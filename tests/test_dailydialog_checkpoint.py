import hashlib
import json
from pathlib import Path

import pytest

from scripts.dailydialog_checkpoint import consolidate, create, digest, load, restore, safe_path, verify
from scripts.audit_dailydialog_quarantine import audit_inputs, verify_evidence


def make_bundle(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'first.jsonl').write_bytes(b'alpha\nbeta\n' * 90)
    (repo / 'same.jsonl').write_bytes((repo / 'first.jsonl').read_bytes())
    (repo / 'empty.json').write_bytes(b'')
    (repo / 'binary.db').write_bytes(bytes(range(256)) * 10)
    bundle = tmp_path / 'bundle'
    selection = {p.name: {'test'} for p in repo.iterdir()}
    manifest = create(repo, bundle, selection, chunk_size=128, volume_size=2304)
    return repo, bundle, manifest


def test_lossless_roundtrip_deduplicates_and_restores_all_files(tmp_path):
    repo, bundle, manifest = make_bundle(tmp_path)
    files = {row['path']: row for row in manifest['files']}
    assert files['first.jsonl']['chunks'] == files['same.jsonl']['chunks']
    assert manifest['totals']['unique_chunk_bytes'] < manifest['totals']['source_bytes']
    assert verify(bundle, repo)['status'] == 'passed'
    dest = tmp_path / 'restored'
    assert restore(bundle, dest)['restored'] == 4
    assert verify(bundle, dest)['status'] == 'passed'
    assert restore(bundle, dest)['already_identical'] == 4


def test_restore_conflict_is_detected_before_writing_other_files(tmp_path):
    _, bundle, _ = make_bundle(tmp_path)
    dest = tmp_path / 'restored'
    dest.mkdir()
    (dest / 'same.jsonl').write_text('user change')
    with pytest.raises(ValueError, match='will not overwrite'):
        restore(bundle, dest)
    assert list(dest.iterdir()) == [dest / 'same.jsonl']
    assert (dest / 'same.jsonl').read_text() == 'user change'


def test_prefix_restore_does_not_restore_unselected_files(tmp_path):
    _, bundle, _ = make_bundle(tmp_path)
    dest = tmp_path / 'restored'
    assert restore(bundle, dest, prefix='first.jsonl')['restored'] == 1
    assert [p.name for p in dest.iterdir()] == ['first.jsonl']


@pytest.mark.parametrize('name', ['../outside', '/outside', 'C:/outside', 'x/../../y',
                                 'x\\..\\y', 'x/./y', ''])
def test_rejects_unsafe_restore_paths(tmp_path, name):
    with pytest.raises(ValueError):
        safe_path(tmp_path, name)


def test_changed_archive_fails_before_restore(tmp_path):
    _, bundle, manifest = make_bundle(tmp_path)
    archive = bundle / next(iter(manifest['archives']))
    archive.write_bytes(archive.read_bytes() + b'corruption')
    dest = tmp_path / 'restored'
    with pytest.raises(ValueError, match='Volume checksum'):
        restore(bundle, dest)
    assert not dest.exists()


def test_wrong_file_digest_and_changed_sources_are_detected(tmp_path):
    repo, bundle, manifest = make_bundle(tmp_path)
    (repo / 'first.jsonl').write_bytes(b'new data')
    with pytest.raises(ValueError, match='Source differs'):
        verify(bundle, repo)
    manifest['files'][0]['sha256'] = hashlib.sha256(b'wrong').hexdigest()
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='File checksum'):
        verify(bundle)


def test_manifest_cannot_alias_windows_paths(tmp_path):
    _, bundle, manifest = make_bundle(tmp_path)
    manifest['files'].append({**manifest['files'][0], 'path': manifest['files'][0]['path'].upper()})
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Duplicate path'):
        load(bundle)


def test_existing_bundle_is_never_overwritten(tmp_path):
    repo, bundle, _ = make_bundle(tmp_path)
    before = (bundle / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        create(repo, bundle, {'first.jsonl': {'test'}})
    assert (bundle / 'manifest.json').read_bytes() == before


def test_one_time_import_survives_removal_of_source_workspace(tmp_path):
    repo = tmp_path / 'repo'
    pool = repo / 'data/mined-templates'
    reviews = 'data/dailydialog/review-parts'
    candidate = 'artifacts/candidate'

    def put(name, value):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + '\n', encoding='utf8')
        return path

    names = {
        candidate + '/bank/manifest.json': {'hybrid_import': {'review_status': 'UNREVIEWED'}},
        candidate + '/bank/templates/00.jsonl.gz': {'pending': True},
        reviews + '/annotations/part-001.jsonl': {'split': 'train', 'dialogue': 1},
        reviews + '/luna-offline-quality-v38/summary.json': {'candidate': candidate},
        reviews + '/luna-offline-source-v34/summary.json': {'evidence': 'original'},
        reviews + '/decision.json': {'meaning': 'reviewed'},
        'data/dailydialog/filtered/dailydialog-train.jsonl': ['Hello.'],
    }
    for name, value in names.items():
        put(name, value)
    pool_manifest = put('data/mined-templates/manifest.json', {'sources': ['accepted-only']})
    original_pool_bytes = pool_manifest.read_bytes()
    evidence_name = reviews + '/luna-offline-source-v34/summary.json'
    register = dict(candidate=candidate, occurrences=[{'decision': 'keep in quarantine'}],
                    evidence={evidence_name: digest(repo / evidence_name)})
    register_path = put('data/mined-templates/reviewed/dailydialog-quarantine.json', register)
    original_register_bytes = register_path.read_bytes()
    bundle = pool / 'audit/dailydialog-v38/checkpoint'
    create(repo, bundle, {name: {'test'} for name in names}, metadata={'candidate': candidate})
    # Only the archive survives; neither the source bank nor review-parts provides inputs.
    for name in names:
        (repo / name).unlink()
    result = consolidate(repo, bundle)
    assert result['files'] == len(names)
    assert result['candidate_published'] is False
    assert pool_manifest.read_bytes() == original_pool_bytes
    updated = json.loads(register_path.read_text())
    assert updated['occurrences'] == register['occurrences']
    assert not any(name.startswith(reviews) for name in updated['evidence'])
    assert (pool / 'audit/dailydialog-v38/evidence/quarantine-register-before-import.json').read_bytes() == original_register_bytes
    verify_evidence(repo, updated)
    inputs = audit_inputs(repo, updated)
    assert inputs['centralized'] and inputs['parts'] is None
    assert json.loads((inputs['candidate'] / 'manifest.json').read_text())['hybrid_import']['review_status'] == 'UNREVIEWED'
    with pytest.raises(ValueError, match='already recorded'):
        consolidate(repo, bundle)
    (inputs['annotations'] / 'part-001.jsonl').write_text('changed review')
    with pytest.raises(ValueError, match='Imported evidence changed'):
        audit_inputs(repo, updated)


def test_copied_candidates_merge_into_shared_files_without_losing_evidence(tmp_path):
    from scripts.dailydialog_checkpoint import merge_pool
    from semantic_corpus.template_bank import TemplateBank
    from test_template_registry import make_bank, publish, write_json, write_rows

    repo = tmp_path / 'repo'
    pool, active_text = publish(repo, tmp_path / 'active')
    candidate, _, _ = make_bank(repo / 'artifacts/candidate', subject='Beth')
    manifest = json.loads((candidate / 'manifest.json').read_text())
    manifest['hybrid_import'] = {'review_status': 'UNREVIEWED'}
    write_json(candidate / 'manifest.json', manifest)
    review = 'data/dailydialog/review-parts'
    names = [p.relative_to(repo).as_posix() for p in candidate.rglob('*') if p.is_file()]
    for name, data in {
        review + '/annotations/part-001.jsonl': {'split': 'train', 'dialogue': 1},
        review + '/luna-offline-quality-v38/summary.json': {'candidate': str(candidate)},
        review + '/luna-offline-source-v34/summary.json': {'evidence': True},
        'data/dailydialog/filtered/dailydialog-train.jsonl': ['Hi.'],
    }.items():
        write_json(repo / name, data)
        names.append(name)
    evidence = review + '/luna-offline-source-v34/summary.json'
    register_path = pool / 'reviewed/dailydialog-quarantine.json'
    write_json(register_path, {'candidate': str(candidate), 'evidence': {evidence: digest(repo / evidence)}})
    bundle = pool / 'audit/dailydialog-v38/checkpoint'
    # consolidate expects a historical candidate's bank/ subdirectory.
    actual = candidate / 'bank'
    actual.mkdir()
    for p in list(candidate.iterdir()):
        if p != actual: p.rename(actual / p.name)
    names = [n for n in names if not n.startswith('artifacts/candidate/')]
    names += [p.relative_to(repo).as_posix() for p in actual.rglob('*') if p.is_file()]
    create(repo, bundle, {n: {'test'} for n in names}, metadata={'candidate': 'artifacts/candidate'})
    consolidate(repo, bundle)
    resources = pool / 'candidates/dailydialog-resources'
    write_json(resources / 'manifest.json', {'review_status': 'UNREVIEWED'})
    write_rows(resources / 'nominals.jsonl.gz', [{'text': 'a book', 'review_status': 'UNREVIEWED'}])
    result = merge_pool(repo)
    assert result['status'] == 'merged'
    assert not (pool / 'candidates').exists()
    assert not (pool / 'audit/dailydialog-v38').exists()
    assert (pool / 'audit/checkpoint/manifest.json').exists()
    updated = json.loads(register_path.read_text())
    verify_evidence(repo, updated)
    inputs = audit_inputs(repo, updated)
    assert inputs['candidate'].path == pool
    assert inputs['candidate'].source_id == 'dailydialog'
    assert next(inputs['candidate'].templates()).render()['text'].startswith('Beth ')
    assert [r['text'] for r in TemplateBank(pool).generate()] == [active_text]
    assert merge_pool(repo)['status'] == 'already_merged'
