"""Unapproved data share physical storage without entering generated datasets."""
import json

import pytest

from semantic_corpus.template_bank import TemplateBank
from semantic_corpus.template_pending import load_pending, store_pending, verify_pending
from test_template_registry import make_bank, publish, write_json, write_rows


def test_pending_ranges_survive_publication_and_preserve_active_snapshot(tmp_path):
    repo = tmp_path / 'repo'
    root, active_text = publish(repo, tmp_path / 'active')
    old_reader = TemplateBank(root)
    manifest = (root / 'manifest.json').read_bytes()
    candidate, source, pending_text = make_bank(tmp_path / 'candidate', subject='Beth')
    entry = store_pending(candidate, root, source_id='pending')
    assert (root / 'manifest.json').read_bytes() == manifest
    assert [r['text'] for r in TemplateBank(root).generate()] == [active_text]
    assert [r['text'] for r in old_reader.generate()] == [active_text]
    pending, = TemplateBank(root).pending_components()
    assert pending.path == root
    assert [t.render()['text'] for t in pending.templates()] == [pending_text]
    assert all(offset > 0 for name, (offset, _) in entry['files'].items() if name == 'verbs.jsonl.gz')
    lengths = {name: (root / name).stat().st_size for name in entry['files']}
    assert store_pending(candidate, root, source_id='pending') == entry
    assert lengths == {name: (root / name).stat().st_size for name in entry['files']}
    publish(repo, tmp_path / 'next', subject='Carol')
    assert verify_pending(root) == [entry]
    assert pending_text not in {r['text'] for r in TemplateBank(root).generate()}
    # No source directory is required to read the stored records.
    for path in candidate.rglob('*'):
        if path.is_file(): path.unlink()
    assert [t.render()['text'] for t in TemplateBank(root).pending_components()[0].templates()] == [pending_text]


def test_historical_reviewed_entities_stay_out_of_typed_generation(tmp_path):
    from semantic_corpus.semantic_generator.substitutions import default_pool
    root, _ = publish(tmp_path / 'repo', tmp_path / 'active')
    before = len(default_pool(mined_bank=root))
    source = tmp_path / 'inventories'
    write_json(source / 'manifest.json', {'review_status': 'UNREVIEWED'})
    write_rows(source / 'entities.jsonl.gz', [{'review_status': 'REVIEWED',
        'entity': {'text': 'Unique Pending Name', 'labels': ['PERSON', 'NAMED_ENTITY'], 'named': True}}])
    store_pending(source, root, source_id='resources', inventories=True)
    assert len(default_pool(mined_bank=root)) == before
    assert TemplateBank(root).pending_components() == ()
    assert load_pending(root)[0]['kind'] == 'inventories'


def test_reading_review_survives_retry_and_new_import_without_publishing(tmp_path):
    from semantic_corpus.semantic_generator.substitutions import default_pool
    from semantic_corpus.template_bank import fingerprint

    root, _ = publish(tmp_path / 'repo', tmp_path / 'active')
    before = list(default_pool(mined_bank=root))
    source = tmp_path / 'inventories'
    write_json(source / 'manifest.json', {'review_status': 'UNREVIEWED'})
    write_rows(source / 'entities.jsonl.gz', [{'review_status': 'REVIEWED',
        'entity': {'text': 'Unique Pending Name', 'labels': ['PERSON', 'NAMED_ENTITY'], 'named': True}}])
    entry = store_pending(source, root, source_id='resources', inventories=True)
    review = {'entities.jsonl.gz': {'decisions': [{'row_index': 0, 'outcome': 'admitted'}]}}
    index = root / 'pending.json'
    data = json.loads(index.read_text())
    entry['review'] = review
    entry['sha256'] = fingerprint({k: v for k, v in entry.items() if k != 'sha256'})
    data['sources'][0] = entry
    write_json(index, data)
    lengths = {name: (root / name).stat().st_size for name in entry['files']}

    assert store_pending(source, root, source_id='resources', inventories=True) == entry
    assert lengths == {name: (root / name).stat().st_size for name in entry['files']}
    other, _, _ = make_bank(tmp_path / 'other')
    store_pending(other, root, source_id='other')
    assert verify_pending(root)[0] == entry
    assert list(default_pool(mined_bank=root)) == before
    assert all(e['generation_enabled'] is False for e in load_pending(root))

    # Review decisions are included in the metadata checksum, too.
    data = json.loads(index.read_text())
    data['sources'][0]['review']['entities.jsonl.gz']['decisions'][0]['outcome'] = 'held'
    write_json(index, data)
    with pytest.raises(ValueError, match='metadata changed'):
        load_pending(root)


def test_interrupted_append_is_invisible_and_retry_preserves_active_data(tmp_path, monkeypatch):
    from semantic_corpus import template_pending
    root, text = publish(tmp_path / 'repo', tmp_path / 'active')
    candidate, _, _ = make_bank(tmp_path / 'candidate')
    original = template_pending._append_file
    def fail_after_append(*args):
        original(*args)
        raise OSError('injected interruption')
    monkeypatch.setattr(template_pending, '_append_file', fail_after_append)
    with pytest.raises(OSError, match='interruption'):
        store_pending(candidate, root, source_id='pending')
    assert load_pending(root) == []
    assert [r['text'] for r in TemplateBank(root).generate()] == [text]
    monkeypatch.setattr(template_pending, '_append_file', original)
    store_pending(candidate, root, source_id='pending')
    assert len(verify_pending(root)) == 1


def test_changed_pending_data_or_conflicting_source_is_rejected(tmp_path):
    root, _ = publish(tmp_path / 'repo', tmp_path / 'active')
    source, _, _ = make_bank(tmp_path / 'candidate')
    entry = store_pending(source, root, source_id='pending')
    different, _, _ = make_bank(tmp_path / 'different', subject='Other')
    with pytest.raises(FileExistsError):
        store_pending(different, root, source_id='pending')
    name = 'verbs.jsonl.gz'
    offset, size = entry['files'][name]
    with (root / name).open('r+b') as stream:
        stream.seek(offset); stream.write(b'X')
    with pytest.raises(ValueError, match='bytes changed'):
        verify_pending(root)


def test_pending_metadata_cannot_silently_enable_generation(tmp_path):
    root, _ = publish(tmp_path / 'repo', tmp_path / 'active')
    source, _, _ = make_bank(tmp_path / 'candidate')
    store_pending(source, root, source_id='pending')
    index = root / 'pending.json'
    data = json.loads(index.read_text())
    data['sources'][0]['generation_enabled'] = True
    write_json(index, data)
    with pytest.raises(ValueError, match='Invalid pending'):
        load_pending(root)


def test_legacy_entity_candidate_name_uses_the_existing_inventory(tmp_path):
    root, _ = publish(tmp_path / 'repo', tmp_path / 'active')
    source, _, _ = make_bank(tmp_path / 'candidate')
    write_rows(source / 'entity-candidates.jsonl.gz', [{'text': 'Paris'}])
    entry = store_pending(source, root, source_id='pending')
    assert 'entity_candidates.jsonl.gz' in entry['files']
    assert not (root / 'entity-candidates.jsonl.gz').exists()
    assert list(TemplateBank(root).pending_components()[0].rows('entity_candidates.jsonl.gz')) == [{'text': 'Paris'}]
    assert store_pending(source, root, source_id='pending') == entry
