from copy import deepcopy

import pytest

from scripts.recheck_dailydialog_quarantine import METHOD, FIELDS, resolve, token_identity
from semantic_corpus.template_bank import fingerprint


def example(finding='fixed'):
    # A retired sentence address points to a new segment. Token numbers from
    # the API response deliberately differ from the current tokens.
    original = dict(id='train:1:1:3', issue_index=0, group='lemma_or_inflection',
                    issue={'tokens': [8], 'explanation': 'Got needs a participle.'})
    original['issue_sha256'] = fingerprint(original['issue'])
    original['status'] = 'fixed_in_candidate'
    lineage = dict(id='train:1:1:2', input_ids=['train:1:1:2', 'train:1:1:3'],
                   text='Have got.', token_analysis='reparsed_unreviewed',
                   plan_sha256='source-revision', turn_id='train:1:1')
    warning = {k: original[k] for k in ('id', 'issue_index', 'issue_sha256', 'group')}
    warning.update(previous_warning=original, previous_warning_sha256=fingerprint(original),
                   output_occurrences=[lineage], status='quarantined_after_source_revision',
                   semantic_approval=False, publication=False)
    body = dict(idx=lineage['id'], example=lineage['text'], provenance=dict(
        origin=dict(split='train', dialogue=1, turn=1, segment=2, span=[0, 9]),
        context_id='train:1'))
    tokens = [dict(i=0, text='Have', span=[0, 4], pos='AUX', tag='VBP',
                   dep='aux', head=1, parser_lemma='have', morph='Tense=Pres'),
              dict(i=1, text='got', span=[5, 8], pos='VERB', tag='VBD',
                   dep='ROOT', head=1, parser_lemma='get', morph='Tense=Past')]
    payload = dict(record={'attributes': []}, ledger=dict(tokens=tokens,
        candidate_source_revision=dict(plan_sha256=lineage['plan_sha256'],
            preimages=[{'body': {'idx': i}} for i in lineage['input_ids']],
            semantic_approval=False)))
    before = dict(body=body, payload=payload)
    after = deepcopy(before)
    after['payload']['ledger']['tokens'][1].update(tag='VBN', morph='Tense=Past|VerbForm=Part')
    context = ['Have got.']
    token = after['payload']['ledger']['tokens'][1]
    review = {k: warning[k] for k in ('id', 'issue_index', 'issue_sha256')}
    review.update(method=METHOD, review_scope='local_warning_only', finding=finding,
        note='Reviewed current got as the participle of get.',
        previous_disposition_sha256=fingerprint(warning), occurrence_id=body['idx'],
        before_sha256=fingerprint(before), after_sha256=fingerprint(after),
        text=body['example'], origin=body['provenance']['origin'], context_sha256=fingerprint(context),
        lineage=deepcopy(lineage), token_proofs=[dict(token=token_identity(token),
        expected={k: token[k] for k in FIELDS}, head_token=token_identity(token))],
        attribute_targets={}, boundary_outputs=[], semantic_approval=False, publication=False)
    return warning, review, before, after, context


def test_rebased_warning_keeps_complete_prior_and_paid_evidence():
    args = example()
    result = resolve(*args)
    assert result['status'] == 'fixed_in_candidate'
    assert result['previous_disposition'] == args[0]
    assert result['previous_warning']['issue'] == args[0]['previous_warning']['issue']
    assert result['local_recheck']['occurrence_id'] != result['id']
    assert result['semantic_approval'] is False and result['publication'] is False
    assert args[0]['status'] == 'quarantined_after_source_revision'


@pytest.mark.parametrize('field', ['before_sha256', 'after_sha256', 'context_sha256',
                                  'previous_disposition_sha256'])
def test_rejects_stale_evidence(field):
    args = example()
    args[1][field] = 'stale'
    with pytest.raises(ValueError):
        resolve(*args)


def test_cannot_reuse_an_old_token_number_or_wrong_lemma():
    args = example()
    args[1]['token_proofs'][0]['token']['i'] = 8
    with pytest.raises(ValueError, match='token'):
        resolve(*args)
    args = example()
    args[1]['token_proofs'][0]['expected']['parser_lemma'] = 'swim'
    with pytest.raises(ValueError, match='lemma'):
        resolve(*args)


def test_cannot_attach_proof_to_another_current_output():
    args = example()
    args[1]['occurrence_id'] = 'train:1:1:1'
    with pytest.raises(ValueError, match='occurrence'):
        resolve(*args)


def test_cannot_change_head_identity_or_property_target():
    args = example()
    args[1]['token_proofs'][0]['head_token']['text'] = 'Have'
    with pytest.raises(ValueError, match='head'):
        resolve(*args)
    args = example()
    args[1]['attribute_targets'] = {'a0': 'n0'}
    with pytest.raises(ValueError, match='Property'):
        resolve(*args)


@pytest.mark.parametrize('field', ['semantic_approval', 'publication'])
def test_local_fix_never_becomes_semantic_admission(field):
    args = example()
    args[1][field] = True
    with pytest.raises(ValueError, match='overbroad'):
        resolve(*args)


def test_boundary_closure_is_explicit_and_covers_every_output():
    args = example('boundary_only')
    w, r, before, after, context = args
    r['token_proofs'] = []
    proof = dict(id=r['occurrence_id'], text=r['text'], origin=r['origin'],
                 context_sha256=r['context_sha256'])
    r['boundary_outputs'] = [proof]
    result = resolve(*args, boundary_outputs=[proof])
    assert result['status'] == 'fixed_boundary_only'
    assert not result['semantic_approval']
    with pytest.raises(ValueError, match='all current output'):
        resolve(*args)


def test_held_case_remains_quarantined_with_specific_reason():
    args = example('unresolved')
    args[1]['token_proofs'] = []
    args[1]['note'] = 'Ambiguous scope still needs contextual review.'
    result = resolve(*args)
    assert result['status'] == 'quarantined_unresolved'
    assert result['reason'] == args[1]['note']


def test_fix_requires_an_actual_change_and_verification_requires_proof():
    w, r, before, after, context = example()
    r['before_sha256'] = fingerprint(after)
    with pytest.raises(ValueError, match='did not change'):
        resolve(w, r, after, after, context)
    r['finding'] = 'verified_current'
    assert resolve(w, r, after, after, context)['status'] == 'closed_verified_after_source_revision'
    r['token_proofs'] = []
    with pytest.raises(ValueError, match='inspected token'):
        resolve(w, r, after, after, context)


def test_revision_cannot_lose_an_input_segment():
    args = example()
    args[3]['payload']['ledger']['candidate_source_revision']['preimages'].pop()
    args[1]['after_sha256'] = fingerprint(args[3])
    with pytest.raises(ValueError, match='lineage'):
        resolve(*args)


def test_original_warning_cannot_be_rewritten_as_part_of_recheck():
    args = example()
    args[0]['previous_warning']['issue']['explanation'] = 'Altered API claim'
    args[1]['previous_disposition_sha256'] = fingerprint(args[0])
    with pytest.raises(ValueError, match='Original warning'):
        resolve(*args)


def test_full_ledger_preserves_unaddressed_rows_and_paid_response_bytes(tmp_path, monkeypatch):
    import gzip
    import json
    import sqlite3
    from scripts import recheck_dailydialog_quarantine as module

    warning, review, before, after, context = example()
    source, candidate, ledger, out = [tmp_path / name for name in ('source', 'candidate', 'ledger', 'out')]
    for p in (source, candidate, ledger):
        p.mkdir()

    def rows(path, values):
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'wt', encoding='utf-8') as stream:
            for value in values:
                stream.write(json.dumps(value) + '\n')

    (candidate / 'report.json').write_text(json.dumps(dict(validation={'status': 'passed'}, conservation={'status': 'passed'})))
    patches = tmp_path / 'patches.jsonl'
    patches.write_text('applied decisions')
    (candidate / 'plan.json').write_text(json.dumps(dict(source=str(source), reviews=str(patches), review_sha256=module.digest(patches))))
    (ledger / 'summary.json').write_text(json.dumps(dict(candidate=str(source))))
    for root in (source, candidate):
        rows(root / 'contexts.jsonl.gz', [dict(id='train:1', turns=context)])
    # Copy the exact compressed stream, too: gzip timestamps must not matter.
    (candidate / 'contexts.jsonl.gz').write_bytes((source / 'contexts.jsonl.gz').read_bytes())
    untouched = dict(id='train:2:1:1', issue_index=0, issue_sha256='another warning',
                     status='quarantined_unresolved', arbitrary_prior_evidence={'keep': [1, 2, 3]}, group='source')
    rows(ledger / 'warning-decisions.jsonl.gz', [warning, untouched])
    paid_bytes = b'unchanged saved response evidence'
    (ledger / 'received-evidence.jsonl.gz').write_bytes(paid_bytes)
    review_path = tmp_path / 'reviews.jsonl'
    rows(review_path, [review])

    def overlay(root, work):
        image = before if root == source else after
        db = sqlite3.connect(':memory:')
        db.executescript('CREATE TABLE occurrences (id TEXT, key TEXT, body TEXT); CREATE TABLE parses (key TEXT, payload TEXT);')
        db.execute('INSERT INTO occurrences VALUES (?,?,?)', (review['occurrence_id'], 'key', json.dumps(image['body'])))
        db.execute('INSERT INTO parses VALUES (?,?)', ('key', json.dumps(image['payload'])))
        return db, None, None, None

    monkeypatch.setattr(module, 'open_overlay', overlay)
    summary = module.run(candidate, ledger, review_path, out)
    saved = list(module.read_rows(out / 'warning-decisions.jsonl.gz'))
    assert saved[0]['previous_disposition'] == warning
    assert saved[1] == untouched
    assert summary['rechecked_warnings'] == 1 and summary['quarantined_warnings'] == 1
    assert (out / 'received-evidence.jsonl.gz').read_bytes() == paid_bytes
    assert json.loads((out / 'verification.json').read_text())['untouched_dispositions'] == 1
