from copy import deepcopy
import json
import sqlite3

import pytest

from scripts.reconcile_dailydialog_api_results import disposition, representation_only, verify_saved_repairs
from semantic_corpus.template_bank import fingerprint


def question():
    return {'tokens': [[0, 'How', 'how', 'SCONJ', 'WRB', 'advmod', 2, '', [0, 3]]]}, {
        'category': 'structure', 'tokens': [0],
        'explanation': 'How should not be SCONJ.',
        'correction': 'Change token 0 POS from SCONJ to ADV; retain its WRB tag and advmod relation.'}


def test_representation_policy_closes_only_the_warning():
    case, issue = question()
    before = deepcopy(case)
    assert representation_only(issue, case)
    result = disposition('test:1:1:1', 0, issue, case, None, set(), {})
    assert result['status'] == 'closed_representation_only'
    assert result['publication'] is False
    assert case == before


@pytest.mark.parametrize('correction', [
    'Change token 0 POS to ADV and attach it to token 1.',
    'Set token 0 POS to ADV; change its head.',
    'Set token 0 POS to ADV and dep to ROOT.',
    'Change token 1 POS from SCONJ to ADV.',
    'Remove How and set POS to ADV.',
])
def test_mixed_or_mismatched_edits_are_not_closed(correction):
    case, issue = question()
    issue['correction'] = correction
    assert not representation_only(issue, case)


@pytest.mark.parametrize('change', ['source', 'multiple_tokens', 'that', 'wrong_tag', 'mark', 'outside'])
def test_schema_exception_cannot_hide_other_defects(change):
    case, issue = question()
    if change == 'source':
        issue['category'] = 'source'
    elif change == 'multiple_tokens':
        issue['tokens'] = [0, 1]
    elif change == 'that':
        case['tokens'][0][1] = 'that'
    elif change == 'wrong_tag':
        case['tokens'][0][4] = 'IN'
    elif change == 'mark':
        case['tokens'][0][5] = 'mark'
    else:
        issue['tokens'] = [100]
    assert not representation_only(issue, case)


def test_local_fix_does_not_close_other_warnings():
    case, issue = question()
    issue['correction'] = 'Attach token 0 to token 1.'
    review = {'api': {'issues': [issue, issue]}, 'issue_reviews': [
        {'detection': 'confirmed', 'note': 'Confirmed first issue.'},
        {'detection': 'confirmed', 'note': 'Second issue needs a separate repair.'}]}
    assert disposition('x', 0, issue, case, review, {'x'}, {'x': [0]})['status'] == 'fixed_in_candidate'
    assert disposition('x', 1, issue, case, review, {'x'}, {'x': [0]})['status'].startswith('pending_')
    assert disposition('x', 0, issue, case, review, set(), {'x': [0]})['status'].startswith('pending_')
    assert disposition('x', 0, issue, case, None, {'x'}, {'x': [0]})['status'].startswith('pending_')


def test_stale_model_issue_cannot_use_old_adjudication():
    case, issue = question()
    review = {'api': {'issues': [dict(issue, correction='Different correction')]}}
    with pytest.raises(ValueError, match='no longer matches'):
        disposition('x', 0, issue, case, review, {'x'}, {'x': [0]})


def test_explicit_review_preserves_bounded_representation_decision():
    case, issue = question()
    review = {'api': {'issues': [issue]}, 'issue_reviews': [
        {'detection': 'representation_only', 'note': 'Keep the existing WRB convention.'}]}
    result = disposition('x', 0, issue, case, review, set(), {})
    assert result['status'] == 'closed_representation_only'
    assert result['publication'] is False


def test_explicit_representation_decision_cannot_hide_attachment_change():
    case, issue = question()
    issue['correction'] = 'Set token 0 POS to ADV and change its head to 1.'
    review = {'api': {'issues': [issue]}, 'issue_reviews': [
        {'detection': 'representation_only', 'note': 'Unsupported attempt to waive syntax repair.'}]}
    with pytest.raises(ValueError, match='exceeds the bounded POS convention'):
        disposition('x', 0, issue, case, review, set(), {})


def retained_repair(identity='test:1:1:1'):
    preimage = {'body': {'example': 'No.'}, 'payload': {'before': True}}
    review = {'id': identity, 'preimage': fingerprint(preimage), 'after_turn': 'No.',
              'outputs': [{'id': identity, 'span': [0, 3]}]}
    key = fingerprint(['test_revision', review, identity])
    body = {'idx': identity, 'example': 'No.', 'provenance': {'cache_key': key}}
    payload = {'ledger': {'text': 'No.', 'source_revision': {'version': 'test_revision',
               'preimage': preimage, 'review': review, 'semantic_approval': False}}}
    return key, body, payload, fingerprint(review)


def save_repair(path, identity, key, body, payload):
    with sqlite3.connect(path/'revision.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS corrections (id TEXT PRIMARY KEY, key TEXT, body TEXT, payload TEXT)')
        db.execute('INSERT INTO corrections VALUES (?,?,?,?)', (identity, key, json.dumps(body), json.dumps(payload)))


def test_prior_repairs_remain_verified_after_next_batch(tmp_path):
    bindings = {}
    for identity in ('old:1:1:1', 'new:2:1:1'):
        key, body, payload, digest = retained_repair(identity)
        save_repair(tmp_path, identity, key, body, payload)
        bindings[identity] = digest
    before = (tmp_path/'revision.sqlite3').read_bytes()
    assert verify_saved_repairs(tmp_path, bindings) == set(bindings)
    assert (tmp_path/'revision.sqlite3').read_bytes() == before


@pytest.mark.parametrize('defect', ['missing', 'overwritten', 'preimage', 'key', 'body', 'ledger', 'approved', 'output'])
def test_stale_or_unrelated_repairs_cannot_close_warnings(tmp_path, defect):
    identity = 'test:1:1:1'
    key, body, payload, digest = retained_repair(identity)
    revision = payload['ledger']['source_revision']
    if defect == 'overwritten':
        revision['review']['note'] = 'A later unrelated edit'
    elif defect == 'preimage':
        revision['preimage']['body']['example'] = 'Yes.'
    elif defect == 'key':
        body['provenance']['cache_key'] = 'wrong'
    elif defect == 'body':
        body['example'] = 'Yes.'
    elif defect == 'ledger':
        payload['ledger']['text'] = 'Yes.'
    elif defect == 'approved':
        revision['semantic_approval'] = True
    elif defect == 'output':
        revision['review']['outputs'][0]['id'] = 'unrelated'
        digest = fingerprint(revision['review'])
        key = body['provenance']['cache_key'] = fingerprint(['test_revision', revision['review'], identity])
    save_repair(tmp_path, 'other' if defect == 'missing' else identity, key, body, payload)
    with pytest.raises(ValueError):
        verify_saved_repairs(tmp_path, {identity: digest})


@pytest.mark.parametrize('defect', [None, 'snapshot', 'input_address'])
def test_joined_input_warning_requires_its_retained_preimage(tmp_path, defect):
    parent, retired = 'test:1:1:1', 'test:1:1:2'
    key, body, payload, digest = retained_repair(parent)
    original = {'body': {'idx': retired, 'example': '50?'}, 'payload': {'old_parse': True}}
    revision = payload['ledger']['source_revision']
    revision['preimage']['payload'] = {'ledger': {'input_segments': [original]}}
    review = revision['review']
    review['input_ids'] = [parent] if defect == 'input_address' else [parent, retired]
    review['preimage'] = fingerprint(revision['preimage'])
    digest = fingerprint(review)
    key = body['provenance']['cache_key'] = fingerprint(['test_revision', review, parent])
    save_repair(tmp_path, parent, key, body, payload)
    snapshot = deepcopy(original)
    if defect == 'snapshot':
        snapshot['body']['example'] = 'Unrelated text'
    with sqlite3.connect(tmp_path/'revision.sqlite3') as db:
        db.execute('CREATE TABLE retired_occurrences (id TEXT, replacement TEXT, snapshot TEXT)')
        db.execute('INSERT INTO retired_occurrences VALUES (?,?,?)', (retired, parent, json.dumps(snapshot)))
    if defect:
        with pytest.raises(ValueError, match='retired input evidence'):
            verify_saved_repairs(tmp_path, {retired: digest})
    else:
        assert verify_saved_repairs(tmp_path, {retired: digest}) == {retired}
