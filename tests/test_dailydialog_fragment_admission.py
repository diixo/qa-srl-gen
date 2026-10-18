"""Context-reviewed adapters must survive import without approving source syntax."""
from copy import deepcopy
import json

import pytest

from scripts.admit_dailydialog_fragments import admit
from semantic_corpus.dialogue_generator.fragments import load_fragment_patterns
from semantic_corpus.template_bank import fingerprint


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


@pytest.fixture
def candidate(tmp_path):
    context = ['I want to say...', 'Go on.', 'I love you.']
    source = dict(split='train', dialogue=1, turn=1, text=context[0], span=[0, len(context[0])])
    pattern = dict(
        id='test_missing_message', kind='partial', context=False,
        assistant='What would you like to tell me?', reply=context[0],
        predicate='say', operators=['want'], tense='present',
        arguments=[['speaker', 'user', 'PERSON']], missing_roles=['message'],
        question='Have I finished saying what I wanted to say?', source=source,
        review=dict(method='direct_assistant_fragment_adapter', scope='adapted_visible_pair_only',
                    adaptation='Keep desire and speaker; do not read the future continuation.',
                    evidence_turns=[1], source_sha256=fingerprint(context[0]),
                    context_sha256=fingerprint(context)),
    )
    review = tmp_path / 'review.json'
    proposal = dict(format='reviewed-fragment-adapters-v1', patterns=[pattern])
    write_json(review, proposal)
    contexts = tmp_path / 'contexts.jsonl'
    contexts.write_text(json.dumps(dict(id='train:1', turns=context))+'\n', encoding='utf-8')
    catalog = tmp_path / 'catalog.json'
    original = dict(format='reviewed-dialogue-fragments-v1', note='keep metadata',
                    patterns=[load_fragment_patterns()[0]])
    write_json(catalog, original)
    return review, proposal, contexts, catalog, original


def test_preflight_publish_idempotence_and_offline_catalog(candidate):
    review, proposal, contexts, catalog, original = candidate
    before = catalog.read_bytes()
    report = admit(review, contexts, catalog=catalog)
    assert not report['published'] and report['new_patterns'] == 1
    assert report['episodes_checked'] == 2 and report['transitions_checked'] == 6
    assert catalog.read_bytes() == before
    report = admit(review, contexts, catalog=catalog, publish=True)
    published = catalog.read_bytes()
    result = json.loads(published)
    assert result['note'] == original['note']
    assert result['patterns'][:-1] == original['patterns']
    assert report['previous_patterns_preserved'] == 1
    assert result['patterns'][-1]['source']['context'][-1] == 'I love you.'
    assert report['template_bank_changed'] is False
    again = admit(review, contexts, catalog=catalog, publish=True)
    assert again['new_patterns'] == 0 and catalog.read_bytes() == published
    contexts.unlink()
    review.unlink()
    assert len(load_fragment_patterns(catalog)) == 2


@pytest.mark.parametrize('defect', ['missing_review', 'future_evidence', 'stale_context',
                                  'wrong_span', 'wrong_format', 'unknown_dialogue'])
def test_invalid_reviews_never_change_catalog(candidate, defect):
    review, proposal, contexts, catalog, original = candidate
    before = catalog.read_bytes()
    pattern = proposal['patterns'][0]
    if defect == 'missing_review':
        del pattern['review']
    elif defect == 'future_evidence':
        pattern['review']['evidence_turns'].append(3)
    elif defect == 'stale_context':
        contexts.write_text(json.dumps(dict(id='train:1', turns=['I want to say...', 'Go on.', 'Never mind.']))+'\n')
    elif defect == 'wrong_span':
        pattern['source']['span'][0] = 1
    elif defect == 'wrong_format':
        proposal['format'] = 'UNREVIEWED'
    else:
        pattern['source']['dialogue'] = 2
    write_json(review, proposal)
    with pytest.raises(ValueError):
        admit(review, contexts, catalog=catalog, publish=True)
    assert catalog.read_bytes() == before


def test_conflicting_review_cannot_overwrite_existing_adapter(candidate):
    review, proposal, contexts, catalog, original = candidate
    admit(review, contexts, catalog=catalog, publish=True)
    before = catalog.read_bytes()
    changed = deepcopy(proposal)
    changed['patterns'][0]['question'] = 'Did I say it?'
    write_json(review, changed)
    with pytest.raises(ValueError, match='cannot be overwritten'):
        admit(review, contexts, catalog=catalog, publish=True)
    assert catalog.read_bytes() == before
