from copy import deepcopy

import pytest

from scripts.resolve_dailydialog_offline_warnings import check_judgment
from semantic_corpus.template_bank import fingerprint


def example():
    tokens=[dict(i=0, text='Hello', dep='ROOT', head=0),
            dict(i=1, text='John', pos='PROPN', tag='NNP', dep='vocative', head=0)]
    revision=dict(preimage={'tokens':[]}, edits=[{'token':1,'set':{'dep':'vocative'}}])
    current=dict(idx='train:1:1:1', text='Hello John', tokens=tokens,
                 origin={'turn':1, 'span':[0,10]}, automatic_revision=revision)
    context={'turns':['Hello John']}
    issue={'category':'structure','tokens':[1], 'correction':'Mark John as vocative'}
    warning=dict(id=current['idx'], issue_index=0, issue=issue, issue_sha256=fingerprint(issue))
    review=dict(id=current['idx'], issue_index=0, issue_sha256=fingerprint(issue),
        method='direct_assistant_local_warning_review', review_scope='local_warning_only',
        semantic_approval=False, publication=False, note='John is the addressee.',
        occurrence_id=current['idx'], text=current['text'],
        context_sha256=fingerprint(context['turns']), tokens_sha256=fingerprint(tokens),
        finding='fixed_local_syntax', repair_sha256=fingerprint(revision),
        verified_tokens=[tokens[1]], context_turns_read=[1])
    return review,warning,current,context


def test_retained_local_fix_is_not_sentence_approval():
    args=example()
    assert check_judgment(*args)=='fixed_in_candidate'
    assert args[0]['publication'] is False


@pytest.mark.parametrize('part', ['warning','tokens','context','repair','address'])
def test_stale_evidence_cannot_close_warning(part):
    review,warning,current,context=deepcopy(example())
    if part=='warning':warning['issue']['correction']='Change another token'
    if part=='tokens':current['tokens'][1]['head']=1
    if part=='context':context['turns']=['Hello John, another context']
    if part=='repair':current['automatic_revision']['edits']=[]
    if part=='address':current['origin']['span']=[0,5]
    with pytest.raises(ValueError):check_judgment(review,warning,current,context)


def test_partial_fix_stays_pending():
    args=example();args[0]['finding']='needs_additional_repair'
    assert check_judgment(*args)=='pending_context_or_source_review'


def test_local_review_cannot_claim_semantic_approval():
    args=example();args[0]['semantic_approval']=True
    with pytest.raises(ValueError):check_judgment(*args)


def test_nonexistent_context_turn_is_rejected():
    args=example();args[0]['context_turns_read']=[1,2]
    with pytest.raises(ValueError):check_judgment(*args)


def test_explicit_local_repair_requires_bound_preimage():
    args = example(); review, _, occurrence, _ = args
    revision = dict(preimage={'body': {}, 'payload': {}}, semantic_approval=False)
    revision['review'] = dict(id=occurrence['idx'], preimage=fingerprint(revision['preimage']),
                              semantic_approval=False, publication=False)
    occurrence['local_syntax_revision'] = revision
    review.update(repair_kind='local_syntax_revision', repair_sha256=fingerprint(revision))
    assert check_judgment(*args) == 'fixed_in_candidate'
    revision['preimage']['body']['changed'] = True
    review['repair_sha256'] = fingerprint(revision)
    with pytest.raises(ValueError):check_judgment(*args)


def test_already_satisfied_is_limited_to_the_checked_local_claim():
    args = example(); args[0]['finding'] = 'already_satisfied_local_syntax'
    assert check_judgment(*args) == 'closed_not_supported'
    args[0]['verified_tokens'] = []
    with pytest.raises(ValueError):check_judgment(*args)


def test_property_review_binds_the_actual_changed_link_not_a_nearby_token():
    args=example();r,_,o,_=args
    o['attributes']=[dict(slot='a0',target_slot=None)]
    preimage=dict(record={'attributes':[dict(slot='a0',target_slot='n0')]})
    rev=dict(preimage=preimage,preimage_sha256=fingerprint(preimage),semantic_approval=False,
             attribute_changes=[dict(slot='a0',previous_target='n0',target_slot=None)])
    o['attribute_target_revision']=rev
    r.update(finding='fixed_property_link',repair_sha256=fingerprint(rev),
             verified_attributes=[dict(slot='a0',target_slot=None)],context_turns_read=[],
             sentence_span_read=o['origin']['span'])
    assert check_judgment(*args)=='fixed_in_candidate'
    o['attributes'][0]['target_slot']='n1'
    with pytest.raises(ValueError):check_judgment(*args)


@pytest.mark.parametrize('kind,token', [
    ('interrogative_adverb_pos', dict(text='How', pos='SCONJ', tag='WRB', dep='advmod')),
    ('participle_morph_not_clause_aspect', dict(text='made', pos='VERB', tag='VBN',
        morph='Aspect=Perf|Tense=Past|VerbForm=Part')),
])
def test_explicit_representation_review_preserves_the_parse(kind, token):
    r,w,o,c = example()
    o['tokens'][1].update(token)
    r.update(finding='representation_only', representation_kind=kind,
             tokens_sha256=fingerprint(o['tokens']), verified_tokens=[dict(o['tokens'][1])])
    before = deepcopy(o)
    assert check_judgment(r,w,o,c) == 'closed_representation_only'
    assert o == before
    r['verified_tokens'] = [o['tokens'][0]]
    with pytest.raises(ValueError):check_judgment(r,w,o,c)


def test_representation_label_cannot_close_an_arbitrary_dependency_error():
    r,w,o,c = example()
    r.update(finding='representation_only', representation_kind='interrogative_adverb_pos')
    with pytest.raises(ValueError):check_judgment(r,w,o,c)
