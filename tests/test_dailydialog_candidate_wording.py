from copy import deepcopy

import pytest

from scripts.apply_dailydialog_candidate_wording import plan_turn, changed_payload, reconcile_duplicate_contexts
from semantic_corpus.template_bank import CorpusTemplate
from tests.test_dailydialog_bank_audit import audit, active
from tests.test_dailydialog_context_revision import sample


def segment(text, identity='train:1:1:1', start=0):
    return dict(id=identity, span=[start,start+len(text)], text=text)


def correction(s, before, after):
    start = s['text'].index(before)
    return dict(id=s['id'],decision='explicit_wording_correction',
        method='direct_assistant_local_source_wording_review',semantic_approval=False,publication=False,
        source_text=s['text'],corrected_text=s['text'].replace(before,after,1),
        edits=[dict(span=[start,start+len(before)],before=before,after=after)])


def boundary(s, action, value):
    return dict(id=s['id'],method='direct_assistant_local_boundary_review',
        semantic_approval=False,publication=False,current_text=s['text'],
        current_origin={'span':s['span']},decision=action,value=value)


def test_compose_independent_wording_fixes_and_adjust_later_offsets():
    a=segment('He go to scool.')
    b=segment('She stays.', 'train:1:1:2',len(a['text'])+1)
    text=a['text']+' '+b['text']
    reviews=[correction(a,'go','goes'),correction(a,'scool','school')]
    before=deepcopy(reviews)
    result=plan_turn(text,[a,b],reviews,[])
    assert result['after']=='He goes to school. She stays.'
    assert result['pieces'][1]['span']==[19,29]
    assert result['pieces'][1]['input_ids']==[b['id']]
    assert reviews==before


def test_duplicate_warning_does_not_apply_correction_twice():
    a=segment('He go.')
    r=correction(a,'go','goes')
    assert plan_turn(a['text'],[a],[r,deepcopy(r)],[])['after']=='He goes.'


def test_conflicting_wording_and_stale_preimage_are_rejected():
    a=segment('He go.')
    with pytest.raises(ValueError,match='conflicting'):
        plan_turn(a['text'],[a],[correction(a,'go','goes'),correction(a,'go','went')],[])
    r=correction(a,'go','goes');r['edits'][0]['before']='came'
    with pytest.raises(ValueError,match='preimage'):
        plan_turn(a['text'],[a],[r],[])


def test_split_preserves_every_character_and_join_preserves_both_inputs():
    a=segment('He left... She stayed.')
    result=plan_turn(a['text'],[a],[],[boundary(a,'split_after',['He left...'])])
    assert [r['text'] for r in result['pieces']]==['He left...','She stayed.']
    a=segment('At 8 p.')
    b=segment('m.', 'train:1:1:2',len(a['text']))
    text=a['text']+b['text']
    result=plan_turn(text,[a,b],[],[boundary(b,'join',[text])])
    assert result['pieces']==[dict(span=[0,len(text)],text=text,input_ids=[a['id'],b['id']])]


def test_split_join_conflict_and_unrelated_join_are_rejected():
    a=segment('He left... She stayed.')
    with pytest.raises(ValueError,match='conflicting'):
        plan_turn(a['text'],[a],[],[boundary(a,'split_after',['He left...']),boundary(a,'join',[a['text']])])
    with pytest.raises(ValueError,match='exact source'):
        plan_turn(a['text'],[a],[],[boundary(a,'join',['Invented sentence.'])])


def test_review_cannot_grant_semantic_approval_or_introduce_curly_apostrophe():
    a=segment('He goes.')
    r=correction(a,'goes',"doesn\u2019t go")
    with pytest.raises(ValueError,match='apostrophe'):
        plan_turn(a['text'],[a],[r],[])
    r=correction(a,'goes','went');r['semantic_approval']=True
    with pytest.raises(ValueError,match='overreaching'):
        plan_turn(a['text'],[a],[r],[])


def test_reparsed_wording_retains_previous_analysis_without_inheriting_approval(audit):
    doc, previous=active(audit)
    previous['ledger']['local_syntax_revision']={'review':'earlier exact token repair'}
    inputs=[dict(body={'example':doc.text},payload=previous)]
    before=deepcopy(inputs)
    result=changed_payload(doc,inputs,{'before':'He here.','after':doc.text},audit)
    rev=result['ledger']['candidate_source_revision']
    assert rev['preimages']==before and inputs==before
    assert rev['semantic_approval'] is False
    assert rev['syntax_status']=='UNREVIEWED_AFTER_TEXT_OR_BOUNDARY_CHANGE'
    assert 'local_syntax_revision' not in result['ledger']
    assert result['review_status']=='UNREVIEWED'
    assert CorpusTemplate(result['record']).render()['text']==doc.text


def test_local_source_view_does_not_claim_stale_duplicate_context():
    contexts=[{'id':'train:1','turns':['He goes.']},
              {'id':'train:2','turns':['He go.'],'duplicate_of':'train:1'},
              {'id':'train:3','turns':['He goes.'],'duplicate_of':'train:1'}]
    reconcile_duplicate_contexts(contexts,{'train:1'})
    assert contexts[1]['previous_duplicate_of']=='train:1'
    assert 'duplicate_of' not in contexts[1]
    assert contexts[2]['duplicate_of']=='train:1'


def test_unchanged_neighbour_keeps_its_earlier_revision_binding(tmp_path):
    import gzip
    import json
    from scripts.apply_dailydialog_candidate_wording import verify_outputs
    source, out = tmp_path/'before', tmp_path/'after'
    old = dict(idx='train:1:1:1', text='Hello.', tokens=[],
               candidate_source_revision=dict(plan_sha256='previous', semantic_approval=False))
    for directory in (source, out):
        (directory/'bank/occurrences').mkdir(parents=True)
        with gzip.open(directory/'bank/occurrences/0.jsonl.gz', 'wt', encoding='utf-8') as stream:
            stream.write(json.dumps(old)+'\n')
    lineage = {old['idx']:dict(input_ids=[old['idx']], text=old['text'],
                               token_analysis='preserved', plan_sha256='new')}
    assert verify_outputs(source,out,lineage,{})['status']=='passed'
    changed = deepcopy(old)
    changed['tokens'] = [{'text':'changed'}]
    with gzip.open(out/'bank/occurrences/0.jsonl.gz','wt',encoding='utf-8') as stream:
        stream.write(json.dumps(changed)+'\n')
    with pytest.raises(ValueError,match='decision binding'):
        verify_outputs(source,out,lineage,{})


def test_changed_context_revokes_approval_but_preserves_review_and_tokens(sample):
    from scripts.apply_dailydialog_quarantine_reviews import apply_bundle
    from scripts.apply_dailydialog_candidate_wording import invalidate_context_review
    from tests.test_dailydialog_quarantine_reviews import pending
    args=pending(sample)
    b,p=apply_bundle(*args)
    old=dict(body=deepcopy(b),payload=deepcopy(p));ctx=args[2]
    updated=ctx+['A corrected later reply.']
    nb,np=invalidate_context_review(b,p,ctx,updated,old)
    assert dict(body=b,payload=p)==old
    assert 'context_review' not in nb['provenance']
    assert nb['provenance']['context_review_history'][-1]==b['provenance']['context_review']
    assert np['ledger']['tokens']==p['ledger']['tokens']
    assert np['ledger']['candidate_context_revision']['preimage']==old
    assert np['record']['usage_constraints']['mode']=='pending'
    assert np['record']['usage_constraints']['context']==updated
    assert np['record']['template_id']!=p['record']['template_id']
    assert CorpusTemplate(np['record']).render()['text']==b['example']


def test_context_invalidation_cannot_touch_a_manual_reference(sample):
    from scripts.apply_dailydialog_candidate_wording import invalidate_context_review
    b,p,ctx,*_=sample
    p=deepcopy(p);p['review_status']='REVIEWED'
    with pytest.raises(ValueError,match='manual reference'):
        invalidate_context_review(b,p,ctx,ctx,dict(body=b,payload=p))


def test_pending_policy_can_be_rebound_again_without_restoring_old_approval(sample):
    from scripts.apply_dailydialog_quarantine_reviews import apply_bundle
    from scripts.apply_dailydialog_candidate_wording import invalidate_context_review
    from tests.test_dailydialog_quarantine_reviews import pending
    args=pending(sample);b,p=apply_bundle(*args);ctx=args[2]
    first=ctx+['First correction.'];second=ctx+['Second correction.']
    nb,np=invalidate_context_review(b,p,ctx,first,dict(body=b,payload=p))
    final_b,final_p=invalidate_context_review(nb,np,first,second,dict(body=nb,payload=np))
    assert 'context_review' not in final_b['provenance']
    assert final_p['record']['usage_constraints']['context']==second
    assert final_p['record']['usage_constraints']['mode']=='pending'
    assert final_p['ledger']['candidate_context_revision']['preimage']['payload']==np
    assert final_p['ledger']['tokens']==p['ledger']['tokens']
