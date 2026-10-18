from copy import deepcopy

import pytest

from scripts.apply_dailydialog_local_syntax import apply
from semantic_corpus.template_bank import CorpusTemplate, fingerprint
from tests.test_dailydialog_bank_audit import audit
from tests.test_dailydialog_offline_rules import addressed


def sample(audit):
    doc,payload=addressed(audit)
    payload['ledger']['automatic_revision']={'old':'retained decision'}
    body=dict(idx='train:1:1:1',example=doc.text,provenance=dict(
        origin={'turn':1,'span':[0,len(doc.text)]},cache_key='old'))
    review=dict(id=body['idx'],method='direct_assistant_local_syntax_review',
        semantic_approval=False,publication=False,note='Sir addresses the listener.',
        preimage=fingerprint(dict(body=body,payload=payload)),context_sha256=fingerprint([doc.text]),
        token_edits=[dict(token=0,set={'dep':'vocative'})])
    return body,payload,[doc.text],review,audit,doc.vocab


def test_local_fix_preserves_text_and_old_reviews_without_semantic_claims(audit):
    args=sample(audit);before=deepcopy(args[:4])
    body,payload=apply(*args)
    assert args[:4]==before
    assert payload['ledger']['tokens'][0]['dep']=='vocative'
    assert payload['ledger']['automatic_revision']=={'old':'retained decision'}
    assert payload['ledger']['local_syntax_revision']['preimage']==dict(body=args[0],payload=args[1])
    assert payload['review_status']=='UNREVIEWED'
    assert 'context_review' not in body['provenance']
    assert 'usage_constraints' not in payload['record']
    assert CorpusTemplate(payload['record']).render()['text']==args[0]['example']


def test_second_explicit_repair_preserves_first_review_and_parse(audit):
    args=sample(audit)
    body,payload=apply(*args)
    first=deepcopy(payload)
    review=dict(args[3],preimage=fingerprint(dict(body=body,payload=payload)),
                token_edits=[dict(token=0,set={'parser_lemma':'sir'})],note='Normalize the address lemma.')
    updated,revised=apply(body,payload,args[2],review,audit,args[5])
    assert revised['ledger']['local_syntax_revision']['preimage']['payload']==first
    assert updated['provenance']['previous_cache_key']==body['provenance']['cache_key']


def test_root_consolidation_requires_exact_explicit_transition(audit):
    args=list(sample(audit))
    # A greeting was incorrectly split off as its own syntactic sentence.
    args[1]['ledger']['tokens'][0].update(dep='ROOT',head=0)
    args[3]['preimage']=fingerprint(dict(body=args[0],payload=args[1]))
    args[3]['token_edits']=[dict(token=0,set={'dep':'vocative','head':3})]
    with pytest.raises(ValueError,match='root count'):
        apply(*args)
    args[3]['root_transition']={'before':[0,3],'after':[3]}
    _,result=apply(*args)
    assert result['review_status']=='UNREVIEWED'
    assert result['ledger']['local_syntax_revision']['preimage']['payload']==args[1]
    assert CorpusTemplate(result['record']).render()['text']==args[0]['example']
    args[3]['root_transition']['before']=[3]
    with pytest.raises(ValueError,match='Stale explicit root'):
        apply(*args)


def test_explicit_root_transition_does_not_allow_rootless_graph(audit):
    args=list(sample(audit))
    args[3]['token_edits']=[dict(token=3,set={'dep':'dep'})]
    args[3]['root_transition']={'before':[3],'after':[]}
    with pytest.raises(ValueError,match='lost all sentence roots'):
        apply(*args)


@pytest.mark.parametrize('defect',['stale','context','reference','duplicate','cycle','root','text'])
def test_invalid_or_overreaching_review_is_rejected(audit,defect):
    args=list(sample(audit));r=args[3]
    if defect=='stale':r['preimage']='wrong'
    elif defect=='context':args[2]=['changed']
    elif defect=='reference':
        args[0]['provenance']['context_review']={}
        args[0]['provenance']['context_review']['unit']='sentence'
        r['preimage']=fingerprint(dict(body=args[0],payload=args[1]))
    elif defect=='duplicate':r['token_edits']*=2
    elif defect=='cycle':r['token_edits'][0]['set'].update(head=0)
    elif defect=='root':r['token_edits'][0]['set'].update(dep='ROOT',head=0)
    elif defect=='text':r['token_edits'][0]['set']['text']='Other'
    with pytest.raises(ValueError):apply(*args)
