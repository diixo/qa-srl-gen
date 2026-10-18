from copy import deepcopy

import pytest

from scripts.dailydialog_quality_rules import VERSION, proposals, repair
from scripts.audit_dailydialog_quality_patterns import diagnose
from tests.test_dailydialog_bank_audit import audit, payload
from tests.test_template_bank import parsed
from semantic_corpus.template_bank import CorpusTemplate, fingerprint


def purpose(audit):
    doc=parsed(['We','stopped','to','help','.'],[1,1,3,1,1],
        ['nsubj','ROOT','aux','xcomp','punct'], ['PRON','VERB','PART','VERB','PUNCT'],
        ['PRP','VBD','TO','VB','.'],['we','stop','to','help','.'])
    return doc,payload(doc,audit)


def test_stop_purpose_retains_evidence_without_completion_claim(audit):
    doc,before=purpose(audit);saved=deepcopy(before)
    after=repair(before,audit,doc.vocab)
    assert before==saved
    assert after['ledger']['tokens'][3]['dep']=='advcl'
    assert after['ledger']['tokens'][2]['dep']=='mark'
    proof=after['ledger']['automatic_revision']
    assert proof['preimage']==before and proof['preimage_sha256']==fingerprint(before)
    assert proof['semantic_approval'] is False and proof['version']==VERSION
    assert after['review_status']=='UNREVIEWED' and 'usage_constraints' not in after['record']
    assert CorpusTemplate(after['record']).render()['text']==doc.text
    assert repair(after,audit,doc.vocab) is None


def test_stop_talking_to_someone_is_cessation_not_purpose(audit):
    doc=parsed(['Stop','talking','to','me','.'],[0,0,1,2,0],
        ['ROOT','xcomp','prep','pobj','punct'], ['VERB','VERB','ADP','PRON','PUNCT'],
        ['VB','VBG','IN','PRP','.'],['stop','talk','to','I','.'])
    data=payload(doc,audit)
    assert any(h['family']=='stop_infinitive_purpose' for h in diagnose(data['ledger']['tokens']))
    assert repair(data,audit,doc.vocab) is None


@pytest.mark.parametrize('reason',['protected','reviewed','source_revision','attribute_revision','attribute_target_revision','quote'])
def test_existing_evidence_is_not_overwritten(audit,reason):
    doc,data=purpose(audit)
    if reason=='reviewed':data['review_status']='REVIEWED_REUSE'
    elif reason=='quote':data['ledger']['text']='"'+data['ledger']['text']+'"'
    elif reason!='protected':data['ledger'][reason]={'old':'decision'}
    assert repair(data,audit,doc.vocab,protected=reason=='protected') is None


@pytest.mark.parametrize('word,dep,child', [('on','prep',False),('on','prep',True),('in','advmod',False)])
def test_particle_requires_no_prepositional_object(audit,word,dep,child):
    verb='move' if word=='in' else 'try'
    doc=parsed(['They',verb,'it',word,'.'],[1,1,1,1,1],
        ['nsubj','ROOT','dobj',dep,'punct'], ['PRON','VERB','PRON','ADP','PUNCT'],
        ['PRP','VBP','PRP','IN','.'],['they',verb,'it',word,'.'])
    data=payload(doc,audit)
    if child:
        data['ledger']['tokens'][2].update(head=3,dep='pobj')
        assert repair(data,audit,doc.vocab) is None
    else:
        after=repair(data,audit,doc.vocab)
        assert after['ledger']['tokens'][3]['dep']=='prt'


def test_hard_infinitive_does_not_guess_referential_it(audit):
    doc=parsed(['It','is','hard','to','say','.'],[1,1,1,4,1,1],
        ['nsubj','ROOT','acomp','aux','xcomp','punct'], ['PRON','AUX','ADJ','PART','VERB','PUNCT'],
        ['PRP','VBZ','JJ','TO','VB','.'],['it','be','hard','to','say','.'])
    data=payload(doc,audit);after=repair(data,audit,doc.vocab)
    assert after['ledger']['tokens'][4]['head']==2
    assert after['ledger']['tokens'][0]==data['ledger']['tokens'][0]
    data['ledger']['tokens'][3]['dep']='mark'
    assert proposals(data)==[]


@pytest.mark.parametrize('first,person,expected',[('Hello',True,True),('Hello',False,False),('Oh',True,False)])
def test_greetings_do_not_treat_every_proper_noun_as_addressee(audit,first,person,expected):
    doc=parsed([first,',','John','.'],[0,0,0,0],
        ['ROOT','punct','npadvmod','punct'],['INTJ','PUNCT','PROPN','PUNCT'],
        ['UH',',','NNP','.'],[first.lower(),',','John','.'])
    data=payload(doc,audit)
    if person:data['entities']=[dict(label='PERSON',span=data['ledger']['tokens'][2]['span'])]
    after=repair(data,audit,doc.vocab)
    assert (after is not None)==expected
    if after:assert after['ledger']['tokens'][2]['dep']=='vocative'
