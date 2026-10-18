from copy import deepcopy

import pytest

from scripts.dailydialog_offline_rules import proposals, repair, VERSION
from tests.test_dailydialog_bank_audit import audit, payload
from tests.test_template_bank import parsed
from semantic_corpus.template_bank import CorpusTemplate, fingerprint


def addressed(audit, name='Sir'):
    doc = parsed([name, ',', 'I', 'need', 'help', '.'], [3,3,3,3,3,3],
        ['npadvmod','punct','nsubj','ROOT','dobj','punct'],
        ['PROPN','PUNCT','PRON','VERB','NOUN','PUNCT'],
        ['NNP',',','PRP','VBP','NN','.'], [name.lower(),',','I','need','help','.'])
    return doc, payload(doc,audit)


def test_local_address_preserves_source_scope_and_preimage(audit):
    doc,before=addressed(audit)
    saved=deepcopy(before)
    after=repair(before,audit,doc.vocab)
    assert before==saved and after['review_status']=='UNREVIEWED'
    assert after['ledger']['tokens'][0]['dep']=='vocative'
    assert after['ledger']['tokens'][0]['head']==3
    revision=after['ledger']['automatic_revision']
    assert revision['version']==VERSION and revision['semantic_approval'] is False
    assert revision['preimage']==before and revision['preimage_sha256']==fingerprint(before)
    assert 'meaning' not in revision and 'usage_constraints' not in after['record']
    assert CorpusTemplate(after['record']).render()['text']==doc.text
    assert repair(after,audit,doc.vocab) is None


@pytest.mark.parametrize('name',['Reservation','service','London','time'])
def test_departments_places_and_other_nouns_are_not_addresses(audit,name):
    _,data=addressed(audit,name)
    assert proposals(data)==[]


@pytest.mark.parametrize('defect',['quote','topic','reviewed','two_roots','modifier','subject','modal_root'])
def test_context_sensitive_addresses_are_not_guessed(audit,defect):
    _,data=addressed(audit)
    ts=data['ledger']['tokens']
    if defect=='quote':data['ledger']['text']='"'+data['ledger']['text']+'"'
    elif defect=='topic':ts[2]['text']='he'
    elif defect=='reviewed':data['review_status']='REVIEWED_REUSE'
    elif defect=='two_roots':ts[0]['dep']='ROOT'
    elif defect=='modifier':ts[4].update(dep='poss',head=0)
    elif defect=='modal_root':ts[3].update(pos='AUX',tag='MD',parser_lemma='will')
    else:ts[0]['dep']='nsubj'
    assert proposals(data)==[]


def test_appositive_name_after_object_is_not_vocative(audit):
    doc=parsed(['I','called','my','brother',',','John','.'],[1,1,3,1,3,3,1],
        ['nsubj','ROOT','poss','dobj','punct','appos','punct'],
        ['PRON','VERB','PRON','NOUN','PUNCT','PROPN','PUNCT'],
        ['PRP','VBD','PRP$','NN',',','NNP','.'],['I','call','my','brother',',','john','.'])
    data=payload(doc,audit)
    span=data['ledger']['tokens'][5]['span']
    data['entities']=[dict(text='John',label='PERSON',span=span)]
    assert proposals(data)==[]


def test_protected_contextual_review_cannot_be_overwritten(audit):
    doc,data=addressed(audit)
    assert repair(data,audit,doc.vocab,protected=True) is None


@pytest.mark.parametrize('own_subject',[False,True])
def test_evaluative_infinitive_is_not_a_completed_event(audit,own_subject):
    doc=parsed(['I','am','glad','to','meet','you','.'],[1,1,1,4,2,4,1],
        ['nsubj','ROOT','acomp','aux','advcl','dobj','punct'],
        ['PRON','AUX','ADJ','PART','VERB','PRON','PUNCT'],
        ['PRP','VBP','JJ','TO','VB','PRP','.'],['I','be','glad','to','meet','you','.'])
    data=payload(doc,audit)
    if own_subject:
        data['ledger']['tokens'][5]['dep']='nsubj'
        assert proposals(data)==[]
    else:
        result=repair(data,audit,doc.vocab)
        assert result['ledger']['tokens'][4]['dep']=='xcomp'
        assert result['ledger']['automatic_revision']['semantic_approval'] is False
