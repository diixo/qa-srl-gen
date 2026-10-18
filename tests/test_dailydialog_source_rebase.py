"""Explicit source corrections must preserve provenance and unrelated analyses."""
from copy import deepcopy
import gzip
import json

import pytest

from scripts import process_dailydialog_hybrid as full
from scripts.rebase_dailydialog_hybrid_source import rewrite, verify_rebase
from semantic_corpus.template_bank import CorpusTemplate, fingerprint
from semantic_corpus.template_compiler import LexicalAudit, compile_record


@pytest.fixture
def sample(tmp_path):
    spacy=pytest.importorskip('spacy')
    from spacy.tokens import Doc
    (tmp_path/'en_verb_inflections.txt').write_text('be\tis\tbeing\twas\tbeen\n')
    (tmp_path/'en_postags_withverb.txt').write_text('be\tverb\n')
    audit=LexicalAudit(tmp_path)
    doc=Doc(spacy.blank('en').vocab,words=['Lt',"'s",'calm','.'],spaces=[False,True,False,False],
        heads=[1,1,1,1],deps=['nsubj','ROOT','acomp','punct'],pos=['PROPN','AUX','ADJ','PUNCT'],
        tags=['NNP','VBZ','JJ','.'],lemmas=['lt','be','calm','.'])
    record,ledger=compile_record(doc,{},audit)
    ledger['automatic_revision']={'rules':['prior repair'],'unresolved_adjective_targets':['a0']}
    payload=dict(record=record,ledger=ledger,method='local_spacy_syntax',review_status='UNREVIEWED',entities=[])
    body=dict(idx='train:1:1:1',example=doc.text,provenance=dict(
        origin=dict(split='train',dialogue=1,turn=1,segment=1,span=[0,len(doc.text)]),
        original=doc.text,cache_key='old',context_id='train:1',boundary_flags=[],
        context_review=dict(unit='pending',note='old typo decision')))
    inputs=[dict(body=body,payload=payload)]
    old=[doc.text];new=[doc.text.replace("Lt's","It's")]
    review=dict(id=body['idx'],method='direct_assistant_source_review',old_ids=[body['idx']],
        preimages=[fingerprint(r) for r in inputs],previous_context_sha256=fingerprint(old),
        context_sha256=fingerprint(new),span=[0,len(doc.text)],text=new[0],unit='sentence',
        meaning={'speech_act':'description','event_status':'state','content':{'state':'calm'},
                 'completed_action_asserted':False,'unknown':[]},evidence_turns=[1],note='It describes a state.',
        token_edits=[dict(token=0,set=dict(text='It',parser_lemma='it',pos='PRON',tag='PRP'))])
    return inputs,old,new,review,audit,doc.vocab


def test_rebase_retains_old_repairs_and_superseded_context_decision(sample):
    inputs,old,new,review,audit,vocab=sample
    before=deepcopy((inputs,old,new,review))
    body,payload=rewrite(*sample)
    assert (inputs,old,new,review)==before
    assert CorpusTemplate(payload['record']).render()['text']=="It's calm."
    assert body['provenance']['context_review_history']==[inputs[0]['body']['provenance']['context_review']]
    assert body['provenance']['context_review']['unit']=='sentence'
    assert not body['provenance']['context_review']['template_semantics_approved']
    assert payload['ledger']['source_revision']['preimages']==inputs
    assert payload['ledger']['tokens'][0]['pos']=='PRON'
    assert payload['ledger']['tokens'][0]['span']==[0,2]
    assert payload['review_status']=='UNREVIEWED'


@pytest.mark.parametrize('key',['preimages','previous_context_sha256','context_sha256'])
def test_rebase_rejects_stale_context_or_parse(sample,key):
    sample[3][key]='stale'
    with pytest.raises(ValueError,match='preimage'): rewrite(*sample)


def test_rebase_rejects_unreviewed_spelling_or_offset_changes(sample):
    inputs,old,new,review,audit,vocab=sample
    new[0]="He's calm."
    review.update(text=new[0],context_sha256=fingerprint(new))
    with pytest.raises(ValueError,match='equal-length'): rewrite(*sample)


def test_rebase_rejects_corrupted_dependency(sample):
    sample[3]['token_edits'].append(dict(token=1,set={'head':0,'dep':'dep'}))
    with pytest.raises(ValueError,match='cyclic'): rewrite(*sample)


def test_rebase_keeps_advice_and_completed_action_decisions_explicit(sample):
    _,payload=rewrite(*sample)
    review=payload['ledger']['source_revision']['review']
    assert review['meaning']==sample[3]['meaning']
    assert not review['meaning']['completed_action_asserted']


def test_number_abbreviation_stays_inside_sentence_but_no_answer_does_not():
    text="It's Flight No. 302, Northwest Orient Airlines. We leave soon."
    spans,_=full.split_sentences(text)
    assert [text[a:b] for a,b in spans]==["It's Flight No. 302, Northwest Orient Airlines.",'We leave soon.']
    text='No. Please wait.'
    spans,_=full.split_sentences(text)
    assert [text[a:b] for a,b in spans]==['No.','Please wait.']


def test_join_allows_only_explained_line_renumbering(tmp_path):
    source,out=tmp_path/'before',tmp_path/'after'
    originals=[dict(idx=str(i),line=i,tokens=[{'text':str(i)}],text=str(i)) for i in range(1,5)]
    inputs=[{'body':r} for r in originals[:2]]
    review=dict(id='1',old_ids=['1','2'],text='12',span=[0,2],preimages=[fingerprint(r) for r in inputs])
    changed=dict(idx='1',line=1,tokens=[{'text':'1'},{'text':'2'}],text='12',origin={'span':[0,2]},
                 source_revision=dict(review=review,preimages=inputs,semantic_approval=False))
    rows=[changed,{**originals[2],'line':2},{**originals[3],'line':3}]
    def save(root,values):
        directory=root/'bank/occurrences';directory.mkdir(parents=True,exist_ok=True)
        with gzip.open(directory/'00000.jsonl.gz','wt',encoding='utf-8') as stream:
            for r in values: stream.write(json.dumps(r)+'\n')
    save(source,originals);save(out,rows)
    result=verify_rebase(source,out,[review])
    assert result['counts']['unchanged_analyses']==2
    assert result['counts']['renumbered_output_lines']==2
    assert result['counts']['new_tokens']==4
    rows[1]['text']='corrupted';save(out,rows)
    with pytest.raises(ValueError,match='unaddressed'): verify_rebase(source,out,[review])
    rows[1]['text']='3';rows[1]['line']=3;save(out,rows)
    with pytest.raises(ValueError,match='physical output line'): verify_rebase(source,out,[review])
