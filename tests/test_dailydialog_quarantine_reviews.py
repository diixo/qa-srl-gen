from copy import deepcopy

import pytest

from scripts.apply_dailydialog_quarantine_reviews import apply_bundle, VERSION
from semantic_corpus.template_bank import CorpusTemplate, fingerprint
from tests.test_dailydialog_context_revision import sample


def pending(sample):
    body, payload, context, step, audit, vocab, _ = sample
    step = dict(step, token_edits=[], unit='pending',
                usage={'mode':'context_only','reasons':['Source remains ambiguous.']},
                meaning=dict(speech_act='utterance',event_status='unknown',
                             content={'observed':'Fire!'},completed_action_asserted=False,
                             unknown=['command or warning']))
    review = dict(id=body['idx'],method=VERSION,preimage=fingerprint(dict(body=body,payload=payload)),
                  context_sha256=fingerprint(context),steps=[step],note='Keep ambiguity open.',
                  semantic_approval=False,publication=False)
    return body,payload,context,review,audit,vocab


def test_partial_meaning_is_retained_without_granting_generation(sample):
    args = pending(sample); original = deepcopy(args[:4])
    body, result = apply_bundle(*args)
    assert args[:4] == original
    assert CorpusTemplate(result['record']).render()['text'] == args[0]['example']
    usage = result['record']['usage_constraints']
    assert usage['mode'] == 'context_only'
    assert usage['meaning']['unknown'] == ['command or warning']
    assert body['provenance']['context_review']['unit'] == 'pending'
    assert result['ledger']['quarantine_revision']['preimage'] == dict(body=args[0],payload=args[1])
    assert result['review_status'] == 'UNREVIEWED'


@pytest.mark.parametrize('defect',['stale','context','approval','publication','empty','admission','guessed_syntax','method','protected'])
def test_bad_quarantine_bundle_is_rejected(sample, defect):
    args = list(pending(sample)); review = args[3]
    if defect == 'stale': review['preimage'] = 'wrong'
    elif defect == 'context': args[2] = ['A different setting',args[0]['example']]
    elif defect == 'approval': review['semantic_approval'] = True
    elif defect == 'publication': review['publication'] = True
    elif defect == 'empty': review['steps'] = []
    elif defect == 'admission': review['steps'][0]['usage']['mode'] = 'attested_sentence'
    elif defect == 'guessed_syntax': review['steps'][0]['token_edits'] = [{'token':0,'set':{'pos':'VERB'}}]
    elif defect == 'method': review['steps'][0]['method'] = 'automatic_semantic_approval'
    else:
        args[0]['provenance']['context_review'] = {'unit':'sentence'}
        review['preimage'] = fingerprint(dict(body=args[0],payload=args[1]))
    with pytest.raises(ValueError): apply_bundle(*args)


@pytest.mark.parametrize('tampered_history',[False,True])
@pytest.mark.parametrize('history_field',['attribute_target_revision','candidate_context_revision'])
def test_compound_activity_stays_one_verb_without_losing_source(sample, tmp_path,tampered_history,history_field):
    from spacy.tokens import Doc
    from semantic_corpus.template_compiler import compile_record
    body,payload,_,_,audit,vocab,_=sample
    doc=Doc(vocab,words=['Trick','-','or','-','treating','?'],spaces=[False]*6,
            heads=[0,0,0,4,0,0],deps=['ROOT','punct','cc','punct','conj','punct'],
            pos=['NOUN','PUNCT','CCONJ','PUNCT','NOUN','PUNCT'],
            tags=['NN','HYPH','CC','HYPH','NN','.'],lemmas=['trick','-','or','-','treating','?'])
    record,ledger=compile_record(doc,{},audit)
    ledger['candidate_source_revision']={'historical_decision':'before retokenization'}
    ledger['candidate_context_revision']={'status':'invalidated_after_wording_change',
        'previous_context_sha256':'old-context','preimage':{'prior_review':'retained'}}
    ledger['attribute_target_revision']={'version':'offline_attribute_targets_v1',
        'attribute_changes':[{'slot':'a0','previous_target':'n0','target_slot':None}]}
    payload=dict(payload,record=record,ledger=ledger)
    body=deepcopy(body);body['example']=doc.text
    body['provenance'].update(original=doc.text,origin=dict(split='test',dialogue=1,turn=2,segment=1,span=[0,len(doc.text)]))
    context=['What Halloween activity?',doc.text]
    meaning=dict(speech_act='question',event_status='questioned',content={'activity':'trick-or-treat'},completed_action_asserted=False,unknown=['whether it occurs'])
    output=dict(id=body['idx'],span=[0,len(doc.text)],unit='fragment',meaning=meaning,
                evidence_turns=[1,2],note='One compound activity.',boundary_flags=[],
                usage={'mode':'context_only','reasons':['Question; no asserted completion.']},
                tokens=[{'from':[0,1,2,3,4],'set':dict(text='Trick-or-treating',parser_lemma='trick-or-treat',pos='VERB',tag='VBG',dep='ROOT',head=0,morph='Aspect=Prog|Tense=Pres|VerbForm=Part')},{'from':[5]}])
    step=dict(id=body['idx'],method='direct_assistant_source_review',preimage=fingerprint(dict(body=body,payload=payload)),context_sha256=fingerprint(context),before_turn=doc.text,after_turn=doc.text,dropped_tokens=[],outputs=[output])
    review=dict(id=body['idx'],method=VERSION,preimage=step['preimage'],context_sha256=step['context_sha256'],steps=[step],note='Merge one compound lexeme.',semantic_approval=False,publication=False)
    b,p=apply_bundle(body,payload,context,review,audit,vocab)
    assert p['record']['verbs']==['trick-or-treat']
    assert len(p['ledger']['tokens'])==2
    assert CorpusTemplate(p['record']).render()['text']==doc.text
    assert p['record']['usage_constraints']['mode']=='context_only'
    assert b['provenance']['context_review']['meaning']['completed_action_asserted'] is False
    assert p['ledger']['quarantine_revision']['preimage']['payload']['ledger']['tokens']==ledger['tokens']
    assert p['ledger']['quarantine_revision']['preimage']['payload']['ledger']['candidate_source_revision']==ledger['candidate_source_revision']
    assert p['ledger']['quarantine_revision']['preimage']['payload']['ledger']['attribute_target_revision']==ledger['attribute_target_revision']
    assert p['ledger']['quarantine_revision']['preimage']['payload']['ledger']['candidate_context_revision']==ledger['candidate_context_revision']
    # The old source binding lives in the retained preimage, not at the root
    # of a newly tokenized ledger. Conservation must allow precisely this case.
    import gzip
    import json
    from scripts.apply_dailydialog_quarantine_reviews import conserve
    source,out=tmp_path/'source',tmp_path/'out'
    for directory,bdy,parsed in [(source,body,payload),(out,b,p)]:
        (directory/'bank/occurrences').mkdir(parents=True)
        row=dict(parsed['ledger'],**bdy['provenance'],idx=bdy['idx'])
        if tampered_history and directory==source:
            row[history_field]={'different_history':True}
        with gzip.open(directory/'bank/occurrences/0.jsonl.gz','wt',encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        for name in ('contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'):
            (directory/name).write_bytes(b'unchanged fixture')
    if tampered_history:
        with pytest.raises(ValueError,match='Prior repair history missing'):
            conserve(source,out,[review],{body['idx']:(b,p)})
        return
    report=conserve(source,out,[review],{body['idx']:(b,p)})
    assert report['status']=='passed' and report['counts']['revised_occurrences']==1


def construction_case(sample):
    from tests.test_template_bank import parsed
    from semantic_corpus.template_compiler import compile_record
    from semantic_corpus.template_usage import constrain_record
    from semantic_corpus.syntax_mining import EXTRACTOR_VERSION
    body,payload,_,_,audit,vocab,_=sample
    doc=parsed(['Anna','turned','it','on','and','off','.'],[1,1,1,1,3,3,1],
        ['nsubj','ROOT','dobj','prt','cc','conj','punct'],['PROPN','VERB','PRON','ADP','CCONJ','ADP','PUNCT'],
        ['NNP','VBD','PRP','RP','CC','RP','.'],['anna','turn','it','on','and','off','.'])
    record,ledger=compile_record(doc,{},audit)
    body=deepcopy(body);body['example']=doc.text
    body['provenance'].update(original=doc.text,origin=dict(split='test',dialogue=1,turn=2,segment=1,span=[0,len(doc.text)]))
    context=['What happened?',doc.text]
    # Retain an older, reviewed scope restriction while refreshing only derived groups.
    body['provenance']['context_review']={'note':'Preserved earlier scope review.'}
    record['constructions']=['turn on']
    for group in (record['predicate_groups'][0],ledger['predicate_groups'][0]):
        group['lemma']='turn on';group['constructions']=group['constructions'][:1]
    constrain_record(record,body,context,dict(unit='sentence',meaning={'event_status':'reported'},
        usage={'mode':'context_only','reasons':['Missing coordinated construction.']}))
    ledger['template_id']=record['template_id']
    payload=deepcopy(payload);payload.update(record=record,ledger=ledger)
    step=dict(id=body['idx'],method='direct_assistant_construction_review',
        preimage=fingerprint(dict(body=body,payload=payload)),context_sha256=fingerprint(context),
        extractor_version=EXTRACTOR_VERSION,expected_constructions={'v0':['turn on','turn off']},
        expected_temporal_adjuncts=[],note='Retain both coordinated particles; no event chosen.',
        semantic_approval=False,publication=False)
    review=dict(id=body['idx'],method=VERSION,preimage=step['preimage'],context_sha256=step['context_sha256'],
        steps=[step],note=step['note'],semantic_approval=False,publication=False)
    return body,payload,context,review,audit,vocab


def test_construction_refresh_preserves_source_context_and_previous_restrictions(sample):
    args=construction_case(sample);before=deepcopy(args[:4])
    body,payload=apply_bundle(*args)
    assert args[:4]==before
    assert payload['ledger']['tokens']==args[1]['ledger']['tokens']
    assert payload['record']['constructions']==['turn off','turn on']
    assert body['provenance']['context_review']==args[0]['provenance']['context_review']
    assert payload['record']['usage_constraints']==args[1]['record']['usage_constraints']
    assert payload['ledger']['construction_revision']['preimage']==dict(body=args[0],payload=args[1])
    assert CorpusTemplate(payload['record']).render()['text']==body['example']
    assert payload['ledger']['template_id']==payload['record']['template_id']


@pytest.mark.parametrize('defect',['stale','context','version','unexpected_group','unexpected_time','approval','published','gold','mixed_steps'])
def test_construction_refresh_refuses_approval_or_silent_scope_rewrite(sample,defect):
    args=list(construction_case(sample));step=args[3]['steps'][0]
    if defect=='stale':step['preimage']='stale'
    elif defect=='context':step['context_sha256']='stale'
    elif defect=='version':step['extractor_version']='0'
    elif defect=='unexpected_group':step['expected_constructions']['v0']=['turn on']
    elif defect=='unexpected_time':step['expected_temporal_adjuncts']=[{}]
    elif defect=='approval':step['semantic_approval']=True
    elif defect=='mixed_steps':args[3]['steps'].append(deepcopy(step))
    else:
        if defect=='published':args[1]['record']['usage_constraints']['mode']='attested_sentence'
        else:args[1]['review_status']='REVIEWED_REUSE'
        args[3]['preimage']=step['preimage']=fingerprint(dict(body=args[0],payload=args[1]))
    with pytest.raises(ValueError):apply_bundle(*args)
