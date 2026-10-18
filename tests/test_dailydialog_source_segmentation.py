"""Source editing/splitting retains exact coverage, prior evidence and restrictions."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pytest

from scripts import resegment_dailydialog_hybrid_source as revision
from scripts import process_dailydialog_hybrid as full
from scripts.audit_dailydialog_hybrid_bank import open_overlay
from scripts.import_dailydialog_sentences import digest
from semantic_corpus.template_bank import CorpusTemplate, fingerprint, read_rows
from semantic_corpus.template_compiler import LexicalAudit, compile_record


@pytest.fixture
def sample(tmp_path):
    spacy=pytest.importorskip('spacy')
    from spacy.tokens import Doc
    lexicon=tmp_path/'lexicon';lexicon.mkdir()
    (lexicon/'en_verb_inflections.txt').write_text('be\tis\tbeing\twas\tbeen\ngo\tgoes\tgoing\twent\tgone\n')
    (lexicon/'en_postags_withverb.txt').write_text('be\tverb\ngo\tverb\n')
    audit=LexicalAudit(lexicon)
    doc=Doc(spacy.blank('en').vocab,words=['I','"','m','ready','?','.','Go','!'],
        spaces=[True,True,True,False,False,True,False,False],
        heads=[2,2,2,2,2,2,6,6],deps=['nsubj','punct','ROOT','acomp','punct','punct','ROOT','punct'],
        pos=['PRON','PUNCT','VERB','ADJ','PUNCT','PUNCT','VERB','PUNCT'],
        tags=['PRP','``','VBP','JJ','.','.','VB','.'],lemmas=['I','"','m','ready','?','.','go','!'])
    record,ledger=compile_record(doc,{},audit)
    payload=dict(record=record,ledger=ledger,entities=[],flags=[],variant={},method='local_spacy_syntax',review_status='UNREVIEWED')
    ledger['automatic_revision']={'rules':['old decision'],'previous_payload_sha256':'old'}
    body=dict(idx='train:1:1:1',example=doc.text,provenance=dict(origin=dict(split='train',dialogue=1,turn=1,segment=1,span=[0,len(doc.text)]),
        context_id='train:1',original=doc.text,cache_key='old',boundary_flags=[],context_review=dict(unit='pending',syntax_corrected=False)))
    text="I'm ready? Go!"
    review=dict(id=body['idx'],method='direct_assistant_source_review',preimage=fingerprint(dict(body=body,payload=payload)),
        context_sha256=fingerprint([doc.text]),before_turn=doc.text,after_turn=text,dropped_tokens=[5],tokenization_note='Merge contraction and delete duplicate terminal period.',outputs=[])
    meaning=dict(speech_act='question',event_status='questioned',content={'state':'ready'},completed_action_asserted=False,unknown=[])
    for n,(span,tokens) in enumerate([([0,10],[{'from':[0]},{'from':[1,2],'set':dict(text="'m",parser_lemma='be',pos='AUX',tag='VBP',dep='ROOT',head=1,morph='')},{'from':[3]},{'from':[4]}]),([11,14],[{'from':[6]},{'from':[7]}])],1):
        review['outputs'].append(dict(id=f'train:1:1:{n}',span=span,tokens=tokens,unit='sentence',meaning=meaning,
            evidence_turns=[1],note='Explicit test review',boundary_flags=[],usage={'mode':'attested_sentence','reasons':['no_factual_qa']}))
    return body,payload,[doc.text],review,audit,doc.vocab


def test_split_and_retokenization_preserve_history_and_context(sample):
    before=deepcopy(sample[:4]);outputs,context=revision.compile_replacement(*sample)
    assert sample[:4]==before
    assert context==["I'm ready? Go!"]
    assert [b['example'] for b,p,r in outputs]==["I'm ready?",'Go!']
    for body,payload,decision in outputs:
        assert payload['ledger']['source_revision']['preimage']==dict(body=sample[0],payload=sample[1])
        assert body['provenance']['context_review_history']==[sample[0]['provenance']['context_review']]
        assert not payload['ledger']['source_revision']['semantic_approval']
        assert CorpusTemplate(payload['record']).render()['text']==body['example']


def internal_segment(sample):
    """Start from an already corrected sentence followed by an unrelated sentence."""
    outputs,context=revision.compile_replacement(*sample)
    body,payload,_=outputs[0]
    review=deepcopy(sample[3]);review.update(
        preimage=fingerprint(dict(body=body,payload=payload)), context_sha256=fingerprint(context),
        before_turn=context[0],after_turn=context[0],dropped_tokens=[])
    review['outputs']=review['outputs'][:1]
    review['outputs'][0]['tokens']=[{'from':[i]} for i in range(len(payload['ledger']['tokens']))]
    review['outputs'][0]['usage']['mode']='context_only'
    return body,payload,context,review,sample[4],sample[5]


def test_internal_context_refinement_keeps_suffix_and_does_not_claim_syntax_edit(sample):
    args=internal_segment(sample)
    outputs,context=revision.compile_replacement(*args)
    body,payload,decision=outputs[0]
    assert context==args[2] and context[0][body['provenance']['origin']['span'][1]:]==' Go!'
    assert decision['token_edits']==[]
    assert not body['provenance']['context_review']['syntax_corrected']
    assert payload['ledger']['source_revision']['preimage']['payload']['ledger']['source_revision']


def test_explicit_entity_removal_preserves_original_evidence(sample):
    args=internal_segment(sample);body,payload,context,review,*_=args
    payload['entities']=[dict(text='ready',label='ORG',span=[4,9],review_status='UNREVIEWED')]
    review['preimage']=fingerprint(dict(body=body,payload=payload))
    review['outputs'][0].update(entities=[],entity_review_note='This is a property, not an organization.')
    outputs,_=revision.compile_replacement(*args)
    result=outputs[0][1]
    assert result['entities']==[]
    assert result['ledger']['source_revision']['preimage']['payload']['entities']==payload['entities']
    assert not result['ledger']['source_revision']['semantic_approval']


def test_explicit_entity_relocation_uses_output_local_span(sample):
    args=whole_turn_review(sample)
    args[3]['outputs'][1].update(entities=[dict(text='Go',label='ORG',span=[0,2],review_status='UNREVIEWED')],
                                entity_review_note='Fixture label relocated to the second output.')
    outputs,_=revision.compile_replacement(*args)
    assert outputs[1][1]['entities']==args[3]['outputs'][1]['entities']
    assert outputs[1][0]['provenance']['origin']['span']==[14,17]


@pytest.mark.parametrize('defect',['missing_note','wrong_text','outside','inside_token','approval','boolean_span'])
def test_explicit_entity_rejects_invalid_evidence(sample,defect):
    args=internal_segment(sample);output=args[3]['outputs'][0]
    entity=dict(text='ready',label='ORG',span=[4,9],review_status='UNREVIEWED')
    output.update(entities=[entity],entity_review_note='Fixture evidence')
    if defect=='missing_note': output.pop('entity_review_note')
    elif defect=='wrong_text': entity['text']='London'
    elif defect=='outside': entity['span']=[40,45]
    elif defect=='inside_token': entity.update(text='eady',span=[5,9])
    elif defect=='approval': entity['review_status']='APPROVED'
    else: entity['span']=[True,9]
    with pytest.raises(ValueError,match='entity'):
        revision.compile_replacement(*args)


def test_duplicate_review_changes_only_its_occurrence(sample):
    args=internal_segment(sample)
    body,payload,context,review,*_=args
    contexts={'train:1':dict(turns=context,duplicate_of='train:2'),
              'train:2':dict(turns=deepcopy(context))}
    review['duplicate_context_of']='train:2'
    before=deepcopy(contexts)
    revision.validate_duplicate_context(contexts,'train:1',review)
    outputs,new_context=revision.compile_replacement(*args)
    assert contexts==before and new_context==context
    assert len(outputs)==1 and outputs[0][0]['idx']=='train:1:1:1'
    assert outputs[0][1]['record']['usage_constraints']['mode']=='context_only'
    assert not outputs[0][0]['provenance']['context_review']['unseen_group_members_approved']


@pytest.mark.parametrize('failure',['unacknowledged','wrong_parent','missing_parent',
                                  'different_context','source_edit','false_duplicate','self_reference'])
def test_duplicate_context_cannot_propagate_review_or_source_edit(sample,failure):
    args=internal_segment(sample);context,review=args[2:4]
    contexts={'train:1':dict(turns=context,duplicate_of='train:2'),
              'train:2':dict(turns=deepcopy(context))}
    review['duplicate_context_of']='train:2'
    if failure=='unacknowledged': review.pop('duplicate_context_of')
    elif failure=='wrong_parent': review['duplicate_context_of']='train:3'
    elif failure=='missing_parent': contexts.pop('train:2')
    elif failure=='different_context': contexts['train:2']['turns'].append('At school.')
    elif failure=='source_edit': review['after_turn']='Changed.'
    elif failure=='false_duplicate': contexts['train:1'].pop('duplicate_of')
    elif failure=='self_reference':
        contexts['train:1']['duplicate_of']='train:1'
        review['duplicate_context_of']='train:1'
    with pytest.raises(ValueError,match='duplicate'): revision.validate_duplicate_context(contexts,'train:1',review)


def test_duplicate_can_reference_preserved_dialogue_outside_candidate(sample,tmp_path):
    args=internal_segment(sample);context,review=args[2:4]
    contexts={'train:1':dict(turns=context,duplicate_of='train:2')}
    path=tmp_path/'data/dailydialog/filtered/dailydialog-train.jsonl';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(context)+'\n'+json.dumps(context)+'\n',encoding='utf-8')
    reference=dict(id='train:2',turns=deepcopy(context),sha256=fingerprint(context),source_file_sha256=digest(path))
    review.update(duplicate_context_of='train:2',duplicate_context_reference=reference)
    before=path.read_bytes()
    revision.validate_duplicate_context(contexts,'train:1',review,repo=tmp_path)
    assert path.read_bytes()==before and set(contexts)=={'train:1'}
    reference['source_file_sha256']='stale'
    with pytest.raises(ValueError,match='stale duplicate'):
        revision.validate_duplicate_context(contexts,'train:1',review,repo=tmp_path)
    reference.update(source_file_sha256=digest(path),turns=['Different.'],sha256=fingerprint(['Different.']))
    with pytest.raises(ValueError,match='does not match source'):
        revision.validate_duplicate_context(contexts,'train:1',review,repo=tmp_path)


@pytest.mark.parametrize('change',['suffix','offset','split'])
def test_internal_edit_cannot_move_unreviewed_sentence(sample,change):
    args=internal_segment(sample);review=args[3]
    if change=='suffix': review['after_turn']=review['after_turn'].replace('Go!','No!')
    elif change=='offset':
        review['after_turn']=review['after_turn'].replace('ready','very ready')
        review['outputs'][0]['span'][1]+=5
    else: review['outputs'].append(deepcopy(review['outputs'][0]))
    with pytest.raises(ValueError,match='suffix'): revision.compile_replacement(*args)


@pytest.mark.parametrize('failure',['preimage','missing_token','duplicate_token','bad_span','cycle','missing_boundary_head','insertion_note'])
def test_invalid_source_review_is_rejected(sample,failure):
    review=sample[3]
    if failure=='preimage': review['preimage']='stale'
    elif failure=='missing_token': review['dropped_tokens']=[]
    elif failure=='duplicate_token': review['dropped_tokens']=[5,5]
    elif failure=='bad_span': review['outputs'][1]['span'][0]=12
    elif failure=='cycle': review['outputs'][0]['tokens'][0]['set']={'head':0}
    elif failure=='missing_boundary_head': review['outputs'][1]['tokens'][0]['set']={'head':10}
    elif failure=='insertion_note': review.pop('tokenization_note')
    with pytest.raises(ValueError): revision.compile_replacement(*sample)


def joined_review(inputs,context,usage,reference_expansions=None):
    body,payload=revision.combine_segments(inputs,context,reference_expansions)
    tokens=[{'from':[i]} for i in range(len(payload['ledger']['tokens']))]
    # In this fixture the second clause is an imperative attached paratactically.
    second=len(inputs[0]['payload']['ledger']['tokens'])
    tokens[second]['set']={'dep':'parataxis','head':1}
    review=dict(id=body['idx'],input_ids=[i['body']['idx'] for i in inputs],
        method='direct_assistant_source_review',preimage=fingerprint(dict(body=body,payload=payload)),
        context_sha256=fingerprint(context),before_turn=context[0],after_turn=context[0],dropped_tokens=[],
        outputs=[dict(id=body['idx'],span=[0,len(context[0])],tokens=tokens,unit='sentence',
            meaning=dict(speech_act='question_and_instruction',event_status='not_completed',content={'state':'ready'},
                         completed_action_asserted=False,unknown=[]),
            evidence_turns=[1],note='Explicit fixture grouping',boundary_flags=[],usage=usage)])
    if reference_expansions:review['reference_expansions']=deepcopy(reference_expansions)
    return body,payload,review


def borrowed_inputs(sample):
    outputs,context=revision.compile_replacement(*sample)
    inputs=[dict(body=b,payload=p) for b,p,r in outputs]
    item=inputs[1];b,p=item['body'],item['payload']
    b['provenance'].pop('context_review')
    b['provenance']['reused_from']='train:2:1:1'
    p.update(method='reviewed_whole_dialogue_reuse',review_status='REVIEWED_REUSE')
    original=deepcopy(p['ledger']['tokens'])
    word={k:v for k,v in original[0].items() if k not in {'i','head'}}
    word['slot']='v0';p['ledger']['tokens']=[word]
    expansion=dict(preimage=fingerprint(item),reused_from='train:2:1:1',
        note='Retain the reviewed verb slot; add its observed exclamation mark.',
        tokens=[{k:v for k,v in t.items() if k in revision.FIELDS|{'i','span'}} for t in original])
    return inputs,context,{b['idx']:expansion}


def test_join_borrowed_reference_retains_original_and_accounts_for_punctuation(sample,tmp_path):
    inputs,context,expansions=borrowed_inputs(sample);before=deepcopy(inputs)
    body,payload,review=joined_review(inputs,context,{'mode':'context_only','reasons':['fixture']},expansions)
    outputs,_=revision.compile_replacement(body,payload,context,review,sample[4],sample[5])
    assert inputs==before
    result=outputs[0][1]
    assert result['ledger']['source_revision']['preimage']['payload']['ledger']['input_segments']==before
    assert CorpusTemplate(result['record']).render()['text']==context[0]
    # Independent conservation compares the old sparse slot ledger, not the
    # expanded virtual parse, so an added punctuation token must be explicit.
    source=tmp_path/'source';out=tmp_path/'out'
    for root in (source,out):(root/'bank/occurrences').mkdir(parents=True)
    with revision.gzip.open(source/'bank/occurrences/00000.jsonl.gz','wt',encoding='utf-8') as stream:
        for i,item in enumerate(before,1):
            revision.write(stream,dict(idx=item['body']['idx'],line=i,text=item['body']['example'],tokens=item['payload']['ledger']['tokens']))
    with revision.gzip.open(out/'bank/occurrences/00000.jsonl.gz','wt',encoding='utf-8') as stream:
        revision.write(stream,dict(idx=body['idx'],line=1,**result['ledger']))
    check=revision.verify_conservation(source,out,[review])
    assert check['explicit_token_delta']==check['reference_expansion_token_delta']==1
    assert check['previous_preimages_retained']


@pytest.mark.parametrize('failure',['missing','stale','wrong_reference','changed_word','changed_head',
                                  'missing_slot','added_word','wrong_text','unused','review_changed'])
def test_borrowed_reference_expansion_cannot_rewrite_or_lose_evidence(sample,failure):
    inputs,context,expansions=borrowed_inputs(sample)
    e=next(iter(expansions.values()))
    if failure=='missing':expansions={}
    elif failure=='stale':e['preimage']='stale'
    elif failure=='wrong_reference':e['reused_from']='train:999:1:1'
    elif failure=='changed_word':e['tokens'][0]['parser_lemma']='stop'
    elif failure=='changed_head':e['tokens'][0]['head']=1
    elif failure=='missing_slot':e['tokens']=e['tokens'][1:]
    elif failure=='added_word':e['tokens'][1]['pos']='NOUN'
    elif failure=='wrong_text':e['tokens'][1]['text']='?'
    elif failure=='unused':expansions['train:7:1:1']=deepcopy(e)
    with pytest.raises(ValueError):
        body,payload,review=joined_review(inputs,context,{'mode':'context_only','reasons':['fixture']},expansions)
        if failure=='review_changed':review.pop('reference_expansions')
        revision.compile_replacement(body,payload,context,review,sample[4],sample[5])


def test_resplitting_fractional_address_stays_before_next_turn():
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE occurrences(rowid REAL,id TEXT)')
        db.executemany('INSERT INTO occurrences VALUES(?,?)',[(1,'first'),(1.75,'previous_split'),(2,'next_turn')])
        positions=revision.insertion_positions(db,1.75,3)
        db.executemany('INSERT INTO occurrences VALUES(?,?)',zip(positions,['new1','new2','new3']))
        assert [i for i, in db.execute('SELECT id FROM occurrences ORDER BY rowid')]==[
            'first','previous_split','new1','new2','new3','next_turn']
        assert revision.insertion_positions(db,2,0)==[]
        assert 2<revision.insertion_positions(db,2,1)[0]<3


def test_exhausted_address_precision_is_rejected_instead_of_reordering():
    from math import nextafter
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE occurrences(rowid REAL)')
        db.executemany('INSERT INTO occurrences VALUES(?)',[(1.0,),(nextafter(1.0,2.0),)])
        with pytest.raises(ValueError,match='ordering precision'):
            revision.insertion_positions(db,1.0,1)


@pytest.mark.parametrize('failure',['different_turn','gap','overlap','reordered','reference','stale_preimage','source_edit','missing_input'])
def test_join_rejects_incomplete_or_changed_evidence(sample,failure):
    outputs,context=revision.compile_replacement(*sample)
    inputs=[dict(body=b,payload=p) for b,p,r in outputs]
    if failure=='different_turn': inputs[1]['body']['provenance']['origin']['turn']=2
    elif failure=='gap': inputs[1]['body']['provenance']['origin']['segment']=3
    elif failure=='overlap': inputs[1]['body']['provenance']['origin']['span'][0]=5
    elif failure=='reordered': inputs.reverse()
    elif failure=='reference': inputs[1]['payload']['method']='reviewed_reference'
    with pytest.raises(ValueError):
        body,payload,review=joined_review(inputs,context,{'mode':'context_only','reasons':['fixture']})
        if failure=='stale_preimage': review['preimage']='stale'
        elif failure=='source_edit': review['after_turn']='Changed.'
        elif failure=='missing_input': review['input_ids'].append('train:1:1:3')
        revision.compile_replacement(body,payload,context,review,sample[4],sample[5])


def whole_turn_review(sample):
    emitted, context = revision.compile_replacement(*sample)
    inputs = [dict(body=b, payload=p) for b,p,r in emitted]
    body, payload, review = joined_review(inputs, context, {'mode':'context_only','reasons':['fixture']})
    review.update(whole_turn_source_edit=True, after_turn=context[0].replace('ready','prepared'))
    review['outputs'] = []
    offset = 0
    for i, ((b,p,r), span) in enumerate(zip(emitted, ([0,13],[14,17]))):
        output = deepcopy(r)
        output.pop('token_edits', None)
        output.update(span=span, tokens=[{'from':[offset+j]} for j in range(len(p['ledger']['tokens']))])
        if i == 0:
            output['tokens'][2]['set'] = {'text':'prepared','parser_lemma':'prepared'}
        review['outputs'].append(output)
        offset += len(p['ledger']['tokens'])
    return body, payload, context, review, sample[4], sample[5]


def test_explicit_whole_turn_edit_moves_addresses_without_renumbering(sample):
    args = whole_turn_review(sample)
    before = deepcopy(args[:4])
    outputs, context = revision.compile_replacement(*args)
    assert args[:4] == before
    assert context == ["I'm prepared? Go!"]
    assert [b['idx'] for b,p,r in outputs] == ['train:1:1:1','train:1:1:2']
    assert [b['provenance']['origin']['span'] for b,p,r in outputs] == [[0,13],[14,17]]
    original = args[1]['ledger']['input_segments'][1]
    assert outputs[1][1]['ledger']['tokens'] == original['payload']['ledger']['tokens']
    assert outputs[1][1]['ledger']['source_revision']['preimage']['payload']['ledger']['input_segments'][1] == original


@pytest.mark.parametrize('defect', ['flag','partial_turn','changed_ids','lost_output'])
def test_grouped_source_edit_requires_full_coverage_and_preserved_ids(sample, defect):
    args = whole_turn_review(sample); body,payload,context,review,*_ = args
    if defect == 'flag': review.pop('whole_turn_source_edit')
    elif defect == 'partial_turn':
        context[0] += ' Later.'
        review.update(context_sha256=fingerprint(context), before_turn=context[0], after_turn=review['after_turn']+' Later.')
    elif defect == 'changed_ids': review['outputs'][1]['id']='train:1:1:3'
    else: review['outputs'].pop()
    with pytest.raises(ValueError, match='complete turn'):
        revision.compile_replacement(*args)


def duplicate_source_batch(sample):
    context=sample[2]
    contexts={'train:1':dict(turns=deepcopy(context),duplicate_of='train:2'),
              'train:2':dict(turns=deepcopy(context))}
    reviews=[deepcopy(sample[3]) for _ in range(2)]
    for i,r in enumerate(reviews,1):
        r.update(id=f'train:{i}:1:1',source_duplicate_group=['train:1','train:2'])
    reviews[0]['duplicate_context_of']='train:2'
    return contexts,reviews


def test_duplicate_source_batch_validates_both_original_contexts_without_mutation(sample):
    contexts,reviews=duplicate_source_batch(sample);before=deepcopy((contexts,reviews))
    revision.validate_review_contexts(contexts,reviews)
    assert (contexts,reviews)==before


@pytest.mark.parametrize('defect',['missing_review','different_change','different_turn','missing_group','stale_context'])
def test_duplicate_source_batch_rejects_partial_or_inconsistent_update(sample,defect):
    contexts,reviews=duplicate_source_batch(sample)
    if defect=='missing_review': reviews.pop()
    elif defect=='different_change': reviews[1]['after_turn']='Different.'
    elif defect=='different_turn': reviews[1]['id']='train:2:2:1'
    elif defect=='missing_group': reviews[1].pop('source_duplicate_group')
    else: reviews[0]['context_sha256']='stale'
    with pytest.raises(ValueError,match='duplicate'):
        revision.validate_review_contexts(contexts,reviews)


def test_source_preparation_rejects_unreviewed_duplicate_outside_candidate(sample,tmp_path):
    root=tmp_path/'data/dailydialog';parts=root/'review-parts';filtered=root/'filtered'
    parts.mkdir(parents=True);filtered.mkdir()
    context=sample[2];review=sample[3]
    for split in ('train','test','val'):
        (filtered/f'dailydialog-{split}.jsonl').write_text((json.dumps(context)+'\n')*2 if split=='train' else '',encoding='utf-8')
    revision.dump(parts/'manifest.json',dict(parts=[],sources={s:digest(filtered/f'dailydialog-{s}.jsonl') for s in ('train','test','val')}))
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='omitted an identical dialogue'):
        revision.source_updates(tmp_path,[review])
    assert before=={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('defect',[None,'missing_snapshot','bad_hash','wrong_turn','wrong_part','duplicate'])
def test_source_version_snapshot_preserves_unrelated_earlier_wording(tmp_path,defect):
    root=tmp_path/'data/dailydialog';parts=root/'review-parts';filtered=root/'filtered'
    parts.mkdir(parents=True);filtered.mkdir()
    source=['Earlier typo.',"I \\'d try."]
    candidate=['Earlier correction.',source[1]]
    untouched=['Another dialogue.']
    rows=[source,source if defect=='duplicate' else untouched]
    for split in ('train','val','test'):
        (filtered/f'dailydialog-{split}.jsonl').write_text(
            ''.join(json.dumps(r)+'\n' for r in rows) if split=='train' else '',encoding='utf-8')
    part=parts/'part-001.jsonl'
    part_rows=[dict(split='train',dialogue=i,turns=r) for i,r in enumerate(rows,1)]
    if defect=='wrong_part':part_rows[0]['turns']=['Changed independently.',source[1]]
    part.write_text(''.join(json.dumps(r)+'\n' for r in part_rows),encoding='utf-8')
    revision.dump(parts/'manifest.json',dict(sources={s:digest(filtered/f'dailydialog-{s}.jsonl') for s in ('train','val','test')},
        parts=[dict(file=part.name,sha256=digest(part))]))
    review=dict(id='train:1:2:1',context_sha256=fingerprint(candidate),before_turn=source[1],after_turn="I'd try.",
        source_context_snapshot=dict(turns=source,sha256=fingerprint(source),note='Candidate-only correction in turn 1; preserve raw turn 1.'))
    if defect=='missing_snapshot':review.pop('source_context_snapshot')
    elif defect=='bad_hash':review['source_context_snapshot']['sha256']='stale'
    elif defect=='wrong_turn':review['before_turn']='Different target.'
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    if defect:
        with pytest.raises(ValueError):revision.source_updates(tmp_path,[review])
    else:
        changes=revision.source_updates(tmp_path,[review])
        updated=[json.loads(s) for s in changes[filtered/'dailydialog-train.jsonl'][1].splitlines()]
        assert updated==[['Earlier typo.',"I'd try."],untouched]
        assert json.loads(changes[part][1].splitlines()[0])['turns']==updated[0]
        assert changes[filtered/'dailydialog-train.jsonl'][1].splitlines()[1]==before[str(filtered/'dailydialog-train.jsonl')].splitlines()[1]
    assert before=={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('rewrite_whole', [False, True])
def test_full_rebuild_and_next_overlay_keep_added_sentences(sample,tmp_path,monkeypatch,rewrite_whole):
    body,payload,context,review,audit,vocab=sample
    repo=tmp_path/'repo';(repo/'data').mkdir(parents=True)
    (tmp_path/'lexicon').rename(repo/'data/wiktionary')
    source=repo/'artifacts/source';source.mkdir(parents=True)
    original=repo/'artifacts/original';original.mkdir()
    lexical=repo/'artifacts/lexical';lexical.mkdir()
    with sqlite3.connect(original/'cache.sqlite3') as db:
        db.execute('CREATE TABLE parses(key TEXT,text TEXT,payload TEXT)')
        db.execute('CREATE TABLE occurrences(id TEXT,key TEXT,body TEXT)')
        db.execute('INSERT INTO parses VALUES(?,?,?)',('old',body['example'],json.dumps(payload)))
        db.execute('INSERT INTO occurrences VALUES(?,?,?)',(body['idx'],'old',json.dumps(body)))
    with sqlite3.connect(lexical/'revision.sqlite3') as db: db.execute('CREATE TABLE replacements(key TEXT,payload TEXT)')
    revision.dump(lexical/'plan.json',dict(source=str(original),source_cache_sha256=digest(original/'cache.sqlite3'),lexical_resources={}))
    revision.dump(source/'plan.json',dict(source=str(lexical)))
    with sqlite3.connect(source/'revision.sqlite3') as db:
        db.execute('CREATE TABLE corrections(id TEXT PRIMARY KEY,key TEXT,text TEXT,body TEXT,payload TEXT,review TEXT)')
    # Materialize the previous candidate through the real compiler/validator.
    temp=repo/'artifacts/base';temp.mkdir()
    db,*_=open_overlay(source,temp)
    coverage=[dict(id='train:1:1',original=body['example'],segments=[dict(id=body['idx'],span=[0,len(body['example'])],text=body['example'])])]
    for name,rows in [('contexts',[dict(id='train:1',turns=context)]),('coverage',coverage)]:
        with revision.gzip.open(source/f'{name}.jsonl.gz','wt',encoding='utf-8') as stream:
            for row in rows: revision.write(stream,row)
    (source/'reviews.jsonl').write_text('')
    full.export(db,source);full.materialize(db,source,{'name':'fixture'},repo/'data/wiktionary');db.close()
    (source/'bank/review').mkdir()
    filtered=repo/'data/dailydialog/filtered';filtered.mkdir(parents=True)
    parts=repo/'data/dailydialog/review-parts';parts.mkdir()
    for split in ('train','val','test'):
        (filtered/f'dailydialog-{split}.jsonl').write_text(json.dumps(context)+'\n' if split=='train' else '')
    part=parts/'part-001.jsonl';part.write_text(json.dumps(dict(split='train',dialogue=1,turns=context))+'\n')
    revision.dump(parts/'manifest.json',dict(sources={s:digest(filtered/f'dailydialog-{s}.jsonl') for s in ('train','val','test')},parts=[dict(file=part.name,sha256=digest(part))]))
    reviews=parts/'reviews.jsonl';reviews.write_text(json.dumps(review)+'\n')
    monkeypatch.setattr(full,'load_reference',lambda _:[])
    out=repo/'artifacts/revised'
    report=revision.run(source,reviews,out,repo)
    assert report['validation']['source_records']==2
    assert report['conservation']['explicit_token_delta']==-2
    assert list(read_rows(part))[0]['turns']==["I'm ready? Go!"]
    assert list(read_rows(filtered/'dailydialog-train.jsonl'))==[["I'm ready? Go!"]]
    later=repo/'artifacts/later';later.mkdir()
    db,*_=open_overlay(out,later)
    assert [i for i, in db.execute('SELECT id FROM occurrences ORDER BY rowid')]==['train:1:1:1','train:1:1:2']
    db.close()
    # Protected old bank remains byte-for-byte intact, including its old text.
    assert list(read_rows(source/'sentences.jsonl'))[0]['example']==body['example']
    # Join the newly created adjacent addresses, retire one explicitly, and reopen twice.
    emitted,new_context=revision.compile_replacement(*sample)
    if rewrite_whole:
        args=whole_turn_review(sample)
        turn_path=parts/'whole-turn.jsonl';turn_path.write_text(json.dumps(args[3])+'\n')
        whole=repo/'artifacts/whole-turn'
        whole_report=revision.run(out,turn_path,whole,repo)
        assert whole_report['validation']['source_records']==2
        assert whole_report['conservation']['explicit_token_delta']==0
        assert whole_report['conservation']['counts'].get('joined_input_segments',0)==0
        assert list(read_rows(part))[0]['turns']==["I'm prepared? Go!"]
        emitted,new_context=revision.compile_replacement(*args)
        out=whole
    inputs=[dict(body=b,payload=p) for b,p,r in emitted]
    _,_,group=joined_review(inputs,new_context,{'mode':'context_only','reasons':['fixture']})
    group_path=parts/'group.jsonl';group_path.write_text(json.dumps(group)+'\n')
    joined=repo/'artifacts/joined'
    joined_report=revision.run(out,group_path,joined,repo)
    assert joined_report['validation']['source_records']==1
    assert joined_report['conservation']['counts']['joined_input_segments']==1
    assert joined_report['conservation']['explicit_token_delta']==0
    assert joined_report['source_files_updated']==0
    reopened=repo/'artifacts/reopened';reopened.mkdir()
    db,*_=open_overlay(joined,reopened)
    assert db.execute('SELECT id FROM occurrences').fetchall()==[('train:1:1:1',)]
    retired=db.execute('SELECT snapshot FROM retired_occurrences WHERE id=?',('train:1:1:2',)).fetchone()
    assert json.loads(retired[0])==inputs[1]
    db.close()
    usage=list(read_rows(joined/'reviews.jsonl'))
    assert len(usage)==1 and usage[0]['id']=='train:1:1:1'
    # Retired evidence also survives another overlay generation.
    revision.dump(reopened/'plan.json',dict(source=str(joined)))
    again=repo/'artifacts/reopened-again';again.mkdir()
    db,*_=open_overlay(reopened,again)
    assert db.execute('SELECT id FROM occurrences').fetchall()==[('train:1:1:1',)]
    assert db.execute('SELECT snapshot FROM retired_occurrences').fetchone()==retired
    db.close()
