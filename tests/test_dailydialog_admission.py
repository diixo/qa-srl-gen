"""Exercise selection, real bank validation, publication and offline rendering."""
import json
import sqlite3
from copy import deepcopy
import pytest

from scripts import admit_dailydialog_hybrid as admission
from scripts.revise_dailydialog_hybrid_context import apply_review
from semantic_corpus.template_bank import TemplateBank
from semantic_corpus.template_bank import fingerprint
from semantic_corpus.resource_pool import BuiltInPool
from tests.test_dailydialog_context_revision import sample


def promotion(sample):
    body,payload,context,review,audit,vocab,_=sample
    review=deepcopy(review)
    review['usage']=dict(mode='context_only',reasons=['Command; no factual action or new combinations.'])
    body,payload=apply_review(body,payload,context,review,audit,vocab)
    decision=dict(review,method='direct_assistant_attestation_admission',token_edits=[],
        source_sha256=fingerprint(body['example']),payload_sha256=fingerprint(payload),
        context_sha256=fingerprint(context),
        previous_context_review_sha256=fingerprint(body['provenance']['context_review']),
        usage={**review['usage'],'mode':'attested_sentence'},semantic_approval=False,publication=False,
        note='Explicit admission of the exact sentence with its retained command interpretation.')
    return body,payload,context,decision


def test_explicit_attestation_promotion_preserves_previous_meaning_and_evidence(sample):
    args=promotion(sample);before=deepcopy(args)
    b,p=admission.apply_admission(*args)
    assert args==before
    assert b['provenance']['context_review_history'][-1]==args[0]['provenance']['context_review']
    assert p['record']['usage_constraints']=={**args[1]['record']['usage_constraints'],'mode':'attested_sentence'}
    assert p['ledger']['tokens']==args[1]['ledger']['tokens']
    assert p['ledger']['attestation_admission']['preimage']==dict(body=args[0],payload=args[1])
    assert p['ledger']['attestation_admission']['review']==args[3]


@pytest.mark.parametrize('defect',['previous','source','context','payload','meaning','usage','tokens',
                                 'unit','old_unit','old_mode','reference','approval','publication','evidence','method',
                                 'old_meaning','embedded_context','source_id'])
def test_attestation_promotion_rejects_stale_or_expanded_permission(sample,defect):
    body,payload,context,review=promotion(sample)
    if defect=='previous':review['previous_context_review_sha256']='wrong'
    elif defect in ('source','context','payload'):review[defect+'_sha256']='wrong'
    elif defect=='meaning':review['meaning']['completed_action_asserted']=True
    elif defect=='usage':review['usage']['reasons']=[]
    elif defect=='tokens':review['token_edits']=[dict(token=0,set={'dep':'ROOT'})]
    elif defect=='unit':review['unit']='fragment'
    elif defect=='old_unit':body['provenance']['context_review']['unit']='fragment'
    elif defect=='old_mode':payload['record']['usage_constraints']['mode']='attested_sentence'
    elif defect=='reference':payload['review_status']='REVIEWED_REUSE'
    elif defect in ('approval','publication'):review['semantic_approval' if defect=='approval' else defect]=True
    elif defect=='evidence':review['evidence_turns']=[]
    elif defect=='method':review['method']='direct_assistant_review'
    elif defect=='old_meaning':body['provenance']['context_review']['meaning']={'event_status':'different'}
    elif defect=='embedded_context':payload['record']['usage_constraints']['context']=['changed']
    elif defect=='source_id':payload['record']['usage_constraints']['source_id']='another:1:1:1'
    with pytest.raises(ValueError):admission.apply_admission(body,payload,context,review)


@pytest.mark.parametrize('multiple', [False, True])
def test_admission_publishes_only_reviewed_attestations_with_embedded_context(sample,tmp_path,monkeypatch,multiple):
    body,payload,context,review,audit,vocab,lexicon=sample
    review['usage']=dict(mode='attested_sentence',reasons=['command_not_completed_action'])
    body,payload=apply_review(body,payload,context,review,audit,vocab)
    source=tmp_path/'revision';source.mkdir()
    database=source/'revision.sqlite3'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE occurrences(id TEXT,key TEXT,body TEXT)')
        db.execute('CREATE TABLE parses(key TEXT,payload TEXT)')
        db.execute('CREATE TABLE corrections(id TEXT,review TEXT)')
        db.execute('INSERT INTO occurrences VALUES(?,?,?)',(body['idx'],'a',json.dumps(body)))
        db.execute('INSERT INTO parses VALUES(?,?)',('a',json.dumps(payload)))
        db.execute('INSERT INTO corrections VALUES(?,?)',(body['idx'],json.dumps(review)))
    db.close()
    monkeypatch.setattr(admission,'open_overlay',lambda *_:(sqlite3.connect(database.as_uri()+'?mode=ro',uri=True),))
    reviews=tmp_path/'reviews.jsonl'
    # A held-back fragment is not compiled, even if its address is otherwise available.
    reviews.write_text(json.dumps(review)+'\n'+json.dumps(dict(id='test:2:1:1',usage={'mode':'context_only'}))+'\n')
    if multiple:
        second = tmp_path/'held-reviews.jsonl'
        first, held = reviews.read_text().splitlines()
        reviews.write_text(first+'\n')
        second.write_text(held+'\n')
        reviews = [reviews, second]
    repo=tmp_path/'repo'
    (repo/'data').mkdir(parents=True)
    lexicon.rename(repo/'data/wiktionary')
    out=tmp_path/'admission'
    report=admission.compile_admission(source,reviews,out,repo)
    assert report['validation']['source_records']==1
    assert report['held_back']==['test:2:1:1']
    assert not (repo/'data/mined-templates').exists()
    result=admission.publish(out,'test-context',repo)
    assert result['published']
    saved=(repo/'data/mined-templates/manifest.json').read_bytes()
    admission.publish(out,'test-context',repo)
    assert (repo/'data/mined-templates/manifest.json').read_bytes()==saved
    # Runtime rendering needs neither the input source nor the review staging bank.
    out.rename(tmp_path/'no-longer-at-original-path')
    source.rename(tmp_path/'removed-source-path')
    bank=TemplateBank(repo/'data/mined-templates')
    outputs=list(bank.generate(count=1,recombine=True))
    assert len(outputs)==1 and outputs[0]['text']==body['example']
    assert outputs[0]['usage_constraints']['context']==context
    assert not (bank.path/'sources/test-context.jsonl').exists()
    pool=BuiltInPool(repo)
    docs=[doc for split in ('train','dev','test') for doc in pool.mined_documents(split=split,recombine=True)]
    assert len(docs)==1 and docs[0].metadata['recombined'] is False
    assert docs[0].metadata['usage_constraints']['context']==context


@pytest.mark.parametrize('damage', [None, 'open_warning', 'stale_context', 'wrong_candidate', 'syntax_edit', 'wrong_count'])
@pytest.mark.parametrize('prior_review', [False, True])
def test_contextual_subset_admission_preserves_candidate_and_blocks_quarantine(sample,tmp_path,monkeypatch,damage,prior_review):
    import gzip
    body,payload,context,review,audit,vocab,lexicon=sample
    if prior_review:
        body,payload,context,review=promotion(sample)
    else:
        review['usage']=dict(mode='attested_sentence',reasons=['imperative_not_completed_action'])
        review['token_edits']=[]
    source=tmp_path/'candidate';source.mkdir()
    database=source/'revision.sqlite3'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE occurrences(id TEXT,key TEXT,body TEXT)')
        db.execute('CREATE TABLE parses(key TEXT,payload TEXT)')
        db.execute('INSERT INTO occurrences VALUES(?,?,?)',(body['idx'],'a',json.dumps(body)))
        db.execute('INSERT INTO parses VALUES(?,?)',('a',json.dumps(payload)))
    before=database.read_bytes()
    (source/'plan.json').write_text('{}')
    (source/'report.json').write_text(json.dumps(dict(validation={'status':'passed'},conservation={'status':'passed'})))
    with gzip.open(source/'contexts.jsonl.gz','wt',encoding='utf-8') as f:
        f.write(json.dumps(dict(id=body['provenance']['context_id'],turns=context))+'\n')
    monkeypatch.setattr(admission,'open_overlay',lambda *_:(sqlite3.connect(database.as_uri()+'?mode=ro',uri=True),))
    quarantine=tmp_path/'quarantine';quarantine.mkdir()
    cases=[]
    if damage=='open_warning':
        image=dict(body=body,payload=payload)
        cases=[dict(id=body['idx'],image=image,image_sha256=fingerprint(image))]
    summary=dict(candidate=str(source if damage!='wrong_candidate' else tmp_path/'elsewhere'),
                 quarantined_current_occurrences=len(cases)+(damage=='wrong_count'))
    (quarantine/'summary.json').write_text(json.dumps(summary))
    with gzip.open(quarantine/'quarantine-cases.jsonl.gz','wt',encoding='utf-8') as f:
        for case in cases:f.write(json.dumps(case)+'\n')
    if damage=='stale_context':review['context_sha256']='outdated'
    if damage=='syntax_edit':review['token_edits']=[dict(token=0,set={'dep':'ROOT'})]
    reviews=tmp_path/'reviews.jsonl';reviews.write_text(json.dumps(review)+'\n')
    repo=tmp_path/'repo';(repo/'data').mkdir(parents=True)
    lexicon.rename(repo/'data/wiktionary')
    out=tmp_path/'admitted'
    if damage:
        with pytest.raises(ValueError):
            admission.compile_admission(source,reviews,out,repo,quarantine=quarantine)
    else:
        report=admission.compile_admission(source,reviews,out,repo,quarantine=quarantine)
        assert report['contextual_reviews']==1 and report['candidate_unchanged']
        assert report['admitted_occurrences']==[body['idx']]
        with gzip.open(out/'contextual-reviews.jsonl.gz','rt',encoding='utf-8') as f:
            snapshot=json.loads(next(f))
        assert snapshot['preimage']==dict(body=body,payload=payload)
        assert snapshot['context']==context
        row=next(TemplateBank(out/'bank').generate(count=1,recombine=True))
        assert row['text']==body['example']
        assert row['usage_constraints']['meaning']==review['meaning']
        assert row['usage_constraints']['mode']=='attested_sentence'
        assert not (repo/'data/mined-templates').exists()
    assert database.read_bytes()==before
