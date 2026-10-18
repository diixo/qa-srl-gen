from copy import deepcopy

import pytest

from scripts.finalize_dailydialog_quarantine import resolve, current_ids
from scripts.apply_dailydialog_quarantine_reviews import apply_bundle
from tests.test_dailydialog_context_revision import sample
from tests.test_dailydialog_quarantine_reviews import pending
from semantic_corpus.template_bank import fingerprint


def local_revision(tmp_path):
    import json
    from scripts.import_dailydialog_sentences import digest
    parent,candidate=tmp_path/'parent',tmp_path/'candidate'
    parent.mkdir();candidate.mkdir()
    protected={}
    for name in ('plan.json','report.json','revision.sqlite3','contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'):
        (parent/name).write_bytes(b'original')
        protected[str(parent/name)]=digest(parent/name)
        if name in ('contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'):
            (candidate/name).write_bytes(b'original')
    applied=tmp_path/'applied.jsonl'
    applied.write_text(json.dumps({'id':'test:1:2:1'})+'\n',encoding='utf8')
    plan=dict(version='explicit_quarantine_review_v1',source=str(parent),reviews=str(applied),
              review_sha256=digest(applied),protected=protected)
    (candidate/'plan.json').write_text(json.dumps(plan),encoding='utf8')
    report=dict(validation={'status':'passed'},conservation=dict(status='passed',source_unchanged=True,
        all_unaddressed_occurrences_identical=True,counts={'revised_occurrences':1}))
    (candidate/'report.json').write_text(json.dumps(report),encoding='utf8')
    return candidate,dict(candidate=str(parent))


def test_local_revision_binds_parent_and_keeps_identity(tmp_path):
    from scripts.finalize_dailydialog_quarantine import revision_inputs
    candidate,previous=local_revision(tmp_path)
    mapping,changed,patched=revision_inputs(candidate,None,previous)
    assert not mapping and not changed and patched=={'test:1:2:1'}


def test_source_and_local_revision_still_rebind_sentence_descendants(tmp_path):
    import json
    from scripts.finalize_dailydialog_quarantine import revision_inputs
    from scripts.import_dailydialog_sentences import digest
    candidate,previous=local_revision(tmp_path)
    source=tmp_path/'source';source.mkdir()
    wording=tmp_path/'wording.jsonl';wording.write_text('source review',encoding='utf8')
    (source/'plan.json').write_text(json.dumps(dict(source=previous['candidate'],
        review_sha256={str(wording):digest(wording)})),encoding='utf8')
    lineage={'test:1:2:1':dict(input_ids=['test:1:2:1'],token_analysis='reparsed_unreviewed'),
             'test:1:2:2':dict(input_ids=['test:1:2:1'],token_analysis='reparsed_unreviewed')}
    (source/'lineage.json').write_text(json.dumps(lineage),encoding='utf8')
    for name in ('contexts.jsonl.gz','coverage.jsonl.gz'):(source/name).write_bytes(b'revised source')
    plan=json.loads((candidate/'plan.json').read_text(encoding='utf8'));plan['source']=str(source)
    plan['protected']={str(source/n):digest(source/n) for n in ('lineage.json','contexts.jsonl.gz','coverage.jsonl.gz')}
    (candidate/'plan.json').write_text(json.dumps(plan),encoding='utf8')
    mapping,changed,patched=revision_inputs(candidate,source,previous)
    assert mapping=={'test:1:2:1':['test:1:2:1','test:1:2:2']}
    assert changed==patched=={'test:1:2:1'}


@pytest.mark.parametrize('defect',[None,'parent','decisions','conservation','unrelated'])
def test_wording_revision_invalidates_context_dependent_closures(tmp_path,defect):
    import json
    from pathlib import Path
    from scripts.finalize_dailydialog_quarantine import revision_inputs
    from scripts.import_dailydialog_sentences import digest
    candidate,previous=local_revision(tmp_path)
    plan=json.loads((candidate/'plan.json').read_text(encoding='utf8'))
    applied=Path(plan.pop('reviews'))
    plan.update(version='explicit_candidate_wording_v1',review_sha256={str(applied):digest(applied)})
    (candidate/'plan.json').write_text(json.dumps(plan),encoding='utf8')
    lineage={'test:1:2:1':dict(input_ids=['test:1:2:1'],token_analysis='reparsed_unreviewed'),
             'test:1:3:1':dict(input_ids=['test:1:3:1'],token_analysis='preserved',context_review_invalidated=True),
             'test:1:2:2':dict(input_ids=['test:1:2:2'],token_analysis='preserved')}
    if defect=='unrelated':lineage['test:1:3:1']['input_ids']=['test:9:3:1']
    (candidate/'lineage.json').write_text(json.dumps(lineage),encoding='utf8')
    report=dict(conservation=dict(status='passed',every_original_accounted_for=True,
        all_unaddressed_analyses_unchanged=True,source_preimages_and_previous_local_decisions_preserved=defect!='conservation'))
    (candidate/'report.json').write_text(json.dumps(report),encoding='utf8')
    if defect=='parent':(Path(previous['candidate'])/'contexts.jsonl.gz').write_bytes(b'changed')
    if defect=='decisions':applied.write_text('changed',encoding='utf8')
    if defect:
        with pytest.raises(ValueError):revision_inputs(candidate,None,previous)
    else:
        mapping,changed,patched=revision_inputs(candidate,None,previous)
        assert changed=={'test:1:2:1','test:1:3:1'} and not patched
        assert mapping=={i:[i] for i in lineage}


@pytest.mark.parametrize('defect',[None,'parent','source','reviews','history','count','unrelated','duplicate'])
def test_direct_source_revision_rebinds_retired_quotes_without_losing_evidence(tmp_path,defect):
    import json
    from pathlib import Path
    from scripts.finalize_dailydialog_quarantine import revision_inputs
    from scripts.import_dailydialog_sentences import digest
    candidate,previous=local_revision(tmp_path)
    plan=json.loads((candidate/'plan.json').read_text(encoding='utf8'))
    plan.update(version='reviewed_source_segmentation_v6',frozen=plan.pop('protected'))
    review=dict(id='test:1:2:1',method='direct_assistant_source_review',
        input_ids=['test:1:2:1','test:1:2:2'],outputs=[dict(id='test:1:2:1')])
    if defect=='unrelated':review['outputs'][0]['id']='test:9:2:1'
    if defect=='duplicate':review['input_ids'].append('test:1:2:2')
    applied=Path(plan['reviews']);applied.write_text(json.dumps(review)+'\n',encoding='utf8')
    plan['review_sha256']=digest(applied)
    (candidate/'plan.json').write_text(json.dumps(plan),encoding='utf8')
    report=dict(conservation=dict(status='passed',previous_preimages_retained=defect!='history',
        counts=dict(old_segments=12,new_segments=11,revised_outputs=1,
                    unchanged_analyses=9 if defect=='count' else 10)))
    (candidate/'report.json').write_text(json.dumps(report),encoding='utf8')
    if defect=='parent':previous['candidate']=str(tmp_path/'another')
    elif defect=='source':(Path(previous['candidate'])/'contexts.jsonl.gz').write_bytes(b'changed')
    elif defect=='reviews':applied.write_text('{}\n',encoding='utf8')
    if defect:
        with pytest.raises(ValueError):revision_inputs(candidate,None,previous)
    else:
        mapping,changed,patched=revision_inputs(candidate,None,previous)
        assert mapping=={'test:1:2:1':['test:1:2:1'],'test:1:2:2':['test:1:2:1']}
        assert changed=={'test:1:2:1','test:1:2:2'} and patched=={'test:1:2:1'}


@pytest.mark.parametrize('defect',['parent','source','context','reviews','conservation','count','version'])
def test_local_revision_rejects_stale_or_unconserved_lineage(tmp_path,defect):
    import json
    from pathlib import Path
    from scripts.finalize_dailydialog_quarantine import revision_inputs
    candidate,previous=local_revision(tmp_path)
    if defect=='parent':previous['candidate']=str(tmp_path/'another')
    elif defect=='source':(Path(previous['candidate'])/'revision.sqlite3').write_bytes(b'changed')
    elif defect=='context':(candidate/'contexts.jsonl.gz').write_bytes(b'changed')
    elif defect=='reviews':(tmp_path/'applied.jsonl').write_text('{}\n',encoding='utf8')
    elif defect=='version':
        plan=json.loads((candidate/'plan.json').read_text(encoding='utf8'));plan['version']='unknown'
        (candidate/'plan.json').write_text(json.dumps(plan),encoding='utf8')
    else:
        report=json.loads((candidate/'report.json').read_text(encoding='utf8'))
        if defect=='count':report['conservation']['counts']['revised_occurrences']=2
        else:report['conservation']['all_unaddressed_occurrences_identical']=False
        (candidate/'report.json').write_text(json.dumps(report),encoding='utf8')
    with pytest.raises(ValueError):revision_inputs(candidate,None,previous)


def test_local_finalization_reopens_changed_cases_without_explicit_recheck(sample,tmp_path,monkeypatch):
    import json,gzip
    from pathlib import Path
    from scripts import finalize_dailydialog_quarantine as module
    from semantic_corpus.template_bank import read_rows
    candidate,previous=local_revision(tmp_path)
    w,review,images,contexts=evidence(sample)
    identity=w['id'];image=images[identity]
    # One explicit resolution must not silently approve another warning on the same parse.
    second=deepcopy(w);second['issue_index']=1;second['status']='fixed_in_candidate'
    ledger=tmp_path/'ledger';ledger.mkdir()
    (ledger/'summary.json').write_text(json.dumps(previous),encoding='utf8')
    for name,rows in [('warning-decisions.jsonl.gz',[w,second]),('received-evidence.jsonl.gz',[{'paid':'kept'}])]:
        with gzip.open(ledger/name,'wt',encoding='utf8') as stream:
            for r in rows:stream.write(json.dumps(r)+'\n')
    with gzip.open(candidate/'contexts.jsonl.gz','wt',encoding='utf8') as stream:
        for k,ctx in contexts.items():stream.write(json.dumps({'id':k,'turns':ctx})+'\n')
    reviews=tmp_path/'rechecks.jsonl';reviews.write_text(json.dumps(review)+'\n',encoding='utf8')
    monkeypatch.setattr(module,'revision_inputs',lambda *a: ({},set(),{identity}))
    class DB:
        def execute(self,query,params):
            assert params==(identity,)
            return self
        def fetchone(self):return json.dumps(image['body']),json.dumps(image['payload'])
        def close(self):pass
    monkeypatch.setattr(module,'open_overlay',lambda *a:(DB(),))
    out=tmp_path/'result'
    result=module.run(candidate,None,ledger,reviews,out)
    assert result['saved_warnings']==2 and result['invalidated_without_new_review']==1
    saved=list(read_rows(out/'warning-decisions.jsonl.gz'))
    assert saved[1]['status']=='quarantined_after_candidate_revision'
    assert saved[1]['previous_disposition']==second
    snapshot=list(read_rows(out/'quarantine-cases.jsonl.gz'))
    assert len(snapshot)==1 and len(snapshot[0]['warnings'])==2
    assert snapshot[0]['image']==image and snapshot[0]['context']==contexts[image['body']['provenance']['context_id']]


def evidence(sample):
    args=pending(sample)
    b,p=apply_bundle(*args)
    image=dict(body=b,payload=p);identity=b['idx'];ctx=args[2]
    issue={'category':'ambiguity','explanation':'Command or warning?'}
    warning=dict(id=identity,issue_index=0,issue_sha256=fingerprint(issue),issue=issue,
                 status='quarantined_unresolved',group='ambiguity')
    proof=dict(id=identity,image_sha256=fingerprint(image),context_sha256=fingerprint(ctx),
               contains=[b['example']],absent=[],token_checks=[])
    review=dict(id=identity,issue_index=0,issue_sha256=warning['issue_sha256'],
                previous_disposition_sha256=fingerprint(warning),
                method='direct_assistant_quarantine_warning_review_v1',
                status='quarantined_unresolved',note='Preserve ambiguity.',outputs=[proof],
                unknown=['command or warning'],semantic_approval=False,publication=False)
    return warning,review,{identity:image},{b['provenance']['context_id']:ctx}


def test_pending_meaning_is_not_counted_as_closed(sample):
    args=evidence(sample);before=deepcopy(args)
    result=resolve(*args)
    assert args==before
    assert result['status']=='quarantined_unresolved'
    assert result['previous_disposition']==args[0]
    assert current_ids(result)==[args[0]['id']]
    assert result['quarantine_recheck']['unknown']==['command or warning']


@pytest.mark.parametrize('field', ['constructions', 'temporal_adjunct_candidates'])
def test_construction_closure_binds_derived_evidence(sample, field):
    args=evidence(sample);w,r,images,_=args;proof=r['outputs'][0]
    p=images[w['id']]['payload']
    proof['constructions']={g['head_slot']:[c['lemma'] for c in g['constructions']]
                            for g in p['record']['predicate_groups']}
    proof['temporal_adjunct_candidates']=p['ledger'].get('temporal_adjunct_candidates', [])
    assert resolve(*args)['quarantine_recheck']['outputs']==[proof]
    proof[field]={'incorrect':'evidence'}
    with pytest.raises(ValueError):resolve(*args)


@pytest.mark.parametrize('defect',['stale','context','token','negative_token','duplicate_token','empty','missing_unknown','unrestricted','approval','evidence'])
def test_finalizer_rejects_missing_or_mismatched_evidence(sample,defect):
    args=evidence(sample);w,r,images,contexts=args;p=r['outputs'][0]
    if defect=='stale':p['image_sha256']='stale'
    elif defect=='context':p['context_sha256']='stale'
    elif defect=='token':p['token_checks']=[dict(i=0,text='Fire',fields={'pos':'VERB'})]
    elif defect=='negative_token':p['token_checks']=[dict(i=-1,text='!',fields={'pos':'PUNCT'})]
    elif defect=='duplicate_token':p['token_checks']=[dict(i=0,text='Fire',fields={'pos':'NOUN'})]*2
    elif defect=='empty':p['contains']=[]
    elif defect=='missing_unknown':r.pop('unknown')
    elif defect=='unrestricted':
        img=images[p['id']];img['payload']['record']['usage_constraints']['mode']='attested_sentence'
        p['image_sha256']=fingerprint(img)
    elif defect=='approval':r['semantic_approval']=True
    elif defect=='evidence':
        w['issue']['explanation']='altered';r['previous_disposition_sha256']=fingerprint(w)
    with pytest.raises(ValueError):resolve(*args)
