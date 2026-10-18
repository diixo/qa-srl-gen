from copy import deepcopy
import gzip
import json
import sqlite3

import pytest

from scripts.catalog_dailydialog_offline_warnings import classify, run
from semantic_corpus.template_bank import fingerprint


def fixture():
    token=dict(i=0,text='Sir',parser_lemma='sir',pos='NOUN',tag='NN',dep='vocative',head=0,span=[0,3])
    issue=dict(category='structure',tokens=[0],explanation='Address',correction='Use vocative.')
    warning=dict(id='test:1:1:1',issue_index=0,issue=issue,issue_sha256=fingerprint(issue),group='direct_address')
    evidence=dict(kind='result',id=warning['id'],issues=[issue],case=dict(text='Sir',tokens=[[0,'Sir']]))
    current=dict(idx=warning['id'],text='Sir',tokens=[token],context_id='test:1',
        origin=dict(turn=1,span=[0,3]),automatic_revision=dict(version='offline_local_syntax_v2',edits=[{'token':0}]))
    return warning,evidence,current


def test_automatic_rule_match_does_not_close_or_approve_warning():
    w,e,c=fixture();before=deepcopy(c)
    r=classify(w,e,c)
    assert r['route']=='local_address_rule_applied_review_still_required'
    assert r['status']=='pending_context_or_source_review'
    assert r['semantic_approval'] is False and r['publication'] is False
    assert c==before


def test_nearby_edit_cannot_resolve_other_warning():
    w,e,c=fixture();w['group']='source'
    assert classify(w,e,c)['route']=='source_wording_needs_explicit_revision'


@pytest.mark.parametrize('field',['text','token'])
def test_stale_api_addresses_are_not_reused(field):
    w,e,c=fixture()
    if field=='text':e['case']['text']='Madam'
    else:e['case']['tokens'][0][1]='Madam'
    r=classify(w,e,c)
    assert r['route']=='api_token_addresses_stale'
    assert r['automatic_rule_touched_issue_tokens']==[]


def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row)+'\n')


@pytest.mark.parametrize('missing',[False,True])
def test_scan_requires_every_warning_and_does_not_claim_review_completion(tmp_path,missing):
    w,e,c=fixture();candidate=tmp_path/'candidate';ledger=tmp_path/'ledger';out=tmp_path/'out'
    candidate.mkdir()
    (candidate/'report.json').write_text(json.dumps(dict(validation={'status':'passed'},conservation={'status':'passed'})))
    context=dict(id='test:1',turns=['Sir'])
    write_rows(candidate/'contexts.jsonl.gz',[context])
    write_rows(candidate/'bank/occurrences/00000.jsonl.gz',[] if missing else [c])
    write_rows(ledger/'pending-warnings.jsonl.gz',[w])
    write_rows(ledger/'received-evidence.jsonl.gz',[dict(context,kind='context'),e])
    if missing:
        with pytest.raises(ValueError,match='every pending warning'):run(candidate,ledger,out)
        assert not (out/'summary.json').exists()
    else:
        report=run(candidate,ledger,out)
        assert report['scan_complete'] is True and report['semantic_review_complete'] is False
        assert report['pending_warnings_scanned']==1 and report['api_calls']==0
        assert report['automatic_matches_remain_pending'] is True and report['published'] is False


@pytest.mark.parametrize('tamper',[False,True])
def test_joined_input_warning_keeps_its_address_and_requires_retained_evidence(tmp_path,tamper):
    w,e,c=fixture();candidate=tmp_path/'candidate';ledger=tmp_path/'ledger';out=tmp_path/'out'
    candidate.mkdir()
    (candidate/'report.json').write_text(json.dumps(dict(validation={'status':'passed'},conservation={'status':'passed'})))
    old=w['id'];c['idx']='test:1:1:2'
    snapshot=dict(body={'idx':old},payload={'original':True})
    c['source_revision']=dict(preimage=dict(payload=dict(ledger=dict(input_segments=[] if tamper else [snapshot]))))
    with sqlite3.connect(candidate/'revision.sqlite3') as db:
        db.execute('CREATE TABLE retired_occurrences(id TEXT,replacement TEXT,snapshot TEXT)')
        db.execute('INSERT INTO retired_occurrences VALUES(?,?,?)',(old,c['idx'],json.dumps(snapshot)))
    context=dict(id='test:1',turns=['Sir'])
    write_rows(candidate/'contexts.jsonl.gz',[context])
    write_rows(candidate/'bank/occurrences/00000.jsonl.gz',[c])
    write_rows(ledger/'pending-warnings.jsonl.gz',[w])
    write_rows(ledger/'received-evidence.jsonl.gz',[dict(context,kind='context'),e])
    if tamper:
        with pytest.raises(ValueError,match='original input evidence'):run(candidate,ledger,out)
    else:
        summary=run(candidate,ledger,out)
        assert summary['pending_warnings_scanned']==1
        with gzip.open(out/'warnings.jsonl.gz','rt',encoding='utf-8') as f:row=json.loads(next(f))
        assert row['id']==old and row['current_occurrence_id']==c['idx'] and row['retired_input'] is True
