import gzip
import json

from scripts.finalize_dailydialog_offline_review import disposition, run
from semantic_corpus.template_bank import fingerprint, read_rows


def warning(status='fixed_in_candidate'):
    return dict(id='train:1:1:1',issue_index=0,issue_sha256='warning',
                group='source',status=status,reason='Meaning cannot be uniquely recovered.')


def test_reparse_revokes_current_fix_claim_but_keeps_original_evidence():
    w=warning()
    r=disposition(w,{}, {},set(),[{'id':w['id'],'token_analysis':'reparsed_unreviewed'}])
    assert r['status']=='quarantined_after_source_revision'
    assert r['previous_warning']==w
    assert r['previous_warning_sha256']==fingerprint(w)
    assert r['semantic_approval'] is False and r['publication'] is False


def test_wording_fix_is_not_a_semantic_or_syntax_approval():
    w=warning('pending_context_or_source_review')
    key=(w['id'],w['issue_index'],w['issue_sha256'])
    local={'decision':'explicit_wording_correction','note':'Repair a typo.'}
    r=disposition(w,{key:local},{},{key},[{'token_analysis':'reparsed_unreviewed'}])
    assert r['status']=='fixed_wording_only'
    assert not r['semantic_approval']


def test_unresolved_reason_is_retained_in_quarantine():
    w=warning('pending_context_or_source_review')
    r=disposition(w,{}, {},set(),[])
    assert r['status']=='quarantined_unresolved'
    assert r['reason']==w['reason']


def test_unchanged_analysis_keeps_its_previous_local_resolution():
    w=warning()
    r=disposition(w,{}, {},set(),[{'token_analysis':'preserved'}])
    assert r['status']=='fixed_in_candidate'
    assert not r['semantic_approval']


def test_retired_warning_uses_current_sentence_address_and_preserves_paid_evidence(tmp_path):
    candidate=tmp_path/'candidate';candidate.mkdir()
    ledger=tmp_path/'ledger';ledger.mkdir()
    (candidate/'report.json').write_text(json.dumps({'validation':{'status':'passed'},'conservation':{'status':'passed'}}))
    (candidate/'applied-turn-plans.jsonl').write_text('')
    (candidate/'lineage.json').write_text(json.dumps({'train:1:1:2':{
        'input_ids':['train:1:1:2'],'token_analysis':'reparsed_unreviewed','text':'Miss Parker.'}}))
    w=warning();w['id']='train:1:1:3'
    w['local_review']={'occurrence_id':'train:1:1:2'}
    w['issue']={'category':'structure','tokens':[],'explanation':'Honorific was split.'}
    w['issue_sha256']=fingerprint(w['issue'])
    with gzip.open(ledger/'warning-decisions.jsonl.gz','wt',encoding='utf-8') as stream:
        stream.write(json.dumps(w)+'\n')
    (ledger/'received-evidence.jsonl.gz').write_bytes(b'unchanged paid evidence')
    words=tmp_path/'words.jsonl';words.write_text('')
    bounds=tmp_path/'bounds.jsonl';bounds.write_text('')
    out=tmp_path/'final'
    result=run(candidate,ledger,words,bounds,out)
    assert result['quarantined_warnings']==1
    saved=list(read_rows(out/'warning-decisions.jsonl.gz'))[0]
    assert saved['status']=='quarantined_after_source_revision'
    assert saved['previous_warning']==w
    assert (out/'received-evidence.jsonl.gz').read_bytes()==b'unchanged paid evidence'
