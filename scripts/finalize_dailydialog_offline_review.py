"""Close the offline review pass without disguising unresolved evidence.

Every paid warning keeps its original payload and previous resolution. Explicit
wording/boundary repairs are counted separately from syntax repairs. Unresolved
and newly reparsed analyses go to a durable quarantine, never into admission.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_dailydialog_hybrid import dump, write
from scripts.import_dailydialog_sentences import digest
from semantic_corpus.template_bank import fingerprint, read_rows


def disposition(warning, wording, boundary, applied, outputs):
    key = (warning['id'], warning['issue_index'], warning['issue_sha256'])
    prior = warning['status']
    result = dict(id=warning['id'],issue_index=warning['issue_index'],
        issue_sha256=warning['issue_sha256'],group=warning['group'],
        previous_warning=warning,previous_warning_sha256=fingerprint(warning),
        output_occurrences=outputs,semantic_approval=False,publication=False)
    local = wording.get(key) or boundary.get(key)
    if local:
        result['explicit_decision'] = local
    if key in applied:
        result.update(status=('fixed_wording_only' if key in wording else 'fixed_boundary_only'),
            reason='Explicit local decision applied. The resulting syntax/semantics is not approved.')
    elif local and local['decision']=='not_supported':
        result.update(status='closed_source_warning_not_supported',reason=local['note'])
    elif any(r['token_analysis']=='reparsed_unreviewed' for r in outputs):
        result.update(status='quarantined_after_source_revision',
            reason='Text or boundaries changed; earlier local syntax decisions remain historical evidence and are not approval of the new parse.')
    elif prior.startswith('pending_'):
        note = (local or {}).get('note') or warning.get('local_review',{}).get('note') or warning.get('reason')
        if not note:
            raise ValueError('Unreviewed warning cannot be closed without an explicit reason')
        result.update(status='quarantined_unresolved',reason=note)
    else:
        result.update(status=prior,reason=warning.get('reason','Previous evidence-bound resolution retained.'))
    return result


def run(candidate, ledger, wording, boundaries, out):
    if out.exists():
        raise ValueError('use a fresh final review directory')
    report=json.loads((candidate/'report.json').read_text(encoding='utf-8'))
    if report['validation']['status']!='passed' or report['conservation']['status']!='passed':
        raise ValueError('candidate must pass both validation and conservation')
    warnings=list(read_rows(ledger/'warning-decisions.jsonl.gz'))
    keys={(r['id'],r['issue_index'],r['issue_sha256']) for r in warnings}
    if len(keys)!=len(warnings):
        raise ValueError('duplicate warning identity')
    decisions=[]
    for path in (wording,boundaries):
        mapping={}
        for r in read_rows(path):
            k=(r['id'],r['issue_index'],r['issue_sha256'])
            if k not in keys or k in mapping or r['semantic_approval'] or r['publication']:
                raise ValueError('invalid local decision')
            mapping[k]=r
        decisions.append(mapping)
    words,bounds=decisions
    applied=set()
    for turn in read_rows(candidate/'applied-turn-plans.jsonl'):
        for r in turn['wording']+turn['boundaries']:
            k=(r['id'],r['issue_index'],r['issue_sha256'])
            if (words.get(k) or bounds.get(k))!=r:
                raise ValueError('applied review differs from retained decision')
            applied.add(k)
    by_input=defaultdict(list)
    for identity,r in json.loads((candidate/'lineage.json').read_text(encoding='utf-8')).items():
        for parent in r['input_ids']:
            by_input[parent].append(dict(id=identity,**r))
    results=[]
    for w in warnings:
        if fingerprint(w['issue'])!=w['issue_sha256']:
            raise ValueError('original API warning changed')
        current_id = w.get('local_review',{}).get('occurrence_id',w['id'])
        results.append(disposition(w,words,bounds,applied,by_input.get(current_id,[])))
    counts=Counter(r['status'] for r in results)
    quarantined=[r for r in results if r['status'].startswith('quarantined_')]
    # No claim is made about cases with no API answer or a response of clear.
    out.mkdir(parents=True)
    for name,rows in [('warning-decisions.jsonl.gz',results),('quarantined-warnings.jsonl.gz',quarantined)]:
        with gzip.open(out/name,'wt',encoding='utf-8',newline='\n') as stream:
            for r in rows:
                write(stream,r)
    shutil.copyfile(ledger/'received-evidence.jsonl.gz',out/'received-evidence.jsonl.gz')
    for path in (wording,boundaries):
        shutil.copyfile(path,out/path.name)
    summary=dict(version='offline_disposition_v1',candidate=str(candidate.resolve()),
        previous_ledger=str(ledger.resolve()),warning_counts=dict(counts),
        saved_warnings=len(warnings),all_saved_warnings_dispositioned=True,
        offline_pass_complete=True,semantic_review_complete=False,
        quarantined_warnings=len(quarantined),quarantined_cases=len({r['id'] for r in quarantined}),
        quarantine_groups=dict(Counter(r['group'] for r in quarantined)),
        source_view_report=report,received_evidence_sha256=digest(out/'received-evidence.jsonl.gz'),
        prior_warning_ledger_sha256=digest(ledger/'warning-decisions.jsonl.gz'),
        api_calls=0,unanswered_api_cases_reviewed=False,clear_is_not_admission=True,
        full_candidate_published=False,
        publication_policy='Only explicit per-occurrence semantic admissions may enter the Built-in pool; all other candidates stay UNREVIEWED.',
        unresolved_policy='Retain the complete case and reason in quarantine; do not guess missing wording or propagate meanings from equal text.')
    dump(out/'summary.json',summary)
    # Re-read durable outputs and prove one-to-one accounting, not just totals.
    saved=list(read_rows(out/'warning-decisions.jsonl.gz'))
    if len(saved)!=len(warnings) or any(r['previous_warning']!=w for r,w in zip(saved,warnings)):
        raise ValueError('saved review lost original evidence')
    if digest(out/'received-evidence.jsonl.gz')!=digest(ledger/'received-evidence.jsonl.gz'):
        raise ValueError('paid response evidence changed')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('candidate','ledger','wording','boundaries','out'):
        p.add_argument('--'+flag,type=Path,required=True)
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
