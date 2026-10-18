"""Account for EVERY remaining saved warning against the current candidate.

This is a scan, not a semantic adjudication. Automatic changes are explicitly
kept pending. Each original warning, current token evidence and routing reason
is retained, so a completed scan cannot be mistaken for a completed review.
No API, source edits, template approval, publication, or changes to the journal.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from semantic_corpus.template_bank import read_rows, fingerprint

ROUTES={
    'source':'source_wording_needs_explicit_revision',
    'sentence_boundary':'neighbouring_sentence_boundaries_need_joint_review',
    'ambiguity':'multiple_readings_need_contextual_choice',
    'negation_or_modality':'scope_or_missing_predicate_needs_context',
    'property_attachment':'property_referent_needs_context',
    'direct_address':'address_or_argument_needs_context',
    'clause_attachment':'clause_role_and_scope_need_context',
    'coordination':'conjunct_scope_and_heads_need_context',
    'particle_or_preposition':'lexical_construction_and_attachment_need_context',
    'lemma_or_inflection':'lexical_reading_and_form_need_verification',
    'other_syntax':'explicit_syntax_adjudication_required',
}


def classify(warning, evidence, current):
    """Never turn a nearby token edit or an API clear into semantic approval."""
    old=evidence['case']['tokens'];new=current['tokens']
    aligned=(evidence['case']['text']==current['text'] and len(old)==len(new)
             and all(a[1]==b['text'] for a,b in zip(old,new)))
    ids=warning['issue']['tokens']
    revision=current.get('automatic_revision',{})
    edits=revision.get('edits',[]) if revision.get('version')=='offline_local_syntax_v2' else []
    touched=sorted(set(ids)&{e['token'] for e in edits}) if aligned else []
    related=(warning['group']=='direct_address' and touched
             and any(new[i]['dep']=='vocative' for i in touched))
    route=('api_token_addresses_stale' if not aligned else
           'local_address_rule_applied_review_still_required' if related else ROUTES[warning['group']])
    return dict(route=route,status='pending_context_or_source_review',
        source_aligned=aligned,automatic_rule_touched_issue_tokens=touched,
        automatic_revision_version=revision.get('version') if edits else None,
        semantic_approval=False,publication=False)


def run(candidate,ledger,out):
    if out.exists():raise ValueError('Use a fresh catalog directory')
    report=json.loads((candidate/'report.json').read_text(encoding='utf-8'))
    if report['validation']['status']!='passed' or report['conservation']['status']!='passed':
        raise ValueError('Candidate must pass validation and conservation first')
    warnings=list(read_rows(ledger/'pending-warnings.jsonl.gz'))
    by_id=defaultdict(list)
    for row in warnings:by_id[row['id']].append(row)
    expected={(r['id'],r['issue_index']) for r in warnings}
    if len(expected)!=len(warnings):raise ValueError('Duplicate pending warning')
    evidence={};contexts={}
    for row in read_rows(ledger/'received-evidence.jsonl.gz'):
        if row['kind']=='context':contexts[row['id']]=row
        elif row['id'] in by_id:evidence[row['id']]=row
    if evidence.keys()!=by_id.keys():raise ValueError('Missing frozen API evidence')
    current_contexts={r['id']:r for r in read_rows(candidate/'contexts.jsonl.gz')}
    retired={}
    database=candidate/'revision.sqlite3'
    if database.exists():
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='retired_occurrences'").fetchone():
                retired={i:(replacement,json.loads(snapshot)) for i,replacement,snapshot in
                         db.execute('SELECT id,replacement,snapshot FROM retired_occurrences')}
    targets=defaultdict(list)
    for identity,items in by_id.items():
        targets[retired[identity][0] if identity in retired else identity].extend(items)
    out.mkdir(parents=True)
    counts=Counter();seen=set();seen_cases=set();used_contexts=set()
    with gzip.open(out/'warnings.jsonl.gz','wt',encoding='utf-8',newline='\n') as stream:
        for path in sorted((candidate/'bank/occurrences').glob('*.jsonl.gz')):
            for row in read_rows(path):
                identity=row['idx']
                if identity not in targets:continue
                context=current_contexts[row['context_id']]
                if context['turns'][row['origin']['turn']-1][slice(*row['origin']['span'])]!=row['text']:
                    raise ValueError('Current source address is invalid')
                used_contexts.add(row['context_id'])
                for warning in targets[identity]:
                    original_id=warning['id']
                    address=(original_id,warning['issue_index'])
                    if address in seen:raise ValueError('Duplicate current warning address')
                    seen.add(address);seen_cases.add(original_id)
                    if original_id in retired:
                        snapshot=retired[original_id][1]
                        revision=row.get('source_revision',{})
                        found=False
                        while revision:
                            before=revision['preimage']['payload']['ledger']
                            if snapshot in before.get('input_segments',[]):found=True;break
                            revision=before.get('source_revision',{})
                        if not found:raise ValueError('Retired warning lost its original input evidence')
                    if warning['issue_sha256']!=fingerprint(warning['issue']):
                        raise ValueError('Saved warning changed')
                    api=evidence[original_id]
                    if api['issues'][warning['issue_index']]!=warning['issue']:
                        raise ValueError('Warning does not match original API evidence')
                    classification=classify(warning,api,row)
                    counts[classification['route']]+=1
                    result=dict(warning,**classification,current_occurrence_id=identity,
                        retired_input=original_id in retired,current_text=row['text'],
                        current_origin=row['origin'],current_context_id=row['context_id'],
                        current_context_sha256=fingerprint(context['turns']),
                        current_tokens=[{k:t[k] for k in ('i','text','parser_lemma','pos','tag','dep','head','span')}
                                        for t in row['tokens']],
                        current_parse_sha256=fingerprint(row['tokens']),
                        frozen_api_evidence_sha256=fingerprint(api))
                    stream.write(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    if seen!=expected:raise ValueError('Catalog did not cover every pending warning')
    with gzip.open(out/'contexts.jsonl.gz','wt',encoding='utf-8',newline='\n') as stream:
        for identity in sorted(used_contexts):
            stream.write(json.dumps(current_contexts[identity],ensure_ascii=False,separators=(',',':'))+'\n')
    summary=dict(version='offline-warning-catalog-v1',scan_complete=True,
        semantic_review_complete=False,candidate=str(candidate.resolve()),
        input_ledger=str(ledger.resolve()),pending_warnings_scanned=len(seen),
        cases_scanned=len(seen_cases),routes=dict(counts),
        automatic_matches_remain_pending=True,api_calls=0,published=False,
        files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
               for p in out.glob('*.gz')})
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(run(args.candidate,args.ledger,args.out),indent=2))
