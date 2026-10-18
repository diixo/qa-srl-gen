"""Bind offline warning decisions and the remaining quarantine to a candidate.

No inference or publication. A repaired parse without a new explicit warning
decision invalidates the previous local closure. Quarantine carries current
text, tokens, full dialogue context, reasons and original paid evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_dailydialog_hybrid_bank import open_overlay
from scripts.benchmark_dailydialog_hybrid import dump, write
from scripts.import_dailydialog_sentences import digest
from semantic_corpus.template_bank import fingerprint, read_rows

STATUSES = {'fixed_wording_only','fixed_boundary_only','fixed_in_candidate',
            'closed_verified_after_source_revision','closed_representation_only',
            'quarantined_unresolved'}


def key(row):
    return row['id'],row['issue_index'],row['issue_sha256']


def issue(row):
    while 'issue' not in row:
        row = row.get('previous_warning') or row['previous_disposition']
    return row['issue']


def current_ids(warning):
    if warning.get('quarantine_recheck'):
        return [r['id'] for r in warning['quarantine_recheck']['outputs']]
    if warning.get('local_recheck'):
        return [warning['local_recheck']['occurrence_id']]
    if warning.get('output_occurrences'):
        return list(dict.fromkeys(r['id'] for r in warning['output_occurrences']))
    old = warning.get('previous_warning',warning)
    return [old.get('local_review',{}).get('occurrence_id',warning['id'])]


def resolve(warning, review, images, contexts):
    if (key(warning) != key(review) or review['previous_disposition_sha256'] != fingerprint(warning)
            or fingerprint(issue(warning)) != warning['issue_sha256']
            or review['method'] != 'direct_assistant_quarantine_warning_review_v1'
            or review['status'] not in STATUSES or not review['note'].strip()
            or review['semantic_approval'] is not False or review['publication'] is not False):
        raise ValueError('Stale or overbroad warning decision')
    seen=set()
    for proof in review['outputs']:
        identity=proof['id']
        if identity in seen:raise ValueError('Duplicate output proof')
        seen.add(identity);image=images[identity];b,p=image['body'],image['payload']
        context=contexts[b['provenance']['context_id']];origin=b['provenance']['origin']
        if (fingerprint(image)!=proof['image_sha256'] or fingerprint(context)!=proof['context_sha256']
                or context[origin['turn']-1][slice(*origin['span'])]!=b['example']):
            raise ValueError('Stale current image or context')
        inspected=set()
        for check in proof['token_checks']:
            if (type(check['i']) is not int or not 0 <= check['i'] < len(p['ledger']['tokens'])
                    or check['i'] in inspected or not check['fields']):
                raise ValueError('Invalid or duplicate token proof')
            inspected.add(check['i'])
            token=p['ledger']['tokens'][check['i']]
            if token['text']!=check['text'] or any(token[k]!=v for k,v in check['fields'].items()):
                raise ValueError('Token proof does not match current parse')
        for text in proof['contains']:
            if text not in b['example']:raise ValueError('Required source evidence absent')
        for text in proof['absent']:
            if text in b['example']:raise ValueError('Rejected source wording still present')
        if 'attribute_targets' in proof and proof['attribute_targets'] != {
                a['slot']:a.get('target_slot') for a in p['record']['attributes']}:
            raise ValueError('Property target proof does not match current record')
        if 'constructions' in proof and proof['constructions'] != {
                g['head_slot']:[c['lemma'] for c in g['constructions']]
                for g in p['record']['predicate_groups']}:
            raise ValueError('Construction proof does not match current record')
        if ('temporal_adjunct_candidates' in proof
                and proof['temporal_adjunct_candidates'] != p['ledger'].get('temporal_adjunct_candidates', [])):
            raise ValueError('Temporal evidence does not match current ledger')
        if not proof['token_checks'] and not proof['contains']:
            raise ValueError('Empty source/token proof')
    if not seen:raise ValueError('No current output evidence')
    if review['status']=='quarantined_unresolved':
        if not review.get('unknown'):raise ValueError('Pending decision must retain unknown information')
        for identity in seen:
            p=images[identity]['payload']
            if p['record'].get('usage_constraints',{}).get('mode') not in {'context_only','pending'}:
                raise ValueError('Partial interpretation is not restricted')
    result=deepcopy(warning)
    result.update(status=review['status'],reason=review['note'],
        previous_disposition=deepcopy(warning),previous_disposition_sha256=fingerprint(warning),
        quarantine_recheck=deepcopy(review),semantic_approval=False,publication=False)
    return result


def revision_inputs(candidate, source_revision, previous):
    """Verify either a source+syntax revision or an unchanged-source local step."""
    candidate_plan=json.loads((candidate/'plan.json').read_text(encoding='utf-8'))
    if source_revision is None and candidate_plan.get('version')=='hybrid_quality_patterns_v1':
        from itertools import zip_longest
        parent=Path(previous['candidate']).resolve()
        if Path(candidate_plan['source']).resolve()!=parent:
            raise ValueError('Ledger and quality revision disagree')
        required={str(parent/name) for name in ('plan.json','report.json','revision.sqlite3','contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl')}
        if not required<=candidate_plan['frozen'].keys():
            raise ValueError('Quality revision has incomplete parent fingerprints')
        for path,sha in candidate_plan['frozen'].items():
            if digest(Path(path))!=sha:
                raise ValueError('Quality revision parent changed: '+path)
        conservation=json.loads((candidate/'report.json').read_text(encoding='utf8'))['conservation']
        if (conservation.get('status')!='passed' or not conservation.get('previous_preimages_retained')
                or not conservation.get('source_and_tokens_unchanged') or not conservation.get('context_reviews_unchanged')):
            raise ValueError('Quality revision lacks conservation')
        def rows(root):
            for path in sorted((root/'bank/occurrences').glob('*.jsonl.gz')):
                yield from read_rows(path)
        changed=set();count=0
        for before,after in zip_longest(rows(parent),rows(candidate)):
            if before is None or after is None or before['idx']!=after['idx']:
                raise ValueError('Quality revision lost an occurrence')
            count+=1
            if before!=after:changed.add(before['idx'])
        if (count!=conservation['counts']['segments']
                or len(changed)!=conservation['counts'].get('revised_occurrences',0)):
            raise ValueError('Quality revision changes disagree with conservation')
        for name in ('contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'):
            if digest(parent/name)!=digest(candidate/name):
                raise ValueError('Quality revision changed source or reviews')
        return defaultdict(list),changed,changed
    if source_revision is None and candidate_plan.get('version')=='explicit_candidate_wording_v1':
        parent=Path(previous['candidate']).resolve()
        if Path(candidate_plan['source']).resolve()!=parent:
            raise ValueError('Ledger and wording revision disagree')
        for name in ('plan.json','report.json','revision.sqlite3','contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'):
            path=parent/name
            if digest(path)!=candidate_plan['protected'].get(str(path)):
                raise ValueError('Wording parent changed: '+name)
        for name,value in candidate_plan['review_sha256'].items():
            if digest(Path(name))!=value:
                raise ValueError('Wording decisions changed')
        report=json.loads((candidate/'report.json').read_text(encoding='utf-8'))
        conservation=report['conservation']
        if (conservation.get('status')!='passed' or not conservation.get('every_original_accounted_for')
                or not conservation.get('all_unaddressed_analyses_unchanged')
                or not conservation.get('source_preimages_and_previous_local_decisions_preserved')):
            raise ValueError('Wording revision lacks complete conservation')
        by_input=defaultdict(list);changed=set()
        for identity,entry in json.loads((candidate/'lineage.json').read_text(encoding='utf-8')).items():
            for old in entry['input_ids']:
                if old.rsplit(':',1)[0]!=identity.rsplit(':',1)[0]:
                    raise ValueError('Wording lineage crosses a source turn')
                by_input[old].append(identity)
                if entry['token_analysis']=='reparsed_unreviewed' or entry.get('context_review_invalidated'):
                    changed.add(old)
        return by_input,changed,set()
    applied=Path(candidate_plan['reviews'])
    if digest(applied) != candidate_plan['review_sha256']:
        raise ValueError('Applied quarantine reviews changed')
    patched_rows=list(read_rows(applied));patched={r['id'] for r in patched_rows}
    if not patched or len(patched)!=len(patched_rows):
        raise ValueError('Empty or duplicate applied reviews')
    by_input=defaultdict(list);changed=set()
    if source_revision is None and candidate_plan.get('version')=='reviewed_source_segmentation_v6':
        parent=Path(previous['candidate']).resolve()
        if Path(candidate_plan['source']).resolve()!=parent:
            raise ValueError('Ledger and explicit source revision disagree')
        for name in ('plan.json','report.json','revision.sqlite3','contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'):
            path=parent/name
            if digest(path)!=candidate_plan['frozen'].get(str(path)):
                raise ValueError('Frozen source parent changed: '+name)
        inputs=[];outputs=[]
        for row in patched_rows:
            before=row.get('input_ids',[row['id']]);after=[o['id'] for o in row['outputs']]
            if (row.get('method')!='direct_assistant_source_review' or not before or before[0]!=row['id']
                    or not after or any(i.rsplit(':',1)[0]!=row['id'].rsplit(':',1)[0] for i in before+after)):
                raise ValueError('Invalid explicit source lineage')
            for identity in before:by_input[identity]=after
            inputs.extend(before);outputs.extend(after)
        if len(set(inputs))!=len(inputs) or len(set(outputs))!=len(outputs):
            raise ValueError('Overlapping explicit source lineage')
        conservation=json.loads((candidate/'report.json').read_text(encoding='utf-8'))['conservation']
        counts=conservation['counts']
        if (conservation.get('status')!='passed' or conservation.get('previous_preimages_retained') is not True
                or counts.get('revised_outputs')!=len(outputs)
                or counts.get('unchanged_analyses',0)!=counts['old_segments']-len(inputs)
                or counts['new_segments']!=counts['old_segments']-len(inputs)+len(outputs)):
            raise ValueError('Explicit source revision lacks complete conservation')
        # A retired quote-only address maps to the joined sentence. Every old
        # closure on revised inputs requires a new review, even after a join.
        return by_input,set(inputs),set(outputs)
    if source_revision is None:
        parent=Path(previous['candidate']).resolve()
        if (candidate_plan.get('version')!='explicit_quarantine_review_v1'
                or Path(candidate_plan['source']).resolve()!=parent):
            raise ValueError('Ledger and local candidate revision disagree')
        report=json.loads((candidate/'report.json').read_text(encoding='utf-8'))
        conservation=report['conservation']
        if (conservation.get('source_unchanged') is not True
                or conservation.get('all_unaddressed_occurrences_identical') is not True
                or conservation['counts']['revised_occurrences']!=len(patched)):
            raise ValueError('Local revision lacks occurrence conservation')
        for name in ('plan.json','report.json','revision.sqlite3','contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'):
            path=parent/name
            if digest(path)!=candidate_plan['protected'].get(str(path)):
                raise ValueError('Protected parent candidate changed: '+name)
        for name in ('contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'):
            if digest(candidate/name)!=digest(parent/name):
                raise ValueError('Local revision changed source or reference reviews')
        return by_input,changed,patched
    source_plan=json.loads((source_revision/'plan.json').read_text(encoding='utf-8'))
    if (Path(candidate_plan['source']).resolve()!=source_revision
            or Path(source_plan['source']).resolve()!=Path(previous['candidate']).resolve()):
        raise ValueError('Ledger and candidate revision chain disagree')
    for name,value in source_plan['review_sha256'].items():
        if digest(Path(name)) != value:raise ValueError('Applied source reviews changed')
    for name in ('lineage.json','contexts.jsonl.gz','coverage.jsonl.gz'):
        path=source_revision/name
        if digest(path) != candidate_plan['protected'][str(path)]:
            raise ValueError('Source revision lineage or context changed')
    for identity,value in json.loads((source_revision/'lineage.json').read_text(encoding='utf-8')).items():
        for parent in value['input_ids']:
            by_input[parent].append(identity)
            if value['token_analysis']=='reparsed_unreviewed' or value.get('context_review_invalidated'):
                changed.add(parent)
    return by_input,changed,patched


def run(candidate, source_revision, ledger, reviews, out):
    candidate,ledger,reviews,out=(Path(p).resolve() for p in (candidate,ledger,reviews,out))
    source_revision=Path(source_revision).resolve() if source_revision is not None else None
    if out.exists():raise ValueError('Use a fresh quarantine ledger')
    report=json.loads((candidate/'report.json').read_text(encoding='utf-8'))
    if report['validation']['status']!='passed' or report['conservation']['status']!='passed':
        raise ValueError('Candidate must pass validation and conservation')
    previous=json.loads((ledger/'summary.json').read_text(encoding='utf-8'))
    by_input,changed,patched=revision_inputs(candidate,source_revision,previous)
    warnings=list(read_rows(ledger/'warning-decisions.jsonl.gz'));decisions=list(read_rows(reviews))
    known={key(w):w for w in warnings};local={key(r):r for r in decisions}
    if len(known)!=len(warnings) or len(local)!=len(decisions) or not local.keys()<=known.keys():
        raise ValueError('Duplicate or unknown warning')
    contexts={r['id']:r['turns'] for r in read_rows(candidate/'contexts.jsonl.gz')}
    targets=set()
    mappings={}
    for w in warnings:
        ids=current_ids(w)
        mapped=list(dict.fromkeys(o for i in ids for o in by_input.get(i,[i])))
        if key(w) in local:
            proposed=[r['id'] for r in local[key(w)]['outputs']]
            # Rechecks can select a more precise descendant of the SAME turn,
            # e.g. a schedule separated from the earlier 'Let me see' prefix.
            if any(i.rsplit(':',1)[0] not in {j.rsplit(':',1)[0] for j in ids} for i in proposed):
                raise ValueError('Warning decision points outside its source turn')
            mapped=proposed
        mappings[key(w)]=mapped
        if w['status'].startswith('quarantined_') or key(w) in local or set(ids)&changed or set(mapped)&patched:
            targets.update(mapped)
    images={}
    with tempfile.TemporaryDirectory(prefix='quarantine-final-',dir=candidate.parent) as directory:
        if Path(directory).resolve().parent != candidate.parent:
            raise ValueError('Temporary overlay must remain beside the candidate')
        db,*_=open_overlay(candidate,Path(directory))
        try:
            for identity in sorted(targets):
                row=db.execute('SELECT o.body,p.payload FROM occurrences o JOIN parses p ON o.key=p.key WHERE o.id=?',(identity,)).fetchone()
                if not row:raise ValueError('Missing current quarantine output: '+identity)
                b,p=map(json.loads,row);images[identity]=dict(body=b,payload=p)
        finally:db.close()
    results=[];invalidated=[]
    for w in warnings:
        k=key(w)
        if k in local:
            new=resolve(w,local[k],images,contexts)
        elif set(current_ids(w))&changed or set(mappings[k])&patched:
            new=deepcopy(w)
            new.update(status='quarantined_after_candidate_revision',
                reason='Earlier local resolution needs rechecking against the revised parse; no approval transferred.',
                previous_disposition=deepcopy(w),previous_disposition_sha256=fingerprint(w),
                semantic_approval=False,publication=False)
            invalidated.append(k)
        else:new=w
        results.append(new)
    quarantine=[w for w in results if w['status'].startswith('quarantined_')]
    out.mkdir(parents=True)
    for name,rows in [('warning-decisions.jsonl.gz',results),('quarantined-warnings.jsonl.gz',quarantine)]:
        with gzip.open(out/name,'wt',encoding='utf-8',newline='\n') as stream:
            for row in rows:write(stream,row)
    case_warnings=defaultdict(list)
    for w in quarantine:
        for identity in mappings[key(w)]:
            case_warnings[identity].append(dict(id=w['id'],issue_index=w['issue_index'],issue_sha256=w['issue_sha256'],group=w['group'],status=w['status'],reason=w['reason']))
    with gzip.open(out/'quarantine-cases.jsonl.gz','wt',encoding='utf-8',newline='\n') as stream:
        for identity,ws in sorted(case_warnings.items()):
            image=images[identity];b,p=image['body'],image['payload'];ctx=contexts[b['provenance']['context_id']]
            origin=b['provenance']['origin']
            if ctx[origin['turn']-1][slice(*origin['span'])]!=b['example']:raise ValueError('Quarantine address mismatch')
            write(stream,dict(id=identity,image=image,image_sha256=fingerprint(image),context=ctx,context_sha256=fingerprint(ctx),
                warnings=ws,status='OPEN',next_action='separate_contextual_review',publication=False,
                extraction_scope=p['record'].get('usage_constraints',{}),missing_information_is_not_inferred=True))
    shutil.copyfile(ledger/'received-evidence.jsonl.gz',out/'received-evidence.jsonl.gz')
    shutil.copyfile(reviews,out/reviews.name)
    saved=list(read_rows(out/'warning-decisions.jsonl.gz'))
    if len(saved)!=len(warnings):raise ValueError('Warning lost')
    for before,after in zip(warnings,saved):
        if key(before)!=key(after) or fingerprint(issue(after))!=after['issue_sha256']:raise ValueError('Evidence changed')
        if key(before) not in local and key(before) not in invalidated:
            if before!=after:raise ValueError('Untouched disposition changed')
        elif after['previous_disposition']!=before:raise ValueError('Prior disposition lost')
    if digest(out/'received-evidence.jsonl.gz')!=digest(ledger/'received-evidence.jsonl.gz'):raise ValueError('Paid evidence changed')
    summary=dict(version='quarantine_current_candidate_v1',candidate=str(candidate),previous_ledger=str(ledger),
        saved_warnings=len(warnings),explicitly_rechecked=len(decisions),
        recheck_findings=dict(Counter(r['status'] for r in decisions)),
        invalidated_without_new_review=len(invalidated),warning_counts=dict(Counter(w['status'] for w in results)),
        quarantined_warnings=len(quarantine),quarantined_original_cases=len({w['id'] for w in quarantine}),
        quarantined_current_occurrences=len(case_warnings),quarantine_groups=dict(Counter(w['group'] for w in quarantine)),
        received_evidence_sha256=digest(out/'received-evidence.jsonl.gz'),review_sha256=digest(reviews),
        current_source_and_context_embedded=True,unknowns_retained=True,
        exclusions=0,semantic_review_complete=False,full_candidate_published=False,api_calls=0)
    dump(out/'summary.json',summary)
    dump(out/'verification.json',dict(status='passed',warnings_preserved=len(warnings),current_quarantine_cases=len(case_warnings),
        paid_evidence_identical=True,prior_dispositions_preserved=True,api_calls=0,publication=False))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','ledger','reviews','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-revision',type=Path,help='Omit for a local revision that preserves source text and IDs.')
    print(json.dumps(run(**vars(parser.parse_args())),indent=2))
