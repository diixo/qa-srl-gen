"""Offline, occurrence-addressed accounting of saved API warnings and repairs.

Never calls the API, executes model prose, or admits a template. Exports received
evidence with shared contexts, plus one disposition per warning. Unresolved
warnings remain in an explicit queue, including warnings on partly fixed cases.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from semantic_corpus.template_bank import fingerprint, read_rows
from scripts.triage_dailydialog_api_results import issue_group, packed


def representation_only(issue, case):
    """Accept our WRB convention only for a narrowly bounded POS-only proposal.

    A policy decision about representation, not a judgment about the sentence's
    meaning. Other words, changed dependencies and mixed proposals are excluded.
    """
    ids = issue['tokens']
    if issue['category'] != 'structure' or len(ids) != 1:
        return False
    index = ids[0]
    if type(index) is not int or not 0 <= index < len(case['tokens']):
        return False
    token = case['tokens'][index]
    if (token[1].lower() not in {'how', 'when', 'where', 'why'}
            or token[3:5] != ['SCONJ', 'WRB'] or token[5] not in {'advmod', 'ROOT'}):
        return False
    text = issue['correction'].lower().replace('\u2019', "'")
    clauses = [part for part in re.split(r';|,?\s+(?=retain(?:ing)?\b|keep(?:ing)?\b)', text) if part.strip()]
    if not clauses:
        return False
    main = re.findall(r'[a-z]+|\d+', clauses[0])
    allowed = {'change', 'set', 'tag', 'retag', 'use', 'token', 'pos', 'from',
               'sconj', 'to', 'adv', 'as', 'how', 'when', 'where', 'why', 'the',
               'of', 's', 'its', 'wrb', str(index)}
    if (not main or main[0] not in {'change', 'set', 'tag', 'retag', 'use'}
            or 'adv' not in main or not set(main) <= allowed):
        return False
    tail_words = {'retain', 'retaining', 'keep', 'keeping', 'its', 'the', 'current',
                  'existing', 'original', 'tag', 'wrb', 'and', 'advmod', 'root',
                  'dep', 'dependency', 'relation', 'head', 'attachment', 'adverbial',
                  'unchanged'}
    for clause in clauses[1:]:
        words = re.findall(r'[a-z]+|\d+', clause)
        if not words or words[0] not in {'retain', 'retaining', 'keep', 'keeping'} or not set(words) <= tail_words:
            return False
    return True


def disposition(identity, index, issue, case, review, applied, resolved):
    decision = dict(id=identity, issue_index=index, issue_sha256=fingerprint(issue),
                    group=issue_group(issue), issue=issue, publication=False)
    if review is not None:
        if review['api']['issues'][index] != issue:
            raise ValueError('Adjudicated warning no longer matches the journal')
        judgment = review['issue_reviews'][index]
        decision['adjudication'] = judgment
        if judgment['detection'] == 'representation_only':
            if not representation_only(issue, case):
                raise ValueError('Explicit representation-only judgment exceeds the bounded POS convention')
            decision.update(status='closed_representation_only', reason=judgment['note'])
            return decision
        if judgment['detection'] == 'not_demonstrated':
            decision.update(status='closed_not_supported', reason=judgment['note'])
            return decision
        if (judgment['detection'] == 'confirmed' and identity in applied
                and index in resolved.get(identity, ())):
            decision.update(status='fixed_in_candidate', reason='Explicit reviewed edit compiled and candidate validation passed.')
            return decision
        decision.update(status='pending_context_or_source_review', reason=judgment['note'])
        return decision
    if representation_only(issue, case):
        decision.update(status='closed_representation_only',
            reason='POS-only replacement of accepted spaCy SCONJ/WRB convention; keep syntax. No semantic approval.')
    else:
        decision.update(status='pending_context_or_source_review',
            reason='Explicit contextual adjudication is still required.' if identity not in applied
            else 'Other local edits were applied; this warning has not been independently closed.')
    return decision


def verify_saved_repairs(candidate, bindings):
    """Verify earlier fixes carried into this candidate, without trusting IDs alone.

    A later rebuild contains earlier corrections too. Bind each closed warning
    to its exact reviewed edit in the CURRENT database; an overwritten repair,
    stale resolution file or missing occurrence cannot silently count as fixed.
    """
    result = set()
    db = sqlite3.connect((candidate/'revision.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)
    try:
        has_retired = db.execute("SELECT 1 FROM sqlite_master WHERE name='retired_occurrences'").fetchone()
        for identity, expected in bindings.items():
            retired = (db.execute('SELECT replacement,snapshot FROM retired_occurrences WHERE id=?',
                                  (identity,)).fetchone() if has_retired else None)
            target = retired[0] if retired else identity
            row = db.execute('SELECT key,body,payload FROM corrections WHERE id=?', (target,)).fetchone()
            if row is None:
                raise ValueError('Resolved warning has no retained repair: '+identity)
            key, body, payload = row[0], json.loads(row[1]), json.loads(row[2])
            revision = payload['ledger'].get('source_revision', {})
            review = revision.get('review', {})
            if (fingerprint(review) != expected or revision.get('semantic_approval') is not False
                    or fingerprint(revision.get('preimage')) != review.get('preimage')
                    or body.get('idx') != target
                    or body['provenance']['cache_key'] != key
                    or key != fingerprint([revision.get('version'), review, target])):
                raise ValueError('Resolved warning refers to a stale or overwritten repair: '+identity)
            if retired:
                inputs = revision['preimage']['payload']['ledger'].get('input_segments', [])
                original = next((item for item in inputs if item['body']['idx'] == identity), None)
                if (target == identity or identity not in review.get('input_ids', [])
                        or original is None or json.loads(retired[1]) != original):
                    raise ValueError('Joined repair lost the retired input evidence: '+identity)
            output = next((o for o in review['outputs'] if o['id'] == target), None)
            if (output is None or body['example'] != review['after_turn'][slice(*output['span'])]
                    or payload['ledger']['text'] != body['example']):
                raise ValueError('Retained repair has different output: '+identity)
            result.add(identity)
    finally:
        db.close()
    return result


def run(source, candidate, sample_path, resolution_path, out):
    if out.exists():
        raise ValueError('Use a fresh export directory')
    report = json.loads((candidate/'report.json').read_text(encoding='utf-8'))
    if (report.get('status') != 'source_segmentation_complete'
            or report['validation']['status'] != 'passed'
            or report['conservation']['status'] != 'passed' or report.get('published') is not False):
        raise ValueError('Candidate must have passed conservation and remain unpublished')
    plan = json.loads((candidate/'plan.json').read_text(encoding='utf-8'))
    archived = candidate/'bank/review/source-segmentation-decisions.jsonl'
    if archived.read_bytes() != Path(plan['reviews']).read_bytes():
        raise ValueError('Reviewed edits differ from the candidate archive')
    changes = list(read_rows(archived))
    applied = {i for r in changes for i in r.get('input_ids', [r['id']])}
    sample = {r['id']:r for r in read_rows(sample_path)}
    resolution = json.loads(resolution_path.read_text(encoding='utf-8'))
    if resolution['review_sha256'] != hashlib.sha256(sample_path.read_bytes()).hexdigest():
        raise ValueError('Explicit resolutions refer to a different sample')
    resolved = resolution['resolved_issue_indices']
    bindings = resolution.get('applied_review_sha256')
    if bindings is not None:
        if set(bindings) != set(resolved):
            raise ValueError('Every resolved case requires an exact retained-review binding')
        applied |= verify_saved_repairs(candidate, bindings)
    for identity, indices in resolved.items():
        if (identity not in applied or identity not in sample or len(set(indices)) != len(indices)
                or any(type(i) is not int or not 0 <= i < len(sample[identity]['issue_reviews'])
                       or sample[identity]['issue_reviews'][i]['detection'] != 'confirmed' for i in indices)):
            raise ValueError('Unsupported claim that an API warning was fixed')
    db = sqlite3.connect((source/'journal.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    out.mkdir(parents=True)
    counts, groups, cases, verdicts = Counter(), Counter(), Counter(), Counter()
    digest = hashlib.sha256()
    try:
        db.execute('BEGIN')
        contexts = {}
        with gzip.open(out/'received-evidence.jsonl.gz', 'wt', encoding='utf-8', newline='\n') as evidence:
            for row in db.execute('SELECT DISTINCT x.id,x.turns FROM contexts x JOIN cases c ON json_extract(c.data,\'$.dialogue\')=x.id JOIN outputs o ON o.id=c.id ORDER BY x.id'):
                turns = json.loads(row['turns']); contexts[row['id']] = turns
                evidence.write(packed(dict(kind='context', id=row['id'], turns=turns, sha256=fingerprint(turns)))+'\n')
            with gzip.open(out/'warning-decisions.jsonl.gz', 'wt', encoding='utf-8', newline='\n') as decisions, gzip.open(out/'pending-warnings.jsonl.gz', 'wt', encoding='utf-8', newline='\n') as pending:
                for row in db.execute('SELECT o.*,c.data,c.evidence,c.flags FROM outputs o JOIN cases c ON c.id=o.id ORDER BY o.id'):
                    identity = row['id']; issue_list = json.loads(row['issues'])
                    case, proof = json.loads(row['data']), json.loads(row['evidence'])
                    context = contexts[case['dialogue']]
                    if (fingerprint(context) != proof['context_sha256']
                            or context[case['turn']-1][slice(*case['span'])] != case['text']):
                        raise ValueError('Frozen source context no longer matches its evidence')
                    digest.update((packed([identity,row['batch'],row['verdict'],row['issues']])+'\n').encode())
                    evidence.write(packed(dict(kind='result',id=identity,batch=row['batch'],verdict=row['verdict'],
                        issues=issue_list,case=case,evidence=proof,flags=json.loads(row['flags']),
                        frozen_pre_repair_input=True,publication=False))+'\n')
                    verdicts[row['verdict']] += 1
                    statuses = []
                    for index, issue in enumerate(issue_list):
                        item = disposition(identity,index,issue,case,sample.get(identity),applied,resolved)
                        decisions.write(packed(item)+'\n');counts[item['status']] += 1;statuses.append(item['status'])
                        if item['status'].startswith('pending_'):
                            pending.write(packed(item)+'\n');groups[item['group']] += 1
                    if any(s.startswith('pending_') for s in statuses):cases['with_pending_warnings'] += 1
                    elif statuses:cases['all_warnings_addressed_not_semantically_approved'] += 1
                    else:cases['api_clear_not_semantically_approved'] += 1
        total = db.execute('SELECT COUNT(*) FROM outputs').fetchone()[0]
        if sum(verdicts.values()) != total:
            raise ValueError('Export lost an API result')
        checkpoint = json.loads((ROOT/'data/dailydialog/review-parts/hybrid-luna-stop-checkpoint-20261014.json').read_text(encoding='utf-8'))
        if digest.hexdigest() != checkpoint['outputs_sha256']:
            raise ValueError('Paid output ledger changed since the stop checkpoint')
        result = dict(version='offline-luna-reconciliation-v1', source=str(source.resolve()), candidate=str(candidate.resolve()),
            received_results=total,verdicts=dict(verdicts),contexts=len(contexts),warning_counts=dict(counts),
            case_counts=dict(cases),pending_groups=dict(sorted(groups.items())),outputs_sha256=digest.hexdigest(),
            files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.glob('*.gz'))},
            sample_sha256=hashlib.sha256(sample_path.read_bytes()).hexdigest(),
            explicit_resolution_sha256=hashlib.sha256(resolution_path.read_bytes()).hexdigest(),
            reviewed_edits=len(changes),input_occurrences_with_edits=len({i for r in changes for i in r.get('input_ids', [r['id']])}),
            resolved_case_bindings_verified=len(bindings or {}),
            source_database_read_only=True,api_calls=0,published=False,
            note='Every received result exported. Pending warnings are NOT resolved by grouping or a nearby edit. Source contexts are the original API inputs, not the corrected candidate.')
        (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return result
    finally:
        db.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--sample',type=Path,required=True)
    parser.add_argument('--resolutions',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.source,args.candidate,args.sample,args.resolutions,args.out),indent=2))
