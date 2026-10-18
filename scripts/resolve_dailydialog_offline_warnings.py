"""Apply explicit, evidence-bound local judgments to a saved warning ledger.

This changes accounting only. It neither changes a parse nor approves a whole
sentence. A closed local warning is not permission to publish its template.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from semantic_corpus.template_bank import fingerprint, read_rows


def check_judgment(judgment, warning, occurrence, context):
    if (judgment['method'] != 'direct_assistant_local_warning_review'
            or judgment['review_scope'] != 'local_warning_only'
            or judgment['semantic_approval'] is not False
            or judgment['publication'] is not False
            or not judgment.get('note')):
        raise ValueError('Missing explicit limited review')
    if (judgment['id'] != warning['id']
            or judgment['issue_index'] != warning['issue_index']
            or judgment['issue_sha256'] != fingerprint(warning['issue'])
            or judgment['issue_sha256'] != warning['issue_sha256']
            or judgment['occurrence_id'] != occurrence['idx']
            or judgment['text'] != occurrence['text']
            or judgment['context_sha256'] != fingerprint(context['turns'])
            or judgment['tokens_sha256'] != fingerprint(occurrence['tokens'])):
        raise ValueError('Stale or mismatched local review')
    origin = occurrence['origin']
    turns = judgment.get('context_turns_read', [])
    sentence_only = judgment.get('sentence_span_read') == origin['span']
    if ((origin['turn'] not in turns and not sentence_only)
            or any(type(t) is not int or not 1 <= t <= len(context['turns']) for t in turns)):
        raise ValueError('Invalid reviewed context turns')
    if context['turns'][origin['turn']-1][slice(*origin['span'])] != occurrence['text']:
        raise ValueError('Source address changed')
    if judgment['finding'] == 'fixed_property_link':
        revision=occurrence.get('attribute_target_revision',{})
        if (judgment.get('repair_sha256')!=fingerprint(revision)
                or not revision.get('preimage')
                or fingerprint(revision['preimage'])!=revision.get('preimage_sha256')
                or revision.get('semantic_approval') is not False
                or not judgment.get('verified_attributes')):
            raise ValueError('Property repair evidence missing or changed')
        changes={r['slot']:r for r in revision['attribute_changes']}
        attributes={r['slot']:r for r in occurrence['attributes']}
        for spec in judgment['verified_attributes']:
            slot=spec['slot']
            if (slot not in changes or slot not in attributes
                    or spec['target_slot']!=changes[slot]['target_slot']
                    or any(attributes[slot].get(k)!=v for k,v in spec.items())):
                raise ValueError('Reviewed property link changed')
        return 'fixed_in_candidate'
    if judgment['finding'] in {'fixed_local_syntax', 'already_satisfied_local_syntax', 'representation_only'}:
        if not judgment.get('verified_tokens'):
            raise ValueError('A local decision needs checked tokens')
        for spec in judgment['verified_tokens']:
            i = spec.get('i')
            if (type(i) is not int or not 0 <= i < len(occurrence['tokens'])
                    or len(spec) < 2
                    or any(occurrence['tokens'][i].get(k) != v for k, v in spec.items())):
                raise ValueError('Reviewed token changed')
        if judgment['finding'] == 'representation_only':
            indices = {s['i'] for s in judgment['verified_tokens']}
            if indices != set(warning['issue']['tokens']):
                raise ValueError('Representation review must cover the exact warning tokens')
            kind = judgment.get('representation_kind')
            for i in indices:
                t = occurrence['tokens'][i]
                if kind == 'interrogative_adverb_pos':
                    valid = (t.get('text', '').lower() in {'how', 'where', 'why', 'when'}
                             and t.get('pos') == 'SCONJ' and t.get('tag') == 'WRB'
                             and t.get('dep') in {'advmod', 'ROOT', 'pobj'})
                elif kind == 'participle_morph_not_clause_aspect':
                    valid = (t.get('pos') == 'VERB' and t.get('tag') == 'VBN'
                             and 'VerbForm=Part' in t.get('morph', '').split('|')
                             and 'Aspect=Perf' in t.get('morph', '').split('|'))
                else:
                    valid = False
                if not valid:
                    raise ValueError('Token does not match the explicitly reviewed convention')
            return 'closed_representation_only'
        if judgment['finding'] == 'already_satisfied_local_syntax':
            return 'closed_not_supported'
        revision_key = judgment.get('repair_kind', 'automatic_revision')
        if revision_key not in {'automatic_revision', 'local_syntax_revision'}:
            raise ValueError('Unsupported repair evidence')
        revision = occurrence.get(revision_key, {})
        if (not revision.get('preimage') or judgment.get('repair_sha256') != fingerprint(revision)
                or revision.get('semantic_approval', False) is not False):
            raise ValueError('A closed fix needs its retained repair and checked tokens')
        if revision_key == 'local_syntax_revision':
            review = revision.get('review', {})
            if (review.get('id') != occurrence['idx']
                    or review.get('preimage') != fingerprint(revision['preimage'])
                    or review.get('semantic_approval') is not False
                    or review.get('publication') is not False):
                raise ValueError('Local syntax review lost its preimage binding')
        return 'fixed_in_candidate'
    if judgment['finding'] == 'needs_additional_repair':
        return 'pending_context_or_source_review'
    raise ValueError('Unsupported finding')


def run(candidate, ledger, reviews, out):
    if out.exists():
        raise ValueError('Use a fresh ledger directory')
    report = json.loads((candidate/'report.json').read_text(encoding='utf-8'))
    if report['validation']['status'] != 'passed' or report['conservation']['status'] != 'passed':
        raise ValueError('Candidate is not validated')
    judgments = list(read_rows(reviews))
    keys = {(r['id'], r['issue_index']): r for r in judgments}
    if len(keys) != len(judgments):
        raise ValueError('Duplicate review address')
    warnings = list(read_rows(ledger/'warning-decisions.jsonl.gz'))
    by_key = {(r['id'], r['issue_index']): r for r in warnings}
    if not keys.keys() <= by_key.keys():
        raise ValueError('Unknown warning address')
    for key in keys:
        if by_key[key]['status'] != 'pending_context_or_source_review':
            raise ValueError('Already closed warning cannot be overwritten')
    ids = {r['occurrence_id'] for r in judgments}
    current = {}
    for path in sorted((candidate/'bank/occurrences').glob('*.jsonl.gz')):
        for row in read_rows(path):
            if row['idx'] in ids:
                if row['idx'] in current:
                    raise ValueError('Duplicate occurrence')
                current[row['idx']] = row
    if current.keys() != ids:
        raise ValueError('Missing reviewed occurrence')
    contexts = {r['id']: r for r in read_rows(candidate/'contexts.jsonl.gz')}
    for key, judgment in keys.items():
        warning = by_key[key]
        occurrence = current[judgment['occurrence_id']]
        status = check_judgment(judgment, warning, occurrence, contexts[occurrence['context_id']])
        warning.update(status=status, local_review=judgment, reason=judgment['note'])
    out.mkdir(parents=True)
    evidence = ledger/'received-evidence.jsonl.gz'
    shutil.copyfile(evidence, out/evidence.name)
    for name, rows in [('warning-decisions.jsonl.gz', warnings),
                       ('pending-warnings.jsonl.gz', [r for r in warnings if r['status'].startswith('pending_')])]:
        with gzip.open(out/name, 'wt', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
    counts = Counter(r['status'] for r in warnings)
    summary = dict(candidate=str(candidate.resolve()), ledger=str(ledger.resolve()),
        review_file=str(reviews.resolve()), review_sha256=hashlib.sha256(reviews.read_bytes()).hexdigest(),
        warning_counts=dict(counts), reviewed_local_warnings=len(judgments),
        newly_closed=counts['fixed_in_candidate']-sum(r['status']=='fixed_in_candidate' for r in read_rows(ledger/'warning-decisions.jsonl.gz')),
        pending_groups=dict(Counter(r['group'] for r in warnings if r['status'].startswith('pending_'))),
        all_original_warnings_retained=True, semantic_review_complete=False, api_calls=0, published=False,
        received_evidence=str((out/evidence.name).resolve()),
        received_evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('candidate', 'ledger', 'reviews', 'out'):
        parser.add_argument('--'+flag, type=Path, required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
