"""Verify explicit local rechecks after a source revision, without admission.

Each decision binds the original paid warning, its prior disposition, current
source lineage, context, complete before/after parse and inspected token heads.
Unaddressed dispositions remain identical. No model or parser is called here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_dailydialog_hybrid_bank import open_overlay
from scripts.benchmark_dailydialog_hybrid import dump, write
from scripts.import_dailydialog_sentences import digest
from semantic_corpus.template_bank import fingerprint, read_rows

METHOD = 'direct_assistant_reparsed_warning_review_v1'
FIELDS = {'pos', 'tag', 'dep', 'head', 'parser_lemma', 'morph'}
STATUSES = {
    'fixed': 'fixed_in_candidate',
    'verified_current': 'closed_verified_after_source_revision',
    'representation': 'closed_representation_only',
    'not_supported': 'closed_not_supported',
    'boundary_only': 'fixed_boundary_only',
    'unresolved': 'quarantined_unresolved',
}


def key(row):
    return row['id'], row['issue_index'], row['issue_sha256']


def token_identity(token):
    return {k: token[k] for k in ('i', 'text', 'span')}


def verify_lineage(warning, review, before, after, context):
    identity = review['occurrence_id']
    lineage = [r for r in warning['output_occurrences'] if r['id'] == identity]
    if len(lineage) != 1 or review['lineage'] != lineage[0]:
        raise ValueError('Wrong current occurrence or source lineage')
    for image, hash_key in ((before, 'before_sha256'), (after, 'after_sha256')):
        if fingerprint(image) != review[hash_key]:
            raise ValueError('Stale parse preimage/postimage')
        body, payload = image['body'], image['payload']
        origin = body['provenance']['origin']
        if (body['idx'] != identity or body['example'] != review['text']
                or origin != review['origin'] or body['example'] != lineage[0]['text']
                or fingerprint(context) != review['context_sha256']
                or context[origin['turn'] - 1][slice(*origin['span'])] != body['example']):
            raise ValueError('Changed source text, address or context')
        revision = payload['ledger'].get('candidate_source_revision', {})
        if (revision.get('plan_sha256') != lineage[0]['plan_sha256']
                or [r['body']['idx'] for r in revision.get('preimages', [])] != lineage[0]['input_ids']
                or revision.get('semantic_approval') is not False):
            raise ValueError('Source revision lost its lineage or preimages')


def resolve(warning, review, before, after, context, boundary_outputs=()):
    """Check one human/model-authored local judgment against actual bank data."""
    if (key(review) != key(warning)
            or review['previous_disposition_sha256'] != fingerprint(warning)
            or warning['status'] != 'quarantined_after_source_revision'
            or review['method'] != METHOD or review['review_scope'] != 'local_warning_only'
            or review['finding'] not in STATUSES or not review['note'].strip()
            or review['semantic_approval'] is not False or review['publication'] is not False):
        raise ValueError('Stale, incomplete or overbroad local decision')
    previous = warning['previous_warning']
    if (fingerprint(previous) != warning['previous_warning_sha256']
            or fingerprint(previous['issue']) != warning['issue_sha256']):
        raise ValueError('Original warning evidence changed')
    verify_lineage(warning, review, before, after, context)
    tokens = after['payload']['ledger']['tokens']
    old_tokens = before['payload']['ledger']['tokens']
    if [token_identity(t) for t in tokens] != [token_identity(t) for t in old_tokens]:
        raise ValueError('Local syntax recheck cannot change tokenization')
    seen, changed = set(), False
    for proof in review['token_proofs']:
        i = proof['token']['i']
        if type(i) is not int or i in seen or not 0 <= i < len(tokens):
            raise ValueError('Duplicate or invalid inspected token')
        seen.add(i)
        token = tokens[i]
        if (token_identity(token) != proof['token']
                or set(proof['expected']) != FIELDS
                or any(token[k] != v for k, v in proof['expected'].items())
                or proof['head_token'] != token_identity(tokens[token['head']])):
            raise ValueError('Token/lemma/head proof does not match the current parse')
        changed |= any(old_tokens[i][k] != token[k] for k in FIELDS)
    attrs = {r['slot']: r.get('target_slot') for r in after['payload']['record']['attributes']}
    if any(k not in attrs or attrs[k] != v for k, v in review['attribute_targets'].items()):
        raise ValueError('Property target proof does not match the compiled record')
    finding = review['finding']
    if finding not in {'unresolved', 'boundary_only'} and not seen:
        raise ValueError('Local closure requires inspected token evidence')
    if finding == 'fixed' and not changed:
        raise ValueError('Claimed fix did not change any inspected token')
    if finding == 'unresolved' and (seen or review['attribute_targets']):
        raise ValueError('Unresolved decision must not carry an approval proof')
    if finding == 'boundary_only':
        expected_ids = [r['id'] for r in warning['output_occurrences']]
        if ([r['id'] for r in review['boundary_outputs']] != expected_ids
                or list(boundary_outputs) != review['boundary_outputs']):
            raise ValueError('Boundary decision must cover all current output segments')
    elif review['boundary_outputs']:
        raise ValueError('Unexpected boundary evidence')
    result = deepcopy(warning)
    result.update(status=STATUSES[finding], reason=review['note'],
                  previous_disposition=deepcopy(warning),
                  previous_disposition_sha256=fingerprint(warning),
                  local_recheck=deepcopy(review), semantic_approval=False, publication=False)
    return result


def run(candidate, ledger, reviews, out):
    candidate, ledger, reviews, out = (Path(p).resolve() for p in (candidate, ledger, reviews, out))
    if out.exists():
        raise ValueError('Use a fresh recheck ledger directory')
    report = json.loads((candidate / 'report.json').read_text(encoding='utf-8'))
    if report['validation']['status'] != 'passed' or report['conservation']['status'] != 'passed':
        raise ValueError('Candidate must pass validation and conservation')
    plan = json.loads((candidate / 'plan.json').read_text(encoding='utf-8'))
    source = Path(plan['source'])
    prior_summary = json.loads((ledger / 'summary.json').read_text(encoding='utf-8'))
    if source.resolve() != Path(prior_summary['candidate']).resolve():
        raise ValueError('Recheck ledger belongs to another candidate')
    if digest(Path(plan['reviews'])) != plan['review_sha256']:
        raise ValueError('Applied local patches changed')
    warnings, decisions = list(read_rows(ledger / 'warning-decisions.jsonl.gz')), list(read_rows(reviews))
    known, local = {key(w): w for w in warnings}, {key(r): r for r in decisions}
    if (len(known) != len(warnings) or len(local) != len(decisions)
            or not local or not local.keys() <= known.keys()):
        raise ValueError('Empty, duplicate or unknown review identity')
    # Only temporary parse overlays are placed under artifacts. Durable review
    # evidence never acquires a large rebuild/cache dependency.
    temporary = tempfile.TemporaryDirectory(prefix='quarantine-recheck-', dir=candidate.parent)
    workspace = Path(temporary.name).resolve()
    if workspace.parent != candidate.parent:
        raise ValueError('Temporary overlay must remain beside the candidate')
    databases = []
    try:
        for name, root in (('before', source), ('after', candidate)):
            work = workspace / name
            work.mkdir()
            db, *_ = open_overlay(root, work)
            databases.append(db)
        contexts = {r['id']: r['turns'] for r in read_rows(candidate / 'contexts.jsonl.gz')}
        if digest(source / 'contexts.jsonl.gz') != digest(candidate / 'contexts.jsonl.gz'):
            raise ValueError('Context changed during local syntax repair')

        def image(db, identity):
            row = db.execute('SELECT o.body,p.payload FROM occurrences o JOIN parses p ON o.key=p.key WHERE o.id=?', (identity,)).fetchone()
            if row is None:
                raise ValueError('Current occurrence is missing: ' + identity)
            body, payload = map(json.loads, row)
            return dict(body=body, payload=payload)

        results = []
        for warning in warnings:
            review = local.get(key(warning))
            if review is None:
                results.append(warning)
                continue
            identity = review['occurrence_id']
            before, after = [image(db, identity) for db in databases]
            boundary = []
            if review['finding'] == 'boundary_only':
                for item in warning['output_occurrences']:
                    body = image(databases[1], item['id'])['body']
                    origin = body['provenance']['origin']
                    context = contexts[body['provenance']['context_id']]
                    if context[origin['turn'] - 1][slice(*origin['span'])] != body['example']:
                        raise ValueError('Boundary evidence does not address the current source')
                    boundary.append(dict(id=body['idx'], text=body['example'], origin=origin,
                                         context_sha256=fingerprint(context)))
            results.append(resolve(warning, review, before, after,
                                   contexts[after['body']['provenance']['context_id']], boundary))
    finally:
        for db in databases:
            db.close()
        temporary.cleanup()
    counts = Counter(r['status'] for r in results)
    quarantine = [r for r in results if r['status'].startswith('quarantined_')]
    out.mkdir(parents=True)
    for name, rows in (('warning-decisions.jsonl.gz', results), ('quarantined-warnings.jsonl.gz', quarantine)):
        with gzip.open(out / name, 'wt', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                write(stream, row)
    shutil.copyfile(ledger / 'received-evidence.jsonl.gz', out / 'received-evidence.jsonl.gz')
    shutil.copyfile(reviews, out / reviews.name)
    summary = dict(version='offline_reparsed_recheck_v1', candidate=str(candidate),
                   previous_ledger=str(ledger), saved_warnings=len(warnings),
                   rechecked_warnings=len(decisions), findings=dict(Counter(r['finding'] for r in decisions)),
                   warning_counts=dict(counts), quarantined_warnings=len(quarantine),
                   quarantined_cases=len({r['id'] for r in quarantine}),
                   quarantine_groups=dict(Counter(r['group'] for r in quarantine)),
                   review_sha256=digest(reviews), prior_warning_ledger_sha256=digest(ledger / 'warning-decisions.jsonl.gz'),
                   received_evidence_sha256=digest(out / 'received-evidence.jsonl.gz'),
                   candidate_report=report, semantic_review_complete=False,
                   full_candidate_published=False, api_calls=0, clear_is_not_admission=True)
    dump(out / 'summary.json', summary)
    saved = list(read_rows(out / 'warning-decisions.jsonl.gz'))
    for old, new in zip(warnings, saved):
        if key(old) in local:
            if new['previous_disposition'] != old:
                raise ValueError('Prior disposition was not retained exactly')
        elif new != old:
            raise ValueError('An unaddressed disposition changed')
    if len(saved) != len(warnings) or digest(out / 'received-evidence.jsonl.gz') != digest(ledger / 'received-evidence.jsonl.gz'):
        raise ValueError('Saved review or paid evidence lost data')
    dump(out / 'verification.json', dict(status='passed', warnings=len(saved),
         rechecked=len(local), untouched_dispositions=len(saved)-len(local),
         exact_prior_dispositions_preserved=True, paid_evidence_identical=True,
         current_tokens_heads_attributes_context_and_lineage_verified=True,
         semantic_approval=False, publication=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'ledger', 'reviews', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
