"""Build an offline review snapshot; grouping never approves API suggestions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from semantic_corpus.template_bank import fingerprint

SEED = '20261014-luna-offline-review-v1'
PATTERNS = (
    ('direct_address', r'vocative|direct.address|address term'),
    ('lemma_or_inflection', r'lemma|inflection|verb form|past participle|past tense'),
    ('sentence_boundary', r'boundary|boundaries|sentence split|segment|multiple roots|restart'),
    ('property_attachment', r'adjective|property|attribute|adjectival'),
    ('particle_or_preposition', r'particle|phrasal|preposition|\bpobj\b|\bpcomp\b'),
    ('clause_attachment', r'\bccomp\b|\bxcomp\b|\brelcl\b|\badvcl\b|relative clause|embedded|complement clause'),
    ('coordination', r'coordina|conjunct|\bconj\b'),
    ('negation_or_modality', r'negation|negative|modal|intention|conditional'),
)


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def issue_group(issue):
    if issue['category'] in {'source', 'ambiguity'}:
        return issue['category']
    text = issue['explanation'] + ' ' + issue['correction']
    return next((name for name, pattern in PATTERNS if re.search(pattern, text, re.I)), 'other_syntax')


def run(source, out, *, clear_count=50, issue_count=50, replace_snapshot=False):
    generated = {'triage.json', 'groups.jsonl', 'review-inputs.jsonl'}
    if (out.exists() and (not replace_snapshot or {p.name for p in out.iterdir()} != generated)
            or clear_count < 1 or issue_count < 1):
        raise ValueError('Use a fresh output and positive sample sizes')
    db = sqlite3.connect((source/'journal.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute('BEGIN')
        plan = json.loads((source/'plan.json').read_text(encoding='utf-8'))
        groups = defaultdict(list)
        categories, verdicts, splits = Counter(), Counter(), Counter()
        clear, flagged, outputs = [], [], {}
        digest = hashlib.sha256()
        rows = db.execute('SELECT o.*,j.phase FROM outputs o JOIN jobs j ON j.id=o.batch ORDER BY o.id')
        for row in rows:
            digest.update((packed([row['id'], row['batch'], row['verdict'], row['issues']])+'\n').encode())
            item = dict(id=row['id'], batch=row['batch'], verdict=row['verdict'], issues=json.loads(row['issues']))
            item['output_sha256'] = fingerprint(item)
            outputs[row['id']] = item
            verdicts[row['verdict']] += 1
            splits[row['id'].split(':')[0]] += 1
            if row['verdict'] == 'clear' and row['phase'] == 'bulk':
                clear.append(row['id'])
            elif row['verdict'] != 'clear' and row['phase'] == 'bulk':
                flagged.append(row['id'])
            case_groups = set()
            for issue in item['issues']:
                categories[issue['category']] += 1
                case_groups.add(issue_group(issue))
            if row['verdict'] == 'uncertain' and not case_groups:
                case_groups.add('uncertain_without_issue')
            for group in sorted(case_groups):
                groups[group].append(row['id'])
        rank = lambda identity: fingerprint([SEED, identity])
        clear_ids = sorted(clear, key=rank)[:clear_count]
        issue_ids = sorted(flagged, key=rank)[:issue_count]
        selected = [(i, 'random_clear') for i in clear_ids] + [(i, 'random_flagged') for i in issue_ids]
        if len({i for i, _ in selected}) != len(selected):
            raise ValueError('Overlapping review samples')
        bundles = []
        for identity, stratum in selected:
            row = db.execute('SELECT data,evidence,flags FROM cases WHERE id=?', (identity,)).fetchone()
            item, evidence = json.loads(row['data']), json.loads(row['evidence'])
            context_row = db.execute('SELECT turns FROM contexts WHERE id=?', (item['dialogue'],)).fetchone()
            context = json.loads(context_row['turns'])
            if fingerprint(context) != evidence['context_sha256']:
                raise ValueError('Frozen context changed: '+identity)
            if context[item['turn']-1][slice(*item['span'])] != item['text']:
                raise ValueError('Source address changed: '+identity)
            bundles.append(dict(id=identity, stratum=stratum, case=item, context=context,
                evidence=evidence, flags=json.loads(row['flags']), api=outputs[identity],
                review_status='UNREVIEWED', publication=False))
        summary = dict(version='offline-api-triage-v1', source=str(source.resolve()),
            bank_source=plan['source'], plan_sha256=fingerprint(plan), outputs_sha256=digest.hexdigest(),
            seed=SEED, outputs=len(outputs), verdicts=dict(verdicts), received_splits=dict(splits),
            issue_categories=dict(categories), groups={k:len(v) for k,v in sorted(groups.items())},
            random_clear_population=len(clear), random_clear_sample=clear_ids,
            random_flagged_population=len(flagged), random_flagged_sample=issue_ids,
            grouping='Keyword routing of API issue prose only; groups may overlap and are not adjudications.',
            sampling_scope='Stratified random sample of received bulk results: clear and non-clear. Requires population weights; not representative of the unprocessed corpus.',
            api_calls=0, source_database_read_only=True, publication=False)
        out.mkdir(parents=True, exist_ok=replace_snapshot)
        (out/'triage.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        with (out/'groups.jsonl').open('w', encoding='utf-8', newline='\n') as f:
            for group, ids in sorted(groups.items()):
                f.write(packed(dict(group=group, cases=ids, status='UNREVIEWED'))+'\n')
        with (out/'review-inputs.jsonl').open('w', encoding='utf-8', newline='\n') as f:
            for item in bundles:
                f.write(packed(item)+'\n')
        return {k:v for k,v in summary.items() if not k.endswith('_sample')}
    finally:
        db.close()


def rate(successes, total):
    """Descriptive rate with a Wilson 95% interval (sampling uncertainty only)."""
    if not 0 <= successes <= total or total == 0:
        raise ValueError('Invalid rate denominator')
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return dict(count=successes, total=total, rate=p,
                wilson_95=[max(0., center-half), min(1., center+half)])


def summarize_review(triage, rows):
    """Summarize explicit judgments, never infer them from API verdicts."""
    expected = dict(random_clear=set(triage['random_clear_sample']),
                    random_flagged=set(triage['random_flagged_sample']))
    if len(rows) != len({r['id'] for r in rows}):
        raise ValueError('Duplicate reviewed case')
    for stratum, ids in expected.items():
        if {r['id'] for r in rows if r['stratum'] == stratum} != ids:
            raise ValueError('Review does not cover the frozen sample')
    for r in rows:
        if r['stratum'] not in expected or r['publication'] is not False:
            raise ValueError('Unknown stratum or publication approval')
        c, b, api = r['case'], r['independent'], r['api']
        if (fingerprint(c) != b['case_sha256']
                or fingerprint(r['context']) != b['context_sha256']
                or r['context'][c['turn']-1][slice(*c['span'])] != c['text']
                or fingerprint({k: api[k] for k in ('id', 'batch', 'verdict', 'issues')}) != api['output_sha256']):
            raise ValueError('Review evidence changed')
        if (r['final_finding'] not in {'error', 'acceptable', 'uncertain'}
                or [d['index'] for d in r['issue_reviews']] != list(range(len(api['issues'])))
                or any(d['detection'] not in {'confirmed', 'not_demonstrated', 'unresolved'} for d in r['issue_reviews'])
                or r['useful_detection'] != any(d['detection'] == 'confirmed' for d in r['issue_reviews'])
                or (api['verdict'] == 'clear') != (r['stratum'] == 'random_clear')):
            raise ValueError('Incomplete or inconsistent adjudication')
    clear = [r for r in rows if r['stratum'] == 'random_clear']
    flagged = [r for r in rows if r['stratum'] == 'random_flagged']
    useful = rate(sum(r['useful_detection'] for r in flagged), len(flagged))
    missed = rate(sum(r['final_finding'] == 'error' for r in clear), len(clear))
    nc, nf = triage['random_clear_population'], triage['random_flagged_population']
    if nc < len(clear) or nf < len(flagged):
        raise ValueError('Sample exceeds stratum population')
    weight = nf / (nc + nf)
    return dict(
        version='luna-usefulness-sample-v1', seed=triage['seed'],
        outputs_sha256=triage['outputs_sha256'], plan_sha256=triage['plan_sha256'],
        received_results=triage['outputs'], received_splits=triage['received_splits'],
        sampling_population=dict(clear=nc, flagged=nf, total=nc+nf, pilot_excluded=triage['outputs']-nc-nf),
        sample=dict(total=len(rows), clear=len(clear), flagged=len(flagged),
            final_findings={s:dict(Counter(r['final_finding'] for r in rows if r['stratum']==s)) for s in expected},
            blind_findings={s:dict(Counter(r['independent']['finding'] for r in rows if r['stratum']==s)) for s in expected}),
        flagged_with_confirmed_useful_detection=useful,
        clear_with_missed_error=missed,
        issue_detection_counts=dict(Counter(d['detection'] for r in rows for d in r['issue_reviews'])),
        correction_review_counts=dict(Counter(d['correction'] for r in rows for d in r['issue_reviews'])),
        weighted_confirmed_detection=dict(
            rate_among_received_bulk=weight*useful['rate'],
            estimated_cases=nf*useful['rate'],
            sampling_interval_cases=[nf*x for x in useful['wilson_95']],
            interpretation='Cases with at least one useful warning, NOT fully corrected cases or training improvement.'),
        limitations=[
            'Only received bulk results; queue ordering is not random across train/test/val or the unprocessed remainder.',
            'Assistant review, not an independent human gold standard; blind initial judgments are preserved separately.',
            'Initial blind token display omitted morphology; case 11 was corrected after API reveal. Post-reveal adjudications are explicit.',
            'Wilson intervals describe random-sampling uncertainty, not reviewer error or correlated dialogue phenomena.',
            'No inference about trained-model accuracy, complete error recall, or publication safety.',
            'Minimal-edit proposals still require compilation and validation. No repairs or publication in this evaluation.'
        ], api_calls=0, publication=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--replace-snapshot', action='store_true', help='Replace only this script\'s three generated files, before manual review')
    parser.add_argument('--review', type=Path, help='Offline adjudicated sample to summarize')
    parser.add_argument('--triage', type=Path, help='Frozen triage.json for the adjudicated sample')
    parser.add_argument('--report', type=Path, help='Write evaluation JSON here')
    args = parser.parse_args()
    if args.review:
        if not args.triage or not args.report or args.source or args.out or args.replace_snapshot:
            parser.error('--review requires --triage and --report, without snapshot options')
        rows = [json.loads(line) for line in args.review.read_text(encoding='utf-8').splitlines()]
        summary = summarize_review(json.loads(args.triage.read_text(encoding='utf-8')), rows)
        summary['review_sha256'] = hashlib.sha256(args.review.read_bytes()).hexdigest()
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(summary, indent=2))
    else:
        if not args.source or not args.out or args.triage or args.report:
            parser.error('Snapshot mode requires --source and --out')
        print(json.dumps(run(args.source, args.out, replace_snapshot=args.replace_snapshot), indent=2))
