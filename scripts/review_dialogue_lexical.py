"""Audit step-4 lexical decisions plus all earlier dialogue export checks.

python scripts/review_dialogue_lexical.py artifacts/dialogue-mined-review-step4-final \
    --baseline artifacts/dialogue-mined-roles-step3 \
    --out artifacts/reviews/dialogue-review-step4-final-validation.json
"""
import argparse
from collections import Counter
import json
from hashlib import sha256
from pathlib import Path

from review_dialogue_roles import review_roles
from semantic_corpus.dialogue_generator.mined_realization import mined_lexeme
from semantic_corpus.dialogue_generator.mined_review import policy


def check_resources(manifest):
    """An audit using other rules cannot certify the recorded generation run."""
    root = Path(__file__).resolve().parents[1] / 'semantic_corpus'
    for name, expected in manifest['resources'].items():
        allowed = root
        if name.startswith('data/'):
            allowed = root.parent / 'data'
            path = root.parent / name
        elif name == 'entities.json':
            path = root / 'semantic_generator/entities.json'
        else:
            path = root / name if '/' in name else root / 'dialogue_generator' / name
        path = path.resolve()
        if not path.is_relative_to(allowed) or not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Audit resource differs from dataset manifest: {name}; use the matching revision')


def check_decisions(dataset):
    """Cross-check independent exported ledgers and the actual training events."""
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    check_resources(manifest)
    coverage = json.loads((dataset / 'mined-coverage.json').read_text(encoding='utf-8'))
    ledger = json.loads((dataset / manifest['mined_bank']['review_file']).read_text(encoding='utf-8'))
    if ledger['identity'] != manifest['mined_bank']['identity']:
        raise ValueError('Review ledger belongs to another bank')
    if ledger['policy'] != policy():
        raise ValueError('Review ledger does not contain the generation policy')
    if set(ledger['predicates']) != set(coverage['predicates']):
        raise ValueError('Review ledger lost raw inventory entries')
    totals, restrictions = Counter(), Counter()
    repaired = []
    for lemma, row in ledger['predicates'].items():
        counts = row['lexical_decisions']
        if counts != coverage['predicates'][lemma]['lexical_decisions']:
            raise ValueError(f'Inconsistent coverage/review decisions: {lemma}')
        if sum(counts.values()) != row['source_occurrences']:
            raise ValueError(f'An occurrence has no lexical decision: {lemma}')
        if sum(row['native_rejections'].values()) + row['compositional_occurrences'] != row['source_occurrences']:
            raise ValueError(f'An occurrence has no construction disposition: {lemma}')
        totals.update(counts)
        restrictions.update(row['native_rejections'])
        if counts.get('normalized_lemma') or counts.get('repaired_finite_tag'):
            repaired.append(lemma)
        for issue, example in row['examples'].items():
            if not counts.get(issue) and not row['native_rejections'].get(issue):
                raise ValueError(f'Example without counted evidence: {lemma}/{issue}')
            if not example['source_text'] or len(example['reference']) != 3:
                raise ValueError(f'Untraceable example: {lemma}/{issue}')
    if dict(totals) != ledger['lexical_decisions'] or sum(totals.values()) != coverage['train']['predicate_occurrences']:
        raise ValueError('Lexical decisions do not cover the complete occurrence inventory')
    exported = {}
    for split in ('train', 'validation', 'test'):
        decisions = Counter()
        with (dataset / (split + '.episodes.jsonl')).open(encoding='utf-8') as stream:
            for line in stream:
                episode = json.loads(line)
                source = episode.get('mined_source')
                if not source:
                    continue
                decision = source['lexical_review']
                decisions[decision['reason']] += 1
                if decision['raw_lemma'] != source['raw_lemma'] or decision['canonical'] != source['canonical_lemma']:
                    raise ValueError('An episode lost the raw/resolved lexical distinction')
                if source['realization'] != 'compositional':
                    if not source['restriction']:
                        raise ValueError('Quotation without a restriction reason')
                    continue
                if source['restriction'] is not None:
                    raise ValueError('Restricted template became compositional')
                events = episode['transitions'][1]['delta']['event_add']
                reviews = [decision, *source['companion_lexical_reviews']]
                if len(events) != len(reviews):
                    raise ValueError('A companion event bypassed lexical review')
                for claim, review in zip(events, reviews):
                    if review['status'] == 'quotation_only':
                        raise ValueError('Ambiguous/rejected form was asserted as a lexical event')
                    event = claim['event']
                    if event['predicate'].split(' ', 1)[0] != review['canonical']:
                        raise ValueError('Generated event uses another lexical head')
                    lexeme = mined_lexeme(event['predicate'])
                    generated = review['generation_paradigm']
                    if (lexeme.past, lexeme.third, lexeme.gerund, lexeme.participle) != (
                            generated['past'], generated['present_singular_3rd'],
                            generated['present_participle'], generated['past_participle']):
                        raise ValueError('Realization ignored the reviewed generation paradigm')
        exported[split] = dict(decisions)
    return {'inventory_decisions': dict(totals), 'construction_restrictions': dict(restrictions),
            'heads_with_normalization': repaired, 'exported_decisions': exported,
            'checks': ['Code/policy resources match the manifest', 'Policy snapshot matches the review ledger',
                       'Every source occurrence accounted for', 'Raw inventory preserved in review ledger',
                       'Primary and companion events passed lexical review',
                       'Generated paradigms match recorded decisions', 'Quotation reasons are explicit']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-length', type=int, default=768)
    args = parser.parse_args()
    lexical = check_decisions(args.dataset)
    report = review_roles(args.dataset, args.baseline, args.max_length)
    report['lexical_review'] = lexical
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(args.out)


if __name__ == '__main__':
    main()
