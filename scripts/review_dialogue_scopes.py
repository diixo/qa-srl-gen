"""Audit step-2 source scopes plus the complete subject/export contract.

python scripts/review_dialogue_scopes.py artifacts/dialogue-mined-scopes-step2 \
    --baseline artifacts/dialogue-mined-subjects-step1 \
    --out artifacts/reviews/dialogue-scopes-step2-validation.json
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from review_dialogue_subjects import review


def review_scopes(dataset, baseline, max_length):
    report = review(dataset, max_length)
    coverage = json.loads((dataset / 'mined-coverage.json').read_text(encoding='utf-8'))
    report['scope_coverage'] = {key: coverage['train'][key]
                              for key in ('source_scope_lemmas', 'source_scope_occurrences')}
    report['scopes'] = {}
    for split in ('train', 'validation', 'test'):
        origins, scopes, examples = Counter(), Counter(), {}
        with (dataset / (split + '.episodes.jsonl')).open(encoding='utf-8') as stream:
            for line in stream:
                episode = json.loads(line)
                source = episode.get('mined_source')
                if not source or source['realization'] != 'compositional':
                    continue
                origin = source['scope_origin']
                origins[origin] += 1
                if origin == 'source':
                    scope = source['source_scope']
                    initial = episode['transitions'][1]['delta']['event_add'][0]['event']
                    if any(initial[key] != value for key, value in scope.items()):
                        raise ValueError(f'Source scope changed in {episode["id"]}')
                    key = f'{">".join(scope["operators"]) or "none"}/{scope["phase"]}/{scope["tense"]}'
                    scopes[key] += 1
                    examples.setdefault(key, {'source': source['source_text'], 'scope': scope,
                                               'turns': episode['turns'], 'answer': episode['answer']})
                expected = episode['condition'].rsplit(':', 1)[-1]
                answer = episode['answer']
                if not answer.startswith({'yes': 'Yes,', 'no': 'No,', 'unknown': "I don't know"}[expected]):
                    raise ValueError(f'Wrong contrast outcome: {episode["id"]}')
        report['scopes'][split] = {'origins': dict(origins), 'profiles': dict(scopes), 'examples': examples}
    if baseline:
        old = json.loads((baseline / 'mined-coverage.json').read_text(encoding='utf-8'))
        if old['train']['sources']['banks'] != coverage['train']['sources']['banks']:
            raise ValueError('Source banks changed relative to baseline')
        if old['train']['sources']['dictionary_sha256'] != coverage['train']['sources']['dictionary_sha256']:
            raise ValueError('Dictionary changed relative to baseline')
        if set(old['predicates']) != set(coverage['predicates']):
            raise ValueError('Predicate inventory changed')
        gained, restricted = [], {}
        for lemma, entry in coverage['predicates'].items():
            was = old['predicates'][lemma]['status'] == 'compositional'
            now = entry['status'] == 'compositional'
            if now and not was:
                gained.append(lemma)
            elif was and not now:
                restricted[lemma] = entry['native_rejections']
        report['baseline_comparison'] = {'baseline': str(baseline), 'archives_unchanged': True,
                                         'dictionary_unchanged': True, 'inventory_unchanged': True,
                                         'before': old['train']['dispositions'], 'after': coverage['train']['dispositions'],
                                         'gained': gained, 'restricted': restricted}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-length', type=int, default=768)
    args = parser.parse_args()
    report = review_scopes(args.dataset, args.baseline, args.max_length)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(args.out)


if __name__ == '__main__':
    main()
