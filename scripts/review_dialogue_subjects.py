"""Audit a subject-aware dialogue export without downloading or training a model.

python scripts/review_dialogue_subjects.py artifacts/dialogue-mined-subjects-step1 \
    --out artifacts/reviews/dialogue-subjects-step1-validation.json
"""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semantic_corpus.dialogue_generator.token_limits import GPT2TokenCounter
from semantic_corpus.dialogue_generator.trainer.data import check_disjoint, file_sha256, load_rows, validate_view


def review(dataset, max_length):
    tokenizer = GPT2TokenCounter()
    report = {'dataset': str(dataset), 'max_length': max_length, 'tokenizer': tokenizer.provenance,
              'checks': ['Extraction targets parse and apply', 'no split prompt/scenario/group overlap',
                         'paired Memory/History answers', 'full ChatML length without truncation'],
              'files': {}, 'subjects': {}}
    paired = {}
    for view in ('memory', 'history', 'extraction'):
        splits = {}
        for split in ('train', 'validation', 'test'):
            path = dataset / view / (split + '.json')
            rows = load_rows(path)
            validate_view(rows, view, split)
            lengths = [tokenizer.length(row.messages) for row in rows]
            if max(lengths) > max_length:
                raise ValueError(f'{view}/{split} exceeds {max_length} tokens: {max(lengths)}')
            splits[split] = rows
            report['files'][f'{view}/{split}'] = {'count': len(rows), 'sha256': file_sha256(path),
                                                 'max_tokens': max(lengths)}
            if view == 'memory':
                paired[split] = [(row.scenario_key, row.messages[-1]['content']) for row in rows]
            elif view == 'history':
                if paired[split] != [(row.scenario_key, row.messages[-1]['content']) for row in rows]:
                    raise ValueError(f'Memory/History answer mismatch: {split}')
            print(view, split, report['files'][f'{view}/{split}'], flush=True)
        for left, right in combinations(splits, 2):
            check_disjoint(splits[left], splits[right])
    coverage = json.loads((dataset / 'mined-coverage.json').read_text(encoding='utf-8'))
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    report['dispositions'] = coverage['train']['dispositions']
    scenario_keys, contrast_groups = {}, {}
    for split in ('train', 'validation', 'test'):
        scenario_keys[split], contrast_groups[split] = set(), set()
        labels, persons, numbers, realizations = Counter(), Counter(), Counter(), Counter()
        outcomes, examples = defaultdict(set), {}
        with (dataset / (split + '.episodes.jsonl')).open(encoding='utf-8') as stream:
            for line in stream:
                episode = json.loads(line)
                scenario_keys[split].add(episode['scenario_key'])
                if episode['contrast_group']:
                    contrast_groups[split].add(episode['contrast_group'])
                source = episode.get('mined_source')
                if not source:
                    continue
                realizations[source['realization']] += 1
                outcomes[source['raw_lemma']].add(episode['condition'].rsplit(':', 1)[-1])
                subject = episode['query']['event'].get('subject')
                if not subject:
                    continue
                labels[subject['label']] += 1
                persons[subject['text'] if subject['text'] in ('user', 'assistant') else 'other'] += 1
                numbers[subject['number']] += 1
                examples.setdefault(subject['label'], {'subject': subject, 'source': source['source_text'],
                                                       'turns': episode['turns']})
        if split == 'train' and manifest['mined_train_requested'] == 0:
            if coverage[split]['unselected_lemmas'] or len(outcomes) != coverage[split]['available_lemmas']:
                raise ValueError('Full inventory coverage was requested but not achieved')
            if any(values != {'yes', 'no', 'unknown'} for values in outcomes.values()):
                raise ValueError('A predicate is missing a requested outcome')
        report['subjects'][split] = {'labels': dict(labels), 'persons': dict(persons),
                                     'numbers': dict(numbers), 'realizations': dict(realizations),
                                     'covered_raw_lemmas': len(outcomes), 'examples': examples}
    # Audit metadata survives even when training exports contain only messages.
    for left, right in combinations(scenario_keys, 2):
        if scenario_keys[left] & scenario_keys[right] or contrast_groups[left] & contrast_groups[right]:
            raise ValueError(f'Audit scenarios or contrast groups overlap: {left}/{right}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-length', type=int, default=768)
    args = parser.parse_args()
    report = review(args.dataset, args.max_length)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(args.out)


if __name__ == '__main__':
    main()
