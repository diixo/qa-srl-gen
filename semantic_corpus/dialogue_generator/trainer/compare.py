"""Evaluate saved runs on identical frozen datasets; never train or select models."""

import argparse
import json
from pathlib import Path
import re


def _binding(value):
    name, separator, path = value.partition('=')
    if not separator or not re.fullmatch(r'[a-zA-Z0-9_-]+', name) or not path:
        raise argparse.ArgumentTypeError('Expected label=directory with a simple label')
    return name, Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=_binding, required=True)
    parser.add_argument('--dataset', action='append', type=_binding, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='test')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Choose a new comparison directory')
    for entries in (args.run, args.dataset):
        if len(dict(entries)) != len(entries):
            parser.error('Duplicate labels')
    from .engine import evaluate
    from .data import file_sha256

    frozen = {name: file_sha256(path / f'{args.split}.json') for name, path in args.dataset}
    args.out.mkdir(parents=True)
    results = []
    for run_name, run_path in args.run:
        for data_name, data_path in args.dataset:
            print(f'Comparing {run_name} / {data_name}', flush=True)
            output = args.out / run_name / data_name
            report = evaluate(run_path, data_path, output, split=args.split, device=args.device)
            if report['source_sha256'] != frozen[data_name]:
                raise ValueError(f'Dataset changed during comparison: {data_name}')
            results.append({'run': run_name, 'dataset': data_name, 'evaluation': str(output / 'evaluation.json'),
                            **{key: report[key] for key in ('count', 'exact_matches', 'exact_match_rate',
                                                           'unknown_count', 'unknown_false_yes', 'answer_loss',
                                                           'contrast_groups', 'source_sha256')}})
            (args.out / 'comparison.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    lines = ['| Model | Dataset | Exact match | Unknown to yes | Whole contrast groups |',
             '| --- | --- | --- | --- | --- |']
    for row in results:
        groups = row['contrast_groups']
        lines.append(f"| {row['run']} | {row['dataset']} | {row['exact_matches']}/{row['count']} "
                     f"({row['exact_match_rate']:.1%}) | {row['unknown_false_yes']}/{row['unknown_count']} | "
                     f"{groups['all_correct']}/{groups['complete']} |")
    table = '\n'.join(lines) + '\n'
    (args.out / 'comparison.md').write_text(table, encoding='utf-8')
    print(table)


if __name__ == '__main__':
    main()
