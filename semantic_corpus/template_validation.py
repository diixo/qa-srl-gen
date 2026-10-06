"""Verify a complete bank against every source record and render every template.

This check uses the compiled data only; no NLP model or network is required.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from .template_bank import TemplateBank, fingerprint, read_rows


def _check_rendered(row):
    text = row['text']
    for name, (start, end) in row['spans'].items():
        if not 0 <= start < end <= len(text) or text[start:end] != row['bindings'][name]:
            raise ValueError('invalid rendered slot offsets')
    for group in row['predicate_groups']:
        if group['head_span'] != row['spans'][group['head_slot']]:
            raise ValueError('invalid predicate head offsets')
        for construction in group['constructions']:
            for marker in construction['markers']:
                start, end = marker['span']
                if not 0 <= start < end <= len(text) or text[start:end] != marker['text']:
                    raise ValueError('invalid predicate marker offsets')
    for attribute in row['attributes']:
        if attribute['span'] != row['spans'][attribute['slot']]:
            raise ValueError('invalid adjective offsets')
        if attribute['target_span'] != row['spans'].get(attribute['target_slot']):
            raise ValueError('invalid adjective target offsets')


def validate_bank(bank, source, *, rendered=None):
    """Reject truncated, unmatched or inconsistent inventories and ledgers."""
    counts, template_counts = Counter(), {}
    attested = set()
    sink = None
    if rendered is not None:
        rendered = Path(rendered)
        rendered.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if rendered.suffix == '.gz' else open
        sink = opener(rendered, 'xt', encoding='utf-8')
    try:
        for template in bank.templates():
            template_counts[template.template_id] = template.record['count']
            counts['templates'] += 1
            for binding in template.record['attested_bindings']:
                row = template.render(binding)
                _check_rendered(row)
                attested.add(bytes.fromhex(fingerprint([template.template_id, row['text']])))
                counts['attested_roundtrips'] += 1
            example = template.sample(seed=0)
            _check_rendered(example)
            if sink is not None:
                sink.write(json.dumps(example, ensure_ascii=False, separators=(',', ':')) + '\n')
            if counts['templates'] % 50000 == 0:
                print(f"Validated {counts['templates']:,} executable templates", flush=True)
    finally:
        if sink is not None:
            sink.close()

    digest = hashlib.sha256()
    occurrences, observed_texts = Counter(), set()
    words = {'verbs': Counter(), 'adjectives': Counter()}
    with Path(source).open('rb') as stream:
        for shard in sorted((bank.path / 'occurrences').glob('*.jsonl.gz')):
            for row in read_rows(shard):
                raw = next(stream, None)
                if raw is None:
                    raise ValueError('ledger has more records than the source')
                digest.update(raw)
                original = json.loads(raw)
                counts['source_records'] += 1
                if (row['line'] != counts['source_records'] or row['idx'] != original.get('idx') or
                        row['text'] != original[bank.manifest['field']]):
                    raise ValueError(f"ledger/source mismatch at record {counts['source_records']}")
                occurrences[row['template_id']] += 1
                signature = bytes.fromhex(fingerprint([row['template_id'], row['text']]))
                if signature not in attested:
                    raise ValueError(f"source record {row['line']} has no executable attested binding")
                observed_texts.add(signature)
                for token in row['tokens']:
                    start, end = token['span']
                    if not 0 <= start < end <= len(row['text']) or row['text'][start:end] != token['text']:
                        raise ValueError('invalid source token offsets')
                    counts['token_spans'] += 1
                    if token['kind'] == 'verb':
                        words['verbs'][token['lemma'], token['text']] += 1
                    if token['pos'] == 'ADJ' or token['tag'] in ('JJ', 'JJR', 'JJS'):
                        words['adjectives'][token['parser_lemma'], token['text']] += 1
            print(f"Verified {counts['source_records']:,} source records", flush=True)
        if stream.read(1):
            raise ValueError('source records are missing from the ledger')
    if digest.hexdigest() != bank.manifest['source_sha256']:
        raise ValueError('source hash mismatch')
    if occurrences != template_counts or observed_texts != attested:
        raise ValueError('template coverage or attested binding counts do not match the source')
    for kind, observed in words.items():
        inventory = Counter({(row['lemma'], form): count for row in read_rows(bank.path / f'{kind}.jsonl.gz')
                             for form, count in row['forms'].items()})
        if inventory != observed:
            raise ValueError(f'{kind} inventory does not account for every occurrence')
        counts[kind + '_occurrences'] = sum(observed.values())
    expected = bank.manifest['counts']
    if (counts['source_records'] != expected['compiled_records'] or counts['templates'] != expected['templates'] or
            counts['token_spans'] != expected['tokens_audited']):
        raise ValueError('manifest counts do not match the data')
    return {'status': 'passed', 'source_sha256': digest.hexdigest(), **counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--rendered', type=Path, help='new JSONL or JSONL.GZ output, one example per template')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = validate_bank(TemplateBank(args.bank), args.source, rendered=args.rendered)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
