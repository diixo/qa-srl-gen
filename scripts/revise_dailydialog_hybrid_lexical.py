"""Recompile dictionary-only predicates from the existing local parse cache.

No inference, network, source rewriting or publication. The old SQLite cache is
attached read-only; the revision stores only changed payloads. All other syntax
and full dialogue contexts are reused unchanged in a new quarantined bank.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import process_dailydialog_hybrid as full
from scripts.benchmark_dailydialog_hybrid import dump, write, risk_flags
from scripts.import_dailydialog_sentences import digest
from semantic_corpus.template_bank import CorpusTemplate, fingerprint, read_rows
from semantic_corpus.template_compiler import LexicalAudit, compile_record
from semantic_corpus.template_validation import _check_rendered

RULE = 'dictionary_readings_are_candidates_v1'


def cached_doc(ledger, vocab):
    """Restore saved tokenization and dependency trees without rerunning NLP."""
    from spacy.tokens import Doc
    text, tokens = ledger['text'], ledger['tokens']
    spaces = []
    for i, token in enumerate(tokens):
        start, end = token['span']
        if token['i'] != i or text[start:end] != token['text']:
            raise ValueError('cached token/source mismatch')
        next_start = tokens[i+1]['span'][0] if i+1 < len(tokens) else len(text)
        gap = text[end:next_start]
        if gap not in ('', ' '):
            raise ValueError('cached whitespace requires an explicit token')
        spaces.append(bool(gap))
    doc = Doc(vocab, words=[t['text'] for t in tokens], spaces=spaces,
        heads=[t['head'] for t in tokens], deps=[t['dep'] for t in tokens],
        pos=[t['pos'] for t in tokens], tags=[t['tag'] for t in tokens],
        lemmas=[t['parser_lemma'] for t in tokens], morphs=[t['morph'] for t in tokens])
    if doc.text != text:
        raise ValueError('cached document restoration changed text')
    return doc


def revise(payload, audit, vocab):
    if payload.get('review_status') != 'UNREVIEWED' or payload.get('method') != 'local_spacy_syntax':
        raise ValueError('Only unreviewed automatic parses may be revised')
    old = payload['ledger']
    indices = {t['i'] for t in old['tokens'] if t['analysis'] == 'imperative_alternative'}
    if not indices:
        raise ValueError('No dictionary-only predicates to revise')
    doc = cached_doc(old, vocab)
    record, ledger = compile_record(doc, {}, audit)
    if indices != {c['token'] for c in ledger['imperative_candidates']}:
        raise ValueError('Reconstructed sentence boundaries changed candidate discovery')
    stable = ('text', 'span', 'pos', 'tag', 'parser_lemma', 'dep', 'head', 'morph',
              'verb_lemmas', 'adjective_lemmas')
    if len(ledger['tokens']) != len(old['tokens']):
        raise ValueError('Token count changed')
    for before, after in zip(old['tokens'], ledger['tokens']):
        if any(before[k] != after[k] for k in stable):
            raise ValueError('Revision changed cached syntax or lexical alternatives')
        if before['i'] in indices:
            if before['lemma'] not in after['verb_lemmas'] or after['kind'] == 'verb':
                raise ValueError('Dictionary reading was lost or still promoted')
        elif before != after:
            raise ValueError('Revision changed an unrelated token')
    rendered = CorpusTemplate(record).render()
    _check_rendered(rendered)
    if rendered['text'] != old['text']:
        raise ValueError('Roundtrip changed')
    result = deepcopy(payload)
    result.update(record=record, ledger=ledger, flags=risk_flags(doc, record, ledger),
        variant=full.variant_signature(record, ledger), syntax_revision=RULE)
    return result


def overlay(source, path):
    db = sqlite3.connect(path.resolve().as_uri(), uri=True)
    db.execute('ATTACH DATABASE ? AS original', (source.resolve().as_uri()+'?mode=ro',))
    db.execute('CREATE TABLE IF NOT EXISTS replacements (key TEXT PRIMARY KEY, payload TEXT NOT NULL, original_sha256 TEXT NOT NULL)')
    db.execute('CREATE TEMP VIEW parses AS SELECT p.key,p.text,COALESCE(r.payload,p.payload) payload '
               'FROM original.parses p LEFT JOIN replacements r ON p.key=r.key')
    db.execute('CREATE TEMP VIEW occurrences AS SELECT rowid,id,key,body FROM original.occurrences')
    return db


def run(source: Path, out: Path, repo=ROOT):
    import spacy
    started = time.monotonic()
    source, out, repo = source.resolve(), out.resolve(), repo.resolve()
    if out.is_relative_to(repo/'data') or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('Use a separate quarantine output outside data/ and the previous run')
    if out.exists():
        raise ValueError('Output already exists; keep previous revisions intact')
    # Frozen dictionary identity is necessary when replaying cached syntax.
    resources = json.loads((source/'local-resources.json').read_text(encoding='utf-8'))
    for name, expected in resources['lexicons'].items():
        if digest(repo/name) != expected:
            raise ValueError('Dictionary changed since the original local pass')
    protected = {str(p.relative_to(repo)): digest(p) for root in (
        repo/'data/mined-templates', repo/'data/dailydialog/filtered',
        repo/'data/dailydialog/review-parts/annotations') for p in root.rglob('*') if p.is_file()}
    out.mkdir(parents=True)
    original_cache_hash = digest(source/'cache.sqlite3')
    dump(out/'plan.json', dict(rule=RULE, source=str(source), source_cache_sha256=original_cache_hash,
        protected=protected, lexical_resources=resources['lexicons'], api_calls=0,
        code={str(p.relative_to(repo)): digest(p) for p in (Path(__file__),
            repo/'semantic_corpus/template_compiler.py', repo/'scripts/benchmark_dailydialog_hybrid.py',
            repo/'scripts/process_dailydialog_hybrid.py')}, published=False))
    db = overlay(source/'cache.sqlite3', out/'revision.sqlite3')
    try:
        vocab = spacy.blank('en').vocab
        audit = LexicalAudit(repo/'data/wiktionary')
        rows = db.execute("SELECT key,payload FROM original.parses "
                          "WHERE json_extract(payload,'$.ledger.analysis_status')='imperative_alternative_added'").fetchall()
        frequencies = dict(db.execute('SELECT key,COUNT(*) FROM occurrences GROUP BY key'))
        counts, pos_counts, token_counts, affected = Counter(), Counter(), Counter(), set()
        representatives = {}
        with gzip.open(out/'changes.jsonl.gz', 'wt', encoding='utf-8') as changes:
            for key, raw in rows:
                before = json.loads(raw)
                after = revise(before, audit, vocab)
                db.execute('INSERT INTO replacements VALUES (?,?,?)',
                           (key, json.dumps(after, ensure_ascii=False), fingerprint(before)))
                affected.add(key)
                weight = frequencies.get(key, 0)
                counts['unique_texts_recompiled'] += 1
                counts['active_unique_texts_recompiled'] += bool(weight)
                counts['unused_cached_inputs_recompiled'] += not weight
                counts['affected_occurrences'] += weight
                tokens = [t for t in before['ledger']['tokens'] if t['analysis']=='imperative_alternative']
                counts['removed_primary_verb_occurrences'] += weight * len(tokens)
                for t in tokens:
                    pos_counts[t['pos']] += weight
                    token_counts[t['text'].lower()] += weight
                    # One example per previous POS/word, selected by stable cache hash.
                    group = (t['pos'], t['text'].lower())
                    if weight and (group not in representatives or key < representatives[group]):
                        representatives[group] = key
                write(changes, dict(key=key, text=before['ledger']['text'], occurrences=weight,
                    previous_template_id=before['record']['template_id'], template_id=after['record']['template_id'],
                    previous_verbs=before['record']['verbs'], verbs=after['record']['verbs'],
                    dictionary_candidates=after['ledger']['imperative_candidates'],
                    rule=RULE, semantic_review='UNREVIEWED'))
        db.commit()
        dump(out/'repair-summary.json', dict(counts=counts, by_previous_pos=pos_counts,
            by_word=dict(token_counts.most_common()), changed_records=len(affected)))
        print(json.dumps(dict(counts)), flush=True)
        for name in ('contexts.jsonl.gz', 'coverage.jsonl.gz'):
            shutil.copyfile(source/name, out/name)
        contexts = {r['id']:r['turns'] for r in read_rows(out/'contexts.jsonl.gz')}
        picks = set()
        # Cover frequent replies and two rare candidates per POS, not just easy examples.
        for word, _ in token_counts.most_common(12):
            keys = [(k,v) for k,v in representatives.items() if k[1]==word]
            picks.update(v for _,v in keys[:1])
        for pos in sorted(pos_counts):
            keys = sorted((key for group,key in representatives.items() if group[0]==pos))
            picks.update(keys[:2])
        with (out/'targeted-sample.jsonl').open('w', encoding='utf-8') as stream:
            for key in sorted(picks):
                identity, body = db.execute('SELECT id,body FROM occurrences WHERE key=? ORDER BY rowid LIMIT 1', (key,)).fetchone()
                occurrence = json.loads(body)
                before = json.loads(db.execute('SELECT payload FROM original.parses WHERE key=?', (key,)).fetchone()[0])
                after = json.loads(db.execute('SELECT payload FROM parses WHERE key=?', (key,)).fetchone()[0])
                write(stream, dict(id=identity, text=occurrence['example'], origin=occurrence['provenance']['origin'],
                    context=contexts[occurrence['provenance']['context_id']],
                    previous_verbs=before['record']['verbs'], verbs=after['record']['verbs'],
                    tokens=[dict(text=t['text'],pos=t['pos'],tag=t['tag']) for t in before['ledger']['tokens']
                            if t['analysis']=='imperative_alternative'],
                    candidates=after['ledger']['imperative_candidates'], semantic_review='UNREVIEWED'))
        totals = full.export(db, out)
        meta = json.loads((source/'bank/manifest.json').read_text(encoding='utf-8')).get('model', {})
        manifest, validation = full.materialize(db, out, meta, repo/'data/wiktionary')
        manifest['hybrid_import']['syntax_revision'] = dict(rule=RULE,
            source_bank_manifest_sha256=digest(source/'bank/manifest.json'),
            recompiled_active_inputs=counts['active_unique_texts_recompiled'])
        dump(out/'bank/manifest.json', manifest)
        old = json.loads((source/'report.json').read_text(encoding='utf-8'))
        if totals['compiled_occurrences'] != old['counts']['compiled_occurrences']:
            raise ValueError('Occurrence coverage changed')
        if validation['token_spans'] != old['validation']['token_spans']:
            raise ValueError('Token coverage changed')
        if digest(source/'cache.sqlite3') != original_cache_hash:
            raise ValueError('Original cache changed')
        if any(not (repo/name).is_file() or digest(repo/name)!=value for name,value in protected.items()):
            raise ValueError('Protected source or bank changed')
        result = dict(status='recompiled_unreviewed', rule=RULE, repairs=dict(counts),
            counts=totals, bank_counts=manifest['counts'], validation=validation,
            original_cache_unchanged=True, protected_files_unchanged=len(protected),
            parser_inference_calls=0, api_calls=0, published=False,
            elapsed_seconds=round(time.monotonic()-started, 2), semantic_review='pending')
        dump(out/'report.json', result)
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'artifacts/dailydialog-hybrid-full')
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/dailydialog-hybrid-lexical-v1')
    args = parser.parse_args()
    run(args.source, args.out)
