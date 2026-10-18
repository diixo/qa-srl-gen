"""Process every unpublished DailyDialog turn locally into a quarantined bank.

No API, publication, or semantic auto-approval. The cache shares syntax, never
the meaning of identical replies in different dialogues. Restart with --resume.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_dailydialog_hybrid import dump, load_reference, risk_flags, write, features
from scripts.import_dailydialog_sentences import digest, manual_record, turn_id
from semantic_corpus.template_bank import CorpusTemplate, TemplateBank, fingerprint, read_rows
from semantic_corpus.template_compiler import LexicalAudit, build_bank, compile_record
from semantic_corpus.template_validation import _check_rendered, validate_bank

VERSION = 'dailydialog-hybrid-full-v1'
APOSTROPHES = str.maketrans({c: "'" for c in '\u2019\u2018\u201b\u02bc\u02bb\uff07`'})
ABBREVIATIONS = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'vs', 'etc', 'e.g', 'i.e', 'a.m', 'p.m', 'am', 'pm'}


def split_sentences(text):
    """Conservative punctuation boundaries, independent of dependency clauses.

    Ambiguous ellipses/quotes stay intact and are queued for boundary review.
    No source character other than boundary whitespace is discarded.
    """
    cuts, flags, quote, i = [], set(), False, 0
    while i < len(text):
        char = text[i]
        if char in '"\u201c\u201d':
            quote = not quote
        if char not in '.!?':
            i += 1
            continue
        end = i + 1
        while end < len(text) and text[end] in '.!?':
            end += 1
        if end - i > 1 and '.' in text[i:end]:
            flags.add('ambiguous_ellipsis_boundary')
            i = end
            continue
        preceding = re.search(r'([\w.]+)$', text[:i])
        word = preceding.group(1) if preceding else ''
        if char == '.' and (word.lower() in ABBREVIATIONS or
                (word.lower() == 'no' and re.match(r'\s*\d', text[end:])) or
                (len(word) == 1 and word.isupper()) or
                (i and end < len(text) and text[i-1].isdigit() and text[end].isdigit())):
            flags.add('abbreviation_boundary')
            i = end
            continue
        if quote:
            flags.add('quoted_boundary')
            i = end
            continue
        # Sentence-final closing brackets belong to the preceding sentence,
        # including standalone stage directions: (A few minutes later.).
        while end < len(text) and text[end] in ')]}':
            end += 1
        if end < len(text) and not text[end].isspace():
            flags.add('unspaced_sentence_boundary')
        cuts.append(end)
        i = end
    if quote:
        flags.add('unbalanced_quote')
    cuts.append(len(text))
    spans, start = [], 0
    for end in cuts:
        a, b = start, end
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b-1].isspace():
            b -= 1
        if a < b:
            spans.append([a, b])
        start = end
    check_spans(text, spans)
    return spans, sorted(flags)


def check_spans(text, spans):
    cursor = 0
    for start, end in spans:
        if not cursor <= start < end <= len(text) or text[cursor:start].strip():
            raise ValueError('incomplete or overlapping segmentation')
        cursor = end
    if not spans or text[cursor:].strip():
        raise ValueError('unaccounted source text')


def reference_groups(references):
    """Reuse only a whole identical conversation with a consistent reference."""
    dialogues = defaultdict(list)
    for row in references:
        dialogues[f"{row['split']}:{row['dialogue']}"].append(row)
    by_text = defaultdict(list)
    for identity, rows in dialogues.items():
        rows.sort(key=lambda r: r['turn'])
        by_text[fingerprint([r['original'] for r in rows])].append((identity, rows))
    reusable, conflicts = {}, set()
    for key, variants in by_text.items():
        signatures = set()
        for identity, rows in variants:
            signatures.add(fingerprint([[dict(status=s['status'], text=s['text'],
                span=s['origin']['span'], syntax=features(s['record']) if 'record' in s else None,
                meaning={k:v for k,v in s.items() if k not in {'record','origin','text','original','status'}})
                for s in r['segments']] for r in rows]))
        # Previously adjudicated faulty adjective target: never propagate it.
        # Do not rewrite approved records (and their slot offsets) implicitly.
        # Reparse a duplicate with normalized punctuation if its reference is
        # noncanonical; the original reviewed conversation remains untouched.
        noncanonical = any(s['text'] != s['text'].translate(APOSTROPHES)
                           for _, rows in variants for r in rows for s in r['segments'])
        if len(signatures) > 1 or any(i == 'train:6' for i, _ in variants) or noncanonical:
            conflicts.add(key)
        else:
            reusable[key] = variants[0]
    return dialogues, reusable, conflicts


def variant_signature(record, ledger):
    """Preserve predicate forms, negation, auxiliary and clause attachments."""
    tokens = ledger['tokens']
    return dict(predicates=[{k:t.get(k) for k in ('lemma','tag','dep','head','morph')}
                            for t in tokens if t['kind'] == 'verb'],
                operators=[{k:t.get(k) for k in ('text','lemma','dep','head')}
                           for t in tokens if t.get('dep') in {'neg','mark','prt','aux','auxpass'}
                           or t['text'].lower() in {'not', 'never', 'no', "n't", 'if', 'unless'}],
                constructions=record['predicate_groups'], question='?' in ledger['text'])


def connect(path):
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE IF NOT EXISTS parses (key TEXT PRIMARY KEY, text TEXT NOT NULL, payload TEXT)')
    db.execute('CREATE INDEX IF NOT EXISTS pending_parse_keys ON parses(key) WHERE payload IS NULL')
    db.execute('CREATE TABLE IF NOT EXISTS occurrences (id TEXT PRIMARY KEY, key TEXT, body TEXT NOT NULL)')
    db.execute('CREATE INDEX IF NOT EXISTS occurrence_key ON occurrences(key)')
    return db


def reference_payload(segment):
    record, ledger = manual_record(segment, {})
    return dict(record=record, ledger=ledger, flags=['exact_reviewed_dialogue_reuse'],
                variant=variant_signature(record, ledger), entities=[],
                method='reviewed_whole_dialogue_reuse', review_status='REVIEWED_REUSE')


def prepare(repo, out, db, references):
    known, reusable, conflicts = reference_groups(references)
    seen, counts = {}, Counter()
    with gzip.open(out/'contexts.jsonl.gz', 'wt', encoding='utf-8') as contexts, \
            gzip.open(out/'coverage.jsonl.gz', 'wt', encoding='utf-8') as coverage:
        for split in ('train', 'val', 'test'):
            for number, turns in enumerate(read_rows(repo/f'data/dailydialog/filtered/dailydialog-{split}.jsonl'), 1):
                if not isinstance(turns, list) or not turns or any(not isinstance(t,str) or not t.strip() for t in turns):
                    raise ValueError('invalid dialogue input')
                identity = f'{split}:{number}'
                key = fingerprint(turns)
                counts['source_dialogues'] += 1
                counts['source_turns'] += len(turns)
                if identity in known:
                    if turns != [r['original'] for r in known[identity]]:
                        raise ValueError('published reference differs from input: '+identity)
                    counts['preserved_reviewed_dialogues'] += 1
                    counts['preserved_reviewed_turns'] += len(turns)
                    continue
                counts['remaining_dialogues'] += 1
                reused = reusable.get(key)
                duplicate = seen.get(key)
                counts['reviewed_dialogue_reuses'] += bool(reused)
                counts['unreviewed_dialogue_duplicates'] += bool(duplicate and not reused)
                counts['reference_conflict_dialogues'] += key in conflicts
                write(contexts, dict(id=identity, turns=turns, sha256=key,
                    duplicate_of=reused[0] if reused else duplicate))
                seen.setdefault(key, identity)
                for turn, original in enumerate(turns, 1):
                    counts['remaining_turns'] += 1
                    tid = f'{identity}:{turn}'
                    reviewed = reused[1][turn-1] if reused else None
                    if reviewed:
                        spans = [s['origin']['span'] for s in reviewed['segments']]
                        boundary_flags = []
                    else:
                        spans, boundary_flags = split_sentences(original)
                    check_spans(original, spans)
                    segments = []
                    for index, span in enumerate(spans, 1):
                        raw = original[slice(*span)]
                        ref = reviewed['segments'][index-1] if reviewed else None
                        text = ref['text'] if ref else raw.translate(APOSTROPHES)
                        correction = ref.get('correction') if ref else ('normalize_apostrophes' if text != raw else None)
                        payload = reference_payload(ref) if ref and 'record' in ref else None
                        cache_key = fingerprint(['reviewed', payload]) if payload else fingerprint(['syntax_only',text])
                        db.execute('INSERT OR IGNORE INTO parses VALUES (?,?,?)',
                                   (cache_key,text,json.dumps(payload) if payload else None))
                        sid = f'{tid}:{index}'
                        provenance = dict(origin=dict(split=split, dialogue=number, turn=turn, segment=index, span=span),
                            original=raw, correction=correction, context_id=identity, boundary_flags=boundary_flags,
                            cache_key=cache_key, semantic_review='UNREVIEWED',
                            reused_from=f'{reused[0]}:{turn}:{index}' if reused else None,
                            reviewed_fragment=ref if ref and 'record' not in ref else None,
                            exact_dialogue_duplicate_of=duplicate,
                            reference_conflict=key in conflicts)
                        body = dict(idx=sid, example=text, provenance=provenance)
                        db.execute('INSERT INTO occurrences VALUES (?,?,?)', (sid,cache_key,json.dumps(body)))
                        segments.append(dict(id=sid, span=span, text=text, correction=correction,
                                             processing='reviewed_syntax_reuse' if payload else 'local_syntax_pending'))
                        counts['segments'] += 1
                    write(coverage, dict(id=tid, context_id=identity, original=original,
                        boundary_flags=boundary_flags, segments=segments))
                if counts['remaining_dialogues'] % 500 == 0:
                    db.commit()
        db.commit()
    if counts['preserved_reviewed_dialogues'] != len(known):
        raise ValueError('published references absent from source')
    counts['unique_parse_inputs'] = db.execute('SELECT COUNT(DISTINCT key) FROM occurrences').fetchone()[0]
    dump(out/'coverage-summary.json', dict(counts))
    return dict(counts)


def parse_all(db, model, wiktionary, out):
    import spacy
    nlp = spacy.load(model)
    if nlp.lang != 'en' or not {'parser','lemmatizer'} <= set(nlp.pipe_names):
        raise ValueError('English parser and lemmatizer required')
    audit = LexicalAudit(wiktionary)
    count = 0
    while True:
        batch = db.execute('SELECT key,text FROM parses WHERE payload IS NULL ORDER BY key LIMIT 256').fetchall()
        if not batch:
            break
        for doc, key in nlp.pipe(((text,key) for key,text in batch), as_tuples=True, batch_size=64):
            try:
                record, ledger = compile_record(doc, {}, audit)
                rendered = CorpusTemplate(record).render()
                _check_rendered(rendered)
                if rendered['text'] != doc.text:
                    raise ValueError('roundtrip mismatch')
                payload = dict(record=record, ledger=ledger, flags=risk_flags(doc,record,ledger),
                    variant=variant_signature(record,ledger), method='local_spacy_syntax', review_status='UNREVIEWED',
                    entities=[dict(text=e.text,label=e.label_,span=[e.start_char,e.end_char],review_status='UNREVIEWED') for e in doc.ents])
            except (ValueError, KeyError, AssertionError) as error:
                payload = dict(error=f'{type(error).__name__}: {error}', text=doc.text,
                               method='local_spacy_syntax', review_status='UNREVIEWED')
            db.execute('UPDATE parses SET payload=? WHERE key=?', (json.dumps(payload),key))
            count += 1
        db.commit()
        if count % 2048 == 0:
            print(f'Parsed {count:,} additional unique texts', flush=True)
            dump(out/'progress.json',dict(phase='parsing',completed=db.execute('SELECT COUNT(*) FROM parses WHERE payload IS NOT NULL').fetchone()[0]))
    return nlp.meta


def export(db, out, *, observer=None):
    counts, groups, families, entities = Counter(), {}, {}, {}
    with (out/'sentences.jsonl').open('w',encoding='utf-8',newline='\n') as source, \
            gzip.open(out/'outcomes.jsonl.gz','wt',encoding='utf-8') as outcomes:
        for identity, body, payload in db.execute('SELECT o.id,o.body,p.payload FROM occurrences o JOIN parses p ON o.key=p.key ORDER BY o.rowid'):
            row, parsed = json.loads(body), json.loads(payload)
            provenance = row['provenance']
            if parsed.get('syntax_revision'):
                provenance['syntax_revision'] = parsed['syntax_revision']
            if 'error' in parsed:
                counts['failed_occurrences'] += 1
                write(outcomes,dict(id=identity,status='parse_failed',error=parsed['error'],provenance=provenance))
                continue
            flags = sorted(set(parsed['flags']+provenance['boundary_flags']))
            variant = dict(syntax=parsed['variant'],flags=flags)
            gid = fingerprint(variant)
            provenance.update(review_status=parsed['review_status'],method=parsed['method'],group_id=gid,
                              candidate_kind='fragment_or_nonverbal' if 'no_verb_or_fragment' in flags else 'sentence',
                              semantic_review='UNREVIEWED')
            context_review = provenance.get('context_review')
            if context_review:
                provenance['candidate_kind'] = context_review['unit']
                counts['context_reviewed_occurrences'] += 1
                counts['context_syntax_corrections'] += context_review['syntax_corrected']
            if observer is not None:
                observer(identity, row, parsed)
            write(source,row)
            write(outcomes,dict(id=identity,status='compiled',group_id=gid,flags=flags,
                               review_status=parsed['review_status'],semantic_review='UNREVIEWED',
                               **({'context_review':context_review} if context_review else {})))
            counts['compiled_occurrences'] += 1
            counts['reviewed_syntax_reuses'] += parsed['review_status'] == 'REVIEWED_REUSE'
            counts['fragment_or_nonverbal_candidates'] += provenance['candidate_kind'] != 'sentence'
            group = groups.setdefault(gid,dict(id=gid,variant=variant,count=0,representatives=[],review_status='UNREVIEWED'))
            group['count'] += 1
            if len(group['representatives']) < 2:
                group['representatives'].append(identity)
            case = dict(id=identity,text=row['example'],origin=provenance['origin'],group_id=gid,
                        syntax=features(parsed['record']), flags=flags,template=CorpusTemplate(parsed['record']).text)
            # Deterministic bounded samples per overlapping risk family, incl. ordinary.
            for flag in flags or ['ordinary_unflagged']:
                family = families.setdefault(flag,dict(count=0,sample=[]))
                family['count'] += 1
                family['sample'].append((fingerprint(['full-hybrid-v1',identity]),case))
                family['sample'] = sorted(family['sample'],key=lambda x:x[0])[:4]
            for entity in parsed['entities']:
                key = (entity['text'], entity['label'])
                entry = entities.setdefault(key,dict(text=key[0],label=key[1],count=0,examples=[],review_status='UNREVIEWED'))
                entry['count'] += 1
                if len(entry['examples']) < 3:
                    entry['examples'].append(dict(id=identity,span=entity['span']))
    with gzip.open(out/'groups.jsonl.gz','wt',encoding='utf-8') as stream:
        for group in sorted(groups.values(),key=lambda g:(-g['count'],g['id'])):
            write(stream,group)
    with gzip.open(out/'entity-candidates.jsonl.gz','wt',encoding='utf-8') as stream:
        for _,row in sorted(entities.items()):
            write(stream,row)
    selected = {}
    for name,family in sorted(families.items()):
        for _,case in family['sample']:
            selected.setdefault(case['id'],dict(**case,selection_families=[]))['selection_families'].append(name)
    contexts = {r['id']:r['turns'] for r in read_rows(out/'contexts.jsonl.gz')}
    with (out/'review-sample.jsonl').open('w',encoding='utf-8') as stream:
        for case in selected.values():
            origin=case['origin']
            case['context']=contexts[f"{origin['split']}:{origin['dialogue']}"]
            write(stream,case)
    dump(out/'families.json',{k:dict(count=v['count'],sample=[c['id'] for _,c in v['sample']]) for k,v in sorted(families.items())})
    counts.update(groups=len(groups),entity_candidates=len(entities),sample_cases=len(selected))
    return dict(counts)


class CachedInputs:
    uses_spacy = False
    note = 'QUARANTINE: automatic sentence/fragment candidates, not approved semantic annotations.'

    def __init__(self, meta):
        self.meta = meta

    def pipe(self, inputs, **kwargs):
        for text,provenance in inputs:
            yield SimpleNamespace(text=text),provenance


def materialize(db, out, meta, wiktionary):
    def cached(doc, provenance, audit):
        payload = json.loads(db.execute('SELECT payload FROM parses WHERE key=?',(provenance['cache_key'],)).fetchone()[0])
        if payload['ledger']['text'] != doc.text:
            raise ValueError('cache/source mismatch')
        record = payload['record']
        record['review_status'] = 'UNREVIEWED'
        return record,dict(**payload['ledger'],**provenance)
    bank = out/'bank'
    if bank.exists():
        raise ValueError('bank exists: inspect it; use a new output for rebuilding')
    manifest = build_bank(out/'sentences.jsonl',bank,out/'shards',CachedInputs(meta),
                          wiktionary=wiktionary,record_compiler=cached)
    for name in ('contexts.jsonl.gz','coverage.jsonl.gz','outcomes.jsonl.gz','groups.jsonl.gz','entity-candidates.jsonl.gz'):
        shutil.copyfile(out/name,bank/name)
    manifest['hybrid_import'] = dict(method=VERSION,review_status='UNREVIEWED',published=False,
        context_file='contexts.jsonl.gz',coverage_file='coverage.jsonl.gz',semantic_approval=False,
        note='Family review does not approve unseen instances; all original contexts are embedded.')
    dump(bank/'manifest.json',manifest)
    validation=validate_bank(TemplateBank(bank),out/'sentences.jsonl')
    dump(bank/'validation.json',validation)
    return manifest,validation


def run(repo, out, model, resume=False):
    started=time.monotonic()
    if out.resolve().is_relative_to((repo/'data').resolve()):
        raise ValueError('quarantine output must be outside data/')
    source_paths=[repo/f'data/dailydialog/filtered/dailydialog-{s}.jsonl' for s in ('train','val','test')]
    protected_paths=source_paths+sorted(p for p in (repo/'data/mined-templates').rglob('*') if p.is_file())
    protected={str(p.relative_to(repo)):digest(p) for p in protected_paths}
    code_paths=[Path(__file__),repo/'scripts/benchmark_dailydialog_hybrid.py',repo/'scripts/import_dailydialog_sentences.py',
                repo/'semantic_corpus/template_compiler.py',repo/'semantic_corpus/syntax_mining.py']
    code={str(p.relative_to(repo)):digest(p) for p in code_paths}
    if resume:
        plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
        if plan['protected'] != protected or plan['code'] != code or plan['model'] != str(model):
            raise ValueError('resume inputs or code changed')
    else:
        out.mkdir(parents=True,exist_ok=False)
        dump(out/'plan.json',dict(format=VERSION,protected=protected,code=code,model=str(model),
            revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
            api_calls=0,publish=False,policy='Preserve reviewed baseline; all new semantic interpretations remain UNREVIEWED.'))
    db=connect(out/'cache.sqlite3')
    if not (out/'coverage-summary.json').exists():
        # A partial preparation has no reusable semantic state. Rebuild indices only.
        db.execute('DELETE FROM occurrences')
        counts=prepare(repo,out,db,load_reference(repo/'data/mined-templates'))
    else:
        counts=json.loads((out/'coverage-summary.json').read_text())
    print(json.dumps(counts),flush=True)
    meta=parse_all(db,model,repo/'data/wiktionary',out)
    counts.update(export(db,out))
    if counts['segments'] != counts['compiled_occurrences']+counts.get('failed_occurrences',0):
        raise ValueError('incomplete processing outcomes')
    dump(out/'processing-summary.json',counts)
    manifest,validation=materialize(db,out,meta,repo/'data/wiktionary')
    if any(digest(repo/name)!=value for name,value in protected.items()):
        raise ValueError('protected sources or working bank changed')
    report=dict(status='processed_unreviewed',counts=counts,native_bank_counts=manifest['counts'],
        validation=validation,protected_files_unchanged=len(protected),api_calls=0,published=False,
        elapsed_seconds=round(time.monotonic()-started,2),semantic_review='pending_in_review-sample.jsonl')
    dump(out/'report.json',report)
    db.close()
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/dailydialog-hybrid-full')
    parser.add_argument('--model',default=str(ROOT/'artifacts/models/spacy-en-3.8.0/en_core_web_sm/en_core_web_sm-3.8.0'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    run(ROOT,args.out.resolve(),args.model,args.resume)
