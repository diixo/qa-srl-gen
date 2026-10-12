"""Partition DailyDialog and publish explicit, model-authored sentence reviews.

Serialization and validation only: no parser, sentence splitting or inferred
annotations. The model supplies boundaries, native templates and named readings.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from semantic_corpus.template_bank import CorpusTemplate, TemplateBank, fingerprint, read_rows
from semantic_corpus.template_compiler import build_bank
from semantic_corpus.template_registry import register_bank
from semantic_corpus.template_validation import validate_bank
from semantic_corpus.semantic_generator.substitutions import entity_from_json


def digest(path, length=None):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        remaining = length
        while remaining is None or remaining > 0:
            block = stream.read(1024*1024 if remaining is None else min(remaining, 1024*1024))
            if not block:
                if remaining:
                    raise ValueError(f'truncated file: {path}')
                break
            result.update(block)
            if remaining is not None:
                remaining -= len(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def write_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')


def turn_id(row):
    return ':'.join(str(row[k]) for k in ('split', 'dialogue', 'turn'))


def partition(directory, target, number=100):
    """Distribute whole dialogue records evenly; preserve split and row ID."""
    dialogues, sources = [], {}
    for split in ('train', 'val', 'test'):
        path = directory/f'dailydialog-{split}.jsonl'
        sources[split] = digest(path)
        for line, turns in enumerate(read_rows(path), 1):
            if not isinstance(turns, list) or not turns or any(not isinstance(t, str) or not t.strip() for t in turns):
                raise ValueError(f'invalid dialogue: {split}:{line}')
            dialogues.append(dict(split=split, dialogue=line, turns=turns))
    if not 1 <= number <= len(dialogues):
        raise ValueError('part count must be positive and not exceed dialogue count')
    if target.exists():
        manifest = json.loads((target/'manifest.json').read_text(encoding='utf-8'))
        if manifest['sources'] != sources or len(manifest['parts']) != number:
            raise ValueError('existing partition has different inputs or part count')
        for part in manifest['parts']:
            if digest(target/part['file']) != part['sha256']:
                raise ValueError('a partition file changed')
        return manifest
    target.mkdir(parents=True)
    (target/'annotations').mkdir()
    quotient, remainder = divmod(len(dialogues), number)
    cursor, parts = 0, []
    for i in range(number):
        rows = dialogues[cursor:cursor+quotient+(i < remainder)]
        path = target/f'part-{i+1:03d}.jsonl'
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                write_row(stream, row)
        parts.append(dict(file=path.name, dialogues=len(rows), turns=sum(len(r['turns']) for r in rows),
                          first={k:rows[0][k] for k in ('split','dialogue')},
                          last={k:rows[-1][k] for k in ('split','dialogue')}, sha256=digest(path)))
        cursor += len(rows)
    manifest = dict(format='dailydialog-review-parts-v1', sources=sources, dialogues=len(dialogues),
                    turns=sum(p['turns'] for p in parts), parts=parts)
    write_json(target/'manifest.json', manifest)
    return manifest


def check_turn(row, original):
    """Verify supplied boundaries and renderings, without guessing missing text."""
    if row['original'] != original:
        raise ValueError(f'original turn changed: {turn_id(row)}')
    cursor = 0
    if not row['segments']:
        raise ValueError('review must account for the entire turn')
    for number, segment in enumerate(row['segments'], 1):
        origin = segment['origin']
        if any(origin[k] != row[k] for k in ('split', 'dialogue', 'turn')) or origin['segment'] != number:
            raise ValueError('segment identity does not match its turn')
        start, end = origin['span']
        if not 0 <= cursor <= start < end <= len(original) or original[cursor:start].strip():
            raise ValueError('overlapping segments or missing source text')
        if original[start:end] != segment['original']:
            raise ValueError('segment does not match original source span')
        if (segment['text'] != segment['original']) != bool(segment.get('correction')):
            raise ValueError('editorial changes must be explicit')
        cursor = end
        if segment['status'] == 'sentence':
            record = segment['record']
            shape = {k:record[k] for k in ('parts','slots','verbs','constructions','predicate_groups','attributes')}
            if record['template_id'] != fingerprint(shape) or CorpusTemplate(record).render()['text'] != segment['text']:
                raise ValueError('native template does not reproduce reviewed sentence')
        elif segment['status'] not in {'fragment', 'pending'}:
            raise ValueError('unknown review status')
    if original[cursor:].strip():
        raise ValueError('unaccounted turn remainder')
    for named in row.get('names', ()):
        if named['text'] not in original:
            raise ValueError('named reading lacks source evidence')


def reviews(parts):
    manifest = json.loads((parts/'manifest.json').read_text(encoding='utf-8'))
    result, progress = [], []
    for part in manifest['parts']:
        source = parts/part['file']
        if digest(source) != part['sha256']:
            raise ValueError('partition source changed')
        dialogues = {(r['split'],r['dialogue']):r['turns'] for r in read_rows(source)}
        annotation = parts/'annotations'/part['file']
        rows = list(read_rows(annotation)) if annotation.exists() else []
        seen, counts = set(), Counter()
        for row in rows:
            key = turn_id(row)
            if key in seen:
                raise ValueError(f'duplicate turn review: {key}')
            seen.add(key)
            turns = dialogues[row['split'],row['dialogue']]
            if not 1 <= row['turn'] <= len(turns):
                raise ValueError('invalid turn number')
            check_turn(row, turns[row['turn']-1])
            counts.update(s['status'] for s in row['segments'])
        complete = sum(all(f'{s}:{d}:{t}' in seen for t in range(1,len(turns)+1))
                       for (s,d),turns in dialogues.items())
        result.extend(rows)
        progress.append(dict(file=part['file'], dialogues=part['dialogues'], turns=part['turns'],
                             reviewed_dialogues=complete, reviewed_turns=len(rows), statuses=dict(counts),
                             complete=len(rows)==part['turns']))
    return result, dict(dialogues=manifest['dialogues'], turns=manifest['turns'],
                       reviewed_dialogues=sum(p['reviewed_dialogues'] for p in progress),
                       reviewed_turns=len(result), complete=all(p['complete'] for p in progress), parts=progress)


def manual_record(segment, provenance):
    record = deepcopy(segment['record'])
    rendered = CorpusTemplate(record).render()
    tokens = []
    for key, slot in record['slots'].items():
        text = record['defaults'][key]
        lemma = slot.get('lemma', text.lower())
        tokens.append(dict(text=text, span=rendered['spans'][key], lemma=lemma, parser_lemma=lemma,
                           kind=slot['kind'], pos=slot['pos'], tag=slot['tag'], morph=slot.get('morph',''),
                           dep=slot['dependency'], analysis='model_reviewed', slot=key,
                           verb_lemmas=[], adjective_lemmas=[]))
    ledger = dict(**provenance, text=rendered['text'], template_id=record['template_id'],
                  bindings=record['defaults'], tokens=sorted(tokens, key=lambda t:t['span']),
                  attributes=rendered['attributes'], predicate_groups=rendered['predicate_groups'],
                  analysis_status='model_reviewed_sentence', sentence_template=CorpusTemplate(record).text,
                  predicate_templates=[])
    return record, ledger


class LiteralInputs:
    meta = {'name':'explicit model-authored sentence reviews', 'parser':None}
    uses_spacy = False
    note = 'Model-authored syntax; semantic generation remains subject to admission rules.'

    def pipe(self, inputs, **kwargs):
        for text, provenance in inputs:
            yield SimpleNamespace(text=text), provenance


def compile_reviews(rows, output, work, wiktionary):
    work.mkdir(parents=True)
    source = work/'sentences.jsonl'
    accepted = {}
    with source.open('x', encoding='utf-8', newline='\n') as sink:
        for turn in rows:
            for segment in turn['segments']:
                if segment['status'] != 'sentence':
                    continue
                idx = f"{turn_id(turn)}:{segment['origin']['segment']}"
                accepted[idx] = segment
                write_row(sink, dict(idx=idx, example=segment['text'], provenance=dict(
                    origin=segment['origin'], original=segment['original'],
                    correction=segment.get('correction'), review_note=segment.get('note',''))))
    manifest = build_bank(source, output, work/'shards', LiteralInputs(), wiktionary=wiktionary,
                          record_compiler=lambda doc,p,a:manual_record(accepted[p['idx']],p))
    entities, candidates, nominals = {}, {}, {}

    def observe(table, key, fields, text, span, origin):
        row = table.setdefault(key, {**fields, 'count':0, 'examples':[]})
        row['count'] += 1
        if len(row['examples']) < 3:
            row['examples'].append(dict(text=text, span=span, origin=origin))

    for turn in rows:
        origin = {k:turn[k] for k in ('split','dialogue','turn')}
        for named in turn.get('names', ()):
            start = turn['original'].index(named['text'])
            span = [start,start+len(named['text'])]
            if named['label']:
                entity = dict(text=named['text'], labels=[named['label'],'NAMED_ENTITY'], named=True)
                entity_from_json(entity)
                observe(entities,(named['text'],named['label']),dict(entity=entity,review_status='REVIEWED',
                         review_method='model_context_review'),turn['original'],span,origin)
            else:
                observe(candidates,named['text'],dict(text=named['text'],review_status='UNREVIEWED',note=named['note']),
                         turn['original'],span,origin)
        for segment in turn['segments']:
            if segment['status'] != 'sentence':
                continue
            record = segment['record']
            rendered = CorpusTemplate(record).render()
            for key, slot in record['slots'].items():
                if slot['kind'] == 'nominal':
                    text = record['defaults'][key]
                    observe(nominals,(text,slot['tag']),dict(text=text,tag=slot['tag'],review_status='REVIEWED_SURFACE'),
                             segment['text'],rendered['spans'][key],segment['origin'])
    for name, table in (('entities',entities),('entity_candidates',candidates),('nominals',nominals)):
        with gzip.open(output/f'{name}.jsonl.gz','wt',encoding='utf-8') as sink:
            for _, row in sorted(table.items()):
                write_row(sink,row)
        manifest['counts'][name] = len(table)
    with gzip.open(output/'sentence_review.jsonl.gz','wt',encoding='utf-8') as sink:
        for turn in rows:
            write_row(sink,{**turn,'segments':[{k:v for k,v in s.items() if k!='record'} for s in turn['segments']]})
    manifest.pop('source',None)
    manifest['manual_import'] = dict(method='model_reviewed',unit='sentence',complete_corpus=False,
                                    reviewed_turns={turn_id(t):fingerprint(t) for t in rows},
                                    statuses=dict(Counter(s['status'] for t in rows for s in t['segments'])),
                                    spans='Explicit lexical slots, including noun phrases; not a complete token dependency parse.',
                                    source_policy='Self-contained templates; no duplicate source ZIP or JSONL.')
    write_json(output/'manifest.json',manifest)
    validation = validate_bank(TemplateBank(output),source)
    write_json(output/'validation.json',validation)
    return source,manifest,validation


def publish(parts, repo):
    rows,progress = reviews(parts)
    shared = repo/'data/mined-templates'
    old = TemplateBank(shared) if (shared/'manifest.json').exists() else None
    published = {}
    if old:
        for component in old.components():
            published.update(component.manifest.get('manual_import',{}).get('reviewed_turns',{}))
    for row in rows:
        if turn_id(row) in published and published[turn_id(row)] != fingerprint(row):
            raise ValueError('a published review changed; explicit correction is required')
    pending = [r for r in rows if turn_id(r) not in published]
    if pending:
        before = {p:(p.stat().st_size,digest(p)) for p in shared.rglob('*.jsonl.gz')} if old else {}
        with tempfile.TemporaryDirectory(prefix='dailydialog-reviewed-') as temp:
            temp = Path(temp)
            source,manifest,validation = compile_reviews(pending,temp/'bank',temp/'work',repo/'data/wiktionary')
            bank_id = 'dailydialog-reviewed-'+fingerprint(manifest['manual_import']['reviewed_turns'])[:16]
            target = register_bank(temp/'bank',source,repo_root=repo,bank_id=bank_id,archive_source=False)
            component = next(c for c in TemplateBank(target).components() if c.source_id==bank_id)
            if validate_bank(component,source) != validation:
                raise ValueError('published data differs from validated review')
            for path,(size,expected) in before.items():
                if digest(path,size)!=expected:
                    raise ValueError(f'old bank data changed: {path}')
            published.update(manifest['manual_import']['reviewed_turns'])
    progress.update(published_turns=len(published),newly_published_turns=len(pending))
    write_json(parts/'progress.json',progress)
    return progress


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=ROOT)
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    parts=args.repo/'data/dailydialog/review-parts'
    partition(args.repo/'data/dailydialog/filtered',parts)
    if args.publish:
        progress=publish(parts,args.repo)
    else:
        _,progress=reviews(parts)
        write_json(parts/'progress.json',progress)
    print(json.dumps({k:v for k,v in progress.items() if k!='parts'},indent=2))


if __name__=='__main__':
    main()
