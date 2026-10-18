"""Rebuild a quarantined candidate after explicitly reviewed source corrections.

Old banks and decisions remain immutable. Each replacement carries its complete
preimage, including superseded reviews; joined segments retain their old IDs.
This migration currently accepts only equal-length Lt's -> It's corrections.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import process_dailydialog_hybrid as full
from scripts.audit_dailydialog_hybrid_bank import open_overlay, WholeBankAudit
from scripts.benchmark_dailydialog_hybrid import dump, write, risk_flags
from scripts.import_dailydialog_sentences import digest
from scripts.revise_dailydialog_hybrid_context import SHAPE
from scripts.revise_dailydialog_hybrid_lexical import cached_doc
from semantic_corpus.template_bank import CorpusTemplate, fingerprint, read_rows
from semantic_corpus.template_compiler import LexicalAudit, compile_record
from semantic_corpus.template_validation import _check_rendered

VERSION = 'reviewed_source_rebase_v1'


def rewrite(inputs, old_context, new_context, review, audit, vocab):
    """Apply the explicitly supplied token decisions, never infer their meaning."""
    if review['method'] != 'direct_assistant_source_review':
        raise ValueError('explicit source review required')
    if [r['body']['idx'] for r in inputs] != review['old_ids'] or review['id'] != review['old_ids'][0]:
        raise ValueError('source identities mismatch')
    if [fingerprint(r) for r in inputs] != review['preimages']:
        raise ValueError('stale parse preimage')
    if fingerprint(old_context) != review['previous_context_sha256'] or fingerprint(new_context) != review['context_sha256']:
        raise ValueError('stale dialogue preimage')
    body = deepcopy(inputs[0]['body'])
    origin = body['provenance']['origin']
    start, end = review['span']
    old_text = old_context[origin['turn']-1][start:end]
    text = new_context[origin['turn']-1][start:end]
    if text != review['text'] or text != old_text.replace("Lt's", "It's") or text == old_text:
        raise ValueError('only the reviewed equal-length spelling correction is allowed')
    if text != text.translate(full.APOSTROPHES):
        raise ValueError('noncanonical apostrophe')
    if start != origin['span'][0] or end != inputs[-1]['body']['provenance']['origin']['span'][1]:
        raise ValueError('replacement boundary mismatch')
    ledger = dict(text=text,tokens=[])
    cursor = start
    for item in inputs:
        row, payload = item['body'],item['payload']
        o = row['provenance']['origin']
        if any(o[k] != origin[k] for k in ('split','dialogue','turn')):
            raise ValueError('a join cannot cross dialogue turns')
        if not cursor <= o['span'][0] or old_context[o['turn']-1][cursor:o['span'][0]].strip():
            raise ValueError('overlap or omitted source characters')
        if old_context[o['turn']-1][slice(*o['span'])] != row['example']:
            raise ValueError('unexpected prior normalization')
        if payload.get('method') != 'local_spacy_syntax':
            raise ValueError('reviewed reference cannot be rebased implicitly')
        offset, shift = len(ledger['tokens']),o['span'][0]-start
        for token in deepcopy(payload['ledger']['tokens']):
            token['i'] += offset
            token['head'] += offset
            token['span'] = [v+shift for v in token['span']]
            ledger['tokens'].append(token)
        cursor = o['span'][1]
    edits = {}
    for edit in review['token_edits']:
        index, fields = edit['token'],edit['set']
        if (type(index) is not int or index in edits or not 0 <= index < len(ledger['tokens']) or
                not fields or set(fields)-{'text','parser_lemma','pos','tag','morph','dep','head'}):
            raise ValueError('invalid token edit')
        edits[index] = fields
        ledger['tokens'][index].update(fields)
    cursor = 0
    for i,token in enumerate(ledger['tokens']):
        a,b = token['span']
        if not cursor <= a < b <= len(text) or text[cursor:a].strip() or text[a:b] != token['text']:
            raise ValueError('token text or coverage mismatch')
        cursor = b
        seen, current = set(),i
        while True:
            if current in seen or not 0 <= current < len(ledger['tokens']):
                raise ValueError('cyclic or invalid dependency')
            seen.add(current)
            parent = ledger['tokens'][current]['head']
            if parent == current:
                if ledger['tokens'][current]['dep'] != 'ROOT':
                    raise ValueError('self head is not ROOT')
                break
            current = parent
    if text[cursor:].strip() or sum(t['dep']=='ROOT' for t in ledger['tokens']) != 1:
        raise ValueError('incomplete tokens or multiple sentence roots')
    doc = cached_doc(ledger,vocab)
    record, compiled = compile_record(doc,{},audit)
    rendered = CorpusTemplate(record).render()
    if len(compiled['tokens']) != len(ledger['tokens']):
        raise ValueError('tokenization changed')
    for i,(old,new) in enumerate(zip(ledger['tokens'],compiled['tokens'])):
        if old['text'] != new['text'] or old['span'] != new['span']:
            raise ValueError('token boundaries changed')
        if i in edits:
            new['analysis'] = 'model_source_review'
            for key,span in rendered['spans'].items():
                if span == new['span']:
                    record['slots'][key]['analysis'] = 'model_source_review'
    for adjective,target in review.get('attribute_targets',{}).items():
        akeys = [k for k,v in record['defaults'].items() if v == adjective and record['slots'][k]['kind']=='adjective']
        targets = [k for k,v in record['defaults'].items() if v == target and record['slots'][k]['kind']=='nominal']
        if len(akeys) != 1 or len(targets) != 1:
            raise ValueError('ambiguous explicit attribute target')
        for items in (record['attributes'],compiled['attributes']):
            next(a for a in items if a['slot']==akeys[0])['target_slot'] = targets[0]
    record['template_id'] = fingerprint({k:record[k] for k in SHAPE})
    compiled['template_id'] = record['template_id']
    # Preimages contain all superseded automatic repairs and contextual decisions.
    revision = dict(version=VERSION,old_ids=review['old_ids'],preimages=deepcopy(inputs),
        previous_context_sha256=review['previous_context_sha256'],context_sha256=review['context_sha256'],
        review=deepcopy(review),semantic_approval=False)
    compiled['source_revision'] = revision
    key = fingerprint([VERSION,review])
    provenance = body['provenance']
    if 'context_review' in provenance:
        provenance.setdefault('context_review_history',[]).append(provenance.pop('context_review'))
    provenance.update(original=text,previous_original=old_text,cache_key=key,
        previous_cache_key=provenance['cache_key'],correction='source_spelling_and_reviewed_syntax',
        context_review=dict(rule=VERSION,status='REVIEWED_OCCURRENCE',method=review['method'],
            unit=review['unit'],meaning=review['meaning'],evidence_turns=review['evidence_turns'],note=review['note'],
            syntax_corrected=True,source_sha256=fingerprint(text),context_sha256=fingerprint(new_context),
            template_semantics_approved=False,unseen_group_members_approved=False))
    origin['span'] = [start,end]
    body['example'] = text
    flags = sorted(set(risk_flags(doc,record,compiled)) | {'context_reviewed_occurrence'})
    if review['unit']=='pending': flags.append('context_pending')
    # Keep NER evidence at surviving spans, discarding any entity overlapping Lt.
    entities=[]
    for item in inputs:
        shift=item['body']['provenance']['origin']['span'][0]-start
        for entity in item['payload']['entities']:
            entity=deepcopy(entity);a,b=entity['span'];entity['span']=[a+shift,b+shift]
            if old_text[a+shift:b+shift] == text[a+shift:b+shift]: entities.append(entity)
    result=dict(record=record,ledger=compiled,flags=flags,variant=full.variant_signature(record,compiled),
        entities=entities,method='local_spacy_syntax',review_status='UNREVIEWED',syntax_revision=VERSION)
    rendered=CorpusTemplate(record).render();_check_rendered(rendered)
    if rendered['text']!=text: raise ValueError('roundtrip mismatch')
    return body,result


def refresh_contexts(source,repo,reviews,out):
    sources={s:list(read_rows(repo/f'data/dailydialog/filtered/dailydialog-{s}.jsonl')) for s in ('train','val','test')}
    # Previously published 510 dialogues are protected independently of the remainder.
    refs=full.load_reference(repo/'data/mined-templates')
    for row in refs:
        if sources[row['split']][row['dialogue']-1][row['turn']-1] != row['original']:
            raise ValueError('published reference source changed')
    contexts=list(read_rows(source/'contexts.jsonl.gz'))
    old={r['id']:r['turns'] for r in contexts}
    expected_turns={r['id'].rsplit(':',1)[0] for r in reviews}
    changed=set();new={}
    for row in contexts:
        split,number=row['id'].split(':');turns=sources[split][int(number)-1]
        if len(turns)!=len(row['turns']): raise ValueError('turn count changed')
        for i,(before,after) in enumerate(zip(row['turns'],turns),1):
            if before!=after:
                if after!=before.replace("Lt's","It's"): raise ValueError('unexpected source edit')
                changed.add(f"{row['id']}:{i}")
        if turns!=row['turns']:
            if row.get('duplicate_of'): raise ValueError('changed duplicate needs explicit review')
            row.update(turns=turns,sha256=fingerprint(turns))
        new[row['id']]=turns
    if changed!=expected_turns: raise ValueError('source changes and reviewed turns disagree')
    if len(refs)+sum(map(len,new.values())) != sum(len(d) for rows in sources.values() for d in rows):
        raise ValueError('source coverage changed')
    with gzip.open(out/'contexts.jsonl.gz','wt',encoding='utf-8') as stream:
        for row in contexts: write(stream,row)
    targets={r['id']:r for r in reviews};removed={i for r in reviews for i in r['old_ids'][1:]}
    with gzip.open(out/'coverage.jsonl.gz','wt',encoding='utf-8') as stream:
        for row in read_rows(source/'coverage.jsonl.gz'):
            cid,turn=row['id'].rsplit(':',1);row['original']=new[cid][int(turn)-1]
            segments=[]
            for s in row['segments']:
                if s['id'] in removed: continue
                if s['id'] in targets:
                    r=targets[s['id']]
                    s.update(span=r['span'],text=r['text'],correction='source_spelling_and_reviewed_syntax',
                             processing='reviewed_source_revision',previous_segment_ids=r['old_ids'])
                segments.append(s)
            row['segments']=segments
            full.check_spans(row['original'],[s['span'] for s in segments])
            write(stream,row)
    return old,new


def verify_rebase(source,out,reviews):
    def rows(root):
        for path in sorted((root/'bank/occurrences').glob('*.jsonl.gz')): yield from read_rows(path)
    targets={r['id']:r for r in reviews};retired={i for r in reviews for i in r['old_ids'][1:]}
    counts=Counter();previous={}
    def surviving():
        for row in rows(source):
            counts['old_segments']+=1;counts['old_tokens']+=len(row['tokens'])
            if row['idx'] in retired:
                previous[row['idx']]=row
                counts['retired_segments']+=1
                continue
            yield row
    for before,after in zip_longest(surviving(),rows(out)):
        if before is None or after is None or before['idx']!=after['idx']: raise ValueError('coverage/order changed')
        counts['new_segments']+=1;counts['new_tokens']+=len(after['tokens'])
        if after['line']!=before['line']-counts['retired_segments']:
            raise ValueError('unexpected physical output line')
        if before['idx'] not in targets:
            if {k:v for k,v in before.items() if k!='line'}!={k:v for k,v in after.items() if k!='line'}:
                raise ValueError('unaddressed occurrence changed')
            counts['unchanged_analyses']+=1
            counts['renumbered_output_lines']+=before['line']!=after['line']
        else:
            revision=after['source_revision'];review=targets[before['idx']]
            if revision['review']!=review or revision['semantic_approval']: raise ValueError('review changed')
            if [fingerprint(r) for r in revision['preimages']]!=review['preimages']: raise ValueError('history lost')
            if after['text']!=review['text'] or after['origin']['span']!=review['span']: raise ValueError('incorrect source')
            counts['revised_occurrences']+=1
    if set(previous)!=retired or counts['old_tokens']!=counts['new_tokens']: raise ValueError('retired records or tokens lost')
    result=dict(status='passed',counts=dict(counts),retired_ids=sorted(retired),history_retained=True,published=False)
    dump(out/'source-conservation-audit.json',result)
    return result


def finish(source,out,reviews_path,db,counts,validation,elapsed):
    reviews=list(read_rows(reviews_path));plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    old_manifest=json.loads((source/'bank/manifest.json').read_text(encoding='utf-8'))
    manifest=json.loads((out/'bank/manifest.json').read_text(encoding='utf-8'))
    coverage=json.loads((out/'audit-summary.json').read_text(encoding='utf-8'))
    latest=list(read_rows(out/'reviews.jsonl'))
    conservation=verify_rebase(source,out,reviews)
    for path,value in {**plan['protected'],**plan['frozen']}.items():
        if digest(Path(path))!=value: raise ValueError('protected input changed: '+path)
    changed={r['id'] for r in reviews}
    if [r for r in db.execute('SELECT * FROM corrections ORDER BY id') if r[0] not in changed] != [r for r in db.execute('SELECT * FROM previous.corrections ORDER BY id') if r[0] not in changed]:
        raise ValueError('previous context decisions changed')
    if list(db.execute('SELECT * FROM automatic_repairs ORDER BY key'))!=list(db.execute('SELECT * FROM previous.automatic_repairs ORDER BY key')):
        raise ValueError('automatic repair history changed')
    review_dir=out/'bank/review';shutil.copytree(source/'bank/review',review_dir)
    shutil.copyfile(source/'reviews.jsonl',review_dir/'context-reviews-before-source-rebase.jsonl')
    shutil.copyfile(out/'reviews.jsonl',review_dir/'context-reviews.jsonl')
    shutil.copyfile(reviews_path,review_dir/'source-reviews.jsonl')
    for name in ('syntax-audit.jsonl.gz','review-queue.jsonl.gz','audit-summary.json','apostrophe-audit.json','source-conservation-audit.json'):
        shutil.copyfile(out/name,review_dir/name)
    manifest['hybrid_import']={**old_manifest['hybrid_import'],**manifest['hybrid_import']}
    manifest['hybrid_import']['source_revision']=dict(version=VERSION,review_file='review/source-reviews.jsonl',
        changed_segments=len(reviews),joined_segments=len(conservation['retired_ids']),latest_context_reviews=len(latest))
    manifest['hybrid_import']['context_revision']=dict(review_file='review/context-reviews.jsonl',occurrences=len(latest),
        syntax_corrections=sum(bool(r['token_edits']) for r in latest))
    manifest['hybrid_import'].pop('source_apostrophe_audit',None)
    dump(out/'bank/manifest.json',manifest)
    report=dict(status='source_rebase_complete',counts=counts,validation=validation,conservation=conservation,
        dialogues=coverage['dialogues'],turns=coverage['turns'],protected_files_unchanged=len(plan['protected']),
        previous_bank_unchanged=True,api_calls=0,parser_inference_calls=0,published=False,
        elapsed_seconds=round(elapsed,2) if elapsed is not None else None)
    dump(out/'report.json',report);print(json.dumps(report,indent=2),flush=True)
    return report


def run(source,reviews_path,out,repo=ROOT):
    import spacy
    start=time.monotonic();source,out,repo=map(Path.resolve,(source,out,repo))
    if out.exists() or out.is_relative_to(repo/'data') or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('use a new separate quarantine directory')
    reviews=list(read_rows(reviews_path))
    if not reviews or len({r['id'] for r in reviews})!=len(reviews): raise ValueError('empty/duplicate reviews')
    protected={str(p):digest(p) for folder in (repo/'data/mined-templates',repo/'data/dailydialog/filtered',repo/'data/dailydialog/review-parts') for p in folder.rglob('*') if p.is_file()}
    frozen={str(p):digest(p) for p in source.rglob('*') if p.is_file() and (p.is_relative_to(source/'bank') or p.name in {'plan.json','revision.sqlite3','contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'})}
    out.mkdir(parents=True)
    db,lexical,original,lex_plan=open_overlay(source,out)
    try:
        if digest(original)!=lex_plan['source_cache_sha256']: raise ValueError('parse cache changed')
        for name,value in lex_plan['lexical_resources'].items():
            if digest(repo/name)!=value: raise ValueError('dictionary changed')
        frozen.update({str(p):digest(p) for p in (original,lexical/'revision.sqlite3')})
        dump(out/'plan.json',dict(version=VERSION,source=str(source),reviews=str(reviews_path.resolve()),
            protected=protected,frozen=frozen,code={str(p):digest(p) for p in (Path(__file__),repo/'scripts/process_dailydialog_hybrid.py',repo/'scripts/audit_dailydialog_hybrid_bank.py',repo/'semantic_corpus/template_compiler.py')},published=False))
        old,new=refresh_contexts(source,repo,reviews,out)
        vocab,audit=spacy.blank('en').vocab,LexicalAudit(repo/'data/wiktionary')
        prior=list(read_rows(source/'reviews.jsonl'));latest={r['id']:r for r in prior}
        for review in reviews:
            inputs=[]
            for identity in review['old_ids']:
                found=db.execute('SELECT o.body,p.payload FROM occurrences o JOIN parses p ON o.key=p.key WHERE o.id=?',(identity,)).fetchone()
                if not found: raise ValueError('unknown source address')
                body,payload=map(json.loads,found);inputs.append(dict(body=body,payload=payload))
            cid=inputs[0]['body']['provenance']['context_id']
            body,payload=rewrite(inputs,old[cid],new[cid],review,audit,vocab)
            db.execute('INSERT OR REPLACE INTO corrections VALUES (?,?,?,?,?,?)',(review['id'],body['provenance']['cache_key'],body['example'],json.dumps(body),json.dumps(payload),json.dumps(review)))
            for item in inputs[1:]:
                db.execute('INSERT INTO retired_occurrences VALUES (?,?,?)',(item['body']['idx'],review['id'],json.dumps(item)))
            latest[review['id']]=review
        db.commit()
        with (out/'reviews.jsonl').open('w',encoding='utf-8') as stream:
            for r in latest.values(): write(stream,r)
        checker=WholeBankAudit(out,out)
        try:
            counts=full.export(db,out,observer=checker.observe);coverage=checker.finish()
        finally: checker.stream.close()
        old_manifest=json.loads((source/'bank/manifest.json').read_text(encoding='utf-8'))
        manifest,validation=full.materialize(db,out,old_manifest['model'],repo/'data/wiktionary')
        return finish(source,out,reviews_path,db,counts,validation,time.monotonic()-start)
    finally: db.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--reviews',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.reviews,args.out)
