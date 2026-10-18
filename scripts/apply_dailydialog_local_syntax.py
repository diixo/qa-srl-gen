"""Execute explicit local token edits without inventing contextual meanings.

Decisions are occurrence-addressed, not propagated to equal strings. Prior
analyses remain in each preimage. Whole sentences remain UNREVIEWED.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
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
from scripts.benchmark_dailydialog_hybrid import dump, risk_flags
from scripts.import_dailydialog_sentences import digest
from scripts.revise_dailydialog_hybrid_lexical import cached_doc
from semantic_corpus.template_bank import CorpusTemplate, fingerprint, read_rows
from semantic_corpus.template_compiler import LexicalAudit, compile_record
from semantic_corpus.template_validation import _check_rendered

VERSION = 'explicit_local_syntax_v2'
FIELDS = {'pos', 'tag', 'dep', 'head', 'parser_lemma', 'morph'}


def apply(body, payload, context, review, audit, vocab):
    if (review['method'] != 'direct_assistant_local_syntax_review'
            or review['id'] != body['idx'] or review['semantic_approval'] is not False
            or review['publication'] is not False or not review['note']
            or review['preimage'] != fingerprint(dict(body=body, payload=payload))
            or review['context_sha256'] != fingerprint(context)):
        raise ValueError('Stale or incomplete local review')
    if (body['provenance'].get('context_review')
            or payload.get('method') != 'local_spacy_syntax'
            or payload.get('review_status') != 'UNREVIEWED'):
        raise ValueError('Local patch cannot overwrite a contextual reference')
    origin = body['provenance']['origin']
    if context[origin['turn']-1][slice(*origin['span'])] != body['example']:
        raise ValueError('Source address mismatch')
    ledger = deepcopy(payload['ledger'])
    edits = review['token_edits']
    seen = set()
    for edit in edits:
        i, fields = edit['token'], edit['set']
        if (type(i) is not int or not 0 <= i < len(ledger['tokens']) or i in seen
                or not fields or set(fields)-FIELDS):
            raise ValueError('Invalid explicit token edit')
        seen.add(i)
        ledger['tokens'][i].update(fields)
    if not edits:
        raise ValueError('Empty local patch')
    ts = ledger['tokens']
    old_roots=[t['i'] for t in payload['ledger']['tokens'] if t['dep']=='ROOT']
    new_roots=[t['i'] for t in ts if t['dep']=='ROOT']
    transition=review.get('root_transition')
    if transition is not None:
        if transition != {'before':old_roots,'after':new_roots}:
            raise ValueError('Stale explicit root transition: '+body['idx'])
    elif len(new_roots)!=len(old_roots):
        raise ValueError('Local patch changed sentence-root count: '+body['idx'])
    if not old_roots or not new_roots:
        raise ValueError('Local patch lost all sentence roots: '+body['idx'])
    for i in range(len(ts)):
        visited = set()
        while True:
            if type(i) is not int or not 0 <= i < len(ts) or i in visited:
                raise ValueError('Invalid dependency tree')
            visited.add(i)
            parent = ts[i]['head']
            if parent == i:
                if ts[i]['dep'] != 'ROOT':
                    raise ValueError('Non-root self head')
                break
            i = parent
    doc = cached_doc(ledger, vocab)
    record, compiled = compile_record(doc, {}, audit)
    for before, after in zip_longest(ts, compiled['tokens']):
        if before is None or after is None:
            raise ValueError('Compiler changed token count: '+body['idx'])
        changed={k:(before[k],after[k]) for k in FIELDS | {'i','text','span','verb_lemmas','adjective_lemmas'} if before[k]!=after[k]}
        if changed:
            raise ValueError('Compiler differs at '+body['idx']+':'+str(before['i'])+': '+str(changed))
    rendered = CorpusTemplate(record).render()
    _check_rendered(rendered)
    if rendered['text'] != body['example']:
        raise ValueError('Local patch changed text')
    # Keep all existing revision chains and their evidence at their old keys.
    ledger.update(compiled)
    ledger['local_syntax_revision'] = dict(version=VERSION, review=deepcopy(review),
        preimage=dict(body=deepcopy(body),payload=deepcopy(payload)), semantic_approval=False)
    result = dict(payload, record=record, ledger=ledger, flags=risk_flags(doc,record,ledger),
                  variant=full.variant_signature(record,ledger), syntax_revision=VERSION)
    updated = deepcopy(body)
    key = fingerprint([VERSION,review])
    updated['provenance'].update(cache_key=key, previous_cache_key=body['provenance']['cache_key'])
    return updated, result


def conserve(source, out, reviews):
    def rows(root):
        for path in sorted((root/'bank/occurrences').glob('*.jsonl.gz')):
            yield from read_rows(path)
    targets = {r['id']:r for r in reviews}
    revised = set();counts=Counter()
    derived = {'template_id','group_id','syntax_revision','tokens','attributes','bindings',
               'predicate_templates','predicate_groups','sentence_template','analysis_status',
               'imperative_candidates','local_syntax_revision','cache_key','previous_cache_key',
               'candidate_kind'}
    for a,b in zip_longest(rows(source),rows(out)):
        if a is None or b is None or a['idx'] != b['idx']:
            raise ValueError('Lost or reordered occurrence')
        counts['segments']+=1;counts['tokens']+=len(a['tokens'])
        if a['idx'] not in targets:
            if a != b:raise ValueError('Unreviewed occurrence changed')
            counts['unchanged_occurrences']+=1
            continue
        r=targets[a['idx']];rev=b['local_syntax_revision']
        if (rev['review']!=r or rev['semantic_approval'] is not False
                or b['cache_key']!=fingerprint([VERSION,r])
                or b['previous_cache_key']!=a['cache_key']
                or fingerprint(rev['preimage'])!=r['preimage']
                or rev['preimage']['payload']['ledger']['tokens']!=a['tokens']):
            raise ValueError('Local patch lost preimage')
        if {k:v for k,v in a.items() if k not in derived}!={k:v for k,v in b.items() if k not in derived}:
            raise ValueError('Local patch changed prior metadata')
        edits={e['token']:e['set'] for e in r['token_edits']}
        for i,(x,y) in enumerate(zip_longest(a['tokens'],b['tokens'])):
            if x is None or y is None:raise ValueError('Tokenization changed')
            expected=dict(x,**edits.get(i,{}))
            if any(expected[k]!=y[k] for k in FIELDS|{'i','text','span','verb_lemmas','adjective_lemmas'}):
                raise ValueError('Unexpected token change')
        revised.add(a['idx']);counts['revised_occurrences']+=1
    if revised!=targets.keys():raise ValueError('Missing reviewed occurrence')
    for name in ('contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'):
        if digest(source/name)!=digest(out/name):raise ValueError('Context or previous reviews changed')
    return dict(status='passed',counts=dict(counts),source_and_tokens_unchanged=True,
                previous_reviews_preserved=True,all_unaddressed_occurrences_identical=True)


def run(source, reviews, out, repo=ROOT):
    import spacy
    start=time.monotonic();source=source.resolve();out=out.resolve()
    if out.exists() or out.is_relative_to(repo/'data') or out.is_relative_to(source):
        raise ValueError('Use a fresh quarantine output outside data/')
    decisions=list(read_rows(reviews))
    if not decisions or len({r['id'] for r in decisions})!=len(decisions):
        raise ValueError('Empty or duplicate decisions')
    protected={str(p):digest(p) for root in (repo/'data/mined-templates',repo/'data/dailydialog',source)
               for p in root.rglob('*') if p.is_file()}
    out.mkdir(parents=True)
    db,lexical,original,lexplan=open_overlay(source,out)
    try:
        for name,value in lexplan['lexical_resources'].items():
            if digest(repo/name)!=value:raise ValueError('Lexicon changed')
        if digest(original)!=lexplan['source_cache_sha256']:raise ValueError('Source cache changed')
        contexts={r['id']:r['turns'] for r in read_rows(source/'contexts.jsonl.gz')}
        audit=LexicalAudit(repo/'data/wiktionary');vocab=spacy.blank('en').vocab
        dump(out/'plan.json',dict(version=VERSION,source=str(source),reviews=str(reviews.resolve()),
            review_sha256=digest(reviews),protected=protected,
            code_sha256=digest(Path(__file__)),api_calls=0,published=False))
        for review in decisions:
            row=db.execute('SELECT o.body,p.payload FROM occurrences o JOIN parses p ON o.key=p.key WHERE o.id=?',(review['id'],)).fetchone()
            if not row:raise ValueError('Unknown occurrence '+review['id'])
            body,payload=map(json.loads,row)
            updated,result=apply(body,payload,contexts[body['provenance']['context_id']],review,audit,vocab)
            # A previously repaired occurrence may need another explicit local
            # correction. apply() binds and retains that complete prior state.
            db.execute('INSERT OR REPLACE INTO corrections VALUES (?,?,?,?,?,?)',
                (updated['idx'],updated['provenance']['cache_key'],updated['example'],
                 json.dumps(updated),json.dumps(result),json.dumps(review)))
        db.commit()
        for name in ('contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'):
            shutil.copyfile(source/name,out/name)
        checker=WholeBankAudit(source,out)
        try:
            totals=full.export(db,out,observer=checker.observe);coverage=checker.finish()
        finally:checker.stream.close()
        meta=json.loads((source/'bank/manifest.json').read_text(encoding='utf-8'))['model']
        manifest,validation=full.materialize(db,out,meta,repo/'data/wiktionary')
        conservation=conserve(source,out,decisions)
        for name,value in protected.items():
            if digest(Path(name))!=value:raise ValueError('Protected file changed: '+name)
        shutil.copytree(source/'bank/review',out/'bank/review')
        shutil.copyfile(reviews,out/'bank/review/local-syntax-decisions.jsonl')
        manifest['hybrid_import']['local_syntax_review']=dict(version=VERSION,
            occurrences=len(decisions),review_status='UNREVIEWED',semantic_approval=False)
        dump(out/'bank/manifest.json',manifest)
        report=dict(status='local_syntax_reviews_complete',validation=validation,conservation=conservation,
            counts=totals,dialogues=coverage['dialogues'],turns=coverage['turns'],
            protected_files_unchanged=len(protected),api_calls=0,parser_inference_calls=0,published=False,
            elapsed_seconds=round(time.monotonic()-start,2))
        dump(out/'report.json',report)
        return report
    finally:db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','reviews','out'):p.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
