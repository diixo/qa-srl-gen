"""Apply explicit, occurrence-scoped reviews to a quarantined hybrid bank.

No inference or publication. Identical text in another context stays untouched.
The original cache and the lexical revision are attached read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import process_dailydialog_hybrid as full
from scripts.benchmark_dailydialog_hybrid import dump, risk_flags, write
from scripts.import_dailydialog_sentences import digest
from scripts.revise_dailydialog_hybrid_lexical import cached_doc
from semantic_corpus.template_bank import CorpusTemplate, fingerprint, read_rows
from semantic_corpus.template_compiler import LexicalAudit, compile_record
from semantic_corpus.template_validation import _check_rendered
from semantic_corpus.template_usage import constrain_record

RULE = 'occurrence_context_review_v1'
SHAPE = ('parts', 'slots', 'verbs', 'constructions', 'predicate_groups', 'attributes')


def overlay(original, lexical, output):
    db = sqlite3.connect(output.resolve().as_uri(), uri=True)
    for alias, path in (('original', original), ('lexical', lexical)):
        db.execute(f'ATTACH DATABASE ? AS {alias}', (path.resolve().as_uri()+'?mode=ro',))
    db.execute('CREATE TABLE IF NOT EXISTS corrections '
               '(id TEXT PRIMARY KEY, key TEXT UNIQUE, text TEXT, body TEXT, payload TEXT, review TEXT)')
    db.execute('CREATE TEMP VIEW base_parses AS SELECT p.key,p.text,COALESCE(r.payload,p.payload) payload '
               'FROM original.parses p LEFT JOIN lexical.replacements r ON p.key=r.key')
    db.execute('CREATE TEMP VIEW parses AS SELECT key,text,payload FROM base_parses UNION ALL '
               'SELECT key,text,payload FROM corrections')
    db.execute('CREATE TEMP VIEW occurrences AS SELECT o.rowid,o.id,COALESCE(c.key,o.key) key,'
               'COALESCE(c.body,o.body) body FROM original.occurrences o LEFT JOIN corrections c ON c.id=o.id')
    return db


def apply_review(body, payload, context, review, audit, vocab):
    """Validate exact preimages, then apply only the model-authored token edits."""
    if body['provenance'].get('context_review'):
        raise ValueError('an existing context review cannot be overwritten')
    origin = body['provenance']['origin']
    identity = ':'.join(str(origin[k]) for k in ('split', 'dialogue', 'turn', 'segment'))
    if body['idx'] != identity or review['id'] != identity:
        raise ValueError('occurrence identity mismatch')
    expected = dict(source_sha256=fingerprint(body['example']),
                    context_sha256=fingerprint(context), payload_sha256=fingerprint(payload))
    if any(review.get(k) != v for k, v in expected.items()):
        raise ValueError('stale source, context or parse preimage')
    if review.get('method') != 'direct_assistant_review':
        raise ValueError('explicit contextual review required')
    reference_overlay = (payload.get('review_status') == 'REVIEWED_REUSE'
                         and payload.get('method') == 'reviewed_whole_dialogue_reuse'
                         and review.get('reference_overlay') is True
                         and not review['token_edits']
                         and review.get('usage', {}).get('mode') == 'context_only')
    if not reference_overlay and (payload.get('review_status') != 'UNREVIEWED'
                                  or payload.get('method') != 'local_spacy_syntax'):
        raise ValueError('only unreviewed automatic occurrences may be revised')
    start, end = origin['span']
    if context[origin['turn']-1][start:end] != body['provenance']['original']:
        raise ValueError('source address mismatch')
    if review['unit'] not in {'sentence', 'fragment', 'pending'}:
        raise ValueError('invalid reviewed unit')
    if review['unit'] == 'pending' and review['token_edits']:
        raise ValueError('unresolved source text cannot receive a guessed syntax repair')
    meaning = review['meaning']
    if (not meaning.get('speech_act') or not meaning.get('event_status') or
            not isinstance(meaning.get('content'), dict) or
            type(meaning.get('completed_action_asserted')) is not bool or
            not isinstance(meaning.get('unknown'), list)):
        raise ValueError('incomplete contextual meaning')
    evidence = review['evidence_turns']
    if not evidence or any(type(t) is not int or not 1 <= t <= len(context) for t in evidence):
        raise ValueError('invalid context evidence')
    updated, result = deepcopy(body), deepcopy(payload)
    changes, seen = review['token_edits'], set()
    if changes:
        if any(k in payload['ledger'] for k in ('automatic_revision', 'attribute_revision')):
            raise ValueError('syntax edits require explicit reconciliation with previous repairs')
        ledger = deepcopy(payload['ledger'])
        for edit in changes:
            index = edit['token']
            if type(index) is not int or not 0 <= index < len(ledger['tokens']) or index in seen:
                raise ValueError('invalid/duplicate token edit')
            seen.add(index)
            fields = edit['set']
            if not fields or set(fields) - {'pos', 'tag', 'dep', 'head', 'parser_lemma', 'morph'}:
                raise ValueError('only explicit syntax fields may change')
            ledger['tokens'][index].update(fields)
        tokens = ledger['tokens']
        for token in tokens:
            head = token['head']
            if type(head) is not int or not 0 <= head < len(tokens):
                raise ValueError('invalid dependency head')
            visited, current = set(), token['i']
            while True:
                if current in visited:
                    raise ValueError('cyclic dependency')
                visited.add(current)
                parent = tokens[current]['head']
                if parent == current:
                    if tokens[current]['dep'] != 'ROOT':
                        raise ValueError('self head requires ROOT')
                    break
                current = parent
        doc = cached_doc(ledger, vocab)
        record, compiled = compile_record(doc, {}, audit)
        rendered = CorpusTemplate(record).render()
        _check_rendered(rendered)
        if rendered['text'] != body['example']:
            raise ValueError('review changed source text')
        for before, after in zip(payload['ledger']['tokens'], compiled['tokens']):
            for name in ('text', 'span', 'verb_lemmas', 'adjective_lemmas'):
                if before[name] != after[name]:
                    raise ValueError('tokenization or dictionary evidence changed')
        for edit in changes:
            index = edit['token']
            token = compiled['tokens'][index]
            if any(token[k] != value for k, value in edit['set'].items()):
                raise ValueError(f'compiled syntax differs from requested token edit: {identity}:{index}')
            token['analysis'] = 'model_context_review'
            token['previous_syntax'] = {k: payload['ledger']['tokens'][index][k]
                                        for k in edit['set']}
            for key, span in rendered['spans'].items():
                if span == token['span']:
                    record['slots'][key]['analysis'] = 'model_context_review'
        record['template_id'] = fingerprint({k: record[k] for k in SHAPE})
        compiled['template_id'] = record['template_id']
        result.update(record=record, ledger=compiled, flags=risk_flags(doc, record, compiled),
                      variant=full.variant_signature(record, compiled), syntax_revision=RULE)
    if 'usage' in review:
        rendered = constrain_record(result['record'], body, context, review)
        result['ledger'].update(template_id=result['record']['template_id'], attributes=rendered['attributes'])
        result['variant'] = full.variant_signature(result['record'], result['ledger'])
    result['flags'] = sorted(set(result['flags']) | {'context_reviewed_occurrence'})
    if review['unit'] == 'pending':
        result['flags'].append('context_pending')
    annotation = dict(rule=RULE, status='REVIEWED_OCCURRENCE', method=review['method'],
        unit=review['unit'], meaning=deepcopy(meaning), evidence_turns=evidence,
        note=review['note'], syntax_corrected=bool(changes), **expected,
        template_semantics_approved=False, unseen_group_members_approved=False)
    if 'usage' in review:
        annotation['usage'] = deepcopy(review['usage'])
        if review['usage'].get('property_bindings'):
            annotation['previous_attributes'] = deepcopy(payload['record']['attributes'])
    if reference_overlay:
        annotation['reference_overlay'] = True
    # Only this occurrence gets the new cache key and meaning. No global text replacement.
    key = fingerprint([RULE, identity, review])
    updated['provenance'].update(cache_key=key, previous_cache_key=body['provenance']['cache_key'],
                                 context_review=annotation)
    return updated, result


def read_reviews(paths):
    """Replay earlier and new decisions together, rejecting conflicting addresses."""
    decisions = [row for path in paths for row in read_rows(path)]
    if not decisions or len({r['id'] for r in decisions}) != len(decisions):
        raise ValueError('empty or duplicate reviews')
    return decisions


def verify_conservation(source, out, decisions):
    """Compare all materialized occurrences, allowing only addressed reviews."""
    def rows(root):
        for path in sorted((root/'bank/occurrences').glob('*.jsonl.gz')):
            yield from read_rows(path)
    reviews = {r['id']: r for r in decisions}
    seen, counts = set(), Counter()
    for before, after in zip_longest(rows(source), rows(out)):
        if before is None or after is None or before['idx'] != after['idx']:
            raise ValueError('missing or reordered occurrence')
        identity = before['idx']
        counts['segments'] += 1
        counts['tokens'] += len(before['tokens'])
        if identity not in reviews:
            if before != after:
                raise ValueError('unaddressed occurrence changed: '+identity)
            counts['identical_occurrences'] += 1
            continue
        seen.add(identity)
        review = reviews[identity]
        reference_overlay = (review.get('reference_overlay') is True
                             and before.get('review_status') == 'REVIEWED_REUSE'
                             and before.get('method') == 'reviewed_whole_dialogue_reuse'
                             and not review['token_edits']
                             and review.get('usage', {}).get('mode') == 'context_only')
        if before.get('context_review') or (before.get('review_status') != 'UNREVIEWED' and not reference_overlay):
            raise ValueError('protected occurrence changed')
        allowed = {'cache_key','previous_cache_key','context_review','candidate_kind','group_id'}
        if 'usage' in review:
            allowed |= {'template_id', 'attributes'}
        if review['token_edits']:
            allowed |= {'template_id','tokens','attributes','syntax_revision','analysis_status',
                        'bindings','predicate_templates','imperative_candidates','sentence_template','predicate_groups'}
        if ({k:v for k,v in before.items() if k not in allowed} !=
                {k:v for k,v in after.items() if k not in allowed}):
            raise ValueError('source or prior repair changed: '+identity)
        if after['previous_cache_key'] != before['cache_key']:
            raise ValueError('review lost its preimage key')
        edits = {e['token']:e['set'] for e in review['token_edits']}
        for index, (old, new) in enumerate(zip_longest(before['tokens'], after['tokens'])):
            if old is None or new is None:
                raise ValueError('tokenization changed')
            if index not in edits:
                if old != new:
                    raise ValueError('unaddressed token changed')
            else:
                allowed_token = set(edits[index]) | {'lemma','kind','analysis','previous_syntax'}
                if ({k:v for k,v in old.items() if k not in allowed_token} !=
                        {k:v for k,v in new.items() if k not in allowed_token}):
                    raise ValueError('token text, spans or lexical evidence changed')
                if any(new[k] != v for k,v in edits[index].items()):
                    raise ValueError('requested token edit missing')
        annotation = after['context_review']
        if annotation['meaning'] != review['meaning'] or annotation['template_semantics_approved']:
            raise ValueError('review meaning changed or template approved')
        if 'usage' in review and annotation.get('usage') != review['usage']:
            raise ValueError('review lost its usage restrictions')
        if reference_overlay and annotation.get('reference_overlay') is not True:
            raise ValueError('reference overlay provenance lost')
        if review.get('usage', {}).get('property_bindings'):
            # Ledgers may also contain spans and modifiers absent from the record.
            prior_attributes = annotation.get('previous_attributes', [])
            previous = {a['slot']: a for a in before['attributes']}
            if (not prior_attributes or len(prior_attributes) != len(previous)
                    or any(a['slot'] not in previous or
                           any(previous[a['slot']].get(k) != v for k, v in a.items())
                           for a in prior_attributes)):
                raise ValueError('previous property bindings lost')
        counts['reviewed_occurrences'] += 1
        counts['syntax_corrections'] += bool(edits)
    if seen != set(reviews):
        raise ValueError('reviewed occurrences missing')
    for name in ('contexts.jsonl.gz','coverage.jsonl.gz'):
        if digest(source/name) != digest(out/name):
            raise ValueError('source context or coverage changed')
    result = dict(status='passed',counts=dict(counts),source_and_tokenization_unchanged=True,
        unaddressed_occurrences_and_previous_repairs_identical=True,published=False)
    dump(out/'context-conservation-audit.json',result)
    return result


def run(source, reviews, out, repo=ROOT):
    import spacy
    from scripts.audit_dailydialog_hybrid_bank import open_overlay, WholeBankAudit
    started = time.monotonic()
    source, out, repo = (p.resolve() for p in (source, out, repo))
    review_paths = [p.resolve() for p in ([reviews] if isinstance(reviews, Path) else reviews)]
    plan = json.loads((source/'plan.json').read_text(encoding='utf-8'))
    incremental = 'lexical_resources' not in plan
    if (out.exists() or out.is_relative_to(repo/'data') or out.is_relative_to(source)
            or source.is_relative_to(out)):
        raise ValueError('use a new, separate quarantine output outside data/')
    protected = {str(p.relative_to(repo)): digest(p) for root in (
        repo/'data/mined-templates', repo/'data/dailydialog/filtered',
        repo/'data/dailydialog/review-parts') for p in root.rglob('*') if p.is_file()}
    frozen = {str(p):digest(p) for p in source.rglob('*') if p.is_file() and
              (p.is_relative_to(source/'bank') or p.name in
               {'revision.sqlite3','plan.json','contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'})}
    frozen.update({str(p):digest(p) for p in review_paths})
    decisions = read_reviews(review_paths)
    prior = list(read_rows(source/'reviews.jsonl')) if incremental else []
    if {r['id'] for r in prior} & {r['id'] for r in decisions}:
        raise ValueError('an existing context review cannot be overwritten')
    contexts = {r['id']: r['turns'] for r in read_rows(source/'contexts.jsonl.gz')}
    out.mkdir(parents=True)
    if incremental:
        db, lexical, original, lex_plan = open_overlay(source,out)
    else:
        lexical, lex_plan = source, plan
        original = Path(plan['source'])/'cache.sqlite3'
        db = overlay(original, source/'revision.sqlite3', out/'revision.sqlite3')
    try:
        for name, value in lex_plan['lexical_resources'].items():
            if digest(repo/name) != value:
                raise ValueError('dictionary changed since lexical revision')
        if digest(original) != lex_plan['source_cache_sha256']:
            raise ValueError('original parse cache changed')
        frozen.update({str(p):digest(p) for p in (original,lexical/'revision.sqlite3')})
        dump(out/'plan.json', dict(rule=RULE, source=str(source), incremental=incremental,
            frozen=frozen, protected=protected, review_inputs=[str(p) for p in review_paths],
            code={str(p.relative_to(repo)):digest(p) for p in (Path(__file__),
                repo/'scripts/audit_dailydialog_hybrid_bank.py',repo/'scripts/process_dailydialog_hybrid.py',
                repo/'scripts/revise_dailydialog_hybrid_lexical.py',repo/'semantic_corpus/template_compiler.py',
                repo/'semantic_corpus/template_usage.py',repo/'semantic_corpus/template_bank.py')},
            api_calls=0, parser_inference_calls=0, published=False))
        previous_corrections = list(db.execute('SELECT * FROM corrections ORDER BY id'))
        if {r[0] for r in previous_corrections} != {r['id'] for r in prior}:
            raise ValueError('previous review file and database disagree')
        with (out/'reviews.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for decision in prior + decisions:
                write(stream, decision)
        vocab, audit = spacy.blank('en').vocab, LexicalAudit(repo/'data/wiktionary')
        for review in decisions:
            found = db.execute('SELECT o.body,p.payload FROM occurrences o JOIN parses p ON o.key=p.key '
                               'WHERE o.id=?', (review['id'],)).fetchone()
            if not found:
                raise ValueError('unknown occurrence: '+review['id'])
            body, payload = map(json.loads, found)
            body, payload = apply_review(body, payload, contexts[body['provenance']['context_id']], review, audit, vocab)
            db.execute('INSERT INTO corrections VALUES (?,?,?,?,?,?)', (review['id'],
                body['provenance']['cache_key'], body['example'], json.dumps(body), json.dumps(payload), json.dumps(review)))
        db.commit()
        for name in ('contexts.jsonl.gz', 'coverage.jsonl.gz'):
            shutil.copyfile(source/name, out/name)
        checker = WholeBankAudit(source,out)
        try:
            counts = full.export(db, out, observer=checker.observe)
            coverage = checker.finish()
        finally:
            checker.stream.close()
        previous_manifest = json.loads((source/'bank/manifest.json').read_text(encoding='utf-8'))
        meta = previous_manifest['model']
        manifest, validation = full.materialize(db, out, meta, repo/'data/wiktionary')
        baseline = json.loads((source/'report.json').read_text(encoding='utf-8'))
        if (counts['compiled_occurrences'] != baseline['counts']['compiled_occurrences'] or
                validation['token_spans'] != baseline['validation']['token_spans']):
            raise ValueError('coverage changed')
        for path, value in frozen.items():
            if digest(Path(path)) != value:
                raise ValueError('frozen input changed: '+path)
        for name, value in protected.items():
            if digest(repo/name) != value:
                raise ValueError('protected input changed: '+name)
        new_ids = {r['id'] for r in decisions}
        if [r for r in db.execute('SELECT * FROM corrections ORDER BY id') if r[0] not in new_ids] != previous_corrections:
            raise ValueError('previous context review changed')
        if incremental:
            has_repairs = db.execute("SELECT 1 FROM previous.sqlite_master WHERE name='automatic_repairs'").fetchone()
            previous_repairs = list(db.execute('SELECT * FROM previous.automatic_repairs ORDER BY key')) if has_repairs else []
            if list(db.execute('SELECT * FROM automatic_repairs ORDER BY key')) != previous_repairs:
                raise ValueError('previous automatic repairs changed')
        conservation = verify_conservation(source,out,decisions)
        review_dir = out/'bank/review'
        if (source/'bank/review').exists():
            shutil.copytree(source/'bank/review',review_dir)
        else:
            review_dir.mkdir()
        shutil.copyfile(out/'reviews.jsonl', review_dir/'context-reviews.jsonl')
        for name in ('syntax-audit.jsonl.gz','review-queue.jsonl.gz','audit-summary.json',
                     'apostrophe-audit.json','context-conservation-audit.json'):
            shutil.copyfile(out/name,review_dir/name)
        # Keep earlier family reviews and repair history alongside this stage.
        manifest['hybrid_import'] = {**previous_manifest.get('hybrid_import',{}), **manifest['hybrid_import']}
        manifest['hybrid_import']['context_revision'] = dict(rule=RULE,
            review_file='review/context-reviews.jsonl', occurrences=len(prior)+len(decisions),
            new_occurrences=len(decisions), syntax_corrections=sum(bool(r['token_edits']) for r in prior+decisions))
        manifest['hybrid_import']['apostrophe_audit'] = 'review/apostrophe-audit.json'
        dump(out/'bank/manifest.json', manifest)
        report = dict(status='context_reviews_applied', counts=counts, validation=validation,
            reviewed_occurrences=len(prior)+len(decisions),new_reviews=len(decisions),
            new_syntax_corrections=sum(bool(r['token_edits']) for r in decisions),
            syntax_corrections=sum(bool(r['token_edits']) for r in prior+decisions),
            conservation=conservation,dialogues=coverage['dialogues'],turns=coverage['turns'],
            protected_files_unchanged=len(protected), frozen_inputs_unchanged=True,
            api_calls=0, parser_inference_calls=0, published=False,
            elapsed_seconds=round(time.monotonic()-started, 2))
        dump(out/'report.json', report)
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'artifacts/dailydialog-hybrid-lexical-v1')
    parser.add_argument('--reviews', type=Path, nargs='+', required=True,
                        help='new decisions for an incremental source; all decisions for a lexical source')
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/dailydialog-hybrid-context-v1')
    args = parser.parse_args()
    run(args.source, args.reviews, args.out)
