"""Apply explicit wording/boundary decisions to an isolated candidate view.

Authoritative source files, manual references and the published pool are immutable.
Only changed segments are reparsed locally; their previous analyses are retained,
and the new analysis is explicitly unreviewed. A wording correction never grants
semantic approval or silently carries a previous token-level approval forward.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from difflib import SequenceMatcher
import gzip
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
from scripts.resegment_dailydialog_hybrid_source import insertion_positions
from semantic_corpus.template_bank import CorpusTemplate, fingerprint, read_rows
from semantic_corpus.template_compiler import LexicalAudit, compile_record
from semantic_corpus.template_validation import _check_rendered
from semantic_corpus.template_usage import constrain_record

VERSION = 'explicit_candidate_wording_v1'


def invalidate_context_review(body, payload, before_context, after_context, preimage):
    """Retain prior evidence but require a fresh review in the revised context."""
    if payload.get('method') != 'local_spacy_syntax' or payload.get('review_status') != 'UNREVIEWED':
        raise ValueError('Cannot invalidate a protected manual reference')
    b, p = deepcopy(body), deepcopy(payload)
    previous = b['provenance'].pop('context_review', None)
    if previous:
        b['provenance'].setdefault('context_review_history', []).append(previous)
    note = 'Candidate wording changed in this dialogue; previous contextual approval is retained only as historical evidence.'
    review = dict(unit='pending', meaning=dict(status='UNREVIEWED_AFTER_CONTEXT_CHANGE',
        unknown=['The interpretation must be checked against the revised dialogue.']),
        usage=dict(mode='pending', reasons=[note]))
    constrain_record(p['record'], b, after_context, review)
    p['ledger']['template_id'] = p['record']['template_id']
    p['ledger']['candidate_context_revision'] = dict(version=VERSION,
        preimage=deepcopy(preimage), before_context=deepcopy(before_context),
        after_context_sha256=fingerprint(after_context), semantic_approval=False,
        previous_review_requires_recheck=True)
    p['variant'] = full.variant_signature(p['record'], p['ledger'])
    p['flags'] = sorted((set(p['flags']) - {'context_reviewed_occurrence'}) | {'context_changed_unreviewed'})
    b['provenance'].update(cache_key=fingerprint([VERSION, b['idx'], fingerprint(preimage), after_context]),
        previous_cache_key=body['provenance']['cache_key'], semantic_review='UNREVIEWED')
    return b, p


def plan_turn(text, segments, wording, boundaries):
    """Compose occurrence-addressed decisions; reject ambiguity and overlap."""
    full.check_spans(text, [s['span'] for s in segments])
    by_id = {s['id']: s for s in segments}
    edits = set()
    for review in wording:
        s = by_id[review['id']]
        if (review['decision'] != 'explicit_wording_correction'
                or review['method'] != 'direct_assistant_local_source_wording_review'
                or review['semantic_approval'] is not False or review['publication'] is not False
                or review['source_text'] != s['text']
                or s['text'] != text[slice(*s['span'])]):
            raise ValueError('stale or overreaching wording review')
        revised = s['text']
        for edit in review['edits']:
            a, b = edit['span']
            if revised[a:b] != edit['before']:
                raise ValueError('wording edit preimage mismatch')
            revised = revised[:a] + edit['after'] + revised[b:]
        if revised != review['corrected_text']:
            raise ValueError('wording output mismatch')
        # Independent warnings may correct different words in the same segment.
        for op, a, b, c, d in SequenceMatcher(None, s['text'], revised, autojunk=False).get_opcodes():
            if op != 'equal':
                edits.add((s['span'][0]+a, s['span'][0]+b, revised[c:d]))
    edits = sorted(edits)
    for left, right in zip(edits, edits[1:]):
        if (left[1] > right[0] or left[0] == right[0]
                or (left[1] == right[0] and right[0] == right[1])):
            raise ValueError('conflicting explicit wording decisions')
    cuts = {0, len(text)} | {s['span'][1] for s in segments}
    joins, splits = [], set()
    for review in boundaries:
        s = by_id[review['id']]
        if (review['method'] != 'direct_assistant_local_boundary_review'
                or review['semantic_approval'] is not False or review['publication'] is not False
                or review['current_text'] != s['text']
                or review['current_origin']['span'] != s['span']):
            raise ValueError('stale or overreaching boundary review')
        if review['decision'] == 'split_after':
            for prefix in review['value']:
                if not s['text'].startswith(prefix) or len(prefix) >= len(s['text']):
                    raise ValueError('split must be an exact internal prefix')
                splits.add(s['span'][0]+len(prefix))
        elif review['decision'] == 'join':
            for phrase in review['value']:
                if not phrase or text.count(phrase) != 1:
                    raise ValueError('join must address one exact source phrase')
                a = text.index(phrase)
                b = a + len(phrase)
                if not (a < s['span'][1] and s['span'][0] < b):
                    raise ValueError('join does not include reviewed segment')
                joins.append((a, b))
        else:
            raise ValueError('unsupported boundary action')
    if any(a < split < b for split in splits for a, b in joins):
        raise ValueError('conflicting explicit split and join')
    cuts = sorted({c for c in cuts if not any(a < c < b for a, b in joins)} | splits)

    def offset(pos):
        delta = 0
        for a, b, replacement in edits:
            if a < pos < b:
                raise ValueError('source edit crosses an explicit boundary')
            if b <= pos:
                delta += len(replacement)-(b-a)
        return pos+delta

    revised = text
    for a, b, replacement in reversed(edits):
        revised = revised[:a]+replacement+revised[b:]
    if revised != revised.translate(full.APOSTROPHES):
        raise ValueError('noncanonical apostrophe')
    pieces = []
    for a, b in zip(cuts, cuts[1:]):
        x, y = offset(a), offset(b)
        while x < y and revised[x].isspace():
            x += 1
        while y > x and revised[y-1].isspace():
            y -= 1
        if x < y:
            pieces.append(dict(span=[x, y], text=revised[x:y],
                input_ids=[s['id'] for s in segments if a < s['span'][1] and s['span'][0] < b]))
    full.check_spans(revised, [p['span'] for p in pieces])
    if set(i for p in pieces for i in p['input_ids']) != by_id.keys():
        raise ValueError('source segment disappeared')
    return dict(before=text, after=revised, edits=[dict(span=[a,b], after=r) for a,b,r in edits],
                pieces=pieces, wording=deepcopy(wording), boundaries=deepcopy(boundaries))


def changed_payload(doc, inputs, plan, audit):
    record, ledger = compile_record(doc, {}, audit)
    rendered = CorpusTemplate(record).render()
    _check_rendered(rendered)
    if rendered['text'] != doc.text:
        raise ValueError('candidate wording roundtrip mismatch')
    ledger['candidate_source_revision'] = dict(version=VERSION, preimages=deepcopy(inputs),
        plan_sha256=fingerprint(plan), semantic_approval=False,
        syntax_status='UNREVIEWED_AFTER_TEXT_OR_BOUNDARY_CHANGE',
        previous_local_decisions='retained in preimages; not transferred to the new parse')
    return dict(record=record, ledger=ledger, flags=sorted(set(risk_flags(doc, record, ledger)) |
        {'wording_or_boundary_changed_unreviewed'}), variant=full.variant_signature(record, ledger),
        method='local_spacy_syntax', review_status='UNREVIEWED', syntax_revision=VERSION,
        entities=[dict(text=e.text, label=e.label_, span=[e.start_char,e.end_char],
                       review_status='UNREVIEWED') for e in doc.ents])


def reconcile_duplicate_contexts(contexts, changed_contexts):
    """A local wording edit must not keep a now-false duplicate declaration."""
    by_id = {r['id']:r for r in contexts}
    for r in contexts:
        parent_id = r.get('duplicate_of')
        if not parent_id:
            continue
        parent = by_id.get(parent_id)
        stale = (parent is not None and r['turns'] != parent['turns'])
        stale |= parent is None and r['id'] in changed_contexts
        if stale:
            r['previous_duplicate_of'] = r.pop('duplicate_of')
            r['duplicate_revision'] = 'local wording view differs; original duplicate evidence retained'


def run(source, wording, boundaries, model, out, repo=ROOT, invalidate_context_reviews=False):
    import spacy
    started = time.monotonic()
    source, out, repo = map(Path.resolve, (source, out, repo))
    if out.exists() or out.is_relative_to(repo/'data') or out.is_relative_to(source):
        raise ValueError('use a fresh candidate directory outside data/')
    report = json.loads((source/'report.json').read_text(encoding='utf-8'))
    if report['validation']['status'] != 'passed':
        raise ValueError('source candidate is not validated')
    word_rows = [r for r in read_rows(wording) if r['decision']=='explicit_wording_correction']
    boundary_rows = [r for r in read_rows(boundaries) if r['decision'] in {'split_after','join'}]
    targets = defaultdict(lambda: [[], []])
    for k, rows in enumerate((word_rows, boundary_rows)):
        for r in rows:
            targets[r['id'].rsplit(':',1)[0]][k].append(r)
    protected = {str(p):digest(p) for root in (repo/'data/mined-templates',repo/'data/dailydialog',source)
                 for p in root.rglob('*') if p.is_file()}
    out.mkdir(parents=True)
    db, lexical, original, lexplan = open_overlay(source, out)
    try:
        if digest(original) != lexplan['source_cache_sha256']:
            raise ValueError('source cache changed')
        for name, value in lexplan['lexical_resources'].items():
            if digest(repo/name) != value:
                raise ValueError('lexical dictionary changed')
        contexts = list(read_rows(source/'contexts.jsonl.gz'))
        by_context = {r['id']:r for r in contexts}
        coverage = list(read_rows(source/'coverage.jsonl.gz'))
        protected_dialogues, manual_dialogues = set(), set()
        reviewed_images = {}
        for raw, parsed in db.execute('SELECT o.body,p.payload FROM occurrences o JOIN parses p ON o.key=p.key'):
            b, p = json.loads(raw), json.loads(parsed)
            cid = b['provenance']['context_id']
            if p.get('method') != 'local_spacy_syntax' or p.get('review_status') != 'UNREVIEWED':
                manual_dialogues.add(cid)
            if b['provenance'].get('context_review') or p.get('record',{}).get('usage_constraints'):
                protected_dialogues.add(cid)
                reviewed_images[b['idx']] = dict(body=b, payload=p)
        plans, held = {}, []
        for row in coverage:
            if row['id'] not in targets:
                continue
            words, bounds = targets[row['id']]
            cid = row['id'].rsplit(':',1)[0]
            try:
                plan = plan_turn(row['original'], row['segments'], words, bounds)
                if plan['after'] != plan['before'] and cid in manual_dialogues:
                    raise ValueError('context contains a protected manual reference')
                if plan['after'] != plan['before'] and cid in protected_dialogues and not invalidate_context_reviews:
                    raise ValueError('context contains a protected semantic review; source cannot be rewritten locally')
                originals = []
                for segment in row['segments']:
                    found = db.execute('SELECT o.rowid,o.body,p.payload FROM occurrences o JOIN parses p ON o.key=p.key WHERE o.id=?', (segment['id'],)).fetchone()
                    if not found:
                        raise ValueError('missing input occurrence')
                    position, b, p = found
                    originals.append(dict(position=position, body=json.loads(b), payload=json.loads(p)))
                # References are immutable, even when only a later span would move.
                for item in originals:
                    p = item['payload']
                    if (p.get('method')!='local_spacy_syntax' or p.get('review_status')!='UNREVIEWED'
                            or (item['body']['provenance'].get('context_review') and not invalidate_context_reviews)):
                        raise ValueError('turn contains a protected reference analysis')
                plans[row['id']] = dict(plan=plan, originals=originals)
            except ValueError as exc:
                held.append(dict(turn_id=row['id'], reason=str(exc), wording=words, boundaries=bounds,
                                 status='quarantined_not_applied', semantic_approval=False))
        if set(plans) | {h['turn_id'] for h in held} != targets.keys():
            raise ValueError('unaccounted review target')
        dump(out/'plan.json', dict(version=VERSION, source=str(source), protected=protected,
            review_sha256={str(wording):digest(wording),str(boundaries):digest(boundaries)},
            code_sha256=digest(Path(__file__)), model=str(model), api_calls=0, published=False,
            invalidate_context_reviews=invalidate_context_reviews))
        dump(out/'held-source-decisions.json', held)
        with (out/'applied-turn-plans.jsonl').open('w',encoding='utf-8') as stream:
            for turn_id, value in plans.items():
                write(stream, dict(turn_id=turn_id, **value['plan']))
        nlp = spacy.load(model)
        audit = LexicalAudit(repo/'data/wiktionary')
        parsed_count, adjusted_count, lineage = 0, 0, {}
        # Reuse the ordered address slots within the explicitly revised turn.
        # Lineage maps every output back to its original segment(s); addresses
        # outside this turn and their relative ordering never change.
        for row in coverage:
            saved = plans.get(row['id'])
            if saved is None:
                continue
            plan, originals = saved['plan'], saved['originals']
            original_by_id = {o['body']['idx']:o for o in originals}
            next_segment = max(o['body']['provenance']['origin']['segment'] for o in originals)+1
            results = []
            for piece_index, piece in enumerate(plan['pieces']):
                identity = (originals[piece_index]['body']['idx'] if piece_index < len(originals)
                            else row['id']+':'+str(next_segment))
                if piece_index >= len(originals):
                    next_segment += 1
                inputs = [dict(body=original_by_id[i]['body'],payload=original_by_id[i]['payload']) for i in piece['input_ids']]
                before = inputs[0]['body']
                b = deepcopy(before)
                b.update(idx=identity, example=piece['text'])
                unchanged = (len(inputs)==1 and identity==before['idx'] and piece['text']==before['example'])
                if unchanged:
                    p = deepcopy(inputs[0]['payload'])
                else:
                    p = changed_payload(nlp(piece['text']), inputs, plan, audit)
                    parsed_count += 1
                prov = b['provenance']
                prov['origin'].update(span=piece['span'], segment=int(identity.rsplit(':',1)[1]))
                prov['original'] = piece['text']
                if b != before or not unchanged:
                    key = fingerprint([VERSION, fingerprint(plan), identity, piece])
                    prov.update(cache_key=key,previous_cache_key=before['provenance']['cache_key'],
                                candidate_source_view=VERSION)
                    if unchanged:
                        p['ledger']['candidate_source_revision'] = dict(version=VERSION,
                            preimages=deepcopy(inputs),plan_sha256=fingerprint(plan),
                            syntax_status='UNCHANGED_TOKENS_RELOCATED',semantic_approval=False)
                        adjusted_count += 1
                results.append((b,p))
                lineage[identity] = dict(input_ids=piece['input_ids'], text=piece['text'],
                    token_analysis='preserved' if unchanged else 'reparsed_unreviewed',
                    turn_id=row['id'],plan_sha256=fingerprint(plan))
            output_ids = {b['idx'] for b,p in results}
            positions = [o['position'] for o in originals]
            extra = insertion_positions(db, positions[-1], max(0,len(results)-len(originals)))
            new_positions = positions[:len(results)]+extra
            # Reassign sort positions for already-existing extra occurrences too.
            # A join never changes the order of outputs outside this turn.
            for item in originals:
                identity = item['body']['idx']
                if identity not in output_ids:
                    db.execute('INSERT INTO retired_occurrences VALUES (?,?,?)',
                               (identity,results[0][0]['idx'],json.dumps(item)))
            for position,(b,p) in zip(new_positions,results):
                identity = b['idx']
                prior = original_by_id.get(identity)
                if prior is not None and b == prior['body'] and p == prior['payload']:
                    # A base parse key can be shared by many occurrences. Do
                    # not add a second copy of it to the corrections UNION.
                    continue
                if identity not in original_by_id:
                    if db.execute('SELECT 1 FROM occurrences WHERE id=?',(identity,)).fetchone() or db.execute('SELECT 1 FROM retired_occurrences WHERE id=?',(identity,)).fetchone():
                        raise ValueError('new sentence ID collision')
                    db.execute('INSERT INTO extra_occurrences VALUES (?,?,?)',(identity,originals[0]['body']['idx'],position))
                db.execute('INSERT OR REPLACE INTO corrections VALUES (?,?,?,?,?,?)',
                    (identity,b['provenance']['cache_key'],b['example'],json.dumps(b),json.dumps(p),json.dumps(lineage[identity])))
            cid, turn = row['id'].rsplit(':',1)
            context = by_context[cid]
            context['turns'][int(turn)-1] = plan['after']
            context['sha256'] = fingerprint(context['turns'])
            row['original'] = plan['after']
            row['segments'] = [dict(id=b['idx'],span=b['provenance']['origin']['span'],text=b['example'],
                correction=VERSION,processing='explicit_candidate_source_view') for b,p in results]
        invalidations = []
        if invalidate_context_reviews:
            old_contexts = {r['id']:r['turns'] for r in read_rows(source/'contexts.jsonl.gz')}
            changed_contexts = {cid for cid, ctx in by_context.items() if ctx['turns'] != old_contexts[cid]}
            # Reparsed sentences and all earlier contextual reviews in a changed
            # dialogue are pending. Unchanged token analyses are not recomputed.
            affected = set(i for i, entry in lineage.items() if entry['token_analysis']=='reparsed_unreviewed')
            affected.update(i for i, image in reviewed_images.items()
                            if image['body']['provenance']['context_id'] in changed_contexts)
            for identity in sorted(affected):
                found = db.execute('SELECT o.body,p.payload FROM occurrences o JOIN parses p ON o.key=p.key WHERE o.id=?', (identity,)).fetchone()
                if found is None:
                    continue  # A retired source segment is retained by the boundary lineage.
                b, p = map(json.loads, found)
                cid = b['provenance']['context_id']
                before = reviewed_images.get(identity, dict(body=deepcopy(b), payload=deepcopy(p)))
                nb, np = invalidate_context_review(b,p,old_contexts[cid],by_context[cid]['turns'],before)
                if identity not in lineage:
                    lineage[identity] = dict(input_ids=[identity], text=b['example'],
                        token_analysis='preserved', turn_id=identity.rsplit(':',1)[0])
                lineage[identity]['context_review_invalidated'] = True
                db.execute('INSERT OR REPLACE INTO corrections VALUES (?,?,?,?,?,?)',
                    (identity,nb['provenance']['cache_key'],nb['example'],json.dumps(nb),json.dumps(np),json.dumps(lineage[identity])))
                invalidations.append(dict(id=identity, previous_context_review=bool(before['body']['provenance'].get('context_review')),
                    preimage_sha256=fingerprint(before), context_id=cid, status='PENDING_CONTEXT_REVIEW'))
        dump(out/'invalidated-context-reviews.json', invalidations)
        db.commit()
        reconcile_duplicate_contexts(contexts, {k.rsplit(':',1)[0] for k,v in plans.items()
                                               if v['plan']['before'] != v['plan']['after']})
        for name,rows in [('contexts.jsonl.gz',contexts),('coverage.jsonl.gz',coverage)]:
            with gzip.open(out/name,'wt',encoding='utf-8') as stream:
                for r in rows:
                    write(stream,r)
        shutil.copyfile(source/'reviews.jsonl',out/'reviews.jsonl')
        dump(out/'lineage.json',lineage)
        checker = WholeBankAudit(out,out)
        try:
            counts = full.export(db,out,observer=checker.observe)
            checked = checker.finish()
        finally:
            checker.stream.close()
        # Independently verify every source view against its explicit plan.
        before_contexts = {r['id']:r['turns'] for r in read_rows(source/'contexts.jsonl.gz')}
        for cid, context in by_context.items():
            before = before_contexts[cid]
            if len(before) != len(context['turns']):
                raise ValueError('dialogue turn lost')
            for index,(a,b) in enumerate(zip(before,context['turns']),1):
                planned = plans.get(cid+':'+str(index))
                if b != (planned['plan']['after'] if planned else a):
                    raise ValueError('unplanned source-view modification')
        meta = json.loads((source/'bank/manifest.json').read_text(encoding='utf-8'))['model']
        manifest, validation = full.materialize(db,out,meta,repo/'data/wiktionary')
        conservation = verify_outputs(source,out,lineage,plans)
        for name,value in protected.items():
            if digest(Path(name)) != value:
                raise ValueError('protected file changed: '+name)
        shutil.copytree(source/'bank/review',out/'bank/review')
        for name in ('applied-turn-plans.jsonl','held-source-decisions.json','lineage.json'):
            shutil.copyfile(out/name,out/'bank/review'/name)
        manifest['hybrid_import']['candidate_source_view'] = dict(version=VERSION,
            source_files_unchanged=True,original_contexts=str(source/'contexts.jsonl.gz'),
            changed_syntax_unreviewed=True,semantic_approval=False)
        dump(out/'bank/manifest.json',manifest)
        apostrophes = json.loads((out/'apostrophe-audit.json').read_text(encoding='utf-8'))
        apostrophes.update(original_contexts_preserved=False,authoritative_source_unchanged=True,
                           original_contexts_retained_in_source_candidate=True)
        dump(out/'apostrophe-audit.json',apostrophes)
        report = dict(status='candidate_wording_complete',validation=validation,conservation=conservation,
            counts=counts,dialogues=checked['dialogues'],turns=checked['turns'],
            applied_turns=len(plans),held_turns=len(held),
            applied_wording_decisions=sum(len(v['plan']['wording']) for v in plans.values()),
            applied_boundary_decisions=sum(len(v['plan']['boundaries']) for v in plans.values()),
            reparsed_segments=parsed_count,relocated_unchanged_segments=adjusted_count,
            protected_files_unchanged=len(protected),authoritative_source_unchanged=True,
            api_calls=0,parser_inference_calls=parsed_count,published=False,
            elapsed_seconds=round(time.monotonic()-started,2))
        dump(out/'report.json',report)
        return report
    finally:
        db.close()


def verify_outputs(source, out, lineage, plans):
    def rows(root):
        for p in sorted((root/'bank/occurrences').glob('*.jsonl.gz')):
            yield from read_rows(p)
    changed_inputs = {i for r in lineage.values() for i in r['input_ids']}
    originals = {}
    old_ids = set()
    for r in rows(source):
        old_ids.add(r['idx'])
        if r['idx'] in changed_inputs:
            originals[r['idx']] = r
    unchanged = (r for r in rows(source) if r['idx'] not in changed_inputs)
    seen, consumed, counts = set(), set(), Counter()
    for after in rows(out):
        identity = after['idx']
        if identity in seen:
            raise ValueError('duplicate materialized occurrence')
        seen.add(identity)
        entry = lineage.get(identity)
        if entry is None:
            before = next(unchanged, None)
            if before is None or before['idx'] != identity:
                raise ValueError('unaddressed occurrence lost or reordered')
            if {k:v for k,v in before.items() if k!='line'} != {k:v for k,v in after.items() if k!='line'}:
                raise ValueError('unaddressed materialized occurrence changed: '+identity)
            consumed.add(identity)
            counts['untouched_occurrences'] += 1
            continue
        consumed.update(entry['input_ids'])
        if after['text'] != entry['text']:
            raise ValueError('planned source text changed')
        rev = after.get('candidate_source_revision')
        before = originals.get(identity)
        identical = (before is not None and
                     {k:v for k,v in before.items() if k!='line'} ==
                     {k:v for k,v in after.items() if k!='line'})
        # A planned turn also includes untouched neighbours. Their existing
        # revision belongs to an earlier pass and must be preserved verbatim.
        if identical:
            if entry['token_analysis'] != 'preserved' or entry['input_ids'] != [identity]:
                raise ValueError('unchanged occurrence has inconsistent lineage')
        elif after.get('candidate_context_revision') is not None:
            ctxrev = after['candidate_context_revision']
            image = ctxrev['preimage']
            if (ctxrev['semantic_approval'] is not False or not entry.get('context_review_invalidated')
                    or after.get('context_review') or image['body']['idx'] != identity):
                raise ValueError('Context revision transferred approval or lost preimage')
            # A reparse stores the old text in its source revision; otherwise
            # the contextual preimage must be byte-for-byte token equivalent.
            source_preimages = rev['preimages'] if entry['token_analysis']=='reparsed_unreviewed' else [image]
            for prior in source_preimages:
                old = originals[prior['body']['idx']]
                if old['text']!=prior['body']['example'] or old['tokens']!=prior['payload']['ledger']['tokens']:
                    raise ValueError('Context invalidation lost source evidence')
        elif rev is not None:
            if rev['semantic_approval'] is not False or rev['plan_sha256']!=entry['plan_sha256']:
                raise ValueError('source revision lost its decision binding')
            for image in rev['preimages']:
                old = originals[image['body']['idx']]
                if old['text']!=image['body']['example'] or old['tokens']!=image['payload']['ledger']['tokens']:
                    raise ValueError('previous source or analysis lost')
            if after.get('context_review'):
                raise ValueError('local wording edit granted semantic review')
        else:
            raise ValueError('changed source view without retained preimage')
        if entry['token_analysis']=='preserved' and after['tokens']!=before['tokens']:
            raise ValueError('unchanged segment syntax changed')
        counts[entry['token_analysis']] += 1
    if next(unchanged, None) is not None or consumed != old_ids:
        raise ValueError('unaccounted original occurrence')
    result = dict(status='passed',counts=dict(counts),old_occurrences=len(old_ids),new_occurrences=len(seen),
        every_original_accounted_for=True,all_unaddressed_analyses_unchanged=True,
        source_preimages_and_previous_local_decisions_preserved=True,semantic_approval=False)
    dump(out/'source-view-conservation.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('source','wording','boundaries','model','out'):
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--invalidate-context-reviews',action='store_true',
        help='Allow isolated wording edits and invalidate dependent candidate reviews; manual references remain protected.')
    print(json.dumps(run(**vars(parser.parse_args())),indent=2))
