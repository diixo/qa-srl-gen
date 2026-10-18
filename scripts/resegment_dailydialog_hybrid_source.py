"""Apply model-authored source edits and sentence splits, retaining all preimages.

This is an executor of explicit decisions, not a parser or spelling corrector.
New records remain quarantined; admission is a separate, restricted operation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
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
from scripts.revise_dailydialog_hybrid_lexical import cached_doc
from semantic_corpus.template_bank import CorpusTemplate, fingerprint, read_rows
from semantic_corpus.template_compiler import LexicalAudit, compile_record
from semantic_corpus.template_usage import constrain_record
from semantic_corpus.template_validation import _check_rendered

VERSION = 'reviewed_source_segmentation_v6'
FIELDS = {'text', 'parser_lemma', 'pos', 'tag', 'dep', 'head', 'morph'}


def input_ids(review):
    identities = review.get('input_ids', [review['id']])
    if (not identities or identities[0] != review['id']
            or len(set(identities)) != len(identities)):
        raise ValueError('invalid reviewed input identities')
    return identities


def insertion_positions(db, last, count):
    """Keep a later split between its parent and the next existing occurrence."""
    if not count:
        return []
    following=db.execute('SELECT MIN(rowid) FROM occurrences WHERE rowid>?',(last,)).fetchone()[0]
    upper=following if following is not None else last+1
    positions=[last+(upper-last)*(i+1)/(count+1) for i in range(count)]
    if len(set(positions))!=count or any(not last<p<upper for p in positions):
        raise ValueError('not enough ordering precision for another sentence split')
    return positions


def expand_reference(item, expansion):
    """Add explicitly reviewed punctuation/heads to a borrowed slot ledger.

    The original reference is never edited. Every existing slot and its syntax
    must survive exactly; only uncovered punctuation can become new tokens.
    """
    b,p=item['body'],item['payload']
    if (p.get('method')!='reviewed_whole_dialogue_reuse' or p.get('review_status')!='REVIEWED_REUSE'
            or b['provenance'].get('context_review') or not b['provenance'].get('reused_from')
            or b['provenance']['reused_from']==b['idx'] or not expansion.get('note')
            or expansion.get('reused_from')!=b['provenance']['reused_from']
            or expansion.get('preimage')!=fingerprint(item)):
        raise ValueError('invalid explicit borrowed-reference expansion')
    tokens=deepcopy(expansion['tokens']);text=b['example'];cursor=0
    old=p['ledger']['tokens'];by_span={tuple(t['span']):t for t in old}
    if len(by_span)!=len(old): raise ValueError('duplicate reference slot spans')
    seen=set();slot_indices={}
    for i,t in enumerate(tokens):
        if set(t)!=FIELDS|{'i','span'} or t['i']!=i:
            raise ValueError('reference expansion requires complete explicit tokens')
        a,z=t['span']
        if (not cursor<=a<z<=len(text) or text[a:z]!=t['text'] or text[cursor:a].strip()
                or type(t['head']) is not int or not 0<=t['head']<len(tokens)):
            raise ValueError('reference expansion changed source coverage')
        cursor=z;before=by_span.get((a,z))
        if before:
            if any(t[k]!=before[k] for k in FIELDS-{'head'}):
                raise ValueError('reference expansion changed reviewed slot syntax')
            seen.add((a,z));slot_indices[before['slot']]=i
        elif t['pos']!='PUNCT' or any(c.isalnum() or c.isspace() for c in t['text']):
            raise ValueError('reference expansion may only add missing punctuation')
    if seen!=set(by_span) or text[cursor:].strip() or sum(t['dep']=='ROOT' for t in tokens)!=1:
        raise ValueError('reference expansion lost reviewed slots or sentence root')
    for before in old:
        slot=p['record']['slots'][before['slot']]
        if tokens[slot_indices[before['slot']]]['head']!=slot_indices.get(slot['head_slot']):
            raise ValueError('reference expansion changed reviewed slot head')
    for i in range(len(tokens)):
        current=i;visited=set()
        while tokens[current]['head']!=current:
            if current in visited: raise ValueError('cyclic reference expansion')
            visited.add(current);current=tokens[current]['head']
        if tokens[current]['dep']!='ROOT': raise ValueError('invalid reference root')
    return tokens


def combine_segments(inputs, context, reference_expansions=None):
    """Build a temporary token ledger; retain every original segment in its preimage."""
    reference_expansions=reference_expansions or {}
    if len(inputs) == 1:
        if reference_expansions: raise ValueError('reference expansion requires a grouped input')
        return inputs[0]['body'], inputs[0]['payload']
    body = deepcopy(inputs[0]['body'])
    origin = body['provenance']['origin']
    turn = context[origin['turn']-1]
    start = cursor = origin['span'][0]
    tokens, entities = [], []
    used_expansions=set()
    for number, item in enumerate(inputs):
        b, p = item['body'], item['payload']
        o = b['provenance']['origin']
        if (any(o[k] != origin[k] for k in ('split', 'dialogue', 'turn'))
                or b['provenance']['context_id'] != body['provenance']['context_id']
                or o['segment'] != origin['segment']+number):
            raise ValueError('joined inputs must be consecutive segments of one turn')
        a, z = o['span']
        if (not cursor <= a < z <= len(turn) or turn[cursor:a].strip()
                or turn[a:z] != b['example'] or p['ledger']['text'] != b['example']):
            raise ValueError('joined source coverage mismatch')
        if b['idx'] in reference_expansions:
            source_tokens=expand_reference(item,reference_expansions[b['idx']])
            used_expansions.add(b['idx'])
        else:
            if p.get('method') != 'local_spacy_syntax' or p.get('review_status') != 'UNREVIEWED':
                raise ValueError('reviewed reference cannot be joined without explicit expansion')
            source_tokens=p['ledger']['tokens']
        offset, shift = len(tokens), a-start
        for token in deepcopy(source_tokens):
            token['i'] += offset
            token['head'] += offset
            token['span'] = [v+shift for v in token['span']]
            tokens.append(token)
        for entity in deepcopy(p['entities']):
            entity['span'] = [v+shift for v in entity['span']]
            entities.append(entity)
        cursor = z
    if used_expansions!=set(reference_expansions): raise ValueError('unused reference expansion')
    body['example'] = turn[start:cursor]
    origin['span'] = [start,cursor]
    payload = dict(method='local_spacy_syntax', review_status='UNREVIEWED', entities=entities,
                   ledger=dict(text=body['example'],tokens=tokens,input_segments=deepcopy(inputs)))
    if reference_expansions: payload['ledger']['reference_expansions']=deepcopy(reference_expansions)
    return body, payload


def compile_replacement(body, payload, context, review, audit, vocab):
    """Every consumed/dropped/inserted token and source character is accounted for."""
    if (review['method'] != 'direct_assistant_source_review' or review['id'] != body['idx']
            or review['preimage'] != fingerprint(dict(body=body, payload=payload))
            or review['context_sha256'] != fingerprint(context)):
        raise ValueError('stale source review preimage')
    if payload.get('method') != 'local_spacy_syntax' or payload.get('review_status') != 'UNREVIEWED':
        raise ValueError('reviewed reference cannot be rewritten')
    if len(input_ids(review)) > 1:
        if [r['body']['idx'] for r in payload['ledger'].get('input_segments', [])] != input_ids(review):
            raise ValueError('grouped segmentation requires complete preimages')
        if review['before_turn'] != review['after_turn']:
            # A typo in an internal sentence moves every later character address.
            # Allow that only with an explicit review of the COMPLETE turn, with
            # all sentence IDs retained and no borrowed reference being rewritten.
            if (review.get('whole_turn_source_edit') is not True
                    or body['provenance']['origin']['span'] != [0, len(context[body['provenance']['origin']['turn']-1])]
                    or [o['id'] for o in review['outputs']] != input_ids(review)
                    or review.get('reference_expansions')):
                raise ValueError('grouped source edit requires a complete turn and preserved sentence IDs')
        rebuilt=combine_segments(payload['ledger']['input_segments'],context,review.get('reference_expansions'))
        if rebuilt!=(body,payload): raise ValueError('grouped input differs from original evidence')
    elif review.get('reference_expansions'):
        raise ValueError('reference expansion requires a grouped input')
    origin = body['provenance']['origin']
    old_turn = context[origin['turn']-1]
    start, end = origin['span']
    new_turn = review['after_turn']
    # Internal segments can be revised only without moving later source addresses.
    outputs = review['outputs']
    new_end = outputs[-1]['span'][1] if outputs else -1
    if (old_turn != review['before_turn']
            or old_turn[start:end] != body['example'] or new_turn[:start] != old_turn[:start]):
        raise ValueError('source edit crosses its reviewed segment')
    if (new_turn[new_end:] != old_turn[end:]
            or (end < len(old_turn) and (new_end != end or len(outputs) != 1))):
        raise ValueError('source edit moves or changes an unreviewed suffix')
    if new_turn != new_turn.translate(full.APOSTROPHES):
        raise ValueError('noncanonical apostrophe')
    new_context = deepcopy(context)
    new_context[origin['turn']-1] = new_turn
    full.check_spans(new_turn[start:new_end], [[a-start, b-start] for a,b in (o['span'] for o in outputs)])
    previous = payload['ledger']['tokens']
    used = Counter(review['dropped_tokens'])
    for output in outputs:
        for token in output['tokens']:
            used.update(token['from'])
    if used != Counter(range(len(previous))):
        raise ValueError('source token was lost, duplicated or invented')
    if (review['dropped_tokens'] or any(not t['from'] for o in outputs for t in o['tokens'])) and not review.get('tokenization_note'):
        raise ValueError('token insertion/deletion requires explanation')
    compiled_outputs = []
    for number, output in enumerate(outputs):
        identity = ':'.join(body['idx'].split(':')[:-1]+[str(origin['segment']+number)])
        if output['id'] != identity or output['unit'] not in {'sentence', 'fragment'}:
            raise ValueError('invalid replacement identity or unit')
        meaning = output['meaning']
        if (not meaning.get('speech_act') or not meaning.get('event_status')
                or not isinstance(meaning.get('content'), dict)
                or type(meaning.get('completed_action_asserted')) is not bool
                or not isinstance(meaning.get('unknown'), list)):
            raise ValueError('incomplete contextual meaning')
        if not output['evidence_turns'] or any(type(t) is not int or not 1 <= t <= len(context) for t in output['evidence_turns']):
            raise ValueError('invalid context evidence')
        a,b = output['span']
        text = new_turn[a:b]
        mapping = {old:i for i,spec in enumerate(output['tokens']) for old in spec['from']}
        tokens, cursor = [], 0
        for i,spec in enumerate(output['tokens']):
            edits = spec.get('set', {})
            if set(edits)-FIELDS:
                raise ValueError('unexpected token field')
            if spec['from']:
                token = {k:deepcopy(previous[spec['from'][0]][k]) for k in FIELDS}
                if 'head' not in edits:
                    if token['head'] not in mapping:
                        raise ValueError('dependency crosses the reviewed sentence boundary')
                    token['head'] = mapping[token['head']]
                token.update(edits)
            else:
                if set(edits) != FIELDS:
                    raise ValueError('inserted token needs a complete explicit analysis')
                token = deepcopy(edits)
            a = text.find(token['text'], cursor)
            if not token['text'] or a < cursor or text[cursor:a].strip():
                raise ValueError('token text does not cover reviewed source')
            cursor = a+len(token['text'])
            token.update(i=i, span=[a,cursor])
            tokens.append(token)
        if text[cursor:].strip() or sum(t['dep']=='ROOT' for t in tokens) != 1:
            raise ValueError('incomplete sentence tokens or multiple roots')
        for i in range(len(tokens)):
            current, seen = i, set()
            while True:
                if type(current) is not int or current in seen or not 0 <= current < len(tokens):
                    raise ValueError('invalid or cyclic dependency')
                seen.add(current)
                parent = tokens[current]['head']
                if parent == current:
                    if tokens[current]['dep'] != 'ROOT':
                        raise ValueError('self head requires ROOT')
                    break
                current = parent
        doc = cached_doc(dict(text=text, tokens=tokens), vocab)
        record, ledger = compile_record(doc, {}, audit)
        if len(ledger['tokens']) != len(tokens) or any(
                any(t[k] != observed[k] for k in FIELDS | {'span', 'i'})
                for t,observed in zip(tokens,ledger['tokens'])):
            raise ValueError('compiled analysis differs from explicit review')
        updated = deepcopy(body)
        updated['idx'], updated['example'] = identity, text
        provenance = updated['provenance']
        if 'context_review' in provenance:
            provenance.setdefault('context_review_history', []).append(provenance.pop('context_review'))
        provenance['origin'].update(segment=origin['segment']+number, span=output['span'])
        syntax_changed = (len(tokens) != len(previous) or any(
            any(t[k] != old[k] for k in FIELDS | {'span', 'i'}) for t,old in zip(tokens,previous)))
        decision = {k:deepcopy(v) for k,v in output.items() if k != 'tokens'}
        decision.update(method=review['method'], source_revision_sha256=fingerprint(review),
                        token_edits=deepcopy(output['tokens']) if syntax_changed else [])
        key = fingerprint([VERSION,review,identity])
        provenance.update(original=text, previous_original=body['example'], cache_key=key,
            previous_cache_key=body['provenance']['cache_key'], boundary_flags=output['boundary_flags'],
            correction=VERSION, context_review=dict(rule=VERSION, status='REVIEWED_OCCURRENCE',
                method=review['method'], unit=output['unit'], meaning=meaning, note=output['note'],
                evidence_turns=output['evidence_turns'], syntax_corrected=syntax_changed,
                source_sha256=fingerprint(text), context_sha256=fingerprint(new_context),
                template_semantics_approved=False, unseen_group_members_approved=False))
        rendered = constrain_record(record, updated, new_context, decision)
        ledger.update(template_id=record['template_id'], attributes=deepcopy(record['attributes']),
            source_revision=dict(version=VERSION, parent_id=body['idx'],
                preimage=dict(body=deepcopy(body),payload=deepcopy(payload)),
                review=deepcopy(review), semantic_approval=False))
        _check_rendered(rendered)
        if rendered['text'] != text:
            raise ValueError('source roundtrip changed')
        # Unchanged NER spans only; removed or retokenized evidence remains in preimage.
        entities=[]
        for entity in payload['entities']:
            ea,eb = entity['span']
            absolute_a,absolute_b = start+ea,start+eb
            a,b = output['span']
            if a <= absolute_a < absolute_b <= b and old_turn[:absolute_b] == new_turn[:absolute_b]:
                entities.append({**deepcopy(entity),'span':[absolute_a-a,absolute_b-a]})
        if 'entities' in output:
            # Explicit NER decisions can relocate unchanged evidence after a
            # source edit, or remove a false label. The old list remains in the
            # preimage. Neither a label nor a span grants template admission.
            if not output.get('entity_review_note') or not isinstance(output['entities'], list):
                raise ValueError('entity replacement requires an explicit review note and list')
            entities = deepcopy(output['entities'])
            starts, ends = {t['span'][0] for t in tokens}, {t['span'][1] for t in tokens}
            for entity in entities:
                if (not isinstance(entity, dict)
                        or set(entity) != {'text', 'label', 'span', 'review_status'}
                        or entity['review_status'] != 'UNREVIEWED'
                        or not isinstance(entity['label'], str) or not entity['label']
                        or not isinstance(entity['span'], list) or len(entity['span']) != 2
                        or any(type(v) is not int for v in entity['span'])):
                    raise ValueError('invalid explicit entity evidence')
                ea, eb = entity['span']
                if (not 0 <= ea < eb <= len(text) or ea not in starts or eb not in ends
                        or text[ea:eb] != entity['text']):
                    raise ValueError('explicit entity does not match reviewed source tokens')
        result=dict(record=record, ledger=ledger, entities=entities, flags=sorted(set(risk_flags(doc,record,ledger))|{'context_reviewed_occurrence'}),
            variant=full.variant_signature(record,ledger), method='local_spacy_syntax',
            review_status='UNREVIEWED', syntax_revision=VERSION)
        compiled_outputs.append((updated,result,decision))
    return compiled_outputs, new_context


def source_context_key(review):
    """Bind an explicitly inspected source version when the candidate differs."""
    snapshot=review.get('source_context_snapshot')
    if snapshot is None:
        return review['context_sha256']
    if (not isinstance(snapshot,dict) or not isinstance(snapshot.get('turns'),list) or not snapshot['turns']
            or any(not isinstance(t,str) for t in snapshot['turns'])
            or snapshot.get('sha256')!=fingerprint(snapshot['turns'])
            or not snapshot.get('note','').strip()):
        raise ValueError('invalid explicit source context snapshot')
    return snapshot['sha256']


def source_updates(repo, reviews):
    """Prepare all writes before changing files; preserve untouched lines byte-for-byte."""
    changes = {}
    manifest_path=repo/'data/dailydialog/review-parts/manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    by_dialogue={r['id'].rsplit(':',2)[0]:r for r in reviews if r['before_turn']!=r['after_turn']}
    if len(by_dialogue) != sum(r['before_turn']!=r['after_turn'] for r in reviews):
        raise ValueError('multiple source edits per dialogue need explicit composition')
    expected={}
    edited_contexts = {source_context_key(r) for r in by_dialogue.values()}
    duplicate_sources = {key: set() for key in edited_contexts}
    for split in ('train','val','test'):
        path=repo/f'data/dailydialog/filtered/dailydialog-{split}.jsonl'
        old=path.read_bytes()
        if digest(path)!=manifest['sources'][split]: raise ValueError('stale parts source digest')
        lines=old.splitlines(keepends=True)
        for i,line in enumerate(lines,1):
            identity=f'{split}:{i}'
            row=json.loads(line)
            key = fingerprint(row)
            if key in duplicate_sources:
                duplicate_sources[key].add(identity)
            if identity not in by_dialogue: continue
            r=by_dialogue[identity];turn=int(r['id'].split(':')[2])-1
            if row[turn]!=r['before_turn'] or fingerprint(row)!=source_context_key(r):
                raise ValueError('filtered source preimage mismatch: '+identity)
            row[turn]=r['after_turn'];expected[identity]=row
            lines[i-1]=(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8')
        new=b''.join(lines)
        if new!=old:
            changes[path]=(old,new)
            manifest['sources'][split]=hashlib.sha256(new).hexdigest()
    for key, members in duplicate_sources.items():
        declared = {cid for cid, r in by_dialogue.items() if source_context_key(r) == key}
        if members != declared:
            raise ValueError('source edit omitted an identical dialogue: '+','.join(sorted(members-declared)))
    found=set()
    for entry in manifest['parts']:
        path=manifest_path.parent/entry['file']
        if digest(path)!=entry['sha256']: raise ValueError('stale part digest')
        old=path.read_bytes();lines=old.splitlines(keepends=True)
        for i,line in enumerate(lines):
            row=json.loads(line);identity=f"{row['split']}:{row['dialogue']}"
            if identity not in expected: continue
            if identity in found: raise ValueError('duplicate part address')
            found.add(identity);r=by_dialogue[identity]
            if fingerprint(row['turns'])!=source_context_key(r): raise ValueError('part/source mismatch')
            row['turns']=expected[identity]
            lines[i]=(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8')
        new=b''.join(lines)
        if new!=old:
            changes[path]=(old,new)
            entry['sha256']=hashlib.sha256(new).hexdigest()
    if found!=set(by_dialogue) or set(expected)!=set(by_dialogue): raise ValueError('missing source address')
    if changes:
        changes[manifest_path]=(manifest_path.read_bytes(),(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    return changes


def verify_conservation(source,out,reviews):
    targets={r['id']:r for r in reviews}
    grouped={i:r for r in reviews for i in input_ids(r)[1:]}
    expected_inputs={};observed_inputs={}
    def occurrences(root):
        for p in sorted((root/'bank/occurrences').glob('*.jsonl.gz')): yield from read_rows(p)
    afters=iter(occurrences(out));counts=Counter();line=0
    for before in occurrences(source):
        counts['old_segments']+=1;counts['old_tokens']+=len(before['tokens'])
        if before['idx'] in grouped:
            observed_inputs[before['idx']]=before
            counts['grouped_input_segments']+=1
            if before['idx'] not in {o['id'] for o in grouped[before['idx']]['outputs']}:
                counts['joined_input_segments']+=1
            continue
        review=targets.get(before['idx'])
        if review and len(input_ids(review))>1: observed_inputs[before['idx']]=before
        for output in review['outputs'] if review else [None]:
            after=next(afters,None);line+=1
            if after is None or after['line']!=line: raise ValueError('missing occurrence or invalid output order')
            counts['new_segments']+=1;counts['new_tokens']+=len(after['tokens'])
            if not review:
                if {k:v for k,v in before.items() if k!='line'}!={k:v for k,v in after.items() if k!='line'}:
                    raise ValueError('unaddressed occurrence changed: '+before['idx'])
                counts['unchanged_analyses']+=1
            else:
                revision=after['source_revision']
                if (revision['review']!=review or fingerprint(revision['preimage'])!=review['preimage']
                        or revision['semantic_approval'] or after['idx']!=output['id']
                        or after['text']!=review['after_turn'][slice(*output['span'])]):
                    raise ValueError('source revision lost its preimage or output')
                counts['revised_outputs']+=1
                if len(input_ids(review))>1:
                    for item in revision['preimage']['payload']['ledger']['input_segments']:
                        expected_inputs[item['body']['idx']]=item
    if next(afters,None) is not None: raise ValueError('unexpected output occurrence')
    if set(expected_inputs)!=set(observed_inputs): raise ValueError('joined input preimage lost')
    for identity,item in expected_inputs.items():
        before=observed_inputs[identity]
        if (before['text']!=item['body']['example'] or before['tokens']!=item['payload']['ledger']['tokens']):
            raise ValueError('joined input evidence changed: '+identity)
    expected_delta=sum(sum(len(o['tokens']) for o in r['outputs'])-len({i for o in r['outputs'] for t in o['tokens'] for i in t['from']}|set(r['dropped_tokens'])) for r in reviews)
    expansion_delta=sum(len(e['tokens'])-len(expected_inputs[identity]['payload']['ledger']['tokens'])
        for r in reviews for identity,e in r.get('reference_expansions',{}).items())
    expected_delta+=expansion_delta
    if counts['new_tokens']-counts['old_tokens']!=expected_delta: raise ValueError('unexplained token count change')
    result=dict(status='passed',counts=dict(counts),explicit_token_delta=expected_delta,
                reference_expansion_token_delta=expansion_delta,previous_preimages_retained=True)
    dump(out/'source-conservation-audit.json',result)
    return result


def validate_duplicate_context(contexts, identity, review, *, repo=None, coordinated_source_edit=False):
    """Require explicit duplicate evidence and coordinated shared source edits."""
    context = contexts[identity]
    duplicate = context.get('duplicate_of')
    if not duplicate:
        if review.get('duplicate_context_of'):
            raise ValueError('review claims an unrecorded duplicate context')
        return
    parent=contexts.get(duplicate)
    if parent is None and repo is not None and review.get('duplicate_context_reference'):
        # The canonical dialogue may be one of the preserved 510 references,
        # intentionally absent from the remaining-corpus candidate. Validate
        # its explicitly embedded snapshot against the authoritative source.
        reference=review['duplicate_context_reference']
        split,number=duplicate.split(':')
        if split not in {'train','val','test'} or not number.isdigit() or int(number)<1:
            raise ValueError('invalid duplicate source address')
        path=Path(repo)/f'data/dailydialog/filtered/dailydialog-{split}.jsonl'
        if (reference.get('id')!=duplicate or reference.get('source_file_sha256')!=digest(path)
                or reference.get('sha256')!=fingerprint(reference.get('turns'))):
            raise ValueError('stale duplicate reference snapshot')
        parent=next((dict(turns=row) for index,row in enumerate(read_rows(path),1) if index==int(number)),None)
        if parent is None or parent['turns']!=reference['turns']:
            raise ValueError('duplicate reference does not match source')
    if (duplicate == identity or review.get('duplicate_context_of') != duplicate
            or parent is None or context['turns'] != parent['turns']):
        raise ValueError('duplicate context requires explicit review of matching full context')
    if review['before_turn'] != review['after_turn'] and not coordinated_source_edit:
        raise ValueError('duplicate source edit requires coordinated context updates')


def validate_review_contexts(contexts, reviews, *, repo=None):
    """Check all original contexts before any candidate context is replaced.

    Matching text does not authorize propagating a parse. Every duplicate needs
    its own explicit review and must receive the same source-text change.
    source_updates additionally checks duplicates outside the candidate.
    """
    changes = {}
    for review in reviews:
        if review['before_turn'] == review['after_turn']:
            continue
        cid = review['id'].rsplit(':', 2)[0]
        if cid in changes:
            raise ValueError('multiple source edits per dialogue need explicit composition')
        changes[cid] = review
    by_hash = {}
    for cid, context in contexts.items():
        by_hash.setdefault(fingerprint(context['turns']), set()).add(cid)
    for cid, review in changes.items():
        if review['context_sha256'] != fingerprint(contexts[cid]['turns']):
            raise ValueError('stale duplicate/source context')
        members = by_hash[review['context_sha256']]
        declared = review.get('source_duplicate_group')
        if len(members) > 1 or declared is not None:
            if (declared != sorted(members) or not members <= changes.keys()
                    or any(changes[other].get('source_duplicate_group') != declared
                           or changes[other]['id'].split(':')[2] != review['id'].split(':')[2]
                           or changes[other]['before_turn'] != review['before_turn']
                           or changes[other]['after_turn'] != review['after_turn'] for other in members)):
                raise ValueError('duplicate source edits require an explicit matching review for every dialogue')
    for review in reviews:
        cid = review['id'].rsplit(':', 2)[0]
        validate_duplicate_context(contexts, cid, review, repo=repo,
                                   coordinated_source_edit=cid in changes)


def run(source,reviews_path,out,repo=ROOT):
    import spacy
    started=time.monotonic();source,out,repo=map(Path.resolve,(source,out,repo))
    if out.exists() or out.is_relative_to(repo/'data') or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('use a new separate quarantine directory')
    reviews=list(read_rows(reviews_path))
    if not reviews or len({r['id'] for r in reviews})!=len(reviews): raise ValueError('empty/duplicate source reviews')
    inputs=[i for r in reviews for i in input_ids(r)]
    if len(inputs)!=len(set(inputs)): raise ValueError('overlapping reviewed input groups')
    out.mkdir(parents=True)
    protected={str(p):digest(p) for folder in (repo/'data/mined-templates',repo/'data/dailydialog/filtered',repo/'data/dailydialog/review-parts') for p in folder.rglob('*') if p.is_file()}
    frozen={str(p):digest(p) for p in source.rglob('*') if p.is_file() and (p.is_relative_to(source/'bank') or p.name in {'plan.json','report.json','revision.sqlite3','contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'})}
    db,lexical,original,lex_plan=open_overlay(source,out)
    try:
        if digest(original)!=lex_plan['source_cache_sha256']: raise ValueError('parse cache changed')
        for name,value in lex_plan['lexical_resources'].items():
            if digest(repo/name)!=value: raise ValueError('dictionary changed')
        frozen.update({str(p):digest(p) for p in (original,lexical/'revision.sqlite3')})
        dump(out/'plan.json',dict(version=VERSION,source=str(source),reviews=str(reviews_path.resolve()),review_sha256=digest(reviews_path),protected=protected,frozen=frozen,
            code={str(p):digest(p) for p in (Path(__file__),ROOT/'scripts/audit_dailydialog_hybrid_bank.py',ROOT/'semantic_corpus/template_compiler.py',ROOT/'semantic_corpus/template_usage.py')},published=False))
        contexts=list(read_rows(source/'contexts.jsonl.gz'));by_context={r['id']:r for r in contexts}
        validate_review_contexts(by_context, reviews, repo=repo)
        vocab,audit=spacy.blank('en').vocab,LexicalAudit(repo/'data/wiktionary')
        replacements={};decisions=[]
        for review in reviews:
            originals=[];orders=[]
            for identity in input_ids(review):
                found=db.execute('SELECT o.rowid,o.body,p.payload FROM occurrences o JOIN parses p ON p.key=o.key WHERE o.id=?',(identity,)).fetchone()
                if not found: raise ValueError('unknown reviewed source address: '+identity)
                rowid,b,p=found;orders.append(rowid)
                originals.append(dict(body=json.loads(b),payload=json.loads(p)))
            rowid=orders[0];cid=originals[0]['body']['provenance']['context_id']
            body,payload=combine_segments(originals,by_context[cid]['turns'],review.get('reference_expansions'))
            values,new_context=compile_replacement(body,payload,by_context[cid]['turns'],review,audit,vocab)
            positions=iter(insertion_positions(db,orders[-1],max(0,len(values)-len(originals))))
            by_context[cid].update(turns=new_context,sha256=fingerprint(new_context))
            replacements[review['id']]=values
            output_ids={b['idx'] for b,p,r in values}
            for original in originals[1:]:
                identity=original['body']['idx'];replacements[identity]=[]
                if identity not in output_ids:
                    db.execute('INSERT INTO retired_occurrences VALUES (?,?,?)',(identity,review['id'],json.dumps(original)))
            for index,(b,p,r) in enumerate(values):
                existing=db.execute('SELECT 1 FROM occurrences WHERE id=?',(b['idx'],)).fetchone()
                if existing and b['idx'] not in input_ids(review): raise ValueError('new address collides')
                if not existing and db.execute('SELECT 1 FROM retired_occurrences WHERE id=?',(b['idx'],)).fetchone():
                    raise ValueError('new address collides with retired history')
                db.execute('INSERT OR REPLACE INTO corrections VALUES (?,?,?,?,?,?)',(b['idx'],b['provenance']['cache_key'],b['example'],json.dumps(b),json.dumps(p),json.dumps(r)))
                if not existing:
                    db.execute('INSERT INTO extra_occurrences VALUES (?,?,?)',(b['idx'],review['id'],next(positions)))
                decisions.append(r)
        # Protect every published reference, even if it shares a source dialogue.
        changed_turns={r['id'].rsplit(':',1)[0] for r in reviews if r['before_turn']!=r['after_turn']}
        for ref in full.load_reference(repo/'data/mined-templates'):
            if f"{ref['split']}:{ref['dialogue']}:{ref['turn']}" in changed_turns: raise ValueError('published reference source changed')
        changes=source_updates(repo,reviews)
        for path,(before,after) in changes.items():
            relative=path.relative_to(repo)
            backup=out/'source-before'/relative;backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(before)
            prepared=out/'source-after'/relative;prepared.parent.mkdir(parents=True,exist_ok=True);prepared.write_bytes(after)
        # All checks precede writes; file bytes are checked again against their preimages.
        if any(path.read_bytes()!=before for path,(before,_) in changes.items()): raise ValueError('source changed while preparing')
        for path,(_,after) in changes.items(): path.write_bytes(after)
        dump(out/'source-files.json',{str(p.relative_to(repo)):dict(before_sha256=protected[str(p)],after_sha256=digest(p)) for p in changes})
        protected.update({str(p):digest(p) for p in changes})
        db.commit()
        with gzip.open(out/'contexts.jsonl.gz','wt',encoding='utf-8') as stream:
            for row in contexts: write(stream,row)
        with gzip.open(out/'coverage.jsonl.gz','wt',encoding='utf-8') as stream:
            previous_ids={r['id']:input_ids(r) for r in reviews}
            for row in read_rows(source/'coverage.jsonl.gz'):
                cid,turn=row['id'].rsplit(':',1);row['original']=by_context[cid]['turns'][int(turn)-1]
                segments=[]
                for s in row['segments']:
                    if s['id'] not in replacements: segments.append(s);continue
                    for b,p,r in replacements[s['id']]:
                        segments.append({**s,'id':b['idx'],'span':b['provenance']['origin']['span'],'text':b['example'],
                                         'correction':VERSION,'processing':'reviewed_source_revision','previous_segment_ids':previous_ids[s['id']]})
                row['segments']=segments;full.check_spans(row['original'],[s['span'] for s in segments]);write(stream,row)
        prior={r['id']:r for r in read_rows(source/'reviews.jsonl')}
        produced={r['id'] for r in decisions}
        for identity in set(inputs)-produced: prior.pop(identity,None)
        prior.update({r['id']:r for r in decisions})
        for name,rows in (('reviews.jsonl',prior.values()),('admission-reviews.jsonl',decisions)):
            with (out/name).open('w',encoding='utf-8') as stream:
                for r in rows: write(stream,r)
        checker=WholeBankAudit(out,out)
        try: counts=full.export(db,out,observer=checker.observe);coverage=checker.finish()
        finally: checker.stream.close()
        apostrophes=json.loads((out/'apostrophe-audit.json').read_text(encoding='utf-8'))
        apostrophes.update(original_contexts_preserved=not changes,previous_contexts_retained_in_source_candidate=True,
                           source_corrections_explicit=True)
        dump(out/'apostrophe-audit.json',apostrophes)
        old=json.loads((source/'bank/manifest.json').read_text(encoding='utf-8'))
        manifest,validation=full.materialize(db,out,old['model'],repo/'data/wiktionary')
        conservation=verify_conservation(source,out,reviews)
        for path,value in {**protected,**frozen}.items():
            if digest(Path(path))!=value: raise ValueError('protected input changed: '+path)
        ids=set(inputs)
        if [r for r in db.execute('SELECT * FROM corrections ORDER BY id') if r[0] not in produced] != [r for r in db.execute('SELECT * FROM previous.corrections ORDER BY id') if r[0] not in produced]: raise ValueError('unrelated review changed')
        previous_repairs = (list(db.execute('SELECT * FROM previous.automatic_repairs ORDER BY key'))
                            if db.execute("SELECT 1 FROM previous.sqlite_master WHERE name='automatic_repairs'").fetchone() else [])
        if list(db.execute('SELECT * FROM automatic_repairs ORDER BY key'))!=previous_repairs: raise ValueError('automatic history changed')
        previous_retired=(list(db.execute('SELECT * FROM previous.retired_occurrences ORDER BY id'))
            if db.execute("SELECT 1 FROM previous.sqlite_master WHERE name='retired_occurrences'").fetchone() else [])
        if [r for r in db.execute('SELECT * FROM retired_occurrences ORDER BY id') if r[0] not in ids]!=previous_retired:
            raise ValueError('retired history changed')
        shutil.copytree(source/'bank/review',out/'bank/review')
        shutil.copyfile(source/'reviews.jsonl',out/'bank/review/context-reviews-before-segmentation.jsonl')
        shutil.copyfile(reviews_path,out/'bank/review/source-segmentation-decisions.jsonl')
        for name in ('reviews.jsonl','admission-reviews.jsonl','source-files.json','syntax-audit.jsonl.gz','review-queue.jsonl.gz','audit-summary.json','apostrophe-audit.json','source-conservation-audit.json'):
            shutil.copyfile(out/name,out/'bank/review'/name)
        manifest['hybrid_import']={**old['hybrid_import'],**manifest['hybrid_import'],
            'source_segmentation_revision':dict(version=VERSION,input_occurrences=len(inputs),output_occurrences=len(decisions),review_file='review/source-segmentation-decisions.jsonl'),
            'context_revision':dict(review_file='review/reviews.jsonl',occurrences=len(prior),syntax_corrections=sum(bool(r['token_edits']) for r in prior.values()))}
        dump(out/'bank/manifest.json',manifest)
        report=dict(status='source_segmentation_complete',counts=counts,validation=validation,conservation=conservation,
            dialogues=coverage['dialogues'],turns=coverage['turns'],previous_bank_unchanged=True,protected_files=len(protected),
            source_files_updated=len(changes),api_calls=0,parser_inference_calls=0,published=False,elapsed_seconds=round(time.monotonic()-started,2))
        dump(out/'report.json',report);print(json.dumps(report,indent=2),flush=True)
        return report
    finally: db.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--reviews',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.reviews,args.out)
