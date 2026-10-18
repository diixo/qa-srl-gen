"""Refresh only candidate property links using the conservative compiler rule."""
from collections import Counter
from copy import deepcopy
from itertools import zip_longest

from semantic_corpus.template_bank import CorpusTemplate, fingerprint, read_rows
from semantic_corpus.template_compiler import adjective_target
from semantic_corpus.template_validation import _check_rendered
from scripts.revise_dailydialog_hybrid_context import SHAPE
from scripts.revise_dailydialog_hybrid_lexical import cached_doc

VERSION='offline_attribute_targets_v2'


def repair(payload, audit, vocab, *, protected=False):
    if (protected or payload.get('method')!='local_spacy_syntax'
            or payload.get('review_status')!='UNREVIEWED'):
        return None
    ledger=payload['ledger'];record=payload['record']
    if any(c in ledger['text'] for c in ('\ufffd','"','\u201c','\u201d')):
        return None
    if not record['attributes']:
        return None
    spans=CorpusTemplate(record).render()['spans']
    by_span={tuple(t['span']):t['i'] for t in ledger['tokens']}
    slot_tokens={slot:by_span[tuple(span)] for slot,span in spans.items()}
    token_slots={i:slot for slot,i in slot_tokens.items()}
    doc=cached_doc(ledger,vocab)
    edits=[]
    for attr in record['attributes']:
        target=adjective_target(doc[slot_tokens[attr['slot']]])
        new=token_slots.get(target.i) if target is not None else None
        if new!=attr['target_slot']:
            edits.append(dict(slot=attr['slot'],previous_target=attr['target_slot'],target_slot=new))
    if not edits:return None
    result=deepcopy(payload)
    for edit in edits:
        for attrs in (result['record']['attributes'], result['ledger']['attributes']):
            next(a for a in attrs if a['slot']==edit['slot'])['target_slot']=edit['target_slot']
    result['record']['template_id']=fingerprint({k:result['record'][k] for k in SHAPE})
    result['ledger']['template_id']=result['record']['template_id']
    result['ledger']['attribute_target_revision']=dict(version=VERSION,
        rules=['syntactically_licensed_nominal_property_target'],attribute_changes=edits,
        preimage=deepcopy(payload),preimage_sha256=fingerprint(payload),
        review_status='UNREVIEWED',semantic_approval=False)
    result['syntax_revision']=VERSION
    rendered=CorpusTemplate(result['record']).render();_check_rendered(rendered)
    if rendered['text']!=ledger['text']:raise ValueError('Attribute revision changed source')
    return result


def conserve(source,out):
    from scripts.import_dailydialog_sentences import digest
    from scripts.benchmark_dailydialog_hybrid import dump
    def rows(root):
        for p in sorted((root/'bank/occurrences').glob('*.jsonl.gz')):yield from read_rows(p)
    counts=Counter()
    allowed={'template_id','group_id','syntax_revision','attributes','attribute_target_revision'}
    for a,b in zip_longest(rows(source),rows(out)):
        if a is None or b is None or a['idx']!=b['idx']:raise ValueError('Occurrence lost')
        counts['segments']+=1;counts['tokens']+=len(a['tokens'])
        if a==b:
            counts['unchanged_occurrences']+=1;continue
        rev=b['attribute_target_revision']
        if (a.get('context_review') or a['review_status']!='UNREVIEWED'
                or rev['version']!=VERSION or rev['semantic_approval'] is not False
                or fingerprint(rev['preimage'])!=rev['preimage_sha256']
                or rev['preimage']['ledger']['attributes']!=a['attributes']):
            raise ValueError('Missing original attribute evidence')
        if {k:v for k,v in a.items() if k not in allowed}!={k:v for k,v in b.items() if k not in allowed}:
            raise ValueError('Source, tokens or old reviews changed')
        attrs=deepcopy(a['attributes'])
        for e in rev['attribute_changes']:
            attr=next(x for x in attrs if x['slot']==e['slot'])
            if attr['target_slot']!=e['previous_target']:raise ValueError('Stale property edit')
            attr['target_slot']=e['target_slot'];counts['links_changed']+=1
        if attrs!=b['attributes']:raise ValueError('Unexpected attribute edit')
        counts['revised_occurrences']+=1
    for name in ('verbs.jsonl.gz','adjectives.jsonl.gz','verb_alternatives.jsonl.gz','adjective_alternatives.jsonl.gz'):
        for a,b in zip_longest(read_rows(source/'bank'/name),read_rows(out/'bank'/name)):
            if a is None or b is None or {k:v for k,v in a.items() if k!='examples'}!={k:v for k,v in b.items() if k!='examples'}:
                raise ValueError('Lexical inventory changed')
    for name in ('contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'):
        if digest(source/name)!=digest(out/name):raise ValueError('Original context or review changed')
    result=dict(status='passed',counts=dict(counts),source_and_tokens_unchanged=True,
                previous_reviews_preserved=True,lexical_inventories_unchanged=True)
    dump(out/'attribute-target-conservation.json',result)
    return result
