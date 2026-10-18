"""Conservative local syntax repairs; never interpret or execute API prose.

These rules do NOT approve semantics or publication. Exact original payloads
are retained. Context-reviewed records, ambiguous addresses and quotations are
excluded. A match changes a syntactic candidate, not a fact about an event.
"""
from copy import deepcopy

from semantic_corpus.template_bank import CorpusTemplate, fingerprint
from semantic_corpus.template_compiler import compile_record
from semantic_corpus.template_validation import _check_rendered
from scripts.revise_dailydialog_hybrid_lexical import cached_doc
from scripts.process_dailydialog_hybrid import variant_signature
from scripts.benchmark_dailydialog_hybrid import risk_flags

VERSION = 'offline_local_syntax_v2'
ADDRESS_WORDS = {'sir', 'madam', "ma'am", 'doctor', 'dad', 'daddy', 'mom', 'mum',
                 'mother', 'father', 'honey', 'darling', 'sweetheart', 'waiter',
                 'miss', 'gentlemen', 'ladies'}
TITLES = {'mr.', 'mr', 'mrs.', 'mrs', 'ms.', 'ms', 'dr.', 'dr', 'miss'}
OPENERS = {'hey', 'hello', 'hi', 'oh', 'well', 'yes', 'no', 'ok', 'okay'}
EVALUATIONS = {'nice', 'glad', 'pleased', 'happy', 'delighted', 'sorry'}


def proposals(payload):
    if payload.get('method') != 'local_spacy_syntax' or payload.get('review_status') != 'UNREVIEWED':
        return []
    ledger = payload.get('ledger', {})
    ts = ledger.get('tokens', [])
    if (not ts or any(c in ledger.get('text', '') for c in ('"', '\u201c', '\u201d', '\ufffd'))
            or 'source_revision' in ledger or 'attribute_revision' in ledger):
        return []
    roots = [t for t in ts if t['dep'] == 'ROOT']
    if len(roots) != 1:
        return []
    root = roots[0]
    if root['pos'] not in {'VERB', 'AUX'} or root['tag'] == 'MD':
        return []
    children = {i: [t for t in ts if t['head'] == i and t['i'] != i] for i in range(len(ts))}
    # Direct address needs an independently parsed interlocutor clause. This
    # excludes topical "John, he ..." and noun appositions after another NP.
    subject = [t for t in children[root['i']] if t['dep'] in {'nsubj', 'nsubjpass'}
               and t['text'].lower() in {'i', 'we', 'you'}]
    edits = []
    for t in ts:
        i, word = t['i'], t['text'].lower()
        if subject and t['dep'] in {'npadvmod', 'appos', 'dep', 'advmod', 'nmod'}:
            dependents = children[i]
            if any(c['dep'] not in {'compound', 'amod', 'punct'} for c in dependents):
                continue
            phrase = sorted([i] + [c['i'] for c in dependents if c['dep'] != 'punct'])
            if phrase != list(range(phrase[0], i+1)):
                continue
            lo, hi = phrase[0], i+1
            common = word in ADDRESS_WORDS
            titled = any(ts[k]['text'].lower() in TITLES for k in phrase[:-1])
            person = any(e['label'] == 'PERSON' and e['span'][0] <= t['span'][0]
                         and t['span'][1] <= e['span'][1] for e in payload.get('entities', []))
            if not (common or titled or person):
                continue
            initial = (hi < len(ts) and ts[hi]['text'] == ',' and i < min(s['i'] for s in subject)
                       and all(x['pos'] == 'PUNCT' or x['text'].lower() in OPENERS for x in ts[:lo]))
            final = (lo > 0 and ts[lo-1]['text'] == ',' and i > root['i']
                     and all(x['pos'] == 'PUNCT' for x in ts[hi:])
                     and (common or (lo >= 2 and ts[lo-2]['pos'] not in {'NOUN', 'PROPN', 'PRON'})))
            if not (initial or final):
                continue
            # "my dear John" and "young man" need contextual interpretation.
            if any(ts[k]['text'].lower() not in TITLES for k in phrase[:-1]):
                continue
            fields = {'dep': 'vocative', 'head': root['i']}
            if common and t['pos'] in {'NOUN', 'PROPN'} and t['tag'] in {'NN', 'NNP'}:
                fields.update(pos='NOUN', tag='NN', parser_lemma=word, morph='Number=Sing')
            fields = {k:v for k,v in fields.items() if t[k] != v}
            if fields:
                edits.append(dict(token=i, rules=['delimited_interlocutor_address'], set=fields))
        # Evaluative copular adjective + bare infinitive with its own "to";
        # excludes purpose adjuncts on lexical verbs and subordinate subjects.
        if t['dep'] == 'advcl' and t['tag'] == 'VB':
            parent = ts[t['head']]
            if (parent['pos'] == 'ADJ' and parent['parser_lemma'] in EVALUATIONS
                    and parent['dep'] == 'acomp' and parent['head'] == root['i']
                    and root['parser_lemma'] == 'be'
                    and any(c['text'].lower() == 'to' and c['dep'] == 'aux' for c in children[i])
                    and not any(c['dep'] in {'nsubj','nsubjpass','mark'} for c in children[i])):
                edits.append(dict(token=i, rules=['copular_evaluation_infinitive'], set={'dep':'xcomp'}))
    return edits


def repair(payload, audit, vocab, *, protected=False):
    if protected:
        return None
    edits = proposals(payload)
    return apply_edits(payload, edits, audit, vocab)


def apply_edits(payload, edits, audit, vocab, *, version=VERSION):
    """Recompile explicit grammar edits while retaining their exact preimage."""
    if not edits:
        return None
    ledger = deepcopy(payload['ledger'])
    for edit in edits:
        ledger['tokens'][edit['token']].update(edit['set'])
    doc = cached_doc(ledger, vocab)
    record, compiled = compile_record(doc, {}, audit)
    for old, new in zip(payload['ledger']['tokens'], compiled['tokens']):
        if any(old[k] != new[k] for k in ('i','text','span','verb_lemmas','adjective_lemmas')):
            raise ValueError('local syntax repair changed token or lexical evidence')
    if len(compiled['tokens']) != len(ledger['tokens']):
        raise ValueError('local syntax repair changed token count')
    for edit in edits:
        t = compiled['tokens'][edit['token']]
        if any(t[k] != v for k,v in edit['set'].items()):
            raise ValueError('compiler lost a local syntax repair')
    rendered = CorpusTemplate(record).render()
    _check_rendered(rendered)
    if rendered['text'] != ledger['text']:
        raise ValueError('local syntax repair changed text')
    for key, value in payload['ledger'].items():
        if key not in compiled and key != 'automatic_revision':
            compiled[key] = deepcopy(value)
    compiled['automatic_revision'] = dict(version=version, rules=sorted({r for e in edits for r in e['rules']}),
        edits=edits, preimage_sha256=fingerprint(payload), preimage=deepcopy(payload),
        semantic_approval=False, review_status='UNREVIEWED',
        note='Local grammar rules only; no API prose executed, contextual meaning inferred or admission granted.')
    return dict(payload, record=record, ledger=compiled, flags=risk_flags(doc,record,compiled),
                variant=variant_signature(record,compiled), syntax_revision=version)


def verify_conservation(source, out, *, version=VERSION):
    from collections import Counter
    from itertools import zip_longest
    from semantic_corpus.template_bank import read_rows
    from scripts.benchmark_dailydialog_hybrid import dump
    from scripts.import_dailydialog_sentences import digest

    def rows(root):
        for p in sorted((root/'bank/occurrences').glob('*.jsonl.gz')):
            yield from read_rows(p)
    counts=Counter()
    allowed={'template_id','group_id','syntax_revision','attributes','automatic_revision',
             'tokens','bindings','predicate_templates','predicate_groups','sentence_template','analysis_status',
             'imperative_candidates'}
    if version=='hybrid_quality_patterns_v1':
        # Particle repair can expose a separate time PP (move in on Wednesday).
        # This is a compiler diagnostic, never a reviewed temporal assertion.
        allowed.add('temporal_adjunct_candidates')
    syntax={'pos','tag','dep','head','parser_lemma','morph'}
    for before,after in zip_longest(rows(source),rows(out)):
        if before is None or after is None or before['idx']!=after['idx']:
            raise ValueError('local rules lost or reordered an occurrence')
        counts['segments']+=1;counts['tokens']+=len(before['tokens'])
        if before==after:
            counts['identical_occurrences']+=1
            continue
        revision=after.get('automatic_revision',{})
        if (before.get('context_review') or before.get('review_status')!='UNREVIEWED'
                or revision.get('version')!=version or revision.get('semantic_approval') is not False):
            raise ValueError('local rules changed a protected occurrence or lack evidence')
        if {k:v for k,v in before.items() if k not in allowed}!={k:v for k,v in after.items() if k not in allowed}:
            raise ValueError('local rules changed source or previous metadata')
        preimage=revision['preimage']
        if (fingerprint(preimage)!=revision['preimage_sha256']
                or preimage['ledger']['tokens']!=before['tokens']):
            raise ValueError('local rule preimage does not match the previous bank')
        edits={e['token']:e['set'] for e in revision['edits']}
        if len(edits)!=len(revision['edits']):
            raise ValueError('duplicate local rule edit')
        for i,(a,b) in enumerate(zip_longest(before['tokens'],after['tokens'])):
            if a is None or b is None:
                raise ValueError('local rules changed tokenization')
            expected=dict(a);expected.update(edits.get(i,{}))
            for k in syntax|{'i','text','span','verb_lemmas','adjective_lemmas'}:
                if expected[k]!=b[k]:
                    raise ValueError('unexpected token change: '+before['idx']+':'+str(i)+':'+k)
        if before.get('temporal_adjunct_candidates')!=after.get('temporal_adjunct_candidates'):
            if version!='hybrid_quality_patterns_v1':
                raise ValueError('Unexpected temporal diagnostic revision')
            for candidate in after.get('temporal_adjunct_candidates',[]):
                if candidate.get('status')!='UNREVIEWED':
                    raise ValueError('Grammar rule promoted a temporal diagnostic')
                for field in ('span','marker','predicate_span'):
                    a,b=candidate[field]
                    if not 0<=a<b<=len(after['text']):
                        raise ValueError('Temporal diagnostic lost its source span')
            counts['updated_unreviewed_temporal_diagnostics']+=1
        counts['revised_occurrences']+=1
    for name in ('contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'):
        if digest(source/name)!=digest(out/name):
            raise ValueError('local rules changed context or existing reviews')
    result=dict(status='passed',counts=dict(counts),source_and_tokens_unchanged=True,
        previous_preimages_retained=True,context_reviews_unchanged=True,published=False)
    dump(out/'offline-conservation-audit.json',result)
    return result
