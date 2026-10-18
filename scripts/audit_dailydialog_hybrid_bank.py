"""Audit every hybrid-bank occurrence and apply narrow, unreviewed syntax rules.

Earlier contextual decisions are immutable. Rules do not infer event completion,
resolve pronouns, approve semantics, publish templates, or change source text.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import json
from itertools import zip_longest
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import process_dailydialog_hybrid as full
from scripts.benchmark_dailydialog_hybrid import dump, risk_flags, write
from scripts.import_dailydialog_sentences import digest
from scripts.revise_dailydialog_hybrid_context import SHAPE, overlay as context_overlay
from scripts.revise_dailydialog_hybrid_lexical import cached_doc
from semantic_corpus.template_bank import CorpusTemplate, fingerprint, read_rows
from semantic_corpus.template_compiler import LexicalAudit, compile_record
from semantic_corpus.template_validation import _check_rendered

VERSION = 'hybrid_whole_bank_rules_v1'
ATTRIBUTE_VERSION = 'hybrid_adjective_coordination_v1'
FINITE_3SG = 'Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin'
PARTICIPLE = 'Aspect=Perf|Tense=Past|VerbForm=Part'


def proposals(ledger, audit):
    """Syntactic candidates only; never use dictionary membership to create verbs."""
    tokens, edits = ledger['tokens'], {}
    if any(c in ledger['text'] for c in ('\ufffd', '"', '\u201c', '\u201d')):
        return []

    def change(index, rule, **fields):
        fields = {k:v for k,v in fields.items() if tokens[index][k] != v}
        if fields:
            entry = edits.setdefault(index, dict(token=index, rules=[], set={}))
            if any(k in entry['set'] and entry['set'][k] != v for k,v in fields.items()):
                raise ValueError('conflicting syntax rules')
            entry['set'].update(fields)
            if rule not in entry['rules']:
                entry['rules'].append(rule)

    for token in tokens:
        if token['pos'] != 'AUX' or token['dep'] not in {'aux', 'auxpass'}:
            continue
        head = tokens[token['head']]
        if head['pos'] not in {'VERB', 'AUX'} or head['i'] <= token['i']:
            continue
        following = token['i'] + 1
        # Intervening adverbs/negation must belong to the same predicate.
        while following < len(tokens) and tokens[following]['pos'] in {'ADV', 'PART'}:
            if tokens[following]['head'] != head['i'] or tokens[following]['dep'] not in {'advmod','neg'}:
                break
            following += 1
        if following >= len(tokens):
            continue
        next_token = tokens[following]
        shared_head = next_token['i'] == head['i'] or next_token['head'] == head['i']
        if token['text'] in {"'s", '\u2019s'}:
            if (next_token['text'].lower() == 'been' and next_token['tag'] == 'VBN'
                    and next_token['parser_lemma'] == 'be' and shared_head
                    and (next_token['i'] == head['i'] or next_token['dep'] in {'aux','auxpass'})):
                rule = 'contracted_has_been'
                change(token['i'], rule, parser_lemma='have', pos='AUX', tag='VBZ', dep='aux', morph=FINITE_3SG)
                passive = any(t['dep']=='auxpass' and t['i']!=token['i'] and t['head']==head['i'] for t in tokens)
                if not passive:
                    for child in tokens:
                        if child['head']==head['i'] and child['dep']=='nsubjpass':
                            change(child['i'], rule, dep='nsubj')
            elif next_token['i']==head['i'] and head['tag']=='VBG':
                # 'is being processed' remains passive through the other auxiliary.
                rule = 'contracted_is_progressive'
                change(token['i'], rule, parser_lemma='be', pos='AUX', tag='VBZ', dep='aux', morph=FINITE_3SG)
                if not any(t['dep']=='auxpass' and t['i']!=token['i'] and t['head']==head['i'] for t in tokens):
                    for child in tokens:
                        if child['head']==head['i'] and child['dep']=='nsubjpass':
                            change(child['i'], rule, dep='nsubj')
        elif (token['parser_lemma']=='have' and token['text'].lower() in {'have','has','had',"'ve",'having'}
                and token['dep']=='aux' and head['tag']=='VBD' and next_token['i']==head['i']
                and any(p.past_participle==head['text'].lower() for p in audit.verbs.paradigms(head['parser_lemma']))):
            change(head['i'], 'perfect_requires_participle', tag='VBN', morph=PARTICIPLE)
    return list(edits.values())


def repair(payload, audit, vocab, *, protected=False):
    if protected or payload.get('method')!='local_spacy_syntax' or payload.get('review_status')!='UNREVIEWED':
        return None
    if 'ledger' not in payload:
        return None
    edits = proposals(payload['ledger'], audit)
    invalid = [a for a in payload['record']['attributes'] if a.get('target_slot') and
               payload['record']['slots'][a['target_slot']]['kind']!='nominal']
    if not edits and not invalid:
        return None
    result = deepcopy(payload)
    previous = []
    if edits:
        ledger = deepcopy(payload['ledger'])
        for edit in edits:
            t = ledger['tokens'][edit['token']]
            previous.append(dict(token=edit['token'],rules=edit['rules'],
                                 before={k:t[k] for k in edit['set']},after=edit['set']))
            t.update(edit['set'])
        doc = cached_doc(ledger, vocab)
        record, compiled = compile_record(doc, {}, audit)
        if len(compiled['tokens']) != len(payload['ledger']['tokens']):
            raise ValueError('token count changed')
        for a,b in zip(payload['ledger']['tokens'], compiled['tokens']):
            for key in ('i','text','span','verb_lemmas','adjective_lemmas'):
                if a[key]!=b[key]:
                    raise ValueError('source or dictionary readings changed')
        rendered = CorpusTemplate(record).render()
        for edit in previous:
            token = compiled['tokens'][edit['token']]
            token.update(analysis='automatic_rule_unreviewed', previous_syntax=edit['before'])
            for key, span in rendered['spans'].items():
                if span == token['span']:
                    record['slots'][key]['analysis'] = 'automatic_rule_unreviewed'
        result.update(record=record,ledger=compiled,flags=risk_flags(doc,record,compiled),
                      variant=full.variant_signature(record,compiled))
    unresolved_targets = []
    for attribute in result['record']['attributes']:
        target = attribute.get('target_slot')
        if target and result['record']['slots'][target]['kind']!='nominal':
            unresolved_targets.append(dict(slot=attribute['slot'],previous_target=target,
                                           previous_target_kind=result['record']['slots'][target]['kind']))
            attribute['target_slot'] = None
            for item in result['ledger']['attributes']:
                if item['slot']==attribute['slot']:
                    item['target_slot'] = None
    record = result['record']
    record['template_id'] = fingerprint({k:record[k] for k in SHAPE})
    result['ledger']['template_id'] = record['template_id']
    rules = sorted({r for edit in previous for r in edit['rules']} |
                   ({'clear_non_nominal_adjective_target'} if unresolved_targets else set()))
    revision = dict(version=VERSION,rules=rules,token_changes=previous,
        unresolved_adjective_targets=unresolved_targets,previous_payload_sha256=fingerprint(payload),
        review_status='UNREVIEWED',semantic_approval=False)
    result['ledger']['automatic_revision'] = revision
    result['syntax_revision'] = VERSION
    result['flags'] = sorted(set(result['flags']) | {'automatic_rule_unreviewed'})
    rendered = CorpusTemplate(record).render()
    _check_rendered(rendered)
    if rendered['text'] != payload['ledger']['text']:
        raise ValueError('rule changed text')
    return result


def issues(payload, provenance):
    """Overlapping review reasons, not a classifier of correct/incorrect meanings."""
    found = set(payload.get('flags', ())) - {'automatic_rule_unreviewed','attribute_rule_unreviewed','context_reviewed_occurrence'}
    found.update(provenance.get('boundary_flags', ()))
    ledger, record = payload['ledger'], payload['record']
    if '\ufffd' in ledger['text']:
        found.add('damaged_source_character')
    if re.search(r'\S+@\S+|https?://|www\.', ledger['text']):
        found.add('contact_or_url_boundary')
    if ledger['text'].count('(') != ledger['text'].count(')'):
        found.add('unbalanced_parentheses')
    for attribute in record['attributes']:
        target = attribute.get('target_slot')
        if attribute.get('target_kind') == 'implicit_participant':
            found.add('adjective_implicit_participant')
        elif attribute.get('target_kind') == 'clause':
            found.add('adjective_clausal_evaluation')
        elif attribute.get('target_kind') == 'free_relative':
            found.add('adjective_free_relative')
        elif target is None:
            found.add('adjective_target_unresolved')
        elif record['slots'][target]['kind']!='nominal':
            found.add('adjective_target_non_nominal')
    for token in ledger['tokens']:
        if token.get('dep')=='conj' and token['kind']=='adjective':
            found.add('coordinated_adjective_scope')
        if token['text'] in {"'s", '\u2019s'} and token['kind']=='verb' and token['lemma'] not in {'be','have'}:
            found.add('contraction_lemma_unresolved')
        if token.get('dep') in {'vocative','npadvmod'}:
            found.add('nominal_role_or_vocative')
        if token.get('i') is not None and token['text'] in {"'s", '\u2019s'} and token['pos']=='AUX':
            head = ledger['tokens'][token['head']]
            if head['tag']=='VBN' and token['lemma']!='have':
                # A copular clause can be attached to a participial predicate
                # elsewhere in the sentence ("it's important, I've heard").
                # Only a direct auxiliary edge raises the is/has question.
                # Keep the nonlocal attachment in review, under its own reason.
                if token.get('dep') in {'aux', 'auxpass', 'aux:pass'}:
                    found.add('participle_state_or_perfect')
                else:
                    found.add('contraction_nonlocal_head')
    if provenance.get('context_review', {}).get('unit')=='pending':
        found.add('context_pending')
    return sorted(found)


def repair_adjective_coordination(payload, audit=None, vocab=None, *, protected=False):
    """Inherit only an explicit nominal attachment within an adjective conjunction.

    This identifies a syntactic target, not whether either property is asserted.
    Clausal complements, dummy/referential 'it', conflicting targets and separate
    subjects require contextual review and are deliberately left unresolved.
    """
    if protected or payload.get('method') != 'local_spacy_syntax' or payload.get('review_status') != 'UNREVIEWED':
        return None
    ledger, record = payload['ledger'], payload['record']
    if any(c in ledger['text'] for c in ('\ufffd', '"', '\u201c', '\u201d')):
        return None
    tokens = ledger['tokens']
    spans = CorpusTemplate(record).render()['spans']
    by_span = {tuple(t['span']): t for t in tokens}
    slot_tokens = {key: by_span[tuple(span)] for key, span in spans.items()}
    token_slots = {t['i']: key for key, t in slot_tokens.items()}
    attributes = {a['slot']: a for a in record['attributes']}
    children = defaultdict(list)
    for t in tokens:
        if t['head'] != t['i']:
            children[t['head']].append(t)
    changes = []
    for slot, attribute in attributes.items():
        token = slot_tokens[slot]
        if attribute.get('target_slot') is not None or token['dep'] != 'conj':
            continue
        chain, current, visited = [], token, set()
        while current['kind'] == 'adjective' and current['i'] not in visited:
            visited.add(current['i'])
            chain.append(current)
            if current['dep'] != 'conj':
                break
            current = tokens[current['head']]
        else:
            continue
        anchor = chain[-1]
        if token['parser_lemma'] == anchor['parser_lemma']:
            # Repeated comparatives such as 'more and more' express a trend,
            # not two independently coordinated properties.
            continue
        anchor_slot = token_slots[anchor['i']]
        target = attributes[anchor_slot].get('target_slot')
        if not target or record['slots'][target]['kind'] != 'nominal':
            continue
        # A separate subject/complement may make apparent adjective coordination
        # a new clause or an event property, rather than this nominal relation.
        if any(c['dep'] in {'nsubj','nsubjpass','csubj','csubjpass','expl','xcomp','ccomp'}
               for t in chain for c in children[t['i']]):
            continue
        coordinators = [c for t in chain for c in children[t['i']] if c['dep'] == 'cc']
        if not coordinators or any(c['text'].lower() not in {'and','but'} for c in coordinators):
            continue
        target_token = slot_tokens[target]
        if anchor['dep'] == 'amod':
            if anchor['head'] != target_token['i']:
                continue
        elif anchor['dep'] == 'acomp':
            governor = tokens[anchor['head']]
            subjects = [c for c in children[governor['i']] if c['dep'] in {'nsubj','nsubjpass','csubj','csubjpass','expl'}]
            if (governor['parser_lemma'] != 'be' or governor['pos'] not in {'VERB','AUX'}
                    or len(subjects) != 1 or subjects[0]['i'] != target_token['i']
                    or target_token['text'].lower() == 'it'):
                continue
        else:
            continue
        changes.append(dict(slot=slot,previous_target=None,target_slot=target,anchor_slot=anchor_slot,
            dependency_path=[t['i'] for t in chain],coordinators=[dict(text=c['text'],span=c['span']) for c in coordinators]))
    if not changes:
        return None
    result = deepcopy(payload)
    for edit in changes:
        for collection in (result['record']['attributes'],result['ledger']['attributes']):
            next(a for a in collection if a['slot']==edit['slot'])['target_slot'] = edit['target_slot']
    record = result['record']
    record['template_id'] = fingerprint({k:record[k] for k in SHAPE})
    result['ledger']['template_id'] = record['template_id']
    revision = dict(version=ATTRIBUTE_VERSION,rules=['coordinated_adjective_nominal_target'],
        attribute_changes=changes,previous_payload_sha256=fingerprint(payload),
        review_status='UNREVIEWED',semantic_approval=False)
    if 'attribute_revision' in result['ledger']:
        revision['previous_revision'] = result['ledger']['attribute_revision']
    result['ledger']['attribute_revision'] = revision
    result['syntax_revision'] = ATTRIBUTE_VERSION
    result['flags'] = sorted(set(result['flags']) | {'attribute_rule_unreviewed'})
    rendered = CorpusTemplate(record).render()
    _check_rendered(rendered)
    if rendered['text'] != ledger['text']:
        raise ValueError('attribute rule changed text')
    return result


def verify_attribute_conservation(source, out):
    """Independently compare every materialized occurrence before and after."""
    def rows(root):
        for path in sorted((root/'bank/occurrences').glob('*.jsonl.gz')):
            yield from read_rows(path)
    counts = Counter()
    for before, after in zip_longest(rows(source), rows(out)):
        if before is None or after is None:
            raise ValueError('missing materialized occurrence')
        counts['segments'] += 1
        counts['tokens'] += len(before['tokens'])
        if before == after:
            counts['identical_occurrences'] += 1
            continue
        revision = after.get('attribute_revision')
        if not revision or revision.get('version') != ATTRIBUTE_VERSION:
            raise ValueError('unexplained occurrence change')
        allowed = {'template_id','group_id','syntax_revision','attributes','attribute_revision'}
        if {k: v for k,v in before.items() if k not in allowed} != {k: v for k,v in after.items() if k not in allowed}:
            raise ValueError('source, tokens, previous repairs or contextual decision changed')
        if before.get('context_review') or before.get('review_status') != 'UNREVIEWED':
            raise ValueError('protected occurrence changed')
        attrs = deepcopy(before['attributes'])
        for change in revision['attribute_changes']:
            target = next(a for a in attrs if a['slot']==change['slot'])
            if target['target_slot'] != change['previous_target'] or change['previous_target'] is not None:
                raise ValueError('attribute preimage mismatch')
            target['target_slot'] = change['target_slot']
            counts['attribute_links_added'] += 1
        if attrs != after['attributes'] or revision['semantic_approval']:
            raise ValueError('unexplained attribute change or semantic approval')
        counts['revised_occurrences'] += 1
    for name in ('verbs.jsonl.gz','adjectives.jsonl.gz','verb_alternatives.jsonl.gz','adjective_alternatives.jsonl.gz'):
        for before,after in zip_longest(read_rows(source/'bank'/name),read_rows(out/'bank'/name)):
            if before is None or after is None or {k:v for k,v in before.items() if k!='examples'} != {k:v for k,v in after.items() if k!='examples'}:
                raise ValueError('lexical inventory changed: '+name)
    for name in ('contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'):
        if digest(source/name) != digest(out/name):
            raise ValueError('source or prior review file changed: '+name)
    result = dict(status='passed',counts=dict(counts),source_and_tokens_unchanged=True,
        previous_repairs_and_context_reviews_unchanged=True,lexical_inventories_unchanged=True,
        contexts_coverage_and_reviews_byte_identical=True,published=False)
    dump(out/'attribute-conservation-audit.json',result)
    return result


class WholeBankAudit:
    def __init__(self, source, out):
        self.out = out
        self.contexts = {r['id']:r['turns'] for r in read_rows(source/'contexts.jsonl.gz')}
        self.expected = {}
        self.turns = 0
        for row in read_rows(source/'coverage.jsonl.gz'):
            self.turns += 1
            cid, turn = row['id'].rsplit(':',1)
            if row['original'] != self.contexts[cid][int(turn)-1]:
                raise ValueError('coverage/context mismatch')
            full.check_spans(row['original'], [s['span'] for s in row['segments']])
            for segment in row['segments']:
                if segment['id'] in self.expected:
                    raise ValueError('duplicate coverage address')
                self.expected[segment['id']] = (segment['span'],segment['text'])
        if self.turns != sum(map(len,self.contexts.values())):
            raise ValueError('missing context turns')
        self.seen, self.counts, self.families, self.groups = set(), Counter(), {}, {}
        self.stream = gzip.open(out/'syntax-audit.jsonl.gz','wt',encoding='utf-8')

    def observe(self, identity, row, payload):
        provenance = row['provenance']
        origin = provenance['origin']
        if identity in self.seen or identity not in self.expected:
            raise ValueError('duplicate or unknown occurrence')
        self.seen.add(identity)
        if self.expected[identity] != (origin['span'],row['example']):
            raise ValueError('source segment changed')
        if row['example'] != row['example'].translate(full.APOSTROPHES):
            raise ValueError('noncanonical apostrophe in normalized candidate: '+identity)
        text = self.contexts[provenance['context_id']][origin['turn']-1]
        if text[slice(*origin['span'])] != provenance['original']:
            raise ValueError('source address changed')
        tokens = payload['ledger']['tokens']
        cursor = 0
        full_tokens = bool(tokens) and all('i' in t and 'head' in t for t in tokens)
        for index, token in enumerate(tokens):
            start,end = token['span']
            if not cursor <= start < end <= len(row['example']) or row['example'][start:end]!=token['text']:
                raise ValueError('invalid token spans')
            if full_tokens and (row['example'][cursor:start].strip() or token['i']!=index or not 0<=token['head']<len(tokens)):
                raise ValueError('incomplete token ledger or invalid head')
            cursor = end
        if full_tokens and row['example'][cursor:].strip():
            raise ValueError('unaccounted trailing tokens')
        self.counts['occurrences'] += 1
        self.counts['tokens'] += len(tokens)
        self.counts['ascii_apostrophes'] += row['example'].count("'")
        self.counts['full_token_ledgers' if full_tokens else 'reviewed_slot_ledgers'] += 1
        reasons = issues(payload, provenance)
        self.counts['with_review_reasons' if reasons else 'no_detected_reason_still_unreviewed'] += 1
        revisions = [payload['ledger'][key] for key in ('automatic_revision','attribute_revision','attribute_target_revision') if key in payload['ledger']]
        self.counts['automatically_revised_occurrences'] += bool(revisions)
        self.counts['preserved_context_reviews'] += 'context_review' in provenance
        key = fingerprint(dict(reasons=reasons,syntax=payload['variant']))
        write(self.stream,dict(id=identity,context_id=provenance['context_id'],group=key,
            reasons=reasons,applied_rules=sorted({rule for revision in revisions for rule in revision['rules']}),
            prior_context_review='context_review' in provenance,semantic_review='UNREVIEWED'))
        group = self.groups.setdefault(key,dict(id=key,reasons=reasons,count=0,examples=[],review_status='UNREVIEWED'))
        group['count'] += 1
        if len(group['examples'])<2:
            group['examples'].append(identity)
        for reason in reasons:
            family = self.families.setdefault(reason,dict(count=0,examples=[]))
            family['count'] += 1
            if len(family['examples'])<3:
                family['examples'].append(identity)
        if self.counts['occurrences']%20000==0:
            print(f"Audited {self.counts['occurrences']:,} occurrences",flush=True)

    def finish(self):
        self.stream.close()
        if self.seen != set(self.expected):
            raise ValueError('coverage has unprocessed segments')
        with gzip.open(self.out/'review-queue.jsonl.gz','wt',encoding='utf-8') as stream:
            for group in sorted(self.groups.values(),key=lambda g:(-g['count'],g['id'])):
                write(stream,group)
        result = dict(status='complete_automatic_audit',counts=dict(self.counts),
            dialogues=len(self.contexts),turns=self.turns,groups=len(self.groups),
            families=self.families,semantic_approval=False)
        dump(self.out/'audit-summary.json',result)
        dump(self.out/'apostrophe-audit.json',dict(status='passed',
            normalized_segments=len(self.seen),ascii_apostrophes=self.counts['ascii_apostrophes'],
            noncanonical_apostrophes=0,target="'",original_contexts_preserved=True))
        return result


def open_overlay(source, out):
    lexical, seen = source, set()
    while True:
        if lexical in seen:
            raise ValueError('cyclic revision chain')
        seen.add(lexical)
        lex_plan = json.loads((lexical/'plan.json').read_text(encoding='utf-8'))
        if 'lexical_resources' in lex_plan:
            break
        lexical = Path(lex_plan['source']).resolve()
    original = Path(lex_plan['source'])/'cache.sqlite3'
    db = context_overlay(original,lexical/'revision.sqlite3',out/'revision.sqlite3')
    db.execute('ATTACH DATABASE ? AS previous',((source/'revision.sqlite3').resolve().as_uri()+'?mode=ro',))
    db.execute('INSERT INTO corrections SELECT * FROM previous.corrections')
    db.execute('CREATE TABLE retired_occurrences (id TEXT PRIMARY KEY, replacement TEXT NOT NULL, snapshot TEXT NOT NULL)')
    if db.execute("SELECT 1 FROM previous.sqlite_master WHERE name='retired_occurrences'").fetchone():
        db.execute('INSERT INTO retired_occurrences SELECT * FROM previous.retired_occurrences')
    # Explicit sentence splits add addresses without renumbering old occurrences.
    db.execute('CREATE TABLE extra_occurrences (id TEXT PRIMARY KEY, parent_id TEXT NOT NULL, sort_order REAL NOT NULL)')
    if db.execute("SELECT 1 FROM previous.sqlite_master WHERE name='extra_occurrences'").fetchone():
        db.execute('INSERT INTO extra_occurrences SELECT * FROM previous.extra_occurrences')
    db.execute('DROP VIEW occurrences')
    db.execute('CREATE TEMP VIEW occurrences AS SELECT o.rowid,o.id,COALESCE(c.key,o.key) key,'
               'COALESCE(c.body,o.body) body FROM original.occurrences o LEFT JOIN corrections c ON c.id=o.id '
               'WHERE NOT EXISTS (SELECT 1 FROM retired_occurrences r WHERE r.id=o.id) '
               'UNION ALL SELECT e.sort_order,e.id,c.key,c.body FROM extra_occurrences e '
               'JOIN corrections c ON c.id=e.id '
               'WHERE NOT EXISTS (SELECT 1 FROM retired_occurrences r WHERE r.id=e.id)')
    db.execute('CREATE TABLE automatic_repairs (key TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    db.execute('DROP VIEW parses')
    db.execute('CREATE TEMP VIEW raw_parses AS SELECT key,text,payload FROM base_parses UNION ALL SELECT key,text,payload FROM corrections')
    if db.execute("SELECT 1 FROM previous.sqlite_master WHERE type='table' AND name='automatic_repairs'").fetchone():
        # Carry all prior repairs forward, including ones this stage will not touch.
        db.execute('INSERT INTO automatic_repairs SELECT * FROM previous.automatic_repairs')
        db.execute('CREATE TEMP VIEW source_parses AS SELECT p.key,p.text,COALESCE(a.payload,p.payload) payload '
                   'FROM raw_parses p LEFT JOIN previous.automatic_repairs a ON p.key=a.key')
    else:
        db.execute('CREATE TEMP VIEW source_parses AS SELECT * FROM raw_parses')
    db.execute('CREATE TEMP VIEW parses AS SELECT p.key,p.text,COALESCE(a.payload,p.payload) payload '
               'FROM source_parses p LEFT JOIN automatic_repairs a ON p.key=a.key')
    db.commit()
    return db, lexical, original, lex_plan


def run(source, out, repo=ROOT, *, ruleset='syntax', ledger=None):
    import spacy
    start = time.monotonic()
    source,out,repo = (p.resolve() for p in (source,out,repo))
    from scripts.dailydialog_offline_rules import VERSION as OFFLINE_VERSION, repair as offline_repair
    from scripts.dailydialog_attribute_targets import VERSION as TARGET_VERSION, repair as target_repair
    from scripts.dailydialog_quality_rules import VERSION as QUALITY_VERSION, repair as quality_repair
    version, repair_fn, revision_field = {
        'syntax': (VERSION, repair, 'automatic_revision'),
        'adjective-coordination': (ATTRIBUTE_VERSION, repair_adjective_coordination, 'attribute_revision'),
        'offline-local': (OFFLINE_VERSION, offline_repair, 'automatic_revision'),
        'attribute-targets': (TARGET_VERSION, target_repair, 'attribute_target_revision'),
        'quality-patterns': (QUALITY_VERSION, quality_repair, 'automatic_revision'),
    }[ruleset]
    if ruleset=='quality-patterns':
        if ledger is None:
            raise ValueError('Quality rules require the current warning ledger')
        ledger=Path(ledger).resolve()
        if Path(json.loads((ledger/'summary.json').read_text(encoding='utf8'))['candidate']).resolve()!=source:
            raise ValueError('Warning ledger does not describe this candidate')
    if out.exists() or out.is_relative_to(repo/'data') or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('use a new, separate quarantine output outside data/')
    protected = {str(p.relative_to(repo)):digest(p) for root in (repo/'data/mined-templates',
        repo/'data/dailydialog/filtered',repo/'data/dailydialog/review-parts')
        for p in root.rglob('*') if p.is_file()}
    # Bank files are the previous candidate snapshot, not production data.
    frozen = {str(p):digest(p) for p in source.rglob('*') if p.is_file() and
              (p.is_relative_to(source/'bank') or p.name in {'revision.sqlite3','plan.json','report.json','contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'})}
    out.mkdir(parents=True)
    db,lexical,original,lex_plan = open_overlay(source,out)
    try:
        for name,value in lex_plan['lexical_resources' if 'lexical_resources' in lex_plan else 'lexicons'].items():
            if digest(repo/name)!=value:
                raise ValueError('lexical resources changed')
        if digest(original)!=lex_plan['source_cache_sha256']:
            raise ValueError('original cache changed')
        for p in (original,lexical/'revision.sqlite3'):
            frozen[str(p)]=digest(p)
        dump(out/'plan.json',dict(version=version,ruleset=ruleset,source=str(source),protected=protected,frozen=frozen,
            code={str(p.relative_to(repo)):digest(p) for p in (Path(__file__),repo/'scripts/process_dailydialog_hybrid.py',
                repo/'scripts/revise_dailydialog_hybrid_lexical.py',repo/'semantic_corpus/template_compiler.py',
                repo/'scripts/dailydialog_offline_rules.py',repo/'scripts/dailydialog_attribute_targets.py',
                repo/'scripts/dailydialog_quality_rules.py')},
            warning_ledger=str(ledger) if ledger else None,
            api_calls=0,parser_inference_calls=0,published=False))
        frequencies = dict(db.execute('SELECT key,COUNT(*) FROM occurrences GROUP BY key'))
        reviewed = {r[0] for r in db.execute('SELECT key FROM corrections')}
        warning_keys=set()
        if ruleset=='quality-patterns':
            from scripts.finalize_dailydialog_quarantine import current_ids
            warning_ids={i for w in read_rows(ledger/'warning-decisions.jsonl.gz') for i in current_ids(w)}
            warning_keys={key for identity,key in db.execute('SELECT id,key FROM occurrences') if identity in warning_ids}
        audit,vocab = LexicalAudit(repo/'data/wiktionary'),spacy.blank('en').vocab
        counts, rules, samples = Counter(),Counter(),{}
        with gzip.open(out/'rule-changes.jsonl.gz','wt',encoding='utf-8') as stream:
            for key,text,raw in db.execute('SELECT key,text,payload FROM source_parses'):
                if key not in frequencies:
                    continue
                counts['unique_inputs_examined'] += 1
                payload = json.loads(raw)
                result = repair_fn(payload,audit,vocab,protected=key in reviewed or key in warning_keys)
                if result is None:
                    continue
                db.execute('INSERT OR REPLACE INTO automatic_repairs VALUES (?,?)',(key,json.dumps(result)))
                counts['unique_inputs_revised'] += 1
                counts['affected_occurrences'] += frequencies[key]
                revision = result['ledger'][revision_field]
                write(stream,dict(key=key,text=text,occurrences=frequencies[key],**revision))
                for rule in revision['rules']:
                    rules[rule] += frequencies[key]
                    samples.setdefault(rule,[])
                    if len(samples[rule])<4:
                        row = db.execute('SELECT id,body FROM occurrences WHERE key=? ORDER BY rowid LIMIT 1',(key,)).fetchone()
                        samples[rule].append(dict(id=row[0],text=text,provenance=json.loads(row[1])['provenance'],changes=revision))
        db.commit()
        dump(out/'rule-summary.json',dict(counts=counts,rule_occurrences=rules))
        for name in ('contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'):
            shutil.copyfile(source/name,out/name)
        checker = WholeBankAudit(source,out)
        try:
            totals = full.export(db,out,observer=checker.observe)
            coverage = checker.finish()
        finally:
            checker.stream.close()
        for cases in samples.values():
            for case in cases:
                case['context']=checker.contexts[case['provenance']['context_id']]
        dump(out/'rule-review-sample.json',samples)
        meta = json.loads((source/'bank/manifest.json').read_text(encoding='utf-8'))['model']
        manifest,validation = full.materialize(db,out,meta,repo/'data/wiktionary')
        baseline = json.loads((source/'report.json').read_text(encoding='utf-8'))['validation']
        if (validation['source_records']!=baseline['source_records'] or validation['token_spans']!=baseline['token_spans']
                or coverage['counts']['occurrences']!=baseline['source_records']):
            raise ValueError('coverage changed')
        for path,value in frozen.items():
            if digest(Path(path))!=value:
                raise ValueError('previous candidate or cache changed: '+path)
        for path,value in protected.items():
            if digest(repo/path)!=value:
                raise ValueError('protected file changed: '+path)
        if list(db.execute('SELECT * FROM corrections ORDER BY id')) != list(db.execute('SELECT * FROM previous.corrections ORDER BY id')):
            raise ValueError('context reviews changed')
        conservation = verify_attribute_conservation(source,out) if ruleset=='adjective-coordination' else None
        if ruleset in {'offline-local','quality-patterns'}:
            from scripts.dailydialog_offline_rules import verify_conservation
            conservation=verify_conservation(source,out,version=version)
        if ruleset=='attribute-targets':
            from scripts.dailydialog_attribute_targets import conserve
            conservation=conserve(source,out)
        shutil.copytree(source/'bank/review',out/'bank/review')
        for name in ('syntax-audit.jsonl.gz','review-queue.jsonl.gz','audit-summary.json','rule-changes.jsonl.gz','rule-summary.json','rule-review-sample.json'):
            shutil.copyfile(out/name,out/'bank/review'/name)
        if conservation:
            name={'offline-local':'offline-conservation-audit.json','quality-patterns':'offline-conservation-audit.json','attribute-targets':'attribute-target-conservation.json'}.get(ruleset,'attribute-conservation-audit.json')
            shutil.copyfile(out/name,out/'bank/review'/name)
        manifest['hybrid_import']['whole_bank_audit']=dict(version=version,review_status='UNREVIEWED',
            audit_file='review/syntax-audit.jsonl.gz',review_queue='review/review-queue.jsonl.gz',
            affected_occurrences=counts['affected_occurrences'],preserved_context_reviews=len(reviewed))
        dump(out/'bank/manifest.json',manifest)
        report=dict(status='whole_bank_audit_complete',ruleset=ruleset,counts=totals,rule_counts=dict(counts),
            rule_occurrences=dict(rules),validation=validation,conservation=conservation,dialogues=coverage['dialogues'],turns=coverage['turns'],
            protected_files_unchanged=len(protected),previous_bank_unchanged=True,preserved_context_reviews=len(reviewed),
            preserved_warning_keys=len(warning_keys),
            api_calls=0,parser_inference_calls=0,published=False,elapsed_seconds=round(time.monotonic()-start,2))
        dump(out/'report.json',report)
        print(json.dumps(report,indent=2),flush=True)
        return report
    finally:
        db.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'artifacts/dailydialog-hybrid-context-v2')
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/dailydialog-hybrid-bank-audit-v1')
    parser.add_argument('--ruleset',choices=('syntax','adjective-coordination','offline-local','attribute-targets','quality-patterns'),default='syntax')
    parser.add_argument('--ledger',type=Path,help='Current warning ledger; mandatory for quality-patterns.')
    args=parser.parse_args()
    run(args.source,args.out,ruleset=args.ruleset,ledger=args.ledger)
