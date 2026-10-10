"""Auditable lexical checks and narrow parser repairs over immutable templates.

Dictionary membership is morphology evidence, not a word-sense annotation.
Repairs never change source words, dependencies, participants or attachments.
Ambiguous/missing analyses stay quotations; the raw inventory stays enumerable.
"""
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re

from ..controlled_verbs import controlled_head

POLICY_PATH = Path(__file__).with_name('mined_morphology.json')
FORM_KEYS = ('stem', 'present_singular_3rd', 'present_participle', 'past', 'past_participle')
TAG_FORM = {'VB': 'stem', 'VBP': 'stem', 'VBZ': 'present_singular_3rd',
            'VBG': 'present_participle', 'VBD': 'past', 'VBN': 'past_participle'}

REASONS = {
    'consistent': 'Observed lexical form agrees with the parser and dictionary.',
    'normalized_lemma': 'A unique dictionary analysis repairs the parser lemma.',
    'repaired_finite_tag': 'A unique inflected form repairs a finite ROOT/conj tag with an explicit subject and no auxiliaries.',
    'nonlexical_token': 'Punctuation, broken encoding or a token fragment is not a realizable verb head.',
    'auxiliary_or_modal': 'This slot is an auxiliary/modal, not a standalone lexical event.',
    'not_a_lexical_verb': 'The parser did not assign a lexical verb POS and supported verb tag.',
    'missing_dictionary_paradigm': 'No complete dictionary paradigm matches every observed binding; absence does not prove a word is invalid.',
    'ambiguous_lemma': 'More than one dictionary head matches the observed form and tag.',
    'finite_homograph': 'A bare finite form can also be another verb\'s past; the parser tag alone cannot resolve this homograph.',
    'ambiguous_generation_paradigm': 'Competing paradigms have no reviewed generation preference.',
    'generation_paradigm_mismatch': 'The source dictionary and the event renderer select different generation forms.',
    'dictionary_form_conflict': 'The observed forms disagree with the parser lemma/tag; repair is not uniquely licensed.',
    'sense_ambiguous_inflection': 'The same surface spans senses with different conjugations; morphology alone cannot choose the meaning.',
    'controlled_alias': 'The resolved head belongs to the manually controlled verb policy.',
}


@lru_cache(maxsize=1)
def _policy_resource():
    raw = POLICY_PATH.read_bytes()
    data = json.loads(raw)
    for head, row in data['generation'].items():
        if any(len(forms) != 5 or forms[0] != head for forms in [row['forms'], *row['source_variants']]) or not row['reason']:
            raise ValueError(f'Invalid reviewed morphology for {head}')
    return data, sha256(raw).hexdigest()


def policy():
    return _policy_resource()[0]


def policy_fingerprint():
    return _policy_resource()[1]


def refresh_policy():
    """Refresh once per bank/job, not once per token or dictionary lookup."""
    if sha256(POLICY_PATH.read_bytes()).hexdigest() == policy_fingerprint():
        return False
    _policy_resource.cache_clear()
    _paradigms_for.cache_clear()
    policy()  # Validate the new file before constructing an index.
    return True


def paradigms_for(head, dictionary):
    """Return accepted source forms and a single generation paradigm, or none.

    An unseen multiple-paradigm entry requires a review, never a first-row vote.
    """
    return _paradigms_for(head, tuple(dict.fromkeys(dictionary.paradigms(head))))


@lru_cache(maxsize=16384)
def _paradigms_for(head, found):
    preference = policy()['generation'].get(head)
    if preference:
        chosen = dict(zip(FORM_KEYS, preference['forms']))
        # Reviewed choices are pinned to evidence in the supplied dictionary.
        if chosen not in [asdict(p) for p in found]:
            return (), None
        accepted = [chosen] + [dict(zip(FORM_KEYS, forms)) for forms in preference['source_variants']
                              if dict(zip(FORM_KEYS, forms)) in [asdict(p) for p in found]]
        return accepted, chosen
    return [asdict(p) for p in found], asdict(found[0]) if len(found) == 1 else None


def normalize_inventory(row, dictionary):
    """Inventory-level hint only. Do not silently discard unmatched surfaces."""
    lemma = row['lemma']
    observed = Counter()
    for surface, count in row.get('forms', {}).items():
        observed[surface.lower()] += count
    candidates = dictionary.paradigms(lemma)
    source = 'dictionary'
    if not candidates:
        matches = [{p.stem for p, _ in dictionary.analyses(w)} for w in observed]
        heads = set.intersection(*matches) if matches else set()
        candidates = dictionary.paradigms(next(iter(heads))) if len(heads) == 1 else ()
        source = 'recovered_from_observed_forms' if candidates else 'observed_only'
    canonical = candidates[0].stem if candidates else lemma
    return {'lemma': lemma, 'canonical': canonical, 'forms': dict(observed),
            'paradigms': [asdict(p) for p in candidates], 'inflection_source': source,
            'controlled': controlled_head(canonical)}


def review_slot(record, key, dictionary):
    slot = record['slots'][key]
    lemma, tag = slot.get('lemma', ''), slot.get('tag')
    observed = sorted({b.get(key, '').lower() for b in record['attested_bindings']})
    result = {'raw_lemma': lemma, 'canonical': lemma, 'original_tag': tag, 'tag': tag,
              'observed_forms': observed, 'candidate_heads': [], 'changes': [],
              'status': 'quotation_only', 'reason': None}

    def reject(reason):
        result['reason'] = reason
        return result

    if not re.fullmatch(r"[a-z][a-z'-]*", lemma) or not observed or any(
            not re.fullmatch(r"[a-z][a-z'-]*", w) for w in observed):
        return reject('nonlexical_token')
    if slot.get('pos') == 'AUX' or tag == 'MD' or slot.get('dependency') in ('aux', 'auxpass', 'aux:pass'):
        return reject('auxiliary_or_modal')
    if slot.get('pos') != 'VERB' or tag not in TAG_FORM:
        return reject('not_a_lexical_verb')

    # Every binding must fit the same head and form. Do not use majority votes.
    reverse = [{p.stem for p, _ in dictionary.analyses(w)} for w in observed]
    heads = set.intersection(*reverse) if reverse else set()
    if dictionary.paradigms(lemma):
        heads.add(lemma)
    result['candidate_heads'] = sorted(heads)

    def matches(head, form):
        accepted, _ = paradigms_for(head, dictionary)
        return bool(accepted) and all(w in {p[form] for p in accepted} for w in observed)

    matching = {h for h in heads if matches(h, TAG_FORM[tag])}
    # The parser's lemma is evidence when its tagged surface actually agrees.
    # Homographic reverse dictionary hits alone do not override that evidence.
    if lemma in matching:
        canonical = lemma
    elif len(matching) == 1:
        canonical = next(iter(matching))
    elif len(matching) > 1:
        return reject('ambiguous_lemma')
    else:
        # Narrow repair: explicit finite subject, root/conj, no auxiliary.
        # Never turn a participle or an infinitive under an operator into past.
        finite = tag in ('VBP', 'VBZ', 'VBD') and slot.get('dependency') in ('ROOT', 'conj')
        finite &= any(s.get('head_slot') == key and s.get('dependency') == 'nsubj'
                      for s in record['slots'].values())
        finite &= not any(s.get('head_slot') == key and s.get('dependency') in ('aux', 'auxpass', 'aux:pass')
                          for s in record['slots'].values())
        repairs = {(h, new_tag) for h in heads for new_tag in ('VBD',)
                   if finite and new_tag != tag and matches(h, TAG_FORM[new_tag])}
        if len(repairs) != 1:
            return reject('dictionary_form_conflict' if heads else 'missing_dictionary_paradigm')
        canonical, tag = repairs.pop()
        result['tag'] = tag
        result['changes'].append({'field': 'tag', 'before': slot['tag'], 'after': tag})
    result['canonical'] = canonical
    if canonical != lemma:
        result['changes'].append({'field': 'lemma', 'before': lemma, 'after': canonical})
    if controlled_head(canonical):
        return reject('controlled_alias')
    if tag in ('VB', 'VBP') and slot.get('dependency') in ('ROOT', 'conj') and not any(
            s.get('head_slot') == key and s.get('dependency') in ('aux', 'auxpass', 'aux:pass')
            for s in record['slots'].values()):
        if any(p.stem != canonical and p.past == word for word in observed
               for p, _ in dictionary.analyses(word)):
            return reject('finite_homograph')
    accepted, generation = paradigms_for(canonical, dictionary)
    if generation is None:
        return reject('ambiguous_generation_paradigm')
    guard = policy()['sense_guards'].get(canonical)
    if guard and not set(observed) <= set(guard['unambiguous_source_forms']):
        return reject('sense_ambiguous_inflection')
    result.update(status='normalized' if result['changes'] else 'consistent',
                  reason='repaired_finite_tag' if tag != slot.get('tag') else
                         'normalized_lemma' if canonical != lemma else 'consistent',
                  source_paradigms=accepted, generation_paradigm=generation)
    return result


def review_template(record, dictionary):
    """Return a copied lexical projection and evidence, leaving archives intact."""
    reviews = {k: review_slot(record, k, dictionary) for k, s in record['slots'].items()
               if s.get('kind') == 'verb'}
    slots = {k: dict(s) for k, s in record['slots'].items()}
    for key, decision in reviews.items():
        if decision['status'] != 'quotation_only':
            slots[key].update(lemma=decision['canonical'], tag=decision['tag'])
            if decision['tag'] != decision['original_tag']:
                slots[key]['morph'] = 'Tense=Past|VerbForm=Fin'
    groups = []
    for group in record['predicate_groups']:
        group = {**group, 'constructions': [dict(c) for c in group.get('constructions', ())]}
        decision = reviews.get(group['head_slot'])
        if decision and decision['status'] != 'quotation_only':
            raw, canonical = decision['raw_lemma'], decision['canonical']
            for item in (group, *group['constructions']):
                value = item.get('lemma', '')
                if value == raw or value.startswith(raw + ' '):
                    item['lemma'] = canonical + value[len(raw):]
        groups.append(group)
    return {**record, 'slots': slots, 'predicate_groups': groups}, reviews


def reason_category(reason):
    if reason in REASONS:
        return 'lexical_review'
    if any(w in reason for w in ('reference', 'subject_number', 'argument_surface', 'attachment', 'overlapping')):
        return 'reference_or_attachment'
    return 'construction_rule'
