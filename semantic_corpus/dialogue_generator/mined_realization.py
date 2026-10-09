"""Conservative realization of an observed active verb phrase.

Only the head is inflected; the full complement, particles and prepositions keep
their attested order. This layer does not assign ACTION/STATE or completion to an
unreviewed sense. `reported` records a past predication, not a completed event.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

from ..controlled_verbs import controlled_head
from ..qasrl_core.inflections import load_inflections


@lru_cache(maxsize=8)
def lexicon(path=None):
    return load_inflections(path or Path(__file__).resolve().parents[2] / 'data/wiktionary/en_verb_inflections.txt')


@dataclass(frozen=True)
class MinedLexeme:
    lemma: str
    past: str
    third: str
    gerund: str
    participle: str
    bounded: bool = False


@lru_cache(maxsize=8192)
def mined_lexeme(predicate):
    head = predicate.split(' ', 1)[0]
    if controlled_head(head) or not re.fullmatch(r"[a-z][a-z'-]*(?: [a-z]+)*", predicate):
        raise ValueError('Unsupported or manually controlled mined predicate')
    paradigm = lexicon().paradigm(head)
    if paradigm is None:
        raise ValueError(f'No dictionary paradigm for mined predicate {predicate!r}')
    return MinedLexeme(head, paradigm.past, paradigm.present_singular_3rd,
                       paradigm.present_participle, paradigm.past_participle)


def active_phrase(record, head_slot, canonical, paradigms, person_names):
    """Return a licensed first-person phrase or a reason for attested-only use.

    No parser is run here. Reject clauses needing control, voice, relative-clause
    or pronoun rewriting. These remain accessible as whole source quotations.
    """
    slots = record['slots']
    slot = slots[head_slot]
    if not paradigms or controlled_head(canonical):
        return None, 'unresolved_or_controlled_inflection'
    head = canonical.split(' ', 1)[0]
    if not re.fullmatch(r"[a-z][a-z'-]*", head):
        return None, 'nonlexical_head'
    if slot.get('dependency') != 'ROOT':
        return None, 'nonroot_predicate'
    form_key = {'VB': 'stem', 'VBP': 'stem', 'VBZ': 'present_singular_3rd',
                'VBD': 'past', 'VBN': 'past_participle', 'VBG': 'present_participle'}.get(slot.get('tag'))
    if slot.get('pos') != 'VERB' or form_key is None:
        return None, 'not_a_lexical_verb'
    if len(record['predicate_groups']) != 1:
        return None, 'embedded_or_coordinated_predicates'
    subjects = [k for k, s in slots.items() if s.get('head_slot') == head_slot and s.get('dependency') == 'nsubj']
    if len(subjects) != 1:
        return None, 'missing_active_subject'
    if any(s.get('dependency') in ('nsubjpass', 'auxpass', 'nsubj:pass', 'aux:pass') for s in slots.values()):
        return None, 'passive'
    # Validate every attested binding used with this structural template.
    for binding in record['attested_bindings']:
        subject = binding[subjects[0]]
        if subject.lower() not in ('i', 'you') and subject not in person_names:
            return None, 'unresolved_subject_animacy'
        if binding[head_slot].lower() not in {paradigm[form_key] for paradigm in paradigms}:
            return None, 'observed_form_mismatch'
    parts = record['parts']
    index = next(i for i, p in enumerate(parts) if p.get('slot') == head_slot)
    subject_index = next(i for i, p in enumerate(parts) if p.get('slot') == subjects[0])
    if subject_index > index:
        return None, 'inverted_subject'
    before = ''.join(p.get('literal', record['defaults'].get(p.get('slot'), '')) for p in parts[:index])
    after = parts[index + 1:]
    # Questions and material outside the single predication cannot be erased.
    whole = ''.join(p.get('literal', record['defaults'].get(p.get('slot'), '')) for p in parts)
    if '?' in whole or '\n' in whole or re.search(r'[.!;:]\s+\w', whole):
        return None, 'question_or_multiple_sentences'
    if re.search(r'\b(if|unless|although|when|because|who|what|which|where|why|how)\b', before, re.I):
        return None, 'nonassertive_scope'
    tail = ''.join(p.get('literal', record['defaults'].get(p.get('slot'), '')) for p in after).strip().rstrip('.!').strip()
    if re.search(r'\b(i|me|my|mine|you|your|yours|his|her|their|he|she|him|they|them|we|us|our|it|its|myself|yourself|himself|herself|itself|themselves|ourselves)\b', tail, re.I):
        return None, 'unresolved_reference'
    if re.search(r'\b(today|yesterday|tomorrow|now|ago|yet|already|ever|never|always|still|again|last|next|previous|monday|tuesday|wednesday|thursday|friday|saturday|sunday|seconds?|minutes?|hours?|days?|nights?|weeks?|months?|years?)\b', tail, re.I):
        return None, 'time_or_adverb_scope'
    if re.search(r'\b(any|anybody|anyone|anything|anywhere|either|neither|no|nobody|nothing|nowhere)\b', tail, re.I):
        return None, 'polarity_sensitive_complement'
    # Only attested noun/adjective slots and grammatical linking words may be
    # carried into a newly inflected VP. Free adverbs/negation need their scope.
    literals = ' '.join(p.get('literal', '') for p in after)
    allowed = set('a an the this that these those some any each every all both either neither no one two three four five six seven eight nine ten to for of in on at from by with without about into onto over under through across along around against between among during until since before after up down out off away back together apart forward upon and or'.split())
    if any(w.lower() not in allowed for w in re.findall(r"[A-Za-z']+", literals)):
        return None, 'unresolved_literal_scope'
    # A particle belongs to the predicate but stays in its original tail position.
    group = record['predicate_groups'][0]
    particle = [c for c in group.get('constructions', ()) if c['kind'] in ('particle', 'particle_preposition')]
    predicate = head
    if particle:
        # predicate_groups.lemma already includes particles (e.g. "take off").
        # Slice against the lexical head slot, not that compound lemma.
        construction = max(particle, key=lambda c: len(c['lemma'].split()))
        raw_head = slot['lemma']
        if not construction['lemma'].startswith(raw_head + ' '):
            return None, 'inconsistent_particle_head'
        predicate = head + construction['lemma'][len(raw_head):]
    return {'predicate': predicate, 'head_slot': head_slot, 'subject_slot': subjects[0],
            'tail_parts': after}, None


def phrase_tail(spec, bindings):
    tail = ''.join(p.get('literal', bindings.get(p.get('slot'), '')) for p in spec['tail_parts'])
    return tail.strip().rstrip('.!').strip()
