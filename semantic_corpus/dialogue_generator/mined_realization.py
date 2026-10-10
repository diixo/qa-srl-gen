"""Conservative realization of licensed active/passive and independent clauses.

Only the head is inflected; the full complement, particles and prepositions keep
their attested order. This layer does not assign ACTION/STATE or completion to an
unreviewed sense. `reported` records a past predication, not a completed event.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import re

from ..controlled_verbs import controlled_head
from ..qasrl_core.inflections import load_inflections
from .mined_subjects import realize_subject, subject_parts, NONPERSON_SOURCE_OPERATORS
from .mined_scope import parse_scope, scope_structure
from .mined_arguments import argument_specs, realize_arguments
from .mined_clauses import independent_clauses


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
    """License every clause before exposing any event in a multi-clause source."""
    clauses, reason = independent_clauses(record)
    if reason:
        return None, reason
    if clauses is None:
        return _single_phrase(record, head_slot, canonical, paradigms, person_names)
    primary, companions, ordered = None, [], []
    for clause in clauses:
        if head_slot in clause['slots']:
            primary, reason = _single_phrase(clause, head_slot, canonical, paradigms, person_names)
            if reason:
                return None, reason
            ordered.append(primary)
        else:
            candidates = []
            for key, slot in clause['slots'].items():
                if slot.get('pos') != 'VERB':
                    continue
                lemma = slot.get('lemma', '')
                forms = [asdict(p) for p in lexicon().paradigms(lemma)]
                candidate, _ = _single_phrase(clause, key, lemma, forms, person_names)
                if candidate:
                    candidates.append(candidate)
            if len(candidates) != 1:
                return None, 'unsupported_companion_clause'
            companions.append(candidates[0])
            ordered.append(candidates[0])
    if primary is None:
        return None, 'unresolved_clause_head'
    for binding in record['attested_bindings']:
        keys = []
        for candidate in ordered:
            subject, _ = realize_subject(record, candidate, binding, person_names)
            values, _ = realize_arguments(record, candidate, binding, person_names)
            key = (candidate['predicate'], subject, values[0], candidate['voice'])
            if key in keys:
                return None, 'overlapping_companion_events'
            keys.append(key)
    primary['companions'] = companions
    primary['source_order'] = [c['head_slot'] for c in ordered]
    return primary, None


def _single_phrase(record, head_slot, canonical, paradigms, person_names):
    """License the VP and retain the original reviewed scope, if present."""
    record = {**record, 'parts': [{'_source_part': i, **p} for i, p in enumerate(record['parts'])]}
    if not paradigms or controlled_head(canonical):
        return None, 'unresolved_or_controlled_inflection'
    structure, reason = scope_structure(record, head_slot)
    if reason:
        return None, reason
    root, subject, chain = structure
    voice = 'passive' if any(s.get('dependency') in ('nsubjpass', 'auxpass', 'nsubj:pass', 'aux:pass')
                             for s in record['slots'].values()) else 'active'
    if voice == 'passive' and record['slots'][head_slot].get('tag') != 'VBN':
        return None, 'passive_without_participle'
    np, reason = subject_parts(record, subject, head_slot, scoped=True)
    if reason:
        return None, reason
    parts = record['parts']
    index = next(i for i, p in enumerate(parts) if p.get('slot') == head_slot)
    scope_parts = parts[len(np):index + 1]
    finite = next((p for p in scope_parts if p.get('slot') or p.get('literal', '').strip()), {})
    finite_tag = record['slots'].get(finite.get('slot'), {}).get('tag')
    # Validate all bindings, including source auxiliaries, before making any
    # realization available. A malformed binding must not be silently stripped.
    for binding in record['attested_bindings']:
        text = ''.join(p.get('literal', binding.get(p.get('slot'), '')) for p in scope_parts)
        scope, reason = parse_scope(text, canonical.split(' ', 1)[0], paradigms, finite_tag, voice=voice)
        if reason:
            return None, reason
    projected = {**record, 'parts': np + [{'literal': ' '}] + parts[index:],
                 'slots': {k: dict(s) for k, s in record['slots'].items() if k not in chain or k == head_slot},
                 'predicate_groups': [g for g in record['predicate_groups'] if g['head_slot'] == head_slot]}
    projected['slots'][head_slot].update(dependency='ROOT', head_slot=head_slot)
    projected['slots'][subject]['head_slot'] = head_slot
    projected['slots'][subject]['dependency'] = 'nsubj'
    # be able to has an adjective marker, not an event complement.
    prefix_slots = {p['slot'] for p in scope_parts[:-1] if 'slot' in p}
    for key in prefix_slots:
        projected['slots'].pop(key, None)
    spec, reason = _active_phrase(projected, head_slot, canonical, paradigms, person_names, voice=voice)
    if spec:
        spec.update(source_parts=scope_parts, finite_tag=finite_tag, voice=voice)
        spec['has_source_scope'] = False
        for binding in record['attested_bindings']:
            subject, _ = realize_subject(record, spec, binding, person_names)
            scope = source_scope(spec, binding, paradigms)
            spec['has_source_scope'] |= bool(scope.operators or scope.phase == 'ongoing' or scope.tense == 'future')
            if subject.label != 'PERSON' and not set(scope.operators) <= NONPERSON_SOURCE_OPERATORS:
                return None, 'unreviewed_subject_operator_combination'
        for phrase in (spec['tail_parts'], spec['subject_parts'], spec['source_parts'],
                       *(a['parts'] for a in spec['arguments'])):
            for part in phrase:
                part.pop('_source_part', None)
    return spec, reason


def source_scope(spec, bindings, paradigms):
    text = ''.join(p.get('literal', bindings.get(p.get('slot'), '')) for p in spec['source_parts'])
    scope, reason = parse_scope(text, spec['predicate'].split(' ', 1)[0], paradigms, spec['finite_tag'], voice=spec.get('voice', 'active'))
    if reason:
        raise ValueError(f'Indexed source scope no longer validates: {reason}')
    return scope


def _active_phrase(record, head_slot, canonical, paradigms, person_names, *, voice='active'):
    """Return a licensed subject + VP or a reason for attested-only use.

    No parser is run here. Scope/voice are licensed by the caller. Relative
    clauses and unresolved references remain whole source quotations.
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
    if ',' in tail or any(s.get('dependency') == 'vocative' for s in slots.values()):
        # "You grew up, Tom" addresses Tom, not the assistant. Detached phrases
        # and comma-separated complements need their own scope/role analysis.
        return None, 'vocative_or_detached_complement'
    arguments, reason = argument_specs(record, head_slot, voice)
    if reason:
        return None, reason
    # Explicit role spans resolve I/you perspective. References outside a
    # licensed noun phrase remain quotation-only.
    unbound = tail
    for arg in arguments:
        phrase = ''.join(p.get('literal', record['defaults'].get(p.get('slot'), '')) for p in arg['parts']).strip()
        unbound = unbound.replace(phrase, '')
    if re.search(r'\b(i|me|my|mine|you|your|yours|his|her|their|he|she|him|they|them|we|us|our|it|its|myself|yourself|himself|herself|itself|themselves|ourselves)\b', unbound, re.I):
        return None, 'unresolved_reference'
    if re.search(r'\b(today|yesterday|tomorrow|now|ago|yet|already|ever|never|always|still|again|last|next|previous|monday|tuesday|wednesday|thursday|friday|saturday|sunday|seconds?|minutes?|hours?|days?|nights?|weeks?|months?|years?)\b', tail, re.I):
        return None, 'time_or_adverb_scope'
    if re.search(r'\b(any|anybody|anyone|anything|anywhere|either|neither|no|nobody|nothing|nowhere)\b', tail, re.I):
        return None, 'polarity_sensitive_complement'
    # Only attested noun/adjective slots and grammatical linking words may be
    # carried into a newly inflected VP. Free adverbs/negation need their scope.
    literals = ' '.join(p.get('literal', '') for p in after)
    allowed = set('a an the this that these those my your some any each every all both either neither no one two three four five six seven eight nine ten to for of in on at from by with without about into onto over under through across along around against between among during until since before after up down out off away back together apart forward upon and or'.split())
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
    parts, reason = subject_parts(record, subjects[0], head_slot)
    if reason:
        return None, reason
    spec = {'predicate': predicate, 'head_slot': head_slot, 'subject_slot': subjects[0],
            'subject_parts': parts, 'tail_parts': after, 'arguments': arguments}
    for binding in record['attested_bindings']:
        _, reason = realize_subject(record, spec, binding, person_names)
        if reason:
            return None, reason
        _, reason = realize_arguments(record, spec, binding, person_names)
        if reason:
            return None, reason
    return spec, None


def phrase_tail(spec, bindings):
    tail = ''.join(p.get('literal', bindings.get(p.get('slot'), '')) for p in spec['tail_parts'])
    return tail.strip().rstrip('.!').strip()
