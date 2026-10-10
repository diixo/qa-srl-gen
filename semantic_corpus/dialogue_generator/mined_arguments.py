"""Grammatical argument candidates; dependency roles are not semantic senses.

Keep the attested complement as an ordering witness. Named role values must
occur exactly once inside it; this also makes perspective changes reversible.
"""
import re

from ..ontology.models import ENTITY_LABELS
from ..template_syntax import nominal_members
from .mined_subjects import MinedSubject, realize_subject

PREPOSITIONS = frozenset('to for of in on at from by with without about into onto over under through across along around against between among upon after'.split())
ROLES = frozenset(('object', 'indirect_object', 'by_phrase', *(f'oblique_{p}' for p in PREPOSITIONS)))
DETERMINER = re.compile(r'\b(?:(?:a|an|the|this|that|these|those|my|your|some|each|every|all|both|[0-9]+)\s+)+$', re.I)


def argument_specs(record, head, voice):
    slots, parts = record['slots'], record['parts']
    head_index = next(i for i, p in enumerate(parts) if p.get('slot') == head)
    result = []
    for key, slot in slots.items():
        if slot.get('kind') != 'nominal':
            continue
        dep = slot.get('dependency')
        if dep in ('dobj', 'obj', 'iobj', 'dative') and slot.get('head_slot') == head:
            role = 'object' if dep in ('dobj', 'obj') else 'indirect_object'
        elif dep in ('pobj', 'obl', 'obl:agent'):
            role = None
        else:
            continue
        members = nominal_members(slots, key)
        if any(k not in members and s.get('head_slot') in members for k, s in slots.items()):
            return None, 'complex_argument'
        indices = [i for i, p in enumerate(parts) if p.get('slot') in members]
        if not indices or min(indices) <= head_index:
            return None, 'nonpostverbal_argument'
        first, last = min(indices), max(indices)
        np = parts[first:last + 1]
        if any(p['slot'] not in members for p in np if 'slot' in p):
            return None, 'noncontiguous_argument'
        before = parts[first - 1].get('literal', '')
        det = DETERMINER.search(before)
        if det:
            np = [{'literal': det[0]}] + np
            before = before[:det.start()]
        if role is None:
            prep = re.search(r'\b([a-z]+)\s+$', before, re.I)
            if not prep or prep[1].lower() not in PREPOSITIONS:
                return None, 'unresolved_prepositional_argument'
            prep_span = prep.span(1)
            prep = prep[1].lower()
            # Voice alone cannot attach "by the river" to "sold" in
            # "sold in the shop by the river". Require a dependency edge or
            # an exact archived marker, including for passive by phrases.
            markers = [m for g in record['predicate_groups'] if g['head_slot'] == head
                       for c in g.get('constructions', ()) if c['kind'] in ('preposition_candidate', 'particle_preposition')
                       for m in c.get('markers', ())]
            marked = any(m.get('part') == parts[first - 1].get('_source_part', first - 1)
                         and (m.get('start'), m.get('end')) == prep_span for m in markers)
            attachment = slot.get('preposition')
            if attachment is not None:
                # A known noun attachment must not be overridden by a marker.
                # A missing governor slot can be an ADP in a marked PP chain.
                attached = (attachment.get('text') == prep and
                            (attachment.get('head_slot') == head or
                             attachment.get('head_slot') is None and marked))
            else:
                attached = slot.get('head_slot') == head or marked
            if not attached:
                return None, 'unresolved_prepositional_attachment'
            role = 'by_phrase' if voice == 'passive' and prep == 'by' else 'oblique_' + prep
        if any(a['role'] == role for a in result):
            return None, 'repeated_argument_role'
        result.append({'role': role, 'slot': key, 'parts': np})
    return result, None


def realize_arguments(record, spec, bindings, person_names, entities=None):
    tail = ''.join(p.get('literal', bindings.get(p.get('slot'), '')) for p in spec['tail_parts']).strip().rstrip('.!').strip()
    arguments = []
    spans = []
    for arg in spec.get('arguments', ()):
        original = ''.join(p.get('literal', bindings.get(p.get('slot'), '')) for p in arg['parts']).strip()
        occurrences = list(re.finditer(r'(?<!\w)' + re.escape(original) + r'(?!\w)', tail))
        if len(occurrences) != 1:
            return None, 'ambiguous_argument_surface'
        span = occurrences[0].span()
        if any(span[0] < b and a < span[1] for a, b, _ in spans):
            return None, 'overlapping_arguments'
        if original.lower() in ('me', 'you'):
            np = MinedSubject('user' if original.lower() == 'me' else 'assistant', 'PERSON')
        else:
            np, reason = realize_subject(record, {'subject_slot': arg['slot'], 'subject_parts': arg['parts'],
                                                 'predicate': '', 'tail_parts': []}, bindings, person_names, entities, require_number=False)
            if reason:
                return None, 'unresolved_argument_reference' if 'reference' in reason else reason.replace('subject', 'argument')
        arguments.append((arg['role'], np.text, np.label))
        spans.append((*span, np.text))
    for start, end, text in sorted(spans, reverse=True):
        tail = tail[:start] + text + tail[end:]
    # Participant aliases can collide with another NP after normalization,
    # e.g. "you to my assistant" -> "assistant to my assistant". Reject at
    # admission, before Event validation can abort an otherwise valid job.
    normalized_spans = []
    for _, text, _ in arguments:
        occurrences = list(re.finditer(r'(?<!\w)' + re.escape(text) + r'(?!\w)', tail))
        if len(occurrences) != 1:
            return None, 'ambiguous_normalized_argument_surface'
        start, end = occurrences[0].span()
        if any(start < b and a < end for a, b in normalized_spans):
            return None, 'overlapping_normalized_arguments'
        normalized_spans.append((start, end))
    return (tail, sorted(arguments)), None


def validate_arguments(arguments, voice):
    roles = [a.role for a in arguments]
    if roles != sorted(set(roles)) or not set(roles) <= ROLES | {'complement'}:
        raise ValueError('Mined argument roles must be distinct, sorted grammatical roles')
    complement = next((a.text for a in arguments if a.role == 'complement'), '')
    spans = []
    for arg in arguments:
        if not arg.text or arg.text != arg.text.strip() or '\n' in arg.text or '\r' in arg.text:
            raise ValueError('Mined argument text must be a nonempty single line')
        if arg.role == 'complement':
            if arg.label != 'UNRESOLVED':
                raise ValueError('A mined complement keeps its unresolved semantic type')
            continue
        if arg.label not in {'UNRESOLVED', *(str(label) for label in ENTITY_LABELS)}:
            raise ValueError('Unsupported mined argument type')
        if arg.text in ('user', 'assistant') and arg.label != 'PERSON':
            raise ValueError('Dialogue participants require PERSON')
        if arg.role == 'by_phrase' and voice != 'passive':
            raise ValueError('A passive by phrase requires passive voice')
        matches = list(re.finditer(r'(?<!\w)' + re.escape(arg.text) + r'(?!\w)', complement))
        if len(matches) != 1:
            raise ValueError('A role must occur exactly once in the attested complement')
        start, end = matches[0].span()
        preposition = 'by' if arg.role == 'by_phrase' else arg.role.removeprefix('oblique_') if arg.role.startswith('oblique_') else None
        if preposition and not re.search(r'\b' + preposition + r'\s+$', complement[:start], re.I):
            raise ValueError('A prepositional role must follow its stated preposition')
        if any(start < b and a < end for a, b in spans):
            raise ValueError('Mined argument surfaces must not overlap')
        spans.append((start, end))


def render_complement(arguments, perspective):
    tail = next((a.text for a in arguments if a.role == 'complement'), '')
    changes = []
    for arg in arguments:
        if arg.role in ROLES:
            match = re.search(r'(?<!\w)' + re.escape(arg.text) + r'(?!\w)', tail)
            changes.append((*match.span(), MinedSubject(arg.text, arg.label).surface(perspective, object_case=True)))
    for start, end, text in sorted(changes, reverse=True):
        tail = tail[:start] + text + tail[end:]
    return tail
