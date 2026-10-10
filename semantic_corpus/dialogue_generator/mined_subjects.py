"""Grammatical subjects of mined clauses, distinct from semantic agents.

Keep attested noun phrases and agreement. Unknown entity types stay unresolved;
dummy weather ``it`` is a grammatical marker, not a new ontology entity type.
"""
from dataclasses import dataclass
import re

from ..ontology.models import ENTITY_LABELS
from ..template_syntax import nominal_members


WEATHER_PREDICATES = frozenset(('rain', 'snow', 'hail', 'drizzle', 'sleet', 'thunder'))
POSSIBILITY_PROFILES = (('may',), ('may', 'not'), ('might',), ('may', 'have'))
NONPERSON_SOURCE_OPERATORS = frozenset(('not', 'can', 'could', 'must', 'may', 'might',
                                      'have', 'be', 'have_to', 'start', 'finish', 'stop', 'manage', 'fail'))
_DETERMINERS = set('a an the this that these those my your some each every all both one two three four five six seven eight nine ten hundred thousand million billion'.split())
_REFERENCES = re.compile(r'\b(he|she|they|we|him|her|them|us|his|their|our|its|it|myself|yourself|himself|herself|itself|themselves|ourselves)\b', re.I)
_SOURCE_AUXILIARIES = frozenset('am is are was were be been being have has had do does did can cannot could may might must will would shall should not ca wo sha'.split())


@dataclass(frozen=True)
class MinedSubject:
    text: str
    label: str = 'UNRESOLVED'
    number: str = 'singular'

    def __post_init__(self):
        if not self.text or self.text != self.text.strip() or '\n' in self.text or '\r' in self.text:
            raise ValueError('A mined subject needs a nonempty, single-line noun phrase')
        if self.number not in ('singular', 'plural'):
            raise ValueError('A mined subject needs singular/plural agreement')
        if self.label not in {'UNRESOLVED', 'IMPERSONAL', *(str(label) for label in ENTITY_LABELS)}:
            raise ValueError('Unsupported mined subject type')
        if self.text in ('user', 'assistant') and (self.label != 'PERSON' or self.number != 'singular'):
            raise ValueError('Conversation participants are singular PERSON subjects')
        if self.label == 'IMPERSONAL' and (self.text != 'it' or self.number != 'singular'):
            raise ValueError('Impersonal weather subjects use singular it')

    def surface(self, perspective='user', *, initial=False, object_case=False):
        if perspective not in ('user', 'memory'):
            raise ValueError('Unsupported subject perspective')
        if self.text in ('user', 'assistant'):
            speaker = (self.text == 'user') == (perspective == 'user')
            text = ('me' if object_case else 'I') if speaker else 'you'
        else:
            text = self.text
            if perspective == 'memory':
                # The stored phrase is always in the user's perspective.
                text = re.sub(r'\b(my|your)\b', lambda m: {'my': 'your', 'your': 'my'}[m[0].lower()], text, flags=re.I)
        return text[:1].upper() + text[1:] if initial else text

    def agreement(self, perspective='user'):
        if self.text in ('user', 'assistant'):
            return self.surface(perspective)
        return 'they' if self.number == 'plural' else 'it'


def subject_parts(record, subject_slot, head_slot, *, scoped=False):
    """Accept contiguous simple NPs; keep unresolved clause material quoted."""
    slots, parts = record['slots'], record['parts']
    members = nominal_members(slots, subject_slot)
    if any(key not in members and slot.get('head_slot') in members for key, slot in slots.items()):
        return None, 'complex_subject'
    head_index = next(i for i, part in enumerate(parts) if part.get('slot') == head_slot)
    indices = [i for i, part in enumerate(parts) if part.get('slot') in members]
    if not indices or max(indices) > head_index:
        return None, 'inverted_subject'
    before = parts[:max(indices) + 1]
    if any(p.get('slot') not in members for p in before if 'slot' in p):
        return None, 'noncontiguous_subject'
    between = parts[max(indices) + 1:head_index]
    if scoped:
        between = []  # The caller validates the entire operator span separately.
    # Existing auxiliaries are replaced by a NEW generated scope, not interpreted
    # as source semantics. Do not silently drop adverbs or another lexical slot.
    if any((slots[p['slot']].get('pos'), slots[p['slot']].get('dependency'), slots[p['slot']].get('head_slot'))
           != ('AUX', 'aux', head_slot) for p in between if 'slot' in p):
        return None, 'unresolved_subject_context'
    expansions = {'m': 'am', 're': 'are', 's': 'is', 've': 'have', 'd': 'had', 'll': 'will'}
    for binding in record['attested_bindings']:
        auxiliaries = ''.join(p.get('literal', binding.get(p.get('slot'), '')) for p in between)
        auxiliaries = re.sub(r"n['’]t\b", ' not', auxiliaries, flags=re.I)
        auxiliaries = re.sub(r"['’](m|re|s|ve|d|ll)\b", lambda m: ' ' + expansions[m[1].lower()], auxiliaries, flags=re.I)
        if any(w.lower() not in _SOURCE_AUXILIARIES for w in auxiliaries.split()):
            return None, 'unresolved_subject_context'
    literals = ''.join(p.get('literal', ' ') for p in before)
    literals = re.sub(r"['’]s\b", '', literals)
    if (any(word.lower() not in _DETERMINERS and not word.isdigit()
            for word in re.findall(r"[\w']+", literals)) or re.search(r'[^\w\s-]', literals)):
        return None, 'unresolved_subject_context'
    return before, None


def realize_subject(record, spec, bindings, person_names, entities=None, *, require_number=True):
    """Classify only explicit dialogue deixis and pool-backed entity readings."""
    key = spec['subject_slot']
    slot = record['slots'][key]
    text = ''.join(p.get('literal', bindings.get(p.get('slot'), '')) for p in spec['subject_parts']).strip()
    if text.lower() in ('i', 'you'):
        return MinedSubject('user' if text.lower() == 'i' else 'assistant', 'PERSON'), None
    if text.lower() == 'it' and spec['predicate'] in WEATHER_PREDICATES:
        tail = ''.join(p.get('literal', bindings.get(p.get('slot'), '')) for p in spec['tail_parts']).strip('.! ')
        if not tail:
            return MinedSubject('it', 'IMPERSONAL'), None
    if _REFERENCES.search(text) or (slot.get('pos') == 'PRON' and text not in person_names):
        return None, 'unresolved_subject_reference'
    if re.search(r'\b(no|nobody|nothing|none|neither|any|anyone|anybody|anything)\b', text, re.I):
        return None, 'polarity_sensitive_subject'
    morph, tag = slot.get('morph', ''), slot.get('tag')
    if 'Number=Plur' in morph or tag in ('NNS', 'NNPS'):
        number = 'plural'
    elif 'Number=Sing' in morph or tag in ('NN', 'NNP') or text in person_names:
        number = 'singular'
    elif require_number:
        return None, 'unresolved_subject_number'
    else:
        number = 'singular'  # Non-subject role values do not carry agreement.
    # Preserve proper-name case; normalize only a known initial determiner.
    first = text.split(' ', 1)[0]
    if first.lower() in _DETERMINERS:
        text = first.lower() + text[len(first):]
    elif (spec['subject_parts'][0].get('slot') in record['slots']
          and record['slots'][spec['subject_parts'][0]['slot']].get('pos') == 'NOUN'
          and first.istitle()):
        text = text[:1].lower() + text[1:]
    label = 'PERSON' if text in person_names else 'UNRESOLVED'
    if entities is not None:
        forms = {text, text.lower()}
        # A determiner is not a different sense; a compound/adjective may be.
        bare = re.sub(r'^(?:a|an|the|my|your|this|that|these|those)\s+', '', text, flags=re.I)
        forms.update((bare, 'a ' + bare, 'an ' + bare, 'the ' + bare))
        readings = {str(label) for form in forms for entity in entities.readings(form)
                    for label in entity.labels.entity_labels}
        if len(readings) == 1:
            label = next(iter(readings))
        elif readings:
            label = 'UNRESOLVED'
    return MinedSubject(text, label, number), None
