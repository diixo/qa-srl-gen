"""Recognize reviewed, same-subject operator chains in archived templates.

Dependency paths license subject control; surface grammar determines tense and
negation scope. Neither dictionary forms nor an xcomp edge alone license control.
Anything outside this grammar remains an attested quotation.
"""
from dataclasses import dataclass
from itertools import product
import re

from .verb_operators import MODALS, OPERATORS, validate_chain

CONTROL = frozenset('want need plan intend decide try manage fail start finish stop'.split())
FORMS = {'base': 'stem', 'present': 'stem', 'third': 'present_singular_3rd',
         'past': 'past', 'gerund': 'present_participle', 'participle': 'past_participle'}


@dataclass(frozen=True)
class SourceScope:
    operators: tuple[str, ...]
    tense: str
    phase: str = 'unspecified'


def _words(text):
    text = text.lower().replace('’', "'")
    text = re.sub(r"n't\b", ' not', text)
    text = re.sub(r'\bcannot\b', 'can not', text)
    text = re.sub(r'\bca\b', 'can', text)
    text = re.sub(r'\bwo\b', 'will', text)
    text = re.sub(r"'(m|re|ve|ll)\b", lambda m: ' ' + {'m': 'am', 're': 'are', 've': 'have', 'll': 'will'}[m[1]], text)
    # Resolve 's/'d by the licensed following form, never by frequency.
    tokens = re.findall(r"'[sd]|[a-z]+(?:'[a-z]+)?|[^\s]", text)
    return product(*[('is', 'has') if w == "'s" else ('had', 'would') if w == "'d" else (w,) for w in tokens])


def parse_scope(text, leaf, paradigms, finite_tag=None, *, voice='active'):
    """Parse the subject-to-leaf VP, excluding the leaf's complement."""
    if len(text.split()) > 24:
        return None, 'unsupported_source_scope'
    def matches(word, lemma, form):
        if lemma == 'be':
            return word in {'base': ('be',), 'present': ('am', 'are'), 'third': ('is',),
                            'past': ('was', 'were'), 'gerund': ('being',), 'participle': ('been',)}[form]
        op = OPERATORS.get(lemma)
        if op:
            expected = lemma if form in ('base', 'present') else (op.participle or op.past) if form == 'participle' else getattr(op, form)
            return word == expected
        return word in {p[FORMS[form]] for p in paradigms} if lemma == leaf else False

    def parse(words, form, finite=False):
        if not words:
            return []
        w, *rest = words
        if w == 'not' and not finite:
            return [('not',) + ops for ops in parse(rest, form)]
        if voice == 'passive' and matches(w, 'be', form):
            if len(rest) == 1 and matches(rest[0], leaf, 'participle'):
                return [()]
            if len(rest) == 2 and rest[0] == 'not' and matches(rest[1], leaf, 'participle'):
                return [('not',)]
        if voice == 'active' and len(words) == 1 and matches(w, leaf, form):
            return [()]
        results = []
        if finite and w in ('do', 'does', 'did') and form == {'do': 'present', 'does': 'third', 'did': 'past'}[w]:
            # Passive be takes direct negation/inversion, not do-support.
            # A lexical controller still can: did not want to be seen.
            results += [ops for ops in parse(rest, 'base')
                        if voice != 'passive' or any(op != 'not' for op in ops)]
        for lemma in (*CONTROL, 'have', 'be'):
            if not matches(w, lemma, form):
                continue
            negative = rest[:1] == ['not']
            tail = rest[1:] if negative else rest
            prefix = ('not',) if negative else ()
            if lemma in CONTROL:
                # Finite lexical negation requires do-support. Inner not belongs
                # to the infinitive, e.g. wanted not to leave.
                prefix = ()
                if OPERATORS[lemma].complement == 'to' or lemma == 'start' and tail[:1] == ['to']:
                    if tail[:1] == ['to']:
                        results += [(lemma,) + (('not',) if negative else ()) + ops for ops in parse(tail[1:], 'base')]
                elif not negative:
                    results += [(lemma,) + ops for ops in parse(rest, 'gerund')]
            elif lemma == 'have':
                if tail[:1] == ['to'] and not negative:
                    results += [('have_to',) + ops for ops in parse(tail[1:], 'base')]
                elif tail[:1] != ['to']:
                    results += [prefix + ('have',) + ops for ops in parse(tail, 'participle')]
            else:
                if tail[:2] in (['able', 'to'], ['going', 'to']):
                    op = 'be_able_to' if tail[0] == 'able' else 'be_going_to'
                    results += [prefix + (op,) + ops for ops in parse(tail[2:], 'base')]
                else:
                    results += [prefix + ('be',) + ops for ops in parse(tail, 'gerund')]
        return results

    candidates = set()
    for words in _words(text):
        if not words:
            continue
        if words[0] in MODALS or words[0] == 'will':
            modal, *rest = words
            prefix = (modal,) if modal != 'will' else ()
            if modal in ('can', 'could') and rest[:1] == ['not']:
                prefix, rest = ('not', modal), rest[1:]
            analyses = [(ops, 'future' if modal == 'will' else 'modal') for ops in
                        (prefix + ops for ops in parse(rest, 'base'))]
        else:
            forms = ('past',) if finite_tag == 'VBD' else ('present', 'third') if finite_tag in ('VBP', 'VBZ') else ('past', 'present', 'third')
            analyses = [(ops, 'past' if form == 'past' else 'present') for form in forms for ops in parse(list(words), form, True)]
        for ops, tense in analyses:
            phase = 'unspecified'
            if ops in (('be',), ('not', 'be')) and tense != 'future':
                ops, phase = ops[:-1], 'ongoing'
            elif not ops or ops == ('not',):
                phase = 'reported' if tense == 'past' else 'unspecified'
            try:
                validate_chain(ops, tense)
            except ValueError:
                continue
            candidates.add(SourceScope(ops, tense, phase))
    if len(candidates) != 1:
        return None, 'ambiguous_source_scope' if candidates else 'unsupported_source_scope'
    return candidates.pop(), None


def scope_structure(record, head):
    """License a ROOT -> reviewed xcomp chain with one shared subject."""
    slots = record['slots']
    chain = [head]
    while slots[chain[-1]].get('dependency') != 'ROOT':
        slot = slots[chain[-1]]
        parent = slot.get('head_slot')
        if slot.get('dependency') != 'xcomp' or parent not in slots or parent in chain:
            return None, 'unsupported_control_path'
        if slots[parent].get('lemma') not in CONTROL | {'have', 'be', 'go'}:
            return None, 'unreviewed_control_operator'
        chain.append(parent)
    root = chain[-1]
    subjects = [k for k, s in slots.items() if s.get('dependency') in ('nsubj', 'nsubjpass', 'nsubj:pass')]
    if len(subjects) != 1 or slots[subjects[0]].get('head_slot') != root:
        return None, 'unresolved_control_subject'
    allowed = set(chain)
    allowed.update(k for k, s in slots.items() if s.get('pos') == 'AUX'
                   and s.get('dependency') in ('aux', 'auxpass', 'aux:pass') and s.get('head_slot') in chain)
    if any(k not in allowed for k, s in slots.items() if s.get('kind') == 'verb'):
        return None, 'embedded_or_coordinated_predicates'
    if any(s.get('head_slot') in chain[1:] and k not in allowed | set(subjects)
           and not (s.get('dependency') == 'acomp' and record['defaults'].get(k, '').lower() == 'able')
           for k, s in slots.items()):
        return None, 'operator_has_separate_arguments'
    return (root, subjects[0], allowed), None
