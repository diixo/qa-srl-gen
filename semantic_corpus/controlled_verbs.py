"""Routing policy for reviewed verbs; raw mined banks remain immutable.

Auxiliary be/have/modals inside another predicate's chain are retained. Lexical
and copular occurrences are handled by controlled frames/operators instead.
"""

from .qasrl_core.models import InflectedForms

VERSION = 'controlled-verbs-v1'
CONTROLLED_VERBS = ('can', 'must', 'may', 'be', 'let', 'have', 'go')
CONTROLLED_PARADIGMS = {
    'be': InflectedForms('be', 'is', 'being', 'was', 'been'),
    'have': InflectedForms('have', 'has', 'having', 'had', 'had'),
    'go': InflectedForms('go', 'goes', 'going', 'went', 'gone'),
    'let': InflectedForms('let', 'lets', 'letting', 'let', 'let'),
}
_ALIASES = {
    'could': 'can', 'might': 'may', 'am': 'be', 'is': 'be', 'are': 'be',
    'was': 'be', 'were': 'be', 'been': 'be', 'being': 'be',
    'has': 'have', 'had': 'have', 'having': 'have', 'lets': 'let', 'letting': 'let',
    'goes': 'go', 'went': 'go', 'gone': 'go', 'going': 'go',
    'ca': 'can', 'cannot': 'can', "can't": 'can', "couldn't": 'can',
    "mustn't": 'must', "mightn't": 'may',
}
AUXILIARY_DEPENDENCIES = frozenset(('aux', 'auxpass', 'aux:pass'))


def controlled_head(lemma):
    head = str(lemma).strip().lower().replace('’', "'").split(' ', 1)[0]
    head = _ALIASES.get(head, head)
    return head if head in CONTROLLED_VERBS else None


def excluded_predicates(record):
    """Names of managed lexical/copular heads, never plain substring matches."""
    return frozenset(head for slot in record.get('slots', {}).values()
                     if slot.get('kind') == 'verb'
                     and slot.get('dependency') not in AUXILIARY_DEPENDENCIES
                     and (head := controlled_head(slot.get('lemma', ''))))
