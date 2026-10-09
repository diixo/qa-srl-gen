"""Scope over complete attested clauses, without guessing event semantics.

An explicit report/denial can establish that a statement was endorsed, not that
every embedded action was completed. Questions and imperatives are quotations
too: confirming that they were said is different from asserting their content.
"""
from dataclasses import dataclass

STATUSES = ('reported', 'denied', 'desired', 'planned', 'suggested', 'possible', 'required', 'permitted')


@dataclass(frozen=True)
class ObservedClause:
    text: str
    status: str = 'reported'

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.strip() or self.text != self.text.strip():
            raise ValueError('An observed clause needs nonempty, trimmed text')
        if self.status not in STATUSES:
            raise ValueError('Unsupported observed-clause status')

    def statement(self, *, perspective='memory', style=0):
        subject = 'I' if perspective == 'user' else 'You'
        # These wrappers deliberately scope over a quotation. Inner modality,
        # negation, questions, idioms and mixed subjects are never stripped.
        phrases = {
            'reported': ('reported this exact wording', 'said these words'),
            'denied': ('denied saying these words', 'said that this was not my wording' if subject == 'I' else 'said that this was not your wording'),
            'desired': ('wanted to say these words', 'wished to use this wording'),
            'planned': ('planned to say these words', 'intended to use this wording'),
            'suggested': ('suggested using this wording', 'proposed saying these words'),
            'possible': ('said this wording might be used', 'said it was possible to use this wording'),
            'required': ('said this wording must be used', 'said it was necessary to use this wording'),
            'permitted': ('said this wording could be used', 'allowed this wording to be used'),
        }
        return f'{subject} {phrases[self.status][style % 2]}: "{self.text}"'

    def question(self):
        return f'Did I report saying these exact words: "{self.text}"?'
