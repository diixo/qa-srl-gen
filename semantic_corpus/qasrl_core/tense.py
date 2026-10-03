"""Finite and non-finite tenses from QA-SRL ``Tense.scala`` (MIT).

All values remain strings at the JSON boundary. ``Frame`` also accepts plain
strings for compatibility with the released Bank reader and existing callers.
The question-template automaton only licenses finite tenses; non-finite tenses
are useful when rendering clauses, including infinitives and gerunds.
"""

from __future__ import annotations

from enum import Enum


class _TenseValue(str, Enum):
    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str):
        """Parse a tense, returning ``None`` for an unrecognized spelling."""
        try:
            return cls(value)
        except ValueError:
            return None

    @classmethod
    def from_json(cls, value: str):
        """Decode the same string representation used by upstream Circe."""
        return cls(value)

    def to_json(self) -> str:
        return self.value


class Finite(_TenseValue):
    PAST = "past"
    PRESENT = "present"
    CAN = "can"
    WILL = "will"
    MIGHT = "might"
    WOULD = "would"
    SHOULD = "should"


class NonFinite(_TenseValue):
    BARE = "bare"
    TO = "to"
    GERUND = "gerund"


class Tense(_TenseValue):
    PAST = "past"
    PRESENT = "present"
    CAN = "can"
    WILL = "will"
    MIGHT = "might"
    WOULD = "would"
    SHOULD = "should"
    BARE = "bare"
    TO = "to"
    GERUND = "gerund"

    @property
    def is_finite(self) -> bool:
        return Finite.from_string(self.value) is not None

    @property
    def is_modal(self) -> bool:
        return self.is_finite and self not in (self.PAST, self.PRESENT)


__all__ = ["Finite", "NonFinite", "Tense"]
