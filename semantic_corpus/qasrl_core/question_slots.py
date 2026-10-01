"""The argument side of the QA-SRL question template.

``state_machine`` owns the verb chain (``aux`` + ``verb``); this module owns
the nominal slots — ``wh``, ``subj``, ``obj``, ``prep``, ``obj2`` — their
closed vocabularies and the conventions that tie them together.

Two conventions are easy to get wrong and are therefore encoded here rather
than left to callers:

* ``prep`` has three states, not two. ``"_"`` means the template has no
  prepositional position; ``""`` means there is a position but it is silent,
  which is how a bare complement is written (``What does something help
  something do?`` has ``prep=""`` and ``obj2="do"``); anything else is a
  surface preposition, possibly multi-word (``to do``, ``out of``).
* ``obj2`` accepts placeholders that ``obj`` does not (``somewhere``, ``do``,
  ``doing``), because the second object position also hosts locatives and
  non-finite complements.
"""

from __future__ import annotations

from .models import QuestionSlots, SLOT_ORDER
from .state_machine import (
    BARE_COMPLEMENT_OBJ2,
    OBJ2_VALUES,
    OBJ_VALUES,
    SUBJ_VALUES,
    WH_WORDS,
)

__all__ = [
    "EMPTY",
    "SILENT_PREP",
    "make_slots",
    "normalise_prep",
    "slot_tokens",
    "tail_tokens",
]

EMPTY = QuestionSlots.EMPTY
#: A prepositional position that is present but has no surface form.
SILENT_PREP = ""


def normalise_prep(prep_words: list[str], obj2: str) -> str:
    """Pick the right ``prep`` value for an empty surface preposition.

    With no preposition words, the slot is ``""`` when ``obj2`` is a bare
    complement and ``"_"`` otherwise. This is the rule that makes the
    ``prep``/``obj2`` split recoverable from the surface string.
    """
    if prep_words:
        return " ".join(prep_words)
    return SILENT_PREP if obj2 in BARE_COMPLEMENT_OBJ2 else EMPTY


def make_slots(
    wh: str,
    verb: str,
    *,
    aux: str = EMPTY,
    subj: str = EMPTY,
    obj: str = EMPTY,
    prep: str | None = None,
    obj2: str = EMPTY,
) -> QuestionSlots:
    """Build :class:`QuestionSlots`, deriving ``prep`` when it is not given."""
    if prep is None:
        prep = SILENT_PREP if obj2 in BARE_COMPLEMENT_OBJ2 else EMPTY
    return QuestionSlots(
        wh=wh, aux=aux, subj=subj, verb=verb, obj=obj, prep=prep, obj2=obj2
    )


def slot_tokens(slots: QuestionSlots, name: str) -> list[str]:
    """Surface tokens contributed by one slot (empty for unfilled or silent)."""
    value = getattr(slots, name)
    if value == EMPTY:
        return []
    return [word for word in value.split(" ") if word]


def tail_tokens(slots: QuestionSlots) -> list[str]:
    """Surface tokens after the verb: ``obj``, ``prep`` and ``obj2`` combined."""
    tokens: list[str] = []
    for name in ("obj", "prep", "obj2"):
        tokens.extend(slot_tokens(slots, name))
    return tokens


#: Permitted values per slot, used by :mod:`.validation`. ``prep`` is open
#: class and therefore absent from this table.
SLOT_VOCABULARIES: dict[str, frozenset[str]] = {
    "wh": frozenset(WH_WORDS),
    "subj": frozenset(SUBJ_VALUES) | {EMPTY},
    "obj": frozenset(OBJ_VALUES) | {EMPTY},
    "obj2": frozenset(OBJ2_VALUES) | {EMPTY},
}

assert set(SLOT_VOCABULARIES) <= set(SLOT_ORDER)
