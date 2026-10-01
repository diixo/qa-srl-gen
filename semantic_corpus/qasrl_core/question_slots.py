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
    ADVERBIAL_WH,
    BARE_COMPLEMENT_OBJ2,
    NOUN_WH,
    OBJ2_VALUES,
    OBJ_VALUES,
    SUBJ_VALUES,
    WH_WORDS,
)

__all__ = [
    "EMPTY",
    "SILENT_PREP",
    "SLOT_VOCABULARIES",
    "make_slots",
    "normalise_prep",
    "slot_tokens",
    "tail_tokens",
    "answer_slot_problems",
    "has_gap",
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


#: Permitted values per slot, used by :mod:`.validation`. ``prep`` is checked
#: separately, token by token, against the closed preposition inventory.
SLOT_VOCABULARIES: dict[str, frozenset[str]] = {
    "wh": frozenset(WH_WORDS),
    "subj": frozenset(SUBJ_VALUES) | {EMPTY},
    "obj": frozenset(OBJ_VALUES) | {EMPTY},
    "obj2": frozenset(OBJ2_VALUES) | {EMPTY},
}

assert set(SLOT_VOCABULARIES) <= set(SLOT_ORDER)


# ---------------------------------------------------------------------------
# Where the answer goes
# ---------------------------------------------------------------------------
#
# A QA-SRL question has exactly one gap: the argument slot being asked about.
# The gap leaves no surface token, so it has to be inferred from which slots
# are empty — and that inference is what makes the ``prep``/``obj`` readings
# of a question decidable. The rules below are the slot-level shadow of the
# completeness guard in ``QuestionProcessor``, which requires that
#
#   * a who/what question has an answer slot;
#   * a bare nominal second object implies a first object;
#   * the preposition state and the second object agree.
#
# All of them hold with zero exceptions over the 710 374 questions of
# QA-SRL Bank 2.0.


def has_gap(slots: QuestionSlots) -> bool:
    """Whether some slot can host the questioned argument.

    ``obj2`` holding ``do``/``doing`` counts: that spelling marks the gapped
    object of a non-finite complement (``What does something stop something
    from doing?``), unlike ``someone``/``something``/``somewhere``, which are
    placeholders for arguments that are *not* being asked about.
    """
    return (
        slots.subj == EMPTY
        or slots.obj == EMPTY
        or slots.obj2 == EMPTY
        or slots.obj2 in BARE_COMPLEMENT_OBJ2
    )


def answer_slot_problems(slots: QuestionSlots) -> list[tuple[str, str]]:
    """Return ``(code, message)`` for each answer-placement rule *slots* breaks."""
    problems: list[tuple[str, str]] = []

    if slots.subj == EMPTY and slots.wh not in NOUN_WH:
        problems.append(
            (
                "subject-gap-wh",
                f"only {sorted(NOUN_WH)} can question the subject, but wh="
                f"{slots.wh!r} has no subject",
            )
        )

    if slots.wh in ADVERBIAL_WH and slots.wh not in NOUN_WH and slots.subj == EMPTY:
        problems.append(
            (
                "adverbial-needs-subject",
                f"an adverbial question (wh={slots.wh!r}) always spells out its subject",
            )
        )

    if slots.wh in NOUN_WH and not has_gap(slots):
        problems.append(
            (
                "no-gap",
                f"wh={slots.wh!r} asks about an argument but every slot is filled, "
                "so there is nowhere for the answer to come from",
            )
        )

    if slots.prep == EMPTY and slots.obj2 in OBJ_VALUES and slots.obj == EMPTY:
        # A bare nominal second object implies a first object, and an empty
        # obj slot can only supply one by being the gap — which in turn
        # requires a nominal wh-word and a subject that is not already the gap.
        if slots.subj == EMPTY:
            problems.append(
                (
                    "obj2-without-obj",
                    f"a bare second object ({slots.obj2!r}) implies a first object, "
                    "but the subject is already the gap so obj cannot be the answer",
                )
            )
        elif slots.wh not in NOUN_WH:
            problems.append(
                (
                    "obj2-without-obj",
                    f"a bare second object ({slots.obj2!r}) implies a first object, "
                    f"so obj would have to be the gap — which wh={slots.wh!r} "
                    "cannot question",
                )
            )

    if slots.obj2 in BARE_COMPLEMENT_OBJ2 and slots.wh not in NOUN_WH:
        problems.append(
            (
                "bare-complement-wh",
                f"obj2={slots.obj2!r} marks a gapped complement object, which only "
                f"{sorted(NOUN_WH)} can question",
            )
        )

    return problems
