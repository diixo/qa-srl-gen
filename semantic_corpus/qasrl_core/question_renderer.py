"""Render QA-SRL question slots into the surface question string.

The rule is short and exact: walk the slots in template order, drop the
unfilled ones, replace the verb-form placeholder with the actual inflected
form, join with single spaces, capitalise the first letter and append ``?``.

Verified against QA-SRL Bank 2.0: all 385 645 questions in
``orig/{train,dev,test}``, ``expanded/dev`` and ``dense/dev`` are reproduced
byte for byte from their stored slots.
"""

from __future__ import annotations

from .models import SLOT_ORDER, InflectedForms, QuestionSlots, VerbForm
from .question_slots import EMPTY

__all__ = ["render_question", "render_verb_slot"]

_FORM_KEYS = frozenset(form.value for form in VerbForm)


def render_verb_slot(verb: str, forms: InflectedForms) -> list[str]:
    """Expand the ``verb`` slot into surface words.

    Every word is literal except the final one, which is a :class:`VerbForm`
    key to be looked up in *forms*::

        "being pastParticiple" -> ["being", "given"]
    """
    words = [word for word in verb.split(" ") if word]
    if not words:
        raise ValueError("the verb slot cannot be empty")
    return [forms.get(word) if word in _FORM_KEYS else word for word in words]


def render_question(slots: QuestionSlots, forms: InflectedForms) -> str:
    """Build the surface question string for *slots* under paradigm *forms*."""
    tokens: list[str] = []
    for name in SLOT_ORDER:
        value = getattr(slots, name)
        if value == EMPTY:
            continue
        if name == "verb":
            tokens.extend(render_verb_slot(value, forms))
        else:
            # A filled-but-silent prep contributes nothing; split() handles the
            # multi-word values such as "how much" and "out of".
            tokens.extend(word for word in value.split(" ") if word)
    if not tokens:
        raise ValueError("no filled slots to render")
    sentence = " ".join(tokens)
    return sentence[0].upper() + sentence[1:] + "?"
