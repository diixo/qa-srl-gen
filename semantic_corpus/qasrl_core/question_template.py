"""The abstract shape of a question, with tense and lexis stripped away.

Ported from ``labeling/QuestionTemplate.scala`` of ``julianmichael/qasrl``
(MIT). A template keeps only what is structural — the wh-word, whether there
is a subject and an object, the voice, the preposition — and discards the
verb, its tense and aspect, and the animacy of the arguments::

    Who gave something to someone?   ->  what verb something to something
    What was given to someone?       ->  what verb[pss] to something

Two questions with the same template ask the same *kind* of thing. That is
what makes this useful here:

**Deduplication that sees through wording.** Comparing question strings
treats *What did Anna give to Rex?* and *What was given to Rex by Anna?* as
different examples. :func:`normalize_to_active` folds the second onto the
first, so a corpus can be deduplicated by meaning rather than by spelling.

**A diversity number that is not just a count.** A corpus can hold three
hundred thousand examples of four shapes. Counting distinct templates says
how varied it actually is, which counting rows cannot.

The animacy collapse (``who`` becomes ``what``, ``someone`` becomes
``something``) is upstream's and is kept: animacy is a property of the
filler, not of the question's shape.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from .models import InflectedForms, QuestionSlots, VerbForm
from .question_renderer import render_question
from .question_slots import EMPTY, SILENT_PREP

__all__ = [
    "QuestionTemplate",
    "GENERIC_FORMS",
    "normalize_to_active",
    "normalize_adverbials",
]

#: A placeholder paradigm, so a template renders without a real verb.
GENERIC_FORMS = InflectedForms(
    stem="verb",
    present_singular_3rd="verbs",
    present_participle="verbing",
    past="verbed",
    past_participle="verbed",
)

#: Auxiliaries whose presence alongside a past participle marks the passive.
_PASSIVE_MARKERS = frozenset({"be", "been", "is", "isn't", "was", "wasn't"})


@dataclass(frozen=True, slots=True)
class QuestionTemplate:
    """One question's structure, without its words."""

    wh: str
    has_subj: bool
    is_passive: bool
    has_obj: bool
    prep: str | None = None
    obj2: str | None = None

    # -- construction ------------------------------------------------------

    @classmethod
    def from_slots(cls, slots: QuestionSlots) -> "QuestionTemplate":
        """Abstract a slot bundle into its shape."""
        prefix = [slots.aux] if slots.aux != EMPTY else []
        prefix += list(slots.verb_prefix)
        is_passive = slots.verb_form is VerbForm.PAST_PARTICIPLE and bool(
            set(prefix) & _PASSIVE_MARKERS
        )
        return cls(
            wh="what" if slots.wh == "who" else slots.wh,
            has_subj=slots.subj != EMPTY,
            is_passive=is_passive,
            has_obj=slots.obj != EMPTY,
            prep=None if slots.prep in (EMPTY, SILENT_PREP) else slots.prep,
            obj2=None if slots.obj2 == EMPTY else slots.obj2.replace("someone", "something"),
        )

    # -- rendering ---------------------------------------------------------

    def to_slots(self) -> QuestionSlots:
        """A concrete slot bundle realising this shape, with a generic verb."""
        if self.is_passive:
            aux, verb = "is", VerbForm.PAST_PARTICIPLE.value
        elif self.has_subj:
            aux, verb = "does", VerbForm.STEM.value
        else:
            aux, verb = EMPTY, VerbForm.PRESENT_SINGULAR_3RD.value
        return QuestionSlots(
            wh=self.wh,
            aux=aux,
            subj="something" if self.has_subj else EMPTY,
            verb=verb,
            obj="something" if self.has_obj else EMPTY,
            prep=self.prep if self.prep is not None else EMPTY,
            obj2=self.obj2 if self.obj2 is not None else EMPTY,
        )

    @property
    def template_string(self) -> str:
        """The shape as a flat string: ``what verb[pss] to something``."""
        parts = [
            self.wh,
            "something" if self.has_subj else None,
            "verb[pss]" if self.is_passive else "verb",
            "something" if self.has_obj else None,
            self.prep,
            self.obj2,
        ]
        return " ".join(part for part in parts if part)

    @property
    def question_string(self) -> str:
        """The shape rendered as an English question about a generic verb."""
        return render_question(self.to_slots(), GENERIC_FORMS)

    def __str__(self) -> str:
        return self.template_string

    # -- serialisation -----------------------------------------------------

    def to_json(self) -> dict[str, str]:
        return {
            "abst-wh": self.wh,
            "abst-subj": "something" if self.has_subj else EMPTY,
            "abst-verb": "verb[pss]" if self.is_passive else "verb",
            "abst-obj": "something" if self.has_obj else EMPTY,
            "prep": self.prep if self.prep is not None else EMPTY,
            "abst-obj2": self.obj2 if self.obj2 is not None else EMPTY,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "QuestionTemplate":
        return cls(
            wh=str(data["abst-wh"]),
            has_subj=data["abst-subj"] != EMPTY,
            is_passive=data["abst-verb"] == "verb[pss]",
            has_obj=data["abst-obj"] != EMPTY,
            prep=None if data["prep"] == EMPTY else str(data["prep"]),
            obj2=None if data["abst-obj2"] == EMPTY else str(data["abst-obj2"]),
        )


def normalize_adverbials(template: QuestionTemplate) -> QuestionTemplate:
    """Collapse every adverbial question of one wh-word onto one shape.

    *Where did something give something to something?* and *Where did
    something run?* ask the same kind of thing about different clauses, and
    for counting shapes the clause is noise.
    """
    if template.wh == "what":
        return template
    return QuestionTemplate(template.wh, True, False, False, None, None)


def normalize_to_active(template: QuestionTemplate) -> QuestionTemplate:
    """Rewrite a passive shape as the active one that asks the same thing.

    A faithful port of upstream's case analysis. The cases are not derivable
    from a general rule: a passive drops its agent, strands its preposition
    or promotes its object depending on which argument is questioned, so
    each combination is handled explicitly. Upstream's worked examples::

        what is something given something on?  -> what gave something something?
        what is something punched by?          -> what punched something?
        what is punched on something?          -> what did something punch on something?
    """
    if not template.is_passive:
        return template

    asking_by = template.prep == "by" and template.obj2 is None
    has_by_placeholder = template.prep == "by" and template.obj2 == "something"

    if template.wh == "what":
        if template.has_subj:
            return replace(
                template,
                has_subj=not asking_by,
                is_passive=False,
                prep=None if asking_by else template.prep,
                obj2=("something" if template.has_obj else template.obj2),
            )
        return replace(
            template,
            has_subj=True,
            is_passive=False,
            prep=None if has_by_placeholder else template.prep,
            obj2=None if has_by_placeholder else template.obj2,
        )

    if template.has_subj:
        return replace(
            template,
            is_passive=False,
            has_obj=True,
            prep=(
                None
                if (template.has_obj or has_by_placeholder)
                else template.prep
            ),
            obj2=(
                None
                if has_by_placeholder
                else ("something" if template.has_obj else template.obj2)
            ),
        )
    return replace(template, has_subj=True, is_passive=False)
