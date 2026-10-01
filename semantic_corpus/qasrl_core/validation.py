"""Invariant checks over QA-SRL slots, spans and sentences.

Every check returns a list of human-readable problems instead of raising, so a
caller scanning a corpus can count and classify defects rather than stopping at
the first one. :func:`check_sentence` wraps the lot for a whole sentence.

The slot checks are grammar-driven: the legal ``(aux, verb)`` inventory comes
from :mod:`.state_machine`, not from a hand-written table, so the validator and
the renderer can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import (
    AnswerJudgment,
    InflectedForms,
    QuestionLabel,
    QuestionSlots,
    Sentence,
    Span,
    VerbEntry,
)
from .question_renderer import render_question
from .question_slots import EMPTY, SILENT_PREP, SLOT_VOCABULARIES
from .state_machine import (
    BARE_COMPLEMENT_OBJ2,
    TENSES,
    features_for_chain,
    is_known_chain,
)

__all__ = [
    "Problem",
    "check_slots",
    "check_span",
    "check_question_label",
    "check_verb_entry",
    "check_sentence",
    "validate_sentence",
    "count_problems",
    "ValidationError",
]


class ValidationError(ValueError):
    """Raised by :func:`validate_sentence` when a sentence fails any check."""


@dataclass(frozen=True, slots=True)
class Problem:
    """One invariant violation, located as precisely as the data allows."""

    code: str
    message: str
    where: str = ""

    def __str__(self) -> str:
        return f"[{self.code}] {self.where + ': ' if self.where else ''}{self.message}"


def check_slots(slots: QuestionSlots, where: str = "") -> list[Problem]:
    """Check one slot bundle against the closed vocabularies and the grammar."""
    problems: list[Problem] = []

    for name, vocabulary in SLOT_VOCABULARIES.items():
        value = getattr(slots, name)
        if value not in vocabulary:
            problems.append(
                Problem(
                    "slot-vocabulary",
                    f"{name}={value!r} is not one of {sorted(vocabulary)}",
                    where,
                )
            )

    try:
        form = slots.verb_form
    except ValueError:
        problems.append(
            Problem("verb-form", f"verb={slots.verb!r} does not end in a verb form", where)
        )
        form = None

    if form is not None and not is_known_chain(slots.aux, slots.verb):
        problems.append(
            Problem(
                "verb-chain",
                f"aux={slots.aux!r} with verb={slots.verb!r} is not a chain the "
                "grammar can produce",
                where,
            )
        )

    if slots.aux == EMPTY and slots.subj != EMPTY:
        problems.append(
            Problem(
                "subject-inversion",
                "nothing was fronted into aux, so the questioned argument must be "
                f"the subject, but subj={slots.subj!r}",
                where,
            )
        )

    if slots.prep == SILENT_PREP and slots.obj2 not in BARE_COMPLEMENT_OBJ2:
        problems.append(
            Problem(
                "silent-prep",
                f"an empty prep marks a bare complement, so obj2 must be one of "
                f"{sorted(BARE_COMPLEMENT_OBJ2)}, got {slots.obj2!r}",
                where,
            )
        )
    if slots.prep == EMPTY and slots.obj2 in BARE_COMPLEMENT_OBJ2:
        problems.append(
            Problem(
                "silent-prep",
                f"obj2={slots.obj2!r} is a bare complement and needs prep='' "
                "rather than '_'",
                where,
            )
        )

    return problems


def check_span(
    span: Span, sentence_tokens: Sequence[str], where: str = ""
) -> list[Problem]:
    """Check that a span addresses real tokens of the sentence."""
    if not span.fits(sentence_tokens):
        return [
            Problem(
                "span-bounds",
                f"span [{span.start}, {span.end}) runs past the sentence "
                f"({len(sentence_tokens)} tokens)",
                where,
            )
        ]
    return []


def check_question_label(
    label: QuestionLabel,
    forms: InflectedForms,
    sentence_tokens: Sequence[str],
    where: str = "",
) -> list[Problem]:
    """Check a question: slots, rendering, stored features, judgments, spans."""
    problems = check_slots(label.question_slots, where)

    try:
        rendered = render_question(label.question_slots, forms)
    except (ValueError, KeyError) as error:
        problems.append(Problem("render", f"cannot render slots: {error}", where))
    else:
        if rendered != label.question_string:
            problems.append(
                Problem(
                    "render-mismatch",
                    f"slots render to {rendered!r} but the stored question is "
                    f"{label.question_string!r}",
                    where,
                )
            )

    if label.tense not in TENSES:
        problems.append(
            Problem("tense", f"unknown tense {label.tense!r}", where)
        )
    else:
        stored = (
            label.tense,
            label.is_perfect,
            label.is_progressive,
            label.is_passive,
            label.is_negated,
        )
        derivable = {
            (f.tense, f.is_perfect, f.is_progressive, f.is_passive, f.is_negated)
            for f in features_for_chain(
                label.question_slots.aux, label.question_slots.verb
            )
        }
        if derivable and stored not in derivable:
            problems.append(
                Problem(
                    "features",
                    f"stored features {stored} cannot produce "
                    f"aux={label.question_slots.aux!r} "
                    f"verb={label.question_slots.verb!r}",
                    where,
                )
            )

    seen_sources: set[str] = set()
    for judgment in label.answer_judgments:
        if judgment.source_id in seen_sources:
            problems.append(
                Problem(
                    "duplicate-judgment",
                    f"source {judgment.source_id!r} judged this question twice",
                    where,
                )
            )
        seen_sources.add(judgment.source_id)
        problems.extend(_check_judgment(judgment, sentence_tokens, where))

    return problems


def _check_judgment(
    judgment: AnswerJudgment, sentence_tokens: Sequence[str], where: str
) -> list[Problem]:
    problems: list[Problem] = []
    if not judgment.is_valid and judgment.spans:
        problems.append(
            Problem(
                "rejected-with-spans",
                f"source {judgment.source_id!r} rejected the question but still "
                "highlighted spans",
                where,
            )
        )
    for span in judgment.spans:
        problems.extend(check_span(span, sentence_tokens, where))
    return problems


def check_verb_entry(
    entry: VerbEntry, sentence_tokens: Sequence[str], where: str = ""
) -> list[Problem]:
    """Check one predicate: its index, and every question asked about it."""
    problems: list[Problem] = []
    if not 0 <= entry.verb_index < len(sentence_tokens):
        problems.append(
            Problem(
                "verb-index",
                f"verb index {entry.verb_index} is outside the sentence "
                f"({len(sentence_tokens)} tokens)",
                where,
            )
        )

    for key, label in entry.question_labels.items():
        location = f"{where}/{key}" if where else key
        if key != label.question_string:
            problems.append(
                Problem(
                    "question-key",
                    f"map key {key!r} differs from questionString "
                    f"{label.question_string!r}",
                    location,
                )
            )
        problems.extend(
            check_question_label(
                label, entry.verb_inflected_forms, sentence_tokens, location
            )
        )
    return problems


def check_sentence(sentence: Sentence) -> list[Problem]:
    """Run every check over a sentence and return all problems found."""
    problems: list[Problem] = []
    if not sentence.sentence_tokens:
        problems.append(
            Problem("empty-sentence", "sentence has no tokens", sentence.sentence_id)
        )
    for key, entry in sentence.verb_entries.items():
        location = f"{sentence.sentence_id}#{key}"
        if key.isdigit() and int(key) != entry.verb_index:
            problems.append(
                Problem(
                    "verb-key",
                    f"map key {key!r} differs from verbIndex {entry.verb_index}",
                    location,
                )
            )
        problems.extend(check_verb_entry(entry, sentence.sentence_tokens, location))
    return problems


def validate_sentence(sentence: Sentence) -> Sentence:
    """Return *sentence* unchanged, or raise :class:`ValidationError`."""
    problems = check_sentence(sentence)
    if problems:
        joined = "\n  ".join(str(p) for p in problems)
        raise ValidationError(f"{sentence.sentence_id}: {len(problems)} problem(s):\n  {joined}")
    return sentence


def count_problems(problems: Iterable[Problem]) -> dict[str, int]:
    """Tally problems by code — handy when scanning a whole split."""
    counts: dict[str, int] = {}
    for problem in problems:
        counts[problem.code] = counts.get(problem.code, 0) + 1
    return counts
