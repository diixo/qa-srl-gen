"""Parse a surface QA-SRL question back into its seven slots.

Rendering is lossy in principle — ``prep="to do"`` and ``prep="to", obj2="do"``
produce the same string — so the parser enumerates every analysis the grammar
licenses, keeps only those that render back to the original question, and
ranks what is left by a fixed, documented preference.

The search is driven by the verb: the predicate's own inflected forms are
known, so the parser locates the inflected token, splits what precedes it into
``aux``/``subj``/auxiliary words, and reads ``obj``/``prep``/``obj2`` off the
remainder. The ``(aux, verb)`` pair is then checked against
:func:`~.state_machine.is_known_chain`, which rejects most spurious analyses
before the render test ever runs.

Round-trip guarantee::

    render_question(parse_question(q, forms), forms) == q

Preference order between surviving analyses (highest priority first):

1. the rightmost admissible position for the inflected verb — this is what
   tells ``What does something do?`` (``aux=does``, ``verb=stem``) apart from
   the reading that mistakes the auxiliary ``does`` for the predicate itself;
2. a filled ``obj2`` over a longer multi-word ``prep``;
3. a filled ``obj`` over a ``prep`` that starts with a placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .models import InflectedForms, QuestionSlots
from .question_renderer import render_question
from .question_slots import EMPTY, answer_slot_problems, normalise_prep
from .state_machine import (
    AUX_WORDS,
    BARE_COMPLEMENT_OBJ2,
    OBJ2_VALUES,
    OBJ_VALUES,
    PREPOSITIONS,
    SUBJ_VALUES,
    VERB_PREFIX_WORDS,
    WH_WORDS,
    is_known_chain,
)

#: Words the ``prep`` slot may be built from. Bounding it is what stops a
#: placeholder being swallowed into a multi-word preposition, which would turn
#: ``What causes something to do something?`` into a reading whose ``prep`` is
#: ``something to do``.
_PREP_WORDS: frozenset[str] = PREPOSITIONS | BARE_COMPLEMENT_OBJ2

__all__ = ["QuestionParseError", "parse_question", "parse_question_all"]

_WH_BY_LENGTH: tuple[tuple[int, frozenset[str]], ...] = (
    (2, frozenset(w for w in WH_WORDS if " " in w)),
    (1, frozenset(w for w in WH_WORDS if " " not in w)),
)


class QuestionParseError(ValueError):
    """Raised when no analysis of the question reproduces the input string."""


@dataclass(frozen=True, slots=True)
class _Candidate:
    slots: QuestionSlots
    head_index: int

    def rank_key(self) -> tuple[int, int]:
        return (
            -self.head_index,
            # The template reaches "the first object is the gap" before it
            # reaches "the first object is a placeholder", so when both
            # analyses survive, the gapped one is the original's choice.
            0 if self.slots.obj == EMPTY and self.slots.obj2 != EMPTY else 1,
        )


def _uncapitalise(token: str) -> str:
    return token[0].lower() + token[1:] if token else token


def _split_prefix(prefix: list[str]) -> Iterator[tuple[str, str, list[str]]]:
    """Split the words before the inflected verb into ``aux``, ``subj``, rest.

    ``aux`` and ``subj`` are closed classes that cannot collide with the
    literal auxiliary words allowed inside the verb slot, so the split is
    deterministic; it is still written as a generator because a prefix that
    does not fit the shape yields nothing at all.
    """
    index = 0
    aux = EMPTY
    if index < len(prefix) and prefix[index] in AUX_WORDS:
        aux = prefix[index]
        index += 1
    subj = EMPTY
    if index < len(prefix) and prefix[index] in SUBJ_VALUES:
        subj = prefix[index]
        index += 1
    rest = prefix[index:]
    if all(word in VERB_PREFIX_WORDS for word in rest):
        yield aux, subj, rest


def _split_tail(tail: list[str]) -> Iterator[tuple[str, str, str]]:
    """Enumerate ``(obj, prep, obj2)`` readings of the words after the verb."""
    seen: set[tuple[str, str, str]] = set()
    obj_options = [0]
    if tail and tail[0] in OBJ_VALUES:
        obj_options.insert(0, 1)
    for obj_len in obj_options:
        obj = tail[0] if obj_len else EMPTY
        middle = tail[obj_len:]
        obj2_options = [0]
        if middle and middle[-1] in OBJ2_VALUES:
            obj2_options.insert(0, 1)
        for obj2_len in obj2_options:
            obj2 = middle[-1] if obj2_len else EMPTY
            prep_words = middle[: len(middle) - obj2_len]
            if not all(word in _PREP_WORDS for word in prep_words):
                continue
            reading = (obj, normalise_prep(prep_words, obj2), obj2)
            if reading not in seen:
                seen.add(reading)
                yield reading


def parse_question_all(question: str, forms: InflectedForms) -> list[QuestionSlots]:
    """Every slot analysis that renders back to *question*, best first."""
    body = question.strip()
    if not body.endswith("?"):
        raise QuestionParseError(f"question must end with '?': {question!r}")
    tokens = body[:-1].split()
    if not tokens:
        raise QuestionParseError(f"empty question: {question!r}")
    tokens[0] = _uncapitalise(tokens[0])

    candidates: list[_Candidate] = []
    for wh_len, vocabulary in _WH_BY_LENGTH:
        if len(tokens) < wh_len + 1:
            continue
        wh = " ".join(tokens[:wh_len])
        if wh not in vocabulary:
            continue
        candidates.extend(_analyse(question, tokens, wh, wh_len, forms))

    candidates.sort(key=_Candidate.rank_key)
    # Distinct analyses only; equal slot tuples can arise from different heads.
    out: list[QuestionSlots] = []
    for candidate in candidates:
        if candidate.slots not in out:
            out.append(candidate.slots)
    return out


def _analyse(
    question: str,
    tokens: list[str],
    wh: str,
    wh_len: int,
    forms: InflectedForms,
) -> Iterator[_Candidate]:
    surfaces = set(forms.all_forms)
    for head in range(wh_len, len(tokens)):
        if tokens[head] not in surfaces:
            continue
        prefix = tokens[wh_len:head]
        tail = tokens[head + 1 :]
        for form in forms.forms_for(tokens[head]):
            for aux, subj, verb_prefix in _split_prefix(prefix):
                verb = " ".join([*verb_prefix, form.value])
                if not is_known_chain(aux, verb):
                    continue
                if aux == EMPTY and subj != EMPTY:
                    # Nothing was fronted, so the subject must be the gap.
                    continue
                for obj, prep, obj2 in _split_tail(tail):
                    slots = QuestionSlots(
                        wh=wh, aux=aux, subj=subj, verb=verb,
                        obj=obj, prep=prep, obj2=obj2,
                    )
                    if answer_slot_problems(slots):
                        # Nowhere for the answer to sit, or a second object
                        # without a first: not a question the template builds.
                        continue
                    if render_question(slots, forms) == question:
                        yield _Candidate(slots, head)


def parse_question(question: str, forms: InflectedForms) -> QuestionSlots:
    """Parse *question* into slots, choosing the preferred analysis.

    Raises :class:`QuestionParseError` if nothing renders back to the input.
    """
    analyses = parse_question_all(question, forms)
    if not analyses:
        raise QuestionParseError(
            f"no slot analysis renders back to {question!r} "
            f"for paradigm {forms.stem!r}"
        )
    return analyses[0]
