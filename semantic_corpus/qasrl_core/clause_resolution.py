"""Resolve question ambiguity using the other questions for the same verb.

Adapted from ``labeling/ClauseResolution.scala`` in julianmichael/qasrl
at 16ab4949, MIT, Copyright (c) 2017 Julian Michael. See THIRD_PARTY_NOTICES.md.
Parsing retains every complete frame. Each question distributes one vote
over its distinct frames; local maxima precede the upstream fallback rules.
Ties with no upstream preference use a stable order instead of hash order.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from functools import lru_cache
from typing import Iterable, Sequence

from .frame import OBJ, OBJ2, ArgStructure, ArgumentSlot, Frame, adv
from .models import InflectedForms, QuestionSlots
from .question_renderer import render_question
from .state_machine import PREPOSITIONS

GENERIC_FORMS = InflectedForms("stem", "presentsingular3rd", "presentparticiple", "past", "pastparticiple")
FramePair = tuple[Frame, ArgumentSlot]

__all__ = [
    "GENERIC_FORMS", "get_frames_with_answer_slots", "locally_resolve",
    "classify_ambiguity", "fallback_resolve", "get_resolved_frame_pairs",
    "get_resolved_structures", "get_clause_template", "question_prepositions",
]


def question_prepositions(question: str) -> frozenset[str]:
    """Question-specific prep unigrams and adjacent bigrams, as upstream."""
    words = question.removesuffix("?").lower().split(" ")
    return frozenset(
        [word for word in words if word in PREPOSITIONS]
        + [f"{a} {b}" for a, b in zip(words, words[1:]) if a in PREPOSITIONS and b in PREPOSITIONS]
    )


def frame_pair_key(pair: FramePair) -> tuple:
    frame, slot = pair
    return (
        tuple(sorted((str(key), repr(value)) for key, value in frame.args.items())),
        frame.structure.is_passive, str(frame.tense), frame.is_perfect,
        frame.is_progressive, frame.is_negated, frame.verb_inflected_forms.all_forms,
        str(slot),
    )


def parse_frame_pairs(sentence_tokens: Sequence[str], forms: InflectedForms,
                      question: str) -> frozenset[FramePair]:
    # Local imports also keep the finite parser independent of this resolver.
    from .question_processor import AggregatedInvalidState, CompleteState, QuestionProcessor
    from .template_state_machine import TemplateStateMachine

    machine = TemplateStateMachine(sentence_tokens, forms, prepositions=question_prepositions(question))
    states = QuestionProcessor(machine).process_string_fully(question)
    if isinstance(states, AggregatedInvalidState):
        return frozenset()
    return frozenset((state.frame, state.answer_slot) for state in states if isinstance(state, CompleteState))


@lru_cache(maxsize=4096)
def _generic_frame_pairs(slots: QuestionSlots) -> frozenset[FramePair]:
    question = render_question(slots, GENERIC_FORMS)
    pairs = parse_frame_pairs((), GENERIC_FORMS, question)
    if not pairs:
        raise ValueError(f"no complete QA-SRL parse for {question!r}")
    return pairs


def get_frames_with_answer_slots(question_slots: QuestionSlots,
                                inflected_forms: InflectedForms = GENERIC_FORMS) -> frozenset[FramePair]:
    """Every reading of a slot label, with a bounded generic-paradigm cache."""
    return frozenset((replace(frame, verb_inflected_forms=inflected_forms), slot)
                     for frame, slot in _generic_frame_pairs(question_slots))


def locally_resolve(frame_pair_sets: Iterable[Iterable[FramePair]]) -> list[frozenset[FramePair]]:
    choices = [frozenset(pairs) for pairs in frame_pair_sets]
    counts: dict[Frame, float] = defaultdict(float)
    for pairs in choices:
        frames = {frame for frame, _ in pairs}
        for frame in frames:
            counts[frame] += 1.0 / len(frames)
    return [frozenset(pair for pair in pairs if counts[pair[0]] == max(counts[f] for f, _ in pairs))
            if pairs else frozenset() for pairs in choices]


def _filled(value: str) -> bool:
    return value != QuestionSlots.EMPTY and bool(value)


def classify_ambiguity(slots: QuestionSlots, frame_pairs: Iterable[FramePair]) -> str:
    pairs = frozenset(frame_pairs)
    answer_slots = {slot for _, slot in pairs}
    if len(pairs) == 1:
        return "unambiguous"
    if (not _filled(slots.obj) and _filled(slots.prep) and not _filled(slots.obj2)
            and answer_slots == {OBJ, OBJ2}):
        return "prepositional"
    if slots.wh == "where" and answer_slots == {adv("where"), OBJ2}:
        return "where"
    if (slots.wh in {"who", "what"} and _filled(slots.obj) != _filled(slots.obj2)
            and answer_slots == {OBJ, OBJ2}):
        return "ditransitive"
    return "other"


def fallback_resolve(slots: QuestionSlots, frame_pairs: Iterable[FramePair]) -> FramePair:
    pairs = sorted(set(frame_pairs), key=frame_pair_key)
    if not pairs:
        raise ValueError("cannot resolve an empty set of frames")
    ambiguity = classify_ambiguity(slots, pairs)
    preferred = {
        "prepositional": OBJ2,
        "where": adv("where"),
        "ditransitive": OBJ2 if slots.wh == "what" else OBJ,
    }.get(ambiguity)
    return next((pair for pair in pairs if pair[1] == preferred), pairs[0])


def get_resolved_frame_pairs(inflected_forms: InflectedForms,
                             question_slots: Sequence[QuestionSlots]) -> list[FramePair]:
    choices = [get_frames_with_answer_slots(slots, inflected_forms) for slots in question_slots]
    return [fallback_resolve(slots, pairs) for slots, pairs in zip(question_slots, locally_resolve(choices))]


def get_clause_template(frame: Frame) -> ArgStructure:
    return frame.structure.forget_animacy()


def get_resolved_structures(question_slots: Sequence[QuestionSlots]) -> list[tuple[ArgStructure, ArgumentSlot]]:
    return [(get_clause_template(frame), slot)
            for frame, slot in get_resolved_frame_pairs(GENERIC_FORMS, question_slots)]
