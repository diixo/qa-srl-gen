"""Contextual slot-label mappers from upstream SlotBasedLabel.scala.

Adapted from julianmichael/qasrl at 16ab4949, MIT, Copyright (c) 2017
Julian Michael. See THIRD_PARTY_NOTICES.md. Surface and abstract verb slots
use distinct types so a realised verb is never mistaken for a VerbForm key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .clause_resolution import question_prepositions
from .frame import OBJ
from .models import InflectedForms, QuestionSlots, SLOT_ORDER, _rendered_slot_fields
from .question_label_mapper import QuestionLabelMapper
from .question_processor import AggregatedInvalidState, CompleteState, QuestionProcessor
from .template_state_machine import TemplateStateMachine

__all__ = [
    "SurfaceQuestionSlots", "get_preferred_complete_state", "read_preferred_state",
    "get_slots_for_question", "get_verb_tense_abstracted_slots_for_question",
    "instantiate_verb_for_tense_slots",
]


@dataclass(frozen=True, slots=True)
class SurfaceQuestionSlots:
    """Seven slots whose verb is surface text, e.g. ``being given``."""

    wh: str
    aux: str
    subj: str
    verb: str
    obj: str
    prep: str
    obj2: str

    def to_json(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in SLOT_ORDER}

    @classmethod
    def from_json(cls, data: Mapping[str, str]) -> "SurfaceQuestionSlots":
        return cls(**{name: data[name] for name in SLOT_ORDER})

    def render(self) -> str:
        text = " ".join(value for value in self.to_json().values() if value and value != "_")
        return text[:1].upper() + text[1:] + "?"

    def slot_strings(self) -> tuple[str, ...]:
        return tuple(value.lower() for value in self.to_json().values())

    def render_with_separator(self, separator: str) -> str:
        if not separator:
            raise ValueError("slot separator must not be empty")
        return separator.join(self.slot_strings())

    @classmethod
    def from_rendered_string(cls, value: str, separator: str) -> "SurfaceQuestionSlots":
        return cls.from_json(_rendered_slot_fields(value, separator))


def get_preferred_complete_state(states: Sequence) -> CompleteState | None:
    """Prefer an analysis with a direct object, then the first complete state."""
    complete = [state for state in states if isinstance(state, CompleteState)]
    return next((state for state in complete if OBJ in state.frame.args),
                complete[0] if complete else None)


def read_preferred_state(sentence_tokens: Sequence[str], forms: InflectedForms,
                         question: str) -> CompleteState | None:
    processor = QuestionProcessor(TemplateStateMachine(
        sentence_tokens, forms, prepositions=question_prepositions(question),
    ))
    states = processor.process_string_fully(question)
    return None if isinstance(states, AggregatedInvalidState) else get_preferred_complete_state(states)


def _abstract_slots(tokens, forms, question):
    state = read_preferred_state(tokens, forms, question)
    return None if state is None else state.frame.to_slots_upstream(state.answer_slot)


def _instantiate(_tokens, forms: InflectedForms, slots: QuestionSlots) -> SurfaceQuestionSlots:
    data = slots.to_json()
    data["verb"] = " ".join((*slots.verb_prefix, forms.get(slots.verb_form)))
    return SurfaceQuestionSlots.from_json(data)


get_verb_tense_abstracted_slots_for_question = QuestionLabelMapper.lift_optional_with_context(_abstract_slots)
instantiate_verb_for_tense_slots = QuestionLabelMapper.lift_with_context(_instantiate)
get_slots_for_question = get_verb_tense_abstracted_slots_for_question >> instantiate_verb_for_tense_slots
