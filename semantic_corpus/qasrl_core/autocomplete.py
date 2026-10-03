"""Grammar continuations and context-sensitive question suggestions.

Port of ``Autocomplete.scala`` at julianmichael/qasrl 16ab4949 (MIT;
see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .frame import ALL_ADV_SLOTS
from .question_processor import (
    AggregatedInvalidState, CompleteState, InProgressState, QuestionProcessor,
)

__all__ = ["Suggestion", "Incomplete", "Complete", "Autocomplete"]


@dataclass(frozen=True, slots=True)
class Suggestion:
    full_text: str
    is_complete: bool


@dataclass(frozen=True, slots=True)
class Incomplete:
    suggestions: tuple[Suggestion, ...]
    bad_start_index: int | None = None


@dataclass(frozen=True, slots=True)
class Complete:
    completion_frames: tuple[CompleteState, ...]


def _unique(values):
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _sorted_suggestions(values: Iterable[Suggestion]) -> tuple[Suggestion, ...]:
    return tuple(sorted(set(values), key=lambda value: (not value.is_complete, value.full_text)))


class Autocomplete:
    def __init__(self, question_processor: QuestionProcessor) -> None:
        self.question_processor = question_processor

    def __call__(
        self, question: str, complete_questions: Iterable[CompleteState] = (),
    ) -> Complete | Incomplete:
        result = self.question_processor.process_string_fully(question)
        if isinstance(result, AggregatedInvalidState):
            suggestions = self._continuations(result.last_good_states)
            if not suggestions:
                # A complete question followed by extraneous characters.
                suggestions = (Suggestion(result.last_good_states[0].full_text, True),)
            return Incomplete(suggestions, result.num_good_characters)
        complete = tuple(_unique(state for state in result if isinstance(state, CompleteState)))
        if complete:
            return Complete(complete)

        previous = _unique(complete_questions)
        answered_text = {state.full_text.lower() for state in previous}
        frames = _unique(state.frame for state in previous)
        # Upstream projects a Set of (Frame, answer-slot) pairs onto a Set of
        # Frames before counting, so every count is one. Its remaining score
        # prefers structures with two arguments. Stable input order breaks ties.
        frames.sort(key=lambda frame: abs(len(frame.args) - 2))
        questions: list[str] = []
        for frame in frames:
            filled = {state.answer_slot for state in previous if state.frame == frame}
            slots = [slot for slot in frame.args if slot not in filled]
            # Scala's allAdvSlots has a semantic order. Preserve it until the
            # four-question cap; alphabetical sorting here changes which
            # adjuncts are offered, not merely how suggestions are displayed.
            slots.extend(slot for slot in ALL_ADV_SLOTS if slot not in filled)
            for slot in slots:
                for generated in frame.questions_for_slot(slot):
                    if (
                        generated.lower().startswith(question.lower())
                        and generated.lower() not in answered_text
                        and generated not in questions
                    ):
                        questions.append(generated)
        question_suggestions: list[Suggestion] = []
        for generated in questions:
            if self.question_processor.is_valid(generated):
                question_suggestions.append(Suggestion(generated, True))
                if len(question_suggestions) == 4:
                    break
        return Incomplete(_sorted_suggestions((*question_suggestions, *self._continuations(result))))

    def _continuations(self, states) -> tuple[Suggestion, ...]:
        return _sorted_suggestions(
            Suggestion(state.full_text, self.question_processor.is_almost_complete(state))
            for state in states if isinstance(state, InProgressState)
        )

    def apply(
        self, question: str, complete_questions: Iterable[CompleteState] = (),
    ) -> Complete | Incomplete:
        """Named equivalent of calling this autocomplete instance."""
        return self(question, complete_questions)
