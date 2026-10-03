"""Character-level nondeterministic QA-SRL question processing.

Port of ``QuestionProcessor.scala`` at julianmichael/qasrl 16ab4949 (MIT;
see THIRD_PARTY_NOTICES.md). Every live branch holds a partial Frame, a
remaining transition string, and a target grammar node. Processing another
character advances those branches without reparsing the prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .frame import OBJ, OBJ2, ArgumentSlot, Frame, Noun, Prep, adv
from .template_state_machine import (
    FrameState, TemplateComplete, TemplateProgress, TemplateState,
    TemplateStateMachine,
)

__all__ = [
    "ValidState", "CompleteState", "InProgressState", "InvalidState",
    "AggregatedInvalidState", "ProcessingState", "QuestionProcessor",
]


class ValidState:
    """One surviving interpretation of the consumed input."""

    @property
    def is_complete(self) -> bool:
        return isinstance(self, CompleteState)


@dataclass(frozen=True, slots=True)
class CompleteState(ValidState):
    full_text: str
    frame: Frame
    answer_slot: ArgumentSlot


@dataclass(frozen=True, slots=True)
class InProgressState(ValidState):
    text_so_far: str
    frame_state: FrameState
    text_remaining: str
    target_state: TemplateState

    @property
    def full_text(self) -> str:
        """The prefix completed to the end of the current transition."""
        return self.text_so_far + self.text_remaining


@dataclass(frozen=True, slots=True)
class InvalidState:
    last_good_state: ValidState
    num_good_characters: int


@dataclass(frozen=True, slots=True)
class AggregatedInvalidState:
    last_good_states: tuple[ValidState, ...]
    num_good_characters: int


@dataclass(frozen=True, slots=True)
class ProcessingState:
    valid_states: tuple[ValidState, ...] = ()
    invalid_states: tuple[InvalidState, ...] = ()


class QuestionProcessor:
    def __init__(self, state_machine: TemplateStateMachine) -> None:
        self.state_machine = state_machine

    def get_states_from_transition(
        self, text_so_far: str, frame_state: FrameState, new_state: TemplateState,
    ) -> tuple[ValidState, ...]:
        if isinstance(new_state, TemplateComplete):
            frame = frame_state.frame
            obj2 = frame.args.get(OBJ2)
            wh = frame_state.wh_word
            if (
                wh is not None
                and (frame_state.preposition is not None) == isinstance(obj2, Prep)
                and (wh not in ("who", "what") or frame_state.answer_slot is not None)
                and (not isinstance(obj2, Noun) or OBJ in frame.args)
            ):
                return (CompleteState(text_so_far, frame, frame_state.answer_slot or adv(wh)),)
            return ()
        if not isinstance(new_state, TemplateProgress):
            raise TypeError(f"unknown template state: {type(new_state).__name__}")
        states: list[ValidState] = []
        for transition in new_state.transitions:
            updated = transition.apply(frame_state)
            if updated is None:
                continue
            fs, target = updated
            if transition.text:
                states.append(InProgressState(text_so_far, fs, transition.text, target))
            else:
                states.extend(self.get_states_from_transition(text_so_far, fs, target))
        return tuple(states)

    @property
    def initial_states(self) -> tuple[ValidState, ...]:
        return self.get_states_from_transition(
            "", self.state_machine.initial_frame_state, self.state_machine.start,
        )

    def process_character(self, state: ValidState, observed_char: str) -> ProcessingState:
        if len(observed_char) != 1:
            raise ValueError("process_character requires exactly one character")
        if isinstance(state, CompleteState):
            return ProcessingState(invalid_states=(InvalidState(state, len(state.full_text)),))
        if not isinstance(state, InProgressState):
            raise TypeError(f"unknown processing state: {type(state).__name__}")
        expected = state.text_remaining[0]
        if expected.lower() != observed_char.lower():
            return ProcessingState(invalid_states=(InvalidState(state, len(state.text_so_far)),))
        text = state.text_so_far + expected
        remaining = state.text_remaining[1:]
        if remaining:
            return ProcessingState((replace(state, text_so_far=text, text_remaining=remaining),))
        return ProcessingState(self.get_states_from_transition(text, state.frame_state, state.target_state))

    def advance(self, states: ProcessingState, text: str) -> ProcessingState:
        """Continue an earlier result with additional input, without reparsing."""
        valid = states.valid_states
        invalid = list(states.invalid_states)
        for character in text:
            next_valid: list[ValidState] = []
            for state in valid:
                result = self.process_character(state, character)
                next_valid.extend(result.valid_states)
                invalid.extend(result.invalid_states)
            valid = tuple(next_valid)
            if not valid:
                break
        return ProcessingState(valid, tuple(invalid))

    def process_string(self, text: str) -> ProcessingState:
        return self.advance(ProcessingState(self.initial_states), text)

    def process_string_fully(self, text: str) -> tuple[ValidState, ...] | AggregatedInvalidState:
        """Return live states, or the latest valid prefix if every branch failed.

        A tuple can contain unfinished states: use :meth:`is_valid` when a
        complete question (including ``?``) is required.
        """
        result = self.process_string(text)
        if result.valid_states:
            return result.valid_states
        length = max(state.num_good_characters for state in result.invalid_states)
        return AggregatedInvalidState(
            tuple(state.last_good_state for state in result.invalid_states if state.num_good_characters == length),
            length,
        )

    def is_valid(self, text: str) -> bool:
        return any(state.is_complete for state in self.process_string(text).valid_states)

    @staticmethod
    def is_almost_complete(state: InProgressState) -> bool:
        return isinstance(state.target_state, TemplateComplete)
