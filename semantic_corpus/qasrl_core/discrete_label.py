"""Discrete syntactic QA-SRL labels; these are not semantic AGENT/THEME roles.

Adapted from ``labeling/DiscreteLabel.scala`` in julianmichael/qasrl at
16ab4949, MIT, Copyright (c) 2017 Julian Michael. See THIRD_PARTY_NOTICES.md.
Passive voice, noun animacy and object position determine the label exactly
as upstream; prepositional labels take precedence over an object ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .clause_resolution import frame_pair_key, parse_frame_pairs
from .frame import OBJ, OBJ2, SUBJ, ArgumentSlot, Frame, Locative, Noun, Prep
from .models import InflectedForms
from .question_label_mapper import QuestionLabelMapper
from .state_machine import ADVERBIAL_WH

CORE_ARG_DISCRETE_LABELS = frozenset({"subj-transitive", "subj-intransitive", "obj", "obj-dative"})
ADV_DISCRETE_LABELS = ADVERBIAL_WH

__all__ = ["DiscreteLabel", "NounRole", "AdvRole", "from_rendered_string",
           "label_for_frame", "labels_for_question", "get_all_discrete_labels",
           "get_discrete_labels", "CORE_ARG_DISCRETE_LABELS", "ADV_DISCRETE_LABELS"]


class DiscreteLabel:
    label: str

    def render(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.render()

    @staticmethod
    def from_rendered_string(value: str) -> "DiscreteLabel":
        return from_rendered_string(value)


@dataclass(frozen=True, slots=True)
class NounRole(DiscreteLabel):
    label: str
    is_animate: bool

    def __post_init__(self):
        object.__setattr__(self, "label", self.label.lower())

    def render(self) -> str:
        return self.label + ("/+" if self.is_animate else "/-")


@dataclass(frozen=True, slots=True)
class AdvRole(DiscreteLabel):
    label: str

    def __post_init__(self):
        object.__setattr__(self, "label", self.label.lower())

    def render(self) -> str:
        return self.label


def from_rendered_string(value: str) -> DiscreteLabel:
    value = value.lower()
    if value in ADV_DISCRETE_LABELS:
        return AdvRole(value)
    parts = value.split("/")
    if len(parts) != 2 or not parts[0] or parts[1] not in {"+", "-"}:
        raise ValueError(f"invalid discrete label: {value!r}")
    return NounRole(parts[0], parts[1] == "+")


def label_for_frame(frame: Frame, answer_slot: ArgumentSlot) -> DiscreteLabel:
    if answer_slot.is_adverbial:
        if answer_slot.wh not in ADV_DISCRETE_LABELS:
            raise ValueError(f"invalid adverbial slot: {answer_slot}")
        return AdvRole(answer_slot.wh)
    argument = frame.args.get(answer_slot)
    passive = frame.structure.is_passive
    if answer_slot == OBJ2:
        if isinstance(argument, Prep):
            label = "subj-transitive" if passive and argument.preposition == "by" else argument.preposition
            return NounRole(label, argument.obj.is_animate if argument.obj is not None else False)
        if isinstance(argument, Noun):
            return NounRole("obj-dative" if argument.is_animate else "obj", argument.is_animate)
        if isinstance(argument, Locative):
            return AdvRole("where")
    if answer_slot == OBJ and isinstance(argument, Noun):
        dative = (isinstance(frame.args.get(OBJ2), Noun) or passive) and argument.is_animate
        return NounRole("obj-dative" if dative else "obj", argument.is_animate)
    if answer_slot == SUBJ and isinstance(argument, Noun):
        if passive:
            label = "obj-dative" if OBJ in frame.args else "obj"
        else:
            label = "subj-transitive" if OBJ in frame.args else "subj-intransitive"
        return NounRole(label, argument.is_animate)
    raise ValueError(f"frame has no answerable argument in {answer_slot}")


def labels_for_question(sentence_tokens: Sequence[str], forms: InflectedForms,
                         question: str) -> tuple[DiscreteLabel, ...]:
    pairs = sorted(parse_frame_pairs(sentence_tokens, forms, question), key=frame_pair_key)
    return tuple(label_for_frame(frame, slot) for frame, slot in pairs)


def _all_labels(tokens, forms, questions):
    results = {}
    for question in questions:
        labels = labels_for_question(tokens, forms, question)
        if labels:
            results[question] = labels
    return results


def _preferred(labels: tuple[DiscreteLabel, ...]) -> DiscreteLabel:
    objects = {"obj", "obj-dative"}
    if len(labels) == 2 and any(label.label in objects for label in labels):
        return next((label for label in labels if label.label not in objects), labels[0])
    return labels[0]


get_all_discrete_labels = QuestionLabelMapper(_all_labels)
get_discrete_labels = get_all_discrete_labels >> QuestionLabelMapper.lift(_preferred)
