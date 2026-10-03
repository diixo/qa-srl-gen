"""Bank annotation provenance, ported from julianmichael/qasrl (MIT).

Source: qasrl-bank/{AnnotationRound,AnswerSource,QuestionSource,Data}.scala
at 16ab4949. See THIRD_PARTY_NOTICES.md. Unknown source strings raise
ValueError rather than being assigned an invented annotation round.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import re

from .dataset import Dataset

__all__ = ["AnnotationRound", "AnswerSource", "QuestionSource", "QuestionSourceKind", "filter_expanded_to_orig"]


class AnnotationRound(str, Enum):
    ORIGINAL = "original"
    EXPANSION = "expansion"
    EVAL = "eval"
    QANOM = "qanom"

    @property
    def is_qasrl_original(self) -> bool:
        return self is AnnotationRound.ORIGINAL

    @property
    def is_qasrl_expansion(self) -> bool:
        return self is AnnotationRound.EXPANSION

    @property
    def is_qasrl_eval(self) -> bool:
        return self is AnnotationRound.EVAL

    @property
    def is_qanom(self) -> bool:
        return self is AnnotationRound.QANOM


@dataclass(frozen=True)
class AnswerSource:
    turker_id: str
    round: AnnotationRound

    @classmethod
    def from_string(cls, value: str) -> "AnswerSource":
        qasrl = re.fullmatch(r"turk-qasrl2\.0-([0-9]+)-?(expansion|eval)?", value)
        if qasrl:
            return cls(qasrl[1], AnnotationRound(qasrl[2]) if qasrl[2] else AnnotationRound.ORIGINAL)
        qanom = re.fullmatch(r"turk-qanom-([0-9]+)", value)
        if qanom:
            return cls(qanom[1], AnnotationRound.QANOM)
        raise ValueError(f"Unknown answer source: {value!r}")

    def __str__(self) -> str:
        if self.round is AnnotationRound.QANOM:
            return f"turk-qanom-{self.turker_id}"
        suffix = "" if self.round is AnnotationRound.ORIGINAL else f"-{self.round.value}"
        return f"turk-qasrl2.0-{self.turker_id}{suffix}"

    @property
    def sort_key(self) -> tuple[int, str]:
        return (tuple(AnnotationRound).index(self.round), self.turker_id)


class QuestionSourceKind(str, Enum):
    MODEL = "model"
    QASRL_TURKER = "qasrl_turker"
    QANOM_TURKER = "qanom_turker"


@dataclass(frozen=True)
class QuestionSource:
    kind: QuestionSourceKind
    identifier: str

    @classmethod
    def from_string(cls, value: str) -> "QuestionSource":
        patterns = (
            (r"model-qasrl2\.0-(.+)", QuestionSourceKind.MODEL),
            (r"turk-qasrl2\.0-([0-9]+)", QuestionSourceKind.QASRL_TURKER),
            (r"turk-qanom-([0-9]+)", QuestionSourceKind.QANOM_TURKER),
        )
        for pattern, kind in patterns:
            match = re.fullmatch(pattern, value)
            if match:
                return cls(kind, match[1])
        raise ValueError(f"Unknown question source: {value!r}")

    @property
    def is_model(self) -> bool:
        return self.kind is QuestionSourceKind.MODEL

    @property
    def is_qasrl_turker(self) -> bool:
        return self.kind is QuestionSourceKind.QASRL_TURKER

    @property
    def is_qanom_turker(self) -> bool:
        return self.kind is QuestionSourceKind.QANOM_TURKER

    @property
    def get_model(self) -> str | None:
        return self.identifier if self.is_model else None

    @property
    def get_qasrl_turker(self) -> str | None:
        return self.identifier if self.is_qasrl_turker else None

    @property
    def get_qanom_turker(self) -> str | None:
        return self.identifier if self.is_qanom_turker else None

    @property
    def sort_key(self) -> tuple[int, str]:
        return (1 if self.is_model else 0, self.identifier)

    def __str__(self) -> str:
        prefix = {QuestionSourceKind.MODEL: "model-qasrl2.0", QuestionSourceKind.QASRL_TURKER: "turk-qasrl2.0", QuestionSourceKind.QANOM_TURKER: "turk-qanom"}[self.kind]
        return f"{prefix}-{self.identifier}"


def filter_expanded_to_orig(dataset: Dataset) -> Dataset:
    """Keep original QA-SRL questions and judgments, retaining empty containers."""
    questions = dataset.filter_question_sources(lambda source: QuestionSource.from_string(source).is_qasrl_turker)
    return questions.map_question_labels(lambda label: replace(
        label, answer_judgments=tuple(judgment for judgment in label.answer_judgments if AnswerSource.from_string(judgment.source_id).round.is_qasrl_original),
    ))
