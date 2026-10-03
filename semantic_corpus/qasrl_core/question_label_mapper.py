"""Context-aware, composable question-label mappings.

Adapted from ``labeling/QuestionLabelMapper.scala`` in julianmichael/qasrl
at 16ab4949, MIT, Copyright (c) 2017 Julian Michael. See THIRD_PARTY_NOTICES.md.
Each mapping sees all labels for one predicate together. Missing labels stay
missing through composition; input order and duplicate inputs are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Iterable, Mapping, Sequence, TypeVar

from .models import InflectedForms

A = TypeVar("A", bound=Hashable)
B = TypeVar("B", bound=Hashable)
C = TypeVar("C", bound=Hashable)
D = TypeVar("D", bound=Hashable)

__all__ = ["QuestionLabelMapper", "map_to_lower_case"]


@dataclass(frozen=True)
class QuestionLabelMapper(Generic[A, B]):
    mapping: Callable[[Sequence[str], InflectedForms, Sequence[A]], Mapping[A, B]]

    def __call__(self, sentence_tokens: Sequence[str], forms: InflectedForms,
                 labels: Iterable[A]) -> list[B | None]:
        labels = list(labels)
        result = self.mapping(sentence_tokens, forms, labels)
        return [result.get(label) for label in labels]

    @classmethod
    def lift_optional_with_context(cls, function):
        def mapping(tokens, forms, labels):
            result = {}
            for label in labels:
                value = function(tokens, forms, label)
                if value is not None:
                    result[label] = value
            return result
        return cls(mapping)

    @classmethod
    def lift_with_context(cls, function):
        return cls(lambda tokens, forms, labels: {
            label: function(tokens, forms, label) for label in labels
        })

    @classmethod
    def lift_optional(cls, function):
        return cls.lift_optional_with_context(lambda _tokens, _forms, label: function(label))

    @classmethod
    def lift(cls, function):
        return cls.lift_with_context(lambda _tokens, _forms, label: function(label))

    @classmethod
    def identity(cls):
        return cls.lift(lambda label: label)

    def and_then(self, other: "QuestionLabelMapper[B, C]") -> "QuestionLabelMapper[A, C]":
        def mapping(tokens, forms, labels):
            intermediate = self.mapping(tokens, forms, labels)
            following = other.mapping(tokens, forms,
                                      [intermediate[a] for a in labels if a in intermediate])
            return {a: following[b] for a, b in intermediate.items() if b in following}
        return QuestionLabelMapper(mapping)

    def compose(self, before: "QuestionLabelMapper[C, A]") -> "QuestionLabelMapper[C, B]":
        return before.and_then(self)

    def __rshift__(self, other: "QuestionLabelMapper[B, C]") -> "QuestionLabelMapper[A, C]":
        return self.and_then(other)

    def first(self) -> "QuestionLabelMapper[tuple[A, C], tuple[B, C]]":
        def mapping(tokens, forms, labels):
            converted = self.mapping(tokens, forms, [a for a, _ in labels])
            return {(a, c): (converted[a], c) for a, c in labels if a in converted}
        return QuestionLabelMapper(mapping)

    def second(self) -> "QuestionLabelMapper[tuple[C, A], tuple[C, B]]":
        def mapping(tokens, forms, labels):
            converted = self.mapping(tokens, forms, [a for _, a in labels])
            return {(c, a): (c, converted[a]) for c, a in labels if a in converted}
        return QuestionLabelMapper(mapping)

    def split(self, other: "QuestionLabelMapper[C, D]") -> "QuestionLabelMapper[tuple[A, C], tuple[B, D]]":
        return self.first().and_then(other.second())

    def fanout(self, other: "QuestionLabelMapper[A, C]") -> "QuestionLabelMapper[A, tuple[B, C]]":
        return QuestionLabelMapper.lift(lambda a: (a, a)).and_then(self.split(other))


map_to_lower_case = QuestionLabelMapper.lift(str.lower)
