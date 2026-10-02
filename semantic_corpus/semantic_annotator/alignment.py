"""Turning quoted spans into verified positions.

A teacher returns text, not offsets, because asking a language model to count
characters is asking for silent corruption. This module locates each quote in
the source and rejects anything it cannot find.

Rejection is the point. The handoff requires that every extracted span
provably exists in the original text, so an annotation that cannot be located
is dropped with a reason rather than approximated into place — an annotation
pointing at the wrong characters is worse than no annotation, because nothing
downstream can detect it.

Matching runs in two steps:

1. **Exact.** The quote is searched for literally, honouring ``occurrence``.
2. **Whitespace-insensitive.** Models reflow text, so a quote differing from
   the source only in runs of whitespace is matched and the *source* text is
   recorded, never the model's version. Anything else is rejected.

Case and punctuation differences are not tolerated: those change what the
span says.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from ..documents import (
    Document,
    EntityMention,
    Passage,
    Predicate,
    Property,
    RelationEdge,
    TextSpan,
)
from ..ontology import Label, LabelSet, Relation, ReviewStatus
from .teacher import ProposedAnnotation, TeacherResponse

__all__ = [
    "Rejection",
    "AlignmentResult",
    "locate",
    "align_response",
    "LocationError",
]


class LocationError(ValueError):
    """Raised by :func:`locate` when a quote is not present in the text."""


@dataclass(frozen=True, slots=True)
class Rejection:
    """An annotation that was dropped, and why."""

    annotation: ProposedAnnotation
    reason: str

    def __str__(self) -> str:
        return f"{self.annotation.text!r}: {self.reason}"


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """What survived alignment, and what did not."""

    mentions: tuple[EntityMention, ...] = ()
    predicates: tuple[Predicate, ...] = ()
    properties: tuple[Property, ...] = ()
    relations: tuple[RelationEdge, ...] = ()
    rejected: tuple[Rejection, ...] = ()

    @property
    def accepted_count(self) -> int:
        return len(self.mentions) + len(self.predicates) + len(self.properties)

    @property
    def rejection_rate(self) -> float:
        total = self.accepted_count + len(self.rejected)
        return len(self.rejected) / total if total else 0.0


def locate(
    quote: str,
    text: str,
    *,
    occurrence: int = 0,
    window: TextSpan | None = None,
) -> TextSpan:
    """Find *quote* in *text* and return its span.

    *window* restricts the search to a passage, which both speeds it up and
    stops a quote matching an identical string elsewhere in the document.
    """
    if not quote or not quote.strip():
        raise LocationError("the quote is empty")

    start_at = window.start_char if window else 0
    end_at = window.end_char if window else len(text)
    haystack = text[start_at:end_at]

    found = _find_exact(quote, haystack, occurrence)
    if found is None:
        found = _find_whitespace_insensitive(quote, haystack, occurrence)
    if found is None:
        raise LocationError(
            f"not found in the passage (occurrence {occurrence})"
        )
    start, end = found
    return TextSpan(start + start_at, end + start_at)


def _find_exact(quote: str, haystack: str, occurrence: int) -> tuple[int, int] | None:
    position = -1
    for _ in range(occurrence + 1):
        position = haystack.find(quote, position + 1)
        if position < 0:
            return None
    return position, position + len(quote)


def _find_whitespace_insensitive(
    quote: str, haystack: str, occurrence: int
) -> tuple[int, int] | None:
    """Match a quote whose internal whitespace was reflowed."""
    parts = quote.split()
    if not parts:
        return None
    pattern = re.compile(r"\s+".join(re.escape(part) for part in parts))
    matches = list(pattern.finditer(haystack))
    if occurrence < len(matches):
        match = matches[occurrence]
        return match.start(), match.end()
    return None


_PREDICATE_LABELS = {"ACTION": Label.ACTION, "STATE": Label.STATE}


def _parse_labels(names: Iterable[str]) -> tuple[LabelSet, list[str]]:
    """Convert label names, reporting the ones that are not in the ontology."""
    labels: set[Label] = set()
    unknown: list[str] = []
    for name in names:
        try:
            labels.add(Label(name.strip().upper()))
        except ValueError:
            unknown.append(name)
    return LabelSet(frozenset(labels)), unknown


def align_response(
    response: TeacherResponse,
    document: Document,
    passage: Passage | None = None,
    *,
    run_id: str,
    source: str = "teacher",
    status: ReviewStatus = ReviewStatus.AUTO_VALIDATED,
) -> AlignmentResult:
    """Locate every annotation of *response* in *document*.

    Annotations are written against the document's own text, so the span of a
    mention found inside a passage is still a document-global offset.
    """
    window = passage.span if passage else None
    text = document.text

    mentions: list[EntityMention] = []
    predicates: list[Predicate] = []
    properties: list[Property] = []
    rejected: list[Rejection] = []
    by_text: dict[tuple[str, int], str] = {}

    for index, annotation in enumerate(response.annotations):
        try:
            span = locate(
                annotation.text, text, occurrence=annotation.occurrence, window=window
            )
        except LocationError as error:
            rejected.append(Rejection(annotation, str(error)))
            continue

        exact_text = span.text_in(text)
        labels, unknown = _parse_labels(annotation.labels)
        if unknown:
            rejected.append(
                Rejection(annotation, f"labels not in the ontology: {unknown}")
            )
            continue
        if not labels.labels:
            rejected.append(Rejection(annotation, "no labels"))
            continue

        problems = labels.problems()
        if problems:
            rejected.append(Rejection(annotation, "; ".join(problems)))
            continue

        identifier = f"{run_id}#{index}"
        by_text[(annotation.text, annotation.occurrence)] = identifier

        predicate_label = next(
            (label for name, label in _PREDICATE_LABELS.items() if label in labels),
            None,
        )
        if annotation.head is not None:
            properties.append(
                Property(
                    property_id=identifier,
                    document_id=document.document_id,
                    span=span,
                    exact_text=exact_text,
                    head=annotation.head,
                    degree=annotation.degree,
                    negated=annotation.negated,
                    labels=labels,
                    confidence=annotation.confidence,
                    source=source,
                    review_status=status,
                )
            )
        elif predicate_label is not None:
            predicates.append(
                Predicate(
                    predicate_id=identifier,
                    document_id=document.document_id,
                    span=span,
                    exact_text=exact_text,
                    lemma=annotation.lemma or exact_text.lower(),
                    predicate_type=predicate_label,
                    confidence=annotation.confidence,
                    source=source,
                    review_status=status,
                )
            )
        else:
            mentions.append(
                EntityMention(
                    mention_id=identifier,
                    document_id=document.document_id,
                    span=span,
                    exact_text=exact_text,
                    labels=labels,
                    confidence=annotation.confidence,
                    source=source,
                    review_status=status,
                )
            )

    relations = tuple(
        _align_relations(response.annotations, by_text, source, rejected)
    )

    result = AlignmentResult(
        mentions=tuple(mentions),
        predicates=tuple(predicates),
        properties=tuple(properties),
        relations=relations,
        rejected=tuple(rejected),
    )
    # Everything kept must survive the document's own check; this is the
    # invariant the whole module exists to guarantee.
    for item in (*result.mentions, *result.predicates, *result.properties):
        item.validate_against(document)
    return result


def _align_relations(
    annotations: Sequence[ProposedAnnotation],
    by_text: dict[tuple[str, int], str],
    source: str,
    rejected: list[Rejection],
) -> Iterator[RelationEdge]:
    for annotation in annotations:
        if annotation.relation is None:
            continue
        source_id = by_text.get((annotation.text, annotation.occurrence))
        if source_id is None:
            continue  # its own span was already rejected, with a reason
        if annotation.target_text is None:
            rejected.append(
                Rejection(annotation, f"relation {annotation.relation} has no target")
            )
            continue
        target_id = by_text.get((annotation.target_text, 0))
        if target_id is None:
            rejected.append(
                Rejection(
                    annotation,
                    f"relation target {annotation.target_text!r} was not annotated",
                )
            )
            continue
        try:
            relation = Relation(annotation.relation.strip().upper())
        except ValueError:
            rejected.append(
                Rejection(annotation, f"unknown relation {annotation.relation!r}")
            )
            continue
        yield RelationEdge(
            source_id=source_id,
            relation=relation,
            target_id=target_id,
            question=annotation.question,
            confidence=annotation.confidence,
            source=source,
        )
