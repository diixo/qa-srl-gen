"""BIO / BILOU export — a derived, lossy view.

The ontology is deliberately multi-label and allows nested spans, and a
per-token tag sequence can represent neither. So this exporter is a
*projection*, never the working representation, exactly as the handoff
requires.

What is lost is counted rather than hidden. Every export returns a
:class:`BioReport` saying how many spans were dropped for overlapping an
already-tagged token and how many labels were discarded from multi-label
spans. An export that silently kept one label per span would look fine and
quietly misrepresent the corpus.

Resolution rules, in order:

1. longer spans win over shorter ones, because the longer span is the more
   specific annotation;
2. a tie goes to the span that starts earlier;
3. within a span, the first label in ontology declaration order is kept, so
   the choice is deterministic rather than set-iteration order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from ..documents import AnnotationRun, Document, EntityMention, Predicate, Property
from ..ontology import Label
from ..semantic_annotator.candidates import tokenize
from .jsonl import write_jsonl

__all__ = ["BioReport", "Tagging", "tag_document", "export_bio", "Scheme"]

Scheme = Literal["BIO", "BILOU"]


@dataclass(slots=True)
class BioReport:
    """What the projection had to throw away."""

    documents: int = 0
    tokens: int = 0
    tagged_spans: int = 0
    dropped_overlapping: int = 0
    dropped_labels: int = 0
    dropped_unrepresentable: int = 0
    dropped_examples: list[str] = field(default_factory=list)

    @property
    def is_lossless(self) -> bool:
        return self.dropped_overlapping == 0 and self.dropped_labels == 0 and self.dropped_unrepresentable == 0

    def note(self, message: str) -> None:
        if len(self.dropped_examples) < 20:
            self.dropped_examples.append(message)

    def __str__(self) -> str:
        head = (
            f"{self.documents} documents, {self.tokens} tokens, "
            f"{self.tagged_spans} spans tagged"
        )
        if self.is_lossless:
            return head + "; nothing dropped"
        return (
            head
            + f"; dropped {self.dropped_overlapping} overlapping spans and "
            f"{self.dropped_labels} extra labels, {self.dropped_unrepresentable} unrepresentable spans"
        )


@dataclass(frozen=True, slots=True)
class Tagging:
    """One document as parallel token and tag sequences."""

    document_id: str
    tokens: tuple[str, ...]
    tags: tuple[str, ...]
    split: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "tokens": list(self.tokens),
            "tags": list(self.tags),
            "split": self.split,
        }


def _span_label(item: EntityMention | Predicate | Property) -> Label | None:
    for label in item.labels.ordered:
        return label
    return None


def _extra_labels(item: EntityMention | Predicate | Property) -> int:
    return max(0, len(item.labels) - 1)


def tag_document(
    document: Document,
    run: AnnotationRun,
    *,
    scheme: Scheme = "BIO",
    report: BioReport | None = None,
) -> Tagging:
    """Project one run onto a per-token tag sequence."""
    report = report if report is not None else BioReport()
    if scheme not in ("BIO", "BILOU"):
        raise ValueError(f"unknown scheme {scheme!r}")
    tokens = tokenize(document.text)
    tags = ["O"] * len(tokens)
    report.documents += 1
    report.tokens += len(tokens)

    annotated = [*run.mentions, *run.predicates, *run.properties]
    # Longer spans first; ties broken by position so the result is stable.
    annotated.sort(key=lambda a: (-(len(a.span)), a.span.start_char))

    occupied = [False] * len(tokens)
    for item in annotated:
        label = _span_label(item)
        if label is None:
            report.dropped_unrepresentable += 1
            report.note(f"{document.document_id}: {item.exact_text!r} has no known label")
            continue
        covered = [
            index
            for index, token in enumerate(tokens)
            if token.start_char >= item.span.start_char
            and token.end_char <= item.span.end_char
        ]
        if (not covered or tokens[covered[0]].start_char != item.span.start_char
                or tokens[covered[-1]].end_char != item.span.end_char):
            report.dropped_unrepresentable += 1
            report.note(f"{document.document_id}: {item.exact_text!r} is not token aligned")
            continue
        if any(occupied[index] for index in covered):
            report.dropped_overlapping += 1
            report.note(f"{document.document_id}: {item.exact_text!r} overlaps")
            continue
        extra = _extra_labels(item)
        if extra:
            report.dropped_labels += extra
            report.note(
                f"{document.document_id}: {item.exact_text!r} kept only {label}"
            )
        for position, index in enumerate(covered):
            occupied[index] = True
            tags[index] = _tag(scheme, label, position, len(covered))
        report.tagged_spans += 1

    return Tagging(
        document_id=document.document_id,
        tokens=tuple(token.text for token in tokens),
        tags=tuple(tags),
        split=document.split,
    )


def _tag(scheme: Scheme, label: Label, position: int, length: int) -> str:
    name = str(label)
    if scheme == "BIO":
        return f"{'B' if position == 0 else 'I'}-{name}"
    if scheme != "BILOU":
        raise ValueError(f"unknown scheme {scheme!r}")
    if length == 1:
        return f"U-{name}"
    if position == 0:
        return f"B-{name}"
    if position == length - 1:
        return f"L-{name}"
    return f"I-{name}"


def export_bio(
    pairs: Iterable[tuple[Document, AnnotationRun]],
    path: Path | str,
    *,
    scheme: Scheme = "BIO",
) -> BioReport:
    """Write a tagged corpus, returning what the projection cost."""
    report = BioReport()

    def rows() -> Iterator[dict[str, Any]]:
        for document, run in pairs:
            yield tag_document(document, run, scheme=scheme, report=report).to_json()

    write_jsonl(path, rows())
    return report
