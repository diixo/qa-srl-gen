"""The second pass: checking an annotation run instead of trusting it.

Two kinds of checking happen here, and they are different in kind.

**Structural verification** is mechanical and always runs: do the spans
cover the text they claim, are the label combinations legal, do relations
point at annotations that exist, is anything duplicated or nested in a way
the ontology forbids. This needs no model and cannot be wrong.

**Independent re-annotation** asks a teacher the same question a second time
and compares. The handoff calls for this pass to be independent, so the
second opinion is formed without being shown the first: a model asked to
confirm its own output agrees with itself, which measures nothing. Spans both
passes agree on are promoted to ``VERIFIED``; spans only one pass produced
are left at their existing status and reported, not deleted — the first pass
may well have been right.

Nothing is ever edited in place. Verification produces a *new*
:class:`~..documents.AnnotationRun`, so the original claim and the judgment
on it both survive.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Iterable, Mapping, Sequence

from ..documents import (
    AnnotationRun,
    Document,
    EntityMention,
    Predicate,
    Property,
    TextSpan,
)
from ..ontology import ReviewStatus
from .alignment import AlignmentResult, align_response
from .teacher import Teacher, TeacherRequest, request_for

__all__ = [
    "StructuralReport",
    "AgreementReport",
    "verify_structure",
    "verify_independently",
    "promote_agreed",
]


@dataclass(frozen=True, slots=True)
class StructuralReport:
    """Mechanical problems found in a run."""

    problems: tuple[str, ...] = ()
    overlapping: tuple[tuple[str, str], ...] = ()

    @property
    def ok(self) -> bool:
        return not self.problems and not self.overlapping

    def __str__(self) -> str:
        if self.ok:
            return "no structural problems"
        lines = list(self.problems)
        lines += [f"spans {a} and {b} overlap partially" for a, b in self.overlapping]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class AgreementReport:
    """How two independent passes compare."""

    agreed: tuple[str, ...] = ()
    only_first: tuple[str, ...] = ()
    only_second: tuple[str, ...] = ()

    @property
    def agreement_rate(self) -> float:
        total = len(self.agreed) + len(self.only_first) + len(self.only_second)
        return len(self.agreed) / total if total else 1.0


def _key(item) -> tuple:
    """All semantic fields, excluding per-run identity and provenance."""
    ignored = {"mention_id", "predicate_id", "property_id", "target_id",
               "confidence", "source", "review_status"}
    return (type(item).__name__, tuple(
        (f.name, getattr(item, f.name)) for f in fields(item) if f.name not in ignored
    ))


def _identity(item) -> tuple[str, str]:
    if hasattr(item, "utterance_id"):
        return ("dialogue", item.utterance_id)
    return ("span", next(getattr(item, name) for name in
                        ("mention_id", "predicate_id", "property_id") if hasattr(item, name)))


def _semantic_keys(run: AnnotationRun) -> dict[tuple[str, str], tuple]:
    items = (*run.mentions, *run.predicates, *run.properties)
    base = {_identity(item)[1]: _key(item) for item in items}
    if len(base) != len(items):
        raise ValueError("cannot compare a run with duplicate annotation IDs")
    # Resolve references by meaning; identifiers differ between passes.
    keys = {}
    for item in items:
        ident = _identity(item)[1]
        links = frozenset(
            (base.get(e.source_id, ("missing", e.source_id)), e.relation,
             base.get(e.target_id, ("missing", e.target_id)), e.question)
            for e in run.relations if ident in (e.source_id, e.target_id)
        )
        target = getattr(item, "target_id", None)
        keys[_identity(item)] = (base[ident], base.get(target, ("missing", target)) if target else None, links)
    for d in run.dialogue:
        ident = _identity(d)
        if ident in keys:
            raise ValueError("cannot compare a run with duplicate dialogue annotations")
        keys[ident] = _key(d)
    return keys


def verify_structure(run: AnnotationRun, document: Document) -> StructuralReport:
    """Check a run against the document it describes, without any model."""
    problems = list(run.validate_against(document))

    # Nesting is allowed; partial overlap is not, because it cannot be
    # expressed as a tree and breaks every span-based export.
    spans: list[tuple[str, TextSpan]] = []
    for mention in run.mentions:
        spans.append((mention.mention_id, mention.span))
    for predicate in run.predicates:
        spans.append((predicate.predicate_id, predicate.span))
    for prop in run.properties:
        spans.append((prop.property_id, prop.span))

    overlapping: list[tuple[str, str]] = []
    ordered = sorted(spans, key=lambda pair: (pair[1].start_char, pair[1].end_char))
    for index, (left_id, left) in enumerate(ordered):
        for right_id, right in ordered[index + 1 :]:
            if right.start_char >= left.end_char:
                break
            if not (left.contains(right) or right.contains(left)):
                overlapping.append((left_id, right_id))

    return StructuralReport(problems=tuple(problems), overlapping=tuple(overlapping))


def verify_independently(
    run: AnnotationRun,
    document: Document,
    teacher: Teacher,
    *,
    passage_index: Mapping[str, object] | None = None,
    run_id: str | None = None,
) -> tuple[AnnotationRun, AgreementReport]:
    """Re-annotate with *teacher* and compare, without showing it the first pass.

    Returns the second run and the comparison. The second run is a complete
    annotation in its own right, not a diff, so it can be stored, inspected
    and compared against a third pass later.
    """
    second_id = run_id or f"{run.run_id}-verify"
    from .pipeline import annotate_document
    outcome = annotate_document(document, teacher, run_id=second_id,
                                status=ReviewStatus.UNREVIEWED)
    if not outcome.ok:
        raise ValueError(str(outcome.structure))
    second = outcome.run
    if run.ontology_version != second.ontology_version:
        raise ValueError("cannot compare different ontology versions")
    first_keys = set(_semantic_keys(run).values())
    second_keys = set(_semantic_keys(second).values())
    report = AgreementReport(
        agreed=tuple(sorted(map(repr, first_keys & second_keys))),
        only_first=tuple(sorted(map(repr, first_keys - second_keys))),
        only_second=tuple(sorted(map(repr, second_keys - first_keys))),
    )
    return second, report


def promote_agreed(
    run: AnnotationRun, second: AnnotationRun, *, run_id: str | None = None
) -> AnnotationRun:
    """A new run in which annotations both passes made are ``VERIFIED``.

    Annotations only the first pass made keep their status. They are not
    rejected: a single disagreement is evidence, not a verdict, and throwing
    them away would quietly bias the corpus towards whatever two models
    happen to share.
    """
    if run.ontology_version != second.ontology_version:
        raise ValueError("cannot compare different ontology versions")
    second_keys = _semantic_keys(second)
    rejected = {_identity(item) for item in
                (*second.mentions, *second.predicates, *second.properties, *second.dialogue)
                if item.review_status == ReviewStatus.REJECTED}
    confirmed = {key for ident, key in second_keys.items() if ident not in rejected}
    first_keys = _semantic_keys(run)

    def promote(item):
        ident = _identity(item)
        if first_keys[ident] in confirmed and item.review_status != ReviewStatus.REJECTED:
            return replace(item, review_status=ReviewStatus.VERIFIED)
        return item

    return replace(
        run,
        run_id=run_id or f"{run.run_id}-reviewed",
        mentions=tuple(promote(m) for m in run.mentions),
        predicates=tuple(promote(p) for p in run.predicates),
        properties=tuple(promote(p) for p in run.properties),
        dialogue=tuple(promote(d) for d in run.dialogue),
    )
