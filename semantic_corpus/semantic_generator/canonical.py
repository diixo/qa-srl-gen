"""Converting a generated sentence into the canonical representation.

The generator knows more about its output than any annotator could: it chose
the entities, so every label, relation and offset is ground truth rather than
a claim. But downstream stages must not need to care where a record came
from — the storage layer stores one kind of row, and the question generator
reads one kind of input.

So a :class:`~.realization.RealizedSituation` becomes an ordinary
:class:`~...documents.Document` plus an :class:`~...documents.AnnotationRun`,
with two honest markers: ``synthetic=True`` on the run, and
``review_status=VERIFIED`` with ``confidence=1.0`` on each annotation,
because these facts are true by construction rather than by agreement.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence

from ..documents import (
    AnnotationRun,
    Document,
    EntityMention,
    ONTOLOGY_VERSION,
    Predicate,
    Property,
    RelationEdge,
    TextSpan,
)
from ..ontology import Label, Polarity, Relation, ReviewStatus
from .realization import RealizedSituation

__all__ = [
    "to_document",
    "to_annotation_run",
    "to_canonical",
    "iter_canonical",
    "combine",
    "GENERATOR_SOURCE",
]

#: Recorded as the ``source`` of every annotation this module produces.
GENERATOR_SOURCE = "semantic_generator"


def to_document(
    realized: RealizedSituation,
    *,
    document_id: str,
    split: str | None = None,
    license: str = "generated",
) -> Document:
    """Wrap a generated sentence as a document."""
    return Document(
        document_id=document_id,
        text=realized.text,
        source=GENERATOR_SOURCE,
        license=license,
        split=split,
        metadata={
            "frame": realized.situation.frame.lemma,
            "pattern": realized.pattern.name,
            "features": realized.situation.features.describe(),
            "omitted_slots": sorted(realized.omitted_slots),
        },
    )


def _aspect(realized: RealizedSituation) -> str | None:
    features = realized.situation.features
    parts = []
    if features.is_perfect:
        parts.append("perfect")
    if features.is_progressive:
        parts.append("progressive")
    return "+".join(parts) if parts else "simple"


def to_annotation_run(
    realized: RealizedSituation,
    document: Document,
    *,
    run_id: str,
    random_seed: int | None = None,
) -> AnnotationRun:
    """Build the annotation run describing *realized* inside *document*.

    Identifiers are derived from the document id and the slot name, so the
    same sentence always produces the same ids — reproducibility has to hold
    for the records as well as for the text.
    """
    situation = realized.situation
    frame = situation.frame
    slots = frame.slot_map

    predicate_id = f"{document.document_id}#pred"
    predicate = Predicate(
        predicate_id=predicate_id,
        document_id=document.document_id,
        span=TextSpan(realized.predicate.start_char, realized.predicate.end_char),
        exact_text=realized.predicate.text,
        lemma=frame.lemma,
        predicate_type=frame.predicate_type,
        tense=situation.features.tense,
        aspect=_aspect(realized),
        voice="passive" if realized.pattern.is_passive else "active",
        polarity=(
            Polarity.NEGATIVE if situation.features.is_negated else Polarity.POSITIVE
        ),
        modality=(
            situation.features.tense
            if situation.features.tense not in ("present", "past")
            else None
        ),
        confidence=1.0,
        source=GENERATOR_SOURCE,
        review_status=ReviewStatus.VERIFIED,
        omitted_slots=tuple(sorted(realized.omitted_slots)),
    )

    mentions: list[EntityMention] = []
    properties: list[Property] = []
    by_slot: dict[str, str] = {}

    for mention in realized.mentions:
        identifier = f"{document.document_id}#{mention.slot}"
        by_slot[mention.slot] = identifier
        span = TextSpan(mention.start_char, mention.end_char)
        if Label.PROPERTY in mention.labels:
            owner = mention.slot.rsplit(":", 1)[0]
            entity = situation.bindings.get(owner)
            properties.append(
                Property(
                    property_id=identifier,
                    document_id=document.document_id,
                    span=span,
                    exact_text=mention.text,
                    head=entity.property_head if entity else mention.text,
                    target_id=f"{document.document_id}#{owner}",
                    degree=entity.property_degree if entity else None,
                    negated=entity.property_negated if entity else False,
                    labels=mention.labels,
                    confidence=1.0,
                    source=GENERATOR_SOURCE,
                    review_status=ReviewStatus.VERIFIED,
                )
            )
            continue
        mentions.append(
            EntityMention(
                mention_id=identifier,
                document_id=document.document_id,
                span=span,
                exact_text=mention.text,
                labels=mention.labels,
                normalized_form=_normalized_form(mention.slot, situation),
                type_grounded=(
                    Label.NAMED_ENTITY not in mention.labels
                    or bool(situation.bindings[mention.slot.split(':')[0]].appositive)
                ),
                confidence=1.0,
                source=GENERATOR_SOURCE,
                review_status=ReviewStatus.VERIFIED,
            )
        )

    relations: list[RelationEdge] = []
    for edge in realized.relations:
        source_id = by_slot.get(edge.source_slot)
        if source_id is None:
            continue
        if edge.relation is Relation.INSTANCE_OF:
            # The target of an appositive link is the gloss, which has its own
            # span rather than being the predicate.
            target_id = by_slot.get(f"{edge.source_slot}:appositive")
            if target_id is None:
                continue
        elif edge.relation is Relation.PROPERTY_OF:
            # A property points at the entity it describes, not at the verb.
            target_id = by_slot.get(f"{edge.source_slot}:property")
            if target_id is None:
                continue
            source_id, target_id = target_id, source_id
        else:
            target_id = predicate_id
        relations.append(
            RelationEdge(
                source_id=source_id,
                relation=edge.relation,
                target_id=target_id,
                confidence=1.0,
                source=GENERATOR_SOURCE,
            )
        )

    run = AnnotationRun(
        run_id=run_id,
        ontology_version=ONTOLOGY_VERSION,
        model_name=GENERATOR_SOURCE,
        prompt_version="n/a",
        random_seed=random_seed,
        synthetic=True,
    )
    return run.extended(
        mentions=mentions,
        predicates=[predicate],
        properties=properties,
        relations=relations,
    )


def _normalized_form(slot: str, situation) -> str | None:
    """The entity's ontology type, when the generator knows one."""
    entity = situation.bindings.get(slot.split(":")[0])
    return entity.type_name if entity is not None else None


def to_canonical(
    realized: RealizedSituation,
    *,
    document_id: str,
    run_id: str | None = None,
    split: str | None = None,
    random_seed: int | None = None,
) -> tuple[Document, AnnotationRun]:
    """A generated sentence as a document and the run describing it."""
    document = to_document(realized, document_id=document_id, split=split)
    run = to_annotation_run(
        realized,
        document,
        run_id=run_id or f"{document_id}-generated",
        random_seed=random_seed,
    )
    return document, run


def iter_canonical(
    realizations: Iterable[RealizedSituation],
    *,
    prefix: str = "gen",
    split: str | None = None,
    random_seed: int | None = None,
):
    """Convert a stream of realisations, numbering the documents."""
    for index, realized in enumerate(realizations):
        yield to_canonical(
            realized,
            document_id=f"{prefix}-{index:06d}",
            split=split,
            random_seed=random_seed,
        )


def _shift(item, offset: int, suffix: str, document_id: str):
    """Move an annotation into a combined document, renaming its id."""
    from dataclasses import replace as _replace

    field_name = next(
        name
        for name in ("mention_id", "predicate_id", "property_id")
        if hasattr(item, name)
    )
    return _replace(
        item,
        span=item.span.shifted(offset),
        document_id=document_id,
        **{field_name: f"{suffix}#{getattr(item, field_name)}"},
    )


def combine(
    pieces: Sequence[tuple[Document, AnnotationRun]],
    *,
    document_id: str,
    separator: str = " ",
    run_id: str | None = None,
    split: str | None = None,
) -> tuple[Document, AnnotationRun]:
    """Join several one-sentence documents into one multi-sentence document.

    Questions whose answer needs two sentences cannot exist while every
    generated document is a single sentence. Joining them is not cosmetic:
    all offsets are shifted into the combined text and re-checked, so the
    merged record stays exactly as verifiable as the pieces were.
    """
    if not pieces:
        raise ValueError("nothing to combine")

    texts: list[str] = []
    cursor = 0
    mentions: list[EntityMention] = []
    predicates: list[Predicate] = []
    properties: list[Property] = []
    relations: list[RelationEdge] = []
    renamed: dict[str, str] = {}
    omitted: list[str] = []

    for index, (piece, run) in enumerate(pieces):
        if run.ontology_version != pieces[0][1].ontology_version:
            raise ValueError("cannot combine different ontology versions")
        if index:
            cursor += len(separator)
        offset = cursor
        texts.append(piece.text)
        cursor += len(piece.text)
        tag = f"{document_id}#s{index}"
        renamed = {}

        for group, sink in (
            (run.mentions, mentions),
            (run.predicates, predicates),
            (run.properties, properties),
        ):
            for item in group:
                moved = _shift(item, offset, tag, document_id)
                if isinstance(moved, Property) and moved.target_id is not None:
                    moved = replace(moved, target_id=f"{tag}#{moved.target_id}")
                identifier = next(
                    getattr(moved, name)
                    for name in ("mention_id", "predicate_id", "property_id")
                    if hasattr(moved, name)
                )
                original = next(
                    getattr(item, name)
                    for name in ("mention_id", "predicate_id", "property_id")
                    if hasattr(item, name)
                )
                renamed[original] = identifier
                sink.append(moved)

        for edge in run.relations:
            source = renamed.get(edge.source_id)
            target = renamed.get(edge.target_id)
            if source and target:
                relations.append(
                    replace(edge, source_id=source, target_id=target)
                )
        omitted.extend(str(s) for s in (piece.metadata.get("omitted_slots") or ()))

    combined = Document(
        document_id=document_id,
        text=separator.join(texts),
        source=GENERATOR_SOURCE,
        license="generated",
        split=split,
        metadata={
            "sentences": len(pieces),
            "omitted_slots": sorted(set(omitted)),
        },
    )
    merged = AnnotationRun(
        run_id=run_id or f"{document_id}-generated",
        ontology_version=pieces[0][1].ontology_version,
        model_name=GENERATOR_SOURCE,
        prompt_version="n/a",
        synthetic=all(run.synthetic for _, run in pieces),
    ).extended(
        mentions=mentions,
        predicates=predicates,
        properties=properties,
        relations=relations,
    )
    problems = merged.validate_against(combined)
    if problems:
        raise AssertionError(
            "combining moved an annotation off its text: " + "; ".join(problems)
        )
    return combined, merged
