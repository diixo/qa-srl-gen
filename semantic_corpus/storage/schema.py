"""What the store holds, and how each record is written down.

The handoff specifies SQLite; the user has ruled out both SQLite and Parquet,
so the store is a directory of append-only JSONL files — one per record kind,
which is what the nine tables become. That choice costs indexing and gains
two things that matter more here: the format needs no dependency, and a
corpus stays readable with `head` when something looks wrong.

Codecs live here rather than on the model classes so that
:mod:`..documents` stays free of storage concerns. The encoding is explicit
and total: every field the handoff requires to be stored has a line in one
of these functions, and :func:`decode_*` reconstructs the object exactly, so
a round trip through disk is lossless.

Two invariants are enforced at this level because they cannot be repaired
later:

* **the ontology version travels with the data.** Annotations made under
  different label inventories mean different things, and merging them
  silently produces a corpus nobody can interpret;
* **records are only ever appended.** Re-annotating writes a new run; it
  never edits an old one.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..documents import (
    AnnotationRun,
    DialogueAnnotation,
    Document,
    EntityMention,
    ONTOLOGY_VERSION,
    Passage,
    Predicate,
    Property,
    RelationEdge,
    TextSpan,
    Utterance,
)
from ..ontology import (
    Label,
    LabelSet,
    MarkerForm,
    MarkerFunction,
    Mood,
    Polarity,
    Relation,
    ReviewStatus,
    SpeechAct,
    Stance,
)

__all__ = [
    "TABLES",
    "STORE_FORMAT",
    "OntologyVersionMismatch",
    "encode_document",
    "decode_document",
    "encode_passage",
    "decode_passage",
    "encode_utterance",
    "decode_utterance",
    "encode_run",
    "decode_run",
    "encode_span",
    "decode_span",
    "encode_qa_example",
    "decode_qa_example",
    "encode_dataset_version",
]

#: The format version of the store layout itself, distinct from the ontology
#: version. Bumped if the file names or record shapes change.
STORE_FORMAT = "jsonl-v1"

#: File name per record kind. These are the handoff's nine tables, flattened:
#: ``spans`` holds mentions, predicates and properties, each tagged with its
#: ``record`` kind, because they share a shape and are always read together.
TABLES: Mapping[str, str] = {
    "documents": "documents.jsonl",
    "passages": "passages.jsonl",
    "utterances": "utterances.jsonl",
    "annotation_runs": "annotation_runs.jsonl",
    "spans": "spans.jsonl",
    "relations": "relations.jsonl",
    "dialogue": "dialogue.jsonl",
    "qa_examples": "qa_examples.jsonl",
    "dataset_versions": "dataset_versions.jsonl",
}


class OntologyVersionMismatch(ValueError):
    """Raised when a record would join data labelled under another ontology."""


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------


def encode_span(span: TextSpan) -> dict[str, Any]:
    return span.to_json()


def decode_span(data: Mapping[str, Any]) -> TextSpan:
    return TextSpan(
        start_char=int(data["start_char"]),
        end_char=int(data["end_char"]),
        start_token=data.get("start_token"),
        end_token=data.get("end_token"),
    )


def _labels(labels: LabelSet) -> list[str]:
    return [str(label) for label in labels.ordered]


def _label_set(names: Any) -> LabelSet:
    return LabelSet(frozenset(Label(name) for name in names or ()))


# ---------------------------------------------------------------------------
# Documents, passages, utterances
# ---------------------------------------------------------------------------


def encode_document(document: Document) -> dict[str, Any]:
    """A document row, carrying the provenance the handoff lists."""
    return {
        "document_id": document.document_id,
        "text": document.text,
        "source_document": document.source,
        "source_hash": document.sha256,
        "source_license": document.license,
        "document_split": document.split,
        "metadata": dict(document.metadata),
    }


def decode_document(data: Mapping[str, Any]) -> Document:
    return Document(
        document_id=data["document_id"],
        text=data["text"],
        source=data.get("source_document", "unknown"),
        license=data.get("source_license", "unknown"),
        split=data.get("document_split"),
        metadata=dict(data.get("metadata") or {}),
    )


def encode_passage(passage: Passage) -> dict[str, Any]:
    return {
        "passage_id": passage.passage_id,
        "document_id": passage.document_id,
        "span": encode_span(passage.span),
        "sentence_spans": [encode_span(s) for s in passage.sentence_spans],
    }


def decode_passage(data: Mapping[str, Any]) -> Passage:
    return Passage(
        passage_id=data["passage_id"],
        document_id=data["document_id"],
        span=decode_span(data["span"]),
        sentence_spans=tuple(decode_span(s) for s in data.get("sentence_spans", ())),
    )


def encode_utterance(utterance: Utterance) -> dict[str, Any]:
    return {
        "utterance_id": utterance.utterance_id,
        "document_id": utterance.document_id,
        "turn_index": utterance.turn_index,
        "speaker": utterance.speaker,
        "span": encode_span(utterance.span),
    }


def decode_utterance(data: Mapping[str, Any]) -> Utterance:
    return Utterance(
        utterance_id=data["utterance_id"],
        document_id=data["document_id"],
        turn_index=int(data["turn_index"]),
        speaker=data["speaker"],
        span=decode_span(data["span"]),
    )


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


def _encode_mention(item: EntityMention, run_id: str) -> dict[str, Any]:
    return {
        "record": "mention",
        "run_id": run_id,
        "id": item.mention_id,
        "document_id": item.document_id,
        "span": encode_span(item.span),
        "exact_text": item.exact_text,
        "labels": _labels(item.labels),
        "head_token": item.head_token,
        "normalized_form": item.normalized_form,
        "confidence": item.confidence,
        "source": item.source,
        "review_status": str(item.review_status),
    }


def _encode_predicate(item: Predicate, run_id: str) -> dict[str, Any]:
    return {
        "record": "predicate",
        "run_id": run_id,
        "id": item.predicate_id,
        "document_id": item.document_id,
        "span": encode_span(item.span),
        "exact_text": item.exact_text,
        "lemma": item.lemma,
        "predicate_type": str(item.predicate_type),
        "tense": item.tense,
        "aspect": item.aspect,
        "voice": item.voice,
        "polarity": str(item.polarity),
        "modality": item.modality,
        "confidence": item.confidence,
        "source": item.source,
        "review_status": str(item.review_status),
    }


def _encode_property(item: Property, run_id: str) -> dict[str, Any]:
    return {
        "record": "property",
        "run_id": run_id,
        "id": item.property_id,
        "document_id": item.document_id,
        "span": encode_span(item.span),
        "exact_text": item.exact_text,
        "head": item.head,
        "target_id": item.target_id,
        "degree": item.degree,
        "negated": item.negated,
        "labels": _labels(item.labels),
        "confidence": item.confidence,
        "source": item.source,
        "review_status": str(item.review_status),
    }


def _decode_span_record(data: Mapping[str, Any]):
    kind = data["record"]
    span = decode_span(data["span"])
    common = {
        "document_id": data["document_id"],
        "span": span,
        "exact_text": data["exact_text"],
        "confidence": data.get("confidence"),
        "source": data.get("source", "unknown"),
        "review_status": ReviewStatus(data.get("review_status", "UNREVIEWED")),
    }
    if kind == "mention":
        return EntityMention(
            mention_id=data["id"],
            labels=_label_set(data.get("labels")),
            head_token=data.get("head_token"),
            normalized_form=data.get("normalized_form"),
            **common,
        )
    if kind == "predicate":
        return Predicate(
            predicate_id=data["id"],
            lemma=data["lemma"],
            predicate_type=Label(data.get("predicate_type", "ACTION")),
            tense=data.get("tense"),
            aspect=data.get("aspect"),
            voice=data.get("voice"),
            polarity=Polarity(data.get("polarity", "POSITIVE")),
            modality=data.get("modality"),
            **common,
        )
    if kind == "property":
        return Property(
            property_id=data["id"],
            head=data["head"],
            target_id=data.get("target_id"),
            degree=data.get("degree"),
            negated=bool(data.get("negated", False)),
            labels=_label_set(data.get("labels")),
            **common,
        )
    raise ValueError(f"unknown span record kind: {kind!r}")


def _encode_relation(edge: RelationEdge, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "source_id": edge.source_id,
        "relation": str(edge.relation),
        "target_id": edge.target_id,
        "question": edge.question,
        "confidence": edge.confidence,
        "source": edge.source,
    }


def _decode_relation(data: Mapping[str, Any]) -> RelationEdge:
    return RelationEdge(
        source_id=data["source_id"],
        relation=Relation(data["relation"]),
        target_id=data["target_id"],
        question=data.get("question"),
        confidence=data.get("confidence"),
        source=data.get("source", "unknown"),
    )


def _encode_dialogue(item: DialogueAnnotation, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "utterance_id": item.utterance_id,
        "speech_acts": [str(a) for a in item.speech_acts],
        "polarity": str(item.polarity) if item.polarity else None,
        "stance": str(item.stance) if item.stance else None,
        "mood": str(item.mood) if item.mood else None,
        "marker_form": str(item.marker_form) if item.marker_form else None,
        "marker_functions": [str(f) for f in item.marker_functions],
        "confidence": item.confidence,
        "source": item.source,
        "review_status": str(item.review_status),
    }


def _decode_dialogue(data: Mapping[str, Any]) -> DialogueAnnotation:
    return DialogueAnnotation(
        utterance_id=data["utterance_id"],
        speech_acts=tuple(SpeechAct(a) for a in data.get("speech_acts", ())),
        polarity=Polarity(data["polarity"]) if data.get("polarity") else None,
        stance=Stance(data["stance"]) if data.get("stance") else None,
        mood=Mood(data["mood"]) if data.get("mood") else None,
        marker_form=MarkerForm(data["marker_form"]) if data.get("marker_form") else None,
        marker_functions=tuple(
            MarkerFunction(f) for f in data.get("marker_functions", ())
        ),
        confidence=data.get("confidence"),
        source=data.get("source", "unknown"),
        review_status=ReviewStatus(data.get("review_status", "UNREVIEWED")),
    )


def encode_run(run: AnnotationRun) -> dict[str, Any]:
    """The run header. Its annotations are written to their own tables."""
    return {
        "run_id": run.run_id,
        "generation_time": run.created_at,
        "ontology_version": run.ontology_version,
        "model_name": run.model_name,
        "prompt_version": run.prompt_version,
        "random_seed": run.random_seed,
        "synthetic": run.synthetic,
        "counts": {
            "mentions": len(run.mentions),
            "predicates": len(run.predicates),
            "properties": len(run.properties),
            "relations": len(run.relations),
            "dialogue": len(run.dialogue),
        },
    }


def encode_run_rows(run: AnnotationRun) -> dict[str, list[dict[str, Any]]]:
    """Every row a run contributes, grouped by the table it belongs in."""
    return {
        "annotation_runs": [encode_run(run)],
        "spans": [
            *[_encode_mention(m, run.run_id) for m in run.mentions],
            *[_encode_predicate(p, run.run_id) for p in run.predicates],
            *[_encode_property(p, run.run_id) for p in run.properties],
        ],
        "relations": [_encode_relation(e, run.run_id) for e in run.relations],
        "dialogue": [_encode_dialogue(d, run.run_id) for d in run.dialogue],
    }


def decode_run(
    header: Mapping[str, Any],
    spans: list[Mapping[str, Any]],
    relations: list[Mapping[str, Any]],
    dialogue: list[Mapping[str, Any]],
) -> AnnotationRun:
    """Rebuild a run from its header and the rows that belong to it."""
    mentions, predicates, properties = [], [], []
    for row in spans:
        item = _decode_span_record(row)
        if isinstance(item, EntityMention):
            mentions.append(item)
        elif isinstance(item, Predicate):
            predicates.append(item)
        else:
            properties.append(item)
    return AnnotationRun(
        run_id=header["run_id"],
        created_at=header.get("generation_time", ""),
        ontology_version=header.get("ontology_version", ONTOLOGY_VERSION),
        model_name=header.get("model_name", "unknown"),
        prompt_version=header.get("prompt_version", "unknown"),
        random_seed=header.get("random_seed"),
        synthetic=bool(header.get("synthetic", False)),
        mentions=tuple(mentions),
        predicates=tuple(predicates),
        properties=tuple(properties),
        relations=tuple(_decode_relation(r) for r in relations),
        dialogue=tuple(_decode_dialogue(d) for d in dialogue),
    )


# ---------------------------------------------------------------------------
# QA examples and dataset versions
# ---------------------------------------------------------------------------


def encode_qa_example(example, *, generation_run_id: str | None = None) -> dict[str, Any]:
    row = example.to_json()
    row["generation_run_id"] = generation_run_id
    return row


def decode_qa_example(data: Mapping[str, Any]):
    from ..question_generator.answers import QAExample, QAKind

    return QAExample(
        context=data["context"],
        question=data["question"],
        answer=data["answer"],
        kind=QAKind(data.get("kind", "atomic")),
        answerable=bool(data.get("answerable", True)),
        document_id=data.get("document_id", ""),
        run_id=data.get("run_id"),
        evidence=tuple(decode_span(s) for s in data.get("evidence", ())),
        metadata=dict(data.get("metadata") or {}),
    )


def encode_dataset_version(
    *,
    version_id: str,
    created_at: str,
    ontology_version: str,
    generator_version: str,
    run_ids: list[str],
    counts: Mapping[str, int],
    content_hash: str,
    notes: str = "",
) -> dict[str, Any]:
    """A manifest naming exactly what a published dataset contains.

    Without one, an export cannot be rebuilt or even identified later: the
    store keeps growing, so "everything in the store" names a different
    dataset every day.
    """
    return {
        "version_id": version_id,
        "created_at": created_at,
        "ontology_version": ontology_version,
        "generator_version": generator_version,
        "store_format": STORE_FORMAT,
        "run_ids": sorted(run_ids),
        "counts": dict(counts),
        "content_hash": content_hash,
        "notes": notes,
    }
