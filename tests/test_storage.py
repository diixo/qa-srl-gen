"""The append-only store: codecs, invariants, versioning and streaming."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from semantic_corpus.documents import (
    AnnotationRun,
    DialogueAnnotation,
    Document,
    EntityMention,
    Predicate,
    Property,
    RelationEdge,
    TextSpan,
    Utterance,
)
from semantic_corpus.ontology import (
    Label,
    LabelSet,
    Mood,
    Polarity,
    Relation,
    ReviewStatus,
    SpeechAct,
    Stance,
)
from semantic_corpus.question_generator import QAExample, QAKind
from semantic_corpus.semantic_annotator import segment_document
from semantic_corpus.storage import (
    CorpusStore,
    DuplicateDocument,
    OntologyVersionMismatch,
    SplitConflict,
    StoreError,
    TABLES,
)
from semantic_corpus.storage.schema import decode_run, encode_run_rows

TEXT = "On Monday, Anna gave her dog Rex a red ball in Kyiv."


@pytest.fixture
def document() -> Document:
    return replace(segment_document(Document("d1", TEXT)), split="train")


@pytest.fixture
def run() -> AnnotationRun:
    anna = EntityMention(
        "m-anna",
        "d1",
        TextSpan(11, 15),
        "Anna",
        LabelSet.of(Label.PERSON, Label.NAMED_ENTITY),
        normalized_form="woman",
        confidence=0.9,
        source="teacher",
        review_status=ReviewStatus.VERIFIED,
    )
    ball = EntityMention(
        "m-ball", "d1", TextSpan(33, 43), "a red ball", LabelSet.of(Label.PHYSICAL_OBJECT)
    )
    predicate = Predicate(
        "p-gave",
        "d1",
        TextSpan(16, 20),
        "gave",
        "give",
        tense="past",
        aspect="simple",
        voice="active",
        polarity=Polarity.POSITIVE,
        modality=None,
    )
    red = Property(
        "pr-red",
        "d1",
        TextSpan(35, 38),
        "red",
        head="red",
        target_id="m-ball",
        degree="very",
        negated=True,
    )
    return AnnotationRun(
        run_id="r1", model_name="local-7b", prompt_version="p1", random_seed=7
    ).extended(
        mentions=[anna, ball],
        predicates=[predicate],
        properties=[red],
        relations=[RelationEdge("m-anna", Relation.AGENT_OF, "p-gave", question="Who?")],
    )


# -- codecs ----------------------------------------------------------------


def test_a_run_survives_a_round_trip_through_json(run):
    rows = encode_run_rows(run)
    restored = decode_run(
        rows["annotation_runs"][0], rows["spans"], rows["relations"], rows["dialogue"]
    )
    assert restored == run


def test_every_required_provenance_field_is_written(run, document, tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(document)
    store.add_run(run, document)

    header = next(store.read("annotation_runs"))
    for name in (
        "ontology_version",
        "model_name",
        "prompt_version",
        "random_seed",
        "generation_time",
        "synthetic",
    ):
        assert name in header, name

    row = next(store.read("documents"))
    for name in ("source_document", "source_hash", "source_license", "document_split"):
        assert name in row, name

    mention = next(r for r in store.read("spans") if r["record"] == "mention")
    for name in ("confidence", "review_status", "source", "normalized_form"):
        assert name in mention, name


def test_predicate_features_are_preserved(run, document, tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(document)
    store.add_run(run, document)
    restored = next(store.runs())
    predicate = restored.predicates[0]
    assert (predicate.tense, predicate.aspect, predicate.voice) == (
        "past",
        "simple",
        "active",
    )
    assert predicate.polarity is Polarity.POSITIVE


def test_property_parts_are_preserved(run, document, tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(document)
    store.add_run(run, document)
    prop = next(store.runs()).properties[0]
    assert (prop.head, prop.degree, prop.negated) == ("red", "very", True)
    assert prop.target_id == "m-ball"


def test_dialogue_annotations_round_trip(tmp_path):
    text = "Hi.\nWow!"
    document = Document(
        "d2",
        text,
        split="train",
        utterances=(
            Utterance("u0", "d2", 0, "A", TextSpan(0, 3)),
            Utterance("u1", "d2", 1, "B", TextSpan(4, 8)),
        ),
    )
    run = AnnotationRun(run_id="r").extended(
        dialogue=[
            DialogueAnnotation(
                utterance_id="u1",
                speech_acts=(SpeechAct.REACTION,),
                polarity=Polarity.POSITIVE,
                stance=Stance.SUPPORTIVE,
                mood=Mood.EXCLAMATIVE,
            )
        ]
    )
    store = CorpusStore(tmp_path / "s")
    store.add_document(document)
    store.add_run(run, document)
    restored = next(store.runs()).dialogue[0]
    assert restored.speech_acts == (SpeechAct.REACTION,)
    assert restored.stance is Stance.SUPPORTIVE
    assert restored.mood is Mood.EXCLAMATIVE


# -- invariants ------------------------------------------------------------


def test_annotations_are_append_only(run, document, tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(document)
    store.add_run(run, document)
    with pytest.raises(StoreError, match="append-only"):
        store.add_run(run, document)
    # A second opinion is a new run, and both survive.
    store.add_run(replace(run, run_id="r2"), document)
    assert {r.run_id for r in store.runs()} == {"r1", "r2"}


def test_ontologies_cannot_be_mixed(run, document, tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(document)
    with pytest.raises(OntologyVersionMismatch):
        store.add_run(replace(run, ontology_version="v2"), document)


def test_reopening_under_another_ontology_is_refused(tmp_path):
    CorpusStore(tmp_path / "s")
    with pytest.raises(OntologyVersionMismatch):
        CorpusStore(tmp_path / "s", ontology_version="v9")


def test_a_run_whose_offsets_do_not_match_is_refused(document, tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(document)
    wrong = AnnotationRun(run_id="bad").extended(
        mentions=[
            EntityMention(
                "m", "d1", TextSpan(11, 15), "Rex", LabelSet.of(Label.ANIMAL)
            )
        ]
    )
    with pytest.raises(StoreError, match="does not describe"):
        store.add_run(wrong, document)


def test_the_same_text_is_stored_once(document, tmp_path):
    store = CorpusStore(tmp_path / "s")
    assert store.add_document(document) is True
    assert store.add_document(replace(document, document_id="copy")) is False
    assert store.count("documents") == 1


def test_a_duplicate_under_another_id_can_be_made_an_error(document, tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(document)
    with pytest.raises(DuplicateDocument):
        store.add_document(
            replace(document, document_id="copy"), skip_duplicates=False
        )


def test_a_document_cannot_change_split(document, tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(document)
    moved = replace(document, text=TEXT + " More.", split="test")
    with pytest.raises(SplitConflict):
        store.add_document(moved)


def test_examples_are_deduplicated(document, tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(document)
    example = QAExample(
        context=TEXT, question="Who gave it?", answer="Anna.", document_id="d1"
    )
    assert store.add_examples([example]) == 1
    assert store.add_examples([example]) == 0
    assert store.count("qa_examples") == 1


def test_deduplication_survives_reopening(document, tmp_path):
    example = QAExample(
        context=TEXT, question="Who gave it?", answer="Anna.", document_id="d1"
    )
    store = CorpusStore(tmp_path / "s")
    store.add_document(document)
    store.add_examples([example])
    reopened = CorpusStore(tmp_path / "s")
    assert reopened.add_examples([example]) == 0


# -- versioning and reading ------------------------------------------------


def test_a_version_names_exactly_what_the_store_held(run, document, tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(document)
    store.add_run(run, document)
    first = store.publish_version("v1", notes="initial")

    assert first["run_ids"] == ["r1"]
    assert first["ontology_version"] == "v1"
    assert first["counts"]["documents"] == 1
    assert first["notes"] == "initial"

    store.add_run(replace(run, run_id="r2"), document)
    second = store.publish_version("v2")
    assert second["content_hash"] != first["content_hash"]
    assert len(store.versions()) == 2


def test_the_content_hash_is_stable_when_nothing_changes(run, document, tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(document)
    store.add_run(run, document)
    before = store.content_hash()
    store.publish_version("v1")
    # Publishing records a version; it must not alter the data it describes.
    assert store.content_hash() == before


def test_documents_come_back_with_their_passages(document, tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(document)
    restored = store.document("d1")
    assert restored is not None
    assert restored.text == TEXT
    assert len(restored.passages) == len(document.passages)
    assert restored.split == "train"


def test_reading_is_filtered_by_split(document, tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(document)
    store.add_document(replace(document, document_id="d2", text="Other text.", split="test"))
    assert [d.document_id for d in store.documents(split="train")] == ["d1"]
    assert [d.document_id for d in store.documents(split="test")] == ["d2"]


def test_examples_are_filtered_by_the_split_of_their_document(document, tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(document)
    store.add_document(replace(document, document_id="d2", text="Other.", split="test"))
    store.add_examples(
        [
            QAExample(context="x.", question="a?", answer="1.", document_id="d1"),
            QAExample(context="y.", question="b?", answer="2.", document_id="d2"),
        ]
    )
    assert [e.question for e in store.examples(split="train")] == ["a?"]
    assert [e.question for e in store.examples(split="test")] == ["b?"]


def test_an_unknown_table_is_an_error(tmp_path):
    store = CorpusStore(tmp_path / "s")
    with pytest.raises(StoreError, match="unknown table"):
        list(store.read("nonsense"))


def test_a_corrupt_line_names_its_location(document, tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(document)
    with store.path_for("documents").open("a", encoding="utf-8") as stream:
        stream.write("not json\n")
    with pytest.raises(StoreError, match="documents.jsonl:2"):
        list(store.read("documents"))


def test_every_table_has_a_file_name():
    assert len(set(TABLES.values())) == len(TABLES)
    assert all(name.endswith(".jsonl") for name in TABLES.values())
