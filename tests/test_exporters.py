"""Exports: SFT loss masking, the lossy BIO projection, and the report."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from semantic_corpus.documents import (
    AnnotationRun,
    Document,
    EntityMention,
    Predicate,
    TextSpan,
)
from semantic_corpus.exporters import (
    build_report,
    check_record,
    export_bio,
    export_sft,
    read_jsonl,
    tag_document,
    to_record,
    write_jsonl,
    write_report,
)
from semantic_corpus.exporters.sft import ANSWER_TAG, CONTEXT_TAG, export_sft_splits
from semantic_corpus.ontology import Label, LabelSet
from semantic_corpus.question_generator import QAExample, QAKind
from semantic_corpus.semantic_generator import Generator
from semantic_corpus.semantic_generator.canonical import to_canonical
from semantic_corpus.semantic_generator.generator import DEFAULT_PARADIGMS
from semantic_corpus.question_generator import QuestionGenerator
from semantic_corpus.storage import CorpusStore

TEXT = "On Monday, Anna gave her dog Rex a red ball in Kyiv."


@pytest.fixture
def example() -> QAExample:
    return QAExample(
        context=TEXT,
        question="Who gave Rex a red ball?",
        answer="Anna.",
        kind=QAKind.ATOMIC,
        document_id="d1",
    )


# -- SFT -------------------------------------------------------------------


def test_the_record_is_split_where_the_loss_starts(example):
    record = to_record(example)
    assert record["prompt"].endswith(f"<{ANSWER_TAG}>\n")
    assert record["completion"].startswith("Anna.")
    assert record["text"] == record["prompt"] + record["completion"]
    assert check_record(record) == []


def test_the_prompt_holds_the_whole_context_and_question(example):
    record = to_record(example)
    assert TEXT in record["prompt"]
    assert example.question in record["prompt"]
    # The answer must be nowhere in the prompt, or the loss mask is a lie.
    assert "Anna." not in record["prompt"]


def test_the_layout_is_the_one_the_handoff_specifies(example):
    text = to_record(example)["text"]
    assert text.index(f"<{CONTEXT_TAG}>") < text.index("<question>") < text.index(
        f"<{ANSWER_TAG}>"
    )


def test_a_record_that_would_corrupt_training_is_rejected():
    assert check_record({"prompt": "a", "completion": "b", "text": "ab"}) == []
    assert any(
        "prompt + completion" in p
        for p in check_record({"prompt": "a", "completion": "b", "text": "zz"})
    )
    assert any(
        "empty" in p for p in check_record({"prompt": "a", "completion": " ", "text": "a "})
    )


def test_exporting_writes_one_json_object_per_line(example, tmp_path):
    path = tmp_path / "sft.jsonl"
    assert export_sft([example, replace(example, question="What was given?")], path) == 2
    rows = list(read_jsonl(path))
    assert len(rows) == 2
    assert all("prompt" in row and "completion" in row for row in rows)


def test_split_export_writes_one_file_each(example, tmp_path):
    counts = export_sft_splits({"train": [example], "dev": []}, tmp_path)
    assert counts == {"train": 1, "dev": 0}
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "dev.jsonl").exists()


def test_output_is_byte_stable_across_runs(example, tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    export_sft([example], first)
    export_sft([example], second)
    assert first.read_bytes() == second.read_bytes()


# -- BIO -------------------------------------------------------------------


def tagged_fixture() -> tuple[Document, AnnotationRun]:
    document = Document("d1", TEXT)
    run = AnnotationRun(run_id="r").extended(
        mentions=[
            EntityMention(
                "m1",
                "d1",
                TextSpan(11, 15),
                "Anna",
                LabelSet.of(Label.PERSON, Label.NAMED_ENTITY),
            ),
            EntityMention(
                "m2", "d1", TextSpan(33, 43), "a red ball", LabelSet.of(Label.PHYSICAL_OBJECT)
            ),
        ],
        predicates=[Predicate("p", "d1", TextSpan(16, 20), "gave", "give")],
    )
    return document, run


def test_tagging_lines_up_with_the_tokens():
    document, run = tagged_fixture()
    tagging = tag_document(document, run)
    assert len(tagging.tokens) == len(tagging.tags)
    assert "Anna" in tagging.tokens
    index = tagging.tokens.index("Anna")
    assert tagging.tags[index] == "B-PERSON"
    assert tagging.tags[tagging.tokens.index("gave")] == "B-ACTION"


def test_bilou_marks_single_token_and_final_tokens():
    document, run = tagged_fixture()
    tagging = tag_document(document, run, scheme="BILOU")
    assert tagging.tags[tagging.tokens.index("Anna")] == "U-PERSON"
    ball = tagging.tokens.index("ball")
    assert tagging.tags[ball] == "L-PHYSICAL_OBJECT"


def test_the_projection_reports_what_it_dropped():
    """A multi-label span cannot be a single tag, and the loss is counted."""
    document, run = tagged_fixture()
    from semantic_corpus.exporters.bio import BioReport

    report = BioReport()
    tag_document(document, run, report=report)
    # Anna is PERSON + NAMED_ENTITY; only one label survives.
    assert report.dropped_labels == 1
    assert not report.is_lossless
    assert "kept only" in " ".join(report.dropped_examples)


def test_overlapping_spans_are_dropped_not_silently_merged():
    document = Document("d", TEXT)
    run = AnnotationRun(run_id="r").extended(
        mentions=[
            EntityMention("a", "d", TextSpan(11, 20), TEXT[11:20], LabelSet.of(Label.PERSON)),
            EntityMention("b", "d", TextSpan(16, 28), TEXT[16:28], LabelSet.of(Label.PERSON)),
        ]
    )
    from semantic_corpus.exporters.bio import BioReport

    report = BioReport()
    tag_document(document, run, report=report)
    assert report.dropped_overlapping == 1
    assert report.tagged_spans == 1


def test_an_unknown_scheme_is_refused():
    document, run = tagged_fixture()
    with pytest.raises(ValueError, match="unknown scheme"):
        tag_document(document, run, scheme="IOBES")  # type: ignore[arg-type]


def test_exporting_bio_returns_the_cost(tmp_path):
    document, run = tagged_fixture()
    report = export_bio([(document, run)], tmp_path / "bio.jsonl")
    assert report.documents == 1
    assert report.tagged_spans == 3
    rows = list(read_jsonl(tmp_path / "bio.jsonl"))
    assert len(rows[0]["tokens"]) == len(rows[0]["tags"])


# -- report ----------------------------------------------------------------


def build_corpus(tmp_path, *, count: int = 20, split: str = "train") -> CorpusStore:
    store = CorpusStore(tmp_path / "store")
    generator = Generator(seed=31, split=split if split == "train" else None)
    questions = QuestionGenerator(paradigms=DEFAULT_PARADIGMS, seed=1)
    for index, realized in enumerate(generator.generate(count)):
        document, run = to_canonical(
            realized, document_id=f"{split}-{index:04d}", split=split
        )
        store.add_document(document)
        store.add_run(run, document)
        store.add_examples(questions.for_document(document, run))
    return store


def test_a_well_split_corpus_reports_clean(tmp_path):
    store = build_corpus(tmp_path)
    held_out = Generator(seed=1).pool.split("test").surface_forms
    report = build_report(store, held_out_names=held_out)
    assert report.is_clean, str(report)
    assert report.documents_by_split["train"] == 20
    assert report.examples_by_split["train"] > 0
    assert report.synthetic_runs == 20


def test_the_report_counts_the_mix(tmp_path):
    report = build_report(build_corpus(tmp_path))
    assert report.examples_by_kind
    assert 0.0 < report.refusal_share < 1.0
    assert report.labels["PERSON"] > 0
    assert report.predicates


def test_a_held_out_name_in_training_is_leakage(tmp_path):
    store = build_corpus(tmp_path)
    # Pick a name the corpus actually uses, then declare it held out.
    name = next(
        row["exact_text"]
        for row in store.read("spans")
        if row.get("record") == "mention" and "NAMED_ENTITY" in row.get("labels", ())
    )
    report = build_report(store, held_out_names=[name])
    assert not report.is_clean
    assert any(f.kind == "held-out-name" for f in report.leakage)


def test_the_same_text_in_two_splits_is_leakage(tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(Document("a", "Shared text.", split="train"))
    # A second store row with the same hash can only arrive by hand, which is
    # exactly the corruption the check exists to find.
    with store.path_for("documents").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "document_id": "b",
                    "text": "Shared text.",
                    "source_document": "x",
                    "source_hash": Document("b", "Shared text.").sha256,
                    "source_license": "unknown",
                    "document_split": "test",
                    "metadata": {},
                },
                sort_keys=True,
            )
            + "\n"
        )
    report = build_report(CorpusStore(tmp_path / "s"))
    assert any(f.kind == "duplicate-text" for f in report.leakage)


def test_an_identical_example_in_two_splits_is_leakage(tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(Document("a", "One.", split="train"))
    store.add_document(Document("b", "Two.", split="test"))
    store.add_examples(
        [
            QAExample(context="Same.", question="q?", answer="a.", document_id="a"),
            QAExample(context="Same.", question="q?", answer="a.", document_id="b"),
        ],
        deduplicate=False,
    )
    report = build_report(store)
    assert any(f.kind == "duplicate-example" for f in report.leakage)


def test_a_repeated_pair_under_different_contexts_is_not_fatal(tmp_path):
    """The model still has to read the context, so this is a warning."""
    store = CorpusStore(tmp_path / "s")
    store.add_document(Document("a", "One.", split="train"))
    store.add_document(Document("b", "Two.", split="test"))
    store.add_examples(
        [
            QAExample(context="One.", question="q?", answer="a.", document_id="a"),
            QAExample(context="Two.", question="q?", answer="a.", document_id="b"),
        ]
    )
    report = build_report(store)
    assert report.is_clean
    assert report.repeated_qa
    assert report.to_json()["repeated_qa_count"] == 1


def test_a_run_under_another_ontology_is_reported(tmp_path):
    store = CorpusStore(tmp_path / "s")
    store.add_document(Document("a", "One.", split="train"))
    with store.path_for("annotation_runs").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"run_id": "x", "ontology_version": "v2"}) + "\n")
    report = build_report(CorpusStore(tmp_path / "s"))
    assert any(f.kind == "ontology-mismatch" for f in report.leakage)


def test_a_report_can_be_saved_and_read(tmp_path):
    report = build_report(build_corpus(tmp_path))
    path = write_report(report, tmp_path / "report.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["ontology_version"] == "v1"
    assert "examples_by_kind" in data
    assert "refusal_share" in data


# -- the whole pipeline ----------------------------------------------------


def test_generate_store_export_report(tmp_path):
    """One pass of everything, with the invariants checked at each hop."""
    store = build_corpus(tmp_path, count=15)
    version = store.publish_version("v1")
    assert version["counts"]["documents"] == 15

    # Annotations still describe their documents after a round trip to disk.
    for document in store.documents():
        for run in store.runs(document_id=document.document_id):
            assert run.validate_against(document) == []

    sft = tmp_path / "sft.jsonl"
    written = export_sft(store.examples(split="train"), sft)
    assert written > 0
    for row in read_jsonl(sft):
        assert check_record(row) == []

    pairs = [
        (document, run)
        for document in store.documents()
        for run in store.runs(document_id=document.document_id)
    ]
    bio = export_bio(pairs, tmp_path / "bio.jsonl")
    assert bio.documents == 15
    assert bio.dropped_overlapping == 0

    report = build_report(store)
    assert report.is_clean
    assert report.content_hash == version["content_hash"]
