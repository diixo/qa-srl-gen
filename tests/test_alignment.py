"""Ingestion, candidate extraction, alignment and the verification pass."""

from __future__ import annotations

import json

import pytest

from semantic_corpus.documents import AnnotationRun, Document, TextSpan
from semantic_corpus.ontology import Label, ReviewStatus
from semantic_corpus.semantic_annotator import (
    CandidateResources,
    LocationError,
    ProposedAnnotation,
    ScriptedTeacher,
    TeacherResponse,
    align_response,
    annotate_document,
    assign_split,
    deduplicate,
    extract_candidates,
    ingest,
    locate,
    promote_agreed,
    read_dialogue_jsonl,
    read_jsonl_documents,
    read_text_file,
    segment_dialogue,
    segment_document,
    tokenize,
    verify_independently,
    verify_structure,
)

TEXT = "On Monday, Anna gave her dog Rex a red ball in Kyiv."


def ann(text: str, *labels: str, **kwargs) -> ProposedAnnotation:
    return ProposedAnnotation(text=text, labels=labels, **kwargs)


@pytest.fixture
def document() -> Document:
    return segment_document(Document("d1", TEXT))


# -- ingestion -------------------------------------------------------------


def test_txt_becomes_one_document(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text(TEXT, encoding="utf-8")
    document = read_text_file(path)
    assert document.document_id == "sample"
    assert document.text == TEXT


def test_jsonl_documents_keep_their_provenance(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(
        json.dumps(
            {
                "document_id": "book-00042",
                "text": TEXT,
                "source": "bookcorpus",
                "license": "unknown",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    document = next(iter(read_jsonl_documents(path)))
    assert (document.document_id, document.source) == ("book-00042", "bookcorpus")


def test_a_record_without_text_is_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"document_id": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no 'text' field"):
        list(read_jsonl_documents(path))


def test_duplicates_are_dropped_by_content_not_by_id():
    documents = [Document("a", TEXT), Document("b", TEXT), Document("c", "Other.")]
    kept = list(deduplicate(documents))
    assert [d.document_id for d in kept] == ["a", "c"]


def test_splits_are_stable_across_runs_and_processes():
    first = {f"doc-{i}": assign_split(f"doc-{i}") for i in range(200)}
    second = {f"doc-{i}": assign_split(f"doc-{i}") for i in range(200)}
    assert first == second
    assert set(first.values()) == {"train", "dev", "test"}


def test_split_shares_are_respected():
    assignments = [assign_split(f"d{i}", dev_share=0.2, test_share=0.2) for i in range(2000)]
    assert 0.15 < assignments.count("test") / len(assignments) < 0.25
    assert 0.15 < assignments.count("dev") / len(assignments) < 0.25


def test_passages_overlap_and_keep_global_offsets():
    text = "One. Two. Three. Four."
    document = segment_document(Document("d", text), window=2, stride=1)
    assert len(document.passages) > 1
    for passage in document.passages:
        assert passage.span.text_in(document.text) == passage.text_in(document)
    # Consecutive windows share a sentence.
    assert document.passages[0].span.overlaps(document.passages[1].span)


def test_a_passage_covers_its_own_sentences():
    document = segment_document(Document("d", "One. Two. Three."), window=2)
    for passage in document.passages:
        for sentence in passage.sentence_spans:
            assert passage.span.contains(sentence)


def test_dialogue_turns_keep_speakers_order_and_offsets(tmp_path):
    path = tmp_path / "dialog.jsonl"
    path.write_text(
        json.dumps(["I finally got the job.", "Wow, that's great!"]) + "\n",
        encoding="utf-8",
    )
    document = next(iter(read_dialogue_jsonl(path)))
    assert [u.speaker for u in document.utterances] == ["A", "B"]
    assert [u.turn_index for u in document.utterances] == [0, 1]
    assert document.utterances[1].text_in(document) == "Wow, that's great!"
    assert document.is_dialogue


def test_a_dialogue_passage_reaches_back_for_context(tmp_path):
    path = tmp_path / "dialog.jsonl"
    path.write_text(json.dumps(["A one.", "B two.", "C three."]) + "\n", encoding="utf-8")
    document = segment_dialogue(next(iter(read_dialogue_jsonl(path))), context_turns=1)
    last = document.passages[-1]
    assert "B two." in last.text_in(document)
    assert last.text_in(document).endswith("C three.")


def test_ingest_deduplicates_then_splits_then_segments(tmp_path):
    (tmp_path / "a.txt").write_text("One. Two. Three.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("One. Two. Three.", encoding="utf-8")
    (tmp_path / "c.txt").write_text("Totally different text.", encoding="utf-8")
    documents = list(ingest(sorted(tmp_path.glob("*.txt"))))
    assert [d.document_id for d in documents] == ["a", "c"]
    assert all(d.split in ("train", "dev", "test") for d in documents)
    assert all(d.passages for d in documents)


# -- candidates ------------------------------------------------------------


def test_tokens_carry_their_offsets():
    tokens = tokenize("Anna gave Rex.")
    assert [t.text for t in tokens] == ["Anna", "gave", "Rex"]
    assert all("Anna gave Rex."[t.start_char : t.end_char] == t.text for t in tokens)


def test_tokenize_respects_an_offset():
    tokens = tokenize("Anna gave", offset=10)
    assert tokens[0].start_char == 10


def test_candidates_are_located_exactly(document):
    for candidate in extract_candidates(document, document.passages[0]):
        candidate.validate_against(document)


def test_candidates_carry_their_evidence(document):
    resources = CandidateResources.load()
    found = {c.exact_text: c for c in extract_candidates(document, None, resources)}
    assert "capitalisation" in found["Anna"].evidence
    assert "inflections" in found["gave"].evidence
    assert "give" in found["gave"].lemmas


def test_the_dictionary_only_proposes():
    """A noun that also has a verb reading is proposed as both, not decided."""
    resources = CandidateResources.load()
    document = Document("d", "The ball rolled.")
    found = {c.exact_text: c for c in extract_candidates(document, None, resources)}
    assert Label.ACTION in found["ball"].proposed_labels
    assert Label.PHYSICAL_OBJECT in found["ball"].proposed_labels
    # ...and the evidence that makes the noun reading likelier is preserved.
    assert found["ball"].tag_counts.get("noun", 0) >= found["ball"].tag_counts.get("verb", 0)


def test_ambiguous_lemmas_are_all_reported():
    """The scrape gives 'ran' both 'run' and a junk entry; neither is hidden."""
    resources = CandidateResources.load()
    document = Document("d", "She ran home.")
    found = {c.exact_text: c for c in extract_candidates(document, None, resources)}
    assert "run" in found["ran"].lemmas
    assert len(found["ran"].lemmas) > 1


def test_function_words_are_not_candidates(document):
    texts = {c.exact_text for c in extract_candidates(document, None)}
    assert "the" not in texts
    assert "a" not in texts


def test_proposed_label_sets_are_always_well_formed():
    resources = CandidateResources.load()
    document = Document("d", TEXT)
    for candidate in extract_candidates(document, None, resources):
        assert candidate.proposed_labels.problems() == []


# -- locating --------------------------------------------------------------


def test_locate_finds_an_exact_quote():
    assert locate("Anna", TEXT) == TextSpan(11, 15)


def test_locate_honours_the_occurrence_index():
    text = "the dog saw the dog"
    assert locate("the dog", text, occurrence=0).start_char == 0
    assert locate("the dog", text, occurrence=1).start_char == 12


def test_locate_tolerates_reflowed_whitespace():
    text = "Anna  gave\nRex a ball."
    span = locate("Anna gave Rex", text)
    assert span.text_in(text) == "Anna  gave\nRex"


def test_locate_refuses_anything_else():
    with pytest.raises(LocationError):
        locate("Berlin", TEXT)
    with pytest.raises(LocationError):
        locate("anna", TEXT), "case differences change what the span says"
    with pytest.raises(LocationError):
        locate("", TEXT)


def test_locate_is_confined_to_its_window():
    text = "Anna here. Anna there."
    window = TextSpan(11, 22)
    assert locate("Anna", text, window=window).start_char == 11


# -- alignment -------------------------------------------------------------


def test_alignment_sorts_annotations_into_kinds(document):
    response = TeacherResponse(
        annotations=(
            ann("Anna", "PERSON", "NAMED_ENTITY"),
            ann("gave", "ACTION", lemma="give"),
            ann("red", "PROPERTY", head="red", degree="very", negated=True),
        )
    )
    result = align_response(response, document, run_id="r1")
    assert [m.exact_text for m in result.mentions] == ["Anna"]
    assert [p.exact_text for p in result.predicates] == ["gave"]
    assert [p.exact_text for p in result.properties] == ["red"]
    assert result.properties[0].degree == "very"
    assert result.properties[0].negated


def test_alignment_rejects_what_it_cannot_justify(document):
    response = TeacherResponse(
        annotations=(
            ann("Berlin", "LOCATION"),
            ann("Anna", "PERSON", "ANIMAL"),
            ann("Monday", "WEEKDAY"),
            ann("Kyiv"),
        )
    )
    result = align_response(response, document, run_id="r1")
    assert result.mentions == ()
    reasons = {r.annotation.text: r.reason for r in result.rejected}
    assert "not found" in reasons["Berlin"]
    assert "at most one entity type" in reasons["Anna"]
    assert "not in the ontology" in reasons["Monday"]
    assert "no labels" in reasons["Kyiv"]
    assert result.rejection_rate == 1.0


def test_alignment_records_the_source_text_not_the_quote():
    document = Document("d", "Anna  gave Rex.")
    response = TeacherResponse(annotations=(ann("Anna gave", "ACTION"),))
    result = align_response(response, document, run_id="r1")
    assert result.predicates[0].exact_text == "Anna  gave"
    result.predicates[0].validate_against(document)


def test_relations_are_resolved_to_ids(document):
    response = TeacherResponse(
        annotations=(
            ann("Anna", "PERSON", relation="AGENT_OF", target_text="gave"),
            ann("gave", "ACTION"),
        )
    )
    result = align_response(response, document, run_id="r1")
    assert len(result.relations) == 1
    edge = result.relations[0]
    ids = {m.mention_id for m in result.mentions} | {
        p.predicate_id for p in result.predicates
    }
    assert edge.source_id in ids and edge.target_id in ids


def test_a_relation_to_an_unannotated_target_is_rejected(document):
    response = TeacherResponse(
        annotations=(ann("Anna", "PERSON", relation="AGENT_OF", target_text="gave"),)
    )
    result = align_response(response, document, run_id="r1")
    assert result.relations == ()
    assert any("was not annotated" in r.reason for r in result.rejected)


def test_an_unknown_relation_is_rejected(document):
    response = TeacherResponse(
        annotations=(
            ann("Anna", "PERSON", relation="BEFRIENDS", target_text="gave"),
            ann("gave", "ACTION"),
        )
    )
    result = align_response(response, document, run_id="r1")
    assert any("unknown relation" in r.reason for r in result.rejected)


# -- the pipeline and verification ----------------------------------------


def scripted(*annotations, name="scripted") -> ScriptedTeacher:
    return ScriptedTeacher(
        default=TeacherResponse(annotations=tuple(annotations)), name=name
    )


def test_the_pipeline_produces_a_self_describing_run(document):
    teacher = scripted(ann("Anna", "PERSON"), ann("gave", "ACTION"), name="local-7b")
    outcome = annotate_document(document, teacher, run_id="run-1", random_seed=3)
    assert outcome.ok
    assert outcome.run.model_name == "local-7b"
    assert outcome.run.random_seed == 3
    assert outcome.run.ontology_version == "v1"
    assert outcome.run.prompt_version
    assert len(outcome.run) >= 2


def test_the_pipeline_shows_candidates_to_the_teacher(document):
    teacher = ScriptedTeacher(default=TeacherResponse())
    annotate_document(document, teacher, run_id="r", resources=CandidateResources.load())
    assert teacher.calls
    assert any(call.candidates for call in teacher.calls)


def test_structural_verification_catches_partial_overlap(document):
    from semantic_corpus.documents import EntityMention
    from semantic_corpus.ontology import LabelSet

    run = AnnotationRun(run_id="r").extended(
        mentions=[
            EntityMention("a", "d1", TextSpan(11, 20), TEXT[11:20], LabelSet.of(Label.PERSON)),
            EntityMention("b", "d1", TextSpan(16, 25), TEXT[16:25], LabelSet.of(Label.PERSON)),
        ]
    )
    report = verify_structure(run, document)
    assert not report.ok
    assert report.overlapping


def test_nesting_is_allowed(document):
    from semantic_corpus.documents import EntityMention
    from semantic_corpus.ontology import LabelSet

    run = AnnotationRun(run_id="r").extended(
        mentions=[
            EntityMention("a", "d1", TextSpan(11, 20), TEXT[11:20], LabelSet.of(Label.PERSON)),
            EntityMention("b", "d1", TextSpan(11, 15), TEXT[11:15], LabelSet.of(Label.PERSON)),
        ]
    )
    assert verify_structure(run, document).ok


def test_the_second_pass_is_a_new_run_and_the_first_is_untouched(document):
    first_teacher = scripted(ann("Anna", "PERSON"), ann("Kyiv", "LOCATION"))
    outcome = annotate_document(document, first_teacher, run_id="run-1")

    second_teacher = scripted(ann("Anna", "PERSON"), name="other-model")
    second, report = verify_independently(outcome.run, document, second_teacher)

    assert second.run_id != outcome.run.run_id
    assert second.model_name == "other-model"
    assert len(outcome.run.mentions) == 2, "the first run must not change"
    assert report.agreed
    assert report.only_first
    assert 0.0 < report.agreement_rate < 1.0


def test_agreement_promotes_only_what_both_passes_found(document):
    first_teacher = scripted(ann("Anna", "PERSON"), ann("Kyiv", "LOCATION"))
    outcome = annotate_document(document, first_teacher, run_id="run-1")
    second_teacher = scripted(ann("Anna", "PERSON"), name="other")
    second, _ = verify_independently(outcome.run, document, second_teacher)

    reviewed = promote_agreed(outcome.run, second)
    statuses = {m.exact_text: m.review_status for m in reviewed.mentions}
    assert statuses["Anna"] is ReviewStatus.VERIFIED
    # Disagreement is evidence, not a verdict: the span survives.
    assert statuses["Kyiv"] is ReviewStatus.AUTO_VALIDATED


def test_every_annotation_in_a_finished_run_is_locatable(document):
    teacher = scripted(
        ann("Anna", "PERSON"),
        ann("gave", "ACTION"),
        ann("Rex", "ANIMAL"),
        ann("Kyiv", "LOCATION"),
    )
    outcome = annotate_document(document, teacher, run_id="run-1")
    assert outcome.run.validate_against(document) == []
