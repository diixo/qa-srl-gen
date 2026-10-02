"""The canonical representation: spans, documents and append-only runs."""

from __future__ import annotations

import pytest

from semantic_corpus.documents import (
    AnnotationRun,
    Document,
    EntityMention,
    Passage,
    Predicate,
    Property,
    RelationEdge,
    SpanError,
    TextSpan,
    Utterance,
    iter_sentence_spans,
    sha256_of,
)
from semantic_corpus.ontology import Label, LabelSet, Relation, ReviewStatus

TEXT = "On Monday, Anna gave her dog Rex a red ball in Kyiv."


@pytest.fixture
def document() -> Document:
    return Document(document_id="d1", text=TEXT, source="test")


def mention(start: int, end: int, *labels: Label, mention_id: str = "m1") -> EntityMention:
    return EntityMention(
        mention_id=mention_id,
        document_id="d1",
        span=TextSpan(start, end),
        exact_text=TEXT[start:end],
        labels=LabelSet(frozenset(labels)),
    )


# -- spans -----------------------------------------------------------------


def test_a_span_knows_what_it_covers(document):
    span = TextSpan(11, 15)
    assert span.text_in(document.text) == "Anna"
    assert len(span) == 4
    span.check(document.text, "Anna")


def test_a_wrong_span_is_rejected_loudly(document):
    with pytest.raises(SpanError, match="expected 'Rex'"):
        TextSpan(11, 15).check(document.text, "Rex")


def test_degenerate_spans_are_refused():
    with pytest.raises(ValueError):
        TextSpan(5, 5)
    with pytest.raises(ValueError):
        TextSpan(5, 2)
    with pytest.raises(ValueError):
        TextSpan(-1, 3)


def test_spans_shift_between_origins():
    assert TextSpan(3, 7).shifted(100) == TextSpan(103, 107)


def test_containment_and_overlap():
    outer, inner = TextSpan(0, 10), TextSpan(2, 5)
    assert outer.contains(inner)
    assert not inner.contains(outer)
    assert outer.overlaps(inner)
    assert not TextSpan(0, 2).overlaps(TextSpan(2, 4))


# -- documents -------------------------------------------------------------


def test_the_hash_is_content_based(document):
    assert document.sha256 == sha256_of(TEXT)
    assert Document("other", TEXT).sha256 == document.sha256


def test_text_is_never_normalised():
    messy = "  Two   spaces\tand a tab.\n"
    assert Document("d", messy).text == messy


def test_documents_round_trip_through_json(document):
    restored = Document.from_json(document.to_json())
    assert restored.text == document.text
    assert restored.sha256 == document.sha256


def test_unknown_json_fields_survive_in_metadata():
    document = Document.from_json(
        {"document_id": "x", "text": "hi", "domain": "wikinews"}
    )
    assert document.metadata["domain"] == "wikinews"


# -- sentence segmentation -------------------------------------------------


def test_sentences_are_split_without_their_whitespace():
    text = "First one. Second one!  Third?"
    spans = list(iter_sentence_spans(text))
    assert [s.text_in(text) for s in spans] == ["First one.", "Second one!", "Third?"]


def test_common_abbreviations_do_not_end_a_sentence():
    text = "Dr. Smith saw it. She left."
    spans = list(iter_sentence_spans(text))
    assert [s.text_in(text) for s in spans] == ["Dr. Smith saw it.", "She left."]


def test_segmentation_respects_an_offset():
    text = "A b. C d."
    spans = list(iter_sentence_spans(text, offset=50))
    assert spans[0].start_char == 50
    assert spans[1].start_char == 55


# -- annotations -----------------------------------------------------------


def test_a_mention_must_match_the_document(document):
    good = mention(11, 15, Label.PERSON)
    good.validate_against(document)

    wrong = EntityMention(
        mention_id="m2",
        document_id="d1",
        span=TextSpan(11, 15),
        exact_text="Rex",
        labels=LabelSet.of(Label.ANIMAL),
    )
    with pytest.raises(SpanError):
        wrong.validate_against(document)


def test_illegal_label_combinations_are_refused_at_construction():
    with pytest.raises(ValueError, match="at most one entity type"):
        mention(11, 15, Label.PERSON, Label.LOCATION)


def test_confidence_is_bounded():
    with pytest.raises(ValueError, match="confidence"):
        EntityMention(
            mention_id="m",
            document_id="d1",
            span=TextSpan(0, 2),
            exact_text="On",
            labels=LabelSet.of(Label.ABSTRACT_ENTITY),
            confidence=1.5,
        )


def test_a_predicate_must_be_action_or_state():
    with pytest.raises(ValueError, match="ACTION or STATE"):
        Predicate(
            predicate_id="p",
            document_id="d1",
            span=TextSpan(16, 20),
            exact_text="gave",
            lemma="give",
            predicate_type=Label.PERSON,
        )


def test_a_property_keeps_its_parts():
    prop = Property(
        property_id="pr1",
        document_id="d1",
        span=TextSpan(35, 38),
        exact_text="red",
        head="red",
        degree="very",
        negated=True,
    )
    assert (prop.head, prop.degree, prop.negated) == ("red", "very", True)


# -- runs ------------------------------------------------------------------


def test_runs_are_append_only(document):
    run = AnnotationRun(run_id="r1")
    extended = run.extended(mentions=[mention(11, 15, Label.PERSON)])
    assert len(run) == 0, "the original run must not change"
    assert len(extended) == 1


def test_a_run_records_its_provenance():
    run = AnnotationRun(
        run_id="r1", model_name="local-7b", prompt_version="p2", random_seed=5
    )
    assert run.ontology_version == "v1"
    assert (run.model_name, run.prompt_version, run.random_seed) == ("local-7b", "p2", 5)
    assert run.created_at


def test_validation_finds_bad_spans_duplicate_ids_and_dangling_relations(document):
    run = AnnotationRun(run_id="r1").extended(
        mentions=[
            mention(11, 15, Label.PERSON, mention_id="m1"),
            mention(29, 32, Label.ANIMAL, mention_id="m1"),  # duplicate id
        ],
        relations=[RelationEdge("m1", Relation.AGENT_OF, "nowhere")],
    )
    problems = run.validate_against(document)
    assert any("duplicate annotation id" in p for p in problems)
    assert any("not annotated" in p for p in problems)


def test_a_dialogue_annotation_must_name_a_real_turn():
    from semantic_corpus.documents import DialogueAnnotation

    document = Document(
        document_id="d2",
        text="Hi.\nWow!",
        utterances=(
            Utterance("d2#u0", "d2", 0, "A", TextSpan(0, 3)),
            Utterance("d2#u1", "d2", 1, "B", TextSpan(4, 8)),
        ),
    )
    run = AnnotationRun(run_id="r").extended(
        dialogue=[DialogueAnnotation(utterance_id="d2#u9")]
    )
    assert any("unknown utterance" in p for p in run.validate_against(document))


def test_review_status_is_changed_by_making_a_new_object(document):
    original = mention(11, 15, Label.PERSON)
    promoted = original.with_status(ReviewStatus.VERIFIED)
    assert original.review_status is ReviewStatus.UNREVIEWED
    assert promoted.review_status is ReviewStatus.VERIFIED


def test_a_passage_locates_itself_in_the_document(document):
    passage = Passage("p0", "d1", TextSpan(11, 20))
    assert passage.text_in(document) == "Anna gave"
    assert passage.start_char == 11


def test_a_leading_ellipsis_is_not_a_sentence_of_its_own():
    """Found on DailyDialog: 30 turns start with '... ' and used to crash."""
    text = "... Okay, I'm through. Here's the form."
    spans = list(iter_sentence_spans(text))
    assert [s.text_in(text) for s in spans] == ["... Okay, I'm through.", "Here's the form."]


def test_punctuation_only_text_does_not_crash():
    for text in ("...", "?!", "... ...", "\"...\""):
        spans = list(iter_sentence_spans(text))
        assert all(s.text_in(text).strip() for s in spans)
