"""Models: spans, paradigms, slots and JSON fidelity."""

from __future__ import annotations

import json

import pytest

from semantic_corpus.qasrl_core.models import (
    AnswerJudgment,
    InflectedForms,
    QuestionSlots,
    Sentence,
    Span,
    VerbForm,
)

TOKENS = ["John", "gave", "Mary", "a", "book", "."]


def test_span_is_half_open():
    span = Span(3, 5)
    assert len(span) == 2
    assert span.tokens(TOKENS) == ["a", "book"]
    assert span.text(TOKENS) == "a book"


def test_span_rejects_degenerate_bounds():
    with pytest.raises(ValueError):
        Span(2, 2)
    with pytest.raises(ValueError):
        Span(3, 1)
    with pytest.raises(ValueError):
        Span(-1, 2)


def test_span_fits_and_overlaps():
    assert Span(0, 6).fits(TOKENS)
    assert not Span(0, 7).fits(TOKENS)
    assert Span(0, 2).overlaps(Span(1, 3))
    assert not Span(0, 2).overlaps(Span(2, 4))


def test_span_is_hashable_and_ordered():
    assert sorted([Span(3, 5), Span(0, 1)]) == [Span(0, 1), Span(3, 5)]
    assert len({Span(0, 1), Span(0, 1)}) == 1


def test_inflected_forms_lookup_by_enum_and_key(give_forms):
    assert give_forms.get(VerbForm.PAST) == "gave"
    assert give_forms.get("pastParticiple") == "given"
    assert give_forms[VerbForm.STEM] == "give"
    with pytest.raises(KeyError):
        give_forms.get("gerund")


def test_inflected_forms_report_every_matching_form():
    put = InflectedForms("put", "puts", "putting", "put", "put")
    assert put.forms_for("put") == (
        VerbForm.STEM,
        VerbForm.PAST,
        VerbForm.PAST_PARTICIPLE,
    )
    assert put.forms_for("puts") == (VerbForm.PRESENT_SINGULAR_3RD,)
    assert put.forms_for("eaten") == ()


def test_question_slots_split_verb_into_prefix_and_form():
    slots = QuestionSlots(
        wh="what", aux="might", subj="_", verb="not have been pastParticiple",
        obj="_", prep="_", obj2="_",
    )
    assert slots.verb_form is VerbForm.PAST_PARTICIPLE
    assert slots.verb_prefix == ("not", "have", "been")
    assert slots.is_filled("aux")
    assert not slots.is_filled("subj")


def test_question_label_aggregates_judgments(fixtures_dir):
    from semantic_corpus.qasrl_core.bank_reader import read_bank

    sentence = next(iter(read_bank(fixtures_dir / "bank_sample.jsonl")))
    _, label = next(sentence.question_labels())
    assert 0.0 <= label.valid_ratio <= 1.0
    assert len(label.valid_judgments) <= len(label.answer_judgments)
    for span, votes in label.span_votes().items():
        assert votes >= 1
        assert span.fits(sentence.sentence_tokens)


def test_rejected_judgment_serialises_without_spans():
    judgment = AnswerJudgment(source_id="turk-1", is_valid=False, spans=())
    assert judgment.to_json() == {"sourceId": "turk-1", "isValid": False}


def test_sentence_json_round_trip(fixtures_dir):
    path = fixtures_dir / "bank_sample.jsonl"
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            original = json.loads(line)
            restored = Sentence.from_json(original).to_json()
            assert restored == original


def test_sentence_keeps_unmodelled_fields():
    sentence = Sentence.from_json(
        {
            "sentenceId": "x:1",
            "sentenceTokens": ["a"],
            "verbEntries": {},
            "nomEntries": {"0": {"anything": True}},
        }
    )
    assert sentence.extra == {"nomEntries": {"0": {"anything": True}}}
    assert "nomEntries" in sentence.to_json()
