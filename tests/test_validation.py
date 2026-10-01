"""Invariant checks: bad spans, bad indices and impossible slot combinations."""

from __future__ import annotations

import itertools

import pytest

from semantic_corpus.qasrl_core.bank_reader import read_bank
from semantic_corpus.qasrl_core.models import (
    AnswerJudgment,
    InflectedForms,
    QuestionLabel,
    QuestionSlots,
    Sentence,
    Span,
    VerbEntry,
)
from semantic_corpus.qasrl_core.validation import (
    ValidationError,
    check_question_label,
    check_sentence,
    check_slots,
    check_span,
    count_problems,
    validate_sentence,
)

TOKENS = ["John", "gave", "Mary", "a", "book", "."]
GIVE = InflectedForms("give", "gives", "giving", "gave", "given")


def codes(problems) -> set[str]:
    return {p.code for p in problems}


def slots(**overrides) -> QuestionSlots:
    base = dict(wh="who", aux="_", subj="_", verb="past", obj="_", prep="_", obj2="_")
    base.update(overrides)
    return QuestionSlots(**base)


def test_valid_slots_produce_no_problems():
    assert check_slots(slots()) == []


def test_unknown_wh_word_is_rejected():
    assert "slot-vocabulary" in codes(check_slots(slots(wh="whence")))


def test_somewhere_is_allowed_in_obj2_but_not_obj():
    assert check_slots(slots(obj2="somewhere")) == []
    assert "slot-vocabulary" in codes(check_slots(slots(obj="somewhere")))


def test_impossible_verb_chain_is_rejected():
    assert "verb-chain" in codes(check_slots(slots(aux="does", verb="pastParticiple")))
    assert "verb-chain" in codes(check_slots(slots(aux="is", verb="stem")))


def test_verb_slot_without_a_form_is_rejected():
    assert "verb-form" in codes(check_slots(slots(verb="gave")))


def test_subject_must_be_the_gap_when_nothing_is_fronted():
    problems = check_slots(slots(aux="_", verb="past", subj="something"))
    assert "subject-inversion" in codes(problems)


def test_bare_complement_requires_the_silent_preposition():
    assert "silent-prep" in codes(check_slots(slots(prep="_", obj2="do")))
    assert "silent-prep" in codes(check_slots(slots(prep="", obj2="something")))
    assert check_slots(slots(prep="", obj2="do")) == []


def test_span_outside_the_sentence_is_rejected():
    assert check_span(Span(0, 6), TOKENS) == []
    assert "span-bounds" in codes(check_span(Span(4, 9), TOKENS))


def test_rejected_judgment_must_not_carry_spans():
    label = QuestionLabel(
        question_string="Who gave something?",
        question_slots=slots(obj="something"),
        tense="past",
        is_perfect=False,
        is_progressive=False,
        is_negated=False,
        is_passive=False,
        answer_judgments=(
            AnswerJudgment("turk-1", is_valid=False, spans=(Span(0, 1),)),
        ),
    )
    assert "rejected-with-spans" in codes(check_question_label(label, GIVE, TOKENS))


def test_duplicate_judgment_source_is_reported():
    label = QuestionLabel(
        question_string="Who gave something?",
        question_slots=slots(obj="something"),
        tense="past",
        is_perfect=False,
        is_progressive=False,
        is_negated=False,
        is_passive=False,
        answer_judgments=(
            AnswerJudgment("turk-1", is_valid=True, spans=(Span(0, 1),)),
            AnswerJudgment("turk-1", is_valid=True, spans=(Span(0, 1),)),
        ),
    )
    assert "duplicate-judgment" in codes(check_question_label(label, GIVE, TOKENS))


def test_question_string_must_match_the_slots():
    label = QuestionLabel(
        question_string="Who gives something?",  # wrong tense for verb="past"
        question_slots=slots(obj="something"),
        tense="past",
        is_perfect=False,
        is_progressive=False,
        is_negated=False,
        is_passive=False,
    )
    assert "render-mismatch" in codes(check_question_label(label, GIVE, TOKENS))


def test_features_inconsistent_with_the_chain_are_reported():
    label = QuestionLabel(
        question_string="Who gave something?",
        question_slots=slots(obj="something"),
        tense="present",  # but verb="past" with no aux can only be past
        is_perfect=False,
        is_progressive=False,
        is_negated=False,
        is_passive=False,
    )
    assert "features" in codes(check_question_label(label, GIVE, TOKENS))


def build_sentence(verb_index: int = 1) -> Sentence:
    label = QuestionLabel(
        question_string="Who gave something?",
        question_slots=slots(obj="something"),
        tense="past",
        is_perfect=False,
        is_progressive=False,
        is_negated=False,
        is_passive=False,
        answer_judgments=(AnswerJudgment("turk-1", is_valid=True, spans=(Span(0, 1),)),),
    )
    entry = VerbEntry(
        verb_index=verb_index,
        verb_inflected_forms=GIVE,
        question_labels={label.question_string: label},
    )
    return Sentence(
        sentence_id="test:1",
        sentence_tokens=tuple(TOKENS),
        verb_entries={str(verb_index): entry},
    )


def test_clean_sentence_validates():
    assert check_sentence(build_sentence()) == []
    assert validate_sentence(build_sentence()).sentence_id == "test:1"


def test_verb_index_outside_the_sentence_is_rejected():
    problems = check_sentence(build_sentence(verb_index=99))
    assert "verb-index" in codes(problems)
    with pytest.raises(ValidationError, match="verb-index"):
        validate_sentence(build_sentence(verb_index=99))


def test_problems_can_be_tallied():
    problems = check_sentence(build_sentence(verb_index=99))
    assert count_problems(problems) == {"verb-index": 1}


def test_sample_corpus_is_clean(fixtures_dir):
    for sentence in read_bank(fixtures_dir / "bank_sample.jsonl"):
        assert check_sentence(sentence) == [], sentence.sentence_id


@pytest.mark.corpus
def test_real_bank_passes_every_structural_invariant(bank_dev):
    """Spans, indices, slots and renderings are flawless in the real bank.

    The one thing that is not: a few questions carry two judgments from the
    same annotator. That is a property of the released data, not of this
    reader, so it is asserted as a known and bounded defect rather than
    quietly tolerated.
    """
    tally: dict[str, int] = {}
    for sentence in itertools.islice(read_bank(bank_dev), 2000):
        for code, count in count_problems(check_sentence(sentence)).items():
            tally[code] = tally.get(code, 0) + count
    assert set(tally) <= {"duplicate-judgment"}, tally
    assert tally.get("duplicate-judgment", 0) < 10
