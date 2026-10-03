"""Validate released-bank and upstream slots without silently mixing formats."""

from dataclasses import replace

import pytest

from semantic_corpus.qasrl_core.models import AnswerJudgment, QuestionLabel, Sentence, Span, VerbEntry
from semantic_corpus.qasrl_core.question_renderer import render_question
from semantic_corpus.qasrl_core.slot_based_label import read_preferred_state
from semantic_corpus.qasrl_core.validation import (
    ValidationError, check_question_label, check_sentence, check_slots,
    check_verb_entry, validate_sentence,
)


QUESTIONS = (
    "What does someone give someone to do?",
    "What did someone give someone doing?",
    "Who did something give to do something?",
    "Who did something give toward doing something?",
    "What did someone give someone about doing?",
    "Who gave someone about doing something?",
    "Who might not have been giving something?",
    "Where did someone give something?",
    "What did someone give with?",
)


def question_label(forms, question):
    state = read_preferred_state((), forms, question)
    assert state is not None, question
    frame = state.frame
    return QuestionLabel(
        question, frame.to_slots_upstream(state.answer_slot), frame.tense,
        frame.is_perfect, frame.is_progressive, frame.is_negated, frame.is_passive,
        (AnswerJudgment("worker", True, (Span(0, 1),)),), ("worker",),
    )


@pytest.mark.parametrize("question", QUESTIONS)
def test_upstream_labels_validate_without_bank_slot_conversion(give_forms, question):
    label = question_label(give_forms, question)
    original = label.to_json()
    assert check_slots(label.question_slots, dialect="upstream") == []
    assert check_question_label(label, give_forms, ("Anna", "gave", "rice"), dialect="upstream") == []
    assert render_question(label.question_slots, give_forms) == question
    assert label.to_json() == original


def test_dialects_are_explicit_and_bank_is_still_the_default(give_forms):
    state = read_preferred_state((), give_forms, "What does someone give someone to do?")
    bank = state.frame.to_slots(state.answer_slot)
    upstream = state.frame.to_slots_upstream(state.answer_slot)
    assert (bank.prep, bank.obj2) == ("to", "do")
    assert (upstream.prep, upstream.obj2) == ("_", "to do")
    assert check_slots(bank) == check_slots(bank, dialect="bank") == []
    assert check_slots(upstream, dialect="bank")
    assert check_slots(upstream, dialect="upstream") == []
    assert {problem.code for problem in check_slots(bank, dialect="upstream")} == {"upstream-slots"}


@pytest.mark.parametrize("changes", [
    {"wh": "whence"}, {"aux": "is", "verb": "stem"},
    {"prep": "_", "obj2": "to do someone"}, {"prep": ""},
    {"prep": "nonsense"}, {"subj": "_", "wh": "why"},
])
def test_upstream_rejects_bad_grammar_and_noncanonical_splits(give_forms, changes):
    label = question_label(give_forms, "What does someone give someone to do?")
    slots = replace(label.question_slots, **changes)
    assert check_slots(slots, dialect="upstream")


def test_upstream_rejects_invalid_verb_form_and_features(give_forms):
    label = question_label(give_forms, "What does someone give someone to do?")
    assert {problem.code for problem in check_slots(replace(label.question_slots, verb="given"), dialect="upstream")} == {"verb-form"}
    assert {problem.code for problem in check_question_label(
        replace(label, tense="past"), give_forms, ("Anna", "gave", "rice"), dialect="upstream",
    )} == {"features"}


def test_dialect_propagates_through_entry_sentence_and_raising_validator(give_forms):
    label = question_label(give_forms, "What does someone give someone to do?")
    entry = VerbEntry(1, give_forms, {label.question_string: label})
    sentence = Sentence("test", ("Anna", "gift", "rice"), {"1": entry})
    assert check_verb_entry(entry, sentence.sentence_tokens, dialect="upstream") == []
    assert check_sentence(sentence, dialect="upstream") == []
    assert validate_sentence(sentence, dialect="upstream") is sentence
    with pytest.raises(ValidationError):
        validate_sentence(sentence)
    invalid = replace(label, answer_judgments=(AnswerJudgment("worker", True, (Span(0, 99),)),))
    broken = replace(sentence, verb_entries={"1": replace(entry, question_labels={label.question_string: invalid})})
    assert {problem.code for problem in check_sentence(broken, dialect="upstream")} == {"span-bounds"}


@pytest.mark.parametrize("dialect", ["auto", "scala", "bank2", None])
def test_unknown_dialect_is_an_error_even_without_any_questions(dialect):
    with pytest.raises(ValueError, match="Unknown slot dialect"):
        check_sentence(Sentence("empty", ("token",)), dialect=dialect)
