"""Rendering and parsing QA-SRL questions, including the milestone round-trip."""

from __future__ import annotations

import itertools

import pytest

from semantic_corpus.qasrl_core.bank_reader import read_bank
from semantic_corpus.qasrl_core.models import InflectedForms, QuestionSlots
from semantic_corpus.qasrl_core.question_parser import (
    QuestionParseError,
    parse_question,
    parse_question_all,
)
from semantic_corpus.qasrl_core.question_renderer import render_question, render_verb_slot
from semantic_corpus.qasrl_core.question_slots import make_slots

PUT = InflectedForms("put", "puts", "putting", "put", "put")
DO = InflectedForms("do", "does", "doing", "did", "done")
HELP = InflectedForms("help", "helps", "helping", "helped", "helped")


def test_render_substitutes_the_verb_form(give_forms):
    slots = make_slots("who", "past", obj="something", prep="to", obj2="someone")
    assert render_question(slots, give_forms) == "Who gave something to someone?"


def test_render_expands_a_multi_word_verb_slot(give_forms):
    assert render_verb_slot("being pastParticiple", give_forms) == ["being", "given"]
    slots = make_slots("what", "being pastParticiple", aux="is")
    assert render_question(slots, give_forms) == "What is being given?"


def test_render_capitalises_only_the_first_letter(give_forms):
    slots = make_slots("how much", "pastParticiple", aux="is")
    assert render_question(slots, give_forms) == "How much is given?"


def test_silent_preposition_contributes_no_token():
    slots = make_slots("what", "stem", aux="does", subj="something",
                       obj="something", obj2="do")
    assert slots.prep == ""
    assert render_question(slots, HELP) == "What does something help something do?"


def test_render_rejects_an_empty_verb_slot(give_forms):
    with pytest.raises(ValueError):
        render_verb_slot("", give_forms)


def test_parse_recovers_slots(give_forms):
    slots = parse_question("Who gave something to someone?", give_forms)
    assert slots == QuestionSlots(
        wh="who", aux="_", subj="_", verb="past",
        obj="something", prep="to", obj2="someone",
    )


def test_parse_separates_auxiliary_from_predicate():
    """``does`` here is the auxiliary, not the predicate ``do``."""
    slots = parse_question("What does something do?", DO)
    assert (slots.aux, slots.subj, slots.verb) == ("does", "something", "stem")


def test_parse_handles_a_predicate_spelled_like_its_own_auxiliary():
    slots = parse_question("What has something had?", InflectedForms(
        "have", "has", "having", "had", "had"))
    assert (slots.aux, slots.subj, slots.verb) == ("has", "something", "pastParticiple")


def test_parse_resolves_a_syncretic_paradigm_by_the_auxiliary():
    """``put`` is stem, past and past participle at once; the aux decides."""
    assert parse_question("What did someone put somewhere?", PUT).verb == "stem"
    assert parse_question("What has someone put somewhere?", PUT).verb == "pastParticiple"
    assert parse_question("What put something somewhere?", PUT).verb == "past"


def test_parse_recovers_the_silent_preposition():
    slots = parse_question("What does something help something do?", HELP)
    assert (slots.obj, slots.prep, slots.obj2) == ("something", "", "do")


def test_parse_requires_a_question_mark(give_forms):
    with pytest.raises(QuestionParseError, match="must end with"):
        parse_question("Who gave something", give_forms)


def test_parse_rejects_a_question_about_another_verb(give_forms):
    with pytest.raises(QuestionParseError, match="no slot analysis"):
        parse_question("Who ate something?", give_forms)


def test_parse_reports_the_competing_analyses(give_forms):
    analyses = parse_question_all("What did someone give something?", give_forms)
    assert len(analyses) > 1
    # Every analysis offered must render back to the original question.
    for slots in analyses:
        assert render_question(slots, give_forms) == "What did someone give something?"


def test_round_trip_on_the_sample(fixtures_dir):
    """The round-trip the milestone requires, over real bank records."""
    checked = 0
    for sentence in read_bank(fixtures_dir / "bank_sample.jsonl"):
        for entry, label in sentence.question_labels():
            forms = entry.verb_inflected_forms
            question = label.question_string

            # slots -> string must reproduce the stored question exactly
            assert render_question(label.question_slots, forms) == question

            # string -> slots -> string must be the identity
            parsed = parse_question(question, forms)
            assert render_question(parsed, forms) == question
            checked += 1
    assert checked >= 80


@pytest.mark.corpus
def test_render_is_exact_on_the_real_bank(bank_dev):
    checked = 0
    for sentence in itertools.islice(read_bank(bank_dev), 2000):
        for entry, label in sentence.question_labels():
            assert (
                render_question(label.question_slots, entry.verb_inflected_forms)
                == label.question_string
            )
            checked += 1
    assert checked > 5000


@pytest.mark.corpus
def test_round_trip_is_exact_on_the_real_bank(bank_dev):
    """String round-trip must never fail; exact slot recovery is near-perfect.

    The two are different claims. Rendering throws information away in a
    handful of constructions — ``make something something`` cannot show which
    of the two objects was questioned — so a small share of questions parse to
    an equally valid but different slot assignment. That share is bounded here
    so a regression cannot slip through unnoticed.
    """
    total = exact = 0
    for sentence in itertools.islice(read_bank(bank_dev), 2000):
        for entry, label in sentence.question_labels():
            forms = entry.verb_inflected_forms
            parsed = parse_question(label.question_string, forms)
            assert render_question(parsed, forms) == label.question_string
            total += 1
            exact += parsed == label.question_slots
    assert total > 5000
    assert exact / total > 0.99
