"""Behavioral checks for the upstream incremental grammar and autocomplete."""

from collections import Counter
from itertools import islice

import pytest

from semantic_corpus.qasrl_core.autocomplete import Autocomplete, Complete, Incomplete
from semantic_corpus.qasrl_core.bank_reader import read_bank
from semantic_corpus.qasrl_core.frame import OBJ, OBJ2, SUBJ, Noun, Prep, adv
from semantic_corpus.qasrl_core.models import InflectedForms
from semantic_corpus.qasrl_core.question_processor import (
    AggregatedInvalidState, CompleteState, InProgressState, QuestionProcessor,
)
from semantic_corpus.qasrl_core.state_machine import PREPOSITIONS
from semantic_corpus.qasrl_core.template_state_machine import TemplateStateMachine


@pytest.fixture
def processor(give_forms):
    return QuestionProcessor(TemplateStateMachine((), give_forms))


def completed(processor, text):
    result = processor.process_string_fully(text)
    assert not isinstance(result, AggregatedInvalidState), (text, result.num_good_characters)
    assert result and all(isinstance(state, CompleteState) for state in result)
    return result


@pytest.mark.parametrize("prefix", ["", "W", "How l", "Who ga", "What did some", "What did someone give"])
def test_prefix_states_are_incremental(processor, prefix):
    result = processor.process_string_fully(prefix)
    assert isinstance(result, tuple) and result
    assert all(isinstance(state, InProgressState) for state in result)
    assert all(state.text_so_far.lower() == prefix.lower() for state in result)
    assert all(state.full_text.lower().startswith(prefix.lower()) for state in result)
    assert not processor.is_valid(prefix)


def test_resume_advances_saved_states_without_reparsing(processor, monkeypatch):
    first = processor.process_string("What did someone")
    monkeypatch.setattr(processor, "process_string", lambda _: pytest.fail("prefix reparsed"))
    final = processor.advance(first, " give?")
    assert final.valid_states
    assert all(state.is_complete for state in final.valid_states)
    assert {state.answer_slot for state in final.valid_states} == {OBJ}


def test_case_insensitive_character_processing_keeps_canonical_surface(processor):
    states = completed(processor, "wHO gAVE sOMETHING?")
    assert states[0].full_text == "Who gave something?"
    assert states[0].answer_slot == SUBJ


def test_subject_it_and_post_subject_negation(processor):
    states = completed(processor, "What did it not give?")
    assert all(state.frame.args[SUBJ] == Noun(False) for state in states)
    assert all(state.frame.is_negated for state in states)
    assert not processor.is_valid("What didn't it not give?")


@pytest.mark.parametrize("question", [
    "What has someone been giving?",
    "What is someone being given?",
    "What would someone have been given?",
    "Who might not give something?",
    "Who does someone give something to?",
    "What does someone give someone to do?",
])
def test_full_auxiliary_and_complement_transitions(processor, question):
    completed(processor, question)


def test_ambiguity_keeps_distinct_argument_structures(processor):
    where = completed(processor, "Where did someone give something?")
    assert {state.answer_slot for state in where} == {OBJ2, adv("where")}
    prep = completed(processor, "What did someone give with?")
    assert {state.answer_slot for state in prep} == {OBJ, OBJ2}
    assert any(state.frame.args[OBJ2] == Prep("with", None) for state in prep)
    assert any(state.frame.args[OBJ2] == Prep("with", Noun(False)) for state in prep)


@pytest.mark.parametrize("question", [
    "When gave something?",          # adjunct needs a subject
    "Who gave something someone something?",
    "Who has been being given?",    # excluded perfect-progressive-passive chain
    "What did someone give someone with something?",  # no gap
])
def test_invalid_questions_are_rejected(processor, question):
    assert not processor.is_valid(question)


def test_bare_second_object_requires_a_first_object(processor):
    single = completed(processor, "Who gave someone?")
    assert all(OBJ in state.frame.args and OBJ2 not in state.frame.args for state in single)
    double = completed(processor, "Who gave someone something?")
    assert all(OBJ in state.frame.args and OBJ2 in state.frame.args for state in double)


def test_failure_reports_longest_valid_prefix(processor):
    question = "What did someone give? garbage"
    result = processor.process_string_fully(question)
    assert isinstance(result, AggregatedInvalidState)
    assert result.num_good_characters == len("What did someone give?")
    assert all(state.is_complete for state in result.last_good_states)
    typo = processor.process_string_fully("What dxd")
    assert isinstance(typo, AggregatedInvalidState)
    assert typo.num_good_characters == len("What d")


def test_sentence_prepositions_and_adjacent_bigrams(give_forms):
    machine = TemplateStateMachine(("OUT", "OF", "one", "near", "two"), give_forms)
    assert {"out", "of", "out of", "near"} <= machine.all_chosen_prepositions
    assert "of near" not in machine.all_chosen_prepositions
    processor = QuestionProcessor(machine)
    assert processor.is_valid("What did someone give out of?")
    assert processor.is_valid("What did someone give near?")
    assert not processor.is_valid("What did someone give across?")
    override = TemplateStateMachine(("near",), give_forms, prepositions={"ACROSS"})
    assert override.all_chosen_prepositions == {"across"}
    assert QuestionProcessor(override).is_valid("What did someone give across?")
    assert not QuestionProcessor(override).is_valid("What did someone give near?")


def test_autocomplete_continuations_and_bad_suffix(processor):
    autocomplete = Autocomplete(processor)
    initial = autocomplete("")
    assert isinstance(initial, Incomplete)
    assert {s.full_text for s in initial.suggestions} == {
        "Who", "What", "When", "Where", "Why", "How", "How much", "How long",
    }
    result = autocomplete("Who gave something")
    assert isinstance(result, Incomplete)
    assert result.suggestions[0].full_text == "Who gave something?"
    assert result.suggestions[0].is_complete
    assert isinstance(autocomplete("Who gave something?"), Complete)
    invalid = autocomplete("Who gave something??")
    assert isinstance(invalid, Incomplete)
    assert invalid.bad_start_index == len("Who gave something?")
    assert invalid.suggestions[0].full_text == "Who gave something?"


def test_autocomplete_suggests_unanswered_slots_without_duplicate_questions(processor):
    previous = completed(processor, "Who gave something?")
    autocomplete = Autocomplete(processor)
    result = autocomplete("What", previous)
    assert isinstance(result, Incomplete)
    assert any(s.full_text == "What did someone give?" and s.is_complete for s in result.suggestions)
    assert all(s.full_text != "Who gave something?" for s in result.suggestions)
    empty = autocomplete("", previous)
    assert sum(s.is_complete for s in empty.suggestions) <= 4


def test_real_bank_questions_are_accepted(bank_dev):
    count = 0
    for sentence in islice(read_bank(bank_dev), 100):
        for verb in sentence.verb_entries.values():
            for label in verb.question_labels.values():
                # Existing Bank questions can use rare prepositions absent
                # from the sentence. Like Scala's question mapper, infer the
                # inventory from the observed question when parsing a corpus.
                processor = QuestionProcessor(TemplateStateMachine(
                    label.question_string[:-1].split(), verb.verb_inflected_forms,
                ))
                completed(processor, label.question_string)
                count += 1
    assert count > 100


def test_upstream_questions_exact_ambiguity_histogram(repo_root):
    """Optional local upstream conformance corpus; no download during tests."""
    path = (repo_root / "artifacts/upstream-qasrl/qasrl-16ab4949"
            / "qasrl/test/resources/question-strings.txt")
    if not path.exists():
        pytest.skip("upstream reference corpus not available")
    forms = InflectedForms("stem", "presentsingular3rd", "presentparticiple", "past", "pastparticiple")
    histogram = Counter()
    questions = path.read_text(encoding="utf-8").splitlines()
    assert len(questions) == 13616
    for question in questions:
        tokens = question[:-1].lower().split()
        preps = {word for word in tokens if word in PREPOSITIONS}
        preps.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if a in PREPOSITIONS and b in PREPOSITIONS)
        processor = QuestionProcessor(TemplateStateMachine((), forms, prepositions=preps))
        states = completed(processor, question)
        histogram[len(states)] += 1
        for state in states:
            assert state.full_text == question
            assert state.frame.questions_for_slot(state.answer_slot) == [question]
    assert histogram == {1: 10882, 2: 2599, 3: 135}
