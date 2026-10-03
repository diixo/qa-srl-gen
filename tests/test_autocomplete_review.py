"""Regression cases from independent review of the Scala autocomplete port."""

from semantic_corpus.qasrl_core.autocomplete import Autocomplete, Incomplete
from semantic_corpus.qasrl_core.question_processor import CompleteState, QuestionProcessor
from semantic_corpus.qasrl_core.template_state_machine import TemplateStateMachine


def test_question_cap_preserves_upstream_adverbial_priority(give_forms):
    processor = QuestionProcessor(TemplateStateMachine((), give_forms))
    previous = (
        *processor.process_string_fully("Who gave something?"),
        *processor.process_string_fully("What did someone give?"),
    )
    assert all(isinstance(state, CompleteState) for state in previous)
    result = Autocomplete(processor)("", previous)
    assert isinstance(result, Incomplete)
    assert [suggestion.full_text for suggestion in result.suggestions if suggestion.is_complete] == [
        "How did someone give something?",
        "When did someone give something?",
        "Where did someone give something?",
        "Why did someone give something?",
    ]


def test_answered_adverbial_frees_place_for_next_upstream_slot(give_forms):
    processor = QuestionProcessor(TemplateStateMachine((), give_forms))
    previous = (
        *processor.process_string_fully("Who gave something?"),
        *processor.process_string_fully("What did someone give?"),
        *processor.process_string_fully("When did someone give something?"),
    )
    result = Autocomplete(processor)("", previous)
    assert isinstance(result, Incomplete)
    assert [suggestion.full_text for suggestion in result.suggestions if suggestion.is_complete] == [
        "How did someone give something?",
        "How long did someone give something?",
        "Where did someone give something?",
        "Why did someone give something?",
    ]
