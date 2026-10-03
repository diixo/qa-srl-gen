"""Behavioral coverage of the Scala Dataset algebra port."""

from dataclasses import replace

import pytest

from semantic_corpus.qasrl_core.dataset import (
    ConsolidatedDataset, ConsolidatedSentence, Dataset, QuestionMergeFailure,
    SentenceMergeFailure, VerbMergeFailure,
)
from semantic_corpus.qasrl_core.models import (
    AnswerJudgment, InflectedForms, QuestionLabel, QuestionSlots, Sentence, Span, VerbEntry,
)


FORMS = InflectedForms("eat", "eats", "eating", "ate", "eaten")


def label(source="turk-qasrl2.0-1", **changes):
    value = QuestionLabel(
        "Who ate something?", QuestionSlots("who", "_", "_", "past", "something", "_", "_"),
        "past", False, False, False, False,
        (AnswerJudgment(source, True, (Span(0, 1),)),), (source,),
    )
    return replace(value, **changes)


def sentence(sid="s", question=None, **changes):
    question = label() if question is None else question
    value = Sentence(sid, ("Anna", "ate", "rice", "."), {"1": VerbEntry(1, FORMS, {question.question_string: question})})
    return replace(value, **changes)


def test_question_merge_unions_sources_and_semantic_judgment_sets():
    left = label(answer_judgments=(AnswerJudgment("a", True, (Span(0, 1), Span(2, 3))),))
    right = label("b", answer_judgments=(AnswerJudgment("a", True, (Span(2, 3), Span(0, 1))), AnswerJudgment("b", False)))
    merged = left.combine_with_like(right)
    assert merged.question_sources == ("turk-qasrl2.0-1", "b")
    assert merged.answer_judgments == (left.answer_judgments[0], right.answer_judgments[1])
    assert left.combine_with_like(left) == left


@pytest.mark.parametrize("changes", [
    {"question_string": "Who ate?"}, {"question_slots": QuestionSlots("who", "_", "_", "past", "_", "_", "_")},
    {"tense": "present"}, {"is_perfect": True}, {"is_progressive": True}, {"is_negated": True}, {"is_passive": True},
])
def test_question_merge_rejects_every_grammatical_conflict(changes):
    with pytest.raises(ValueError, match="like questions"):
        label().combine_with_like(label(**changes))


def test_dataset_filtering_does_not_implicitly_cull_containers():
    full = Dataset.from_sentences([sentence("a"), sentence("b", verb_entries={})])
    empty_labels = full.filter_question_sources(lambda _: False)
    assert list(empty_labels.sentences) == ["a", "b"]
    assert list(empty_labels.sentences["a"].verb_entries) == ["1"]
    assert empty_labels.cull_questionless_sentences() == Dataset()
    assert empty_labels.cull_questionless_verbs().cull_verbless_sentences() == Dataset()
    assert list(full.cull_verbless_sentences().sentences) == ["a"]
    assert full.filter_sentence_ids(lambda sid: sid == "a") == full.filter_sentences(lambda value: value.sentence_id == "a")
    assert full.filter_question_labels(lambda value: value.valid_ratio == 1) == full
    assert full.sentences["a"].verb_entries["1"].question_labels


def test_source_filter_removes_only_questions_whose_sources_are_all_removed():
    value = label(question_sources=("human", "model"))
    full = Dataset.from_sentences([sentence(question=value)])
    filtered = full.filter_question_sources(lambda source: source == "human")
    kept = filtered.sentences["s"].verb_entries["1"].question_labels[value.question_string]
    assert kept.question_sources == ("human",)
    assert kept.answer_judgments == value.answer_judgments


def test_merge_identity_associativity_and_unknown_field_preservation():
    a = Dataset.from_sentences([sentence(question=label("a"), extra={"nomEntries": {"2": {"isVerbal": True}}})])
    b = Dataset.from_sentences([sentence(question=label("b"), extra={"provenance": [1]})])
    c = Dataset.from_sentences([sentence("z"), sentence(question=label("c"))])
    original = a.to_json()
    assert a.merge(Dataset()).dataset == a == Dataset().merge(a).dataset
    assert a.merge(b).dataset.merge(c) == a.merge(b.merge(c).dataset)
    result = a.merge(b)
    assert not result.failures
    assert result.dataset.sentences["s"].extra == {"nomEntries": {"2": {"isVerbal": True}}, "provenance": [1]}
    assert a.to_json() == original
    assert Dataset.from_json(result.dataset.to_json()) == result.dataset


def test_merge_reports_conflicting_question_and_keeps_other_questions():
    left = sentence()
    conflict = label(is_passive=True)
    added = label(question_string="What did someone eat?", question_slots=QuestionSlots("what", "did", "someone", "stem", "_", "_", "_"))
    right = sentence(verb_entries={"1": VerbEntry(1, FORMS, {conflict.question_string: conflict, added.question_string: added})})
    result = Dataset.from_sentences([left]).merge(Dataset.from_sentences([right]))
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert isinstance(failure, QuestionMergeFailure)
    assert failure.q1 == label() and failure.q2 == conflict
    merged = result.dataset.sentences["s"].verb_entries["1"]
    assert merged.question_labels[label().question_string] == label()
    assert merged.question_labels[added.question_string] == added
    with pytest.raises(ValueError, match="is_passive"):
        left.combine_with_like(right)


def test_verb_conflict_retains_left_verb_and_records_both_inputs():
    left = sentence()
    different = replace(FORMS, stem="devour")
    right = sentence(verb_entries={"1": VerbEntry(1, different)})
    result = Dataset.from_sentences([left]).merge(Dataset.from_sentences([right]))
    assert result.dataset.sentences["s"] == left
    assert isinstance(result.failures[0], VerbMergeFailure)
    assert result.failures[0].v2.verb_inflected_forms == different
    with pytest.raises(ValueError, match="same verb"):
        left.verb_entries["1"].combine_with_like(right.verb_entries["1"])


@pytest.mark.parametrize("changes", [
    {"sentence_tokens": ("rice", "ate", "Anna", ".")},
    {"extra": {"nomEntries": {"2": "different"}}},
])
def test_sentence_conflict_does_not_mix_token_offsets_or_lose_metadata(changes):
    left = sentence(extra={"nomEntries": {"2": "original"}})
    right = replace(left, **changes)
    result = Dataset.from_sentences([left]).merge(Dataset.from_sentences([right, sentence("z")]))
    assert result.dataset.sentences["s"] == left
    assert "z" in result.dataset.sentences
    assert isinstance(result.failures[0], SentenceMergeFailure)
    assert result.failures[0].s2 == right


def test_constructing_dataset_requires_unique_consistent_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        Dataset.from_sentences([sentence(), sentence()])
    with pytest.raises(ValueError, match="differs"):
        Dataset({"wrong": sentence()})
    with pytest.raises(ValueError, match="same sentence ID"):
        sentence().combine_with_like(sentence("other"))


def test_monoid_combine_reports_conflicts_through_required_sink():
    left = Dataset.from_sentences([sentence()])
    right = Dataset.from_sentences([sentence(question=label(is_passive=True))])
    reports = []
    assert left.combine(right, reports.append) == left
    assert len(reports) == 1 and isinstance(reports[0][0], QuestionMergeFailure)
    assert left.combine(Dataset(), reports.append) == left
    assert reports[-1] == ()


def test_consolidated_merge_unions_non_predicates_and_preserves_extra():
    original = sentence(extra={"nonPredicates": {"0": "anna"}, "nomEntries": {"n": {"x": 1}}})
    left = ConsolidatedDataset.from_dataset(Dataset.from_sentences([original]))
    right = ConsolidatedDataset({"s": replace(left.sentences["s"], non_predicates={"2": "rice"})})
    result = left.merge(right)
    assert not result.failures
    merged = result.dataset.sentences["s"]
    assert merged.non_predicates == {"0": "anna", "2": "rice"}
    assert merged.extra == {"nomEntries": {"n": {"x": 1}}}
    assert ConsolidatedDataset.from_json(result.dataset.to_json()) == result.dataset
    assert "nonPredicates" not in merged.to_sentence().extra
    assert left.merge(left).dataset == left


def test_consolidated_conflicts_do_not_leak_right_non_predicates():
    left = ConsolidatedDataset({"s": ConsolidatedSentence.from_sentence(sentence(extra={"nonPredicates": {"0": "anna"}}))})
    right = ConsolidatedDataset({"s": replace(left.sentences["s"], sentence_tokens=("different",), non_predicates={"2": "rice"})})
    result = left.merge(right)
    assert result.dataset == left
    assert isinstance(result.failures[0], SentenceMergeFailure)
    right = ConsolidatedDataset({"s": replace(left.sentences["s"], non_predicates={"0": "other"})})
    assert left.merge(right).dataset == left
    assert left.merge(right).failures
