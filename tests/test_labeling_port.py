"""Behavioral coverage of the upstream labeling and resolution APIs."""

from dataclasses import replace

import pytest

from semantic_corpus.qasrl_core.clause_resolution import (
    classify_ambiguity, fallback_resolve, get_frames_with_answer_slots,
    get_resolved_frame_pairs, get_resolved_structures, locally_resolve,
)
from semantic_corpus.qasrl_core.discrete_label import (
    AdvRole, NounRole, from_rendered_string, get_all_discrete_labels, get_discrete_labels,
)
from semantic_corpus.qasrl_core.frame import SUBJ, OBJ, OBJ2, Noun, adv
from semantic_corpus.qasrl_core.models import QuestionSlots, VerbForm
from semantic_corpus.qasrl_core.question_label_mapper import QuestionLabelMapper, map_to_lower_case
from semantic_corpus.qasrl_core.question_renderer import render_question
from semantic_corpus.qasrl_core.slot_based_label import (
    SurfaceQuestionSlots, get_slots_for_question,
    get_verb_tense_abstracted_slots_for_question as get_slots,
    instantiate_verb_for_tense_slots, read_preferred_state,
)


def test_mapper_preserves_missing_order_and_duplicates(give_forms):
    mapper = map_to_lower_case >> QuestionLabelMapper.lift_optional(lambda x: len(x) if x else None)
    assert mapper([], give_forms, iter(["ABC", "", "X", "ABC"])) == [3, None, 1, 3]
    assert QuestionLabelMapper.identity()([], give_forms, ["a", "a"]) == ["a", "a"]


def test_mapper_composition_retains_batch_context(give_forms):
    def frequencies(tokens, forms, labels):
        assert tokens == ["context"] and forms == give_forms
        return {label: labels.count(label) for label in labels}
    mapper = map_to_lower_case >> QuestionLabelMapper(frequencies)
    assert mapper(["context"], give_forms, ["A", "a", "B"]) == [2, 2, 1]


def test_mapper_arrows_preserve_unconverted_values(give_forms):
    lower = map_to_lower_case
    length = QuestionLabelMapper.lift(len)
    assert lower.first()([], give_forms, [("AB", 1)]) == [("ab", 1)]
    assert lower.second()([], give_forms, [(1, "AB")]) == [(1, "ab")]
    assert lower.split(length)([], give_forms, [("AB", "CDE")]) == [("ab", 3)]
    assert lower.fanout(length)([], give_forms, ["AB"]) == [("ab", 2)]
    assert length.compose(lower)([], give_forms, ["AB"]) == [2]


@pytest.mark.parametrize("question", [
    "Who gave something?", "What would someone have been given?",
    "What does someone give someone to do?", "What did someone give with?",
    "What is someone being given?", "Who might not give something?",
])
def test_slot_mappers_render_surface_and_abstract(give_forms, question):
    slots = get_slots([], give_forms, [question])[0]
    surface = get_slots_for_question([], give_forms, [question])[0]
    assert render_question(slots, give_forms) == question
    assert surface.render() == question
    assert instantiate_verb_for_tense_slots([], give_forms, [slots]) == [surface]
    assert SurfaceQuestionSlots.from_json(surface.to_json()) == surface


def test_mapper_uses_upstream_complement_split_and_normalizes_it(give_forms):
    complement, pronoun, invalid = get_slots([], give_forms, [
        "What does someone give someone to do?", "What did it give?", "What did",
    ])
    assert (complement.prep, complement.obj2) == ("_", "to do")
    assert pronoun.subj == "something"
    assert invalid is None
    assert read_preferred_state([], give_forms, "nonsense") is None


@pytest.mark.parametrize("question,category,answer_slot", [
    ("Who gave something?", "unambiguous", SUBJ),
    ("What did someone give with?", "prepositional", OBJ2),
    ("Where did someone give something?", "where", adv("where")),
    ("What did someone give someone?", "ditransitive", OBJ2),
    ("Who did someone give something?", "ditransitive", OBJ),
])
def test_fallback_resolution_matches_upstream_preferences(give_forms, question, category, answer_slot):
    slots = get_slots([], give_forms, [question])[0]
    choices = get_frames_with_answer_slots(slots, give_forms)
    assert classify_ambiguity(slots, choices) == category
    frame, resolved_slot = fallback_resolve(slots, choices)
    assert resolved_slot == answer_slot
    assert frame.verb_inflected_forms == give_forms
    assert frame.questions_for_slot(resolved_slot) == [question]
    assert fallback_resolve(slots, reversed(sorted(choices, key=repr))) == (frame, resolved_slot)


def test_context_vote_overrides_where_fallback(give_forms):
    slots = get_slots([], give_forms, [
        "Where did someone give something?", "Who gave something somewhere?",
    ])
    resolved = get_resolved_frame_pairs(give_forms, slots)
    assert resolved[0][1] == OBJ2
    assert resolved[0][0] == resolved[1][0]
    structures = get_resolved_structures(slots)
    assert structures[0][0] == structures[1][0]
    assert structures[0][0].args[SUBJ] == Noun(False)


def test_local_votes_are_per_distinct_frame_not_answer_slot(give_forms):
    state = read_preferred_state([], give_forms, "Who gave something?")
    first = state.frame
    second = replace(first, tense="present")
    choices = {(first, SUBJ), (first, OBJ), (second, SUBJ)}
    assert locally_resolve([choices]) == [frozenset(choices)]
    assert locally_resolve([choices, {(second, SUBJ)}])[0] == {(second, SUBJ)}
    assert locally_resolve([set()]) == [frozenset()]


@pytest.mark.parametrize("question,label", [
    ("Who gave?", "subj-intransitive/+"),
    ("Who gave something?", "subj-transitive/+"),
    ("What did someone give?", "obj/-"),
    ("Who did someone give something?", "obj-dative/+"),
    ("Who was given?", "obj/+"),
    ("Who was given something?", "obj-dative/+"),
    ("Who was something given by?", "subj-transitive/+"),
    ("What did someone give with?", "with/-"),
    ("Where did someone give something?", "where"),
    ("When did someone give something?", "when"),
])
def test_discrete_roles(give_forms, question, label):
    result = get_discrete_labels([], give_forms, [question, "invalid question"])
    assert str(result[0]) == label
    assert result[1] is None
    assert from_rendered_string(label) == result[0]


def test_discrete_all_readings_are_retained(give_forms):
    results = get_all_discrete_labels([], give_forms, ["What did someone give with?"])[0]
    assert set(results) == {NounRole("obj", False), NounRole("with", False)}
    assert from_rendered_string("WHERE") == AdvRole("where")


@pytest.mark.parametrize("label", ["obj", "obj/yes", "obj/+/extra", "/+", "unknown"])
def test_discrete_invalid_encoding_is_rejected(label):
    with pytest.raises(ValueError):
        from_rendered_string(label)


def test_rendered_slot_codecs_use_literal_separator_and_preserve_verb_prefix(give_forms):
    question = "What would someone have been given?"
    slots = get_slots([], give_forms, [question])[0]
    encoded = slots.render_with_separator(" | ")
    assert "have been pastparticiple" in encoded
    assert QuestionSlots.from_rendered_string(encoded, " | ") == slots
    encoded_surface = slots.render_with_separator(" | ", render_verb=give_forms.get)
    surface = SurfaceQuestionSlots.from_rendered_string(encoded_surface, " | ")
    assert surface.render() == question
    assert surface.render_with_separator(" | ") == encoded_surface
    assert QuestionSlots.from_rendered_string(encoded_surface, " | ",
        read_verb=lambda word: VerbForm.PAST_PARTICIPLE if word == "given" else None) == slots


@pytest.mark.parametrize("text", ["who|_|_|past|_|_", "_|_|_|past|_|_|_", "who|_|_|unknown|_|_|_"])
def test_rendered_slot_decoder_rejects_incomplete_fields(text):
    with pytest.raises(ValueError):
        QuestionSlots.from_rendered_string(text, "|")


def test_rendered_slot_decoder_normalizes_empty_optional_fields():
    slots = QuestionSlots.from_rendered_string("WHO|||PAST|||", "|")
    assert (slots.wh, slots.verb, slots.prep, slots.obj2) == ("who", "past", "_", "_")
