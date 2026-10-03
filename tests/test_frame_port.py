"""Behavioral examples for Frame/Tense/Argument ports from upstream 16ab4949.

The expected strings exercise the branching and complement substitution in
``Frame.scala``, including paths the released Bank's seven slots cannot express.
"""

from itertools import islice, product
import json

import pytest

from semantic_corpus.qasrl_core.bank_reader import read_bank
from semantic_corpus.qasrl_core.clausal_question import ClausalQuestion
from semantic_corpus.qasrl_core.frame import (
    ALL_ADV_SLOTS, LOCATIVE, OBJ, OBJ2, SUBJ, ArgStructure, Argument,
    ArgumentSlot, ArgumentValue, Frame, Noun, Prep, adv, frame_from_slots,
)
from semantic_corpus.qasrl_core.models import InflectedForms, VerbForm
from semantic_corpus.qasrl_core.question_renderer import render_question
from semantic_corpus.qasrl_core.tense import Finite, NonFinite, Tense


GIVE = InflectedForms("give", "gives", "giving", "gave", "given")
HELP = InflectedForms("help", "helps", "helping", "helped", "helped")


def test_missing_nouns_branch_and_missing_obj2_aborts():
    empty = Frame.empty(GIVE)
    assert empty.questions_for_slot(SUBJ) == ["Who gave?", "What gave?"]
    assert empty.questions_for_slot(OBJ) == [
        "Who did someone give?", "Who did something give?",
        "What did someone give?", "What did something give?",
    ]
    assert empty.questions_for_slot(OBJ2) == []
    assert empty.questions_for_slot(adv("how long")) == [
        "How long did someone give?", "How long did something give?",
    ]
    assert empty.clauses() == ["someone gave", "something gave"]
    # The historical singular API intentionally preserves its old behavior.
    assert empty.clause() == "gave"


def test_all_argument_positions_and_yes_no_question():
    frame = Frame(GIVE, ArgStructure({SUBJ: Noun(True), OBJ: Noun(False),
                                      OBJ2: Prep("to", Noun(True))}))
    assert frame.questions_for_slot(SUBJ) == ["Who gave something to someone?"]
    assert frame.questions_for_slot(OBJ) == ["What did someone give to someone?"]
    assert frame.questions_for_slot(OBJ2) == ["Who did someone give something to?"]
    assert frame.questions_for_slot(adv("why")) == ["Why did someone give something to someone?"]
    # Upstream capitalizes wh words, but does not capitalize inverted questions.
    assert frame.questions_for_slot_with_args(None, {}) == ["did someone give something to someone?"]
    assert frame.questions_for_slot_with_args(OBJ, {SUBJ: "Anna", OBJ2: "Rex"}) == [
        "What did Anna give to Rex?",
    ]


def test_locative_and_objectless_preposition():
    frame = Frame(GIVE, ArgStructure({SUBJ: Noun(True), OBJ2: LOCATIVE}))
    assert frame.questions_for_slot(OBJ2) == ["Where did someone give?"]
    assert frame.clauses_with_args({OBJ2: "Kyiv"}) == ["someone gave Kyiv"]
    particle = frame.with_arg(OBJ2, Prep("up"))
    assert particle.questions_for_slot(OBJ2) == []
    assert particle.structure.valid_answer_slots() == {SUBJ}
    assert particle.questions_for_slot(SUBJ) == ["Who gave up?"]


@pytest.mark.parametrize("complement,value,expected", [
    ("do", "wash dishes", "wash dishes"),
    ("to do", "wash dishes", "to wash dishes"),
    ("doing", "washing dishes", "washing dishes"),
    ("from doing", "washing dishes", "from washing dishes"),
])
def test_real_complement_replaces_do_placeholder(complement, value, expected):
    frame = Frame(HELP, ArgStructure({SUBJ: Noun(True), OBJ: Noun(True),
                                     OBJ2: Prep(complement, Noun(False))}))
    assert frame.clauses_with_args({SUBJ: "Anna", OBJ: "Rex", OBJ2: value}) == [
        "Anna helped Rex " + expected,
    ]
    assert frame.questions_for_slot_with_args(SUBJ, {OBJ: "Rex", OBJ2: value}) == [
        "Who helped Rex " + expected + "?",
    ]
    assert frame.questions_for_slot_with_args(OBJ2, {SUBJ: "Anna", OBJ: "Rex", OBJ2: value}) == [
        "What did Anna help Rex " + complement + "?",
    ]


def test_argument_markers_preserve_generic_values_and_token_boundaries():
    frame = Frame(HELP, ArgStructure({SUBJ: Noun(True), OBJ2: Prep("in doing", Noun(False))}))
    payload = {"span": [4, 6]}
    assert frame.gen_clauses_with_args({SUBJ: "Anna", OBJ2: payload}) == [[
        ArgumentValue("Anna"), "helped", "in", ArgumentValue(payload),
    ]]
    assert frame.clauses_with_arg_markers() == [[
        ArgumentValue(SUBJ), "helped", "in", ArgumentValue(OBJ2),
    ]]
    # An argMap entry cannot invent an argument absent from the frame.
    assert Frame.empty(GIVE).clauses_with_args({SUBJ: "Anna", OBJ: "a ball"}) == [
        "someone gave", "something gave",
    ]


_NONFINITE_STACKS = {
    # Keys: (perfect, progressive, passive), from Frame.scala's two branches.
    "bare": {
        (False, False, False): "give",
        (False, False, True): "be given",
        (False, True, False): "be giving",
        (False, True, True): "be being given",
        (True, False, False): "have given",
        (True, False, True): "have been given",
        (True, True, False): "have been giving",
        (True, True, True): "have been being given",
    },
    "gerund": {
        (False, False, False): "giving",
        (False, False, True): "being given",
        (False, True, False): "giving",
        (False, True, True): "being given",
        (True, False, False): "having given",
        (True, False, True): "having been given",
        (True, True, False): "having been giving",
        (True, True, True): "having been given",
    },
}


@pytest.mark.parametrize("tense", list(NonFinite))
def test_nonfinite_aspect_voice_negation_matrix(tense):
    for perfect, progressive, passive, negated in product((False, True), repeat=4):
        frame = Frame(GIVE, ArgStructure({SUBJ: Noun(True)}, is_passive=passive),
                      tense=tense, is_perfect=perfect, is_progressive=progressive,
                      is_negated=negated)
        key = perfect, progressive, passive
        table = _NONFINITE_STACKS["gerund" if tense is NonFinite.GERUND else "bare"]
        expected = ("not " if negated else "") + ("to " if tense is NonFinite.TO else "") + table[key]
        assert frame.get_verb_stack() == expected.split()
        assert frame.clauses() == ["someone " + expected]
        # No do-support is introduced into non-finite clauses.
        assert frame.split_verb_stack_if_necessary(frame.get_verb_stack()) == expected.split()


def test_nonfinite_conjugations_and_tense_codecs():
    assert Frame(GIVE, tense=NonFinite.BARE).get_verb_conjugation(False) is VerbForm.STEM
    assert Frame(GIVE, tense=NonFinite.TO).get_verb_conjugation(False) is VerbForm.STEM
    assert Frame(GIVE, tense=NonFinite.GERUND).get_verb_conjugation(False) is VerbForm.PRESENT_PARTICIPLE
    for tense in Tense:
        assert Tense.from_json(json.loads(json.dumps(tense))) is tense
        assert Tense.from_string(tense.to_json()) is tense
        assert Tense(tense).is_finite == (NonFinite.from_string(str(tense)) is None)
    assert NonFinite.from_string("past") is None
    assert Finite.from_string("gerund") is None
    assert Tense.from_string("invented") is None
    with pytest.raises(ValueError):
        Frame(GIVE, tense="invented")


def test_structure_hash_remains_valid_after_caller_mutates_original_mapping():
    args = {SUBJ: Noun(True), OBJ2: Prep("to", Noun(False)), adv("when"): None}
    frame = Frame(GIVE, ArgStructure(args))
    frames = {frame}
    args[SUBJ] = Noun(False)
    assert frame in frames
    assert frame.args[SUBJ] == Noun(True)
    with pytest.raises(TypeError):
        frame.args[OBJ] = Noun(False)
    same = Frame(GIVE, ArgStructure(dict(reversed(list(frame.args.items())))))
    assert same == frame and same in frames
    assert frame.structure.forget_animacy().args == {
        SUBJ: Noun(False), OBJ2: Prep("to", Noun(False)), adv("when"): None,
    }


def test_scala_frame_json_includes_dependent_argument_codecs():
    data = {
        "verbInflectedForms": GIVE.to_json(),
        "structure": {"args": {
            "subj": {"isAnimate": True}, "obj": {"isAnimate": False},
            "obj2": {"Prep": {"preposition": "to", "objOpt": {"isAnimate": True}}},
            "why": None,
        }, "isPassive": False},
        "tense": "to", "isPerfect": True, "isProgressive": False, "isNegated": True,
    }
    frame = Frame.from_json(data)
    assert frame.clauses() == ["someone not to have given something to someone"]
    assert frame.to_json() == data
    for arg in (Noun(True), Noun(False), LOCATIVE, Prep("up"), Prep("to", Noun(False))):
        assert Argument.from_json(json.loads(json.dumps(arg.to_json()))) == arg
    for slot in (SUBJ, OBJ, OBJ2, *ALL_ADV_SLOTS):
        assert ArgumentSlot.from_json(slot.to_json()) == slot
    assert ArgumentSlot.from_string("invalid") is None
    assert Noun(True).is_noun and not Noun(True).is_prep
    assert LOCATIVE.is_locative and Prep("up").is_prep


def test_clausal_question_round_trip_and_animacy_template():
    frame = Frame(GIVE, ArgStructure({SUBJ: Noun(True), OBJ: Noun(False)}))
    question = ClausalQuestion(frame, OBJ)
    assert question.question_string == "What did someone give?"
    assert question.clause_template.args[SUBJ] == Noun(False)
    assert ClausalQuestion.from_json(question.to_json()) == question


@pytest.mark.parametrize("prep,new_prep,new_obj2", [
    ("do", "_", "do"), ("doing", "_", "doing"),
    ("to do", "_", "to do"), ("to doing", "_", "to doing"),
    ("in doing", "in", "doing"), ("in to do", "in", "to do"),
])
def test_upstream_complement_slots_are_separate_from_released_bank(prep, new_prep, new_obj2):
    frame = Frame(HELP, ArgStructure({SUBJ: Noun(True), OBJ2: Prep(prep, Noun(False))}))
    upstream = frame.to_slots_upstream(OBJ2)
    assert (upstream.prep, upstream.obj2) == (new_prep, new_obj2)
    assert render_question(upstream, HELP) == frame.question(OBJ2)
    upstream_subject = frame.to_slots_upstream(SUBJ)
    assert (upstream_subject.prep, upstream_subject.obj2) == (new_prep, new_obj2 + " something")
    assert render_question(upstream_subject, HELP) == frame.question(SUBJ)
    if prep == "to do":
        assert (frame.to_slots(OBJ2).prep, frame.to_slots(OBJ2).obj2) == ("to", "do")


@pytest.mark.corpus
def test_full_frame_rendering_preserves_released_bank_questions(bank_dev):
    count = 0
    for sentence in islice(read_bank(bank_dev), 500):
        for entry, label in sentence.question_labels():
            frame, slot = frame_from_slots(label.question_slots, entry.verb_inflected_forms)
            assert frame.questions_for_slot(slot) == [label.question_string]
            assert frame.to_slots(slot) == label.question_slots
            count += 1
    assert count > 1000
