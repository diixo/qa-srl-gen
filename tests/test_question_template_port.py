"""Remaining upstream QuestionTemplate construction and codec behavior."""

import pytest

from semantic_corpus.qasrl_core.clausal_question import ClausalQuestion
from semantic_corpus.qasrl_core.frame import OBJ, OBJ2, SUBJ, ArgStructure, Frame, Noun, Prep
from semantic_corpus.qasrl_core.question_template import QuestionTemplate


def test_template_from_clausal_question_normalizes_animacy_and_tense(give_forms):
    frame = Frame(give_forms, ArgStructure({SUBJ: Noun(True), OBJ: Noun(False)}),
                  tense="might", is_perfect=True, is_negated=True)
    template = QuestionTemplate.from_clausal_question(ClausalQuestion(frame, SUBJ))
    assert template == QuestionTemplate("what", False, False, True)
    assert template.question_string == "What verbs something?"


def test_template_from_incomplete_frame_fills_missing_subject(give_forms):
    template = QuestionTemplate.from_clausal_question(ClausalQuestion(Frame.empty(give_forms), SUBJ))
    assert template == QuestionTemplate("what", False, False, False)


def test_template_from_clausal_question_uses_upstream_complement_slots(give_forms):
    frame = Frame(give_forms, ArgStructure({SUBJ: Noun(True), OBJ2: Prep("to do", Noun(False))}))
    template = QuestionTemplate.from_clausal_question(ClausalQuestion(frame, OBJ2))
    assert template == QuestionTemplate("what", True, False, False, None, "to do")


def test_template_from_clausal_question_rejects_no_renderings(give_forms):
    question = ClausalQuestion(Frame.empty(give_forms), OBJ2)
    with pytest.raises(ValueError, match="no question"):
        QuestionTemplate.from_clausal_question(question)


def test_template_from_nonfinite_question_rejects_unparseable_rendering(give_forms):
    question = ClausalQuestion(Frame(give_forms, tense="gerund"), SUBJ)
    with pytest.raises(ValueError, match="outside the QA-SRL grammar"):
        QuestionTemplate.from_clausal_question(question)


def test_template_decoder_lowercases_lexical_fields_like_scala():
    template = QuestionTemplate.from_json({
        "abst-wh": "WHAT", "abst-subj": "something", "abst-verb": "verb[pss]",
        "abst-obj": "_", "prep": "OUT OF", "abst-obj2": "SOMETHING",
    })
    assert template == QuestionTemplate("what", True, True, False, "out of", "something")
    assert template.to_json()["prep"] == "out of"
