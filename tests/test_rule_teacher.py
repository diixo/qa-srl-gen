"""Rule-based annotation: lexicons, candidates, the offline teacher.

Every number quoted in a docstring here was measured on DailyDialog's
81 869 turns before the fix it motivates.
"""

from __future__ import annotations

import json

import pytest

from semantic_corpus.documents import Document, iter_sentence_spans
from semantic_corpus.ontology import (
    Label,
    MarkerForm,
    MarkerFunction,
    Mood,
    Polarity,
    SpeechAct,
    Stance,
)
from semantic_corpus.question_generator import QuestionGenerator
from semantic_corpus.semantic_annotator import (
    CandidateResources,
    RuleBasedTeacher,
    annotate_document,
    classify_turn,
    extract_candidates,
    marker_for,
    rank_lemmas,
    read_dialogue_jsonl,
    segment_dialogue,
    tokenize,
)
from semantic_corpus.semantic_annotator.lexicons import CONTRACTIONS, STATIVE_VERBS


@pytest.fixture(scope="module")
def resources() -> CandidateResources:
    return CandidateResources.load()


def labels_for(text: str, resources: CandidateResources) -> dict[str, str]:
    document = Document("t", text)
    return {
        c.exact_text: str(c.proposed_labels)
        for c in extract_candidates(document, None, resources)
    }


# -- sentence splitting ----------------------------------------------------


def test_an_initialism_is_not_cut_in_half():
    """~450 turns split inside 'p.m.', 'U.S.', 'O.K.' and 'B.A.'."""
    cases = {
        "I'll come at 7 p.m. But I've got to check in at 12.": [
            "I'll come at 7 p.m.",
            "But I've got to check in at 12.",
        ],
        "Actually, the U.S. is the richest.": ["Actually, the U.S. is the richest."],
        "I did a B.A. in economics.": ["I did a B.A. in economics."],
        "O.K., maybe later.": ["O.K., maybe later."],
    }
    for text, expected in cases.items():
        assert [s.text_in(text) for s in iter_sentence_spans(text)] == expected, text


def test_a_lone_letter_can_still_end_a_sentence():
    """What settles it is the other side, not the letter."""
    for text, expected in {
        "So do I. Let's go.": ["So do I.", "Let's go."],
        "Carrots contain Vitamin C. It is good for you.": [
            "Carrots contain Vitamin C.",
            "It is good for you.",
        ],
    }.items():
        assert [s.text_in(text) for s in iter_sentence_spans(text)] == expected, text


# -- contractions ----------------------------------------------------------


def test_contractions_split_so_negation_stays_visible():
    """4 002 'don't' tokens were one unknown word, hiding the negation."""
    tokens = tokenize("I don't know")
    assert [t.text for t in tokens] == ["I", "do", "n't", "know"]


def test_splitting_a_contraction_keeps_offsets_exact():
    text = "It's fine, I don't mind"
    for token in tokenize(text):
        assert text[token.start_char : token.end_char] == token.text


def test_splitting_can_be_turned_off():
    assert [t.text for t in tokenize("don't", split_contractions=False)] == ["don't"]


def test_the_negation_clitic_is_recorded_as_one(resources):
    document = Document("t", "I don't know.")
    negations = [c for c in extract_candidates(document, None, resources) if c.negates]
    assert [c.exact_text for c in negations] == ["n't"]


def test_every_contraction_splits_without_losing_characters():
    for word, (head, tail) in CONTRACTIONS.items():
        assert head + tail == word, word


# -- stative verbs ---------------------------------------------------------


def test_stative_verbs_are_states_not_actions(resources):
    """STATE appeared on 0.3% of candidates; know, think and want are everywhere."""
    found = labels_for("I know what you think and want.", resources)
    assert "STATE" in found["know"]
    assert "ACTION" not in found["know"]
    assert "STATE" in found["think"]


def test_dynamic_verbs_are_still_actions(resources):
    found = labels_for("She gave him the ball and ran.", resources)
    assert "ACTION" in found["gave"]
    assert "STATE" not in found["gave"]


def test_action_and_state_never_combine(resources):
    document = Document("t", "I know she gave it to him and I think she ran.")
    for candidate in extract_candidates(document, None, resources):
        assert candidate.proposed_labels.problems() == []


def test_the_stative_list_covers_the_handoff_examples():
    for verb in ("know", "own", "contain", "remain", "belong", "exist"):
        assert verb in STATIVE_VERBS


# -- positional capitals ---------------------------------------------------


def test_a_sentence_initial_capital_is_not_a_name(resources):
    """47% of proposed names were the first word of a second sentence."""
    found = labels_for("I don't. Remember last time?", resources)
    assert "NAMED_ENTITY" not in found.get("Remember", "")


def test_a_capital_the_position_does_not_explain_is_still_a_name(resources):
    found = labels_for("Yesterday Anna called.", resources)
    assert "NAMED_ENTITY" in found["Anna"]


def test_the_first_word_of_a_turn_is_not_a_name(resources):
    found = labels_for("Let's go now.", resources)
    assert "NAMED_ENTITY" not in found.get("Let", "")


# -- lemma ranking ---------------------------------------------------------


def test_junk_lemmas_are_demoted(resources):
    """The scrape offered 'thought -> thinck' and 'went -> gan' first."""
    expected_first = {
        "thought": "think",
        "went": "go",
        "made": "make",
        "took": "take",
        "talking": "talk",
    }
    for surface, lemma in expected_first.items():
        ranked = rank_lemmas(resources.inflections.lemmas_for(surface), resources)
        assert ranked[0] == lemma, f"{surface} -> {ranked}"


def test_ranking_keeps_every_candidate(resources):
    raw = resources.inflections.lemmas_for("thought")
    assert set(rank_lemmas(raw, resources)) == set(raw)


def test_candidates_report_the_ranked_lemma_first(resources):
    document = Document("t", "She thought about it.")
    thought = next(
        c for c in extract_candidates(document, None, resources) if c.exact_text == "thought"
    )
    assert thought.lemmas[0] == "think"


# -- discourse markers -----------------------------------------------------


def test_position_decides_whether_a_word_is_a_marker():
    """The handoff's rule: 'Good.' is approval, 'a good car' is a property."""
    assert marker_for("good", turn_initial=True, followed_by_comma=False, stands_alone=False)
    assert marker_for("good", turn_initial=False, followed_by_comma=False, stands_alone=True)
    assert (
        marker_for("good", turn_initial=False, followed_by_comma=False, stands_alone=False)
        is None
    )


def test_markers_are_proposed_where_they_were_missed(resources):
    """A quarter of turns open with one; not one was ever proposed."""
    for text, word in [
        ("Oh, I see.", "Oh"),
        ("Yes. I agree.", "Yes"),
        ("Ok, let's go.", "Ok"),
        ("Well, maybe.", "Well"),
    ]:
        found = labels_for(text, resources)
        assert found.get(word) == "DISCOURSE_MARKER", (text, found)


def test_a_marker_word_in_an_ordinary_position_is_not_a_marker(resources):
    found = labels_for("It is a good car.", resources)
    assert "DISCOURSE_MARKER" not in found.get("good", "")


# -- the teacher -----------------------------------------------------------


def test_mood_comes_from_the_punctuation():
    assert classify_turn("Are you coming?").mood is Mood.INTERROGATIVE
    assert classify_turn("That's great!").mood is Mood.EXCLAMATIVE
    assert classify_turn("I am coming.").mood is Mood.DECLARATIVE


def test_polarity_sees_a_contracted_negation():
    assert classify_turn("I don't know.").polarity is Polarity.NEGATIVE
    assert classify_turn("I never go there.").polarity is Polarity.NEGATIVE
    assert classify_turn("I know.").polarity is Polarity.POSITIVE


def test_a_turn_that_asks_then_asserts_is_still_a_question():
    """It ends in a full stop, but 'What do you mean?' is in there."""
    reading = classify_turn("What do you mean? It will help us.")
    assert SpeechAct.QUESTION in reading.speech_acts


def test_thanking_counts_wherever_it_appears():
    reading = classify_turn("Oh, I don't think so. Thanks anyway.")
    assert SpeechAct.THANKING in reading.speech_acts


def test_surface_cues_settle_the_speech_act():
    assert SpeechAct.GREETING in classify_turn("Hello, how are you?").speech_acts
    assert SpeechAct.APOLOGY in classify_turn("Sorry about that.").speech_acts
    assert SpeechAct.REQUEST in classify_turn("Could you help me?").speech_acts


def test_stance_follows_the_marker_function():
    assert classify_turn("Yes, of course.").stance is Stance.SUPPORTIVE
    assert classify_turn("No, I won't.").stance is Stance.OPPOSED
    assert classify_turn("Well, I'm not sure.").stance is Stance.UNCERTAIN


def test_a_turn_with_nothing_special_still_informs():
    assert classify_turn("The train leaves at six.").speech_acts == (SpeechAct.INFORM,)


def test_the_teacher_says_what_it_will_not_decide():
    """Empty mentions must read as 'not attempted', not as 'none found'."""
    teacher = RuleBasedTeacher()
    assert "speech_act" in teacher.covers
    assert "entity" not in " ".join(teacher.covers)


def test_the_teacher_claims_no_entity_types(resources):
    """Rubber-stamping the candidate layer would record guesses as fact."""
    document = segment_dialogue(_dialogue(["Anna gave Rex a ball."]))
    outcome = annotate_document(
        document, RuleBasedTeacher(), run_id="r", resources=resources
    )
    assert not any(
        Label.PERSON in m.labels or Label.ANIMAL in m.labels for m in outcome.run.mentions
    )


def _dialogue(turns: list[str]) -> Document:
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "d.jsonl"
    path.write_text(json.dumps(turns) + "\n", encoding="utf-8")
    return next(iter(read_dialogue_jsonl(path)))


# -- end to end, offline ---------------------------------------------------


def test_the_pipeline_now_produces_dialogue_annotations(resources):
    """Before this, annotate_document never built a single one."""
    document = segment_dialogue(
        _dialogue(["I finally got the job.", "Wow, that's great!"])
    )
    outcome = annotate_document(
        document, RuleBasedTeacher(), run_id="r", resources=resources
    )
    assert len(outcome.run.dialogue) == 2
    assert outcome.structure.ok
    assert outcome.run.validate_against(document) == []

    second = next(d for d in outcome.run.dialogue if d.utterance_id.endswith("u1"))
    assert SpeechAct.REACTION in second.speech_acts
    assert MarkerFunction.SURPRISE in second.marker_functions
    assert second.mood is Mood.EXCLAMATIVE


def test_real_text_now_yields_qa_examples(resources):
    """The handoff's own dialogue example, annotated with no model at all."""
    document = segment_dialogue(
        _dialogue(["I finally got the job.", "Wow, that's great!"])
    )
    outcome = annotate_document(
        document, RuleBasedTeacher(), run_id="r", resources=resources
    )
    examples = QuestionGenerator(lexicon=resources.inflections, seed=0).for_document(
        outcome.document, outcome.run
    )
    assert examples
    produced = {e.question: e.answer for e in examples}
    expresses = next(q for q in produced if "express" in q)
    assert "surprise" in produced[expresses].lower()
    for example in examples:
        assert example.check() == []
        # Context must reach back far enough to interpret the reaction.
        assert "I finally got the job." in example.context


def test_evidence_spans_point_inside_the_context_they_ship_with(resources):
    """A document-global span would run past a context that is a slice."""
    document = segment_dialogue(_dialogue(["One.", "Two.", "Three.", "Four."]))
    outcome = annotate_document(
        document, RuleBasedTeacher(), run_id="r", resources=resources
    )
    for example in QuestionGenerator(seed=0).for_document(
        outcome.document, outcome.run
    ):
        for span in example.evidence:
            assert span.end_char <= len(example.context)


def test_a_quote_is_not_asked_about_with_doubled_punctuation(resources):
    """`...dinner?"?` would be learned as a typo."""
    document = segment_dialogue(_dialogue(["Are you coming?", "Yes!"]))
    outcome = annotate_document(
        document, RuleBasedTeacher(), run_id="r", resources=resources
    )
    for example in QuestionGenerator(seed=0).for_document(
        outcome.document, outcome.run
    ):
        assert '?"?' not in example.question
        assert '!"?' not in example.question


def test_a_repeated_marker_is_attached_to_the_turn_that_used_it(resources):
    """A quote searched across the context lands on an earlier speaker."""
    document = segment_dialogue(
        _dialogue(["Oh, I see.", "The train leaves.", "Oh, really?"])
    )
    outcome = annotate_document(
        document, RuleBasedTeacher(), run_id="r", resources=resources
    )
    spans = [(m.span.start_char, m.span.end_char) for m in outcome.run.mentions]
    assert len(set(spans)) == len(spans), "a marker was located in the wrong turn"
    turns = {
        u.turn_index
        for m in outcome.run.mentions
        for u in document.utterances
        if u.span.contains(m.span)
    }
    assert turns == {0, 2}
