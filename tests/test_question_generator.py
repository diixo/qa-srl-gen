"""The question generator: templates, negatives, paraphrases, determinism."""

from __future__ import annotations

import pytest

from semantic_corpus.documents import (
    AnnotationRun,
    DialogueAnnotation,
    Document,
    EntityMention,
    Predicate,
    RelationEdge,
    TextSpan,
)
from semantic_corpus.ontology import (
    Label,
    LabelSet,
    MarkerFunction,
    Relation,
    SpeechAct,
)
from semantic_corpus.qasrl_core.models import InflectedForms
from semantic_corpus.question_generator import (
    NO_ANSWER_REPLIES,
    QAExample,
    QAKind,
    QuestionGenerator,
    atomic_questions,
    dialogue_act_questions,
    entity_type_questions,
    missing_role_questions,
    ontology_questions,
    paraphrase,
    regular_forms,
    yes_no_questions,
)
from semantic_corpus.question_generator.templates import ParadigmResolver
from semantic_corpus.semantic_generator import Generator
from semantic_corpus.semantic_generator.canonical import to_canonical
from semantic_corpus.semantic_generator.generator import DEFAULT_PARADIGMS

TEXT = "On Monday, Anna gave her dog Rex a red ball in Kyiv."
GIVE = InflectedForms("give", "gives", "giving", "gave", "given")


@pytest.fixture
def resolver() -> ParadigmResolver:
    return ParadigmResolver(known={"give": GIVE})


@pytest.fixture
def annotated() -> tuple[Document, AnnotationRun]:
    """The handoff's running example, as if an annotator had produced it."""
    document = Document("d1", TEXT)

    def mention(mid, start, end, *labels, normalized=None):
        return EntityMention(
            mention_id=mid,
            document_id="d1",
            span=TextSpan(start, end),
            exact_text=TEXT[start:end],
            labels=LabelSet(frozenset(labels)),
            normalized_form=normalized,
        )

    anna = mention("m-anna", 11, 15, Label.PERSON, Label.NAMED_ENTITY, normalized="woman")
    rex = mention("m-rex", 29, 32, Label.ANIMAL, Label.NAMED_ENTITY, normalized="dog")
    ball = mention("m-ball", 33, 43, Label.PHYSICAL_OBJECT)
    kyiv = mention("m-kyiv", 47, 51, Label.LOCATION, Label.NAMED_ENTITY, normalized="city")
    predicate = Predicate(
        predicate_id="p-gave",
        document_id="d1",
        span=TextSpan(16, 20),
        exact_text="gave",
        lemma="give",
        tense="past",
        aspect="simple",
        voice="active",
    )
    run = AnnotationRun(run_id="r1").extended(
        mentions=[anna, rex, ball, kyiv],
        predicates=[predicate],
        relations=[
            RelationEdge("m-anna", Relation.AGENT_OF, "p-gave"),
            RelationEdge("m-rex", Relation.RECIPIENT_OF, "p-gave"),
            RelationEdge("m-ball", Relation.THEME_OF, "p-gave"),
            RelationEdge("m-kyiv", Relation.LOCATION_OF, "p-gave"),
        ],
    )
    assert run.validate_against(document) == []
    return document, run


# -- the example type ------------------------------------------------------


def test_an_example_without_context_is_refused():
    with pytest.raises(ValueError, match="context"):
        QAExample(context="  ", question="Who?", answer="Anna.")


def test_an_example_checks_its_own_consistency():
    bad = QAExample(context="x", question="Who gave it", answer="Anna.")
    assert any("does not end with" in p for p in bad.check())

    lying = QAExample(
        context="x", question="Who?", answer=NO_ANSWER_REPLIES[0], answerable=True
    )
    assert any("answers with a refusal" in p for p in lying.check())

    also_lying = QAExample(
        context="x", question="Who?", answer="Anna.", answerable=False
    )
    assert any("substantive answer" in p for p in also_lying.check())


def test_the_training_layout_is_the_one_the_handoff_specifies():
    example = QAExample(context="Anna gave Rex a ball.", question="Who?", answer="Anna.")
    rendered = example.to_prompt()
    assert "<context>" in rendered and "</context>" in rendered
    assert "<question>" in rendered and "<answer>" in rendered
    assert rendered.index("<question>") < rendered.index("<answer>")


# -- templates -------------------------------------------------------------


def test_atomic_questions_cover_each_argument(annotated, resolver):
    document, run = annotated
    produced = {e.question: e.answer for e in atomic_questions(document, run, resolver=resolver)}
    assert produced["Who gave a red ball to Rex?"] == "Anna."
    assert produced["What did Anna give to Rex?"] == "A red ball."
    assert produced["Who did Anna give a red ball to?"] == "Rex."
    assert produced["Where did Anna give a red ball?"] == "In Kyiv."
    assert produced["What did Anna do?"].startswith("Anna gave")


def test_do_support_appears_only_when_the_subject_is_spelled_out(annotated, resolver):
    """The subject gap takes a finite verb; anything else takes ``did`` + stem."""
    document, run = annotated
    questions = [e.question for e in atomic_questions(document, run, resolver=resolver)]
    assert "Who gave a red ball to Rex?" in questions
    assert all(not q.startswith("Who did gave") for q in questions)
    assert "What did Anna give to Rex?" in questions


def test_a_common_noun_is_lowercased_inside_a_question(resolver):
    """A span lifted from sentence-initial position must not keep its capital."""
    text = "The woman gave Rex a ball."
    document = Document("d", text)
    run = AnnotationRun(run_id="r").extended(
        mentions=[
            EntityMention("m1", "d", TextSpan(0, 9), "The woman", LabelSet.of(Label.PERSON)),
            EntityMention("m2", "d", TextSpan(19, 25), "a ball", LabelSet.of(Label.PHYSICAL_OBJECT)),
        ],
        predicates=[
            Predicate("p", "d", TextSpan(10, 14), "gave", "give", tense="past", voice="active")
        ],
        relations=[
            RelationEdge("m1", Relation.AGENT_OF, "p"),
            RelationEdge("m2", Relation.THEME_OF, "p"),
        ],
    )
    questions = [e.question for e in atomic_questions(document, run, resolver=resolver)]
    assert "What did the woman give?" in questions
    assert not any("The woman" in q for q in questions)


def test_adjunct_answers_use_the_preposition_the_text_uses(resolver):
    """*before dawn* must not be answered with *On dawn.*"""
    text = "Anna gave a ball before dawn."
    document = Document("d", text)
    run = AnnotationRun(run_id="r").extended(
        mentions=[
            EntityMention("m1", "d", TextSpan(0, 4), "Anna", LabelSet.of(Label.PERSON)),
            EntityMention("m2", "d", TextSpan(24, 28), "dawn", LabelSet.of(Label.TIME)),
        ],
        predicates=[
            Predicate("p", "d", TextSpan(5, 9), "gave", "give", tense="past", voice="active")
        ],
        relations=[
            RelationEdge("m1", Relation.AGENT_OF, "p"),
            RelationEdge("m2", Relation.TIME_OF, "p"),
        ],
    )
    answers = [e.answer for e in atomic_questions(document, run, resolver=resolver)]
    assert "Before dawn." in answers


def test_entity_type_questions_read_the_recorded_labels(annotated):
    document, run = annotated
    produced = {e.question: e.answer for e in entity_type_questions(document, run)}
    assert produced["What kind of entity is Rex?"] == "Animal."
    assert produced["What kind of entity is Anna?"] == "Person."


def test_ontology_questions_state_the_inference(annotated):
    document, run = annotated
    produced = {e.question: e.answer for e in ontology_questions(document, run)}
    answer = produced["Is Kyiv a location?"]
    assert "every city is a location" in answer
    # Articles have to agree, or the corpus teaches broken English.
    assert produced["Is Rex an animal?"].startswith("Yes.")


def test_yes_no_answers_carry_their_justification(annotated, resolver):
    document, run = annotated
    example = yes_no_questions(document, run, resolver=resolver)[0]
    assert example.question.startswith("Did Anna")
    assert example.answer.startswith("Yes. Anna gave")


def test_regular_inflection_is_a_fallback_not_a_model():
    assert regular_forms("walk").past == "walked"
    assert regular_forms("carry").present_singular_3rd == "carries"
    assert regular_forms("move").present_participle == "moving"
    assert regular_forms("push").present_singular_3rd == "pushes"
    # It is wrong on irregulars, which is why a real paradigm wins.
    assert regular_forms("give").past != "gave"


# -- negatives -------------------------------------------------------------


def test_absence_from_an_annotated_record_is_not_a_refusal(annotated):
    """The text says *On Monday*; nothing annotated it. Refusing would be wrong."""
    document, run = annotated
    assert not run.synthetic
    assert missing_role_questions(document, run) == []


def test_a_synthetic_record_is_complete_so_silence_means_absence():
    generator = Generator(seed=11)
    realized = next(iter(generator.generate(1)))
    document, run = to_canonical(realized, document_id="g")
    assert run.synthetic
    produced = missing_role_questions(document, run)
    assert produced
    assert all(not e.answerable for e in produced)
    assert all(e.answer in NO_ANSWER_REPLIES for e in produced)


def test_an_omitted_agent_is_asked_about_in_the_active_voice():
    """The record says *given*; the question must say *gave*."""
    generator = Generator(seed=4)
    for realized in generator.omitted_argument_examples(6):
        if "agent" not in realized.omitted_slots:
            continue
        document, run = to_canonical(realized, document_id="omit")
        questions = [
            e.question
            for e in QuestionGenerator(paradigms=DEFAULT_PARADIGMS).for_document(
                document, run
            )
            if e.metadata.get("omitted_slot") == "agent"
        ]
        assert questions
        assert all("given" not in q for q in questions), questions
        return
    pytest.fail("no agentless example was produced")


# -- paraphrases -----------------------------------------------------------


def test_paraphrases_keep_the_answer():
    example = QAExample(
        context="Anna gave a ball in Kyiv.",
        question="Where did Anna give a ball?",
        answer="In Kyiv.",
    )
    variants = paraphrase(example)
    assert variants
    for variant in variants:
        assert variant.answer == example.answer
        assert variant.kind is QAKind.PARAPHRASE
        assert variant.metadata["paraphrase_of"] == example.question
        assert variant.question != example.question


def test_paraphrasing_leaves_unrecognised_shapes_alone():
    example = QAExample(context="x.", question="Why does this matter?", answer="Because.")
    assert paraphrase(example) == []


def test_no_paraphrase_rule_breaks_do_support():
    """A regex cannot refinitise a verb, so the polar rewrite was dropped."""
    example = QAExample(context="x.", question="Did the cat feel sadness?", answer="Yes.")
    for variant in paraphrase(example):
        assert "that the cat feel sadness" not in variant.question


# -- dialogue --------------------------------------------------------------


def test_a_reaction_is_asked_about_with_the_turn_it_answers():
    from semantic_corpus.documents import Utterance

    text = "I finally got the job.\nWow, that's great!"
    document = Document(
        "d",
        text,
        utterances=(
            Utterance("u0", "d", 0, "A", TextSpan(0, 22)),
            Utterance("u1", "d", 1, "B", TextSpan(23, len(text))),
        ),
    )
    run = AnnotationRun(run_id="r").extended(
        dialogue=[
            DialogueAnnotation(
                utterance_id="u1",
                speech_acts=(SpeechAct.REACTION, SpeechAct.APPROVAL),
                marker_functions=(MarkerFunction.SURPRISE,),
            )
        ]
    )
    produced = dialogue_act_questions(document, run)
    assert produced
    for example in produced:
        # Without the previous turn, "Wow" is uninterpretable.
        assert "I finally got the job." in example.context
        assert example.kind is QAKind.SPEECH_ACT
    assert any("express" in e.question for e in produced)
    assert any("reaction and approval" in e.answer.lower() for e in produced)


# -- the driver ------------------------------------------------------------


def test_generation_is_reproducible_and_deduplicated(annotated):
    document, run = annotated
    generator = QuestionGenerator(paradigms={"give": GIVE}, seed=3)
    first = [e.question for e in generator.for_document(document, run)]
    second = [e.question for e in generator.for_document(document, run)]
    assert first == second
    assert len(first) == len(set(first))


def test_every_generated_example_passes_its_own_check(annotated):
    document, run = annotated
    for example in QuestionGenerator(paradigms={"give": GIVE}).for_document(document, run):
        assert example.check() == []
        assert example.context == document.text


def test_generation_over_a_whole_synthetic_corpus_stays_clean():
    generator = Generator(seed=21)
    questions = QuestionGenerator(
        paradigms=DEFAULT_PARADIGMS,
        ambiguous_forms=generator.pool.ambiguous_forms(),
        seed=1,
    )
    total = 0
    kinds: set[QAKind] = set()
    for index, realized in enumerate(generator.generate(40)):
        document, run = to_canonical(realized, document_id=f"g{index}")
        for example in questions.for_document(document, run):
            assert example.check() == []
            assert example.answer.strip()
            kinds.add(example.kind)
            total += 1
    assert total > 200
    assert {QAKind.ATOMIC, QAKind.ENTITY_TYPE, QAKind.NO_ANSWER, QAKind.PARAPHRASE} <= kinds


def test_passage_mode_keeps_each_example_to_its_own_window():
    from semantic_corpus.semantic_annotator import segment_document

    text = "Anna gave Rex a ball. Later, Rex ran to the gate."
    document = segment_document(Document("d", text), window=1)
    run = AnnotationRun(run_id="r").extended(
        mentions=[
            EntityMention("m1", "d", TextSpan(0, 4), "Anna", LabelSet.of(Label.PERSON)),
        ],
        predicates=[
            Predicate("p", "d", TextSpan(5, 9), "gave", "give", tense="past", voice="active")
        ],
        relations=[RelationEdge("m1", Relation.AGENT_OF, "p")],
    )
    produced = QuestionGenerator(paradigms={"give": GIVE}).for_passages(document, run)
    assert produced
    for example in produced:
        assert example.context in text
        assert "ran to the gate" not in example.context
