"""Purchase situations, irregular verb forms and their training examples."""

from dataclasses import replace

import pytest

from semantic_corpus.ontology import Label, LabelSet, Relation
from semantic_corpus.question_generator import NO_ANSWER_REPLIES, QuestionGenerator
from semantic_corpus.semantic_generator import (
    DEFAULT_PARADIGMS, BindingError, Entity, Features, Generator, Situation,
    default_pool, frame_by_lemma, realize,
)
from semantic_corpus.semantic_generator.canonical import to_canonical


@pytest.fixture
def purchase():
    pool = default_pool()
    bindings = {
        slot: next(entity for entity in pool.readings(text) if entity.appositive is None)
        for slot, text in (
            ("agent", "Anna"), ("theme", "a red ball"),
            ("location", "Kyiv"), ("time", "Monday"),
        )
    }
    return Situation(frame_by_lemma("buy"), bindings, DEFAULT_PARADIGMS["buy"])


@pytest.mark.parametrize("pattern, text", [
    ("transitive", "Anna bought a red ball in Kyiv on Monday."),
    ("passive", "A red ball was bought by Anna in Kyiv on Monday."),
    ("passive_agentless", "A red ball was bought in Kyiv on Monday."),
])
def test_purchase_patterns_preserve_roles_spans_and_answers(purchase, pattern, text):
    realized = realize(purchase, pattern)
    assert realized.text == text
    assert realized.check() == []
    document, run = to_canonical(realized, document_id="purchase", split="train")
    predicate, = run.predicates
    assert predicate.lemma == "buy"
    assert predicate.predicate_type is Label.ACTION
    mentions = {mention.mention_id: mention for mention in run.mentions}
    roles = {
        edge.relation: mentions[edge.source_id].exact_text
        for edge in run.relations if edge.target_id == predicate.predicate_id
    }
    assert roles[Relation.THEME_OF].lower() == "a red ball"
    assert roles[Relation.LOCATION_OF] == "Kyiv"
    assert roles[Relation.TIME_OF] == "Monday"
    examples = QuestionGenerator(paradigms=DEFAULT_PARADIGMS).for_document(document, run)
    assert examples and all(example.check() == [] for example in examples)
    answers = {example.question: example for example in examples}
    buyer = answers["Who bought a red ball?"]
    if pattern == "passive_agentless":
        assert Relation.AGENT_OF not in roles
        assert predicate.omitted_slots == ("agent",)
        assert "Anna" not in document.text
        assert not buyer.answerable
        assert buyer.answer in NO_ANSWER_REPLIES
        assert buyer.metadata["omitted_slot"] == "agent"
    else:
        assert roles[Relation.AGENT_OF] == "Anna"
        assert buyer.answerable and buyer.answer == "Anna."
        assert [span.text_in(document.text) for span in buyer.evidence] == ["Anna"]
        assert answers["What did Anna buy?"].answer == "A red ball."
        assert answers["Where did Anna buy a red ball?"].answer == "In Kyiv."
        assert answers["When did Anna buy a red ball?"].answer == "On Monday."


@pytest.mark.parametrize("features, text", [
    (Features(tense="present"), "Anna buys a red ball."),
    (Features(tense="past", is_negated=True), "Anna didn't buy a red ball."),
    (Features(tense="present", is_perfect=True), "Anna has bought a red ball."),
    (Features(tense="present", is_progressive=True), "Anna is buying a red ball."),
    (Features(tense="will"), "Anna will buy a red ball."),
])
def test_purchase_verb_forms_without_optional_adjuncts(purchase, features, text):
    situation = replace(purchase, features=features,
                        bindings={key: purchase.bindings[key] for key in ("agent", "theme")})
    assert realize(situation, "transitive").text == text


@pytest.mark.parametrize("slot, entity", [
    ("agent", Entity("the dog", LabelSet.of(Label.ANIMAL))),
    ("theme", Entity("an idea", LabelSet.of(Label.ABSTRACT_ENTITY))),
    ("theme", Entity("sadness", LabelSet.of(Label.ABSTRACT_ENTITY, Label.EMOTION))),
])
def test_purchase_rejects_invalid_participants(purchase, slot, entity):
    situation = replace(purchase, bindings={**purchase.bindings, slot: entity})
    with pytest.raises(BindingError, match=f"slot '{slot}'"):
        realize(situation)


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_purchase_generation_works_with_each_builtin_split(split):
    generator = Generator(seed=42, frames=(frame_by_lemma("buy"),), split=split)
    examples = list(generator.generate(50))
    assert len(examples) == 50
    for realized in examples:
        assert realized.check() == []
        assert realized.situation.validate() == []
        assert all(entity in generator.pool.entities for entity in realized.situation.bindings.values())
