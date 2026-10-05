"""Collectives keep their base label, member constraints and verb agreement."""

from dataclasses import replace

import pytest

from semantic_corpus.ontology import Label, LabelSet, load_default_hierarchy
from semantic_corpus.semantic_generator import (
    BindingError, DEFAULT_PARADIGMS, Entity, EntityPool, Features, Generator,
    Situation, default_pool, frame_by_lemma, realize,
)
from semantic_corpus.semantic_generator.canonical import to_canonical
from semantic_corpus.semantic_generator.substitutions import entity_from_json, entity_to_json
from semantic_corpus.question_generator import QuestionGenerator
from semantic_corpus.storage import CorpusStore
from semantic_corpus.exporters.bio import tag_document


def group(text="the team", type_name="human_group", number="singular"):
    return Entity(text, LabelSet.of(Label.GROUP), type_name=type_name, number=number)


def purchase(buyer=None, goods=None, **features):
    return Situation(
        frame_by_lemma("buy"),
        {"agent": buyer or group(),
         "theme": goods or Entity("a book", LabelSet.of(Label.PHYSICAL_OBJECT))},
        DEFAULT_PARADIGMS["buy"], Features(**features),
    )


def test_builtin_pool_loads_groups_and_corrected_pity():
    pool = default_pool(load_default_hierarchy())
    assert pool.readings("pity") and not pool.readings("a pity")
    groups = pool.matching({Label.GROUP})
    assert groups
    assert any(e.type_name == "human_group" for e in pool.readings("the team"))
    assert all(e.number == "plural" for e in pool.readings("the police"))
    situation = Situation(frame_by_lemma("feel"),
                          {"experiencer": group(), "emotion": pool.readings("pity")[0]},
                          DEFAULT_PARADIGMS["feel"])
    assert realize(situation).text == "The team felt pity."


def test_group_inheritance_does_not_make_a_group_a_person():
    hierarchy = load_default_hierarchy()
    for subtype in ("human_group", "animal_group", "physical_object_group"):
        assert hierarchy.labels_of(subtype).entity_labels == (Label.GROUP,)
    assert not hierarchy.is_a("human_group", Label.PERSON)
    with pytest.raises(ValueError, match="at most one entity type"):
        Entity("a team", LabelSet.of(Label.GROUP, Label.PERSON))


@pytest.mark.parametrize("lemma,slot,human,animal,objects", [
    ("buy", "agent", True, False, False),
    ("buy", "theme", False, False, True),
    ("say", "speaker", True, False, False),
    ("own", "owner", True, False, False),
    ("give", "recipient", True, True, False),
    ("feel", "experiencer", True, True, False),
    ("visit", "visitor", True, True, False),
    ("move", "agent", True, True, False),
    ("see", "stimulus", True, True, True),
])
def test_group_members_constrain_slots(lemma, slot, human, animal, objects):
    constraint = frame_by_lemma(lemma).slot_map[slot]
    for subtype, accepted in zip(
        ("human_group", "animal_group", "physical_object_group"), (human, animal, objects),
    ):
        assert constraint.accepts({Label.GROUP}, type_name=subtype) is accepted


def test_sampling_validation_and_custom_descendants_share_constraints():
    hierarchy = load_default_hierarchy()
    hierarchy.add_type("rescue_team", "human_group")
    team = group(type_name="rescue_team")
    bundle = group("a bundle of books", "physical_object_group")
    unknown = group("a group", None)
    pool = EntityPool([team, bundle, unknown])
    buyer_slot = frame_by_lemma("buy").slot_map["agent"]
    assert pool.matching_slot(buyer_slot, hierarchy) == (team,)
    for invalid in (bundle, unknown, group(type_name="invented"), group(type_name="human")):
        with pytest.raises(BindingError):
            realize(purchase(invalid))
    generator = Generator(seed=3, pool=pool, hierarchy=hierarchy,
                          frames=(frame_by_lemma("buy"),))
    for result in generator.generate(10):
        assert result.situation.bindings == {"agent": team, "theme": bundle}
        assert result.check() == []


@pytest.mark.parametrize("features,expected", [
    ({"tense": "present"}, "The police buy a book."),
    ({"tense": "past"}, "The police bought a book."),
    ({"tense": "present", "is_negated": True}, "The police don't buy a book."),
    ({"tense": "present", "is_perfect": True}, "The police have bought a book."),
    ({"tense": "present", "is_perfect": True, "is_negated": True}, "The police haven't bought a book."),
    ({"tense": "present", "is_progressive": True}, "The police are buying a book."),
    ({"tense": "past", "is_progressive": True, "is_negated": True}, "The police weren't buying a book."),
    ({"tense": "will"}, "The police will buy a book."),
])
def test_plural_collective_agreement(features, expected):
    result = realize(purchase(group("the police", number="plural"), **features), "transitive")
    assert result.text == expected
    assert result.check() == []


def test_passive_agrees_with_theme_not_buyer():
    situation = purchase(group("the police", number="plural"), tense="present")
    assert realize(situation, "passive").text == "A book is bought by the police."
    goods = group("the bundles", "physical_object_group", "plural")
    result = realize(purchase(goods=goods, tense="past"), "passive")
    assert result.text == "The bundles were bought by the team."


def test_plural_questions_and_storage_round_trip(tmp_path):
    result = realize(purchase(group("the police", number="plural"), tense="present"), "transitive")
    document, run = to_canonical(result, document_id="police-buy")
    store = CorpusStore(tmp_path / "store")
    store.add_document(document)
    store.add_run(run, document)
    restored, = store.runs(document_id=document.document_id)
    assert restored == run
    examples = QuestionGenerator(paradigms=DEFAULT_PARADIGMS).for_document(document, restored)
    answers = {e.question: e.answer for e in examples}
    assert answers["What do the police buy?"] == "A book."
    assert answers["Which group buys a book?"] == "The police."
    assert answers["Do the police buy a book?"] == "Yes. The police buy a book."
    assert answers["What kind of entity are the police?"] == "Group."
    assert answers["Are the police a group?"] == (
        "Yes. The police are a human group, and every human group is a group."
    )
    assert all(e.check() == [] for e in examples)
    tagging = tag_document(document, restored)
    assert "B-GROUP" in tagging.tags


@pytest.mark.parametrize("features,question,answer", [
    ({"tense": "present", "is_progressive": True},
     "What are the police buying?", "The police are buying a book."),
    ({"tense": "present", "is_perfect": True, "is_negated": True},
     "What haven't the police bought?", "The police haven't bought a book."),
])
def test_question_auxiliaries_and_rebuilt_answers_agree(features, question, answer):
    result = realize(purchase(group("the police", number="plural"), **features), "transitive")
    document, run = to_canonical(result, document_id="auxiliaries")
    examples = QuestionGenerator(paradigms=DEFAULT_PARADIGMS).for_document(document, run)
    assert question in {e.question for e in examples}
    assert answer in {e.answer for e in examples}


def test_passive_question_uses_the_plural_theme_as_subject():
    result = realize(purchase(goods=group("the bundles", "physical_object_group", "plural"),
                              tense="past"), "passive")
    document, run = to_canonical(result, document_id="passive-group")
    examples = QuestionGenerator(paradigms=DEFAULT_PARADIGMS).for_document(document, run)
    assert "By which group were the bundles bought?" in {e.question for e in examples}


def test_number_round_trip_and_validation():
    entity = group("the police", number="plural")
    assert entity_from_json(entity_to_json(entity)) == entity
    with pytest.raises(ValueError, match="number"):
        replace(entity, number="guess")
