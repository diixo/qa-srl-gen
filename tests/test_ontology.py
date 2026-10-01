"""Ontology v1: label combinations and type inheritance."""

from __future__ import annotations

import pytest

from semantic_corpus.ontology import (
    ENTITY_LABELS,
    Label,
    LabelSet,
    MarkerForm,
    Relation,
    SpeechAct,
    TypeHierarchy,
    load_default_hierarchy,
)


@pytest.fixture(scope="module")
def hierarchy() -> TypeHierarchy:
    return load_default_hierarchy()


# -- labels ----------------------------------------------------------------


def test_labels_are_not_mutually_exclusive():
    """The handoff's own examples, which a single-label scheme cannot express."""
    assert str(LabelSet.of(Label.STATE, Label.PROPERTY, Label.EMOTION)) == (
        "STATE + PROPERTY + EMOTION"
    )
    assert str(LabelSet.of(Label.ABSTRACT_ENTITY, Label.EMOTION)) == (
        "ABSTRACT_ENTITY + EMOTION"
    )
    assert str(LabelSet.of(Label.PERSON, Label.NAMED_ENTITY)) == "PERSON + NAMED_ENTITY"


def test_label_order_is_stable_regardless_of_insertion():
    first = LabelSet.of(Label.EMOTION, Label.STATE, Label.PROPERTY)
    second = LabelSet.of(Label.PROPERTY, Label.EMOTION, Label.STATE)
    assert first.ordered == second.ordered
    assert first == second


def test_action_and_state_cannot_combine():
    assert LabelSet.of(Label.ACTION).problems() == []
    assert LabelSet.of(Label.STATE).problems() == []
    assert LabelSet.of(Label.ACTION, Label.STATE).problems()


def test_a_span_has_at_most_one_entity_type():
    assert LabelSet.of(Label.PERSON, Label.NAMED_ENTITY).problems() == []
    assert LabelSet.of(Label.PERSON, Label.LOCATION).problems()


def test_named_object_implies_named_entity():
    good = LabelSet.of(Label.PHYSICAL_OBJECT, Label.NAMED_OBJECT, Label.NAMED_ENTITY)
    assert good.problems() == []
    assert LabelSet.of(Label.PHYSICAL_OBJECT, Label.NAMED_OBJECT).problems()


def test_entity_labels_are_reported_separately():
    labels = LabelSet.of(Label.ABSTRACT_ENTITY, Label.EMOTION)
    assert labels.entity_labels == (Label.ABSTRACT_ENTITY,)
    assert Label.EMOTION not in ENTITY_LABELS


def test_the_discourse_inventories_are_complete():
    assert MarkerForm.FILLED_PAUSE in MarkerForm
    assert len(SpeechAct) == 22
    assert len(Relation) == 13


# -- hierarchy -------------------------------------------------------------


def test_inheritance_reaches_the_ontology_label(hierarchy):
    """Paris is a city; every city is a location; therefore Paris is a location."""
    assert hierarchy.ancestors("city") == ("city", "place", "LOCATION")
    assert hierarchy.is_a("city", Label.LOCATION)
    assert hierarchy.is_a("city", "place")
    assert not hierarchy.is_a("city", Label.PERSON)


def test_the_handoff_examples_resolve(hierarchy):
    expected = {
        "city": Label.LOCATION,
        "dog": Label.ANIMAL,
        "woman": Label.PERSON,
        "car": Label.PHYSICAL_OBJECT,
        "ship": Label.PHYSICAL_OBJECT,
    }
    for type_name, label in expected.items():
        assert hierarchy.is_a(type_name, label), type_name
    assert hierarchy.labels_of("feeling") == LabelSet.of(
        Label.ABSTRACT_ENTITY, Label.EMOTION
    )


def test_extra_labels_are_inherited_too(hierarchy):
    assert hierarchy.is_a("feeling", Label.EMOTION)
    assert hierarchy.labels_of("affective_state") == LabelSet.of(
        Label.PROPERTY, Label.STATE, Label.EMOTION
    )


def test_lexicon_only_proposes(hierarchy):
    """A word with several readings returns all of them, not a guess."""
    assert hierarchy.is_ambiguous("jaguar")
    assert set(hierarchy.candidates("jaguar")) == {"big_cat", "car"}
    labels = {ls.entity_labels[0] for ls in hierarchy.candidate_labels("jaguar")}
    assert labels == {Label.ANIMAL, Label.PHYSICAL_OBJECT}


def test_unknown_words_are_empty_not_an_error(hierarchy):
    assert hierarchy.candidates("zzzz") == ()
    assert hierarchy.candidate_labels("zzzz") == ()
    assert not hierarchy.is_ambiguous("zzzz")


def test_descendants_and_children(hierarchy):
    assert "city" in hierarchy.children("place")
    assert "city" in hierarchy.descendants("LOCATION")
    assert "dog" not in hierarchy.descendants("LOCATION")


def test_unknown_parent_is_rejected():
    hierarchy = TypeHierarchy()
    with pytest.raises(KeyError, match="unknown parent"):
        hierarchy.add_type("hamlet", "settlement")


def test_an_ontology_label_cannot_become_a_subtype():
    hierarchy = TypeHierarchy()
    with pytest.raises(ValueError, match="cannot be a subtype"):
        hierarchy.add_type("PERSON", "ANIMAL")


def test_unknown_type_for_a_word_is_rejected(hierarchy):
    fresh = TypeHierarchy()
    fresh.add_type("city", "LOCATION")
    with pytest.raises(KeyError, match="unknown type"):
        fresh.add_word("hamlet", ["settlement"])


def test_defaults_round_trip_through_json(hierarchy, tmp_path):
    path = tmp_path / "ontology.json"
    hierarchy.save(path)
    restored = TypeHierarchy.load(path)
    assert restored.types == hierarchy.types
    assert restored.words == hierarchy.words
    assert restored.labels_of("feeling") == hierarchy.labels_of("feeling")


def test_types_may_be_declared_before_their_parents():
    data = {
        "types": {
            "capital": {"parent": "city"},
            "city": {"parent": "LOCATION"},
        },
        "lexicon": {"Kyiv": ["capital"]},
    }
    hierarchy = TypeHierarchy.from_dict(data)
    assert hierarchy.is_a("capital", Label.LOCATION)


def test_an_unreachable_parent_is_reported():
    with pytest.raises(ValueError, match="unresolvable parents"):
        TypeHierarchy.from_dict({"types": {"capital": {"parent": "nowhere"}}})
