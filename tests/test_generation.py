"""The semantic generator: typed frames, realisation, variation, determinism."""

from __future__ import annotations

from dataclasses import replace

import pytest

from semantic_corpus.ontology import Label, Relation, load_default_hierarchy
from semantic_corpus.qasrl_core.models import InflectedForms
from semantic_corpus.semantic_generator import (
    DEFAULT_FRAMES,
    BindingError,
    Entity,
    Features,
    Generator,
    Situation,
    all_realizations,
    default_pool,
    enumerate_features,
    frame_by_lemma,
    pattern_variants,
    realize,
)
from semantic_corpus.semantic_generator.substitutions import SPLITS

GIVE_FORMS = InflectedForms("give", "gives", "giving", "gave", "given")


@pytest.fixture(scope="module")
def pool():
    return default_pool(load_default_hierarchy())


@pytest.fixture(scope="module")
def give():
    return frame_by_lemma("give")


def entity(pool, text: str, *, appositive: str | None = None) -> Entity:
    for candidate in pool.readings(text):
        if candidate.appositive == appositive:
            return candidate
    raise AssertionError(f"{text!r} not in the pool")


@pytest.fixture
def anna_gives(pool, give):
    return Situation(
        frame=give,
        bindings={
            "agent": entity(pool, "Anna"),
            "recipient": entity(pool, "Rex", appositive="a German shepherd"),
            "theme": entity(pool, "a red ball"),
            "location": entity(pool, "Kyiv"),
        },
        forms=GIVE_FORMS,
        features=Features(tense="past"),
    )


# -- frames ----------------------------------------------------------------


def test_every_default_frame_is_well_formed():
    for frame in DEFAULT_FRAMES:
        assert frame.predicate_type in (Label.ACTION, Label.STATE)
        assert frame.required_slots
        assert frame.patterns
        for pattern in frame.patterns:
            assert pattern.mentioned() <= set(frame.slot_map)


def test_a_frame_rejects_a_pattern_over_unknown_slots(give):
    from semantic_corpus.semantic_generator.frames import (
        SemanticFrame,
        SurfacePattern,
    )

    with pytest.raises(ValueError, match="unknown slots"):
        SemanticFrame(
            lemma="x",
            predicate_type=Label.ACTION,
            slots=give.slots,
            patterns=(SurfacePattern("bad", subject="nobody"),),
        )


def test_slot_types_are_enforced(give):
    agent = give.slot_map["agent"]
    assert agent.accepts({Label.PERSON})
    assert agent.accepts({Label.ORGANIZATION})
    assert not agent.accepts({Label.ANIMAL})


def test_slot_exclusions_override_the_type(give):
    """``give`` takes an abstract theme but not an emotional one."""
    theme = give.slot_map["theme"]
    assert theme.accepts({Label.ABSTRACT_ENTITY})
    assert not theme.accepts({Label.ABSTRACT_ENTITY, Label.EMOTION})


# -- realisation -----------------------------------------------------------


def test_the_handoff_example(anna_gives):
    assert realize(anna_gives, "ditransitive").text == (
        "Anna gave Rex, a German shepherd, a red ball in Kyiv."
    )


def test_patterns_reorder_the_same_meaning(pool, give):
    situation = Situation(
        frame=give,
        bindings={
            "agent": entity(pool, "Anna"),
            "recipient": entity(pool, "Rex"),
            "theme": entity(pool, "a red ball"),
        },
        forms=GIVE_FORMS,
        features=Features(tense="past"),
    )
    produced = {p.name: realize(situation, p).text for p in pattern_variants(situation)}
    assert produced["ditransitive"] == "Anna gave Rex a red ball."
    assert produced["prepositional"] == "Anna gave a red ball to Rex."
    assert produced["passive"] == "A red ball was given to Rex by Anna."
    assert produced["passive_agentless"] == "A red ball was given to Rex."


def test_tense_negation_and_modality(pool, give):
    situation = Situation(
        frame=give,
        bindings={
            "agent": entity(pool, "Anna"),
            "recipient": entity(pool, "Rex"),
            "theme": entity(pool, "a red ball"),
        },
        forms=GIVE_FORMS,
    )

    def say(**features):
        return realize(
            replace(situation, features=Features(**features)), "ditransitive"
        ).text

    assert say(tense="past") == "Anna gave Rex a red ball."
    assert say(tense="present") == "Anna gives Rex a red ball."
    assert say(tense="will") == "Anna will give Rex a red ball."
    assert say(tense="past", is_negated=True) == "Anna didn't give Rex a red ball."
    assert say(tense="present", is_perfect=True) == "Anna has given Rex a red ball."
    assert say(tense="present", is_progressive=True) == "Anna is giving Rex a red ball."
    assert say(tense="might", is_negated=True) == "Anna might not give Rex a red ball."


def test_every_span_is_exact(anna_gives):
    for pattern in pattern_variants(anna_gives):
        realized = realize(anna_gives, pattern)
        assert realized.check() == []
        for mention in realized.mentions:
            assert realized.text[mention.start_char : mention.end_char] == mention.text


def test_relations_follow_the_frame(anna_gives):
    realized = realize(anna_gives, "ditransitive")
    found = {(e.source_slot, e.relation) for e in realized.relations}
    assert ("agent", Relation.AGENT_OF) in found
    assert ("recipient", Relation.RECIPIENT_OF) in found
    assert ("theme", Relation.THEME_OF) in found
    assert ("location", Relation.LOCATION_OF) in found
    # The appositive is recorded as an instance link, not as another argument.
    assert ("recipient", Relation.INSTANCE_OF) in found


def test_an_appositive_is_its_own_span(anna_gives):
    realized = realize(anna_gives, "ditransitive")
    gloss = realized.mention("recipient:appositive")
    assert gloss is not None
    assert gloss.text == "a German shepherd"
    assert realized.text[gloss.start_char : gloss.end_char] == "a German shepherd"


def test_a_sentence_final_appositive_loses_its_comma(pool, give):
    situation = Situation(
        frame=give,
        bindings={
            "agent": entity(pool, "Anna"),
            "recipient": entity(pool, "Rex", appositive="a German shepherd"),
            "theme": entity(pool, "a red ball"),
        },
        forms=GIVE_FORMS,
    )
    text = realize(situation, "prepositional").text
    assert text == "Anna gave a red ball to Rex, a German shepherd."
    assert ",." not in text


def test_a_mistyped_binding_is_refused(pool, give):
    situation = Situation(
        frame=give,
        bindings={
            "agent": entity(pool, "Rex"),  # an animal cannot be give's agent
            "recipient": entity(pool, "Anna"),
            "theme": entity(pool, "a red ball"),
        },
        forms=GIVE_FORMS,
    )
    with pytest.raises(BindingError, match="agent"):
        realize(situation, "ditransitive")


def test_a_missing_required_slot_is_refused(pool, give):
    situation = Situation(
        frame=give,
        bindings={"agent": entity(pool, "Anna")},
        forms=GIVE_FORMS,
    )
    with pytest.raises(BindingError, match="unbound"):
        realize(situation, "ditransitive")


# -- variation -------------------------------------------------------------


def test_unlicensed_feature_bundles_are_never_offered():
    passive = enumerate_features(tenses=("can",), is_passive=True)
    assert passive
    assert not any(f.is_progressive for f in passive)
    active = enumerate_features(tenses=("present",), is_passive=True)
    assert any(f.is_progressive for f in active)


def test_variation_keeps_the_record_intact(anna_gives):
    produced = list(all_realizations(anna_gives, tenses=("past", "present")))
    assert len(produced) > 8
    assert len({r.text for r in produced}) == len(produced)
    for realized in produced:
        assert realized.check() == []
        assert realized.situation.bindings == anna_gives.bindings


# -- the generator ---------------------------------------------------------


def test_generation_is_reproducible_from_the_seed():
    first = [r.text for r in Generator(seed=7).generate(25)]
    second = [r.text for r in Generator(seed=7).generate(25)]
    assert first == second
    assert [r.text for r in Generator(seed=8).generate(25)] != first


def test_reset_rewinds_the_stream():
    generator = Generator(seed=11)
    first = [r.text for r in generator.generate(10)]
    generator.reset()
    assert [r.text for r in generator.generate(10)] == first


def test_generated_sentences_always_type_check_and_locate_their_spans():
    for realized in Generator(seed=3).generate(200):
        assert realized.situation.validate() == []
        assert realized.check() == []
        assert realized.text.endswith(".")
        assert realized.text[0].isupper()


def test_held_out_names_never_appear_in_training():
    """``unseen_names_test`` is only meaningful if this holds."""
    train = {r.text for r in Generator(seed=5, split="train").generate(300)}
    held_out = Generator(seed=5, split="test").pool.surface_forms
    assert held_out
    for name in held_out:
        assert not any(name in sentence for sentence in train), name


def test_splits_partition_the_pool():
    pool = default_pool(load_default_hierarchy())
    sizes = {name: len(pool.split(name).surface_forms) for name in SPLITS}
    assert sum(sizes.values()) == len(pool.surface_forms)
    assert all(size > 0 for size in sizes.values())


def test_split_assignment_does_not_depend_on_run_or_order():
    pool = default_pool(load_default_hierarchy())
    assignment = {form: pool.split_of(form) for form in pool.surface_forms}
    shuffled = default_pool(load_default_hierarchy())
    assert {f: shuffled.split_of(f) for f in shuffled.surface_forms} == assignment


def test_every_reading_of_a_name_lands_in_one_split():
    pool = default_pool(load_default_hierarchy())
    for form in pool.ambiguous_forms():
        splits = {pool.split_of(e.text) for e in pool.readings(form)}
        assert len(splits) == 1, form


def test_ambiguous_names_are_disambiguated_by_context():
    pairs = {p.surface_form: p for p in Generator(seed=2).ambiguity_pairs()}
    assert set(pairs) >= {"Rex", "Paris", "Jaguar"}
    rex = pairs["Rex"]
    assert {rex.left_label, rex.right_label} == {Label.ANIMAL, Label.PERSON}
    # Each sentence must carry the gloss that settles the reading.
    for realized in (rex.left, rex.right):
        gloss = [m for m in realized.mentions if m.slot.endswith(":appositive")]
        assert gloss, realized.text


def test_omitted_arguments_produce_genuinely_unanswerable_questions():
    produced = list(Generator(seed=4).omitted_argument_examples(10))
    assert len(produced) == 10
    for realized in produced:
        assert realized.omitted_slots
        for slot in realized.omitted_slots:
            # The record knows the slot exists; the sentence does not name it.
            assert realized.mention(slot) is None
            assert slot in realized.situation.frame.slot_map


def test_the_pool_and_the_ontology_agree():
    """Construction fails loudly if a common noun contradicts the hierarchy."""
    hierarchy = load_default_hierarchy()
    pool = default_pool(hierarchy)
    for item in pool:
        if item.type_name and item.type_name in hierarchy:
            inherited = hierarchy.labels_of(item.type_name)
            entity_label = item.entity_label
            if entity_label is not None and inherited.entity_labels:
                assert entity_label in inherited.labels, item.text
