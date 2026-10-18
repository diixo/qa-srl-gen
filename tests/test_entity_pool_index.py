"""Entity lookup optimisations must preserve existing dataset partitions."""

from __future__ import annotations

import hashlib

import pytest

from semantic_corpus.ontology import Label, LabelSet
from semantic_corpus.semantic_generator.substitutions import (
    SPLITS,
    Entity,
    EntityPool,
    default_pool,
)


def entity(text: str, label: Label = Label.PERSON, **kwargs) -> Entity:
    return Entity(text, LabelSet(frozenset({label})), **kwargs)


def legacy_split_key(text: str, forms: tuple[str, ...]) -> str:
    """Reference behaviour before indexing, including whitespace and ties."""
    words = text.split()
    best = text
    for other in forms:
        if other == text or len(other) >= len(best):
            continue
        other_words = other.split()
        span = len(other_words)
        if any(
            words[i : i + span] == other_words
            for i in range(len(words) - span + 1)
        ):
            best = other
    return best


def legacy_split_of(key: str, dev: float, test: float, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}\x00{key}".encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    if position < test:
        return "test"
    if position < test + dev:
        return "dev"
    return "train"


@pytest.mark.parametrize(
    ("forms", "external"),
    [
        (
            ("Bo", "Al", "Al Bo", "North America", "America"),
            ("Bo Al", "Al Bo Bo", "South America", "American", ""),
        ),
        (
            ("New  York", "New\tYork", "New York", "Yorkshire"),
            ("New\nYork", "the New\tYork office", "New York.", " York "),
        ),
        (
            ("Ann", "Ann-Marie", "Éva", "E\u0301va", "A\u00a0B"),
            ("ann", "Ann-Mary", "Dr Ann", "Éva's", "A B", " A\nB "),
        ),
        (
            ("   ", "\t", "Alice"),
            ("", " ", "Alice Smith", "Bob"),
        ),
        (
            ("Alice", "", "\n"),
            ("Bob", " ", "Alice Smith"),
        ),
        ((), ("", "Alice", "Alice Smith")),
    ],
)
def test_split_keys_preserve_legacy_matching(forms, external):
    pool = EntityPool(entity(text) for text in forms)
    for text in (*forms, *external):
        assert pool.split_key(text) == legacy_split_key(text, forms), repr(text)


def test_equal_length_matches_keep_pool_order_not_word_order():
    pool = EntityPool([entity("Bo"), entity("Al")])
    reversed_pool = EntityPool(reversed(pool.entities))
    assert pool.split_key("Al Bo") == "Bo"
    assert reversed_pool.split_key("Al Bo") == "Al"
    assert pool.split_key("Bo Al") == "Bo"


def test_new_surface_updates_existing_and_external_split_keys():
    long_name = entity("North America")
    pool = EntityPool([long_name])
    independent = EntityPool(pool)
    external = "the North America office"
    assert pool.split_key(long_name.text) == "North America"
    assert pool.split_key(external) == "North America"
    pool.split("train")
    pool.extend([entity("America"), entity("America")])
    assert pool.split_key(long_name.text) == "America"
    assert pool.split_key(external) == "America"
    assert independent.split_key(external) == "North America"
    assert pool.surface_forms == ("North America", "America")
    assert len(pool) == 2
    for split in SPLITS:
        assert pool.split(split).entities == tuple(
            item for item in pool
            if legacy_split_of("America", 0.1, 0.1, "") == split
        )


def test_shorter_whitespace_variant_updates_matching_candidates():
    pool = EntityPool([entity("New  York")])
    assert pool.split_key("the New York office") == "New  York"
    pool.add(entity("New\tYork"))
    assert pool.split_key("the New York office") == "New\tYork"
    pool.add(entity("New York"))
    # Equal raw lengths preserve the first inserted surface, even though the
    # token sequences and the words in the query are identical.
    assert pool.split_key("the New York office") == "New\tYork"


def test_readings_ambiguity_and_order_follow_mutations_without_duplicates():
    person = entity("Rex")
    glossed_person = entity("Rex", appositive="a mechanic")
    animal = entity("Rex", Label.ANIMAL)
    paris = entity("Paris")
    pool = EntityPool([person, paris, glossed_person, person])
    initial_readings = pool.readings("Rex")
    initial_forms = pool.surface_forms
    assert initial_readings == (person, glossed_person)
    assert pool.ambiguous_forms() == ()
    assert pool.readings("missing") == ()
    pool.extend([animal, animal, entity("Paris", Label.LOCATION)])
    assert pool.readings("Rex") == (person, glossed_person, animal)
    assert pool.ambiguous_forms() == ("Rex", "Paris")
    assert initial_readings == (person, glossed_person)
    assert pool.surface_forms == initial_forms == ("Rex", "Paris")
    assert pool.entities == (
        person, paris, glossed_person, animal, entity("Paris", Label.LOCATION),
    )


def test_builtin_keys_stay_stable_and_compounds_respect_every_held_out_name():
    pool = default_pool()
    forms = pool.surface_forms
    # Calculate the quadratic reference once, then reuse it for salts/shares.
    expected_keys = {text: legacy_split_key(text, forms) for text in forms}
    assert {text: pool.split_key(text) for text in forms} == expected_keys
    form_words = {text: tuple(text.split()) for text in forms}
    contained = {}
    for text, words in form_words.items():
        pieces = {words[start:end] for start in range(len(words) + 1)
                  for end in range(start, len(words) + 1)}
        contained[text] = [other for other, parts in form_words.items()
                           if len(other) < len(text) and parts in pieces]
    for dev, test, salt in (
        (0.1, 0.1, ""),
        (0.25, 0.2, "custom split"),
        (0.0, 0.0, ""),
        (0.01, 0.8, "\u00e9"),
    ):
        primary = {
            text: legacy_split_of(key, dev, test, salt)
            for text, key in expected_keys.items()
        }
        # Independent, slow reference: every complete contained pool phrase
        # can reserve its enclosing compound for a held-out subset.
        priority = {"train": 0, "dev": 1, "test": 2}
        assignments = {
            text: max([primary[text], *(primary[other] for other in contained[text])],
                      key=priority.__getitem__)
            for text in forms
        }
        assert {
            text: pool.split_of(text, dev_share=dev, test_share=test, salt=salt)
            for text in forms
        } == assignments
        for split in SPLITS:
            assert pool.split(
                split, dev_share=dev, test_share=test, salt=salt,
            ).entities == tuple(item for item in pool if assignments[item.text] == split)


def test_short_surname_cannot_leak_a_held_out_given_name():
    pool = EntityPool(entity(text) for text in ("Nathaniel", "Brown", "Nathaniel Brown"))
    assert pool.split_key("Nathaniel Brown") == "Brown"
    assert pool.split_of("Brown") == "train"
    assert pool.split_of("Nathaniel") == "test"
    assert pool.split_of("Nathaniel Brown") == "test"
    assert "Nathaniel Brown" not in pool.split("train").surface_forms


def test_compound_reservations_use_custom_shares_salt_and_new_readings():
    names = [f"Name{i}" for i in range(200)]
    shares = dict(dev_share=0.25, test_share=0.2, salt="entity regression")
    by_split = {s: next(n for n in names if legacy_split_of(n, 0.25, 0.2, shares['salt']) == s)
                for s in SPLITS}
    pool = EntityPool(entity(n) for n in by_split.values())
    train_dev = by_split['train'] + ' ' + by_split['dev']
    train_test = by_split['train'] + ' ' + by_split['test']
    assert pool.split_of(train_dev, **shares) == 'dev'
    assert pool.split_of(train_test, **shares) == 'test'
    pool.extend([entity(train_test), entity(train_test, Label.LOCATION)])
    assert len(pool.split('test', **shares).readings(train_test)) == 2
    assert not pool.split('train', **shares).readings(train_test)
    assert pool.split_of(train_test, dev_share=0, test_share=0) == 'train'
