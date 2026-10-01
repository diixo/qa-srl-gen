"""The verb-chain grammar, checked against the inventory the bank actually uses."""

from __future__ import annotations

import gzip
import json

import pytest

from semantic_corpus.qasrl_core.state_machine import (
    AUX_WORDS,
    VERB_PREFIX_WORDS,
    TenseFeatures,
    VerbChain,
    build_verb_chain,
    enumerate_verb_chains,
    features_for_chain,
    is_known_chain,
)


def chain(**kwargs) -> tuple[str, str]:
    built = build_verb_chain(TenseFeatures(**kwargs))
    return built.aux, built.verb


def test_simple_present_with_and_without_do_support():
    # The questioned argument is the subject: nothing to invert.
    assert chain(tense="present", subject_is_questioned=True) == (
        "_",
        "presentSingular3rd",
    )
    # Any other argument: do-support supplies something to front.
    assert chain(tense="present") == ("does", "stem")


def test_simple_past_mirrors_the_present():
    assert chain(tense="past", subject_is_questioned=True) == ("_", "past")
    assert chain(tense="past") == ("did", "stem")


def test_passive_fronts_the_copula():
    assert chain(tense="present", is_passive=True) == ("is", "pastParticiple")
    assert chain(tense="past", is_passive=True) == ("was", "pastParticiple")


def test_progressive_passive_stacks_two_copulas():
    assert chain(tense="present", is_progressive=True, is_passive=True) == (
        "is",
        "being pastParticiple",
    )


def test_perfect_fronts_have():
    assert chain(tense="present", is_perfect=True) == ("has", "pastParticiple")
    assert chain(tense="past", is_perfect=True) == ("had", "pastParticiple")
    assert chain(tense="present", is_perfect=True, is_passive=True) == (
        "has",
        "been pastParticiple",
    )
    assert chain(tense="present", is_perfect=True, is_progressive=True) == (
        "has",
        "been presentParticiple",
    )


def test_modals_take_bare_complements():
    assert chain(tense="can") == ("can", "stem")
    assert chain(tense="can", is_passive=True) == ("can", "be pastParticiple")
    assert chain(tense="might", is_perfect=True) == ("might", "have pastParticiple")
    assert chain(tense="might", is_perfect=True, is_passive=True) == (
        "might",
        "have been pastParticiple",
    )


def test_negation_contracts_onto_the_fronted_auxiliary():
    assert chain(tense="present", is_negated=True) == ("doesn't", "stem")
    assert chain(tense="past", is_negated=True) == ("didn't", "stem")
    assert chain(tense="present", is_passive=True, is_negated=True) == (
        "isn't",
        "pastParticiple",
    )
    assert chain(tense="can", is_negated=True) == ("can't", "stem")
    assert chain(tense="will", is_negated=True) == ("won't", "stem")


def test_might_has_no_contraction_so_not_moves_into_the_verb_slot():
    assert chain(tense="might", is_negated=True) == ("might", "not stem")
    assert chain(tense="might", is_passive=True, is_negated=True) == (
        "might",
        "not be pastParticiple",
    )


def test_negation_forces_do_support_even_for_a_questioned_subject():
    assert chain(tense="present", is_negated=True, subject_is_questioned=True) == (
        "doesn't",
        "stem",
    )


def test_unknown_tense_is_rejected():
    with pytest.raises(ValueError, match="unknown tense"):
        TenseFeatures(tense="pluperfect")


def test_aux_and_verb_prefix_vocabularies_do_not_overlap():
    """What makes a question prefix splittable without backtracking."""
    assert not (AUX_WORDS & VERB_PREFIX_WORDS)


def test_inverse_mapping_recovers_the_features():
    features = TenseFeatures(tense="might", is_perfect=True, is_passive=True)
    built = build_verb_chain(features)
    recovered = features_for_chain(built.aux, built.verb)
    assert any(
        (f.tense, f.is_perfect, f.is_passive) == ("might", True, True) for f in recovered
    )


def test_illegal_chains_are_rejected():
    assert not is_known_chain("does", "pastParticiple")
    assert not is_known_chain("is", "stem")
    assert not is_known_chain("_", "stem")
    assert is_known_chain("is", "being pastParticiple")


def _attested_chains(path):
    found = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            sentence = json.loads(line)
            for entry in sentence["verbEntries"].values():
                for label in entry["questionLabels"].values():
                    slots = label["questionSlots"]
                    found[VerbChain(slots["aux"], slots["verb"])] = (
                        label["tense"],
                        label["isPerfect"],
                        label["isProgressive"],
                        label["isPassive"],
                        label["isNegated"],
                        slots["subj"],
                    )
    return found


@pytest.mark.corpus
def test_grammar_covers_every_chain_the_bank_uses(bank_dev, bank_dense_dev):
    """No question in the bank uses an ``(aux, verb)`` pair this grammar cannot build."""
    generated = set(enumerate_verb_chains())
    for path in (bank_dev, bank_dense_dev):
        attested = _attested_chains(path)
        missing = sorted((c.aux, c.verb) for c in set(attested) - generated)
        assert not missing, f"{path.name}: chains the grammar cannot produce: {missing}"


@pytest.mark.corpus
def test_stored_features_always_explain_the_stored_chain(bank_dev):
    """The features recorded with a question must be able to produce its slots."""
    for chain_key, stored in _attested_chains(bank_dev).items():
        tense, perfect, progressive, passive, negated, subj = stored
        candidates = features_for_chain(chain_key.aux, chain_key.verb)
        assert candidates, f"{chain_key} is not a chain the grammar knows"
        assert any(
            (f.tense, f.is_perfect, f.is_progressive, f.is_passive, f.is_negated)
            == (tense, perfect, progressive, passive, negated)
            for f in candidates
        ), f"{chain_key} cannot come from {stored}"
        if chain_key.aux == "_":
            assert subj == "_", "an unfronted question must be questioning its subject"
