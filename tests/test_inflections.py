"""Wiktionary inflection indexes: forward lookup, reverse lookup, homonymy."""

from __future__ import annotations

import pytest

from semantic_corpus.qasrl_core.inflections import (
    SUPPLETIVE_PARADIGMS,
    load_inflections,
    load_postags,
    load_verb_phrases,
)
from semantic_corpus.qasrl_core.models import VerbForm


@pytest.fixture(scope="module")
def sample(fixtures_dir):
    return load_inflections(fixtures_dir / "inflections_sample.txt")


def test_forward_index_returns_the_paradigm(sample):
    give = sample.paradigm("give")
    assert give is not None
    assert give.all_forms == ("give", "gives", "giving", "gave", "given")
    go = sample.paradigm("go")
    assert go.past == "went"
    assert go.past_participle == "gone"


def test_forward_index_keeps_competing_paradigms(sample):
    # The scrape records two past tenses for these lemmas; neither is dropped.
    assert len(sample.paradigms("awaken")) == 2
    assert {p.past for p in sample.paradigms("awaken")} == {"awoke", "awakened"}
    assert len(sample.paradigms("chide")) == 2


def test_unknown_lemma_is_empty_not_an_error(sample):
    assert sample.paradigms("zzzz") == ()
    assert sample.paradigm("zzzz") is None


def test_junk_rows_are_dropped(sample):
    assert "-" not in sample
    assert sample.analyses("-") == ()


def test_reverse_index_resolves_inflected_forms(sample):
    assert sample.lemmas_for("gave") == ("give",)
    assert sample.lemmas_for("given") == ("give",)
    assert sample.lemmas_for("went") == ("go",)
    assert sample.lemmas_for("gone") == ("go",)
    assert sample.forms_of("gave") == (VerbForm.PAST,)


def test_reverse_index_reports_several_forms_for_one_surface(sample):
    # One paradigm, three forms spelled the same.
    assert sample.forms_of("put") == (
        VerbForm.STEM,
        VerbForm.PAST,
        VerbForm.PAST_PARTICIPLE,
    )
    assert sample.lemmas_for("put") == ("put",)


def test_reverse_index_reports_several_lemmas_for_one_surface(sample):
    # "left" is both the past of "leave" and, in the full dictionary, a lemma
    # of its own; within this sample it must at least find "leave".
    assert "leave" in sample.lemmas_for("left")
    analyses = sample.analyses("left")
    assert any(form is VerbForm.PAST for _, form in analyses)


def test_be_is_absent_from_the_scrape_and_supplied_separately(fixtures_dir):
    raw = load_inflections(fixtures_dir / "inflections_sample.txt", include_suppletive=False)
    assert "be" not in raw
    with_be = load_inflections(fixtures_dir / "inflections_sample.txt")
    assert with_be.paradigm("be") == SUPPLETIVE_PARADIGMS["be"]
    assert with_be.lemmas_for("been") == ("be",)
    assert with_be.lemmas_for("is") == ("be",)


def test_malformed_row_is_reported_with_its_line(tmp_path):
    path = tmp_path / "broken.txt"
    path.write_text("give\tgives\tgiving\tgave\tgiven\nonly\ttwo\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken\.txt:2"):
        load_inflections(path)


@pytest.mark.corpus
def test_full_dictionary_milestone_requirements(lexicon):
    """The milestone names these lookups explicitly."""
    give = lexicon.paradigm("give")
    assert (give.stem, give.past, give.past_participle) == ("give", "gave", "given")
    go = lexicon.paradigm("go")
    assert (go.stem, go.past, go.past_participle) == ("go", "went", "gone")

    for surface in ("give", "gave", "given", "go", "went", "gone"):
        assert lexicon.analyses(surface), f"{surface} not found in the reverse index"

    # Homonymy must surface as several candidates, not a single guess.
    # One surface, one lemma, several forms:
    assert len(lexicon.forms_of("read")) > 1
    # One surface, several lemmas:
    assert set(lexicon.lemmas_for("ground")) >= {"grind", "ground"}
    assert set(lexicon.lemmas_for("saw")) >= {"see", "saw"}
    assert len(lexicon.analyses("found")) > 2


@pytest.mark.corpus
def test_postags_preserve_sense_frequency(postags_path):
    tags = load_postags(postags_path)
    assert tags["dictionary"] == ("noun", "verb")
    # Repetition is a frequency signal and must not be collapsed.
    assert tags["cat"].count("noun") > 1


@pytest.mark.corpus
def test_verb_phrases_are_tokenised(verb_phrases_path):
    phrases = load_verb_phrases(verb_phrases_path)
    assert len(phrases) > 2000
    assert all(len(p) >= 1 for p in phrases)
    assert any(len(p) > 1 for p in phrases)
