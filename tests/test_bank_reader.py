"""Streaming readers for the bank and for the legacy ``*.qa`` format."""

from __future__ import annotations

import gzip
import itertools
import json

import pytest

from semantic_corpus.qasrl_core.bank_reader import (
    read_bank,
    read_jsonl,
    read_qa_text,
)
from semantic_corpus.qasrl_core.models import Sentence


def test_reads_plain_jsonl(fixtures_dir):
    sentences = list(read_bank(fixtures_dir / "bank_sample.jsonl"))
    assert len(sentences) == 20
    assert all(isinstance(s, Sentence) for s in sentences)
    assert all(s.sentence_tokens for s in sentences)


def test_reads_gzip_without_unpacking(tmp_path, fixtures_dir):
    source = (fixtures_dir / "bank_sample.jsonl").read_bytes()
    packed = tmp_path / "sample.jsonl.gz"
    with gzip.open(packed, "wb") as stream:
        stream.write(source)

    plain = list(read_bank(fixtures_dir / "bank_sample.jsonl"))
    gzipped = list(read_bank(packed))
    assert [s.sentence_id for s in plain] == [s.sentence_id for s in gzipped]


def test_reading_is_lazy(fixtures_dir):
    stream = read_bank(fixtures_dir / "bank_sample.jsonl")
    first = next(stream)
    assert isinstance(first, Sentence)
    # Taking one sentence must not have consumed the rest.
    assert len(list(stream)) == 19


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "with_blanks.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert list(read_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_bad_json_names_the_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken\.jsonl:2"):
        list(read_jsonl(path))


def test_sentence_structure_is_fully_populated(fixtures_dir):
    sentences = list(read_bank(fixtures_dir / "bank_sample.jsonl"))
    for sentence in sentences:
        for key, entry in sentence.verb_entries.items():
            assert key.isdigit()
            assert int(key) == entry.verb_index
            assert 0 <= entry.verb_index < len(sentence.sentence_tokens)
            assert entry.verb_inflected_forms.stem
            for question, label in entry.question_labels.items():
                assert question == label.question_string
                assert label.question_string.endswith("?")


@pytest.mark.corpus
def test_streams_the_real_bank(bank_dev):
    sentences = list(itertools.islice(read_bank(bank_dev), 50))
    assert len(sentences) == 50
    assert any(sentence.verb_entries for sentence in sentences)


def test_reads_legacy_qa_text(qa_text_dev):
    sentences = list(itertools.islice(read_qa_text(qa_text_dev), 25))
    assert len(sentences) == 25
    for sentence in sentences:
        assert sentence.sentence_id
        assert sentence.tokens
        for predicate in sentence.predicates:
            assert 0 <= predicate.index < len(sentence.tokens)
            assert sentence.tokens[predicate.index] == predicate.verb
            for question in predicate.questions:
                assert len(question.slots) == 7
                assert question.text.endswith("?")
                assert question.answers


def test_legacy_qa_text_splits_alternative_answers(tmp_path):
    path = tmp_path / "tiny.qa"
    path.write_text(
        "WIKI1_1\t1\n"
        "A B C\n"
        "1\tB\t1\n"
        "what\t_\t_\tB\t_\t_\t_\t?\tA ### C\n"
        "\n",
        encoding="utf-8",
    )
    sentence = next(iter(read_qa_text(path)))
    assert sentence.sentence_id == "WIKI1_1"
    assert sentence.tokens == ("A", "B", "C")
    question = sentence.predicates[0].questions[0]
    assert question.answers == ("A", "C")
    assert question.text == "What B?"


def test_legacy_qa_text_rejects_unknown_lines(tmp_path):
    path = tmp_path / "bad.qa"
    path.write_text("WIKI1_1\t1\nA B C\nnonsense\tline\tx\ty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unrecognised line"):
        list(read_qa_text(path))
