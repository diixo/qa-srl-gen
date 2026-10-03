"""Regression checks for encoded writes and persistent membership caching."""

from dataclasses import replace

import pytest

from semantic_corpus.documents import AnnotationRun, Document
from semantic_corpus.question_generator import QAExample
from semantic_corpus.storage import CorpusStore, StoreError


def example_setup(path):
    store = CorpusStore(path)
    doc = Document("d", "Anna visited Málaga 🎉.")
    store.add_document(doc)
    store.add_run(AnnotationRun("r"), doc)
    example = QAExample(doc.text, "Who visited?", "Anna.", document_id="d", run_id="r",
                        metadata={"unicode": "Київ 🎉", "newline": "a\nb"})
    return store, doc, example


def test_multiple_live_indexes_follow_encoded_appends_and_reopen(tmp_path):
    store, doc, example = example_setup(tmp_path)
    for key in ("document_id", "run_id", "question"):
        assert list(store._rows_for("qa_examples", key, "missing")) == []
    assert store.add_examples([example]) == 1
    second = replace(example, question="Where did Anna go?", answer="Málaga.")
    assert store.add_examples([second]) == 1
    assert len(list(store._rows_for("qa_examples", "document_id", "d"))) == 2
    assert len(list(store._rows_for("qa_examples", "run_id", "r"))) == 2
    assert list(store._rows_for("qa_examples", "question", second.question))[0]["answer"] == "Málaga."
    assert list(CorpusStore(tmp_path).examples()) == [example, second]


def test_duplicate_rows_are_written_when_deduplication_is_disabled(tmp_path):
    store, _, example = example_setup(tmp_path)
    assert store.add_examples([example, example], deduplicate=False) == 2
    assert store.count("qa_examples") == 2
    assert store.add_examples([example]) == 0
    assert CorpusStore(tmp_path).add_examples([example]) == 0


@pytest.mark.parametrize("live_index", [False, True])
def test_spooled_large_utf8_batch_is_identical_with_live_indexes(tmp_path, live_index):
    store, _, example = example_setup(tmp_path)
    if live_index:
        list(store._rows_for("qa_examples", "document_id", "d"))
    examples = [replace(example, question=f"Who visited in case {i}?",
                        metadata={"payload": "🎉" * 20000}) for i in range(20)]
    assert store.add_examples(iter(examples)) == 20
    assert store.path_for("qa_examples").stat().st_size > 1024 * 1024
    assert list(store.examples()) == examples
    if live_index:
        assert len(list(store._rows_for("qa_examples", "document_id", "d"))) == 20


def test_persistent_cache_does_not_accept_wrong_document_on_later_batch(tmp_path):
    store, _, example = example_setup(tmp_path)
    store.add_document(Document("other", "Another document."))
    assert store.add_examples([example]) == 1
    before = store.content_hash()
    with pytest.raises(StoreError, match="does not annotate"):
        store.add_examples([replace(example, document_id="other")])
    assert store.content_hash() == before


def test_serialization_error_with_live_index_does_not_write_any_rows(tmp_path):
    store, _, example = example_setup(tmp_path)
    list(store._rows_for("qa_examples", "question", "missing"))
    before = store.content_hash()
    bad = replace(example, question="Bad?", metadata={"path": tmp_path})
    with pytest.raises(StoreError, match="cannot serialize"):
        store.add_examples([example, bad])
    assert store.content_hash() == before
    assert list(store._rows_for("qa_examples", "question", example.question)) == []
    assert store.add_examples([example]) == 1
