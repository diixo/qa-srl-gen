"""Storage I/O stays bounded per run without weakening ownership checks."""

from dataclasses import replace
from pathlib import Path

import pytest

from semantic_corpus.documents import AnnotationRun, DialogueAnnotation, Document, TextSpan, Utterance
from semantic_corpus.storage import CorpusStore, StoreError
from semantic_corpus.question_generator import QAExample


def dialogue(doc_id, count=3):
    words = [f"{doc_id}-{index}" for index in range(count)]
    text = "\n".join(words)
    cursor = 0
    utterances = []
    for index, word in enumerate(words):
        utterances.append(Utterance(f"{doc_id}-u{index}", doc_id, index, "speaker",
                                    TextSpan(cursor, cursor + len(word))))
        cursor += len(word) + 1
    return Document(doc_id, text, utterances=tuple(utterances))


@pytest.mark.parametrize("count", [2, 40])
@pytest.mark.parametrize("reopen", [False, True])
def test_run_reads_document_at_most_once_and_batches_utterance_owners(tmp_path, monkeypatch, count, reopen):
    store = CorpusStore(tmp_path)
    document = dialogue("d", count)
    store.add_document(document)
    if reopen:
        store = CorpusStore(tmp_path)
    # Exclude lazy one-time index construction from the per-run I/O count.
    store.document("d")
    list(store._rows_for("utterances", "utterance_id", "missing"))
    reads, loads = [], []
    original_open, original_document = Path.open, store.document

    def counted_open(path, mode="r", *args, **kwargs):
        if path == store.path_for("utterances") and mode == "rb":
            reads.append(path)
        return original_open(path, mode, *args, **kwargs)

    def counted_document(document_id):
        loads.append(document_id)
        return original_document(document_id)

    monkeypatch.setattr(Path, "open", counted_open)
    monkeypatch.setattr(store, "document", counted_document)
    run = AnnotationRun("r", dialogue=tuple(DialogueAnnotation(u.utterance_id) for u in document.utterances))
    store.add_run(run, document)
    assert loads == (["d"] if reopen else [])
    # The validated document supplies ownership; the global index checks that
    # each utterance ID is unique, without reopening the file per annotation.
    assert len(reads) == (1 if reopen else 0)
    assert next(store.runs()) == run


@pytest.mark.parametrize("provided", [False, True])
@pytest.mark.parametrize("problem", ["missing", "ambiguous"])
def test_batch_owner_lookup_rejects_missing_and_duplicate_ids_without_writes(tmp_path, provided, problem):
    store = CorpusStore(tmp_path)
    document = dialogue("a")
    store.add_document(document)
    uid = document.utterances[0].utterance_id
    # Build the index before adding a conflicting document; appends must update it.
    assert len(list(store._rows_for("utterances", "utterance_id", uid))) == 1
    if problem == "ambiguous":
        other = dialogue("b", 1)
        store.add_document(replace(other, utterances=(replace(other.utterances[0], utterance_id=uid),)))
    else:
        uid = "absent"
    run = AnnotationRun("bad", dialogue=(DialogueAnnotation(uid),))
    before = store.content_hash()
    with pytest.raises(StoreError, match="unknown or ambiguous utterance"):
        store.add_run(run, document if provided else None)
    assert store.content_hash() == before
    assert list(store.runs()) == []


def test_owner_batch_supports_a_run_over_multiple_documents(tmp_path):
    store = CorpusStore(tmp_path)
    documents = [dialogue("a"), dialogue("b")]
    for document in documents:
        store.add_document(document)
    run = AnnotationRun("multi", dialogue=tuple(
        DialogueAnnotation(u.utterance_id) for d in documents for u in d.utterances
    ))
    store.add_run(run)
    assert next(store.read("annotation_runs"))["document_ids"] == ["a", "b"]
    assert next(store.runs()) == run
    for document in documents:
        restored, = store.runs(document_id=document.document_id)
        assert {d.utterance_id for d in restored.dialogue} == {u.utterance_id for u in document.utterances}


def test_supplied_document_is_still_compared_with_stored_content(tmp_path):
    store = CorpusStore(tmp_path)
    document = dialogue("a")
    store.add_document(document)
    before = store.content_hash()
    with pytest.raises(StoreError, match="stored unchanged"):
        store.add_run(AnnotationRun("bad"), replace(document, metadata={"changed": True}))
    assert store.content_hash() == before
    store.add_run(AnnotationRun("good"), document)
    assert next(store.runs()).run_id == "good"


def test_batched_reads_match_single_reads_after_reopen_and_append(tmp_path):
    store = CorpusStore(tmp_path)
    first = dialogue("a")
    store.add_document(first)
    store = CorpusStore(tmp_path, create=False)
    uid = first.utterances[0].utterance_id
    assert list(store._rows_for_many("utterances", "utterance_id", [uid]))
    second = dialogue("b")
    store.add_document(second)
    keys = [uid, second.utterances[-1].utterance_id]
    expected = [row for key in keys for row in store._rows_for("utterances", "utterance_id", key)]
    assert list(store._rows_for_many("utterances", "utterance_id", [*keys, uid, "absent"])) == expected


def test_recent_document_fingerprint_does_not_trust_mutable_metadata(tmp_path):
    store = CorpusStore(tmp_path)
    nested = {"source": {"tags": ["original"]}}
    document = Document("d", "Hello.", metadata=nested)
    store.add_document(document)
    before = store.content_hash()
    nested["source"]["tags"].append("changed")
    with pytest.raises(StoreError, match="stored unchanged"):
        store.add_run(AnnotationRun("r"), document)
    assert store.content_hash() == before
    assert store.document("d").metadata == {"source": {"tags": ["original"]}}


def test_validated_document_cannot_hide_a_foreign_dialogue_annotation(tmp_path):
    store = CorpusStore(tmp_path)
    first, second = dialogue("a"), dialogue("b")
    store.add_document(first)
    store.add_document(second)
    run = AnnotationRun("bad", dialogue=(DialogueAnnotation(second.utterances[0].utterance_id),))
    before = store.content_hash()
    with pytest.raises(StoreError, match="does not describe"):
        store.add_run(run, first)
    assert store.content_hash() == before


def test_recent_run_membership_avoids_disk_reads_but_rejects_wrong_document(tmp_path, monkeypatch):
    store = CorpusStore(tmp_path)
    document = dialogue("d", 1)
    store.add_document(document)
    store.add_document(dialogue("other", 1))
    store.add_run(AnnotationRun("r"), document)
    original = store._rows_for

    def no_header_read(table, key, value):
        if table == "annotation_runs":
            pytest.fail("just-written run header was reopened")
        return original(table, key, value)

    monkeypatch.setattr(store, "_rows_for", no_header_read)
    example = QAExample(document.text, "Who?", "d.", document_id="d", run_id="r")
    assert store.add_examples([example], generation_run_id="r") == 1
    before = store.content_hash()
    with pytest.raises(StoreError, match="does not annotate"):
        store.add_examples([replace(example, document_id="other")])
    assert store.content_hash() == before


def test_recent_write_eviction_falls_back_to_disk_and_preserves_membership(tmp_path, monkeypatch):
    # Reduce the bound to exercise eviction without timing-dependent assertions.
    monkeypatch.setattr("semantic_corpus.storage.repository._RECENT_WRITES", 2)
    store = CorpusStore(tmp_path)
    for index in range(3):
        doc = dialogue(str(index), 1)
        store.add_document(doc)
        store.add_run(AnnotationRun(f"r{index}"), doc)
    old = dialogue("0", 1)
    loads = []
    original = store.document

    def counted_document(document_id):
        loads.append(document_id)
        return original(document_id)

    monkeypatch.setattr(store, "document", counted_document)
    store.add_run(AnnotationRun("another"), old)
    assert loads == ["0"]
    example = QAExample(old.text, "Who?", "zero.", document_id="0", run_id="r0")
    assert store.add_examples([example]) == 1
    before = store.content_hash()
    with pytest.raises(StoreError, match="does not annotate"):
        store.add_examples([replace(example, document_id="1")])
    assert store.content_hash() == before
