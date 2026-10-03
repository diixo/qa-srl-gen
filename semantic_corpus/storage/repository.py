"""An append-only corpus store on the filesystem.

A store is a directory of JSONL files plus a manifest. Writes only ever
append; nothing is rewritten in place. That is not a stylistic choice — the
handoff requires annotations to be append-only so that a later pass can
disagree with an earlier one and both opinions survive, and the same
property is what makes an export reproducible: the bytes that produced a
dataset version are still on disk afterwards.

Four guards are built in, each covering a failure that is silent otherwise:

**Ontology versions cannot mix.** The manifest records which ontology the
store was opened under; a run labelled under another one is refused. Two
label inventories merged by accident produce a corpus whose labels mean
different things in different rows, and nothing downstream can detect it.

**Documents are deduplicated by content.** The same text under two
identifiers is one document. If copies land in different splits, every
evaluation on that corpus is contaminated.

**Splits are immutable once assigned.** Re-adding a document with a
different split is refused rather than appended, because the store would
then contain two contradictory answers to "which split is this in".

**Reads stream.** Documents and annotations are decoded one document/run at
a time. Identity, deduplication and byte-offset indexes remain in memory and
grow with the number of records; this is not constant-memory storage.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, Iterable, Iterator, Mapping

from ..documents import AnnotationRun, Document, ONTOLOGY_VERSION, sha256_of
from .schema import (
    STORE_FORMAT,
    TABLES,
    OntologyVersionMismatch,
    decode_document,
    decode_passage,
    decode_qa_example,
    decode_run,
    decode_utterance,
    encode_dataset_version,
    encode_document,
    encode_passage,
    encode_qa_example,
    encode_run_rows,
    encode_utterance,
)

__all__ = ["CorpusStore", "StoreError", "DuplicateDocument", "SplitConflict"]

#: Bumped when the generator's output changes in a way that matters for
#: reproducibility. Recorded on every dataset version.
GENERATOR_VERSION = "0.2.0"

_MANIFEST = "manifest.json"
_RECENT_WRITES = 128


class StoreError(RuntimeError):
    """Raised when a write would break one of the store's invariants."""


class DuplicateDocument(StoreError):
    """Raised when the same text is added twice under different identifiers."""


class SplitConflict(StoreError):
    """Raised when a document would change split."""


@dataclass(frozen=True, slots=True)
class _Index:
    """What the store needs to know before accepting a write."""

    hashes: dict[str, str]
    splits: dict[str, str | None]
    run_ids: set[str]
    example_keys: set[str]
    document_hashes: dict[str, str]


class CorpusStore:
    """Append-only JSONL store for documents, annotations and QA examples."""

    def __init__(
        self,
        root: Path | str,
        *,
        ontology_version: str = ONTOLOGY_VERSION,
        create: bool = True,
    ) -> None:
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.is_dir():
            raise StoreError(f"no store at {self.root}")
        if not create and not self.manifest_path.exists():
            raise StoreError(f"no manifest at {self.root}")
        self.ontology_version = self._open_manifest(ontology_version)
        # Only offsets are retained; text and annotations are decoded on demand.
        self._offsets: dict[tuple[str, str], dict[Any, list[int]]] = {}
        self._runs_by_document: dict[str, list[str]] | None = None
        self._index = self._build_index()
        # Avoid opening just-written files again during validate/store loops.
        # Fingerprints, not mutable Document objects, retain the accepted identity.
        self._recent_document_fingerprints: OrderedDict[str, str] = OrderedDict()
        # Which documents each run covers, for every run written or looked up
        # this session; ``None`` marks a legacy header without document_ids.
        # Unbounded on purpose: a build writes every run before adding any
        # QA, so a window of recent runs would miss on nearly every lookup
        # and reopen the runs file once per run. One small frozenset per run
        # is the same order of memory as the run-ID index already kept.
        self._run_documents: dict[str, frozenset[str] | None] = {}

    # -- manifest ----------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.root / _MANIFEST

    def _open_manifest(self, ontology_version: str) -> str:
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            stored = manifest.get("ontology_version")
            if stored != ontology_version:
                raise OntologyVersionMismatch(
                    f"store at {self.root} holds ontology {stored!r}, "
                    f"but was opened as {ontology_version!r}"
                )
            if manifest.get("store_format") != STORE_FORMAT:
                raise StoreError(
                    f"store format {manifest.get('store_format')!r} is not "
                    f"{STORE_FORMAT!r}"
                )
            return stored
        self.manifest_path.write_text(
            json.dumps(
                {
                    "store_format": STORE_FORMAT,
                    "ontology_version": ontology_version,
                    "created_at": _now(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return ontology_version

    # -- low-level io ------------------------------------------------------

    def path_for(self, table: str) -> Path:
        try:
            return self.root / TABLES[table]
        except KeyError:
            raise StoreError(f"unknown table {table!r}") from None

    def _append(self, table: str, rows: Iterable[Mapping[str, Any]]) -> int:
        return self._append_encoded(table, (_encode_row(row) for row in rows))

    def _append_encoded(self, table: str, lines: Iterable[bytes]) -> int:
        """Append lines already encoded by :func:`_encode_row`.

        A caller that has validated a batch by serializing it hands over the
        bytes as they are, rather than having them parsed and serialized a
        second time. A line is decoded only when an offset index over this
        table is live and needs the row's key.
        """
        indexes = [(key, index) for (indexed_table, key), index in self._offsets.items()
                   if indexed_table == table]
        path = self.path_for(table)
        count = 0
        with path.open("ab") as stream:
            for data in lines:
                offset = stream.tell()
                stream.write(data)
                if indexes:
                    row = json.loads(data)
                    for key, index in indexes:
                        index.setdefault(row.get(key), []).append(offset)
                count += 1
        return count

    def _offset_index(self, table: str, key: str) -> dict[Any, list[int]]:
        index_key = (table, key)
        if index_key not in self._offsets:
            path = self.path_for(table)
            index: dict[Any, list[int]] = {}
            if path.exists():
                with path.open("rb") as stream:
                    while True:
                        offset = stream.tell()
                        line = stream.readline()
                        if not line:
                            break
                        if line.strip():
                            row = json.loads(line)
                            index.setdefault(row.get(key), []).append(offset)
            self._offsets[index_key] = index
        return self._offsets[index_key]

    def _rows_for(self, table: str, key: str, value: Any) -> Iterator[dict[str, Any]]:
        offsets = self._offset_index(table, key).get(value, ())
        if offsets:
            with self.path_for(table).open("rb") as stream:
                for offset in offsets:
                    stream.seek(offset)
                    yield json.loads(stream.readline())

    def _rows_for_many(
        self, table: str, key: str, values: Iterable[Any],
    ) -> Iterator[dict[str, Any]]:
        """Read selected rows through one handle, in file order.

        Only the requested offsets are collected. Distinct rows sharing the
        same key are retained, so callers can still detect ambiguous IDs.
        """
        values = set(values)
        if not values:
            return
        index = self._offset_index(table, key)
        offsets = sorted(offset for value in values for offset in index.get(value, ()))
        if offsets:
            with self.path_for(table).open("rb") as stream:
                for offset in offsets:
                    stream.seek(offset)
                    yield json.loads(stream.readline())

    def read(self, table: str) -> Iterator[dict[str, Any]]:
        """Stream the rows of one table."""
        path = self.path_for(table)
        if not path.exists():
            return
        with path.open("rt", encoding="utf-8") as stream:
            for number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise StoreError(f"{path}:{number}: {error}") from error

    def count(self, table: str) -> int:
        return sum(1 for _ in self.read(table))

    def _build_index(self) -> _Index:
        hashes: dict[str, str] = {}
        splits: dict[str, str | None] = {}
        document_hashes: dict[str, str] = {}
        for row in self.read("documents"):
            hashes[row["source_hash"]] = row["document_id"]
            splits[row["document_id"]] = row.get("document_split")
            document_hashes[row["document_id"]] = row["source_hash"]
        run_ids = {row["run_id"] for row in self.read("annotation_runs")}
        example_keys = {
            _example_key(row) for row in self.read("qa_examples")
        }
        return _Index(hashes, splits, run_ids, example_keys, document_hashes)

    # -- writing -----------------------------------------------------------

    def add_document(self, document: Document, *, skip_duplicates: bool = True) -> bool:
        """Append a document with its passages and utterances.

        Returns whether anything was written. A document whose text is
        already present is skipped — or refused, if *skip_duplicates* is off
        and the identifier differs, since that is almost always a mistake.
        """
        digest = document.sha256
        previous_split = self._index.splits.get(document.document_id, _MISSING)
        if previous_split is not _MISSING:
            if previous_split != document.split:
                raise SplitConflict(f"{document.document_id!r} cannot change split")
            if _document_identity(self.document(document.document_id)) != _document_identity(document):
                raise StoreError(f"document {document.document_id!r} is immutable")
        existing = self._index.hashes.get(digest)
        if existing is not None:
            if self._index.splits[existing] != document.split:
                raise SplitConflict(f"duplicate text {existing!r} cannot cross splits")
            if not skip_duplicates and existing != document.document_id:
                raise DuplicateDocument(
                    f"{document.document_id!r} has the same text as {existing!r}"
                )
            return False

        fingerprint = sha256_of(_document_identity(document))
        self._append("documents", [encode_document(document)])
        self._append("passages", [encode_passage(p) for p in document.passages])
        self._append("utterances", [encode_utterance(u) for u in document.utterances])
        self._index.hashes[digest] = document.document_id
        self._index.splits[document.document_id] = document.split
        self._index.document_hashes[document.document_id] = digest
        _remember_written(self._recent_document_fingerprints, document.document_id, fingerprint)
        return True

    def add_run(self, run: AnnotationRun, document: Document | None = None) -> None:
        """Append an annotation run.

        Referenced documents must already exist. When *document* is omitted,
        validation loads each referenced document from this store.
        """
        if run.ontology_version != self.ontology_version:
            raise OntologyVersionMismatch(
                f"run {run.run_id!r} is labelled under ontology "
                f"{run.ontology_version!r}, store holds {self.ontology_version!r}"
            )
        if run.run_id in self._index.run_ids:
            raise StoreError(
                f"run {run.run_id!r} is already stored; annotations are "
                "append-only, so re-annotating needs a new run id"
            )
        document_ids = {item.document_id for item in
                        (*run.mentions, *run.predicates, *run.properties)}
        if len(run.ids()) != len(run.mentions) + len(run.predicates) + len(run.properties):
            raise StoreError("run contains duplicate annotation IDs")
        provided_stored = None
        if document is not None:
            document_ids.add(document.document_id)
            identity = _document_identity(document)
            fingerprint = self._recent_document_fingerprints.get(document.document_id)
            if fingerprint is not None:
                unchanged = fingerprint == sha256_of(identity)
                provided_stored = document
            else:
                provided_stored = self.document(document.document_id)
                unchanged = _document_identity(provided_stored) == identity
            if not unchanged:
                raise StoreError("run document must already be stored unchanged")
        # The validated document already identifies its own utterances. Use
        # the global offset index to detect collisions even on this fast path.
        known_utterances = ({u.utterance_id for u in provided_stored.utterances}
                            if provided_stored is not None else set())
        owner_reads: set[str] = set()
        if run.dialogue:
            utterance_index = self._offset_index("utterances", "utterance_id")
            for annotation in run.dialogue:
                uid = annotation.utterance_id
                if len(utterance_index.get(uid, ())) != 1:
                    raise StoreError(f"unknown or ambiguous utterance {uid!r}")
                if uid not in known_utterances:
                    owner_reads.add(uid)
        for row in self._rows_for_many(
            "utterances", "utterance_id", owner_reads,
        ):
            document_ids.add(row["document_id"])
        for doc_id in document_ids:
            stored = (provided_stored if document is not None and doc_id == document.document_id
                      else self.document(doc_id))
            if stored is None:
                raise StoreError(f"unknown document {doc_id!r}")
            utterance_ids = {u.utterance_id for u in stored.utterances}
            subset = replace(run,
                mentions=tuple(m for m in run.mentions if m.document_id == doc_id),
                predicates=tuple(p for p in run.predicates if p.document_id == doc_id),
                properties=tuple(p for p in run.properties if p.document_id == doc_id),
                dialogue=tuple(d for d in run.dialogue if d.utterance_id in utterance_ids),
            )
            ids = {m.mention_id for m in subset.mentions} | {p.predicate_id for p in subset.predicates} | {p.property_id for p in subset.properties}
            subset = replace(subset, relations=tuple(e for e in run.relations
                             if e.source_id in ids or e.target_id in ids))
            problems = subset.validate_against(stored)
            if problems:
                raise StoreError(f"run {run.run_id!r} does not describe {doc_id!r}: " + "; ".join(problems[:5]))
        all_ids = {m.mention_id for m in run.mentions} | {p.predicate_id for p in run.predicates} | {p.property_id for p in run.properties}
        if any(e.source_id not in all_ids or e.target_id not in all_ids for e in run.relations):
            raise StoreError("run contains dangling relations")
        if document is not None:
            problems = run.validate_against(document)
            if problems:
                raise StoreError(
                    f"run {run.run_id!r} does not describe "
                    f"{document.document_id!r}: " + "; ".join(problems[:5])
                )
        rows = encode_run_rows(run)
        rows["annotation_runs"][0]["document_ids"] = sorted(document_ids)
        for table, table_rows in rows.items():
            self._append(table, table_rows)
        self._index.run_ids.add(run.run_id)
        self._run_documents[run.run_id] = frozenset(document_ids)
        if self._runs_by_document is not None:
            for doc_id in sorted(document_ids):
                self._runs_by_document.setdefault(doc_id, []).append(run.run_id)

    def add_examples(
        self,
        examples: Iterable[Any],
        *,
        generation_run_id: str | None = None,
        deduplicate: bool = True,
    ) -> int:
        """Append QA examples, dropping ones already present.

        Deduplication here is the second half of what the handoff asks for:
        documents are deduplicated before annotation, examples after
        generation, because two templates can word the same question
        identically from different records.

        Validate and serialize the whole batch before appending anything.
        Staged JSONL spills to disk above 1 MiB; only deduplication hashes
        grow with the batch, not example objects or their contexts. Input,
        validation and serialization errors leave the store unchanged.
        As with other writes, this is not a transaction against I/O failures
        or process crashes during the final append.
        """
        def run_documents(rid: str) -> frozenset[str] | None:
            if rid in self._run_documents:
                return self._run_documents[rid]
            header = next(self._rows_for("annotation_runs", "run_id", rid), None)
            if header is None:
                raise StoreError(f"unknown run {rid!r}")
            # Legacy headers (None) need the document membership index.
            documents = frozenset(header["document_ids"]) if "document_ids" in header else None
            self._run_documents[rid] = documents
            return documents

        batch_keys: set[str] = set()
        written = 0
        with SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b") as staged:
            for example in examples:
                if example.document_id not in self._index.document_hashes:
                    raise StoreError(f"unknown document {example.document_id!r}")
                for rid in (example.run_id, generation_run_id):
                    if rid is not None:
                        document_ids = run_documents(rid)
                        if document_ids is not None:
                            belongs = example.document_id in document_ids
                        else:
                            if self._runs_by_document is None:
                                self._index_run_documents()
                            belongs = rid in self._runs_by_document.get(example.document_id, ())
                        if not belongs:
                            raise StoreError(f"run {rid!r} does not annotate {example.document_id!r}")
                problems = example.check()
                if problems:
                    raise StoreError("invalid example: " + "; ".join(problems))
                row = encode_qa_example(example, generation_run_id=generation_run_id)
                key = _example_key(row)
                if deduplicate and (key in self._index.example_keys or key in batch_keys):
                    continue
                try:
                    data = _encode_row(row)
                except (TypeError, ValueError, UnicodeError) as error:
                    raise StoreError(f"invalid example: cannot serialize JSON: {error}") from error
                staged.write(data)
                batch_keys.add(key)
            if batch_keys:
                staged.seek(0)
                # The staged bytes are what gets written: no second encode.
                written = self._append_encoded("qa_examples", staged)
                self._index.example_keys.update(batch_keys)
        return written

    def publish_version(
        self,
        version_id: str,
        *,
        notes: str = "",
        generator_version: str = GENERATOR_VERSION,
        build_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record what the store currently contains, under a name.

        The content hash covers every table, so two versions with the same
        hash are the same corpus and a changed hash proves something moved.
        """
        digest = self.content_hash()
        for previous in self.read("dataset_versions"):
            if previous["version_id"] == version_id:
                same_config = build_config is None or previous.get("build_config") == dict(build_config)
                if (previous["content_hash"] == digest
                        and previous["generator_version"] == generator_version and same_config):
                    return previous
                raise StoreError(f"version {version_id!r} already names different content")
        counts = {table: self.count(table) for table in TABLES}
        manifest = encode_dataset_version(
            version_id=version_id,
            created_at=_now(),
            ontology_version=self.ontology_version,
            generator_version=generator_version,
            run_ids=sorted(self._index.run_ids),
            counts=counts,
            content_hash=digest,
            notes=notes,
        )
        if build_config is not None:
            manifest["build_config"] = dict(build_config)
        self._append("dataset_versions", [manifest])
        return manifest

    def content_hash(self) -> str:
        """A hash over every table except the version log itself."""
        digest = hashlib.sha256()
        for table in sorted(TABLES):
            if table == "dataset_versions":
                continue
            path = self.path_for(table)
            digest.update(table.encode("utf-8"))
            if path.exists():
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
        return digest.hexdigest()

    # -- reading -----------------------------------------------------------

    def documents(self, *, split: str | None = None) -> Iterator[Document]:
        """Stream documents, with their passages and utterances attached."""
        for row in self.read("documents"):
            if split is not None and row.get("document_split") != split:
                continue
            yield self._document_from_row(row)

    def _document_from_row(self, row: Mapping[str, Any]) -> Document:
        doc = decode_document(row)
        return replace(doc,
            passages=tuple(decode_passage(p) for p in self._rows_for("passages", "document_id", doc.document_id)),
            utterances=tuple(decode_utterance(u) for u in self._rows_for("utterances", "document_id", doc.document_id)),
        )

    def document(self, document_id: str) -> Document | None:
        row = next(self._rows_for("documents", "document_id", document_id), None)
        return self._document_from_row(row) if row is not None else None

    def runs(self, *, document_id: str | None = None) -> Iterator[AnnotationRun]:
        """Stream annotation runs, reassembled from their rows."""
        headers = self.read("annotation_runs")
        if document_id is not None:
            if self._runs_by_document is None:
                self._index_run_documents()
            headers = (row for rid in self._runs_by_document.get(document_id, ())
                       for row in self._rows_for("annotation_runs", "run_id", rid))
        for header in headers:
            run_id = header["run_id"]
            if document_id is not None and "document_ids" in header and document_id not in header["document_ids"]:
                continue
            spans = list(self._rows_for("spans", "run_id", run_id))
            dialogue = list(self._rows_for("dialogue", "run_id", run_id))
            relations = list(self._rows_for("relations", "run_id", run_id))
            if document_id is not None:
                spans = [s for s in spans if s["document_id"] == document_id]
                doc = self.document(document_id)
                utterance_ids = {u.utterance_id for u in doc.utterances} if doc else set()
                dialogue = [d for d in dialogue if d["utterance_id"] in utterance_ids]
                if not spans and not dialogue and document_id not in header.get("document_ids", ()):
                    continue
                ids = {s["id"] for s in spans}
                relations = [r for r in relations if r["source_id"] in ids and r["target_id"] in ids]
            yield decode_run(
                header, spans, relations, dialogue,
            )

    def _index_run_documents(self) -> None:
        membership: dict[str, set[str]] = {}
        legacy: set[str] = set()
        order: dict[str, int] = {}
        for index, header in enumerate(self.read("annotation_runs")):
            rid = header["run_id"]
            order[rid] = index
            if "document_ids" not in header:
                legacy.add(rid)
            for doc_id in header.get("document_ids", ()):
                membership.setdefault(doc_id, set()).add(rid)
        if legacy:
            for row in self.read("spans"):
                if row["run_id"] in legacy:
                    membership.setdefault(row["document_id"], set()).add(row["run_id"])
            owners = {u["utterance_id"]: u["document_id"] for u in self.read("utterances")}
            for row in self.read("dialogue"):
                doc_id = owners.get(row["utterance_id"])
                if row["run_id"] in legacy and doc_id is not None:
                    membership.setdefault(doc_id, set()).add(row["run_id"])
        self._runs_by_document = {doc_id: sorted(ids, key=order.__getitem__)
                                  for doc_id, ids in membership.items()}

    def examples(self, *, split: str | None = None) -> Iterator[Any]:
        """Stream QA examples, optionally restricted to one split."""
        allowed = None
        if split is not None:
            allowed = {
                document_id
                for document_id, value in self._index.splits.items()
                if value == split
            }
        for row in self.read("qa_examples"):
            if allowed is not None and row.get("document_id") not in allowed:
                continue
            yield decode_qa_example(row)

    def versions(self) -> list[dict[str, Any]]:
        return list(self.read("dataset_versions"))

    def split_of(self, document_id: str) -> str | None:
        return self._index.splits.get(document_id)

    @property
    def splits(self) -> Mapping[str, str | None]:
        return dict(self._index.splits)


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


_MISSING = _Missing()


def _encode_row(row: Mapping[str, Any]) -> bytes:
    """The one serialization of a stored row, so every writer agrees byte for byte."""
    return (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _remember_written(cache: OrderedDict, key: str, value: Any) -> None:
    """Keep only a bounded window of successfully appended records."""
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > _RECENT_WRITES:
        cache.popitem(last=False)


def _example_key(row: Mapping[str, Any]) -> str:
    """Identity of a QA example: what it shows, asks and answers."""
    return sha256_of(
        "\x00".join(
            (
                str(row.get("context", "")),
                str(row.get("question", "")),
                str(row.get("answer", "")),
            )
        )
    )


def _document_identity(document: Document | None) -> str | None:
    """Compare persisted content, including JSON-normalized metadata."""
    if document is None:
        return None
    return json.dumps({
        "document": encode_document(document),
        "passages": [encode_passage(p) for p in document.passages],
        "utterances": [encode_utterance(u) for u in document.utterances],
    }, ensure_ascii=False, sort_keys=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
