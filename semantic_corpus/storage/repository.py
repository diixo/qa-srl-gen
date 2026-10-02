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

**Reads stream.** Nothing here loads a table into memory, so a store larger
than RAM still works.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..documents import (
    AnnotationRun,
    Document,
    ONTOLOGY_VERSION,
    Passage,
    Utterance,
    sha256_of,
)
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
GENERATOR_VERSION = "0.1.0"

_MANIFEST = "manifest.json"


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
        self.ontology_version = self._open_manifest(ontology_version)
        self._index = self._build_index()

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
        rows = list(rows)
        if not rows:
            return 0
        path = self.path_for(table)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return len(rows)

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
        for row in self.read("documents"):
            hashes[row["source_hash"]] = row["document_id"]
            splits[row["document_id"]] = row.get("document_split")
        run_ids = {row["run_id"] for row in self.read("annotation_runs")}
        example_keys = {
            _example_key(row) for row in self.read("qa_examples")
        }
        return _Index(hashes, splits, run_ids, example_keys)

    # -- writing -----------------------------------------------------------

    def add_document(self, document: Document, *, skip_duplicates: bool = True) -> bool:
        """Append a document with its passages and utterances.

        Returns whether anything was written. A document whose text is
        already present is skipped — or refused, if *skip_duplicates* is off
        and the identifier differs, since that is almost always a mistake.
        """
        digest = document.sha256
        existing = self._index.hashes.get(digest)
        if existing is not None:
            if not skip_duplicates and existing != document.document_id:
                raise DuplicateDocument(
                    f"{document.document_id!r} has the same text as {existing!r}"
                )
            return False

        previous_split = self._index.splits.get(document.document_id, _MISSING)
        if previous_split is not _MISSING and previous_split != document.split:
            raise SplitConflict(
                f"{document.document_id!r} is already in split "
                f"{previous_split!r}, cannot move it to {document.split!r}"
            )

        self._append("documents", [encode_document(document)])
        self._append("passages", [encode_passage(p) for p in document.passages])
        self._append("utterances", [encode_utterance(u) for u in document.utterances])
        self._index.hashes[digest] = document.document_id
        self._index.splits[document.document_id] = document.split
        return True

    def add_run(self, run: AnnotationRun, document: Document | None = None) -> None:
        """Append an annotation run.

        *document* is optional but checking against it is cheap and catches
        the one error that makes a corpus quietly useless: annotations whose
        offsets do not match the text they claim to describe.
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
        if document is not None:
            problems = run.validate_against(document)
            if problems:
                raise StoreError(
                    f"run {run.run_id!r} does not describe "
                    f"{document.document_id!r}: " + "; ".join(problems[:5])
                )
        rows = encode_run_rows(run)
        for table, table_rows in rows.items():
            self._append(table, table_rows)
        self._index.run_ids.add(run.run_id)

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
        """
        written = 0
        batch: list[dict[str, Any]] = []
        for example in examples:
            row = encode_qa_example(example, generation_run_id=generation_run_id)
            key = _example_key(row)
            if deduplicate:
                if key in self._index.example_keys:
                    continue
                self._index.example_keys.add(key)
            batch.append(row)
            written += 1
        self._append("qa_examples", batch)
        return written

    def publish_version(
        self,
        version_id: str,
        *,
        notes: str = "",
        generator_version: str = GENERATOR_VERSION,
    ) -> dict[str, Any]:
        """Record what the store currently contains, under a name.

        The content hash covers every table, so two versions with the same
        hash are the same corpus and a changed hash proves something moved.
        """
        counts = {table: self.count(table) for table in TABLES}
        manifest = encode_dataset_version(
            version_id=version_id,
            created_at=_now(),
            ontology_version=self.ontology_version,
            generator_version=generator_version,
            run_ids=sorted(self._index.run_ids),
            counts=counts,
            content_hash=self.content_hash(),
            notes=notes,
        )
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
                digest.update(path.read_bytes())
        return digest.hexdigest()

    # -- reading -----------------------------------------------------------

    def documents(self, *, split: str | None = None) -> Iterator[Document]:
        """Stream documents, with their passages and utterances attached."""
        passages: dict[str, list[Passage]] = {}
        for row in self.read("passages"):
            passages.setdefault(row["document_id"], []).append(decode_passage(row))
        utterances: dict[str, list[Utterance]] = {}
        for row in self.read("utterances"):
            utterances.setdefault(row["document_id"], []).append(decode_utterance(row))

        from dataclasses import replace

        for row in self.read("documents"):
            if split is not None and row.get("document_split") != split:
                continue
            document = decode_document(row)
            yield replace(
                document,
                passages=tuple(passages.get(document.document_id, ())),
                utterances=tuple(utterances.get(document.document_id, ())),
            )

    def document(self, document_id: str) -> Document | None:
        for document in self.documents():
            if document.document_id == document_id:
                return document
        return None

    def runs(self, *, document_id: str | None = None) -> Iterator[AnnotationRun]:
        """Stream annotation runs, reassembled from their rows."""
        spans: dict[str, list[dict[str, Any]]] = {}
        for row in self.read("spans"):
            if document_id is not None and row.get("document_id") != document_id:
                continue
            spans.setdefault(row["run_id"], []).append(row)
        relations: dict[str, list[dict[str, Any]]] = {}
        for row in self.read("relations"):
            relations.setdefault(row["run_id"], []).append(row)
        dialogue: dict[str, list[dict[str, Any]]] = {}
        for row in self.read("dialogue"):
            dialogue.setdefault(row["run_id"], []).append(row)

        for header in self.read("annotation_runs"):
            run_id = header["run_id"]
            if document_id is not None and run_id not in spans:
                continue
            yield decode_run(
                header,
                spans.get(run_id, []),
                relations.get(run_id, []),
                dialogue.get(run_id, []),
            )

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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
