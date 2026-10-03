"""QA-SRL Bank metadata codecs, ported from julianmichael/qasrl (MIT).

Based on qasrl-bank/src/qasrl/bank at 16ab4949; see THIRD_PARTY_NOTICES.md.
These metadata objects do not load the sentence corpus. ``read_index`` reads
one (small) JSON index, either plain or gzip-compressed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
import gzip
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .dataset import ConsolidatedSentence

__all__ = ["Domain", "DatasetPartition", "DocumentId", "SentenceId", "DocumentMetadata", "Document", "DataIndex", "read_index"]


class Domain(str, Enum):
    WIKIPEDIA = "wikipedia"
    WIKINEWS = "wikinews"
    TQA = "tqa"

    @classmethod
    def from_string(cls, value: str) -> "Domain":
        return cls(value.lower())

    def __str__(self) -> str:
        return "TQA" if self is Domain.TQA else self.value

    @property
    def sort_key(self) -> int:
        return (Domain.WIKIPEDIA, Domain.WIKINEWS, Domain.TQA).index(self)


class DatasetPartition(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"

    @classmethod
    def from_string(cls, value: str) -> "DatasetPartition":
        return cls(value)

    def __str__(self) -> str:
        return self.value


@total_ordering
@dataclass(frozen=True)
class DocumentId:
    domain: Domain
    id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", Domain.from_string(self.domain))
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Document ID must be a nonempty string")

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, DocumentId):
            return NotImplemented
        return (self.domain.sort_key, self.id) < (other.domain.sort_key, other.id)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "DocumentId":
        return cls(Domain.from_string(data["domain"]), data["id"])

    def to_json(self) -> dict[str, str]:
        return {"domain": self.domain.value, "id": self.id}


@total_ordering
@dataclass(frozen=True)
class SentenceId:
    document_id: DocumentId
    paragraph_num: int
    sentence_num: int

    def __post_init__(self) -> None:
        for number in (self.paragraph_num, self.sentence_num):
            if not isinstance(number, int) or isinstance(number, bool) or number < 0:
                raise ValueError("Sentence and paragraph numbers must be nonnegative integers")
        if self.domain is Domain.TQA and self.paragraph_num != 0:
            raise ValueError("TQA sentence IDs have paragraph number zero")
        if ":" in self.document_id.id:
            raise ValueError("Sentence document IDs cannot contain ':'")

    @property
    def domain(self) -> Domain:
        return self.document_id.domain

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SentenceId):
            return NotImplemented
        return (self.document_id, self.paragraph_num, self.sentence_num) < (other.document_id, other.paragraph_num, other.sentence_num)

    def __str__(self) -> str:
        if self.domain is Domain.TQA:
            return f"TQA:{self.document_id.id}_{self.sentence_num}"
        return f"Wiki1k:{self.domain}:{self.document_id.id}:{self.paragraph_num}:{self.sentence_num}"

    @classmethod
    def from_string(cls, value: str) -> "SentenceId":
        tqa = re.fullmatch(r"TQA:([^:]+)_([0-9]+)", value)
        if tqa:
            return cls(DocumentId(Domain.TQA, tqa[1]), 0, int(tqa[2]))
        wiki = re.fullmatch(r"Wiki1k:(wikipedia|wikinews):([^:]+):([0-9]+):([0-9]+)", value)
        if wiki:
            return cls(DocumentId(Domain.from_string(wiki[1]), wiki[2]), int(wiki[3]), int(wiki[4]))
        raise ValueError(f"Invalid QA-SRL Bank sentence ID: {value!r}")

    @classmethod
    def from_json(cls, data: str) -> "SentenceId":
        return cls.from_string(data)

    def to_json(self) -> str:
        return str(self)


@total_ordering
@dataclass(frozen=True)
class DocumentMetadata:
    id: DocumentId
    part: DatasetPartition
    title: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "part", DatasetPartition(self.part))

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, DocumentMetadata):
            return NotImplemented
        return (self.title, self.id) < (other.title, other.id)

    @property
    def id_string(self) -> str:
        prefix = "TQA" if self.id.domain is Domain.TQA else f"Wiki1k:{self.id.domain}"
        return f"{prefix}:{self.id.id}"

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "DocumentMetadata":
        return cls(DocumentId.from_json(data), DatasetPartition(data["part"]), data["title"])

    def to_json(self) -> dict[str, str]:
        return {"part": self.part.value, "idString": self.id_string, **self.id.to_json(), "title": self.title}


@dataclass(frozen=True)
class Document:
    metadata: DocumentMetadata
    sentences: tuple[ConsolidatedSentence, ...] = ()

    def __post_init__(self) -> None:
        indexed = {}
        for sentence in self.sentences:
            sid = SentenceId.from_string(sentence.sentence_id)
            if sid.document_id != self.metadata.id:
                raise ValueError(f"Sentence {sid} belongs to a different document")
            if sid in indexed and indexed[sid] != sentence:
                raise ValueError(f"Conflicting duplicate sentence: {sid}")
            indexed[sid] = sentence
        object.__setattr__(self, "sentences", tuple(indexed[sid] for sid in sorted(indexed)))

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Document":
        return cls(DocumentMetadata.from_json(data["metadata"]), tuple(ConsolidatedSentence.from_json(sentence) for sentence in data["sentences"]))

    def to_json(self) -> dict[str, Any]:
        return {"metadata": self.metadata.to_json(), "sentences": [sentence.to_json() for sentence in self.sentences]}


@dataclass(frozen=True)
class DataIndex:
    documents: dict[DatasetPartition, tuple[DocumentMetadata, ...]] = field(default_factory=dict)
    dense_ids: tuple[SentenceId, ...] = ()
    qa_nom_ids: tuple[SentenceId, ...] = ()
    qasrl_gs_ids: tuple[SentenceId, ...] = ()

    def __post_init__(self) -> None:
        documents = {}
        seen: dict[DocumentId, DocumentMetadata] = {}
        for part, metadata in self.documents.items():
            part = DatasetPartition(part)
            for value in metadata:
                if value.part != part:
                    raise ValueError(f"Document {value.id} has inconsistent partition")
                if value.id in seen and seen[value.id] != value:
                    raise ValueError(f"Conflicting document metadata: {value.id}")
                seen[value.id] = value
            documents[part] = tuple(sorted(set(metadata)))
        object.__setattr__(self, "documents", documents)
        for name in ("dense_ids", "qa_nom_ids", "qasrl_gs_ids"):
            object.__setattr__(self, name, tuple(sorted(set(getattr(self, name)))))

    @property
    def all_document_metas(self) -> tuple[DocumentMetadata, ...]:
        return tuple(sorted({meta for values in self.documents.values() for meta in values}))

    @property
    def all_document_ids(self) -> tuple[DocumentId, ...]:
        return tuple(sorted({meta.id for meta in self.all_document_metas}))

    @property
    def num_documents(self) -> int:
        return len(self.all_document_metas)

    def get_part(self, document_id: DocumentId) -> DatasetPartition:
        for part, metadata in self.documents.items():
            if any(value.id == document_id for value in metadata):
                return part
        raise KeyError(document_id)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "DataIndex":
        def sentence_ids(key: str, optional: bool = False) -> tuple[SentenceId, ...]:
            values = data.get(key, ()) if optional else data[key]
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"{key} must be an array of sentence IDs")
            return tuple(SentenceId.from_json(value) for value in values)
        return cls(
            {DatasetPartition(key): tuple(DocumentMetadata.from_json(value) for value in values) for key, values in data["documents"].items()},
            sentence_ids("denseIds"), sentence_ids("qaNomIds", True), sentence_ids("qasrlGsIds", True),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "documents": {part.value: [meta.to_json() for meta in values] for part, values in self.documents.items()},
            "denseIds": [value.to_json() for value in self.dense_ids],
            "qaNomIds": [value.to_json() for value in self.qa_nom_ids],
            "qasrlGsIds": [value.to_json() for value in self.qasrl_gs_ids],
        }


def read_index(path: str | Path) -> DataIndex:
    """Read a complete index JSON object; sentences are never loaded here.

    Missing optional ID arrays default to empty; malformed arrays raise
    ValueError instead of being silently discarded by the upstream decoder.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
        return DataIndex.from_json(data)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise ValueError(f"{path}: invalid QA-SRL Bank index: {error}") from error
