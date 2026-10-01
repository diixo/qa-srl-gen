"""Streaming readers for the QA-SRL Bank.

Nothing here loads a whole file into memory: every entry point is a generator
over one sentence at a time, and ``.jsonl.gz`` is decompressed on the fly, so
``orig/train.jsonl.gz`` can be scanned without being unpacked first.

Two formats are supported:

``read_bank``
    QA-SRL Bank 2.0 / 2.1 — one JSON object per line, gzip-compressed or not.
    Unmodelled fields (2.1 adds nominal predicates) survive in
    :attr:`~.models.Sentence.extra` rather than being dropped.

``read_qa_text``
    The older tab-separated ``*.qa`` release. Its question lines carry
    *surface* verb strings instead of form placeholders, so they are kept as
    raw strings and not forced into :class:`~.models.QuestionSlots`.
"""

from __future__ import annotations

import gzip
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .models import Sentence, Span

__all__ = [
    "read_jsonl",
    "read_bank",
    "iter_question_labels",
    "QaTextQuestion",
    "QaTextPredicate",
    "QaTextSentence",
    "read_qa_text",
]


def _open_text(path: Path) -> io.TextIOBase:
    """Open a plain or gzipped text file for streaming UTF-8 reads."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield each JSON object of a ``.jsonl`` / ``.jsonl.gz`` file."""
    path = Path(path)
    with _open_text(path) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error


def read_bank(path: Path | str) -> Iterator[Sentence]:
    """Stream a QA-SRL Bank file as :class:`~.models.Sentence` objects."""
    for record in read_jsonl(path):
        yield Sentence.from_json(record)


def iter_question_labels(paths: Iterable[Path | str]) -> Iterator[tuple[Sentence, Any, Any]]:
    """Stream ``(sentence, verb_entry, question_label)`` across several files."""
    for path in paths:
        for sentence in read_bank(path):
            for entry, label in sentence.question_labels():
                yield sentence, entry, label


# --------------------------------------------------------------------------
# Legacy tab-separated *.qa format
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QaTextQuestion:
    """One question line of a ``*.qa`` file.

    ``slots`` holds the seven raw fields in template order. The fourth of them
    is a surface verb phrase (``be oxygenated``), not a form placeholder, so
    this is intentionally not a :class:`~.models.QuestionSlots`.
    """

    slots: tuple[str, str, str, str, str, str, str]
    answers: tuple[str, ...]

    @property
    def text(self) -> str:
        words = [w for slot in self.slots if slot != "_" for w in slot.split()]
        sentence = " ".join(words)
        return sentence[:1].upper() + sentence[1:] + "?"


@dataclass(frozen=True, slots=True)
class QaTextPredicate:
    index: int
    verb: str
    questions: tuple[QaTextQuestion, ...]


@dataclass(frozen=True, slots=True)
class QaTextSentence:
    sentence_id: str
    tokens: tuple[str, ...]
    predicates: tuple[QaTextPredicate, ...]

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


def read_qa_text(path: Path | str) -> Iterator[QaTextSentence]:
    """Stream the legacy ``*.qa`` release (``wiki1.train.qa`` and friends).

    Layout per sentence::

        SENT_ID <tab> n_predicates
        the sentence text
        token_index <tab> verb <tab> n_questions
        wh <tab> aux <tab> subj <tab> verb <tab> obj <tab> prep <tab> obj2 <tab> ? <tab> answers
        <blank line>

    Answers on one line are alternatives separated by ``###``.
    """
    path = Path(path)
    with _open_text(path) as stream:
        lines = (line.rstrip("\n") for line in stream)
        yield from _parse_qa_text(lines, str(path))


def _parse_qa_text(lines: Iterator[str], origin: str) -> Iterator[QaTextSentence]:
    pending = None
    for raw in lines:
        if not raw.strip():
            continue
        fields = raw.split("\t")
        if len(fields) == 2 and fields[1].isdigit():
            if pending is not None:
                yield _finish_qa_sentence(pending)
            pending = {"id": fields[0], "tokens": None, "predicates": []}
            continue
        if pending is None:
            raise ValueError(f"{origin}: data before the first sentence header: {raw!r}")
        if pending["tokens"] is None:
            pending["tokens"] = tuple(raw.split(" "))
            continue
        if len(fields) == 3 and fields[0].isdigit() and fields[2].isdigit():
            pending["predicates"].append(
                {"index": int(fields[0]), "verb": fields[1], "questions": []}
            )
            continue
        if len(fields) >= 9 and fields[7] == "?":
            if not pending["predicates"]:
                raise ValueError(f"{origin}: question line before any predicate: {raw!r}")
            answers = tuple(
                answer.strip()
                for field in fields[8:]
                for answer in field.split("###")
                if answer.strip()
            )
            pending["predicates"][-1]["questions"].append(
                QaTextQuestion(slots=tuple(fields[:7]), answers=answers)  # type: ignore[arg-type]
            )
            continue
        raise ValueError(f"{origin}: unrecognised line: {raw!r}")
    if pending is not None:
        yield _finish_qa_sentence(pending)


def _finish_qa_sentence(pending: dict[str, Any]) -> QaTextSentence:
    if pending["tokens"] is None:
        raise ValueError(f"sentence {pending['id']} has no text line")
    return QaTextSentence(
        sentence_id=pending["id"],
        tokens=pending["tokens"],
        predicates=tuple(
            QaTextPredicate(
                index=p["index"], verb=p["verb"], questions=tuple(p["questions"])
            )
            for p in pending["predicates"]
        ),
    )


def span_text(span: Span, sentence: Sentence) -> str:
    """Convenience: the surface text of *span* inside *sentence*."""
    return span.text(sentence.sentence_tokens)
