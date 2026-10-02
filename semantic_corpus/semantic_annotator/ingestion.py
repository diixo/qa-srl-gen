"""Reading source texts into documents, without losing anything.

Four rules from the handoff are enforced here rather than left to callers,
because each of them is impossible to repair after the fact.

**Text is stored verbatim.** No normalisation, no whitespace collapsing, no
unicode folding. Every annotation records character offsets into this text; a
cleanup pass applied later would silently invalidate all of them.

**Duplicates are removed before annotation, not after.** Two copies of the
same document annotated separately produce two sets of annotations that
disagree for no reason, and if the copies land in different splits the
evaluation is contaminated.

**Splits are assigned per document.** Splitting sentences would put two
sentences of one paragraph on both sides of the train/test line, and a model
that saw one can often answer about the other.

**Passages overlap and keep global offsets.** The handoff asks for a few
neighbouring sentences of context, so a sentence appears in more than one
window; every passage records where it starts in the document so a local
offset can always be converted back.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from ..documents import (
    Document,
    Passage,
    TextSpan,
    Utterance,
    iter_sentence_spans,
    sha256_of,
)

__all__ = [
    "read_text_file",
    "read_jsonl_documents",
    "read_dialogue_jsonl",
    "deduplicate",
    "assign_split",
    "segment_document",
    "segment_dialogue",
    "ingest",
    "SPLITS",
]

SPLITS: tuple[str, ...] = ("train", "dev", "test")


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_text_file(
    path: Path | str,
    *,
    document_id: str | None = None,
    source: str = "txt",
    license: str = "unknown",
) -> Document:
    """Read a whole ``.txt`` file as one document."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return Document(
        document_id=document_id or path.stem,
        text=text,
        source=source,
        license=license,
    )


def read_jsonl_documents(
    path: Path | str, *, source: str | None = None
) -> Iterator[Document]:
    """Read the recommended document JSONL.

    Each line is ``{"document_id": ..., "text": ..., "source": ...,
    "license": ...}``. Unknown keys are kept in ``metadata`` rather than
    dropped, so a corpus can carry its own provenance fields through.
    """
    path = Path(path)
    with path.open("rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if "text" not in record:
                raise ValueError(f"{path}:{line_number}: record has no 'text' field")
            record.setdefault("document_id", f"{path.stem}-{line_number:06d}")
            document = Document.from_json(record)
            if source is not None:
                document = replace(document, source=source)
            yield document


def read_dialogue_jsonl(
    path: Path | str,
    *,
    speakers: Sequence[str] = ("A", "B"),
    source: str = "dialogue",
    license: str = "unknown",
) -> Iterator[Document]:
    """Read a dialogue corpus where each line is a JSON list of turns.

    This is the DailyDialog shape. Turns are joined with newlines and each one
    becomes an :class:`~..documents.Utterance` with its offsets, so a reaction
    can later be annotated against the turns that precede it. Speakers
    alternate through *speakers*, which is all the information the format
    carries.
    """
    path = Path(path)
    with path.open("rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            turns = json.loads(line)
            if not isinstance(turns, list) or not all(isinstance(t, str) for t in turns):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON list of strings"
                )
            turns = [t.strip() for t in turns if t.strip()]
            if not turns:
                continue
            document_id = f"{path.stem}-{line_number:06d}"
            text_parts: list[str] = []
            utterances: list[Utterance] = []
            cursor = 0
            for index, turn in enumerate(turns):
                if index:
                    cursor += 1  # the newline joining the previous turn
                utterances.append(
                    Utterance(
                        utterance_id=f"{document_id}#u{index}",
                        document_id=document_id,
                        turn_index=index,
                        speaker=speakers[index % len(speakers)],
                        span=TextSpan(cursor, cursor + len(turn)),
                    )
                )
                text_parts.append(turn)
                cursor += len(turn)
            yield Document(
                document_id=document_id,
                text="\n".join(text_parts),
                source=source,
                license=license,
                utterances=tuple(utterances),
            )


# ---------------------------------------------------------------------------
# Deduplication and splits
# ---------------------------------------------------------------------------


def deduplicate(
    documents: Iterable[Document], *, seen: set[str] | None = None
) -> Iterator[Document]:
    """Drop documents whose text has already been seen.

    Matching is on the content hash, not the identifier: the same text under
    two identifiers is still one document, and that is the case that
    contaminates splits.
    """
    seen = seen if seen is not None else set()
    for document in documents:
        digest = document.sha256
        if digest in seen:
            continue
        seen.add(digest)
        yield document


def assign_split(
    document_id: str,
    *,
    dev_share: float = 0.1,
    test_share: float = 0.1,
    salt: str = "",
) -> str:
    """Assign a document to a split by a stable hash of its identifier.

    ``hash()`` is salted per process, so it cannot be used: a split that
    changed between runs would make every held-out claim unverifiable.
    """
    if not 0 <= dev_share + test_share < 1:
        raise ValueError("dev_share + test_share must be in [0, 1)")
    digest = hashlib.sha256(f"{salt}\x00{document_id}".encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    if position < test_share:
        return "test"
    if position < test_share + dev_share:
        return "dev"
    return "train"


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def segment_document(
    document: Document, *, window: int = 3, stride: int = 1
) -> Document:
    """Split *document* into overlapping passages of *window* sentences.

    A window wider than one sentence is what lets an annotator resolve a
    pronoun or a short reply; ``stride`` controls how much context is shared
    between neighbouring passages.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    if not 1 <= stride <= window:
        raise ValueError("stride must be between 1 and window")

    sentences = tuple(iter_sentence_spans(document.text))
    if not sentences:
        return replace(document, passages=())

    passages: list[Passage] = []
    for index in range(0, len(sentences), stride):
        chunk = sentences[index : index + window]
        if not chunk:
            break
        span = TextSpan(chunk[0].start_char, chunk[-1].end_char)
        passages.append(
            Passage(
                passage_id=f"{document.document_id}#p{len(passages)}",
                document_id=document.document_id,
                span=span,
                sentence_spans=chunk,
            )
        )
        if index + window >= len(sentences):
            break
    return replace(document, passages=tuple(passages))


def segment_dialogue(document: Document, *, context_turns: int = 2) -> Document:
    """Build one passage per turn, including the turns before it.

    A reaction is unannotatable in isolation, so each passage reaches back
    *context_turns* turns. The passage still ends at the turn being
    annotated, which keeps the record of what was visible when it was judged.
    """
    if not document.utterances:
        raise ValueError(f"{document.document_id} has no utterances")
    passages: list[Passage] = []
    for index, utterance in enumerate(document.utterances):
        first = document.utterances[max(0, index - context_turns)]
        span = TextSpan(first.span.start_char, utterance.span.end_char)
        passages.append(
            Passage(
                passage_id=f"{document.document_id}#p{index}",
                document_id=document.document_id,
                span=span,
                sentence_spans=(utterance.span,),
            )
        )
    return replace(document, passages=tuple(passages))


# ---------------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------------


def ingest(
    paths: Iterable[Path | str],
    *,
    window: int = 3,
    stride: int = 1,
    context_turns: int = 2,
    dev_share: float = 0.1,
    test_share: float = 0.1,
    salt: str = "",
    dialogue_suffixes: Sequence[str] = (),
) -> Iterator[Document]:
    """Read, deduplicate, split and segment, in that order.

    Deduplication happens before splitting so that a duplicate cannot be
    assigned to two different splits, and splitting happens before
    segmentation so that every passage of a document inherits one split.
    """
    seen: set[str] = set()
    for path in paths:
        path = Path(path)
        if path.suffix == ".txt":
            documents: Iterable[Document] = [read_text_file(path)]
        elif path.suffix == ".jsonl":
            if any(str(path).endswith(s) for s in dialogue_suffixes):
                documents = read_dialogue_jsonl(path)
            else:
                documents = read_jsonl_documents(path)
        else:
            raise ValueError(f"unsupported input {path}; expected .txt or .jsonl")

        for document in deduplicate(documents, seen=seen):
            document = replace(
                document,
                split=assign_split(
                    document.document_id,
                    dev_share=dev_share,
                    test_share=test_share,
                    salt=salt,
                ),
            )
            if document.is_dialogue:
                yield segment_dialogue(document, context_turns=context_turns)
            else:
                yield segment_document(document, window=window, stride=stride)


def sha256_of_file(path: Path | str) -> str:
    """The content hash of a file, for provenance records."""
    return sha256_of(Path(path).read_text(encoding="utf-8"))
