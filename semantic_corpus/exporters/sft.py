"""Supervised fine-tuning export for a decoder-only model.

The layout is the handoff's::

    <context>
    ...
    </context>
    <question>
    ...
    </question>
    <answer>
    ...
    </answer>

and the one thing that must not be got wrong is the loss mask. The handoff
says the loss is computed over the answer alone, so each record is emitted
as a ``prompt``/``completion`` pair rather than a single ``text`` field. A
trainer given one blob has to find the boundary by string matching, and a
question containing the tag text would break that silently — the model would
be trained to reproduce the context, which looks like it is learning and is
not.

``text`` is still written for inspection, and it is exactly
``prompt + completion``, which :func:`check_record` asserts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ..question_generator.answers import QAExample
from .jsonl import write_jsonl

__all__ = [
    "CONTEXT_TAG",
    "QUESTION_TAG",
    "ANSWER_TAG",
    "to_record",
    "check_record",
    "export_sft",
    "iter_records",
    "export_sft_splits",
]

CONTEXT_TAG = "context"
QUESTION_TAG = "question"
ANSWER_TAG = "answer"


def _block(tag: str, body: str) -> str:
    return f"<{tag}>\n{body}\n</{tag}>\n"


def to_record(example: QAExample) -> dict[str, Any]:
    """One training record, split at the point the loss starts."""
    prompt = _block(CONTEXT_TAG, example.context) + _block(
        QUESTION_TAG, example.question
    ) + f"<{ANSWER_TAG}>\n"
    completion = f"{example.answer}\n</{ANSWER_TAG}>\n"
    return {
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion,
        "kind": str(example.kind),
        "answerable": example.answerable,
        "document_id": example.document_id,
        "run_id": example.run_id,
        "metadata": dict(example.metadata),
    }


def check_record(record: Mapping[str, Any]) -> list[str]:
    """Problems that would corrupt training rather than merely look odd."""
    problems: list[str] = []
    prompt, completion = record.get("prompt", ""), record.get("completion", "")
    if record.get("text") != prompt + completion:
        problems.append("text is not prompt + completion")
    if not completion.strip():
        problems.append("the completion is empty, so the record teaches nothing")
    if f"</{ANSWER_TAG}>" in prompt:
        problems.append("the prompt already closes the answer block")
    if f"<{CONTEXT_TAG}>" in completion:
        problems.append("the completion reopens the context block")
    return problems


def iter_records(examples: Iterable[QAExample]) -> Iterator[dict[str, Any]]:
    """Convert examples to records, refusing any that would corrupt training."""
    for example in examples:
        record = to_record(example)
        problems = check_record(record)
        if problems:
            raise ValueError(
                f"unusable SFT record for {example.question!r}: "
                + "; ".join(problems)
            )
        yield record


def export_sft(
    examples: Iterable[QAExample],
    path: Path | str,
) -> int:
    """Write an SFT file. Returns the number of records."""
    return write_jsonl(path, iter_records(examples))


def export_sft_splits(
    examples_by_split: Mapping[str, Iterable[QAExample]],
    directory: Path | str,
) -> dict[str, int]:
    """Write one SFT file per split, named ``<split>.jsonl``."""
    directory = Path(directory)
    counts: dict[str, int] = {}
    for split, examples in examples_by_split.items():
        counts[split] = export_sft(examples, directory / f"{split}.jsonl")
    return counts
