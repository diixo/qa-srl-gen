"""Streaming JSONL output.

Everything written here goes out one record at a time. A corpus is expected
to outgrow memory long before it outgrows disk, and an exporter that
materialises the whole thing first would be the first component to fail on a
real run.

Keys are sorted on write. The point is diffability: two exports of the same
data produce byte-identical files, so a change in a corpus shows up as a
change in the file rather than being hidden by dictionary ordering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

__all__ = ["write_jsonl", "read_jsonl", "write_records", "ExportCounts"]


def write_jsonl(
    path: Path | str,
    rows: Iterable[Mapping[str, Any]],
    *,
    append: bool = False,
) -> int:
    """Write *rows* to *path*, one JSON object per line. Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a" if append else "w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
    return written


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream a JSONL file back."""
    path = Path(path)
    with path.open("rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: {error}") from error


class ExportCounts(dict):
    """How many records each output file received."""

    def total(self) -> int:
        return sum(self.values())

    def __str__(self) -> str:
        return ", ".join(f"{name}: {count}" for name, count in sorted(self.items()))


def write_records(
    directory: Path | str, groups: Mapping[str, Iterable[Mapping[str, Any]]]
) -> ExportCounts:
    """Write several named streams into one directory as ``<name>.jsonl``."""
    directory = Path(directory)
    counts = ExportCounts()
    for name, rows in groups.items():
        counts[name] = write_jsonl(directory / f"{name}.jsonl", rows)
    return counts
