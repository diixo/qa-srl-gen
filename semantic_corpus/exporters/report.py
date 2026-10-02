"""Distribution and leakage report over a store.

The handoff asks for this as the last step of the pipeline, and it is the
only component whose job is to find problems rather than produce data. Two
kinds of problem matter, and they fail differently.

**Leakage is fatal and silent.** A document whose text appears in both train
and test makes every number measured on that corpus meaningless, and nothing
else in the pipeline would notice: each split looks internally consistent.
So leakage is checked by content hash, not by identifier — the same text
under two ids is the case that slips through an id-based check — and by
entity surface form, because a held-out name that reappears in training
invalidates exactly the test it was held out for.

**Imbalance is survivable but needs to be visible.** A corpus that is 60%
refusals teaches a model to refuse. The report counts examples by kind,
answerability and split so the mix is a number somebody looked at rather
than whatever the templates happened to emit.

Nothing here modifies the store. A report is a reading.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..documents import sha256_of
from ..storage.repository import CorpusStore

__all__ = ["LeakageFinding", "CorpusReport", "build_report", "write_report"]


@dataclass(frozen=True, slots=True)
class LeakageFinding:
    """One thing that crosses a split boundary and should not."""

    kind: str
    key: str
    splits: tuple[str, ...]
    detail: str = ""

    def __str__(self) -> str:
        where = ", ".join(self.splits)
        return f"[{self.kind}] {self.key!r} appears in {where}" + (
            f" ({self.detail})" if self.detail else ""
        )


@dataclass(slots=True)
class CorpusReport:
    """Everything worth knowing about a corpus before training on it."""

    documents_by_split: Counter = field(default_factory=Counter)
    examples_by_split: Counter = field(default_factory=Counter)
    examples_by_kind: Counter = field(default_factory=Counter)
    answerable: Counter = field(default_factory=Counter)
    labels: Counter = field(default_factory=Counter)
    predicates: Counter = field(default_factory=Counter)
    runs_by_model: Counter = field(default_factory=Counter)
    synthetic_runs: int = 0
    annotated_runs: int = 0
    leakage: list[LeakageFinding] = field(default_factory=list)
    #: Weaker than leakage: the same question and answer under *different*
    #: contexts on both sides of the split line. The model still has to read
    #: the context to answer, so this is not fatal, but a test set full of
    #: pairs it has already seen measures less than it appears to.
    repeated_qa: list[LeakageFinding] = field(default_factory=list)
    ontology_version: str = ""
    content_hash: str = ""

    @property
    def is_clean(self) -> bool:
        return not self.leakage

    @property
    def refusal_share(self) -> float:
        total = sum(self.answerable.values())
        return self.answerable[False] / total if total else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "ontology_version": self.ontology_version,
            "content_hash": self.content_hash,
            "documents_by_split": dict(self.documents_by_split),
            "examples_by_split": dict(self.examples_by_split),
            "examples_by_kind": dict(self.examples_by_kind),
            "answerable": {str(k): v for k, v in self.answerable.items()},
            "refusal_share": round(self.refusal_share, 4),
            "labels": dict(self.labels),
            "predicates": dict(self.predicates.most_common(50)),
            "runs_by_model": dict(self.runs_by_model),
            "synthetic_runs": self.synthetic_runs,
            "annotated_runs": self.annotated_runs,
            "leakage": [_finding_json(f) for f in self.leakage],
            "repeated_qa": [_finding_json(f) for f in self.repeated_qa[:50]],
            "repeated_qa_count": len(self.repeated_qa),
        }

    def __str__(self) -> str:
        lines = [
            f"ontology {self.ontology_version}  content {self.content_hash[:12]}",
            f"documents: {_counts(self.documents_by_split)}",
            f"examples:  {_counts(self.examples_by_split)}",
            f"kinds:     {_counts(self.examples_by_kind)}",
            f"refusals:  {self.refusal_share:.1%}",
            f"runs:      {self.synthetic_runs} synthetic, "
            f"{self.annotated_runs} annotated",
        ]
        if self.labels:
            lines.append(f"labels:    {_counts(self.labels)}")
        if self.leakage:
            lines.append(f"LEAKAGE:   {len(self.leakage)} finding(s)")
            lines.extend(f"  {finding}" for finding in self.leakage[:10])
        else:
            lines.append("leakage:   none")
        if self.repeated_qa:
            lines.append(
                f"repeated:  {len(self.repeated_qa)} question/answer pairs "
                "recur across splits under different contexts"
            )
        return "\n".join(lines)


def _finding_json(finding: "LeakageFinding") -> dict[str, Any]:
    return {
        "kind": finding.kind,
        "key": finding.key,
        "splits": list(finding.splits),
        "detail": finding.detail,
    }


def _counts(counter: Counter) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items(), key=str))


def build_report(
    store: CorpusStore, *, held_out_names: Iterable[str] = ()
) -> CorpusReport:
    """Read a store and report its composition and any leakage.

    *held_out_names* are entity surface forms that must not appear in
    training text. Checking them is what turns ``unseen_names_test`` from a
    claim into a verified property.
    """
    report = CorpusReport(
        ontology_version=store.ontology_version, content_hash=store.content_hash()
    )

    text_by_hash: dict[str, list[tuple[str, str | None]]] = {}
    split_of: dict[str, str | None] = {}
    train_text: list[str] = []

    for row in store.read("documents"):
        split = row.get("document_split")
        split_of[row["document_id"]] = split
        report.documents_by_split[split or "unassigned"] += 1
        text_by_hash.setdefault(row["source_hash"], []).append(
            (row["document_id"], split)
        )
        if split == "train":
            train_text.append(row["text"])

    # Same text in two splits: the failure an id-based check cannot see.
    for digest, entries in text_by_hash.items():
        splits = {split for _, split in entries}
        if len(splits) > 1:
            report.leakage.append(
                LeakageFinding(
                    kind="duplicate-text",
                    key=digest[:12],
                    splits=tuple(sorted(str(s) for s in splits)),
                    detail=", ".join(document_id for document_id, _ in entries),
                )
            )

    for row in store.read("spans"):
        if row.get("record") == "predicate":
            report.predicates[row.get("lemma", "?")] += 1
        for label in row.get("labels", ()):
            report.labels[label] += 1

    for header in store.read("annotation_runs"):
        report.runs_by_model[header.get("model_name", "unknown")] += 1
        if header.get("synthetic"):
            report.synthetic_runs += 1
        else:
            report.annotated_runs += 1
        if header.get("ontology_version") != store.ontology_version:
            report.leakage.append(
                LeakageFinding(
                    kind="ontology-mismatch",
                    key=header["run_id"],
                    splits=(),
                    detail=f"run uses {header.get('ontology_version')!r}",
                )
            )

    exact: dict[str, set[str]] = {}
    pairs: dict[str, set[str]] = {}
    for row in store.read("qa_examples"):
        split = split_of.get(row.get("document_id"), None)
        report.examples_by_split[split or "unassigned"] += 1
        report.examples_by_kind[row.get("kind", "?")] += 1
        report.answerable[bool(row.get("answerable", True))] += 1
        question, answer = row.get("question", ""), row.get("answer", "")
        exact.setdefault(
            sha256_of(f"{row.get('context','')}\x00{question}\x00{answer}"), set()
        ).add(str(split))
        pairs.setdefault(sha256_of(f"{question}\x00{answer}"), set()).add(str(split))

    # An identical example — same context as well — on both sides of the
    # split line is leakage outright.
    for key, splits in exact.items():
        real = {s for s in splits if s not in ("None", "unassigned")}
        if len(real) > 1:
            report.leakage.append(
                LeakageFinding(
                    kind="duplicate-example", key=key[:12], splits=tuple(sorted(real))
                )
            )

    # The same question and answer under different contexts is weaker: the
    # model still has to read the context, so it is reported apart.
    for key, splits in pairs.items():
        real = {s for s in splits if s not in ("None", "unassigned")}
        if len(real) > 1 and key not in exact:
            report.repeated_qa.append(
                LeakageFinding(
                    kind="repeated-qa", key=key[:12], splits=tuple(sorted(real))
                )
            )

    corpus = "\n".join(train_text)
    for name in held_out_names:
        if name and name in corpus:
            report.leakage.append(
                LeakageFinding(
                    kind="held-out-name",
                    key=name,
                    splits=("train",),
                    detail="appears in training text",
                )
            )

    return report


def write_report(report: CorpusReport, path: Path | str) -> Path:
    """Save a report as JSON next to the data it describes."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
