"""Run the local DailyDialog train corpus through annotation, storage and export.

No model or network access. Use a fresh output directory for every run.
Example: python scripts/run_dailydialog.py --out artifacts/dailydialog-audit
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from semantic_corpus.exporters import build_report, export_bio, export_sft, write_report
from semantic_corpus.question_generator import QuestionGenerator
from semantic_corpus.semantic_annotator import (
    CandidateResources, RuleBasedTeacher, annotate_document,
    read_dialogue_jsonl, segment_dialogue,
)
from semantic_corpus.storage import CorpusStore


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=REPO_ROOT / "data/dailydialog/filtered/dailydialog-train.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-turns", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Read only this many input dialogues for a smoke run")
    args = parser.parse_args()
    if args.context_turns < 0 or (args.limit is not None and args.limit < 1):
        parser.error("context-turns must be nonnegative and limit must be positive")
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if args.out.exists():
        parser.error(f"output already exists; choose a fresh directory: {args.out}")

    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    args.out.mkdir(parents=True)
    timings: Counter = Counter()
    counts: Counter = Counter()

    @contextmanager
    def timed(phase):
        begin = perf_counter()
        try:
            yield
        finally:
            timings[phase] += perf_counter() - begin

    with timed("load_resources"):
        resources = CandidateResources.load(wiktionary_dir=str(REPO_ROOT / "data/wiktionary"))
    teacher = RuleBasedTeacher()
    questions = QuestionGenerator(seed=args.seed)
    store = CorpusStore(args.out / "store")
    config = {
        "input": str(args.input.resolve()), "input_sha256": file_hash(args.input),
        "seed": args.seed, "context_turns": args.context_turns,
        "limit": args.limit, "split": "train", "teacher": teacher.name,
    }
    source_hashes = {
        str(path.relative_to(REPO_ROOT)): file_hash(path)
        for path in sorted((REPO_ROOT / "semantic_corpus").rglob("*.py"))
    }
    source_hashes[str(Path(__file__).resolve().relative_to(REPO_ROOT))] = file_hash(Path(__file__))
    seen: set[str] = set()
    documents = read_dialogue_jsonl(args.input, source="DailyDialog")
    if args.limit is not None:
        documents = islice(documents, args.limit)
    for document in documents:
        counts["input_documents"] += 1
        counts["input_utterances"] += len(document.utterances)
        if document.sha256 in seen:
            counts["duplicate_documents"] += 1
            continue
        seen.add(document.sha256)
        with timed("segment_and_store_document"):
            document = segment_dialogue(replace(document, split="train"), context_turns=args.context_turns)
            if not store.add_document(document):
                raise AssertionError("deduplicated document was unexpectedly rejected")
        with timed("annotate"):
            outcome = annotate_document(document, teacher, resources=resources,
                                        run_id=f"{document.document_id}-rules", random_seed=args.seed)
        if not outcome.ok:
            raise AssertionError(f"{document.document_id}: {outcome.structure}")
        counts["structural_errors"] += len(outcome.structure.problems) + len(outcome.structure.overlapping)
        counts["rejected_annotations"] += len(outcome.rejected)
        counts["candidates"] += len(outcome.candidates)
        counts["utterances"] += len(document.utterances)
        with timed("store_annotation"):
            store.add_run(outcome.run, document)
        with timed("generate_qa"):
            examples = questions.for_document(document, outcome.run)
        counts["generated_qa"] += len(examples)
        with timed("store_qa"):
            counts["stored_qa"] += store.add_examples(examples, generation_run_id=outcome.run.run_id)
        counts["documents"] += 1
        if counts["documents"] % 500 == 0:
            print(f"{counts['documents']} documents, {counts['stored_qa']} QA, "
                  f"{perf_counter() - started:.1f}s", flush=True)

    print("Annotation complete; exporting saved data.", flush=True)
    with timed("publish_version"):
        version = store.publish_version("dailydialog-audit-v1", build_config=config)
    with timed("export_sft"):
        sft_count = export_sft(store.examples(split="train"), args.out / "sft.jsonl")

    def saved_pairs():
        for document in store.documents(split="train"):
            run, = store.runs(document_id=document.document_id)
            yield document, run

    with timed("export_bio"):
        bio = export_bio(saved_pairs(), args.out / "bio.jsonl")
    with timed("report"):
        report = build_report(store)
        write_report(report, args.out / "report.json")
    if sft_count != counts["stored_qa"] or bio.documents != counts["documents"]:
        raise AssertionError("export counts differ from stored data")
    summary = {
        "started_at": started_at, "python": platform.python_version(),
        "platform": platform.platform(), "config": config, "source_sha256": source_hashes,
        "counts": dict(counts), "tables": version["counts"],
        "timings_seconds": {key: round(value, 3) for key, value in timings.items()},
        "total_seconds": round(perf_counter() - started, 3),
        "store_bytes": sum(p.stat().st_size for p in store.root.iterdir() if p.is_file()),
        "content_hash": version["content_hash"], "sft_records": sft_count,
        "bio": asdict(bio), "report_clean": report.is_clean,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "source_sha256"}, indent=2), flush=True)
    return 0 if report.is_clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
