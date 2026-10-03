"""Command-line entry point: ``python -m semantic_corpus.cli <command>``.

Commands
--------
``inspect``    print the first sentences of a bank file
``roundtrip``  render and re-parse every question of a bank file, reporting
               how often each direction is exact
``validate``   run the invariant checks over a bank file
``lookup``     look a verb up in the Wiktionary inflection indexes
``frames``     print the typed frames the generator knows
``generate``   generate sentences with their spans and relations
``ambiguity``  show name pairs whose type only context settles
``ingest``     read source texts into documents, with splits and passages
``candidates`` show what candidate extraction proposes for a text
``annotate``   run the annotator against an HTTP teacher
``questions``  generate QA examples from generated situations
``build``      generate, store, export SFT/BIO and report on the result
``report``     read an existing store and report distribution and leakage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, Sequence

from .qasrl_core.bank_reader import read_bank
from .qasrl_core.inflections import load_inflections
from .qasrl_core.models import Sentence
from .qasrl_core.question_parser import QuestionParseError, parse_question
from .qasrl_core.question_renderer import render_question
from .qasrl_core.validation import check_sentence, count_problems

DEFAULT_INFLECTIONS = Path("data/wiktionary/en_verb_inflections.txt")


def _limited(sentences: Iterator[Sentence], limit: int | None) -> Iterator[Sentence]:
    for index, sentence in enumerate(sentences):
        if limit is not None and index >= limit:
            return
        yield sentence


def cmd_inspect(args: argparse.Namespace) -> int:
    for sentence in _limited(read_bank(args.path), args.limit):
        print(f"{sentence.sentence_id}: {sentence.text}")
        for entry in sentence:
            verb = sentence.sentence_tokens[entry.verb_index]
            print(f"  [{entry.verb_index}] {verb} (lemma {entry.verb_inflected_forms.stem})")
            for label in entry:
                spans = sorted(label.span_votes().items(), key=lambda kv: (-kv[1], kv[0]))
                answers = "; ".join(
                    f"{span.text(sentence.sentence_tokens)} ({votes})"
                    for span, votes in spans
                )
                print(f"      {label.question_string:55s} {answers}")
        print()
    return 0


def cmd_roundtrip(args: argparse.Namespace) -> int:
    rendered_ok = rendered_bad = 0
    parsed_ok = parsed_bad = slots_exact = 0
    failures: list[str] = []
    for sentence in _limited(read_bank(args.path), args.limit):
        for entry, label in sentence.question_labels():
            forms = entry.verb_inflected_forms
            if render_question(label.question_slots, forms) == label.question_string:
                rendered_ok += 1
            else:
                rendered_bad += 1
                if len(failures) < args.show:
                    failures.append(f"render: {label.question_string!r}")
            try:
                slots = parse_question(label.question_string, forms)
            except QuestionParseError as error:
                parsed_bad += 1
                if len(failures) < args.show:
                    failures.append(f"parse: {error}")
                continue
            if render_question(slots, forms) == label.question_string:
                parsed_ok += 1
            else:
                parsed_bad += 1
            if slots == label.question_slots:
                slots_exact += 1
            elif len(failures) < args.show:
                failures.append(
                    f"slots: {label.question_string!r}\n"
                    f"       stored {label.question_slots.to_json()}\n"
                    f"       parsed {slots.to_json()}"
                )

    total = rendered_ok + rendered_bad
    print(f"questions:            {total}")
    _report("render -> string", rendered_ok, total)
    _report("string -> render", parsed_ok, total)
    _report("exact slot recovery", slots_exact, total)
    for failure in failures:
        print(f"  {failure}")
    return 0 if rendered_bad == 0 and parsed_bad == 0 else 1


def _report(label: str, ok: int, total: int) -> None:
    share = 100.0 * ok / total if total else 0.0
    print(f"{label:22s} {ok}/{total} ({share:.4f}%)")


def cmd_validate(args: argparse.Namespace) -> int:
    all_counts: dict[str, int] = {}
    sentences = 0
    shown = 0
    for sentence in _limited(read_bank(args.path), args.limit):
        sentences += 1
        problems = check_sentence(sentence)
        for code, count in count_problems(problems).items():
            all_counts[code] = all_counts.get(code, 0) + count
        for problem in problems:
            if shown < args.show:
                print(problem)
                shown += 1
    print(f"sentences checked: {sentences}")
    if not all_counts:
        print("no problems found")
        return 0
    for code, count in sorted(all_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {code}: {count}")
    return 1


def cmd_lookup(args: argparse.Namespace) -> int:
    lexicon = load_inflections(args.inflections)
    print(f"{len(lexicon)} lemmas loaded from {args.inflections}")
    for word in args.words:
        print(f"\n{word}")
        paradigms = lexicon.paradigms(word)
        if paradigms:
            for paradigm in paradigms:
                print("  as lemma: " + " / ".join(paradigm.all_forms))
        else:
            print("  as lemma: not in the dictionary")
        analyses = lexicon.analyses(word)
        if analyses:
            print(f"  as surface form: {len(analyses)} analysis/es")
            for paradigm, form in analyses[: args.limit or 10]:
                print(f"    {form.value:20s} of {paradigm.stem}")
            if args.limit and len(analyses) > args.limit:
                print(f"    ... and {len(analyses) - args.limit} more")
        else:
            print("  as surface form: no analysis")
    return 0


def cmd_frames(args: argparse.Namespace) -> int:
    from .semantic_generator import DEFAULT_FRAMES

    for frame in DEFAULT_FRAMES:
        if args.lemma and frame.lemma != args.lemma:
            continue
        print(frame)
        for pattern in frame.patterns:
            parts = [f"{{{pattern.subject}}}", f"<{frame.lemma}>"]
            for complement in pattern.complements:
                if complement.preposition:
                    parts.append(complement.preposition)
                parts.append(f"{{{complement.slot}}}")
            flag = " [passive]" if pattern.is_passive else ""
            print(f"  {pattern.name:20s} {' '.join(parts)}.{flag}")
        print()
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from .semantic_generator import Generator

    generator = Generator(seed=args.seed, split=args.split)
    for realized in generator.generate(args.count):
        print(realized.text)
        if args.spans:
            for mention in realized.mentions:
                print(
                    f"    {mention.slot:22s} "
                    f"[{mention.start_char:3d},{mention.end_char:3d}) "
                    f"{mention.text!r} {mention.labels}"
                )
            for edge in realized.relations:
                print(f"    {edge.source_slot} --{edge.relation}--> {edge.target}")
    return 0


def cmd_ambiguity(args: argparse.Namespace) -> int:
    from .semantic_generator import Generator

    generator = Generator(seed=args.seed)
    for pair in generator.ambiguity_pairs():
        print(f"{pair.surface_form}: {pair.left_label} vs {pair.right_label}")
        print(f"  {pair.left.text}")
        print(f"  {pair.right.text}")
        print(f"  {pair.question()}")
        print()
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from .semantic_annotator import ingest

    counts: dict[str, int] = {}
    shown = 0
    for document in ingest(
        args.paths, window=args.window, dialogue_suffixes=args.dialogue_suffix
    ):
        counts[document.split or "?"] = counts.get(document.split or "?", 0) + 1
        if shown < args.limit:
            shown += 1
            print(
                f"{document.document_id} [{document.split}] "
                f"{len(document.passages)} passages, "
                f"{len(document.utterances)} turns, sha256={document.sha256[:12]}"
            )
            if args.passages:
                for passage in document.passages:
                    body = passage.text_in(document).replace("\n", " / ")
                    print(f"    {passage.passage_id}: {body[:100]!r}")
    print(f"documents by split: {dict(sorted(counts.items()))}")
    return 0


def cmd_candidates(args: argparse.Namespace) -> int:
    from .documents import Document
    from .semantic_annotator import CandidateResources, extract_candidates

    text = args.text if args.text else Path(args.file).read_text(encoding="utf-8")
    resources = CandidateResources.load(wiktionary_dir=args.wiktionary)
    document = Document(document_id="cli", text=text)
    for candidate in extract_candidates(document, None, resources):
        print(
            f"  {candidate.exact_text:18s} {str(candidate.proposed_labels):44s} "
            f"{','.join(candidate.evidence)}"
        )
        if args.verbose:
            print(f"      lemmas={candidate.lemmas} tags={dict(candidate.tag_counts)}")
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    from .ontology import Label
    from .semantic_annotator import (
        CandidateResources,
        HttpTeacher,
        annotate_document,
        ingest,
    )

    teacher = HttpTeacher(
        args.teacher_url, name=args.model, labels=[str(label) for label in Label]
    )
    resources = CandidateResources.load(wiktionary_dir=args.wiktionary)
    failures = 0
    for index, document in enumerate(ingest(args.paths)):
        if args.limit and index >= args.limit:
            break
        outcome = annotate_document(
            document, teacher, run_id=f"{args.run_id}-{index:05d}", resources=resources
        )
        print(outcome.summary())
        if not outcome.ok:
            failures += 1
            print(outcome.structure)
        for rejection in outcome.rejected[: args.show]:
            print(f"    rejected {rejection}")
    return 1 if failures else 0


def cmd_questions(args: argparse.Namespace) -> int:
    from .question_generator import QuestionGenerator
    from .semantic_generator import Generator
    from .semantic_generator.canonical import to_canonical
    from .semantic_generator.generator import DEFAULT_PARADIGMS

    generator = Generator(seed=args.seed, split=args.split)
    questions = QuestionGenerator(
        paradigms=DEFAULT_PARADIGMS,
        ambiguous_forms=generator.pool.ambiguous_forms(),
        seed=args.seed,
        include_paraphrases=not args.no_paraphrases,
    )
    produced = 0
    for index, realized in enumerate(generator.generate(args.count)):
        document, run = to_canonical(realized, document_id=f"gen-{index:05d}")
        examples = questions.for_document(document, run)
        if args.prompts:
            for example in examples:
                print(example.to_prompt())
                print()
        else:
            print(document.text)
            for example in examples:
                flag = "" if example.answerable else "   [unanswerable]"
                print(f"  [{str(example.kind):11s}] {example.question}"
                      f"  ->  {example.answer}{flag}")
            print()
        produced += len(examples)
    print(f"{produced} examples from {args.count} situations")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from .exporters import build_report, export_bio, export_sft, write_report
    from .question_generator import QuestionGenerator, balance
    from .semantic_generator import Generator
    from .semantic_generator.canonical import to_canonical
    from .semantic_generator.generator import DEFAULT_PARADIGMS
    from .storage import CorpusStore
    from .storage.repository import GENERATOR_VERSION, StoreError

    store = CorpusStore(args.store)
    generator = Generator(seed=args.seed, split=args.split)
    questions = QuestionGenerator(
        paradigms=DEFAULT_PARADIGMS,
        ambiguous_forms=generator.pool.ambiguous_forms(),
        seed=args.seed,
    )

    # A repeated build is idempotent. Check identity/version conflicts before
    # any document is appended, including conflicts late in the batch.
    build_config = {key: getattr(args, key) for key in
                    ("seed", "split", "prefix", "count", "max_per_kind", "no_answer_share")}
    previous = next((v for v in store.versions() if v["version_id"] == args.version), None)
    if previous is not None and previous.get("build_config") != build_config:
        raise StoreError(f"version {args.version!r} exists; use a new version")
    if previous is not None and (
        previous.get("generator_version") != GENERATOR_VERSION
        or previous["content_hash"] != store.content_hash()
    ):
        raise StoreError(f"version {args.version!r} names different content or generator; use a new version")
    known_content = {r["source_hash"]: r.get("document_split") for r in store.read("documents")}
    known_runs = {r["run_id"] for r in store.read("annotation_runs")}
    for index, realized in enumerate(generator.generate(args.count)):
        document, run = to_canonical(realized, document_id=f"{args.prefix}-{index:06d}",
                                   split=args.split, random_seed=args.seed)
        existing = store.document(document.document_id)
        if existing is not None and existing != document:
            raise StoreError(f"document {document.document_id!r} is immutable; use a new prefix")
        if existing is None and run.run_id in known_runs:
            raise StoreError(f"run {run.run_id!r} already exists; use a new prefix")
        if document.sha256 in known_content and known_content[document.sha256] != document.split:
            raise StoreError("duplicate text cannot cross splits")
        if previous is not None and existing is None:
            # Content duplicates can legitimately have skipped identifiers.
            if document.sha256 not in known_content:
                raise StoreError(f"version {args.version!r} exists; use a new version")
    generator = Generator(seed=args.seed, split=args.split)
    for index, realized in enumerate(generator.generate(args.count)):
        document, run = to_canonical(
            realized,
            document_id=f"{args.prefix}-{index:06d}",
            split=args.split,
            random_seed=args.seed,
        )
        if not store.add_document(document) and store.document(document.document_id) is None:
            continue
        if run.run_id not in known_runs:
            store.add_run(run, document)
            known_runs.add(run.run_id)
        examples = questions.for_document(document, run)
        if args.no_answer_share is not None or args.max_per_kind:
            examples = balance(
                examples,
                seed=args.seed,
                max_per_kind=args.max_per_kind or None,
                no_answer_share=args.no_answer_share,
            )
        store.add_examples(examples, generation_run_id=run.run_id)

    version = store.publish_version(args.version, notes=args.notes, build_config=build_config)
    print(f"stored {version['counts']['documents']} documents, "
          f"{version['counts']['qa_examples']} examples "
          f"(version {version['version_id']}, {version['content_hash'][:12]})")

    out = Path(args.out)
    written = export_sft(store.examples(split=args.split), out / "sft.jsonl")
    print(f"sft: {written} records -> {out / 'sft.jsonl'}")
    def stored_pairs():
        for doc in store.documents(split=args.split):
            latest = None
            for stored_run in store.runs(document_id=doc.document_id):
                latest = stored_run
            if latest is not None:
                yield doc, latest
    bio = export_bio(stored_pairs(), out / "bio.jsonl", scheme=args.scheme)
    print(f"bio: {bio}")

    # Names held out for the unseen-names test must not be in training text.
    from .semantic_generator import default_pool

    held_out = (
        list(default_pool().split("test").surface_forms)
        if args.split == "train"
        else []
    )
    report = build_report(store, held_out_names=held_out)
    write_report(report, out / "report.json")
    print()
    print(report)
    return 0 if report.is_clean else 1


def cmd_report(args: argparse.Namespace) -> int:
    from .exporters import build_report, write_report
    from .storage import CorpusStore

    report = build_report(CorpusStore(args.store, create=False))
    print(report)
    if args.out:
        write_report(report, args.out)
    return 0 if report.is_clean else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semantic-corpus", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="print sentences from a bank file")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--limit", type=int, default=3)
    inspect.set_defaults(func=cmd_inspect)

    roundtrip = sub.add_parser("roundtrip", help="render/parse every question")
    roundtrip.add_argument("path", type=Path)
    roundtrip.add_argument("--limit", type=int, default=None)
    roundtrip.add_argument("--show", type=int, default=10)
    roundtrip.set_defaults(func=cmd_roundtrip)

    validate = sub.add_parser("validate", help="run invariant checks")
    validate.add_argument("path", type=Path)
    validate.add_argument("--limit", type=int, default=None)
    validate.add_argument("--show", type=int, default=20)
    validate.set_defaults(func=cmd_validate)

    lookup = sub.add_parser("lookup", help="look verbs up in the inflection indexes")
    lookup.add_argument("words", nargs="+")
    lookup.add_argument("--inflections", type=Path, default=DEFAULT_INFLECTIONS)
    lookup.add_argument("--limit", type=int, default=10)
    lookup.set_defaults(func=cmd_lookup)

    frames = sub.add_parser("frames", help="print the generator's typed frames")
    frames.add_argument("--lemma", default=None)
    frames.set_defaults(func=cmd_frames)

    generate = sub.add_parser("generate", help="generate sentences from frames")
    generate.add_argument("--count", type=int, default=10)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--split", choices=("train", "dev", "test"), default=None)
    generate.add_argument("--spans", action="store_true", help="show spans and relations")
    generate.set_defaults(func=cmd_generate)

    ambiguity = sub.add_parser("ambiguity", help="name pairs settled only by context")
    ambiguity.add_argument("--seed", type=int, default=0)
    ambiguity.set_defaults(func=cmd_ambiguity)

    ingest_cmd = sub.add_parser("ingest", help="read source texts into documents")
    ingest_cmd.add_argument("paths", nargs="+", type=Path)
    ingest_cmd.add_argument("--window", type=int, default=3)
    ingest_cmd.add_argument("--limit", type=int, default=5)
    ingest_cmd.add_argument("--passages", action="store_true")
    ingest_cmd.add_argument(
        "--dialogue-suffix",
        nargs="*",
        default=[],
        help="filename endings to read as dialogue, e.g. dailydialog-val.jsonl",
    )
    ingest_cmd.set_defaults(func=cmd_ingest)

    candidates = sub.add_parser("candidates", help="show proposed annotation spans")
    group = candidates.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--file", type=Path)
    candidates.add_argument("--wiktionary", default="data/wiktionary")
    candidates.add_argument("--verbose", action="store_true")
    candidates.set_defaults(func=cmd_candidates)

    annotate = sub.add_parser("annotate", help="annotate documents with an HTTP teacher")
    annotate.add_argument("paths", nargs="+", type=Path)
    annotate.add_argument("--teacher-url", required=True)
    annotate.add_argument("--model", default="http-teacher")
    annotate.add_argument("--run-id", default="run")
    annotate.add_argument("--wiktionary", default="data/wiktionary")
    annotate.add_argument("--limit", type=int, default=0)
    annotate.add_argument("--show", type=int, default=5)
    annotate.set_defaults(func=cmd_annotate)

    questions = sub.add_parser("questions", help="generate QA examples")
    questions.add_argument("--count", type=int, default=5)
    questions.add_argument("--seed", type=int, default=0)
    questions.add_argument("--split", choices=("train", "dev", "test"), default=None)
    questions.add_argument("--no-paraphrases", action="store_true")
    questions.add_argument("--prompts", action="store_true",
                           help="print the decoder-only training layout")
    questions.set_defaults(func=cmd_questions)

    build = sub.add_parser("build", help="generate, store, export and report")
    build.add_argument("--store", type=Path, default=Path("build/store"))
    build.add_argument("--out", type=Path, default=Path("build/export"))
    build.add_argument("--count", type=int, default=100)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--split", choices=("train", "dev", "test"), default="train")
    build.add_argument("--prefix", default="gen")
    build.add_argument("--version", default="v1")
    build.add_argument("--notes", default="")
    build.add_argument("--scheme", choices=("BIO", "BILOU"), default="BIO")
    build.add_argument("--max-per-kind", type=int, default=0)
    build.add_argument("--no-answer-share", type=float, default=None)
    build.set_defaults(func=cmd_build)

    report = sub.add_parser("report", help="distribution and leakage of a store")
    report.add_argument("--store", type=Path, required=True)
    report.add_argument("--out", type=Path, default=None)
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command. Store refusals propagate, so callers can catch them."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


def _run_from_shell() -> int:
    """:func:`main` for a terminal: a refused write is a message, not a traceback.

    A :class:`~.storage.StoreError` here is the store doing its job — an
    existing version, an immutable document — so the user needs the reason
    and the exit status, not a stack trace into the storage layer.
    """
    from .storage.repository import StoreError

    try:
        return main()
    except StoreError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(_run_from_shell())
