"""Streaming chat exports, paired response tasks and a separate delta task."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata
from typing import Callable

from ..semantic_generator.substitutions import ENTITIES_PATH
from .generator import GENERATOR_VERSION, DialogueGenerator, fingerprint
from .memory_text import MEMORY_TEXT_VERSION, render_memory
from .actions import CONTRAST_TASKS
from .extraction_text import EXTRACT_SYSTEM, EXTRACTION_TEXT_VERSION, render_prompt, render_update
from .models import Episode, SCHEMA_VERSION, Transition

PROMPT_VERSION = "dialogue-prompts-v6"
RESPONSE_SYSTEM = (
    "You are a friendly AI personal assistant. Reply briefly and naturally to the latest user message. "
    "Use only the supplied conversation context. If information is missing, say you don't know; "
    "if a reference is ambiguous, ask for clarification. A wish or suggestion is not a completed action. "
    "This chat cannot send messages, set reminders or use external tools."
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _messages(system: str, prompt: str, target: str) -> list[dict[str, str]]:
    # Exactly one assistant target in both response views: equal loss scope.
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt},
            {"role": "assistant", "content": target}]


def response_row(episode: Episode, view: str, *, memory_style: int | None = None) -> dict:
    if view == "history":
        context = "Conversation so far:\n" + "\n".join(
            f"{turn.role.title()}: {turn.text}" for turn in episode.turns[:-1])
    elif view == "memory":
        # Semantic memory is supplied by the simulator, not an extracted model.
        previous = episode.turns[-2].text if len(episode.turns) > 1 else "(none)"
        style = episode.memory_style if memory_style is None else memory_style
        context = render_memory(episode.memory, style=style) + "\n\nLast assistant message: " + previous
    else:
        raise ValueError(f"Unknown response view: {view}")
    prompt = context + "\nLatest user message: " + episode.turns[-1].text
    metadata = ({'contrast_group': episode.contrast_group, 'condition': episode.condition,
                 'answer_kind': ('unknown' if episode.expected_action == 'unknown' else
                                 'no' if episode.answer.startswith('No,') else 'yes')}
                if episode.contrast_group else {})
    return {"id": episode.id, "scenario_key": episode.scenario_key, "task": episode.task, **metadata,
            "messages": _messages(RESPONSE_SYSTEM, prompt, episode.answer)}


def extraction_row(episode: Episode, transition: Transition) -> dict:
    prompt = render_prompt(transition.before, transition.turn, transition.previous_assistant)
    return {"id": episode.id + ":" + transition.turn.id, "scenario_key": episode.scenario_key,
            "task": episode.task,
            "messages": _messages(EXTRACT_SYSTEM, prompt, render_update(transition))}


def prompt_key(row: dict) -> str:
    """Match the trainer's case/punctuation-insensitive prompt deduplication."""
    text = row["messages"][1]["content"]
    normalized = " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))
    return fingerprint(normalized)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_report(path: Path | None, seed: int) -> dict | None:
    if path is None:
        return None
    source = json.loads(path.read_text(encoding="utf-8"))
    settings = source["settings"]
    digest = source["chat_template_sha256"]
    if (not isinstance(settings["model"], str) or not settings["model"].strip()
            or type(settings["max_tokens"]) is not int or settings["max_tokens"] <= 0
            or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
        raise ValueError("Invalid model, max_tokens or chat_template_sha256 in training report")
    return {"settings": {"model": settings["model"], "max_tokens": settings["max_tokens"], "seed": seed},
            "chat_template_sha256": digest, "token_validation": "deferred_to_training_loader",
            "tokenizer_contract_source_sha256": _sha256(path)}


def build_dialogue_dataset(out: Path, *, train: int = 200, validation: int = 20,
                           test: int = 20, seed: int = 42,
                           include_metadata: bool = True,
                           user_assistant_only: bool = False,
                           training_report: Path | None = None,
                           contrast_multiplier: int = 1,
                           progress: Callable[[str, int, int], None] | None = None) -> dict:
    """Write per-task quotas into a new directory, with optional contrast weighting.

    Memory/history are paired plain-text final-answer examples. Extraction transitions
    are deduplicated independently. No tokenizer, model, network or DB is used.
    Disabling metadata exports only messages in training rows; internal grouping,
    validation and the episode audit records retain their identifiers.
    user_assistant_only selects 15 tasks without personal names;
    the default includes the additional named-people task (16 tasks).
    contrast_multiplier scales only the two completion contrast tasks in all
    splits. Its default of 1 preserves the UI's uniform per-task quotas.
    An optional existing trainer report supplies its tokenizer contract; the
    training loader must still validate token lengths before loading weights.
    """
    counts = {"train": train, "validation": validation, "test": test}
    if any(type(count) is not int or count <= 0 for count in counts.values()):
        raise ValueError("All split quotas must be positive integers per task")
    if type(include_metadata) is not bool:
        raise ValueError("include_metadata must be a boolean")
    if type(contrast_multiplier) is not int or contrast_multiplier <= 0:
        raise ValueError('contrast_multiplier must be a positive integer')
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    report = _training_report(training_report, seed)
    generator = DialogueGenerator(seed=seed, user_assistant_only=user_assistant_only)
    tasks = generator.tasks
    weights = {task: contrast_multiplier if task in CONTRAST_TASKS else 1 for task in tasks}
    total = sum(counts.values()) * sum(weights.values())
    processed = 0
    views = ("history", "memory", "extraction")
    seen: dict[str, dict[str, tuple[str, str]]] = {view: {} for view in views}
    scenario_splits: dict[str, str] = {}
    manifest = {"schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION,
                "prompt_version": PROMPT_VERSION, "seed": seed, "quota_per_task": counts,
                "task_multipliers": weights,
                "content_formats": {"history": "history-text-v1", "memory": MEMORY_TEXT_VERSION,
                                    "extraction": EXTRACTION_TEXT_VERSION},
                "include_metadata": include_metadata,
                "user_assistant_only": user_assistant_only,
                "dialogue_profile": "user-assistant-v1" if user_assistant_only else "named-pool-v1",
                "tasks": list(tasks), "splits": {}, "resources": {},
                "split_policy": ("held-out user contexts (location, feeling, drink), hashed 80/10/10; "
                                 "one realization per distinct event plan" if user_assistant_only else
                                 "held-out pool names; one realization per distinct event plan"),
                "shared_between_splits": ["task families", "surface patterns", "common vocabulary"],
                "memory_source": "scenario simulator (oracle)",
                "review_status": "synthetic pilot; inspect before large-scale training",
                "training_report_supplied": report is not None}
    resources = sorted(Path(__file__).parent.glob("*.py"))
    resources.append(ENTITIES_PATH)  # Typed event fillers use the common pool in both profiles.
    for path in resources:
        manifest["resources"][path.name] = _sha256(path)
    frame_path = Path(__file__).parent.parent / 'semantic_generator' / 'frames.py'
    manifest['resources']['semantic_generator/frames.py'] = _sha256(frame_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Failure never publishes a half-finished dataset or overwrites an old one.
    with TemporaryDirectory(prefix=".dialogue-build-", dir=out.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        for view in views:
            (staging / view).mkdir()
        for split, count in counts.items():
            stats = {"episodes": 0, "by_task": {}, "actions": Counter(),
                     "extraction_rows": 0, "extraction_duplicates": 0, "delta_operations": Counter()}
            with ExitStack() as stack:
                audit = stack.enter_context((staging / f"{split}.episodes.jsonl").open("w", encoding="utf-8", newline="\n"))
                writers = {view: stack.enter_context((staging / view / f"{split}.json").open(
                    "w", encoding="utf-8", newline="\n")) for view in views}
                written = Counter()
                for writer in writers.values():
                    writer.write("[\n")

                def write(view: str, row: dict) -> bool:
                    key = prompt_key(row)
                    target = row["messages"][-1]["content"]
                    previous = seen[view].get(key)
                    if previous is not None:
                        if previous != (split, target):
                            raise ValueError(f"Prompt leakage or conflicting targets in {view}/{split}")
                        return False
                    seen[view][key] = (split, target)
                    payload = row if include_metadata else {"messages": row["messages"]}
                    formatted = json.dumps(payload, ensure_ascii=False, indent=2)
                    writers[view].write((",\n" if written[view] else "")
                                        + "  " + formatted.replace("\n", "\n  "))
                    written[view] += 1
                    return True

                for task in tasks:
                    task_count = count * weights[task]
                    accepted = 0
                    phase = f"Generating {split}: {task.replace('_', ' ')}"
                    if progress:
                        progress(phase, processed, total)
                    # Resampling has a bound; quotas are not silently reduced.
                    for episode in generator.generate(task, task_count * 20, split=split):
                        rows = {view: response_row(episode, view) for view in ("history", "memory")}
                        for view, row in rows.items():
                            previous = seen[view].get(prompt_key(row))
                            if previous is not None and previous != (split, episode.answer):
                                raise ValueError(f"Prompt leakage or conflicting targets in {view}/{split}")
                        if any(prompt_key(rows[view]) in seen[view] for view in rows):
                            continue
                        if episode.scenario_key in scenario_splits:
                            raise ValueError("Repeated scenario across tasks or splits")
                        scenario_splits[episode.scenario_key] = split
                        for view, row in rows.items():
                            write(view, row)
                        for transition in episode.transitions:
                            if write("extraction", extraction_row(episode, transition)):
                                delta = transition.delta
                                stats["delta_operations"].update({"add": len(delta.add), "replace": len(delta.replace),
                                                                  "invalidate": len(delta.invalidate),
                                                                  "event_add": len(delta.event_add),
                                                                  "event_replace": len(delta.event_replace),
                                                                  "event_invalidate": len(delta.event_invalidate),
                                                                  "no_change": int(delta == type(delta)())})
                            else:
                                stats["extraction_duplicates"] += 1
                        audit.write(_json(asdict(episode)) + "\n")
                        stats["episodes"] += 1
                        stats["actions"][episode.expected_action] += 1
                        accepted += 1
                        processed += 1
                        if progress and (accepted % 25 == 0 or accepted == task_count):
                            progress(phase, processed, total)
                        if accepted == task_count:
                            break
                    if accepted != task_count:
                        raise ValueError(f"{task}/{split}: only {accepted}/{task_count} unique prompts")
                    stats["by_task"][task] = accepted
                for writer in writers.values():
                    writer.write("\n]\n")
                stats["extraction_rows"] = written["extraction"]
            manifest["splits"][split] = stats
        if progress:
            progress("Checking dataset files", processed, total)
        if report is not None:
            for view in views:
                payload = {**report, "dataset_view": view, "generator_version": GENERATOR_VERSION,
                           "source_manifest": "../manifest.json"}
                (staging / view / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifest["files"] = {path.relative_to(staging).as_posix(): _sha256(path)
                             for path in sorted(staging.rglob("*")) if path.is_file()}
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staging.rename(out)
    return manifest
