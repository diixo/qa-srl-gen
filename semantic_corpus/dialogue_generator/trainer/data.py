"""Load one exported dialogue view and reject ambiguous or leaking splits."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from ..export import prompt_key
from .chatml import validate_messages


@dataclass(frozen=True)
class Example:
    messages: list[dict[str, str]]
    task: str
    scenario_key: str | None
    prompt_key: str
    contrast_group: str | None = None
    condition: str | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[Example]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a nonempty JSON array")
    examples = []
    seen = set()
    for index, row in enumerate(rows, 1):
        try:
            if not isinstance(row, dict):
                raise ValueError("Expected a messages record")
            messages = validate_messages(row.get("messages"))
            key = prompt_key(row)
            if key in seen:
                raise ValueError("Repeated normalized user prompt (duplicate or conflicting answer)")
            task = row.get("task", "unlabeled")
            scenario = row.get("scenario_key")
            group, condition = row.get('contrast_group'), row.get('condition')
            if any(value is not None and (not isinstance(value, str) or not value)
                   for value in (group, condition)):
                raise ValueError('Invalid contrast metadata')
            if not isinstance(task, str) or not task or (scenario is not None and not isinstance(scenario, str)):
                raise ValueError("Invalid task or scenario_key")
            seen.add(key)
            examples.append(Example(messages, task, scenario, key, group, condition))
        except ValueError as error:
            raise ValueError(f"{path}, record {index}: {error}") from error
    return examples


def check_disjoint(left: list[Example], right: list[Example]) -> None:
    if {row.contrast_group for row in left if row.contrast_group} & \
            {row.contrast_group for row in right if row.contrast_group}:
        raise ValueError('Contrast group leakage between dataset splits')
    if {row.prompt_key for row in left} & {row.prompt_key for row in right}:
        raise ValueError("Prompt leakage between dataset splits")
    left_scenarios = {row.scenario_key for row in left if row.scenario_key}
    if left_scenarios & {row.scenario_key for row in right if row.scenario_key}:
        raise ValueError("Scenario leakage between dataset splits")
