"""Publish immutable mined banks as components of the built-in resource pool.

The full validator runs on a staged copy before publication. Each successful job
gets its own directory, so concurrent jobs never rewrite a shared catalog or
the original eng-base bank. Small publication records make discovery cheap.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import uuid

from .template_bank import TemplateBank, fingerprint
from .template_validation import validate_bank

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_FORMAT = "built-in-template-component-v1"
_BANK_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")


def _registry_root(repo_root):
    return (Path(repo_root) if repo_root is not None else REPO_ROOT) / "data" / "mined-templates"


def _read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _validated_report(bank):
    report = _read_json(bank.path / "validation.json")
    source_sha256 = bank.manifest.get("source_sha256")
    if (not isinstance(source_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
            or report.get("status") != "passed"
            or report.get("source_sha256") != source_sha256):
        raise ValueError("a passed validation report for this bank's source is required")
    for observed, declared in (("source_records", "compiled_records"),
                               ("templates", "templates"), ("token_spans", "tokens_audited")):
        actual = report.get(observed)
        expected = bank.manifest["counts"].get(declared)
        if type(actual) is not int or actual < 0 or actual != expected:
            raise ValueError(f"validation report does not match bank count {declared!r}")
    return report


def _publication(bank, report, bank_id):
    return {"format": PUBLICATION_FORMAT, "bank_id": bank_id,
            "source_sha256": bank.manifest["source_sha256"],
            "manifest_sha256": fingerprint(bank.manifest),
            "validation_sha256": fingerprint(report)}


def _registered_bank(path):
    bank = TemplateBank(path)
    report = _validated_report(bank)
    if (_read_json(path / "publication.json") != _publication(bank, report, path.name)
            or not (path / "source.jsonl").is_file()):
        raise ValueError(f"invalid template bank publication: {path.name}")
    return bank


def registered_banks(repo_root=None):
    """Discover published banks without decompressing data or hashing sources.

    A damaged published component is an error, not a silently omitted input.
    Hidden staging directories left by interrupted jobs are never discoverable.
    """
    root = _registry_root(repo_root)
    if not root.exists():
        return ()
    banks = []
    for path in sorted(root.iterdir()):
        if path.name.startswith(".") or not path.is_dir():
            continue
        if path.is_symlink() or not _BANK_ID.fullmatch(path.name):
            raise ValueError(f"invalid registered template bank directory: {path.name}")
        try:
            banks.append(_registered_bank(path))
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid registered template bank {path.name!r}: {error}") from error
    return tuple(banks)


def register_bank(bank_path, source_path, *, repo_root=None, bank_id, before_publish=None):
    """Validate and publish a bank; preserve the source and existing banks.

    Validation runs on the exact copied bank and source before an atomic rename.
    Any report in the input bank is replaced by this authoritative result.
    Repeating the same publication is harmless; a conflicting ID is rejected.
    ``before_publish`` can reject a cancelled job after validation and before
    its resources become visible; an exception removes only this job's staging.
    """
    if not isinstance(bank_id, str) or not _BANK_ID.fullmatch(bank_id):
        raise ValueError("bank_id must contain only lowercase letters, digits, '_' or '-'")
    bank = TemplateBank(bank_path)
    source = Path(source_path)
    root = _registry_root(repo_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / bank_id
    staging = root / f".{bank_id}-{uuid.uuid4().hex}.staging"
    try:
        shutil.copytree(bank.path, staging)
        shutil.copyfile(source, staging / "source.jsonl")
        copied = TemplateBank(staging)
        if fingerprint(copied.manifest) != fingerprint(bank.manifest):
            raise ValueError("bank manifest changed during publication")
        report = validate_bank(copied, staging / "source.jsonl")
        (staging / "validation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        publication = _publication(copied, _validated_report(copied), bank_id)
        (staging / "publication.json").write_text(
            json.dumps(publication, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if before_publish is not None:
            before_publish()
        if target.exists():
            existing = _registered_bank(target)
            if _publication(existing, _validated_report(existing), bank_id) == publication:
                return target
            raise FileExistsError(f"a different template bank is already registered as {bank_id!r}")
        try:
            staging.rename(target)
        except OSError:
            # Two workers can finish the exact same publication together. An
            # existing different target is never overwritten or removed.
            if target.exists():
                existing = _registered_bank(target)
                if _publication(existing, _validated_report(existing), bank_id) == publication:
                    return target
            raise
    finally:
        # Only our unique, resolved staging directory may be removed.
        if staging.exists() and staging.resolve().parent == root and staging.name.startswith("."):
            shutil.rmtree(staging)
    return target
