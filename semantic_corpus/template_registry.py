"""Append validated sources to the shared bank and atomically commit their ranges."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import zipfile

from .template_bank import POOL_FORMAT, TemplateBank, bank_files, fingerprint
from .template_validation import validate_bank

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_FORMAT = "built-in-template-component-v1"
_BANK_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")


def _registry_root(repo_root):
    return (Path(repo_root) if repo_root is not None else REPO_ROOT) / "data" / "mined-templates"


def _validated_report(bank, report=None):
    report = report if report is not None else json.loads((bank.path / "validation.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError('validation report must be a JSON object')
    source = bank.manifest.get("source_sha256")
    if (not isinstance(source, str) or re.fullmatch(r"[0-9a-f]{64}", source) is None
            or report.get("status") != "passed" or report.get("source_sha256") != source):
        raise ValueError("a passed validation report for this bank's source is required")
    for observed, declared in (("source_records", "compiled_records"),
                               ("templates", "templates"), ("token_spans", "tokens_audited")):
        actual = report.get(observed)
        if type(actual) is not int or actual < 0 or actual != bank.manifest["counts"].get(declared):
            raise ValueError(f"validation report does not match bank count {declared!r}")
    return report


def _publication(bank, report, bank_id):
    return dict(format=PUBLICATION_FORMAT, bank_id=bank_id,
                source_sha256=bank.manifest["source_sha256"],
                manifest_sha256=fingerprint(bank.manifest), validation_sha256=fingerprint(report))


def _pool_manifest(entries):
    counts, validation = Counter(), Counter()
    for entry in entries:
        counts.update({k: v for k, v in entry["manifest"]["counts"].items() if type(v) is int})
        validation.update({k: v for k, v in entry["validation"].items() if type(v) is int})
    identity = fingerprint([e["publication"] for e in entries])
    return dict(format=POOL_FORMAT, status="complete", sources=entries, source_sha256=identity,
                counts=dict(counts), validation=dict(status="passed", source_sha256=identity, **validation))


def validate_pool_manifest(manifest):
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("the shared bank must record its sources")
    seen = set()
    for entry in sources:
        try:
            identity = entry["id"]
            if not isinstance(identity, str) or not _BANK_ID.fullmatch(identity) or identity in seen:
                raise ValueError("invalid or repeated source ID")
            seen.add(identity)
            declared = entry["publication"]
            if (declared.get("format") != PUBLICATION_FORMAT or declared.get("bank_id") != identity
                    or declared.get("manifest_sha256") != fingerprint(entry["manifest"])
                    or declared.get("source_sha256") != entry["manifest"]["source_sha256"]
                    or declared.get("validation_sha256") != fingerprint(entry["validation"])
                    or entry["validation"].get("status") != "passed" or not isinstance(entry["files"], dict)):
                raise ValueError("invalid source publication")
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("invalid source publication") from error
    if manifest != _pool_manifest(sources):
        raise ValueError("shared bank counts or source identity do not match its publications")


def registered_banks(repo_root=None):
    """Compatibility API: source views within the single shared directory."""
    root = _registry_root(repo_root)
    if not (root / "manifest.json").is_file():
        return ()
    bank = TemplateBank(root)
    return bank.components() if bank.manifest["format"] == POOL_FORMAT else ()


def publication_record(bank_id, *, repo_root=None):
    root = _registry_root(repo_root)
    if not (root / "manifest.json").is_file():
        return None
    bank = TemplateBank(root)
    return next((entry for entry in bank.manifest.get("sources", ()) if entry["id"] == bank_id), None)


@contextmanager
def _writer_lock(root):
    # Never unlink the lock: otherwise workers could lock different file objects.
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / ".mined-templates.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        deadline = time.monotonic() + 60
        while True:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("another worker is updating the mined bank")
                time.sleep(0.1)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _commit(root, entries):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                     prefix=".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(_pool_manifest(entries), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, root / "manifest.json")
    finally:
        temporary.unlink(missing_ok=True)


def _append_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as sink, source.open("rb") as stream:
        offset = sink.tell()
        shutil.copyfileobj(stream, sink, length=1024 * 1024)
        sink.flush()
        os.fsync(sink.fileno())
        return [offset, sink.tell() - offset]


def register_bank(bank_path, source_path, *, repo_root=None, bank_id, before_publish=None,
                  archive_source=True):
    """Validate and append; return the shared directory. Repeated IDs are idempotent.

    Existing file ranges are never rewritten. Readers use committed byte ranges,
    so interrupted tails remain invisible, including during a retry.
    archive_source=False still validates every source record, but omits the
    duplicate input/attachment archive for a one-time, self-contained import.
    """
    if not isinstance(bank_id, str) or not _BANK_ID.fullmatch(bank_id):
        raise ValueError("bank_id must contain only lowercase letters, digits, '_' or '-'")
    bank = TemplateBank(bank_path)
    if bank.manifest["format"] == POOL_FORMAT:
        raise ValueError("register a newly compiled source, not the shared bank itself")
    destination = _registry_root(repo_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Resolve an existing parent, not a path another worker is just creating.
    # Python 3.10 on Windows can otherwise alternate C:\ and \\?\C:\ forms.
    root = destination.parent.resolve() / destination.name
    if root.is_symlink():
        raise ValueError('the shared bank directory must not be a symlink')
    with tempfile.TemporaryDirectory(prefix=".mined-templates-", suffix=".staging", dir=root.parent) as tmp:
        staging = Path(tmp)
        shutil.copytree(bank.path, staging / "bank")
        shutil.copyfile(source_path, staging / "source.jsonl")
        copied = TemplateBank(staging / "bank")
        if copied.manifest != bank.manifest:
            raise ValueError("bank manifest changed during publication")
        report = _validated_report(copied, validate_bank(copied, staging / "source.jsonl"))
        publication = _publication(copied, report, bank_id)
        with _writer_lock(root):
            entries, legacy = [], None
            if (root / "manifest.json").is_file():
                current = TemplateBank(root)
                if current.manifest["format"] == POOL_FORMAT:
                    current.components()  # Reject damaged component metadata on retries too.
                    entries = current.manifest["sources"]
                else:
                    legacy = current
                    previous = _validated_report(current)
                    entries = [dict(id=current.source_id, source_label=current.source_label,
                                    manifest=current.manifest, validation=previous, files=current.files,
                                    publication=_publication(current, previous, current.source_id))]
            elif root.exists() and any(not p.name.startswith('.') and p.is_dir()
                                       and (p / 'manifest.json').is_file() for p in root.iterdir()):
                raise ValueError("migrate legacy per-job banks before adding resources")
            existing = next((e for e in entries if e["id"] == bank_id), None)
            if existing:
                if (existing["publication"] != publication
                        or bool(existing.get("source_file")) != archive_source):
                    raise FileExistsError(f"a different template bank is already registered as {bank_id!r}")
                return root
            if before_publish is not None:
                before_publish()
            root.mkdir(exist_ok=True)
            if legacy:
                # Seal legacy lengths BEFORE touching data. Readers retry if
                # this manifest transition raced their initial file snapshot.
                _commit(root, entries)
            files = {p.relative_to(copied.path).as_posix():
                     _append_file(p, root / p.relative_to(copied.path)) for p in bank_files(copied.path)}
            if archive_source:
                sources = root / "sources"
                sources.mkdir(exist_ok=True)
                source_name = f"sources/{bank_id}.jsonl"
                os.replace(staging / "source.jsonl", root / source_name)
                # Licenses, documentation and original metadata remain auditable.
                archive = staging / "audit.zip"
                with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as audit:
                    for path in sorted(copied.path.rglob("*")):
                        if path.is_file() and path.relative_to(copied.path).as_posix() not in files:
                            audit.write(path, path.relative_to(copied.path).as_posix())
                os.replace(archive, sources / f"{bank_id}.zip")
            if before_publish is not None:
                before_publish()
            entry = dict(id=bank_id, manifest=copied.manifest, validation=report, publication=publication,
                         files=files)
            if archive_source:
                entry['source_file'] = source_name
            _commit(root, [*entries, entry])
    return root
