"""Keep unapproved resources in the shared files without enabling generation.

pending.json records immutable byte ranges, just as manifest.json does for
published sources. Publishing another source never rewrites these ranges.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

from .template_bank import POOL_FORMAT, TemplateBank, bank_files, fingerprint
from .template_registry import _BANK_ID, _append_file, _validated_report, _writer_lock

FORMAT = 'mined-pending-resources-v1'


def shared_name(name):
    """Use the existing inventory name for the older hyphenated spelling."""
    return 'entity_candidates.jsonl.gz' if name == 'entity-candidates.jsonl.gz' else name


def range_digest(path, offset=0, size=None):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        stream.seek(offset)
        remaining = size
        while remaining is None or remaining:
            block = stream.read(1024 * 1024 if remaining is None else min(remaining, 1024 * 1024))
            if not block:
                if remaining:
                    raise ValueError('Truncated pending resource: ' + str(path))
                break
            checksum.update(block)
            if remaining is not None:
                remaining -= len(block)
    return checksum.hexdigest()


def _safe_name(name):
    if (not isinstance(name, str) or '\\' in name or ':' in name
            or PurePosixPath(name).is_absolute() or any(p in ('', '.', '..') for p in name.split('/'))
            or not name.endswith('.jsonl.gz')):
        raise ValueError('Invalid pending resource path')


def load_pending(root):
    root = Path(root)
    path = root / 'pending.json'
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding='utf8'))
    if data.get('format') != FORMAT or not isinstance(data.get('sources'), list):
        raise ValueError('Invalid pending resources index')
    seen = set()
    for entry in data['sources']:
        if (not _BANK_ID.fullmatch(entry['id']) or entry['id'] in seen
                or entry.get('review_status') != 'UNREVIEWED'
                or entry.get('generation_enabled') is not False
                or entry.get('kind') not in {'templates', 'inventories'}):
            raise ValueError('Invalid pending resource source')
        seen.add(entry['id'])
        expected = {k: v for k, v in entry.items() if k != 'sha256'}
        if fingerprint(expected) != entry.get('sha256'):
            raise ValueError('Pending resource metadata changed')
        if set(entry['files']) != set(entry['file_sha256']):
            raise ValueError('Pending file checksums are incomplete')
        for name, extent in entry['files'].items():
            _safe_name(name)
            if (not isinstance(extent, list) or len(extent) != 2
                    or any(type(v) is not int or v < 0 for v in extent)
                    or not (root / name).resolve().is_relative_to(root.resolve())):
                raise ValueError('Invalid pending resource extent')
        if entry['kind'] == 'templates':
            TemplateBank(root, _entry=entry)
    return data['sources']


def verify_pending(root):
    root = Path(root)
    entries = load_pending(root)
    for entry in entries:
        for name, (offset, size) in entry['files'].items():
            if range_digest(root / name, offset, size) != entry['file_sha256'][name]:
                raise ValueError('Pending resource bytes changed: ' + name)
    return entries


def pending_banks(root):
    """Explicit audit access; ordinary components()/generation exclude these."""
    return tuple(TemplateBank(root, _entry=e) for e in load_pending(root) if e['kind'] == 'templates')


def store_pending(source, root, *, source_id, inventories=False):
    """Append exact gzip members and atomically record their unapproved status."""
    source, root = Path(source).resolve(), Path(root).resolve()
    if not _BANK_ID.fullmatch(source_id) or source == root:
        raise ValueError('Invalid pending source')
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf8'))
    files = list(source.glob('*.jsonl.gz')) if inventories else bank_files(source)
    if not files:
        raise ValueError('No pending resource files')
    proof = {shared_name(p.relative_to(source).as_posix()): range_digest(p) for p in files}
    if len(proof) != len(files):
        raise ValueError('Conflicting inventory filenames')
    identity = dict(id=source_id, kind='inventories' if inventories else 'templates',
                    review_status='UNREVIEWED', generation_enabled=False,
                    manifest=manifest, file_sha256=proof)
    if not inventories:
        bank = TemplateBank(source)
        if bank.manifest['format'] == POOL_FORMAT:
            raise ValueError('Expected a standalone source')
        identity['validation'] = _validated_report(bank)
    with _writer_lock(root):
        current = TemplateBank(root)
        if current.manifest['format'] != POOL_FORMAT:
            raise ValueError('Seal the shared bank before adding pending ranges')
        if any(e['id'] == source_id for e in current.manifest['sources']):
            raise ValueError('Pending source ID is already published')
        entries = load_pending(root)
        existing = next((e for e in entries if e['id'] == source_id), None)
        if existing:
            if {k: v for k, v in existing.items() if k not in {'files', 'sha256'}} != identity:
                raise FileExistsError('A different pending source has this ID')
            verify_pending(root)
            return existing
        extents = {}
        for path in sorted(files):
            name = shared_name(path.relative_to(source).as_posix())
            _safe_name(name)
            if not (root / name).resolve().is_relative_to(root):
                raise ValueError('Pending resource escapes the shared bank')
            offset, size = _append_file(path, root / name)
            if range_digest(root / name, offset, size) != proof[name]:
                raise ValueError('Pending source changed during import')
            extents[name] = [offset, size]
        entry = {**identity, 'files': extents}
        entry['sha256'] = fingerprint(entry)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', dir=root,
                                             prefix='.pending-', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(dict(format=FORMAT, sources=[*entries, entry]), stream, ensure_ascii=False, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, root / 'pending.json')
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return entry
