"""Lossless DailyDialog release archive; never delete sources or publish candidates.

ZIP volumes contain SHA-256-addressed chunks. The manifest maps every original
repository-relative path to its exact bytes, including duplicate historical
ledgers. Only the Python standard library is needed to verify or restore it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'dailydialog-checkpoint-v1'
CHUNK_SIZE = 8 * 1024 * 1024
VOLUME_SIZE = 48 * 1024 * 1024
HEX = re.compile(r'[0-9a-f]{64}\Z')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b''):
            value.update(chunk)
    return value.hexdigest()


def safe_path(root, name):
    relative = PurePosixPath(name)
    if (not name or '\\' in name or ':' in name or '\x00' in name
            or relative.is_absolute() or any(p in {'.', '..'} for p in name.split('/'))):
        raise ValueError('Unsafe archive path: ' + name)
    result = root.joinpath(*relative.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError('Archive path escapes destination: ' + name)
    return result


def collect_release(repo):
    """Archive the current candidate and the actual inputs needed by open_overlay."""
    selected = {}

    def add(path, role):
        if not path.is_file():
            raise ValueError('Missing release input: ' + str(path))
        relative = path.resolve().relative_to(repo.resolve()).as_posix()
        selected.setdefault(relative, set()).add(role)

    def tree(path, role, excluded=()):
        for file in sorted(path.rglob('*')):
            if file.is_file() and not set(file.relative_to(path).parts) & set(excluded):
                add(file, role)

    reviews = repo / 'data/dailydialog/review-parts'
    progress = json.loads((reviews / 'hybrid-luna-offline-progress-quality-v38.json').read_text(encoding='utf8'))
    candidate = repo / progress['candidate']
    tree(reviews, 'reviews_and_annotations')
    tree(candidate, 'current_candidate_unreviewed')
    tree(repo / 'data/dailydialog/filtered', 'corrected_source')
    chain, current = [], candidate.resolve()
    while True:
        if current in chain:
            raise ValueError('Cyclic candidate chain')
        chain.append(current)
        add(current / 'plan.json', 'overlay_dependency')
        plan = json.loads((current / 'plan.json').read_text(encoding='utf8'))
        if 'lexical_resources' in plan:
            original = Path(plan['source'])
            # Root files include the two required databases and their provenance.
            for directory in (original, current):
                for file in sorted(directory.iterdir()):
                    if file.is_file():
                        add(file, 'overlay_dependency')
            break
        current = Path(plan['source'])
    quality = repo / 'artifacts/dailydialog-quality-v37'
    tree(quality, 'validation_and_admission', excluded=(
        'gpt2-sft', 'extraction-pilot', 'evaluation-existing-extraction',
        'evaluation-pretrained', 'evaluation-sft'))
    tree(repo / progress['dataset']['path'], 'generated_dataset')
    tree(repo / 'artifacts/dailydialog-sentence-pilot', 'manual_pilot_history')
    # Original request/response journal, accounting and stop state, without SQLite
    # shared-memory files. A nonempty WAL must be resolved before taking a snapshot.
    api = repo / 'artifacts/dailydialog-luna-medium-bank-v2'
    for wal in api.glob('*-wal'):
        if wal.stat().st_size:
            raise ValueError('Live/uncheckpointed SQLite WAL: ' + str(wal))
    for file in sorted(api.iterdir()):
        if file.is_file() and not file.name.endswith(('-wal', '-shm')):
            add(file, 'paid_api_journal_no_restart')
    # Preserve one-off verification/preparation helpers that otherwise stay ignored.
    helpers = ['verify_quality_v37.py', 'prepare_admission_v37.py', 'quality_pairs_v37.py',
               'quality_readings_v37.py', 'quality_decisions_v37.py', 'review_quality_v37.py',
               'prepare_quality_v37.py', 'finish_quality_rules_v37.py']
    for name in helpers:
        add(repo / 'artifacts' / name, 'release_helpers')
    # The working pool is committed normally; bind the checkpoint to that revision.
    required = {}
    for directory in ('data/mined-templates', 'semantic_corpus', 'scripts'):
        for file in sorted((repo / directory).rglob('*')):
            if file.is_file() and ('__pycache__' not in file.parts) and file.suffix != '.pyc':
                relative = file.relative_to(repo).as_posix()
                if relative == 'scripts/evaluate_dailydialog_extraction_pilot.py':
                    continue
                required[relative] = digest(file)
    return selected, dict(
        source_root=str(repo.resolve()), candidate=candidate.relative_to(repo).as_posix(),
        candidate_status='UNREVIEWED', candidate_published=False,
        overlay_chain=[p.relative_to(repo).as_posix() for p in chain],
        lexical_revision=current.relative_to(repo).as_posix(),
        original_cache=(original / 'cache.sqlite3').relative_to(repo).as_posix(),
        working_tree_fingerprints=required,
        limitations=[
            'Historical candidate banks/SQLite revisions other than the current candidate and base inputs are not archived.',
            'Historical absolute paths are preserved verbatim. Existing revision tools expect the original workspace path.',
            'API journal is evidence, not permission to restart requests.',
            'Training checkpoints and model evaluation experiments are excluded.',
        ])


def create(repo, out, selected, metadata=None, *, chunk_size=CHUNK_SIZE, volume_size=VOLUME_SIZE):
    if chunk_size <= 0 or volume_size < chunk_size + 2048:
        raise ValueError('Invalid archive chunk/volume size')
    out.mkdir(parents=True, exist_ok=False)
    objects, files, volumes = {}, [], []
    archive = None
    try:
        for number, (name, roles) in enumerate(sorted(selected.items()), 1):
            path = safe_path(repo, name)
            before = path.stat()
            checksum, chunks = hashlib.sha256(), []
            with path.open('rb') as stream:
                for data in iter(lambda: stream.read(chunk_size), b''):
                    checksum.update(data)
                    key = hashlib.sha256(data).hexdigest()
                    chunks.append(key)
                    if key in objects:
                        continue
                    if (archive is None or archive.fp.tell() + len(data)
                            + 256 * len(archive.filelist) + 2048 > volume_size):
                        if archive is not None:
                            archive.close()
                        volume = f'pack-{len(volumes) + 1:04d}.zip'
                        volumes.append(volume)
                        archive = zipfile.ZipFile(out / volume, 'x', zipfile.ZIP_DEFLATED, compresslevel=6)
                    info = zipfile.ZipInfo(key, date_time=(2026, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, data, compresslevel=6)
                    objects[key] = dict(archive=volumes[-1], size=len(data))
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Source changed during snapshot: ' + name)
            files.append(dict(path=name, size=after.st_size, sha256=checksum.hexdigest(),
                              chunks=chunks, roles=sorted(roles)))
            if number % 200 == 0:
                print(f'Archived {number}/{len(selected)} files', flush=True)
    finally:
        if archive is not None:
            archive.close()
    archives = {name: dict(size=(out / name).stat().st_size, sha256=digest(out / name)) for name in volumes}
    result = dict(format=FORMAT, metadata=metadata or {}, files=files, objects=objects,
                  archives=archives, totals=dict(files=len(files), chunks=len(objects),
                  source_bytes=sum(f['size'] for f in files),
                  unique_chunk_bytes=sum(o['size'] for o in objects.values()),
                  archive_bytes=sum(a['size'] for a in archives.values())))
    (out / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return result


def load(bundle):
    result = json.loads((bundle / 'manifest.json').read_text(encoding='utf8'))
    if result.get('format') != FORMAT:
        raise ValueError('Unsupported checkpoint format')
    seen = set()
    for file in result['files']:
        safe_path(bundle, file['path'])
        folded = file['path'].casefold()
        if folded in seen or not HEX.fullmatch(file['sha256']):
            raise ValueError('Duplicate path or invalid file checksum')
        seen.add(folded)
        if sum(result['objects'][key]['size'] for key in file['chunks']) != file['size']:
            raise ValueError('Invalid file size')
    for key, obj in result['objects'].items():
        if not HEX.fullmatch(key) or obj['size'] <= 0 or obj['archive'] not in result['archives']:
            raise ValueError('Invalid chunk metadata')
    for name in result['archives']:
        safe_path(bundle, name)
        if not re.fullmatch(r'pack-[0-9]{4}\.zip', name):
            raise ValueError('Invalid volume name')
    return result


class Reader:
    def __init__(self, bundle, manifest):
        self.bundle, self.manifest, self.handles = bundle, manifest, {}

    def chunk(self, key):
        obj = self.manifest['objects'][key]
        name = obj['archive']
        if name not in self.handles:
            self.handles[name] = zipfile.ZipFile(safe_path(self.bundle, name))
        data = self.handles[name].read(key)
        if len(data) != obj['size'] or hashlib.sha256(data).hexdigest() != key:
            raise ValueError('Chunk checksum mismatch: ' + key)
        return data

    def close(self):
        for handle in self.handles.values():
            handle.close()


def verify(bundle, sources=None):
    manifest = load(bundle)
    for name, proof in manifest['archives'].items():
        path = safe_path(bundle, name)
        if path.stat().st_size != proof['size'] or digest(path) != proof['sha256']:
            raise ValueError('Volume checksum mismatch: ' + name)
    reader = Reader(bundle, manifest)
    try:
        for file in manifest['files']:
            checksum = hashlib.sha256()
            for key in file['chunks']:
                checksum.update(reader.chunk(key))
            if checksum.hexdigest() != file['sha256']:
                raise ValueError('File checksum mismatch: ' + file['path'])
            if sources is not None:
                path = safe_path(sources, file['path'])
                if path.stat().st_size != file['size'] or digest(path) != file['sha256']:
                    raise ValueError('Source differs from snapshot: ' + file['path'])
    finally:
        reader.close()
    return dict(status='passed', **manifest['totals'], compared_to_sources=sources is not None)


def restore(bundle, destination, *, prefix=''):
    manifest = load(bundle)
    if prefix:
        safe_path(destination, prefix.rstrip('/'))
    selected = [f for f in manifest['files'] if not prefix or f['path'] == prefix.rstrip('/')
                or f['path'].startswith(prefix.rstrip('/') + '/')]
    if not selected:
        raise ValueError('No files selected')
    # Check all conflicts before creating anything; never overwrite local changes.
    pending = []
    for file in selected:
        target = safe_path(destination, file['path'])
        if target.exists():
            if not target.is_file() or digest(target) != file['sha256']:
                raise ValueError('Destination differs; will not overwrite: ' + file['path'])
        else:
            pending.append((file, target))
    verify(bundle)
    reader = Reader(bundle, manifest)
    try:
        for file, target in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                for key in file['chunks']:
                    stream.write(reader.chunk(key))
            if digest(target) != file['sha256']:
                raise ValueError('Restored file checksum mismatch: ' + file['path'])
    finally:
        reader.close()
    return dict(status='passed', restored=len(pending), already_identical=len(selected) - len(pending))


def consolidate(repo, bundle):
    """One-time import into the common pool; candidate records stay unpublished."""
    manifest = load(bundle)
    pool = 'data/mined-templates'
    target = pool + '/audit/dailydialog-v38'
    state_path = repo / target / 'import.json'
    if state_path.exists() or (repo / pool / 'audit/import.json').exists():
        raise ValueError('Import is already recorded; use its audit, not a repeated import')
    reviews = 'data/dailydialog/review-parts'
    candidate_source = manifest['metadata']['candidate'] + '/bank/'
    candidate_target = pool + '/candidates/dailydialog-v38'
    register_path = repo / pool / 'reviewed/dailydialog-quarantine.json'
    register_bytes = register_path.read_bytes()
    register = json.loads(register_bytes)
    mapping = {}
    for file in manifest['files']:
        name = file['path']
        new = None
        if name.startswith(candidate_source):
            new = candidate_target + '/' + name[len(candidate_source):]
        elif name.startswith(reviews + '/annotations/'):
            new = target + '/annotations/' + name.rsplit('/', 1)[1]
        elif name.startswith(reviews + '/luna-offline-quality-v38/'):
            new = target + '/ledger/' + name.rsplit('/', 1)[1]
        elif name.startswith('data/dailydialog/filtered/'):
            new = target + '/source/' + name.rsplit('/', 1)[1]
        elif PurePosixPath(name).parent.as_posix() == reviews:
            base = PurePosixPath(name).name
            if not base.startswith('part-') and base != 'README.md':
                new = target + '/decisions/' + base
        if name in register['evidence'] and name.startswith(reviews + '/'):
            new = target + '/evidence/' + name.rsplit('/', 1)[1]
        if new is not None:
            if new in mapping:
                raise ValueError('Duplicate imported destination: ' + new)
            mapping[new] = file
    archived = {file['path'] for file in mapping.values()}
    for name in register['evidence']:
        if name.startswith(reviews + '/') and name not in archived:
            raise ValueError('Missing quarantine proof in archive: ' + name)
    if not mapping or candidate_target + '/manifest.json' not in mapping:
        raise ValueError('Candidate missing from archive')
    reader = Reader(bundle, manifest)
    try:
        for name, file in sorted(mapping.items()):
            path = safe_path(repo, name)
            if path.exists():
                if digest(path) != file['sha256']:
                    raise ValueError('Imported destination differs: ' + name)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                for chunk in file['chunks']:
                    stream.write(reader.chunk(chunk))
            if digest(path) != file['sha256']:
                raise ValueError('Imported file checksum mismatch: ' + name)
    finally:
        reader.close()
    candidate_manifest = json.loads((repo / candidate_target / 'manifest.json').read_text(encoding='utf8'))
    if candidate_manifest.get('hybrid_import', {}).get('review_status') != 'UNREVIEWED':
        raise ValueError('Expected an unreviewed candidate; do not change its admission state')
    # Preserve the exact old register, then rebind only storage locations.
    backup = repo / target / 'evidence/quarantine-register-before-import.json'
    with backup.open('xb') as stream:
        stream.write(register_bytes)
    original_evidence = register['evidence']
    register['evidence'] = {
        (target + '/evidence/' + name.rsplit('/', 1)[1] if name.startswith(reviews + '/') else name): sha
        for name, sha in original_evidence.items()
    }
    register['import_state'] = target + '/import.json'
    register['historical_candidate'] = register['candidate']
    register['candidate'] = candidate_target
    state = dict(format='dailydialog-import-state-v1', review_parts_required=False,
                 candidate=candidate_target, candidate_status='UNREVIEWED', candidate_published=False,
                 source=target + '/source', annotations=target + '/annotations',
                 ledger=target + '/ledger', decisions=target + '/decisions',
                 checkpoint=bundle.resolve().relative_to(repo.resolve()).as_posix(),
                 checkpoint_manifest_sha256=digest(bundle / 'manifest.json'),
                 pool_manifest_sha256=digest(repo / pool / 'manifest.json'),
                 previous_register_sha256=hashlib.sha256(register_bytes).hexdigest(),
                 files={name: dict(source=file['path'], sha256=file['sha256'], size=file['size'])
                        for name, file in sorted(mapping.items())},
                 note='One-time import. Archive paths are historical provenance, not runtime dependencies. '
                      'Only the existing registered components may generate data; candidate and audit folders are not registered.')
    register_path.write_text(json.dumps(register, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
    return dict(status='imported', files=len(mapping), state=state_path.relative_to(repo).as_posix(),
                candidate_published=False, originals_deleted=False)


def merge_pool(repo):
    """Replace the initial copied candidate directories with shared file ranges."""
    # Local import also permits invoking this utility as a standalone script.
    import sys
    sys.path.insert(0, str(repo))
    from semantic_corpus.template_pending import shared_name, store_pending, verify_pending

    pool_name = 'data/mined-templates'
    old_name = pool_name + '/audit/dailydialog-v38'
    new_name = pool_name + '/audit'
    old, target = safe_path(repo, old_name), safe_path(repo, new_name)
    if not old.exists():
        state = json.loads((target / 'import.json').read_text(encoding='utf8'))
        if state.get('candidate_source_id') != 'dailydialog':
            raise ValueError('Unexpected shared pool import state')
        verify_pending(repo / pool_name)
        return dict(status='already_merged')
    state_bytes = (old / 'import.json').read_bytes()
    state = json.loads(state_bytes)
    register_path = repo / pool_name / 'reviewed/dailydialog-quarantine.json'
    register_bytes = register_path.read_bytes()
    register = json.loads(register_bytes)
    candidates = repo / pool_name / 'candidates'
    backup = safe_path(repo, 'artifacts/dailydialog-pool-layout-before-merge/candidates')
    if backup.exists():
        raise ValueError('Layout backup already exists; inspect before continuing')
    # Preflight every directory move. Never replace a user's existing file.
    moves = [(item, target / item.name) for item in old.iterdir()]
    for source, destination in [*moves, (candidates, backup)]:
        if (not source.resolve().is_relative_to(repo.resolve())
                or not destination.resolve().is_relative_to(repo.resolve()) or destination.exists()):
            raise ValueError('Unsafe or conflicting layout destination: ' + str(destination))
    entries = {
        pool_name + '/candidates/dailydialog-v38': store_pending(
            candidates / 'dailydialog-v38', repo / pool_name, source_id='dailydialog'),
        pool_name + '/candidates/dailydialog-resources': store_pending(
            candidates / 'dailydialog-resources', repo / pool_name,
            source_id='dailydialog-resources', inventories=True),
    }
    verify_pending(repo / pool_name)
    archive = load(old / 'checkpoint')
    archived = {f['path']: f for f in archive['files']}
    files, ranges, archive_proofs = {}, {}, {}
    for name, proof in state['files'].items():
        candidate = next((prefix for prefix in entries if name.startswith(prefix + '/')), None)
        if candidate is None:
            files[name.replace(old_name + '/', new_name + '/', 1)] = proof
            continue
        relative = shared_name(name[len(candidate) + 1:])
        entry = entries[candidate]
        if relative in entry['files']:
            if proof['sha256'] != entry['file_sha256'][relative]:
                raise ValueError('Source differs from the stored range: ' + name)
            ranges[proof['source']] = dict(source_id=entry['id'], file=relative,
                sha256=proof['sha256'], size=proof['size'])
        elif proof['source'] in archived:
            if archived[proof['source']]['sha256'] != proof['sha256']:
                raise ValueError('Source differs from its archive: ' + name)
            archive_proofs[proof['source']] = dict(sha256=proof['sha256'], size=proof['size'])
        elif candidate.endswith('dailydialog-resources') and relative == 'manifest.json':
            # Preserve exact bytes of the later inventory manifest too.
            name2 = new_name + '/evidence/resource-inventory-before-merge.json'
            dest = old / 'evidence/resource-inventory-before-merge.json'
            with dest.open('xb') as stream:
                stream.write((repo / name).read_bytes())
            files[name2] = proof
        else:
            raise ValueError('No durable copy for: ' + name)
    for name, data in [('import-before-merge.json', state_bytes),
                       ('quarantine-register-before-merge.json', register_bytes)]:
        with (old / 'evidence' / name).open('xb') as stream:
            stream.write(data)
    for field in ('source', 'annotations', 'ledger', 'decisions', 'checkpoint'):
        state[field] = state[field].replace(old_name, new_name, 1)
    state.update(candidate=pool_name, candidate_source_id='dailydialog',
                 pending_resource_source_id='dailydialog-resources', files=files,
                 shared_ranges=ranges, archived_files=archive_proofs,
                 note='One shared set of resource files. pending.json indexes unapproved ranges; '
                      'manifest.json continues to select the published generation components.')
    state.pop('resource_candidates', None)
    register['import_state'] = new_name + '/import.json'
    register['candidate'] = pool_name
    register['candidate_source_id'] = 'dailydialog'
    register['evidence'] = {name.replace(old_name + '/', new_name + '/', 1): sha
                            for name, sha in register['evidence'].items()}
    # All former directories remain recoverable in an ignored local backup;
    # the durable byte ranges and original archive have already been verified.
    for source, destination in moves:
        source.rename(destination)
    old.rmdir()
    backup.parent.mkdir(parents=True, exist_ok=True)
    candidates.rename(backup)
    for path, value in [(target / 'import.json', state), (register_path, register)]:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
    return dict(status='merged', shared_files=sum(len(e['files']) for e in entries.values()),
                audit=new_name, backup=backup.relative_to(repo).as_posix(), candidate_published=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('create')
    build.add_argument('--repo', type=Path, default=ROOT)
    build.add_argument('--out', type=Path, required=True)
    check = sub.add_parser('verify')
    check.add_argument('--bundle', type=Path, required=True)
    check.add_argument('--sources', type=Path)
    unpack = sub.add_parser('restore')
    unpack.add_argument('--bundle', type=Path, required=True)
    unpack.add_argument('--destination', type=Path, required=True)
    unpack.add_argument('--prefix', default='')
    merge = sub.add_parser('consolidate')
    merge.add_argument('--repo', type=Path, default=ROOT)
    merge.add_argument('--bundle', type=Path, required=True)
    shared = sub.add_parser('merge-pool')
    shared.add_argument('--repo', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == 'create':
        repo = args.repo.resolve()
        selection, metadata = collect_release(repo)
        result = create(repo, args.out, selection, metadata)['totals']
    elif args.command == 'verify':
        result = verify(args.bundle, args.sources)
    elif args.command == 'restore':
        result = restore(args.bundle, args.destination, prefix=args.prefix)
    elif args.command == 'consolidate':
        result = consolidate(args.repo.resolve(), args.bundle)
    else:
        result = merge_pool(args.repo.resolve())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
