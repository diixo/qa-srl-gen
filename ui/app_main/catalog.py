"""Lightweight read views: never construct a full CorpusStore during a GET."""

import json
from itertools import islice
from pathlib import Path

from django.conf import settings

from semantic_corpus.semantic_generator import DEFAULT_FRAMES, default_pool
from semantic_corpus.semantic_generator.substitutions import load_pool
from semantic_corpus.storage import TABLES


def data_path(value, *, directory=False):
    path = (settings.REPO_ROOT / value).resolve()
    roots = [(settings.REPO_ROOT / name).resolve() for name in ('data', 'artifacts')]
    roots.append(settings.CORPUS_UI_ROOT.resolve())
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError('The path must be inside data/ or artifacts/.')
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError('File or directory not found.')
    return path


def relative(path):
    try:
        return path.relative_to(settings.REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def job_root(job):
    return settings.CORPUS_UI_ROOT / 'jobs' / str(job.pk)


def read_json(path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def sources():
    rows = []
    for root in (settings.REPO_ROOT / 'data', settings.CORPUS_UI_ROOT / 'uploads'):
        rows.extend(dict(path=relative(p), size=p.stat().st_size) for p in sorted(root.rglob('*'))
                    if p.is_file() and p.name.endswith(('.txt', '.jsonl', '.jsonl.gz', '.qa'))
                    and p.resolve().is_relative_to(root.resolve()))
    return rows


def stores():
    found = {}
    for root in (settings.REPO_ROOT / 'artifacts', settings.CORPUS_UI_ROOT,
                 settings.REPO_ROOT / 'data'):
        for manifest in root.rglob('manifest.json'):
            try:
                path = data_path(relative(manifest.parent), directory=True)
                meta = read_json(manifest)
                if meta.get('store_format') != 'jsonl-v1':
                    continue
                found[str(path)] = dict(path=relative(path), created=meta.get('created_at'),
                    ontology=meta.get('ontology_version'),
                    size=sum((path / f).stat().st_size for f in TABLES.values() if (path / f).is_file()))
            except (OSError, ValueError, AttributeError):
                continue
    return sorted(found.values(), key=lambda row: row['path'])


def store_path(value):
    path = data_path(value, directory=True)
    if read_json(path / 'manifest.json', {}).get('store_format') != 'jsonl-v1':
        raise ValueError('Select a store with a jsonl-v1 manifest.json.')
    # A live or interrupted output is not a stable input to another job.
    from .models import Job
    for job in Job.objects.exclude(status=Job.Status.SUCCEEDED).only('id'):
        if path.is_relative_to(job_root(job).resolve()):
            raise ValueError('This job has not completed successfully.')
    return path


def table_page(path, table, offset=0, size=30):
    filename = path / TABLES[table]
    if not filename.is_file():
        return [], False
    with filename.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in islice(stream, offset, offset + size + 1)]
    return rows[:size], len(rows) > size


def pool_choices():
    root = settings.CORPUS_UI_ROOT / 'pools'
    return [('', 'Built-in pool')] + [(relative(p), p.stem) for p in sorted(root.glob('*.json'))]


def get_pool(value=''):
    if not value:
        return default_pool()
    if value not in dict(pool_choices()):
        raise ValueError('Unknown entity pool.')
    from semantic_corpus.ontology import load_default_hierarchy
    return load_pool(data_path(value), load_default_hierarchy())


def pool_monitor(value=''):
    pool = get_pool(value)
    partitions = {name: pool.split(name) for name in ('train', 'dev', 'test')}
    slots = []
    for frame in DEFAULT_FRAMES:
        for slot in frame.slots:
            counts = [len(part.matching(slot.types, slot.excludes)) for part in partitions.values()]
            slots.append(dict(frame=frame.lemma, name=slot.name, role=str(slot.role),
                types=', '.join(sorted(map(str, slot.types))),
                excludes=', '.join(sorted(map(str, slot.excludes))), optional=slot.optional,
                counts=counts, empty=any(c == 0 for c in counts)))
    return dict(pool=pool, slots=slots, frames=DEFAULT_FRAMES,
        splits=[dict(name=name, entities=len(part), forms=len(part.surface_forms))
                for name, part in partitions.items()], ambiguous=pool.ambiguous_forms())
