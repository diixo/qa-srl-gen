"""Prepare text sources and run validated template mining in a worker process."""

import gzip
import io
import json
import uuid
from pathlib import Path

from django.conf import settings

from .catalog import data_path


def source_file_suffix(name):
    for suffix in ('.jsonl.gz', '.jsonl', '.txt'):
        if name.lower().endswith(suffix):
            return suffix
    raise ValueError('Choose a UTF-8 .txt, .jsonl or .jsonl.gz file.')


def resolve_source_path(value):
    path = data_path(value)
    source_file_suffix(path.name)
    return path


def save_source_upload(uploaded):
    """Keep a private copy for the worker and retries, streaming large uploads."""
    suffix = source_file_suffix(uploaded.name)
    folder = settings.CORPUS_UI_ROOT / 'uploads'
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f'template-mining-{uuid.uuid4().hex}{suffix}'
    stream = destination.open('xb')
    try:
        with stream:
            for chunk in uploaded.chunks():
                stream.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination.resolve()


def prepare_source(config, destination, *, max_length, progress=None):
    """Stream all records into an auditable snapshot, without trimming their text."""
    pasted = config.get('text', '')
    if not isinstance(pasted, str):
        raise ValueError('English text must be a string.')
    path = resolve_source_path(config['source_path']) if config.get('source_path') else None
    counts = {'records': 0, 'pasted_records': 0, 'file_records': 0}

    with Path(destination).open('x', encoding='utf-8', newline='\n') as sink:
        def append(text, origin, line, original_idx=None):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'{origin}, line {line}: expected a non-empty example or text string.')
            if len(text) > max_length:
                raise ValueError(f'{origin}, line {line}: {len(text):,} characters exceeds the '
                                 f'model limit of {max_length:,}. Split this record into shorter records.')
            row = {'idx': counts['records'], 'example': text,
                   'origin': {'source': origin, 'line': line, 'idx': original_idx}}
            sink.write(json.dumps(row, ensure_ascii=False) + '\n')
            counts['records'] += 1
            if progress and counts['records'] % 1000 == 0:
                progress('Preparing source snapshot', counts['records'])

        for line, text in enumerate(io.StringIO(pasted.lstrip('\ufeff'), newline=None), 1):
            text = text.rstrip('\r\n')
            if text.strip():
                append(text, 'pasted text', line)
                counts['pasted_records'] += 1
        if path is not None:
            opener = gzip.open if path.name.lower().endswith('.gz') else open
            is_json = path.name.lower().endswith(('.jsonl', '.jsonl.gz'))
            with opener(path, 'rt', encoding='utf-8-sig') as source:
                for line, raw in enumerate(source, 1):
                    if not raw.strip():
                        continue
                    if is_json:
                        try:
                            row = json.loads(raw)
                        except ValueError as error:
                            raise ValueError(f'{path}, line {line}: invalid JSON: {error}') from error
                        if not isinstance(row, dict):
                            raise ValueError(f'{path}, line {line}: expected a JSON object.')
                        append(row.get('example', row.get('text')), str(path), line, row.get('idx'))
                    else:
                        append(raw.rstrip('\r\n'), str(path), line)
                    counts['file_records'] += 1
    if not counts['records']:
        raise ValueError('The inputs contain no non-empty records.')
    return {**counts, 'source_path': str(path) if path else ''}


def load_mining_model():
    """Resolve locally installed resources; never download a model in a job."""
    try:
        import spacy
    except ImportError as error:
        raise ValueError('Template mining requires requirements-mining.txt and an English spaCy model.') from error

    model = getattr(settings, 'CORPUS_MINING_MODEL', '')
    if model:
        candidate = settings.REPO_ROOT / model
        model = str(candidate) if candidate.is_dir() else model
    elif spacy.util.is_package('en_core_web_sm'):
        model = 'en_core_web_sm'
    else:
        candidates = sorted((settings.REPO_ROOT / 'artifacts/models').glob(
            'spacy-en-*/en_core_web_sm/en_core_web_sm-*'))
        model = next((str(path) for path in reversed(candidates)
                      if (path / 'config.cfg').is_file() and (path / 'meta.json').is_file()), '')
    if not model:
        raise ValueError('No English spaCy model found. Set CORPUS_MINING_MODEL to an installed '
                         'package name or a local model directory, then restart the worker.')
    try:
        nlp = spacy.load(model, disable=['ner'])
    except (OSError, ValueError) as error:
        raise ValueError(f'Cannot load mining model {model!r}: {error}') from error
    if nlp.lang != 'en' or not {'parser', 'lemmatizer'} <= set(nlp.pipe_names):
        raise ValueError('Template mining requires an English model with parser and lemmatizer.')
    return nlp


def execute_template_mining(job, out, config, progress):
    from semantic_corpus.template_compiler import build_bank
    from semantic_corpus.template_registry import publication_record, register_bank

    from .models import Job

    progress('Loading English syntax model')
    nlp = load_mining_model()
    progress('Preparing source snapshot')
    source = out / 'source.jsonl'
    inputs = prepare_source(config, source, max_length=nlp.max_length, progress=progress)
    total = inputs['records']
    wordnet = settings.REPO_ROOT / 'artifacts/models/wordnet.zip'
    manifest = build_bank(
        source, out / 'bank', out / 'work', nlp,
        wiktionary=settings.REPO_ROOT / 'data/wiktionary',
        wordnet=wordnet if wordnet.is_file() else None,
        batch_size=32,
        progress=lambda phase, processed: progress(phase, processed, total),
    )

    def check_cancelled():
        if Job.objects.filter(pk=job.pk, cancel_requested=True).exists():
            raise ValueError('Cancellation requested before adding resources to Built-in pool.')

    check_cancelled()
    # Adding validated resources to the shared pool is part of every mining job,
    # including queued or retried jobs with the former publish_to_pool=False flag.
    progress('Validating and adding to Built-in pool', total, total)
    published = register_bank(out / 'bank', source, repo_root=settings.REPO_ROOT,
                              bank_id=str(job.pk), before_publish=check_cancelled)
    validation = publication_record(str(job.pk), repo_root=settings.REPO_ROOT)['validation']
    (out / 'bank/validation.json').write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary = {'inputs': inputs, 'counts': manifest['counts'], 'validation': validation,
               'review_status': 'UNREVIEWED', 'published_to_pool': True,
               'pool_path': str(published),
               'note': 'Mined syntax needs semantic annotation before QA and training export.'}
    (out / 'mining_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    progress('Built-in pool updated', total, total,
             templates=manifest['counts']['templates'], published_to_pool=True)
