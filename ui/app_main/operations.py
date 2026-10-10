"""UI adapters around the existing corpus APIs; every write has a fresh output."""

import gzip
import json
import re
import runpy
import shutil
import sys
from dataclasses import replace
from contextlib import redirect_stdout
from itertools import islice
from time import monotonic

from django.conf import settings

from semantic_corpus.exporters import build_report, export_bio, export_sft, write_report
from semantic_corpus.question_generator import QuestionGenerator, balance
from semantic_corpus.semantic_generator import Generator
from semantic_corpus.semantic_generator.canonical import iter_canonical, to_canonical
from semantic_corpus.semantic_generator.generator import DEFAULT_PARADIGMS
from semantic_corpus.storage import CorpusStore

from .catalog import data_path, get_pool, job_root, store_path
from .chat_dataset import execute_chat_dataset
from .models import Job
from .operation_registry import OperationRegistry
from .template_mining import execute_template_mining


class Progress:
    def __init__(self, job):
        self.job = job
        self.last = 0
        self.phase = ''

    def __call__(self, phase, processed=0, total=0, **metrics):
        if phase != self.phase or monotonic() - self.last > 0.5 or (total and processed == total):
            Job.objects.filter(pk=self.job.pk).update(phase=phase, processed=processed, total=total, metrics=metrics)
            if phase != self.phase:
                print(phase, flush=True)
            self.phase, self.last = phase, monotonic()


class DailyDialogOutput:
    """Observe the existing script's progress without per-example DB writes."""

    def __init__(self, stream, job, total):
        self.stream, self.job, self.total = stream, job, total
        self.pending = ''

    def write(self, text):
        self.stream.write(text)
        self.pending += text
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            match = re.match(r'(\d+) documents, (\d+) QA,', line)
            if match:
                Job.objects.filter(pk=self.job.pk).update(processed=int(match[1]), total=self.total,
                    metrics={'qa': int(match[2])})
            elif line.startswith('Annotation complete;'):
                Job.objects.filter(pk=self.job.pk).update(phase='DailyDialog: export and validation')
        return len(text)

    def flush(self):
        self.stream.flush()


def export_store(store, out, config, progress):
    progress('Exporting SFT')
    count = export_sft(store.examples(), out / 'sft.jsonl')

    def pairs():
        for doc in store.documents():
            latest = None
            for run in store.runs(document_id=doc.document_id):
                latest = run
            if latest is not None:
                yield doc, latest

    progress('Exporting BIO / BILOU', qa=count)
    bio = export_bio(pairs(), out / 'bio.jsonl', scheme=config.get('scheme', 'BIO'))
    print(f'SFT: {count}; BIO: {bio}', flush=True)
    return count


def quality_report(store, out, config, progress):
    progress('Checking integrity and leakage')
    held_out = (get_pool(config.get('pool', '')).split('test').surface_forms
                if config.get('operation') == 'build' else ())
    report = build_report(store, held_out_names=held_out)
    write_report(report, out / 'report.json')
    print(report, flush=True)
    if not report.is_clean:
        raise ValueError('Validation found issues. Open report.json and the job log.')


def _execute_templates(job, out, c, progress):
    from semantic_corpus.documents import Document
    from semantic_corpus.template_bank import TemplateBank
    bank = TemplateBank(settings.REPO_ROOT / 'data/mined-templates')
    store = CorpusStore(out / 'store')
    processed = stored = 0
    rows = bank.generate(count=c.get('template_count', 1), seed=c['seed'],
        lemma=c.get('template_verb') or None, adjective=c.get('template_adjective') or None,
        recombine=c.get('recombine', False))
    progress('Rendering corpus templates')
    with gzip.open(out / 'generated.jsonl.gz', 'xt', encoding='utf-8') as stream:
        for processed, row in enumerate(rows, 1):
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stored += store.add_document(Document(f'template-{processed:08d}', row['text'],
                source='mined-templates', split=c['split'], metadata={
                    **{key: value for key, value in row.items() if key != 'text'},
                    'bank_source_sha256': bank.manifest['source_sha256'],
                    'recombined': c.get('recombine', False)}))
            progress('Rendering corpus templates', processed, stored=stored)
    store.publish_version(str(job.pk), build_config={**c, 'bank_source_sha256': bank.manifest['source_sha256']})
    quality_report(store, out, c, progress)
    progress('Templates rendered; corpus ready for annotation', processed, processed, stored=stored)
    return


def _execute_dailydialog(job, out, c, progress):
    limit = c.get('limit')
    source = data_path(c['source'])
    sys.argv = ['run_dailydialog.py', '--input', str(source), '--out', str(out / 'result'),
                '--seed', str(c['seed']), '--context-turns', str(c['context_turns'])]
    if limit:
        sys.argv += ['--limit', str(limit)]
    progress('DailyDialog: annotation → QA → export → validation')
    try:
        with redirect_stdout(DailyDialogOutput(sys.stdout, job, limit or 0)):
            runpy.run_path(str(settings.REPO_ROOT / 'scripts/run_dailydialog.py'), run_name='__main__')
    except SystemExit as error:
        if error.code not in (None, 0):
            raise ValueError(f'DailyDialog exited with code {error.code}') from error
    summary = json.loads((out / 'result/summary.json').read_text(encoding='utf-8'))
    counts = summary['counts']
    progress('DailyDialog completed', counts['documents'], counts['documents'], qa=counts['stored_qa'])
    return


def _execute_cli(job, out, c, progress):
    operation = job.operation
    limit = c.get('limit')
    from semantic_corpus.cli import main
    argv = [operation]
    if operation in ('validate', 'roundtrip'):
        argv += [str(data_path(c['source']))]
        if limit:
            argv += ['--limit', str(limit)]
    elif operation == 'lookup':
        argv += c['text'].split()
    else:
        argv += ['--text', c['text'], '--verbose']
    progress(OperationRegistry.default().require(operation).label)
    if main(argv):
        raise ValueError('Validation found issues; see the log for details.')
    return


def _execute_mining(job, out, c, progress):
    limit = c.get('limit')
    from semantic_corpus.semantic_annotator import ingest
    from semantic_corpus.semantic_generator.mining import mine_entities, write_proposals
    progress('Mining entity pool proposals')
    documents = ingest([data_path(c['source'])])
    report = mine_entities((doc.text for doc in islice(documents, limit)), min_support=2)
    write_proposals(report, out / 'proposals.json')
    return


def _execute_exports(job, out, c, progress):
    operation = job.operation
    store = CorpusStore(store_path(c['store']), create=False)
    if operation == 'export':
        export_store(store, out, c, progress)
    quality_report(store, out, c, progress)
    return


def _execute_corpus(job, out, c, progress):
    operation = job.operation
    limit = c.get('limit')
    annotation_ids = None
    if operation in ('annotate', 'questions'):
        progress('Creating an independent copy of the corpus')
        source_path = store_path(c['store'])
        if operation == 'annotate':
            from semantic_corpus.template_semantics import needs_annotation
            source = CorpusStore(source_path, create=False)
            documents = source.documents()
            if c.get('pending_only', False):
                documents = (doc for doc in documents
                             if needs_annotation(doc, source.runs(document_id=doc.document_id)))
            annotation_ids = {doc.document_id for doc in islice(documents, limit)}
        # Build a fresh QA table before opening the new store. Old candidates
        # must neither survive reannotation nor win deduplication over teacher
        # QA with identical wording. Keep the exact rows of untouched documents,
        # including generation-run provenance. Historical runs stay in the copy.
        ignored = shutil.ignore_patterns('qa_examples.jsonl', 'dataset_versions.jsonl')
        shutil.copytree(source_path, out / 'store', ignore=ignored)
        if annotation_ids is not None and (source_path / 'qa_examples.jsonl').is_file():
            with (source_path / 'qa_examples.jsonl').open('rb') as source_qa, \
                    (out / 'store/qa_examples.jsonl').open('xb') as retained_qa:
                for row in source_qa:
                    if row.strip() and json.loads(row)['document_id'] not in annotation_ids:
                        retained_qa.write(row)
    store = CorpusStore(out / 'store')
    generator = None
    resources = None
    build_summary = None
    if operation == 'build':
        from semantic_corpus.resource_pool import BuiltInPool, RESOURCE_SCOPE
        if c.get('resource_scope') not in (None, 'curated_v1', RESOURCE_SCOPE):
            raise ValueError(f"Unknown resource scope: {c['resource_scope']!r}")
        if c.get('resource_scope') == RESOURCE_SCOPE:
            resources = BuiltInPool(settings.REPO_ROOT, entities=get_pool(c.get('pool', '')))
            c = {**c, 'resource_pool': resources.metadata(include_mined=c.get('template_count', 1) > 0)}
            generator = Generator(seed=c['seed'], split=c['split'], pool=resources.entities,
                                  frames=resources.frames, paradigms=resources.paradigms)
            build_summary = dict(curated_generated=0, curated_stored=0, curated_duplicates=0,
                                 mined=dict(stored=0, duplicates=0), resource_pool=c['resource_pool'])
        else:
            # Saved jobs retain the resource scope they were created with.
            generator = Generator(seed=c['seed'], split=c['split'], pool=get_pool(c.get('pool', '')))
    questions = QuestionGenerator(seed=c['seed'], paradigms=generator.paradigms if generator else DEFAULT_PARADIGMS,
        ambiguous_forms=generator.pool.ambiguous_forms() if generator else ())
    qa = 0
    processed = 0
    if operation == 'build':
        scope = c.get('count_scope', 'total')
        if scope == 'per_frame':
            total = c['count'] * len(generator.frames)
            pairs = iter_canonical(generator.generate_per_frame(c['count']), prefix='ui',
                                   split=c['split'], random_seed=c['seed'], per_frame_ids=True)
        elif scope == 'total':
            # Jobs saved before per-frame quotas retain their original count and IDs.
            total = c['count']
            pairs = (to_canonical(realized, document_id=f'ui-{i:06d}', split=c['split'], random_seed=c['seed'])
                     for i, realized in enumerate(generator.generate(c['count']), 1))
        else:
            raise ValueError(f'Unknown count scope: {scope!r}')
        progress('Generation → annotations → QA', 0, total, qa=qa)
        for i, (doc, run) in enumerate(pairs, 1):
            if store.add_document(doc):
                if build_summary is not None:
                    build_summary['curated_stored'] += 1
                store.add_run(run, doc)
                examples = balance(questions.for_document(doc, run), seed=c['seed'],
                    max_per_kind=c.get('max_per_kind'), no_answer_share=c.get('no_answer_share'))
                qa += store.add_examples(examples, generation_run_id=run.run_id)
            processed = i
            progress('Generation → annotations → QA', i, total, qa=qa)
        if resources is not None:
            from semantic_corpus.template_semantics import accumulate_coverage
            build_summary['curated_generated'] = processed
            build_summary['curated_duplicates'] = processed - build_summary['curated_stored']
            mined = build_summary['mined']
            def mined_progress(stats):
                progress('Generating from the built-in mined templates', stats['rendered'],
                         stored=mined['stored'], split_excluded=stats['split_excluded'], qa=qa)
            progress('Generating from the built-in mined templates')
            for doc, run in resources.mined_records(count=c.get('template_count', 1), seed=c['seed'],
                    split=c['split'], lemma=c.get('template_verb') or None,
                    adjective=c.get('template_adjective') or None,
                    recombine=c.get('recombine', False), stats=mined, progress=mined_progress):
                if store.add_document(doc):
                    mined['stored'] += 1
                    added_qa = 0
                    if run is not None:
                        store.add_run(run, doc)
                        examples = balance(questions.for_document(doc, run), seed=c['seed'],
                            max_per_kind=c.get('max_per_kind'), no_answer_share=c.get('no_answer_share'))
                        added_qa = store.add_examples(examples, generation_run_id=run.run_id)
                        qa += added_qa
                    accumulate_coverage(mined, doc, run, added_qa)
                else:
                    mined['duplicates'] += 1
            processed += mined['rendered']
            build_summary.update(qa_examples=qa, pending_annotation=mined.get('pending_documents', 0))
            (out / 'generation-summary.json').write_text(
                json.dumps(build_summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    elif operation == 'bank':
        from semantic_corpus.qasrl_bridge import iter_bank_canonical
        from semantic_corpus.question_generator.templates import native_questions
        for doc, run in islice(iter_bank_canonical(data_path(c['source']), split=c['split']), limit):
            if store.add_document(doc):
                store.add_run(run, doc)
                qa += store.add_examples(native_questions(doc, run), generation_run_id=run.run_id)
            processed += 1
            progress('Bank import → annotations → QA', processed, limit or 0, qa=qa)
    elif operation == 'ingest':
        from semantic_corpus.semantic_annotator import ingest, read_dialogue_jsonl, segment_dialogue
        source = data_path(c['source'])
        documents = (segment_dialogue(replace(doc, split=c['split']), context_turns=c['context_turns'])
                     for doc in read_dialogue_jsonl(source)) if c['input_format'] == 'dialogue' else ingest([source])
        for doc in islice(documents, limit):
            store.add_document(doc)
            processed += 1
            progress('Importing and deduplicating documents', processed, limit or 0)
    elif operation == 'annotate':
        from semantic_corpus.semantic_annotator import CandidateResources, RuleBasedTeacher, HttpTeacher, annotate_document
        from semantic_corpus.ontology import Label
        resources = CandidateResources.load(wiktionary_dir=str(settings.REPO_ROOT / 'data/wiktionary'))
        teacher = RuleBasedTeacher() if c['teacher'] == 'rules' else HttpTeacher(
            c['teacher_url'], name=c['model'], labels=[str(label) for label in Label])
        documents = (doc for doc in store.documents() if doc.document_id in annotation_ids)
        for doc in documents:
            outcome = annotate_document(doc, teacher, run_id=f'{job.pk}-{processed}', resources=resources,
                                        random_seed=c['seed'])
            if not outcome.ok:
                raise ValueError(str(outcome.structure))
            store.add_run(outcome.run, doc)
            qa += store.add_examples(questions.for_document(doc, outcome.run), generation_run_id=outcome.run.run_id)
            processed += 1
            progress('Annotation → QA', processed, limit or 0, qa=qa, rejected=len(outcome.rejected))
    elif operation == 'questions':
        for doc in islice(store.documents(), limit):
            latest = None
            for run in store.runs(document_id=doc.document_id):
                latest = run
            if latest is not None:
                examples = balance(questions.for_document(doc, latest), seed=c['seed'],
                    max_per_kind=c.get('max_per_kind'), no_answer_share=c.get('no_answer_share'))
                qa += store.add_examples(examples, generation_run_id=latest.run_id)
            processed += 1
            progress('QA generation', processed, limit or 0, qa=qa)
    else:
        raise ValueError(f'Unknown operation: {operation}')
    progress('Saving the dataset version', processed, processed, qa=qa)
    store.publish_version(str(job.pk), build_config=c)
    if operation != 'ingest':
        export_store(store, out, c, progress)
    quality_report(store, out, c, progress)
    metrics = {'qa': qa}
    if build_summary is not None:
        metrics.update(pending_annotation=build_summary['pending_annotation'],
                       mined_stored=build_summary['mined']['stored'])
    progress('Processing complete', processed, processed, **metrics)


# Explicit Python handlers keep JSON as metadata, never executable import paths.
HANDLERS = {
    'build': _execute_corpus,
    'dailydialog': _execute_dailydialog,
    'templates': _execute_templates,
    'ingest': _execute_corpus,
    'bank': _execute_corpus,
    'annotate': _execute_corpus,
    'questions': _execute_corpus,
    'export': _execute_exports,
    'report': _execute_exports,
    'validate': _execute_cli,
    'roundtrip': _execute_cli,
    'candidates': _execute_cli,
    'lookup': _execute_cli,
    'mining': _execute_mining,
    'template_mining': execute_template_mining,
    'chat_dataset': execute_chat_dataset,
}


def validate_operation(code):
    operation = OperationRegistry.default().require(code)
    if not callable(HANDLERS.get(code)):
        raise ValueError(f'No handler registered for operation: {code!r}')
    return operation


def execute(job):
    operation = validate_operation(job.operation)
    c = job.config
    if c.get('operation', operation.code) != operation.code:
        raise ValueError('Job configuration operation does not match the job operation.')
    out = job_root(job)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding='utf-8')
    progress = Progress(job)
    progress('Preparing')
    HANDLERS[operation.code](job, out, c, progress)
