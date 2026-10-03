"""UI adapters around the existing corpus APIs; every write has a fresh output."""

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
from semantic_corpus.semantic_generator.canonical import to_canonical
from semantic_corpus.semantic_generator.generator import DEFAULT_PARADIGMS
from semantic_corpus.storage import CorpusStore

from .catalog import data_path, get_pool, job_root, store_path
from .models import Job


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
                Job.objects.filter(pk=self.job.pk).update(phase='DailyDialog: экспорт и проверка')
        return len(text)

    def flush(self):
        self.stream.flush()


def export_store(store, out, config, progress):
    progress('Экспорт SFT')
    count = export_sft(store.examples(), out / 'sft.jsonl')

    def pairs():
        for doc in store.documents():
            latest = None
            for run in store.runs(document_id=doc.document_id):
                latest = run
            if latest is not None:
                yield doc, latest

    progress('Экспорт BIO / BILOU', qa=count)
    bio = export_bio(pairs(), out / 'bio.jsonl', scheme=config.get('scheme', 'BIO'))
    print(f'SFT: {count}; BIO: {bio}', flush=True)
    return count


def quality_report(store, out, config, progress):
    progress('Проверка целостности и утечек')
    held_out = (get_pool(config.get('pool', '')).split('test').surface_forms
                if config.get('operation') == 'build' else ())
    report = build_report(store, held_out_names=held_out)
    write_report(report, out / 'report.json')
    print(report, flush=True)
    if not report.is_clean:
        raise ValueError('Проверка выявила проблемы. Откройте report.json и лог запуска.')


def execute(job):
    c = job.config
    out = job_root(job)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding='utf-8')
    progress = Progress(job)
    progress('Подготовка')
    operation = job.operation
    limit = c.get('limit')

    if operation == 'dailydialog':
        source = data_path(c['source'])
        sys.argv = ['run_dailydialog.py', '--input', str(source), '--out', str(out / 'result'),
                    '--seed', str(c['seed']), '--context-turns', str(c['context_turns'])]
        if limit:
            sys.argv += ['--limit', str(limit)]
        progress('DailyDialog: аннотация → QA → экспорт → проверка')
        try:
            with redirect_stdout(DailyDialogOutput(sys.stdout, job, limit or 0)):
                runpy.run_path(str(settings.REPO_ROOT / 'scripts/run_dailydialog.py'), run_name='__main__')
        except SystemExit as error:
            if error.code not in (None, 0):
                raise ValueError(f'DailyDialog завершился с кодом {error.code}') from error
        summary = json.loads((out / 'result/summary.json').read_text(encoding='utf-8'))
        counts = summary['counts']
        progress('DailyDialog завершён', counts['documents'], counts['documents'], qa=counts['stored_qa'])
        return

    if operation in ('validate', 'roundtrip', 'lookup', 'candidates'):
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
        progress(dict(Job._meta.get_field('operation').choices)[operation])
        if main(argv):
            raise ValueError('Проверка выявила проблемы; подробности в логе.')
        return

    if operation == 'mining':
        from semantic_corpus.semantic_annotator import ingest
        from semantic_corpus.semantic_generator.mining import mine_entities, write_proposals
        progress('Поиск кандидатов в пул')
        documents = ingest([data_path(c['source'])])
        report = mine_entities((doc.text for doc in islice(documents, limit)), min_support=2)
        write_proposals(report, out / 'proposals.json')
        return

    if operation in ('export', 'report'):
        store = CorpusStore(store_path(c['store']), create=False)
        if operation == 'export':
            export_store(store, out, c, progress)
        quality_report(store, out, c, progress)
        return

    if operation in ('annotate', 'questions'):
        progress('Создание независимой копии корпуса')
        # Regeneration must honour the new balance settings even if the source
        # already has QA. The source itself remains untouched.
        ignored = shutil.ignore_patterns('qa_examples.jsonl', 'dataset_versions.jsonl') if operation == 'questions' else None
        shutil.copytree(store_path(c['store']), out / 'store', ignore=ignored)
    store = CorpusStore(out / 'store')
    generator = None
    if operation == 'build':
        generator = Generator(seed=c['seed'], split=c['split'], pool=get_pool(c.get('pool', '')))
    questions = QuestionGenerator(seed=c['seed'], paradigms=DEFAULT_PARADIGMS,
        ambiguous_forms=generator.pool.ambiguous_forms() if generator else ())
    qa = 0
    processed = 0
    if operation == 'build':
        for i, realized in enumerate(generator.generate(c['count']), 1):
            doc, run = to_canonical(realized, document_id=f'ui-{i:06d}', split=c['split'], random_seed=c['seed'])
            if store.add_document(doc):
                store.add_run(run, doc)
                examples = balance(questions.for_document(doc, run), seed=c['seed'],
                    max_per_kind=c.get('max_per_kind'), no_answer_share=c.get('no_answer_share'))
                qa += store.add_examples(examples, generation_run_id=run.run_id)
            processed = i
            progress('Генерация → аннотации → QA', i, c['count'], qa=qa)
    elif operation == 'bank':
        from semantic_corpus.qasrl_bridge import iter_bank_canonical
        from semantic_corpus.question_generator.templates import native_questions
        for doc, run in islice(iter_bank_canonical(data_path(c['source']), split=c['split']), limit):
            if store.add_document(doc):
                store.add_run(run, doc)
                qa += store.add_examples(native_questions(doc, run), generation_run_id=run.run_id)
            processed += 1
            progress('Импорт Bank → аннотации → QA', processed, limit or 0, qa=qa)
    elif operation == 'ingest':
        from semantic_corpus.semantic_annotator import ingest, read_dialogue_jsonl, segment_dialogue
        source = data_path(c['source'])
        documents = (segment_dialogue(replace(doc, split=c['split']), context_turns=c['context_turns'])
                     for doc in read_dialogue_jsonl(source)) if c['input_format'] == 'dialogue' else ingest([source])
        for doc in islice(documents, limit):
            store.add_document(doc)
            processed += 1
            progress('Импорт и дедупликация документов', processed, limit or 0)
    elif operation == 'annotate':
        from semantic_corpus.semantic_annotator import CandidateResources, RuleBasedTeacher, HttpTeacher, annotate_document
        from semantic_corpus.ontology import Label
        resources = CandidateResources.load(wiktionary_dir=str(settings.REPO_ROOT / 'data/wiktionary'))
        teacher = RuleBasedTeacher() if c['teacher'] == 'rules' else HttpTeacher(
            c['teacher_url'], name=c['model'], labels=[str(label) for label in Label])
        for doc in islice(store.documents(), limit):
            outcome = annotate_document(doc, teacher, run_id=f'{job.pk}-{processed}', resources=resources,
                                        random_seed=c['seed'])
            if not outcome.ok:
                raise ValueError(str(outcome.structure))
            store.add_run(outcome.run, doc)
            qa += store.add_examples(questions.for_document(doc, outcome.run), generation_run_id=outcome.run.run_id)
            processed += 1
            progress('Аннотация → QA', processed, limit or 0, qa=qa, rejected=len(outcome.rejected))
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
            progress('Генерация QA', processed, limit or 0, qa=qa)
    else:
        raise ValueError(f'Неизвестная операция: {operation}')
    progress('Сохранение версии', processed, processed, qa=qa)
    store.publish_version(str(job.pk), build_config=c)
    if operation != 'ingest':
        export_store(store, out, c, progress)
    quality_report(store, out, c, progress)
    progress('Обработка завершена', processed, processed, qa=qa)
