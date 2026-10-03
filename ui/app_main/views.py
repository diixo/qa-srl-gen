import json
import uuid
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.utils.text import slugify
from django.views.decorators.http import require_POST, require_safe, require_http_methods

from semantic_corpus.storage import TABLES
from semantic_corpus.semantic_generator.substitutions import load_pool
from semantic_corpus.ontology import load_default_hierarchy

from . import catalog
from .forms import JobForm
from .models import Job, Worker
from .worker import start_worker


SECTIONS = {
    'pipeline': ('Новый запуск', 'От генерации или исходного корпуса до QA, экспорта и проверки качества.', None),
    'ingestion': ('Источники и импорт', 'Тексты, диалоги и QA-SRL Bank. Каждый импорт создаёт новый корпус.', ['ingest', 'bank', 'dailydialog']),
    'annotation': ('Аннотация', 'Новый проход учителя и генерация вопросов в независимой копии корпуса.', ['annotate']),
    'questions': ('Генерация QA', 'Вопросы по последней аннотации каждого документа, балансировка и экспорт.', ['questions']),
    'exports': ('Экспорт', 'SFT, BIO / BILOU и отчёт качества из сохранённых аннотаций.', ['export']),
    'quality': ('Качество и валидация', 'Целостность, распределения, утечки между split и round-trip вопросов.', ['report', 'validate', 'roundtrip']),
    'tools': ('Инструменты', 'Кандидаты разметки, глагольные формы и предложения для расширения пула.', ['candidates', 'lookup', 'mining']),
}
NAV = [
    ('Рабочее пространство', [('main', 'Обзор', 'dashboard'), ('pipeline', 'Новый запуск', 'play_circle'),
        ('jobs', 'История запусков', 'history'), ('workers', 'Воркеры и очередь', 'memory')]),
    ('Пайплайн', [('ingestion', 'Источники и импорт', 'upload_file'), ('annotation', 'Аннотация', 'edit_note'),
        ('questions', 'Генерация QA', 'question_answer'), ('exports', 'Экспорт', 'file_download'),
        ('quality', 'Качество', 'verified')]),
    ('Данные и ресурсы', [('corpora', 'Корпуса', 'storage'), ('pools', 'Пулы сущностей', 'category'),
        ('slots', 'Фреймы и слоты', 'account_tree'), ('tools', 'Инструменты', 'build')]),
]


def page(request, template, **context):
    worker = Worker.objects.filter(pk=1).first()
    return render(request, 'app_main/' + template, {
        'navigation': NAV, 'worker_online': worker.online if worker else False,
        'nav_active': context.pop('nav_active', request.resolver_match.url_name), **context})


@require_safe
def main(request):
    rows = Job.objects.all()
    return page(request, 'index.html', title='Обзор', recent=rows[:8],
        running=rows.filter(status=Job.Status.RUNNING).count(),
        queued=rows.filter(status=Job.Status.QUEUED).count(),
        completed=rows.filter(status=Job.Status.SUCCEEDED).count(),
        failed=rows.filter(status=Job.Status.FAILED).count(), stores=catalog.stores(),
        monitor=catalog.pool_monitor())


@require_http_methods(['GET', 'POST'])
def operation(request, section='pipeline'):
    title, description, allowed = SECTIONS[section]
    initial = {'operation': (allowed or ['build'])[0]}
    form = JobForm(request.POST if request.method == 'POST' else None, operations=allowed, initial=initial)
    if request.method == 'POST' and form.is_valid():
        job = Job.objects.create(operation=form.cleaned_data['operation'], config=form.cleaned_data)
        try:
            start_worker()
        except OSError as error:
            messages.error(request, f'Запуск сохранён в очереди, но воркер не стартовал: {error}')
        return redirect('app_main:job', pk=job.pk)
    return page(request, 'operation.html', nav_active=section, title=title, description=description,
                form=form, section=section, sources=catalog.sources() if section == 'ingestion' else None)


@require_safe
def jobs(request):
    rows = Job.objects.all()
    status = request.GET.get('status', '')
    if status in Job.Status.values:
        rows = rows.filter(status=status)
    return page(request, 'jobs.html', title='История запусков',
        jobs=Paginator(rows, 30).get_page(request.GET.get('page')), statuses=Job.Status.choices, selected=status)


def artifacts(job):
    root = catalog.job_root(job)
    if not root.is_dir():
        return []
    return [dict(name=p.relative_to(root).as_posix(), size=p.stat().st_size) for p in sorted(root.rglob('*'))
            if p.is_file() and p.resolve().is_relative_to(root.resolve())
            and p.suffix in ('.json', '.jsonl', '.log', '.txt')]


def log_tail(job):
    path = catalog.job_root(job) / 'run.log'
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 32000))
        return stream.read().decode('utf-8', errors='replace')


@require_safe
def job_detail(request, pk):
    job = get_object_or_404(Job, pk=pk)
    reports = []
    quality = None
    for name in ('report.json', 'result/report.json', 'result/summary.json'):
        try:
            report_data = catalog.read_json(catalog.job_root(job) / name)
            if report_data:
                reports.append((name, json.dumps(report_data, ensure_ascii=False, indent=2)))
                if name.endswith('report.json'):
                    quality = report_data
        except (ValueError, OSError):
            pass
    store = next((catalog.relative(p.parent) for p in catalog.job_root(job).rglob('manifest.json')), '')
    return page(request, 'job.html', title=job.get_operation_display(), nav_active='jobs', job=job,
        log=log_tail(job), files=artifacts(job), reports=reports,
        quality=quality, result_store=store, config_json=json.dumps(job.config, ensure_ascii=False, indent=2))


@require_safe
def job_status(request, pk):
    job = get_object_or_404(Job, pk=pk)
    worker = Worker.objects.filter(pk=1).first()
    return JsonResponse(dict(status=job.status, label=job.get_status_display(), phase=job.phase,
        processed=job.processed, total=job.total, percent=job.percent, metrics=job.metrics,
        elapsed_seconds=job.elapsed_seconds,
        active=job.active, cancel_requested=job.cancel_requested, log=log_tail(job),
        worker_online=bool(worker and worker.online)))


@require_POST
def cancel_job(request, pk):
    job = get_object_or_404(Job, pk=pk)
    Job.objects.filter(pk=pk, status=Job.Status.QUEUED).update(
        status=Job.Status.CANCELLED, phase='Отменён', cancel_requested=True, finished_at=timezone.now())
    Job.objects.filter(pk=pk, status=Job.Status.RUNNING).update(cancel_requested=True)
    return redirect('app_main:job', pk=job.pk)


@require_POST
def retry_job(request, pk):
    previous = get_object_or_404(Job, pk=pk)
    if previous.active:
        messages.error(request, 'Сначала завершите или отмените текущий запуск.')
        return redirect('app_main:job', pk=pk)
    form = JobForm(previous.config)
    if not form.is_valid():
        messages.error(request, 'Параметры больше не действительны. Создайте запуск с актуальными источниками.')
        return redirect('app_main:pipeline')
    job = Job.objects.create(operation=previous.operation, config=form.cleaned_data)
    try:
        start_worker()
    except OSError as error:
        messages.error(request, f'Запуск в очереди: {error}')
    return redirect('app_main:job', pk=job.pk)


@require_safe
def workers(request):
    worker = Worker.objects.filter(pk=1).first() or Worker()
    running = {j.slot: j for j in Job.objects.filter(status=Job.Status.RUNNING) if j.slot}
    return page(request, 'workers.html', title='Воркеры и очередь', worker=worker,
        slots=[dict(number=n, job=running.get(n)) for n in range(1, max([worker.capacity, *running.keys()]) + 1)],
        queue=Job.objects.filter(status=Job.Status.QUEUED).order_by('created_at')[:30])


@require_POST
def worker_control(request):
    Worker.objects.get_or_create(pk=1)
    action = request.POST.get('action')
    if action == 'capacity':
        try:
            capacity = int(request.POST.get('capacity', ''))
            if not 1 <= capacity <= 4:
                raise ValueError
            Worker.objects.filter(pk=1).update(capacity=capacity)
        except ValueError:
            messages.error(request, 'Выберите от 1 до 4 вычислительных слотов.')
    elif action == 'stop':
        Worker.objects.filter(pk=1).update(stop_requested=True)
        messages.info(request, 'Воркер завершит текущие задания и остановит выдачу новых.')
    elif action == 'start':
        Worker.objects.filter(pk=1).update(stop_requested=False)
        try:
            start_worker()
        except OSError as error:
            messages.error(request, str(error))
    return redirect('app_main:workers')


@require_safe
def corpora(request):
    rows = catalog.stores()
    query = request.GET.get('q', '').strip().lower()
    if query:
        rows = [r for r in rows if query in r['path'].lower()]
    return page(request, 'corpora.html', title='Корпуса', stores=Paginator(rows, 30).get_page(request.GET.get('page')), query=query)


@require_safe
def corpus(request):
    try:
        path = catalog.data_path(request.GET.get('store', ''), directory=True)
        meta = catalog.read_json(path / 'manifest.json', {})
        if meta.get('store_format') != 'jsonl-v1':
            raise ValueError('Нет манифеста корпуса.')
        table = request.GET.get('table', 'documents')
        if table not in TABLES:
            raise ValueError('Неизвестная таблица.')
        offset = max(0, int(request.GET.get('offset', 0)))
        rows, more = catalog.table_page(path, table, offset)
        tables = [dict(name=name, size=(path / file).stat().st_size if (path / file).exists() else 0)
                  for name, file in TABLES.items()]
    except (ValueError, OSError) as error:
        return page(request, 'error.html', title='Корпус недоступен', error=str(error))
    return page(request, 'corpus.html', title=path.name, nav_active='corpora', store=catalog.relative(path),
        tables=tables, table=table, rows=[json.dumps(r, ensure_ascii=False, indent=2) for r in rows],
        offset=offset, previous=max(0, offset - 30), next=offset + 30, more=more)


@require_safe
def pools(request):
    value = request.GET.get('pool', '')
    try:
        monitor = catalog.pool_monitor(value)
    except ValueError as error:
        return page(request, 'error.html', title='Пул недоступен', error=str(error))
    pool = monitor['pool']
    q = request.GET.get('q', '').strip().lower()
    split = request.GET.get('split', '')
    rows = [dict(text=e.text, labels=str(e.labels), type=e.type_name, named=e.is_named,
                 gloss=e.appositive, split=pool.split_of(e.text)) for e in pool]
    rows = [r for r in rows if (not q or q in r['text'].lower() or q in r['labels'].lower())
            and (not split or split == r['split'])]
    return page(request, 'pools.html', title='Пулы сущностей', monitor=monitor, pool_choices=catalog.pool_choices(),
        selected_pool=value, entities=Paginator(rows, 40).get_page(request.GET.get('page')), query=q, selected_split=split)


@require_safe
def slots(request):
    value = request.GET.get('pool', '')
    try:
        monitor = catalog.pool_monitor(value)
    except ValueError as error:
        return page(request, 'error.html', title='Пул недоступен', error=str(error))
    return page(request, 'slots.html', title='Фреймы и слоты', monitor=monitor,
                pool_choices=catalog.pool_choices(), selected_pool=value)


@require_POST
def upload(request):
    file = request.FILES.get('file')
    is_pool = request.POST.get('kind') == 'pool'
    destination = 'app_main:pools' if is_pool else 'app_main:ingestion'
    if not file or file.size > 20 * 1024 * 1024:
        messages.error(request, 'Выберите файл размером до 20 МБ.')
        return redirect(destination)
    suffix = Path(file.name).suffix.lower()
    if suffix not in (('.json',) if is_pool else ('.txt', '.jsonl', '.qa')):
        messages.error(request, 'Неподдерживаемое расширение файла.')
        return redirect(destination)
    folder = settings.CORPUS_UI_ROOT / ('pools' if is_pool else 'uploads')
    folder.mkdir(parents=True, exist_ok=True)
    name = slugify(Path(file.name).stem, allow_unicode=True)[:64] or 'upload'
    target = folder / f'{name}-{uuid.uuid4().hex[:12]}{suffix}'
    try:
        with target.open('xb') as stream:
            for chunk in file.chunks():
                stream.write(chunk)
        if is_pool:
            payload = catalog.read_json(target)
            if not isinstance(payload, dict) or not isinstance(payload.get('entities'), list):
                raise ValueError('Ожидается JSON с массивом entities.')
            if not len(load_pool(target, load_default_hierarchy())):
                raise ValueError('Пул не содержит сущностей.')
        messages.success(request, f'Файл {file.name} сохранён: {catalog.relative(target)}')
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        target.unlink(missing_ok=True)
        messages.error(request, f'Некорректный файл: {error}')
    return redirect(destination)


@require_safe
def download(request, pk, name):
    job = get_object_or_404(Job, pk=pk)
    root = catalog.job_root(job).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.suffix not in ('.json', '.jsonl', '.txt', '.log'):
        raise Http404
    return FileResponse(path.open('rb'), as_attachment=True, filename=path.name)


@require_safe
def report(request):
    return redirect('app_main:quality')

