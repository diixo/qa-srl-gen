"""Persistent local queue with one supervisor and isolated process slots."""

import os
import subprocess
import sys
import time
from contextlib import contextmanager

from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone

from .catalog import job_root
from .models import Job, Worker


def process_options():
    return {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}


def start_worker():
    worker, _ = Worker.objects.get_or_create(pk=1)
    if worker.stop_requested:
        return None
    # Competing launch requests are harmless: only the OS lock owner runs.
    return subprocess.Popen(
        [sys.executable, str(settings.BASE_DIR / 'manage.py'), 'pipeline_worker'],
        cwd=settings.REPO_ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, env={**os.environ, 'CORPUS_UI_ROOT': str(settings.CORPUS_UI_ROOT)},
        **process_options(),
    )


@contextmanager
def supervisor_lock():
    path = settings.CORPUS_UI_ROOT / 'worker.lock'
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def finish(job_id, status, error=''):
    Job.objects.filter(pk=job_id, status=Job.Status.RUNNING).update(
        status=status, error=error, finished_at=timezone.now(),
        phase=dict(Job.Status.choices)[status], pid=None,
    )


def run_worker(*, once=False, idle_seconds=60):
    with supervisor_lock() as acquired:
        if not acquired:
            return
        worker, _ = Worker.objects.get_or_create(pk=1)
        Worker.objects.filter(pk=1).update(pid=os.getpid(), heartbeat=timezone.now())
        # Never silently retry a partially written corpus after a supervisor crash.
        Job.objects.filter(status=Job.Status.RUNNING).update(
            status=Job.Status.FAILED, phase='Прерванный запуск', finished_at=timezone.now(),
            error='Воркер был прерван. Создайте новый запуск; результат может быть неполным.', pid=None)
        children = {}
        idle_since = time.monotonic()
        try:
            while True:
                close_old_connections()
                Worker.objects.filter(pk=1).update(heartbeat=timezone.now())
                worker.refresh_from_db()
                for slot, (job_id, process, log) in list(children.items()):
                    cancelled = Job.objects.filter(pk=job_id, cancel_requested=True).exists()
                    if cancelled and process.poll() is None:
                        process.terminate()
                    code = process.poll()
                    if code is not None:
                        log.close()
                        finish(job_id, Job.Status.CANCELLED if cancelled else
                               Job.Status.SUCCEEDED if code == 0 else Job.Status.FAILED,
                               '' if code == 0 or cancelled else f'Процесс завершился с кодом {code}. Подробности в логе.')
                        del children[slot]
                if not worker.stop_requested:
                    for slot in range(1, max(1, min(worker.capacity, 4)) + 1):
                        if len(children) >= max(1, min(worker.capacity, 4)):
                            break
                        if slot in children:
                            continue
                        job = Job.objects.filter(status=Job.Status.QUEUED).order_by('created_at').first()
                        if job is None:
                            break
                        claimed = Job.objects.filter(pk=job.pk, status=Job.Status.QUEUED,
                                                     cancel_requested=False).update(
                            status=Job.Status.RUNNING, started_at=timezone.now(), slot=slot,
                            phase='Подготовка', processed=0, total=0)
                        if not claimed:
                            continue
                        log = None
                        try:
                            root = job_root(job)
                            root.mkdir(parents=True, exist_ok=False)
                            log = (root / 'run.log').open('wb')
                            process = subprocess.Popen(
                                [sys.executable, '-u', str(settings.BASE_DIR / 'manage.py'),
                                 'execute_pipeline_job', str(job.pk)], cwd=settings.REPO_ROOT,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, 'PYTHONIOENCODING': 'utf-8',
                                     'CORPUS_UI_ROOT': str(settings.CORPUS_UI_ROOT)}, **process_options())
                            children[slot] = (job.pk, process, log)
                            Job.objects.filter(pk=job.pk).update(pid=process.pid)
                        except Exception as error:
                            if log is not None:
                                log.close()
                            finish(job.pk, Job.Status.FAILED, str(error))
                pending = Job.objects.filter(status=Job.Status.QUEUED).exists()
                if children or (pending and not worker.stop_requested):
                    idle_since = time.monotonic()
                elif once or worker.stop_requested or time.monotonic() - idle_since >= idle_seconds:
                    break
                time.sleep(0.5)
        finally:
            for job_id, process, log in children.values():
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=10)
                log.close()
                finish(job_id, Job.Status.FAILED, 'Воркер остановлен во время выполнения.')
            Worker.objects.filter(pk=1).update(pid=None, heartbeat=timezone.now())
