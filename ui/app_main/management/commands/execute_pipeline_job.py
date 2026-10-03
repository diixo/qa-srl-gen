import os
import threading
import traceback

from django.core.management.base import BaseCommand, CommandError

from django.db import close_old_connections
from app_main.models import Job, Worker
from app_main.operations import execute


class Command(BaseCommand):
    help = 'Execute a claimed job; used by pipeline_worker.'

    def add_arguments(self, parser):
        parser.add_argument('job_id')

    def handle(self, *args, **options):
        job = Job.objects.get(pk=options['job_id'])
        if job.status != Job.Status.RUNNING:
            raise CommandError('Job must be claimed by the worker first.')
        supervisor = Worker.objects.get(pk=1).pid
        done = threading.Event()

        def watch_supervisor():
            # A crashed supervisor must not leave an expensive orphan job alive.
            while not done.wait(3):
                close_old_connections()
                try:
                    worker = Worker.objects.get(pk=1)
                    if worker.pid != supervisor or not worker.online:
                        os._exit(3)
                finally:
                    close_old_connections()

        threading.Thread(target=watch_supervisor, daemon=True).start()
        try:
            execute(job)
        except Exception as error:
            traceback.print_exc()
            raise CommandError(str(error)) from error
        finally:
            done.set()
