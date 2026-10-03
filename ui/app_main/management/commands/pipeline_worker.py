from django.core.management.base import BaseCommand

from app_main.worker import run_worker


class Command(BaseCommand):
    help = 'Run the local corpus queue (also started automatically by the UI).'

    def add_arguments(self, parser):
        parser.add_argument('--once', action='store_true', help='Drain the queue and exit.')

    def handle(self, *args, **options):
        run_worker(once=options['once'])
