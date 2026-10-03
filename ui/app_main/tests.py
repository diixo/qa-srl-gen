import json
import io
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from semantic_corpus.documents import Document
from semantic_corpus.storage import CorpusStore

from . import catalog
from .forms import JobForm
from .models import Job, Worker
from .operations import DailyDialogOutput, execute
from .views import SECTIONS
from .worker import run_worker, start_worker, supervisor_lock


class WorkspaceTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.original_root = settings.REPO_ROOT
        self.override = override_settings(REPO_ROOT=self.root, CORPUS_UI_ROOT=self.root / 'artifacts/ui')
        self.override.enable()
        self.addCleanup(self.override.disable)
        settings.CORPUS_UI_ROOT.mkdir(parents=True)
        (self.root / 'data').mkdir()
        (self.root / 'data/dialogue.jsonl').write_text('["Hello!", "Thank you!"]\n', encoding='utf-8')
        shutil.copyfile(self.original_root / 'tests/fixtures/bank_sample.jsonl', self.root / 'data/bank.jsonl')
        self.store = CorpusStore(self.root / 'artifacts/source')
        self.store.add_document(Document('d', 'Anna visited Kyiv.', split='train'))

    def config(self, operation='build', **extra):
        form = JobForm(dict(operation=operation, count=3, seed=42, split='train', **extra))
        self.assertTrue(form.is_valid(), form.errors)
        return form.cleaned_data

    def execute(self, operation='build', **extra):
        job = Job.objects.create(operation=operation, config=self.config(operation, **extra),
                                 status=Job.Status.RUNNING)
        with patch('sys.stdout', new_callable=io.StringIO):
            execute(job)
        job.status = Job.Status.SUCCEEDED
        job.save(update_fields=['status'])
        return job

    def test_all_navigation_pages_render_real_data(self):
        for route in ('main', 'jobs', 'workers', 'corpora', 'pools', 'slots', *SECTIONS):
            with self.subTest(route=route):
                response = self.client.get(reverse('app_main:' + route))
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, 'aria-current="page"')
                self.assertNotContains(response, 'Coming soon')
        response = self.client.get(reverse('app_main:corpus'), {'store': 'artifacts/source'})
        self.assertContains(response, 'Anna visited Kyiv.')

    @patch('app_main.views.start_worker')
    def test_post_queues_and_redirects_to_persistent_job(self, start):
        response = self.client.post(reverse('app_main:pipeline'), {'operation': 'build', 'count': 4})
        job = Job.objects.get()
        self.assertRedirects(response, reverse('app_main:job', args=[job.pk]))
        self.assertEqual(job.config['count'], 4)
        self.assertEqual(job.config['seed'], 42)
        self.assertEqual(job.status, Job.Status.QUEUED)
        start.assert_called_once()

    def test_invalid_input_does_not_enqueue(self):
        for payload in ({'operation': 'build', 'count': -1},
                        {'operation': 'ingest', 'source': '../README.md'},
                        {'operation': 'annotate', 'store': 'artifacts/source', 'teacher': 'http'},
                        {'operation': 'export', 'store': 'artifacts/missing'}):
            self.client.post(reverse('app_main:pipeline'), payload)
        self.assertEqual(Job.objects.count(), 0)

    def test_writes_require_post_and_csrf(self):
        job = Job.objects.create(operation='build')
        for route, args in (('cancel_job', [job.pk]), ('retry_job', [job.pk]), ('worker_control', []), ('upload', [])):
            url = reverse('app_main:' + route, args=args)
            self.assertEqual(self.client.get(url).status_code, 405)
            self.assertEqual(Client(enforce_csrf_checks=True).post(url).status_code, 403)

    def test_cancellation_distinguishes_queued_and_running(self):
        for status in (Job.Status.QUEUED, Job.Status.RUNNING):
            job = Job.objects.create(operation='build', status=status)
            self.client.post(reverse('app_main:cancel_job', args=[job.pk]))
            job.refresh_from_db()
            self.assertTrue(job.cancel_requested)
            self.assertEqual(job.status, Job.Status.CANCELLED if status == Job.Status.QUEUED else status)

    def test_build_export_and_balanced_regeneration_do_not_change_source(self):
        original = self.store.content_hash()
        job = self.execute()
        root = catalog.job_root(job)
        store = CorpusStore(root / 'store', create=False)
        self.assertGreater(store.count('qa_examples'), 0)
        self.assertTrue((root / 'sft.jsonl').is_file())
        self.assertFalse(json.loads((root / 'report.json').read_text())['integrity_issues'])
        before = store.content_hash()
        export = self.execute('export', store=catalog.relative(root / 'store'))
        self.assertTrue((catalog.job_root(export) / 'bio.jsonl').is_file())
        regenerated = self.execute('questions', store=catalog.relative(root / 'store'), no_answer_share=0)
        result = CorpusStore(catalog.job_root(regenerated) / 'store', create=False)
        self.assertTrue(all(e.answerable for e in result.examples()))
        self.assertEqual(CorpusStore(root / 'store').content_hash(), before)
        self.assertEqual(self.store.content_hash(), original)

    def test_dialogue_ingestion_and_rule_annotation(self):
        from semantic_corpus.semantic_annotator import CandidateResources
        job = self.execute('ingest', source='data/dialogue.jsonl', input_format='dialogue')
        source = catalog.relative(catalog.job_root(job) / 'store')
        with patch('semantic_corpus.semantic_annotator.CandidateResources.load', return_value=CandidateResources()):
            annotated = self.execute('annotate', store=source)
        store = CorpusStore(catalog.job_root(annotated) / 'store')
        self.assertEqual(store.count('documents'), 1)
        self.assertGreater(store.count('dialogue'), 0)
        self.assertGreater(store.count('qa_examples'), 0)

    def test_bank_import_keeps_native_qa_and_run_ownership(self):
        job = self.execute('bank', source='data/bank.jsonl', limit=1)
        store = CorpusStore(catalog.job_root(job) / 'store')
        self.assertEqual(store.count('documents'), 1)
        examples = list(store.examples())
        self.assertTrue(examples)
        self.assertTrue(all(e.run_id for e in examples))

    def test_bank_validation_and_mining_adapters(self):
        self.execute('validate', source='data/bank.jsonl', limit=1)
        self.execute('roundtrip', source='data/bank.jsonl', limit=1)
        (self.root / 'data/mining.txt').write_text('We went to Paris. We stayed in Paris.', encoding='utf-8')
        job = self.execute('mining', source='data/mining.txt')
        self.assertIn('proposed', catalog.read_json(catalog.job_root(job) / 'proposals.json'))

    def test_dailydialog_progress_observer_handles_fragmented_prints(self):
        job = Job.objects.create(operation='dailydialog', status=Job.Status.RUNNING)
        output = io.StringIO()
        observer = DailyDialogOutput(output, job, 1000)
        observer.write('500 documents, ')
        observer.write('12500 QA, 20.1s\n')
        job.refresh_from_db()
        self.assertEqual(job.processed, 500)
        self.assertEqual(job.metrics, {'qa': 12500})
        observer.write('Annotation complete; exporting saved data.\n')
        job.refresh_from_db()
        self.assertIn('DailyDialog', job.phase)
        self.assertIn('12500 QA', output.getvalue())

    def test_pool_upload_rejects_bad_ontology_and_preserves_default(self):
        url = reverse('app_main:upload')
        bad = SimpleUploadedFile('bad.json', b'{"entities": [{"text": "A", "labels": ["WRONG"]}]}')
        self.client.post(url, {'kind': 'pool', 'file': bad})
        self.assertEqual(len(catalog.pool_choices()), 1)
        good = SimpleUploadedFile('pool.json', b'{"entities": [{"text": "Anna", "labels": ["PERSON"]}]}')
        self.client.post(url, {'kind': 'pool', 'file': good})
        choices = catalog.pool_choices()
        self.assertEqual(len(choices), 2)
        self.assertEqual(len(catalog.get_pool(choices[1][0])), 1)
        self.assertTrue(any(slot['empty'] for slot in catalog.pool_monitor(choices[1][0])['slots']))

    def test_paths_and_downloads_cannot_escape_results(self):
        with self.assertRaises(ValueError):
            catalog.data_path(str(self.original_root / 'README.md'))
        job = Job.objects.create(operation='build')
        root = catalog.job_root(job)
        root.mkdir(parents=True)
        (root / 'report.json').write_text('{}', encoding='utf-8')
        good = self.client.get(reverse('app_main:download', args=[job.pk, 'report.json']))
        self.assertEqual(good.status_code, 200)
        good.close()
        bad = self.client.get(reverse('app_main:download', args=[job.pk, '../../db.sqlite3']))
        self.assertEqual(bad.status_code, 404)

    def test_incomplete_store_is_not_a_job_input(self):
        job = Job.objects.create(operation='build', status=Job.Status.RUNNING)
        CorpusStore(catalog.job_root(job) / 'store')
        for status in (Job.Status.RUNNING, Job.Status.FAILED, Job.Status.CANCELLED):
            job.status = status
            job.save(update_fields=['status'])
            with self.assertRaises(ValueError):
                catalog.store_path(catalog.relative(catalog.job_root(job) / 'store'))

    def test_job_output_is_escaped_and_status_endpoint_is_read_only(self):
        job = Job.objects.create(operation='build', phase='<script>alert(1)</script>')
        response = self.client.get(reverse('app_main:job', args=[job.pk]))
        self.assertNotContains(response, '<script>alert(1)</script>')
        payload = self.client.get(reverse('app_main:job_status', args=[job.pk])).json()
        self.assertEqual(payload['status'], 'queued')
        self.assertFalse(payload['worker_online'])

    def test_pool_coverage_uses_exclusions(self):
        data = catalog.pool_monitor()
        give_theme = next(s for s in data['slots'] if s['frame'] == 'give' and s['name'] == 'theme')
        self.assertIn('EMOTION', give_theme['excludes'])
        self.assertTrue(all(c > 0 for c in give_theme['counts']))

    def test_worker_settings_and_paused_queue(self):
        self.client.post(reverse('app_main:worker_control'), {'action': 'capacity', 'capacity': 4})
        self.assertEqual(Worker.objects.get().capacity, 4)
        self.client.post(reverse('app_main:worker_control'), {'action': 'capacity', 'capacity': 999})
        self.assertEqual(Worker.objects.get().capacity, 4)
        self.client.post(reverse('app_main:worker_control'), {'action': 'stop'})
        with patch('app_main.worker.subprocess.Popen') as spawn:
            start_worker()
            spawn.assert_not_called()

    def test_worker_claims_queue_and_records_failure(self):
        Worker.objects.create(capacity=1)
        good = Job.objects.create(operation='build')
        bad = Job.objects.create(operation='build')
        processes = [Mock(pid=10), Mock(pid=11)]
        processes[0].poll.return_value = 0
        processes[1].poll.return_value = 2
        with patch('app_main.worker.subprocess.Popen', side_effect=processes), patch('app_main.worker.time.sleep'):
            run_worker(once=True)
        good.refresh_from_db()
        bad.refresh_from_db()
        self.assertEqual(good.status, Job.Status.SUCCEEDED)
        self.assertEqual(bad.status, Job.Status.FAILED)
        self.assertEqual(good.slot, 1)
        self.assertIsNotNone(good.finished_at)
        self.assertIsNone(Worker.objects.get().pid)

    def test_os_lock_allows_only_one_supervisor(self):
        with supervisor_lock() as first:
            with supervisor_lock() as second:
                self.assertTrue(first)
                self.assertFalse(second)

    def test_worker_terminates_only_cancelled_child(self):
        Worker.objects.create(capacity=1)
        job = Job.objects.create(operation='build')
        process = Mock(pid=10)
        process.poll.side_effect = [None, -15]

        def cancel_after_spawn(*args, **kwargs):
            Job.objects.filter(pk=job.pk).update(cancel_requested=True)
            return process

        with patch('app_main.worker.subprocess.Popen', side_effect=cancel_after_spawn), patch('app_main.worker.time.sleep'):
            run_worker(once=True)
        job.refresh_from_db()
        self.assertEqual(job.status, Job.Status.CANCELLED)
        process.terminate.assert_called_once()

    def test_restart_marks_interrupted_job_failed_without_reusing_output(self):
        job = Job.objects.create(operation='build', status=Job.Status.RUNNING)
        run_worker(once=True)
        job.refresh_from_db()
        self.assertEqual(job.status, Job.Status.FAILED)
        self.assertFalse(catalog.job_root(job).exists())

    @patch('app_main.views.start_worker')
    def test_retry_creates_a_fresh_job_and_keeps_previous_result(self, start):
        job = self.execute()
        before = CorpusStore(catalog.job_root(job) / 'store').content_hash()
        response = self.client.post(reverse('app_main:retry_job', args=[job.pk]))
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Job.objects.count(), 2)
        self.assertEqual(CorpusStore(catalog.job_root(job) / 'store').content_hash(), before)
        start.assert_called_once()
