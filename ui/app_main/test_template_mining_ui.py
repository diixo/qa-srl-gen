"""The resource-mining page uploads local files and keeps them for the worker."""

import gzip
import json
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, SimpleTestCase, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .catalog import job_root
from .models import Job
from .template_mining import prepare_source, save_source_upload
from .template_mining_forms import TemplateMiningForm


class MiningInputFixtures:
    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'data').mkdir()
        self.source = self.root / 'data' / 'sentences.txt'
        self.source.write_text('They picked up the book.\n', encoding='utf-8')
        overrides = override_settings(
            REPO_ROOT=self.root,
            CORPUS_UI_ROOT=self.root / 'artifacts' / 'ui',
        )
        overrides.enable()
        self.addCleanup(overrides.disable)


class TemplateMiningFormTests(MiningInputFixtures, SimpleTestCase):
    def test_requires_text_or_file_including_for_whitespace_only_input(self):
        for payload in ({}, {'text': ' \n\t ', 'source_path': '   '}):
            with self.subTest(payload=payload):
                form = TemplateMiningForm(payload)
                self.assertFalse(form.is_valid())
                self.assertIn('Enter English text or choose a source file.', str(form.non_field_errors()))

    def test_pasted_text_preserves_whitespace_and_needs_no_file(self):
        text = '  She bought a book.\n\nThey read it.\n'
        form = TemplateMiningForm({'text': text})
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data['text'], text)
        self.assertEqual(form.cleaned_data['source_path'], '')
        self.assertNotIn('publish_to_pool', form.cleaned_data)

    def test_file_only_and_combined_inputs_accept_uploaded_source(self):
        for text in ('', 'She visited Kyiv.'):
            with self.subTest(text=text):
                uploaded = SimpleUploadedFile('sentences.txt', self.source.read_bytes())
                form = TemplateMiningForm({'text': text}, {'source_file': uploaded})
                self.assertTrue(form.is_valid(), form.errors)
                self.assertIs(form.cleaned_data['source_file'], uploaded)
                self.assertEqual(form.cleaned_data['source_path'], '')
                self.assertEqual(form.cleaned_data['text'], text)
                self.assertNotIn('publish_to_pool', form.cleaned_data)
        self.assertEqual(self.source.read_text(encoding='utf-8'), 'They picked up the book.\n')

    def test_retry_can_reuse_saved_source_but_browser_cannot_submit_a_path(self):
        form = TemplateMiningForm({}, source_path=str(self.source))
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(Path(form.cleaned_data['source_path']), self.source.resolve())
        form = TemplateMiningForm({'source_path': str(self.source)})
        self.assertFalse(form.is_valid())
        self.assertNotIn('source_path', form.fields)

    def test_invalid_saved_source_is_rejected_on_retry_even_with_valid_pasted_text(self):
        outside = self.root / 'private.txt'
        outside.write_text('Private text.', encoding='utf-8')
        unsupported = self.root / 'data' / 'table.csv'
        unsupported.write_text('text\nA sentence.\n', encoding='utf-8')
        for path in ('data/missing.txt', 'data', 'data/../private.txt', str(outside), 'data/table.csv'):
            with self.subTest(path=path):
                form = TemplateMiningForm({'text': 'Go.'}, source_path=path)
                self.assertFalse(form.is_valid())
                self.assertTrue(form.non_field_errors())

    def test_file_extensions_and_empty_uploads_are_validated(self):
        for name in ('source.txt', 'source.jsonl', 'source.JSONL.GZ'):
            form = TemplateMiningForm({}, {'source_file': SimpleUploadedFile(name, b'input')})
            self.assertTrue(form.is_valid(), form.errors)
        for name, content in (('source.csv', b'input'), ('source.jsonl.zip', b'input'), ('empty.txt', b'')):
            form = TemplateMiningForm({'text': 'Go.'}, {'source_file': SimpleUploadedFile(name, content)})
            self.assertFalse(form.is_valid())
            self.assertIn('source_file', form.errors)

    def test_interrupted_upload_removes_the_partial_copy(self):
        class BrokenUpload:
            name = 'source.txt'

            def chunks(self):
                yield b'An incomplete source'
                raise OSError('Read interrupted')

        with self.assertRaisesRegex(OSError, 'Read interrupted'):
            save_source_upload(BrokenUpload())
        self.assertEqual(list((self.root / 'artifacts/ui/uploads').iterdir()), [])


class TemplateMiningPageTests(MiningInputFixtures, TestCase):
    def setUp(self):
        super().setUp()
        self.url = reverse('app_main:template_mining')

    @patch('app_main.views.start_worker')
    def test_get_renders_dedicated_resource_page_without_starting_work(self, start):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<h1>Template mining</h1>', html=True)
        self.assertContains(response, 'name="text"')
        self.assertContains(response, 'name="source_file"')
        self.assertContains(response, 'type="file"')
        self.assertContains(response, 'enctype="multipart/form-data"')
        self.assertNotContains(response, 'name="source_path"')
        self.assertNotContains(response, 'name="publish_to_pool"')
        self.assertContains(response, 'Validated resources are automatically added to Built-in pool.')
        self.assertContains(response, 'UNREVIEWED')
        self.assertContains(response, 'aria-current="page"')
        self.assertEqual(response.context['nav_active'], 'template_mining')
        self.assertNotIn('publish_to_pool', response.context['form'].fields)
        self.assertEqual(Job.objects.count(), 0)
        start.assert_not_called()

    @patch('app_main.views.start_worker')
    def test_post_queues_combined_sources_and_redirects_to_job(self, start):
        text = 'She bought a beautiful book.\n'
        response = self.client.post(self.url, {
            'text': text,
            'source_file': SimpleUploadedFile('sentences.txt', self.source.read_bytes()),
        })
        job = Job.objects.get()
        self.assertRedirects(response, reverse('app_main:job', args=[job.pk]), fetch_redirect_response=False)
        self.assertEqual(job.operation, 'template_mining')
        self.assertEqual(job.config['operation'], 'template_mining')
        self.assertEqual(job.config['text'], text)
        saved = Path(job.config['source_path'])
        self.assertTrue(saved.is_relative_to(self.root / 'artifacts/ui/uploads'))
        self.assertEqual(saved.read_bytes(), self.source.read_bytes())
        self.assertEqual(job.config['source_name'], 'sentences.txt')
        self.assertNotIn('source_file', job.config)
        self.assertFalse(job_root(job).exists())
        snapshot = self.root / 'snapshot.jsonl'
        counts = prepare_source(job.config, snapshot, max_length=1000)
        self.assertEqual(counts['records'], 2)
        self.assertEqual([json.loads(line)['example'] for line in snapshot.read_text().splitlines()],
                         [text.rstrip('\n'), self.source.read_text().rstrip('\n')])
        self.assertNotIn('publish_to_pool', job.config)
        self.assertEqual(job.status, Job.Status.QUEUED)
        start.assert_called_once()

    @patch('app_main.views.start_worker')
    def test_legacy_opt_out_is_not_saved_for_new_jobs(self, start):
        response = self.client.post(self.url, {'text': 'Go.', 'publish_to_pool': ''})
        self.assertEqual(response.status_code, 302)
        self.assertNotIn('publish_to_pool', Job.objects.get().config)
        start.assert_called_once()

    @patch('app_main.views.start_worker')
    def test_invalid_inputs_remain_on_page_without_queueing(self, start):
        for payload in ({}, {'text': 'Go.', 'source_file': SimpleUploadedFile('private.csv', b'input')}):
            with self.subTest(payload=payload):
                response = self.client.post(self.url, payload)
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.context['form'].errors)
                self.assertEqual(Job.objects.count(), 0)
                self.assertFalse((self.root / 'artifacts/ui/uploads').exists())
        start.assert_not_called()

    @override_settings(FILE_UPLOAD_MAX_MEMORY_SIZE=0)
    @patch('app_main.views.start_worker')
    def test_file_only_disk_uploads_with_identical_names_do_not_overwrite(self, start):
        saved_paths = []
        for text in ('A first record.', 'A different record.'):
            content = gzip.compress((json.dumps({'text': text}) + '\n').encode('utf-8'))
            response = self.client.post(self.url, {
                'source_file': SimpleUploadedFile('same.jsonl.gz', content),
            })
            self.assertEqual(response.status_code, 302)
            job = Job.objects.get(pk=response.url.rsplit('/', 1)[-1])
            saved = Path(job.config['source_path'])
            saved_paths.append(saved)
            self.assertEqual(saved.read_bytes(), content)
            self.assertEqual(job.config['text'], '')
            self.assertTrue(saved.name.endswith('.jsonl.gz'))
            snapshot = self.root / f'{job.pk}.jsonl'
            self.assertEqual(prepare_source(job.config, snapshot, max_length=1000)['records'], 1)
            self.assertEqual(json.loads(snapshot.read_text())['example'], text)
        self.assertNotEqual(*saved_paths)
        self.assertEqual(start.call_count, 2)

    @patch('app_main.views.save_source_upload', side_effect=OSError('Disk full'))
    @patch('app_main.views.start_worker')
    def test_upload_failure_is_shown_without_queueing(self, start, save):
        response = self.client.post(self.url, {'source_file': SimpleUploadedFile('source.txt', b'Go.')})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Could not save the uploaded file')
        self.assertEqual(Job.objects.count(), 0)
        start.assert_not_called()

    @patch('app_main.views.start_worker')
    def test_retry_of_uploaded_file_reuses_the_saved_copy(self, start):
        response = self.client.post(self.url, {
            'source_file': SimpleUploadedFile('sentences.txt', b'An uploaded sentence.'),
        })
        previous = Job.objects.get()
        previous.status = Job.Status.FAILED
        previous.save(update_fields=['status'])
        saved = Path(previous.config['source_path'])
        before = saved.read_bytes()
        response = self.client.post(reverse('app_main:retry_job', args=[previous.pk]))
        new = Job.objects.exclude(pk=previous.pk).get()
        self.assertRedirects(response, reverse('app_main:job', args=[new.pk]), fetch_redirect_response=False)
        self.assertEqual(new.config, previous.config)
        self.assertEqual(saved.read_bytes(), before)
        self.assertEqual(len(list(saved.parent.iterdir())), 1)

    @patch('app_main.views.start_worker')
    def test_post_requires_csrf_and_put_is_not_supported(self, start):
        response = Client(enforce_csrf_checks=True).post(self.url, {'text': 'Go.'})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.put(self.url).status_code, 405)
        self.assertEqual(Job.objects.count(), 0)
        start.assert_not_called()

    def test_recent_jobs_are_limited_to_the_latest_template_mining_runs(self):
        start = timezone.now() - timedelta(minutes=1)
        mining_jobs = [
            Job.objects.create(operation='template_mining', created_at=start + timedelta(seconds=i))
            for i in range(12)
        ]
        unrelated = Job.objects.create(operation='build')
        response = self.client.get(self.url)
        actual = list(response.context['mining_jobs'])
        self.assertEqual([job.pk for job in actual], [job.pk for job in reversed(mining_jobs[-10:])])
        self.assertContains(response, reverse('app_main:job', args=[mining_jobs[-1].pk]))
        self.assertNotContains(response, reverse('app_main:job', args=[unrelated.pk]))

    @patch('app_main.views.start_worker')
    def test_retry_preserves_sources_and_drops_legacy_opt_out(self, start):
        previous = Job.objects.create(
            operation='template_mining',
            config={
                'operation': 'template_mining',
                'text': 'She picked up the book.',
                'source_path': str(self.source),
                'publish_to_pool': False,
            },
            status=Job.Status.FAILED,
        )
        response = self.client.post(reverse('app_main:retry_job', args=[previous.pk]))
        new = Job.objects.exclude(pk=previous.pk).get()
        self.assertRedirects(response, reverse('app_main:job', args=[new.pk]), fetch_redirect_response=False)
        self.assertEqual(new.operation, 'template_mining')
        self.assertEqual(new.config, {key: value for key, value in previous.config.items()
                                     if key != 'publish_to_pool'})
        start.assert_called_once()

    @patch('app_main.views.start_worker')
    def test_retry_with_missing_source_returns_to_template_mining(self, start):
        previous = Job.objects.create(
            operation='template_mining',
            config={'source_path': 'data/deleted.txt', 'publish_to_pool': True},
            status=Job.Status.FAILED,
        )
        response = self.client.post(reverse('app_main:retry_job', args=[previous.pk]))
        self.assertRedirects(response, self.url, fetch_redirect_response=False)
        self.assertEqual(Job.objects.count(), 1)
        start.assert_not_called()

    @patch('app_main.views.start_worker')
    def test_mining_has_dedicated_inputs_and_is_not_a_new_job_operation(self, start):
        pipeline = reverse('app_main:pipeline')
        response = self.client.get(pipeline)
        self.assertNotContains(response, '<option value="template_mining"')
        response = self.client.post(pipeline, {'operation': 'template_mining', 'text': 'Go.'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('operation', response.context['form'].errors)
        self.assertEqual(Job.objects.count(), 0)
        start.assert_not_called()

    def test_mined_job_shows_summary_and_compressed_outputs_without_corpus_link(self):
        job = Job.objects.create(operation='template_mining', status=Job.Status.SUCCEEDED)
        root = job_root(job)
        bank = root / 'bank'
        templates = bank / 'templates'
        templates.mkdir(parents=True)
        (bank / 'manifest.json').write_text(json.dumps({
            'format': 'corpus-templates-v1', 'status': 'complete',
        }), encoding='utf-8')
        (root / 'mining_summary.json').write_text(json.dumps({
            'review_status': 'UNREVIEWED', 'published_to_pool': False,
        }), encoding='utf-8')
        compressed = gzip.compress(b'{"template_id": "example"}\n')
        (templates / '00.jsonl.gz').write_bytes(compressed)
        work = root / 'work'
        work.mkdir()
        (work / 'temporary.jsonl').write_text('{}\n', encoding='utf-8')

        response = self.client.get(reverse('app_main:job', args=[job.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['result_store'], '')
        self.assertNotContains(response, 'Browse corpus records')
        self.assertContains(response, 'mining_summary.json')
        self.assertContains(response, 'UNREVIEWED')
        self.assertIsNone(response.context['quality'])
        self.assertNotContains(response, 'work/temporary.jsonl')

        download = reverse('app_main:download', args=[job.pk, 'bank/templates/00.jsonl.gz'])
        self.assertContains(response, download)
        response = self.client.get(download)
        self.assertEqual(response.status_code, 200)
        try:
            self.assertEqual(b''.join(response.streaming_content), compressed)
            self.assertIn('attachment;', response['Content-Disposition'])
        finally:
            response.close()

    def test_corpus_job_still_offers_browse_link_when_a_store_manifest_exists(self):
        job = Job.objects.create(operation='build', status=Job.Status.SUCCEEDED)
        store = job_root(job) / 'store'
        store.mkdir(parents=True)
        (store / 'manifest.json').write_text(json.dumps({'store_format': 'jsonl-v1'}), encoding='utf-8')

        response = self.client.get(reverse('app_main:job', args=[job.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Browse corpus records')
        self.assertEqual(response.context['result_store'], f'artifacts/ui/jobs/{job.pk}/store')

    def test_failed_job_still_shows_published_bank_without_a_job_summary(self):
        job = Job.objects.create(operation='template_mining', status=Job.Status.FAILED)
        published = self.root / 'data' / 'mined-templates' / str(job.pk)
        published.mkdir(parents=True)
        (published / 'publication.json').write_text(json.dumps({
            'bank_id': str(job.pk),
        }), encoding='utf-8')
        self.assertFalse((job_root(job) / 'mining_summary.json').exists())

        response = self.client.get(reverse('app_main:job', args=[job.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['published_bank'], f'data/mined-templates/{job.pk}')
        self.assertContains(response, 'Added to Built-in pool')
        self.assertContains(response, f'data/mined-templates/{job.pk}')
        self.assertContains(response, reverse('app_main:pools'))
        self.assertNotContains(response, 'Browse corpus records')
