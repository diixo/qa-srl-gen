"""Dedicated chat generation: page, queue, worker output and downloads."""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from django.test import Client, SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from semantic_corpus.dialogue_generator import TASKS, USER_ASSISTANT_TASKS

from .catalog import job_root
from .chat_dataset import ChatDatasetForm
from .models import Job
from .operations import execute


class ChatDatasetFormTests(SimpleTestCase):
    def test_user_only_defaults_and_legacy_jobs_keep_their_task_sets(self):
        config = {'train': 1, 'validation': 1, 'test': 1, 'seed': 42}
        self.assertTrue(ChatDatasetForm()['user_assistant_only'].value())
        for form, expected in (
            (ChatDatasetForm(config), False),
            (ChatDatasetForm({**config, 'user_assistant_only': 'on'}), True),
            (ChatDatasetForm.from_config(config), False),
            (ChatDatasetForm.from_config({**config, 'user_assistant_only': True}), True),
        ):
            self.assertTrue(form.is_valid(), form.errors)
            self.assertIs(form.cleaned_data['user_assistant_only'], expected)
            tasks = USER_ASSISTANT_TASKS if expected else TASKS
            self.assertEqual(form.tasks(), tasks)
            self.assertEqual(form.preview()['train'], len(tasks))

    def test_metadata_defaults_differ_for_new_posts_and_legacy_configs(self):
        config = {'train': 1, 'validation': 1, 'test': 1, 'seed': 42}
        self.assertTrue(ChatDatasetForm()['include_metadata'].value())
        for form, expected in (
            (ChatDatasetForm(config), False),
            (ChatDatasetForm({**config, 'include_metadata': 'on'}), True),
            (ChatDatasetForm.from_config(config), True),
            (ChatDatasetForm.from_config({**config, 'include_metadata': False}), False),
        ):
            self.assertTrue(form.is_valid(), form.errors)
            self.assertIs(form.cleaned_data['include_metadata'], expected)

    def test_defaults_and_invalid_quota_previews(self):
        self.assertEqual(ChatDatasetForm().preview(), {
            'train': 200 * len(USER_ASSISTANT_TASKS), 'validation': 20 * len(USER_ASSISTANT_TASKS),
            'test': 20 * len(USER_ASSISTANT_TASKS), 'total': 240 * len(USER_ASSISTANT_TASKS),
        })
        for value in ('', '0', '-1', '1.5', '1000001', 'bad'):
            with self.subTest(value=value):
                form = ChatDatasetForm({'train': value, 'validation': 2, 'test': 3, 'seed': 42})
                self.assertFalse(form.is_valid())
                self.assertIsNone(form.preview()['train'])
                self.assertIsNone(form.preview()['total'])
                self.assertEqual(form.preview()['validation'], 2 * len(TASKS))

    def test_seed_zero_is_preserved_and_invalid_seed_is_rejected(self):
        config = {'train': 1, 'validation': 1, 'test': 1, 'seed': 0}
        form = ChatDatasetForm(config)
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data['seed'], 0)
        for seed in ('', -1, 2147483648):
            self.assertFalse(ChatDatasetForm({**config, 'seed': seed}).is_valid())


class ChatDatasetPageTests(TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        overrides = override_settings(CORPUS_UI_ROOT=self.root / 'ui')
        overrides.enable()
        self.addCleanup(overrides.disable)
        self.url = reverse('app_main:chat_dataset')
        self.config = {'train': 1, 'validation': 1, 'test': 1, 'seed': 42,
                       'include_metadata': True, 'user_assistant_only': False}

    @patch('app_main.views.start_worker')
    def test_page_and_sidebar_render_without_queueing(self, start):
        self.assertEqual(self.url, '/chat-dataset')
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<h1>Chat dataset</h1>', html=True)
        self.assertContains(response, 'name="train"')
        self.assertContains(response, 'name="validation"')
        self.assertContains(response, 'name="test"')
        self.assertContains(response, 'name="seed"')
        self.assertContains(response, 'name="include_metadata"')
        self.assertContains(response, 'Include metadata')
        self.assertContains(response, 'plain-text facts')
        self.assertContains(response, 'Inputs and targets still use JSON')
        self.assertTrue(response.context['form']['include_metadata'].value())
        self.assertContains(response, f'data-task-count="{len(USER_ASSISTANT_TASKS)}"')
        self.assertContains(response, 'User and assistant only')
        self.assertContains(response, 'data-chat-people-only hidden')
        self.assertTrue(response.context['form']['user_assistant_only'].value())
        self.assertContains(response, 'aria-current="page"')
        self.assertEqual(response.context['nav_active'], 'chat_dataset')
        self.assertEqual(sum(not row['hidden'] for row in response.context['task_rows']), len(USER_ASSISTANT_TASKS))
        self.assertEqual(Job.objects.count(), 0)
        start.assert_not_called()
        # A separate page, not an option backed by the generic corpus form.
        from .forms import JobForm
        with patch('app_main.forms.catalog.sources', return_value=[]), \
                patch('app_main.forms.catalog.stores', return_value=[]):
            self.assertNotIn('chat_dataset', dict(JobForm().fields['operation'].choices))

    @patch('app_main.views.start_worker')
    def test_post_queues_only_validated_parameters_and_redirects(self, start):
        response = self.client.post(self.url, {**self.config, 'operation': 'build', 'out': '../bad'})
        job = Job.objects.get()
        self.assertRedirects(response, reverse('app_main:job', args=[job.pk]), fetch_redirect_response=False)
        self.assertEqual(job.operation, 'chat_dataset')
        self.assertEqual(job.config, {**self.config, 'operation': 'chat_dataset'})
        self.assertEqual(job.status, Job.Status.QUEUED)
        self.assertFalse(job_root(job).exists())
        start.assert_called_once()
        response = self.client.get(self.url)
        self.assertContains(response, reverse('app_main:job', args=[job.pk]))

    @patch('app_main.views.start_worker')
    def test_invalid_post_and_csrf_do_not_queue(self, start):
        for config in ({}, {**self.config, 'validation': 0}, {**self.config, 'seed': -1}):
            response = self.client.post(self.url, config)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.context['form'].errors)
        self.assertEqual(Client(enforce_csrf_checks=True).post(self.url, self.config).status_code, 403)
        self.assertEqual(self.client.put(self.url, self.config).status_code, 405)
        self.assertEqual(Job.objects.count(), 0)
        start.assert_not_called()

    @patch('app_main.views.start_worker', side_effect=OSError('Worker unavailable'))
    def test_worker_start_failure_keeps_queued_job_and_explains_it(self, start):
        response = self.client.post(self.url, self.config, follow=True)
        self.assertEqual(Job.objects.get().status, Job.Status.QUEUED)
        self.assertContains(response, 'worker could not start')
        self.assertContains(response, 'Worker unavailable')

    @patch('app_main.views.start_worker')
    def test_user_only_mode_is_saved_and_preserved_on_retry(self, start):
        self.client.post(self.url, {**self.config, 'user_assistant_only': 'on'})
        previous = Job.objects.get()
        self.assertIs(previous.config['user_assistant_only'], True)
        previous.status = Job.Status.FAILED
        previous.save(update_fields=['status'])
        self.client.post(reverse('app_main:retry_job', args=[previous.pk]))
        retry = Job.objects.exclude(pk=previous.pk).get()
        self.assertEqual(retry.config, previous.config)

    @patch('app_main.views.start_worker')
    def test_retry_preserves_chat_settings_in_a_new_job(self, start):
        legacy_config = {key: value for key, value in self.config.items()
                         if key not in ('include_metadata', 'user_assistant_only')}
        previous = Job.objects.create(operation='chat_dataset', config={**legacy_config, 'operation': 'build'},
                                      status=Job.Status.FAILED)
        response = self.client.post(reverse('app_main:retry_job', args=[previous.pk]))
        job = Job.objects.exclude(pk=previous.pk).get()
        self.assertRedirects(response, reverse('app_main:job', args=[job.pk]), fetch_redirect_response=False)
        self.assertEqual(job.config, {**self.config, 'operation': 'chat_dataset'})
        self.assertNotEqual(job_root(previous), job_root(job))
        start.assert_called_once()

    @patch('app_main.views.start_worker')
    def test_unchecked_metadata_is_saved_and_preserved_on_retry(self, start):
        config = {key: value for key, value in self.config.items() if key != 'include_metadata'}
        self.client.post(self.url, config)
        previous = Job.objects.get()
        self.assertIs(previous.config['include_metadata'], False)
        previous.status = Job.Status.FAILED
        previous.save(update_fields=['status'])
        self.client.post(reverse('app_main:retry_job', args=[previous.pk]))
        retry = Job.objects.exclude(pk=previous.pk).get()
        self.assertEqual(retry.config, previous.config)
        self.assertIs(retry.config['include_metadata'], False)

    @patch('app_main.views.start_worker')
    def test_invalid_retry_returns_to_chat_page_and_active_retry_is_refused(self, start):
        for status, config, route in (
            (Job.Status.FAILED, {**self.config, 'test': 0}, self.url),
            (Job.Status.QUEUED, self.config, None),
        ):
            previous = Job.objects.create(operation='chat_dataset', config=config, status=status)
            count = Job.objects.count()
            response = self.client.post(reverse('app_main:retry_job', args=[previous.pk]))
            self.assertRedirects(response, route or reverse('app_main:job', args=[previous.pk]),
                                 fetch_redirect_response=False)
            self.assertEqual(Job.objects.count(), count)
        start.assert_not_called()

    def test_real_generation_shows_results_and_downloads_complete_archive(self):
        legacy_config = {key: value for key, value in self.config.items()
                         if key not in ('include_metadata', 'user_assistant_only')}
        job = Job.objects.create(operation='chat_dataset', config=legacy_config, status=Job.Status.RUNNING)
        with patch('sys.stdout', new_callable=io.StringIO):
            execute(job)
        job.refresh_from_db()
        self.assertEqual(job.phase, 'Chat dataset ready')
        self.assertEqual((job.processed, job.total), (3 * len(TASKS),) * 2)
        self.assertEqual(job.metrics['train'], len(TASKS))
        job.status = Job.Status.SUCCEEDED
        job.save(update_fields=['status'])
        root = job_root(job)
        manifest = json.loads((root / 'dataset/manifest.json').read_text(encoding='utf-8'))
        self.assertIs(manifest['include_metadata'], True)
        self.assertIs(manifest['user_assistant_only'], False)
        self.assertEqual(manifest['content_formats']['memory'], 'memory-text-v2')
        rows = json.loads((root / 'dataset/memory/train.json').read_text(encoding='utf-8'))
        for row in rows:
            fields = {'messages', 'id', 'scenario_key', 'task'}
            if row['task'] in ('desire_vs_done', 'suggestion_vs_done'):
                fields |= {'contrast_group', 'condition', 'answer_kind'}
                self.assertIn(row['answer_kind'], ('yes', 'no', 'unknown'))
            self.assertEqual(set(row), fields)
        for row in rows:
            content = row['messages'][1]['content']
            self.assertTrue(content.startswith('Known facts:\n'))
            self.assertNotIn('{', content)
        with ZipFile(root / 'chat-dataset.zip') as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(set(archive.namelist()), set(manifest['files']) | {'manifest.json'})
            for name in archive.namelist():
                self.assertEqual(archive.read(name), (root / 'dataset' / name).read_bytes())
        response = self.client.get(reverse('app_main:job', args=[job.pk]))
        self.assertContains(response, 'Chat dataset results')
        self.assertContains(response, 'Start with Memory')
        self.assertContains(response, 'Download dataset ZIP')
        self.assertNotContains(response, 'Result quality')
        for name in ('chat-dataset.zip', 'dataset/memory/train.json', 'dataset/history/test.json',
                     'dataset/extraction/validation.json'):
            url = reverse('app_main:download', args=[job.pk, name])
            self.assertContains(response, url)
            downloaded = self.client.get(url)
            self.assertEqual(downloaded.status_code, 200)
            self.assertEqual(b''.join(downloaded.streaming_content), (root / name).read_bytes())
            downloaded.close()
        status = self.client.get(reverse('app_main:job_status', args=[job.pk])).json()
        self.assertEqual(status['percent'], 100)
        self.assertFalse(status['active'])

    def test_worker_and_archive_export_only_messages_when_metadata_is_disabled(self):
        job = Job.objects.create(operation='chat_dataset', config={**self.config, 'include_metadata': False,
                                                                 'user_assistant_only': True})
        with patch('sys.stdout', new_callable=io.StringIO):
            execute(job)
        root = job_root(job)
        manifest = json.loads((root / 'dataset/manifest.json').read_text(encoding='utf-8'))
        self.assertIs(manifest['include_metadata'], False)
        self.assertIs(manifest['user_assistant_only'], True)
        self.assertEqual(manifest['tasks'], list(USER_ASSISTANT_TASKS))
        self.assertEqual(manifest['splits']['train']['episodes'], len(USER_ASSISTANT_TASKS))
        with ZipFile(root / 'chat-dataset.zip') as archive:
            for view in ('history', 'memory', 'extraction'):
                for split in ('train', 'validation', 'test'):
                    name = f'{view}/{split}.json'
                    self.assertEqual(archive.read(name), (root / 'dataset' / name).read_bytes())
                    rows = json.loads(archive.read(name))
                    self.assertTrue(rows)
                    self.assertTrue(all(set(row) == {'messages'} for row in rows))
                    if view == 'memory':
                        for row in rows:
                            content = row['messages'][1]['content']
                            self.assertTrue(content.startswith('Known facts:\n'))
                            self.assertNotIn('turn_id', content)
                            self.assertNotIn('{', content)

    def test_bad_saved_settings_and_cancellation_do_not_publish_results(self):
        for config, cancelled, message in (
            ({**self.config, 'train': 0}, False, 'Invalid chat dataset settings'),
            (self.config, True, 'cancelled'),
        ):
            job = Job.objects.create(operation='chat_dataset', config=config, cancel_requested=cancelled)
            with patch('sys.stdout', new_callable=io.StringIO), self.assertRaisesRegex(ValueError, message):
                execute(job)
            self.assertFalse((job_root(job) / 'dataset').exists())
            self.assertFalse((job_root(job) / 'chat-dataset.zip').exists())
