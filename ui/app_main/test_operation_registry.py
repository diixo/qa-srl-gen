"""Operation metadata can evolve without changing the job table schema."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from . import operations
from .forms import JobForm
from .models import Job
from .operation_registry import OperationRegistry


class RegistryFixtures:
    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.registry_path = Path(temporary.name) / 'operations.json'

    def registry(self, entries):
        self.registry_path.write_text(json.dumps(entries), encoding='utf-8')
        return OperationRegistry(self.registry_path)

    def isolate_catalog(self):
        for name, result in (
            ('sources', []), ('stores', []), ('pool_choices', [('', 'Default pool')]),
        ):
            mock = patch(f'app_main.forms.catalog.{name}', return_value=result)
            mock.start()
            self.addCleanup(mock.stop)


class OperationRegistryTests(RegistryFixtures, SimpleTestCase):
    def test_catalog_preserves_order_and_supports_explicit_empty_selection(self):
        registry = self.registry([
            {'code': 'lookup', 'label': 'Verb forms'},
            {'code': 'build', 'label': 'Synthetic corpus'},
        ])
        self.assertEqual(registry.choices(), [
            ('lookup', 'Verb forms'), ('build', 'Synthetic corpus'),
        ])
        self.assertEqual(registry.choices(['build']), [('build', 'Synthetic corpus')])
        self.assertEqual(registry.choices([]), [])

    def test_catalog_rejects_malformed_entries(self):
        invalid_payloads = [
            {'code': 'build', 'label': 'Build'},
            [None],
            [{'code': 'build'}],
            [{'label': 'Build'}],
            [{'code': None, 'label': 'Build'}],
            [{'code': '', 'label': 'Build'}],
            [{'code': 'Build', 'label': 'Build'}],
            [{'code': '../build', 'label': 'Build'}],
            [{'code': 'build job', 'label': 'Build'}],
            [{'code': 'x' * 25, 'label': 'Build'}],
            [{'code': 'build', 'label': ''}],
            [{'code': 'build', 'label': '   '}],
            [{'code': 'build', 'label': None}],
            [{'code': 'build', 'label': 'One'}, {'code': 'build', 'label': 'Two'}],
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.registry(payload)

    def test_catalog_rejects_invalid_json(self):
        self.registry_path.write_text('[{"code":', encoding='utf-8')
        with self.assertRaises(ValueError):
            OperationRegistry(self.registry_path)

    def test_lookup_requires_known_code_and_operation_is_immutable(self):
        registry = self.registry([{'code': 'lookup', 'label': 'Verb forms'}])
        operation = registry.require('lookup')
        self.assertEqual((operation.code, operation.label), ('lookup', 'Verb forms'))
        with self.assertRaises(AttributeError):
            operation.label = 'Changed'
        with self.assertRaises(ValueError):
            registry.require('removed')
        self.assertEqual(registry.label('removed'), 'removed')

    def test_checked_in_catalog_has_executable_handlers(self):
        registry = OperationRegistry.default()
        self.assertIs(registry, OperationRegistry.default())
        self.assertTrue(registry.choices())
        self.assertIn('templates', dict(registry.choices()))
        for code, label in registry.choices():
            with self.subTest(operation=code):
                self.assertTrue(label)
                operations.validate_operation(code)

    def test_catalog_updates_do_not_change_model_field_serialization(self):
        field = Job._meta.get_field('operation')
        before = field.deconstruct()
        updated = self.registry([{'code': 'new_operation', 'label': 'New operation'}])
        with patch.object(OperationRegistry, 'default', return_value=updated):
            self.assertEqual(Job._meta.get_field('operation').deconstruct(), before)
        self.assertNotIn('choices', before[3])
        self.assertEqual(before[3]['max_length'], 24)

    def test_job_labels_follow_catalog_and_preserve_unknown_history(self):
        registry = self.registry([{'code': 'lookup', 'label': 'Renamed lookup'}])
        with patch.object(OperationRegistry, 'default', return_value=registry):
            self.assertEqual(Job(operation='lookup').get_operation_display(), 'Renamed lookup')
            self.assertEqual(Job(operation='retired').get_operation_display(), 'retired')


class OperationValidationTests(RegistryFixtures, SimpleTestCase):
    def setUp(self):
        super().setUp()
        self.isolate_catalog()

    def test_forms_load_current_catalog_on_instantiation(self):
        registry = self.registry([{'code': 'lookup', 'label': 'Current label'}])
        with patch.object(OperationRegistry, 'default', return_value=registry):
            form = JobForm({'operation': 'lookup', 'text': 'buy'})
        self.assertEqual(list(form.fields['operation'].choices), [('lookup', 'Current label')])
        with patch.object(OperationRegistry, 'default', return_value=registry):
            self.assertTrue(form.is_valid(), form.errors)

    def test_form_allowed_subset_and_empty_subset_are_enforced(self):
        form = JobForm({'operation': 'build'}, operations=['lookup'])
        self.assertEqual(list(form.fields['operation'].choices), [('lookup', 'Verb forms')])
        self.assertFalse(form.is_valid())
        self.assertIn('operation', form.errors)
        empty = JobForm({'operation': 'build'}, operations=[])
        self.assertEqual(list(empty.fields['operation'].choices), [])
        self.assertFalse(empty.is_valid())
        self.assertIn('operation', empty.errors)

    def test_form_rejects_registered_operation_without_handler(self):
        registry = self.registry([{'code': 'not_implemented', 'label': 'Future operation'}])
        with patch.object(OperationRegistry, 'default', return_value=registry):
            form = JobForm({'operation': 'not_implemented'})
            self.assertFalse(form.is_valid())
            self.assertIn('operation', form.errors)

    def assert_rejected_before_execution(self, job):
        with patch('app_main.operations.job_root') as root, \
                patch('app_main.operations.Progress') as progress:
            with self.assertRaises(ValueError):
                operations.execute(job)
        root.assert_not_called()
        progress.assert_not_called()

    def test_execute_rejects_unknown_operation_before_any_side_effects(self):
        self.assert_rejected_before_execution(Job(operation='removed', config={}))

    def test_execute_rejects_missing_handler_before_any_side_effects(self):
        with patch.dict(operations.HANDLERS, {}, clear=True):
            self.assert_rejected_before_execution(Job(operation='lookup', config={'text': 'buy'}))

    def test_execute_rejects_mismatched_config_operation_before_any_side_effects(self):
        self.assert_rejected_before_execution(Job(
            operation='lookup', config={'operation': 'build', 'text': 'buy'},
        ))


class OperationRetryTests(RegistryFixtures, TestCase):
    def setUp(self):
        super().setUp()
        self.isolate_catalog()

    def retry(self, previous):
        with patch('app_main.views.start_worker') as start:
            response = self.client.post(reverse('app_main:retry_job', args=[previous.pk]))
        return response, start

    def test_retry_uses_stored_operation_instead_of_stale_config(self):
        previous = Job.objects.create(
            operation='lookup', config={'operation': 'build', 'text': 'buy'},
            status=Job.Status.FAILED,
        )
        response, start = self.retry(previous)
        new = Job.objects.exclude(pk=previous.pk).get()
        self.assertRedirects(response, reverse('app_main:job', args=[new.pk]), fetch_redirect_response=False)
        self.assertEqual(new.operation, 'lookup')
        self.assertEqual(new.config['operation'], 'lookup')
        self.assertEqual(new.config['text'], 'buy')
        previous.refresh_from_db()
        self.assertEqual(previous.config['operation'], 'build')
        start.assert_called_once()

    def test_retry_accepts_historical_config_without_operation_key(self):
        previous = Job.objects.create(
            operation='lookup', config={'text': 'buy'}, status=Job.Status.FAILED,
        )
        response, start = self.retry(previous)
        new = Job.objects.exclude(pk=previous.pk).get()
        self.assertRedirects(response, reverse('app_main:job', args=[new.pk]), fetch_redirect_response=False)
        self.assertEqual(new.config['operation'], 'lookup')
        start.assert_called_once()

    def test_retry_rejects_removed_operation_even_if_config_looks_valid(self):
        previous = Job.objects.create(
            operation='retired', config={'operation': 'build', 'count': 1},
            status=Job.Status.FAILED,
        )
        response, start = self.retry(previous)
        self.assertRedirects(response, reverse('app_main:pipeline'), fetch_redirect_response=False)
        self.assertEqual(Job.objects.count(), 1)
        start.assert_not_called()

    def test_retry_rejects_operation_without_handler(self):
        previous = Job.objects.create(
            operation='lookup', config={'operation': 'lookup', 'text': 'buy'},
            status=Job.Status.FAILED,
        )
        with patch.dict(operations.HANDLERS, {}, clear=True):
            response, start = self.retry(previous)
        self.assertRedirects(response, reverse('app_main:pipeline'), fetch_redirect_response=False)
        self.assertEqual(Job.objects.count(), 1)
        start.assert_not_called()
