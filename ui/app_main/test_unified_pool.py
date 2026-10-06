"""Unified generation controls and compatibility for existing queued jobs."""

import gzip
import io
import json
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from semantic_corpus.template_bank import FORMAT
from semantic_corpus.semantic_generator import default_pool
from semantic_corpus.semantic_generator.substitutions import EntityPool

from . import catalog
from .forms import JobForm
from .models import Job


class PoolFixtures:
    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        settings = override_settings(REPO_ROOT=self.root, CORPUS_UI_ROOT=self.root / 'artifacts/ui')
        settings.enable()
        self.addCleanup(settings.disable)
        # Keep realistic type coverage without repeatedly splitting thousands
        # of proper names in tests of UI controls and rendered summaries.
        counts = Counter()
        entities = []
        for entity in default_pool():
            if counts[entity.entity_label] < 20:
                entities.append(entity)
                counts[entity.entity_label] += 1
        mocked_pool = patch('app_main.catalog.default_pool', return_value=EntityPool(entities))
        mocked_pool.start()
        self.addCleanup(mocked_pool.stop)
        for name, result in (('sources', []), ('stores', []), ('pool_choices', [('', 'Built-in pool')])):
            mocked = patch(f'app_main.forms.catalog.{name}', return_value=result)
            mocked.start()
            self.addCleanup(mocked.stop)

    def bank(self):
        path = self.root / 'data/eng-base-templates'
        path.mkdir(parents=True)
        (path / 'manifest.json').write_text(json.dumps({
            'format': FORMAT, 'status': 'complete', 'source_sha256': 'a' * 64,
            'counts': {'templates': 123, 'verb_lemmas': 12, 'adjective_lemmas': 13,
                       'constructions': 14, 'verb_alternative_lemmas': 15,
                       'adjective_alternative_forms': 16, 'compiled_records': 200},
        }), encoding='utf-8')
        return path


class UnifiedPoolFormTests(PoolFixtures, SimpleTestCase):
    def test_annotation_form_selects_pending_documents_initially_and_preserves_explicit_choice(self):
        initial = JobForm(initial={'operation': 'annotate'})
        self.assertTrue(initial['pending_only'].value())
        for value, expected in (('on', True), ('', False)):
            form = JobForm({'operation': 'annotate', 'store': 'artifacts/source', 'pending_only': value})
            form.fields['store'].choices = [('artifacts/source', 'Stored corpus')]
            with patch('app_main.forms.catalog.store_path', return_value=self.root):
                self.assertTrue(form.is_valid(), form.errors)
            self.assertEqual(form.cleaned_data['pending_only'], expected)

    def test_synthetic_corpus_includes_mined_resources_by_default(self):
        self.bank()
        form = JobForm({'operation': 'build', 'count': 2,
                        'template_verb': ' Pick Up ', 'template_adjective': ' Beautiful '})
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data['resource_scope'], 'built_in_v1')
        self.assertEqual(form.cleaned_data['count_scope'], 'per_frame')
        self.assertEqual(form.cleaned_data['count'], 2)
        self.assertEqual(form.cleaned_data['template_count'], 1)
        for name in ('template_verb', 'template_adjective'):
            self.assertNotIn(name, form.fields)
            self.assertNotIn(name, form.cleaned_data)
        self.assertEqual(form.fields['template_count'].widget.attrs['data-template-count'], 123)

    def test_missing_bank_requires_explicit_skip(self):
        missing = JobForm({'operation': 'build'})
        self.assertFalse(missing.is_valid())
        self.assertIn('template_count', missing.errors)
        skipped = JobForm({'operation': 'build', 'template_count': 0})
        self.assertTrue(skipped.is_valid(), skipped.errors)
        self.assertEqual(skipped.cleaned_data['template_count'], 0)
        self.assertEqual(skipped.cleaned_data['resource_scope'], 'built_in_v1')

    def test_negative_template_count_is_rejected(self):
        form = JobForm({'operation': 'build', 'template_count': -1})
        self.assertFalse(form.is_valid())
        self.assertIn('template_count', form.errors)

    def test_separate_template_mode_is_hidden_but_explicit_legacy_form_is_supported(self):
        self.bank()
        default = JobForm({'operation': 'templates'})
        self.assertNotIn('templates', dict(default.fields['operation'].choices))
        self.assertFalse(default.is_valid())
        legacy = JobForm({'operation': 'templates'}, operations=['templates'])
        self.assertTrue(legacy.is_valid(), legacy.errors)
        zero = JobForm({'operation': 'templates', 'template_count': 0}, operations=['templates'])
        self.assertFalse(zero.is_valid())
        self.assertIn('template_count', zero.errors)

    def test_resource_monitor_keeps_all_inventory_counts_without_reading_shards(self):
        self.bank()
        with patch('semantic_corpus.resource_pool.read_rows', side_effect=AssertionError('No shard reads on GET')):
            summary = catalog.resource_summary()
        self.assertTrue(summary['available'])
        self.assertGreater(summary['entities'], 0)
        self.assertGreater(summary['frames'], 0)
        self.assertEqual({key: summary[key] for key in (
            'templates', 'verbs', 'adjectives', 'constructions', 'verb_alternatives',
            'adjective_alternatives', 'records')},
            dict(templates=123, verbs=12, adjectives=13, constructions=14,
                 verb_alternatives=15, adjective_alternatives=16, records=200))


class UnifiedPoolViewTests(PoolFixtures, TestCase):
    def test_pending_annotation_filters_before_limit_and_preserves_curated_annotations(self):
        from semantic_corpus.documents import Document
        from semantic_corpus.semantic_annotator import CandidateResources
        from semantic_corpus.semantic_generator import DEFAULT_FRAMES, Generator
        from semantic_corpus.semantic_generator.canonical import to_canonical
        from semantic_corpus.question_generator import QuestionGenerator
        from semantic_corpus.semantic_generator.generator import DEFAULT_PARADIGMS
        from semantic_corpus.storage import CorpusStore
        from .operations import execute

        source = CorpusStore(self.root / 'artifacts/source')
        generator = Generator(seed=42, frames=(DEFAULT_FRAMES[0],), pool=catalog.get_pool(), split='train')
        curated, gold = to_canonical(next(generator.generate(1)), document_id='curated',
                                    split='train', random_seed=42)
        source.add_document(curated)
        source.add_run(gold, curated)
        questions = QuestionGenerator(seed=42, paradigms=DEFAULT_PARADIGMS)
        source.add_examples(questions.for_document(curated, gold), generation_run_id=gold.run_id)
        pending = Document('pending', 'Hello!', source='eng-base-templates', split='train',
                           metadata={'review_status': 'UNREVIEWED'})
        source.add_document(pending)
        original = source.content_hash()
        original_runs = list(source.runs(document_id=curated.document_id))
        original_questions = list(source.examples())
        form = JobForm({'operation': 'annotate', 'store': 'artifacts/source',
                        'pending_only': 'on', 'limit': 1, 'teacher': 'rules'})
        form.fields['store'].choices = [('artifacts/source', 'Stored corpus')]
        self.assertTrue(form.is_valid(), form.errors)
        job = Job.objects.create(operation='annotate', config=form.cleaned_data, status=Job.Status.RUNNING)
        with patch('semantic_corpus.semantic_annotator.CandidateResources.load', return_value=CandidateResources()), \
                patch('sys.stdout', new_callable=io.StringIO):
            execute(job)
        target = CorpusStore(catalog.job_root(job) / 'store', create=False)
        self.assertEqual(list(target.runs(document_id=curated.document_id)), original_runs)
        self.assertEqual(len(list(target.runs(document_id=pending.document_id))), 1)
        preserved_questions = [example for example in target.examples()
                               if example.document_id == curated.document_id]
        self.assertEqual(preserved_questions, original_questions)
        self.assertEqual(CorpusStore(self.root / 'artifacts/source', create=False).content_hash(), original)
        job.refresh_from_db()
        self.assertEqual(job.processed, 1)

    def test_unified_build_stores_mined_documents_without_exporting_unverified_labels(self):
        from semantic_corpus.storage import CorpusStore
        from semantic_corpus.template_bank import fingerprint, read_rows
        from .operations import execute

        bank = self.bank()
        (bank / 'templates').mkdir()
        record = dict(parts=[{'literal': 'A '}, {'slot': 'a0'}, {'literal': ' book.'}],
            slots={'a0': {'kind': 'adjective', 'pos': 'ADJ', 'tag': 'JJ'}}, verbs=[], constructions=[],
            predicate_groups=[], attributes=[{'slot': 'a0', 'target_slot': None, 'degree': ['Pos']}],
            adjective_lemmas=['beautiful'], defaults={'a0': 'beautiful'},
            attested_bindings=[{'a0': 'beautiful'}], fillers={'a0': ['beautiful']})
        record['template_id'] = fingerprint(record)
        shard = bank / 'templates' / (record['template_id'][:2] + '.jsonl.gz')
        with gzip.open(shard, 'wt', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        resources = catalog.get_resources()
        selected_split = next(split for split in ('train', 'dev', 'test')
                              if list(resources.mined_documents(count=1, split=split)))
        # Select the source document's deterministic split, then check the real
        # operation stores both sources but exports only annotated documents.
        form = JobForm({'operation': 'build', 'count': 1, 'template_count': 1, 'split': selected_split})
        self.assertTrue(form.is_valid(), form.errors)
        job = Job.objects.create(operation='build', config=form.cleaned_data, status=Job.Status.RUNNING)
        with patch('sys.stdout', new_callable=io.StringIO):
            execute(job)
        root = catalog.job_root(job)
        store = CorpusStore(root / 'store', create=False)
        mined = [doc for doc in store.documents() if doc.source == 'eng-base-templates']
        self.assertEqual(len(mined), 1)
        self.assertEqual(mined[0].text, 'A beautiful book.')
        self.assertEqual(mined[0].metadata['review_status'], 'UNREVIEWED')
        self.assertEqual(list(store.runs(document_id=mined[0].document_id)), [])
        curated = [doc for doc in store.documents() if doc.source != 'eng-base-templates']
        self.assertEqual(len(curated), len(resources.frames))
        self.assertGreater(store.count('qa_examples'), 0)
        bio = list(read_rows(root / 'bio.jsonl'))
        self.assertNotIn(mined[0].document_id, {row['document_id'] for row in bio})
        summary = json.loads((root / 'generation-summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['pending_annotation'], 1)
        self.assertEqual(summary['mined']['stored'], 1)

    def retry(self, previous):
        with patch('app_main.views.start_worker') as start:
            response = self.client.post(reverse('app_main:retry_job', args=[previous.pk]))
        new = Job.objects.exclude(pk=previous.pk).get()
        self.assertRedirects(response, reverse('app_main:job', args=[new.pk]), fetch_redirect_response=False)
        start.assert_called_once()
        return new

    def test_retry_of_old_build_does_not_add_mined_resources_or_change_quota(self):
        previous = Job.objects.create(operation='build', config={'count': 2}, status=Job.Status.FAILED)
        new = self.retry(previous)
        self.assertNotIn('resource_scope', new.config)
        self.assertEqual(new.config['count_scope'], 'total')
        self.assertEqual(new.config['count'], 2)

    def test_retry_preserves_unified_scope_and_explicit_skip(self):
        previous = Job.objects.create(operation='build', config={
            'count': 3, 'count_scope': 'per_frame', 'resource_scope': 'built_in_v1', 'template_count': 0,
        }, status=Job.Status.FAILED)
        new = self.retry(previous)
        self.assertEqual(new.config['resource_scope'], 'built_in_v1')
        self.assertEqual(new.config['count_scope'], 'per_frame')
        self.assertEqual(new.config['template_count'], 0)

    def test_historical_template_job_can_still_be_retried(self):
        self.bank()
        previous = Job.objects.create(operation='templates', config={
            'template_count': 1, 'template_verb': 'pick up', 'template_adjective': 'beautiful',
        }, status=Job.Status.FAILED)
        new = self.retry(previous)
        self.assertEqual(new.operation, 'templates')
        self.assertEqual(new.config['template_verb'], 'pick up')
        self.assertEqual(new.config['template_adjective'], 'beautiful')

    def test_pages_show_shared_pool_without_promoting_separate_mode(self):
        self.bank()
        response = self.client.get(reverse('app_main:pipeline'))
        self.assertEqual(response.status_code, 200)
        for name in ('template_verb', 'template_adjective'):
            self.assertNotContains(response, f'name="{name}"')
        for route in ('pools', 'slots'):
            with self.subTest(route=route):
                response = self.client.get(reverse('app_main:' + route))
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, 'Built-in pool resources')
                self.assertContains(response, 'Alternative adjective / participle entries')
                self.assertContains(response, 'UNREVIEWED')
                self.assertNotContains(response, 'Corpus templates: verbs and adjectives')
