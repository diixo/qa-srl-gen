"""Unified generation controls and compatibility for existing queued jobs."""

import gzip
import io
import json
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
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
        path = self.root / 'data/mined-templates'
        path.mkdir(parents=True)
        (path / 'manifest.json').write_text(json.dumps({
            'format': FORMAT, 'status': 'complete', 'source_sha256': 'a' * 64,
            'counts': {'templates': 123, 'verb_lemmas': 12, 'adjective_lemmas': 13,
                       'constructions': 14, 'verb_alternative_lemmas': 15,
                       'adjective_alternative_forms': 16, 'compiled_records': 200},
        }), encoding='utf-8')
        return path


class UnifiedPoolFormTests(PoolFixtures, SimpleTestCase):
    def test_build_limits_prefill_new_forms_but_preserve_explicit_values(self):
        for initial in ({}, {'operation': 'build'}):
            form = JobForm(initial=initial)
            self.assertEqual(form['limit'].value(), 1000)
            self.assertEqual(form['qa_limit'].value(), 1000)
        explicit = JobForm(initial={'operation': 'build', 'limit': 250, 'qa_limit': None})
        self.assertEqual(explicit['limit'].value(), 250)
        self.assertIsNone(explicit['qa_limit'].value())
        other = JobForm(initial={'operation': 'annotate'})
        self.assertIsNone(other['limit'].value())
        self.assertIsNone(other['qa_limit'].value())

    def test_build_limits_are_optional_positive_and_independent(self):
        for values in ({}, {'limit': '', 'qa_limit': ''}, {'limit': 1000}, {'qa_limit': 1000},
                       {'limit': 1000, 'qa_limit': 1000}):
            with self.subTest(values=values):
                form = JobForm({'operation': 'build', 'template_count': 0, **values})
                self.assertTrue(form.is_valid(), form.errors)
                for key in ('limit', 'qa_limit'):
                    self.assertEqual(form.cleaned_data[key], values.get(key) or None)
        for key in ('limit', 'qa_limit'):
            for value in (0, -1, '1.5'):
                form = JobForm({'operation': 'build', 'template_count': 0, key: value})
                self.assertFalse(form.is_valid())
                self.assertIn(key, form.errors)

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
        with patch('semantic_corpus.template_bank.TemplateBank.rows', side_effect=AssertionError('No shard reads on GET')):
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
            'limit': 1000, 'qa_limit': 2000,
        }, status=Job.Status.FAILED)
        new = self.retry(previous)
        self.assertEqual(new.config['resource_scope'], 'built_in_v1')
        self.assertEqual(new.config['count_scope'], 'per_frame')
        self.assertEqual(new.config['template_count'], 0)
        self.assertEqual(new.config['limit'], 1000)
        self.assertEqual(new.config['qa_limit'], 2000)

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
        self.assertContains(response, 'name="limit"')
        self.assertContains(response, 'name="qa_limit"')
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


class BuildLimitTests(TestCase):
    """Exercise the real store/export path with bounded input streams."""

    def build(self, *, limit=None, qa_limit=None, curated_count=2, mined_count=2):
        from semantic_corpus.documents import AnnotationRun, Document
        from semantic_corpus.question_generator.answers import QAExample
        from .operations import execute

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        pool = EntityPool([])
        seen = {'curated': 0, 'mined': 0}

        def pair(key):
            doc = Document(key, f'Anna visited place {key}.', split='train',
                           metadata={'template_semantics': {'complete': True, 'skipped': []}})
            return doc, AnnotationRun(f'{key}-run', synthetic=True)

        def curated(*args, **kwargs):
            for i in range(curated_count):
                seen['curated'] += 1
                doc, run = pair(f'curated-{i}')
                yield doc, run
                # A duplicate must not spend a document or QA quota.
                yield replace(doc, document_id=f'duplicate-{i}'), run

        def mined(**kwargs):
            stats = kwargs['stats']
            stats.update(rendered=0, eligible=0, split_excluded=0)
            # This duplicate crosses the curated/mined boundary.
            if curated_count:
                stats['rendered'] += 1
                stats['eligible'] += 1
                doc, run = pair('curated-0')
                yield replace(doc, document_id='mined-duplicate'), run
            for i in range(mined_count):
                seen['mined'] += 1
                stats['rendered'] += 1
                stats['eligible'] += 1
                yield pair(f'mined-{i}')

        def questions(doc, run):
            rows = [QAExample(context=doc.text, question=f'Which place, variant {i}?',
                    answer=doc.text, document_id=doc.document_id, run_id=run.run_id)
                    for i in range(3)]
            return [rows[0], rows[0], *rows[1:]]

        generator = SimpleNamespace(frames=[None], pool=pool, paradigms={}, generate_per_frame=lambda _: ())
        resources = SimpleNamespace(entities=pool, frames=[], paradigms={},
                                    metadata=lambda **kwargs: {}, mined_records=mined)
        with override_settings(REPO_ROOT=root, CORPUS_UI_ROOT=root / 'jobs'), \
                patch('app_main.operations.get_pool', return_value=pool), \
                patch('semantic_corpus.resource_pool.BuiltInPool', return_value=resources), \
                patch('app_main.operations.Generator', return_value=generator), \
                patch('app_main.operations.iter_canonical', side_effect=curated), \
                patch('app_main.operations.QuestionGenerator.for_document', side_effect=questions), \
                patch('app_main.operations.balance', side_effect=lambda rows, **kwargs: rows), \
                patch('sys.stdout', new_callable=io.StringIO):
            job = Job.objects.create(operation='build', config={
                'operation': 'build', 'count': curated_count, 'count_scope': 'per_frame',
                'resource_scope': 'built_in_v1', 'template_count': 1, 'seed': 42, 'split': 'train',
                'limit': limit, 'qa_limit': qa_limit}, status=Job.Status.RUNNING)
            execute(job)
            out = catalog.job_root(job)
        from semantic_corpus.storage import CorpusStore
        store = CorpusStore(out / 'store', create=False)
        summary = json.loads((out / 'generation-summary.json').read_text(encoding='utf-8'))
        for name, table in (('sft.jsonl', 'qa_examples'), ('bio.jsonl', 'documents')):
            with (out / name).open(encoding='utf-8') as stream:
                self.assertEqual(sum(1 for line in stream if line.strip()), store.count(table))
        self.assertEqual(store.count('annotation_runs'), store.count('documents'))
        self.assertEqual(summary['stored_documents'], store.count('documents'))
        self.assertEqual(summary['qa_examples'], store.count('qa_examples'))
        return store, summary, seen

    def test_document_limit_stops_before_next_curated_or_mined_record(self):
        store, summary, seen = self.build(limit=1000, curated_count=1005)
        self.assertEqual(store.count('documents'), 1000)
        self.assertEqual(store.count('qa_examples'), 3000)
        self.assertEqual(seen, {'curated': 1000, 'mined': 0})
        self.assertEqual(summary['stopped_by'], ['documents'])
        self.assertEqual(summary['mined']['rendered'], 0)

    def test_document_limit_is_shared_with_mined_and_ignores_duplicates(self):
        store, summary, seen = self.build(limit=3)
        self.assertEqual(store.count('documents'), 3)
        self.assertEqual(store.count('qa_examples'), 9)
        self.assertEqual(seen, {'curated': 2, 'mined': 1})
        self.assertEqual(summary['curated_duplicates'], 2)
        self.assertEqual(summary['mined']['duplicates'], 1)

    def test_qa_limit_stops_within_a_document_after_deduplication(self):
        store, summary, seen = self.build(qa_limit=1000, curated_count=1005)
        self.assertEqual(store.count('qa_examples'), 1000)
        self.assertEqual(store.count('documents'), 334)
        self.assertEqual(seen, {'curated': 334, 'mined': 0})
        self.assertEqual(summary['stopped_by'], ['qa_examples'])

    def test_qa_limit_is_shared_with_mined(self):
        store, summary, seen = self.build(qa_limit=8)
        self.assertEqual(store.count('qa_examples'), 8)
        self.assertEqual(store.count('documents'), 3)
        self.assertEqual(seen, {'curated': 2, 'mined': 1})
        self.assertEqual(summary['stopped_by'], ['qa_examples'])

    def test_combined_limits_and_unlimited_build(self):
        for limit, qa_limit, documents, qa, stopped in (
            (1, 5, 1, 3, ['documents']),
            (3, 4, 2, 4, ['qa_examples']),
            (2, 6, 2, 6, ['documents', 'qa_examples']),
            (None, None, 4, 12, []),
            (1000, 1000, 4, 12, []),
        ):
            with self.subTest(limit=limit, qa_limit=qa_limit):
                store, summary, seen = self.build(limit=limit, qa_limit=qa_limit)
                self.assertEqual(store.count('documents'), documents)
                self.assertEqual(store.count('qa_examples'), qa)
                self.assertEqual(summary['stopped_by'], stopped)
