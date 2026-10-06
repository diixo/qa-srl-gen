"""Synthetic corpus creates mined annotations, QA and exports in one UI job."""

import gzip
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase, override_settings
from django.urls import reverse

from semantic_corpus.resource_pool import BuiltInPool
from semantic_corpus.semantic_generator import Generator
from semantic_corpus.storage import CorpusStore
from semantic_corpus.template_bank import FORMAT, fingerprint
from semantic_corpus.template_semantics import SOURCE

from . import catalog
from .models import Job
from .operations import execute


class MinedSemanticJobTests(TestCase):
    def test_annotation_replaces_candidate_qa_and_preserves_unselected_documents(self):
        from semantic_corpus.documents import AnnotationRun, Document
        from semantic_corpus.question_generator import QuestionGenerator
        from semantic_corpus.question_generator.answers import QAExample, QAKind
        from semantic_corpus.semantic_annotator import CandidateResources

        with tempfile.TemporaryDirectory() as temp, override_settings(
                REPO_ROOT=Path(temp), CORPUS_UI_ROOT=Path(temp) / 'artifacts/ui'):
            root = Path(temp)
            source = CorpusStore(root / 'artifacts/source')
            for key, name, pending in [('gold', 'Anna', False), ('partial', 'Ben', True), ('later', 'Tom', True)]:
                doc = Document(key, f'{name} bought a book.', split='train',
                    metadata={'template_semantics': {'complete': not pending}})
                run = AnnotationRun(f'{key}-old', model_name=SOURCE if pending else 'curated')
                source.add_document(doc)
                source.add_run(run, doc)
                source.add_examples([QAExample(context=doc.text, question='What was bought?', answer='A book.',
                    kind=QAKind.ATOMIC, document_id=key, run_id=run.run_id,
                    metadata={'source': 'old'})], generation_run_id=run.run_id)
            old_doc = source.document('partial')
            source.add_examples([QAExample(context=old_doc.text, question='What was sold?', answer='A book.',
                kind=QAKind.ATOMIC, document_id='partial', run_id='partial-old')])
            original_hash = source.content_hash()
            preserved = [row for row in source.read('qa_examples') if row['document_id'] != 'partial']
            corrected = AnnotationRun('corrected', model_name='teacher')
            replacement = QAExample(context=old_doc.text, question='What was bought?', answer='A book.',
                kind=QAKind.ATOMIC, document_id='partial', run_id='corrected', metadata={'source': 'teacher'})
            job = Job.objects.create(operation='annotate', config={
                'operation': 'annotate', 'store': str(root / 'artifacts/source'), 'seed': 42,
                'split': 'train', 'teacher': 'rules', 'pending_only': True, 'limit': 1},
                status=Job.Status.RUNNING)
            with patch('semantic_corpus.semantic_annotator.CandidateResources.load', return_value=CandidateResources()), \
                    patch('semantic_corpus.semantic_annotator.annotate_document',
                          return_value=SimpleNamespace(ok=True, run=corrected, rejected=())) as annotate, \
                    patch.object(QuestionGenerator, 'for_document', return_value=[replacement]), \
                    patch('sys.stdout', new_callable=io.StringIO):
                execute(job)
            self.assertEqual(annotate.call_count, 1)
            self.assertEqual(annotate.call_args.args[0].document_id, 'partial')
            target = CorpusStore(catalog.job_root(job) / 'store', create=False)
            self.assertEqual([q for q in target.examples() if q.document_id == 'partial'], [replacement])
            self.assertEqual([row for row in target.read('qa_examples') if row['document_id'] != 'partial'], preserved)
            self.assertEqual(len(list(target.runs(document_id='partial'))), 2)
            self.assertEqual(CorpusStore(root / 'artifacts/source', create=False).content_hash(), original_hash)

    def test_build_and_question_regeneration_keep_mined_qa(self):
        with tempfile.TemporaryDirectory() as temp, override_settings(
                REPO_ROOT=Path(temp), CORPUS_UI_ROOT=Path(temp) / 'artifacts/ui'):
            root = Path(temp)
            record = dict(parts=[{'slot': 'n0'}, {'literal': ' '}, {'slot': 'v0'},
                                 {'literal': ' a '}, {'slot': 'n1'}, {'literal': '.'}],
                slots={
                    'n0': {'kind': 'nominal', 'pos': 'PROPN', 'tag': 'NNP', 'dependency': 'nsubj', 'head_slot': 'v0'},
                    'v0': {'kind': 'verb', 'pos': 'VERB', 'tag': 'VBD', 'lemma': 'buy', 'dependency': 'ROOT', 'head_slot': 'v0'},
                    'n1': {'kind': 'nominal', 'pos': 'NOUN', 'tag': 'NN', 'dependency': 'dobj', 'head_slot': 'v0'}},
                verbs=['buy'], constructions=[], attributes=[],
                predicate_groups=[{'head_slot': 'v0', 'lemma': 'buy', 'constructions': []}])
            record['template_id'] = fingerprint(record)
            bindings = {'n0': 'Anna', 'v0': 'bought', 'n1': 'book'}
            record.update(defaults=bindings, attested_bindings=[bindings], fillers={k: [v] for k, v in bindings.items()},
                          count=1, adjective_lemmas=[])
            bank = root / 'data/eng-base-templates'
            (bank / 'templates').mkdir(parents=True)
            with gzip.open(bank / 'templates' / (record['template_id'][:2] + '.jsonl.gz'), 'wt', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
            (bank / 'manifest.json').write_text(json.dumps({'format': FORMAT, 'status': 'complete',
                'source_sha256': 'a' * 64, 'counts': {'templates': 1, 'compiled_records': 1}}), encoding='utf-8')
            resources = BuiltInPool(root)
            split = next(s for s in ('train', 'dev', 'test') if list(resources.mined_documents(split=s)))
            config = dict(operation='build', resource_scope='built_in_v1', count_scope='per_frame',
                          count=1, template_count=1, seed=42, split=split)
            job = Job.objects.create(operation='build', config=config, status=Job.Status.RUNNING)
            with patch.object(Generator, 'generate_per_frame', return_value=iter(())), \
                    patch('sys.stdout', new_callable=io.StringIO):
                execute(job)
            store = CorpusStore(catalog.job_root(job) / 'store', create=False)
            self.assertEqual(store.count('documents'), 1)
            self.assertEqual(store.count('annotation_runs'), 1)
            self.assertEqual(store.count('qa_examples'), 2)
            run, = store.runs()
            self.assertEqual(run.model_name, SOURCE)
            questions = {qa.question: qa.answer for qa in store.examples()}
            self.assertEqual(questions['What did Anna buy?'], 'A book.')
            summary = json.loads((catalog.job_root(job) / 'generation-summary.json').read_text())
            self.assertEqual(summary['mined']['qa_examples'], 2)
            self.assertEqual(summary['pending_annotation'], 0)
            self.assertTrue((catalog.job_root(job) / 'sft.jsonl').read_text().strip())
            self.assertFalse((catalog.job_root(job) / 'bio.jsonl').read_text().strip())
            Job.objects.filter(pk=job.pk).update(status=Job.Status.SUCCEEDED)
            response = self.client.get(reverse('app_main:job', args=[job.pk]))
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, 'Mined template QA')
            regenerate = Job.objects.create(operation='questions', config={
                'operation': 'questions', 'store': str(catalog.job_root(job) / 'store'), 'seed': 42, 'split': split},
                status=Job.Status.RUNNING)
            with patch('sys.stdout', new_callable=io.StringIO):
                execute(regenerate)
            rebuilt = CorpusStore(catalog.job_root(regenerate) / 'store', create=False)
            self.assertEqual({qa.question: qa.answer for qa in rebuilt.examples()}, questions)
