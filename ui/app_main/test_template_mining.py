"""Source coverage and the worker's compile/validate/publish boundary."""

import gzip
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase, override_settings

from semantic_corpus.resource_pool import BuiltInPool
from semantic_corpus.semantic_generator.substitutions import EntityPool
from semantic_corpus.template_bank import TemplateBank

from .models import Job
from .template_mining import execute_template_mining, prepare_source


class MiningFixtures:
    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'data').mkdir()
        self.settings_override = override_settings(
            REPO_ROOT=self.root, CORPUS_UI_ROOT=self.root / 'artifacts/ui')
        self.settings_override.enable()
        self.addCleanup(self.settings_override.disable)


class SourceSnapshotTests(MiningFixtures, SimpleTestCase):
    def test_combines_pasted_text_and_utf8_txt_preserving_text_and_line_numbers(self):
        source = self.root / 'data/source.txt'
        source.write_bytes('\ufeff  A file record.  \r\n\r\nSecond record.\r\n'.encode('utf-8'))
        output = self.root / 'source.jsonl'
        counts = prepare_source({'text': '  Pasted.  \r\n\nLast pasted.', 'source_path': str(source)},
                                output, max_length=1000)
        rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
        self.assertEqual([r['example'] for r in rows],
                         ['  Pasted.  ', 'Last pasted.', '  A file record.  ', 'Second record.'])
        self.assertEqual([r['idx'] for r in rows], [0, 1, 2, 3])
        self.assertEqual([r['origin']['line'] for r in rows], [1, 3, 1, 3])
        self.assertEqual(counts['records'], 4)
        self.assertEqual((counts['pasted_records'], counts['file_records']), (2, 2))

    def test_jsonl_and_gzip_accept_both_text_fields_and_preserve_multiline_records(self):
        content = '\ufeff' + json.dumps({'idx': 'original-7', 'example': 'First.\nSecond.'}) + '\n\n'
        content += json.dumps({'text': 'A final record.'}) + '\n'
        for extension in ('jsonl', 'jsonl.gz'):
            with self.subTest(extension=extension):
                source = self.root / f'data/source.{extension}'
                opener = gzip.open if extension.endswith('gz') else open
                with opener(source, 'wt', encoding='utf-8') as stream:
                    stream.write(content)
                output = self.root / f'{extension}.snapshot'
                counts = prepare_source({'source_path': str(source)}, output, max_length=1000)
                rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
                self.assertEqual(counts['records'], 2)
                self.assertEqual(rows[0]['example'], 'First.\nSecond.')
                self.assertEqual(rows[0]['origin']['idx'], 'original-7')
                self.assertEqual(rows[1]['origin']['line'], 3)

    def test_bad_rows_fail_with_source_line_instead_of_silently_skipping_them(self):
        for index, bad in enumerate(('not-json', '[]', '{}', '{"text": 7}', '{"example": "  "}')):
            with self.subTest(bad=bad):
                source = self.root / 'data/source.jsonl'
                source.write_text('{"text": "Valid."}\n' + bad, encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'line 2'):
                    prepare_source({'source_path': str(source)}, self.root / f'bad-{index}', max_length=1000)

    def test_empty_and_oversize_records_fail_without_truncation(self):
        with self.assertRaisesRegex(ValueError, 'no non-empty records'):
            prepare_source({'text': '  \n'}, self.root / 'empty', max_length=3)
        with self.assertRaisesRegex(ValueError, 'line 1.*model limit'):
            prepare_source({'text': 'Long record.'}, self.root / 'too-long', max_length=3)


class TemplateMiningExecutionTests(MiningFixtures, TestCase):
    def setUp(self):
        super().setUp()
        try:
            import spacy
            from spacy.tokens import Doc
        except ImportError:
            self.skipTest('Optional spaCy mining dependency is not installed')
        doc = Doc(spacy.blank('en').vocab,
                  words=['Anna', 'picked', 'a', 'beautiful', 'book', 'up', '.'],
                  heads=[1, 1, 4, 4, 1, 1, 1],
                  deps=['nsubj', 'ROOT', 'det', 'amod', 'dobj', 'prt', 'punct'],
                  pos=['PROPN', 'VERB', 'DET', 'ADJ', 'NOUN', 'ADP', 'PUNCT'],
                  tags=['NNP', 'VBD', 'DT', 'JJ', 'NN', 'RP', '.'],
                  lemmas=['anna', 'pick', 'a', 'beautiful', 'book', 'up', '.'])

        class Pipeline:
            meta = {'name': 'controlled-trees'}
            max_length = 1000000

            def pipe(self, inputs, **kwargs):
                for text, provenance in inputs:
                    if text != doc.text:
                        raise AssertionError('Unexpected source text')
                    yield doc, provenance

        self.text = doc.text
        mocked = patch('app_main.template_mining.load_mining_model', return_value=Pipeline())
        mocked.start()
        self.addCleanup(mocked.stop)
        lexicon = self.root / 'data/wiktionary'
        lexicon.mkdir()
        (lexicon / 'en_verb_inflections.txt').write_text(
            'pick\tpicks\tpicking\tpicked\tpicked\n', encoding='utf-8')
        (lexicon / 'en_postags_withverb.txt').write_text('pick\tnoun verb\n', encoding='utf-8')
        self.job = Job.objects.create(operation='template_mining')
        self.out = self.root / 'artifacts/ui/jobs' / str(self.job.pk)
        self.out.mkdir(parents=True)
        self.stages = []

    def run_mining(self, **config):
        execute_template_mining(self.job, self.out, {'text': self.text, **config},
                                lambda phase, *args, **kwargs: self.stages.append(phase))

    def test_complete_pipeline_publishes_all_resources_in_one_shared_pool(self):
        self.run_mining()
        summary = json.loads((self.out / 'mining_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['validation']['status'], 'passed')
        self.assertTrue(summary['published_to_pool'])
        bank = TemplateBank(summary['pool_path'])
        self.assertEqual((bank.path / 'sources' / f'{self.job.pk}.jsonl').read_bytes(),
                         (self.out / 'source.jsonl').read_bytes())
        pool = BuiltInPool(self.root, entities=EntityPool([]))
        self.assertEqual(pool.summary()['source_banks'], 1)
        self.assertEqual(pool.summary()['records'], 1)
        self.assertEqual([row['lemma'] for row in pool.verbs()], ['pick'])
        self.assertEqual([row['lemma'] for row in pool.adjectives()], ['beautiful'])
        self.assertIn('pick up', [row['lemma'] for row in pool.constructions()])
        documents = [doc for split in ('train', 'dev', 'test') for doc in pool.mined_documents(split=split)]
        self.assertEqual([doc.text for doc in documents], [self.text])
        self.assertEqual(documents[0].metadata['review_status'], 'UNREVIEWED')
        self.assertIn('Compiling templates, verbs and adjectives', self.stages)
        self.assertEqual(self.stages[-1], 'Built-in pool updated')

        from semantic_corpus.dialogue_generator.generator import DialogueGenerator
        from semantic_corpus.dialogue_generator.mined_bank import DialogueMinedBank

        dialogue_bank = DialogueMinedBank(self.root, cache=False)
        self.assertEqual(dialogue_bank.lemmas, ('pick',))
        for user_only in (False, True):
            with self.subTest(user_assistant_only=user_only):
                generator = DialogueGenerator(user_assistant_only=user_only, mined_bank=dialogue_bank)
                episodes = list(generator.generate('mined_predicates', 3, split='train'))
                self.assertEqual(len(episodes), 3)
                for episode in episodes:
                    episode.validate()
                    self.assertEqual(episode.mined_source['source_text'], self.text)
                    self.assertEqual(Path(episode.mined_source['bank']), bank.path)

    def test_legacy_opt_out_cannot_bypass_adding_resources_to_pool(self):
        self.run_mining(publish_to_pool=False)
        summary = json.loads((self.out / 'mining_summary.json').read_text(encoding='utf-8'))
        self.assertTrue(summary['published_to_pool'])
        self.assertEqual(summary['validation']['status'], 'passed')
        pool = BuiltInPool(self.root, entities=EntityPool([]))
        documents = [doc for split in ('train', 'dev', 'test') for doc in pool.mined_documents(split=split)]
        self.assertEqual([doc.text for doc in documents], [self.text])
        self.assertEqual(self.stages[-1], 'Built-in pool updated')

    def test_two_jobs_share_files_and_dialogue_lookup_preserves_each_source(self):
        from semantic_corpus.dialogue_generator.mined_bank import DialogueMinedBank

        first_id = str(self.job.pk)
        self.run_mining()
        self.job = Job.objects.create(operation='template_mining')
        self.out = self.root / 'artifacts/ui/jobs' / str(self.job.pk)
        self.out.mkdir(parents=True)
        self.run_mining()
        pool = BuiltInPool(self.root, entities=EntityPool([]))
        self.assertEqual(pool.summary()['source_banks'], 2)
        self.assertEqual({bank.path for bank in pool.banks}, {self.root / 'data/mined-templates'})
        bank = DialogueMinedBank(self.root, cache=False)
        references = bank.entries['pick']['references']
        self.assertEqual({ref[0] for ref in references}, {0, 1})
        self.assertEqual({source.source_id for source in bank.banks}, {first_id, str(self.job.pk)})
        bank.preload(references)
        for reference in references:
            template, _ = bank.template(reference)
            self.assertEqual(template.render()['text'], self.text)
        bank._selected.clear()
        for reference in references:
            template, _ = bank.template(reference)
            self.assertEqual(template.render()['text'], self.text)

    def test_cancel_request_prevents_publication(self):
        Job.objects.filter(pk=self.job.pk).update(cancel_requested=True)
        with self.assertRaisesRegex(ValueError, 'Cancellation requested'):
            self.run_mining()
        self.assertFalse((self.root / 'data/mined-templates').exists())

    def test_validation_failure_never_becomes_an_available_pool_component(self):
        with patch('semantic_corpus.template_registry.validate_bank', side_effect=ValueError('invalid spans')):
            with self.assertRaisesRegex(ValueError, 'invalid spans'):
                self.run_mining()
        self.assertFalse((self.root / 'data/mined-templates/manifest.json').exists())
        self.assertFalse((self.out / 'mining_summary.json').exists())
