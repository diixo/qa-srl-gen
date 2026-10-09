"""Atomic length filtering, preserved pairs/quotas and explicit coverage failures."""

from dataclasses import replace
import json

import pytest

from semantic_corpus.dialogue_generator import export, token_limits
from semantic_corpus.dialogue_generator.generator import DialogueGenerator


class Bank:
    identity = 'test-bank'
    lemmas = ('alpha', 'beta')

    def coverage(self, selected):
        return {'selected_lemmas': sorted(selected), 'unselected_lemmas': sorted(set(self.lemmas) - set(selected))}

    def inventory_report(self):
        return {}


def fixture_generator(monkeypatch, *, mined=False):
    originals = DialogueGenerator(seed=21, user_assistant_only=True)
    episodes = {split: list(originals.generate('recall', 10, split=split))
                for split in ('train', 'validation', 'test')}
    if mined:
        # Alpha/yes is retried after beta is already complete. Repeated beta
        # outcomes must not consume alpha's remaining full-coverage quota.
        outcomes = [('alpha', 'yes'), ('alpha', 'no'), ('alpha', 'unknown'),
                    ('beta', 'yes'), ('beta', 'no'), ('beta', 'unknown'),
                    ('beta', 'yes'), ('alpha', 'yes')]
        episodes = {split: [replace(e, task='mined_predicates', condition='want:' + outcome,
                                    mined_source={'raw_lemma': head, 'realization': 'compositional', 'template_id': e.id})
                            for e, (head, outcome) in zip(rows, outcomes)] for split, rows in episodes.items()}

    class Generator:
        tasks = ('mined_predicates',) if mined else ('recall',)
        mined_bank = Bank()

        def __init__(self, **kwargs):
            pass

        def generate(self, task, count, *, split):
            yield from episodes[split][:count]

    monkeypatch.setattr(export, 'DialogueGenerator', Generator)
    return episodes


def counter(monkeypatch, rejected_prompts=()):
    class Counter:
        provenance = {'model': 'deterministic-test-counter'}

        def length(self, messages):
            return 769 if messages[1]['content'] in rejected_prompts else 768

    monkeypatch.setattr(token_limits, 'GPT2TokenCounter', Counter)


@pytest.mark.parametrize('view', ['memory', 'history', 'extraction'])
def test_overlong_example_rejects_whole_episode_and_resamples_paired_views(tmp_path, monkeypatch, view):
    candidates = fixture_generator(monkeypatch)
    rejected = candidates['train'][0]
    row = (export.extraction_row(rejected, rejected.transitions[-1]) if view == 'extraction'
           else export.response_row(rejected, view))
    counter(monkeypatch, {row['messages'][1]['content']})
    out = tmp_path / 'data'
    manifest = export.build_dialogue_dataset(out, train=1, validation=1, test=1, mined_bank=Bank(), max_length=768)
    assert manifest['length_filter']['rejected_episodes'] == 1
    for view in ('memory', 'history', 'extraction'):
        rows = json.loads((out / view / 'train.json').read_text())
        assert rows and all(r['scenario_key'] != rejected.scenario_key for r in rows)
    audit = [json.loads(line) for line in (out / 'train.episodes.jsonl').read_text().splitlines()]
    assert [e['id'] for e in audit] == [candidates['train'][1].id]
    report = json.loads((out / 'token-lengths.json').read_text())
    assert report['splits']['train']['max_exported_tokens'] == dict.fromkeys(('history', 'memory', 'extraction'), 768)
    records = [json.loads(line) for line in (out / 'length-rejections.jsonl').read_text().splitlines()]
    assert len(records) == 1 and records[0]['episode_id'] == rejected.id
    assert manifest['splits']['train']['episodes'] == 1


def test_full_mined_coverage_retries_missing_outcomes_instead_of_repeating_covered_heads(tmp_path, monkeypatch):
    candidates = fixture_generator(monkeypatch, mined=True)
    e = candidates['train'][0]
    counter(monkeypatch, {export.extraction_row(e, e.transitions[-1])['messages'][1]['content']})
    out = tmp_path / 'full'
    manifest = export.build_dialogue_dataset(out, train=1, validation=1, test=1,
                                            mined_train=0, mined_bank=Bank(), max_length=768)
    rows = [json.loads(line) for line in (out / 'train.episodes.jsonl').read_text().splitlines()]
    assert len(rows) == 6
    assert {(r['mined_source']['raw_lemma'], r['condition']) for r in rows} == {
        (head, 'want:' + outcome) for head in Bank.lemmas for outcome in ('yes', 'no', 'unknown')}
    assert manifest['mined_bank']['selected_lemmas']['train'] == 2


def test_unfillable_limit_fails_without_partial_dataset(tmp_path, monkeypatch):
    episodes = fixture_generator(monkeypatch)
    counter(monkeypatch, {export.response_row(e, 'history')['messages'][1]['content'] for e in episodes['train']})
    with pytest.raises(ValueError, match='10 candidate episodes exceeded max_length=768'):
        export.build_dialogue_dataset(tmp_path / 'data', train=1, validation=1, test=1,
                                      mined_bank=Bank(), max_length=768)
    assert list(tmp_path.iterdir()) == []


def test_disabled_limit_does_not_load_tokenizer(tmp_path, monkeypatch):
    fixture_generator(monkeypatch)
    def refuse():
        raise AssertionError('Disabled limits must not load optional packages or a tokenizer')
    monkeypatch.setattr(token_limits, 'GPT2TokenCounter', refuse)
    manifest = export.build_dialogue_dataset(tmp_path / 'data', train=1, validation=1, test=1, mined_bank=Bank())
    assert 'length_filter' not in manifest
    assert not (tmp_path / 'data/token-lengths.json').exists()


@pytest.mark.parametrize('value', [0, -1, 1025, True, 768.0, '768'])
def test_invalid_gpt2_limits_fail_before_generation(tmp_path, value):
    with pytest.raises(ValueError, match='max_length'):
        export.build_dialogue_dataset(tmp_path / 'data', max_length=value)
    assert not (tmp_path / 'data').exists()


def test_uncached_tokenizer_explains_offline_setup(monkeypatch):
    hub = pytest.importorskip('huggingface_hub')
    pytest.importorskip('tokenizers')
    monkeypatch.setattr(hub, 'try_to_load_from_cache', lambda *a, **kw: None)
    with pytest.raises(ValueError, match='GPT-2 tokenizer.json is not cached'):
        token_limits.GPT2TokenCounter()
