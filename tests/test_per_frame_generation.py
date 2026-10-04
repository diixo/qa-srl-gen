"""Per-frame quotas, stable records and the synthetic CLI contract."""

from collections import Counter, defaultdict

import pytest

from semantic_corpus.cli import main
from semantic_corpus.question_generator import QuestionGenerator
from semantic_corpus.semantic_generator import DEFAULT_FRAMES, DEFAULT_PARADIGMS, Generator, frame_by_lemma
from semantic_corpus.semantic_generator.canonical import iter_canonical
from semantic_corpus.semantic_generator.substitutions import EntityPool
from semantic_corpus.storage import CorpusStore


def grouped(generator, count):
    result = defaultdict(list)
    for realized in generator.generate_per_frame(count):
        assert not realized.check()
        result[realized.situation.frame.lemma].append(realized.text)
    return dict(result)


@pytest.mark.parametrize('split,count', [('train', 100), ('dev', 10), ('test', 10)])
def test_exact_distinct_quota_for_every_frame(split, count):
    result = grouped(Generator(seed=42, split=split), count)
    assert set(result) == {f.lemma for f in DEFAULT_FRAMES}
    assert all(len(texts) == len(set(texts)) == count for texts in result.values())


def test_frames_can_be_added_reordered_or_extended_without_changing_existing_examples():
    base = grouped(Generator(seed=19, frames=DEFAULT_FRAMES[:-1]), 12)
    full = grouped(Generator(seed=19), 12)
    assert {key: full[key] for key in base} == base
    assert grouped(Generator(seed=19, frames=DEFAULT_FRAMES[::-1]), 12) == full
    larger = grouped(Generator(seed=19), 18)
    assert {key: texts[:12] for key, texts in larger.items()} == full
    assert grouped(Generator(seed=20), 12) != full


def test_calls_restart_per_frame_streams_without_consuming_legacy_rng():
    generator = Generator(seed=11)
    first = grouped(generator, 10)
    assert grouped(generator, 10) == first
    assert [r.text for r in generator.generate(10)] == [r.text for r in Generator(seed=11).generate(10)]


def test_document_ids_annotations_and_qa_are_stable_across_frame_additions():
    def records(frames, count):
        generator = Generator(seed=13, frames=frames, split='train')
        questions = QuestionGenerator(seed=13, paradigms=DEFAULT_PARADIGMS,
                                      ambiguous_forms=generator.pool.ambiguous_forms())
        pairs = iter_canonical(generator.generate_per_frame(count), prefix='stable',
                               split='train', random_seed=13, per_frame_ids=True)
        return {doc.document_id: (doc, run, questions.for_document(doc, run)) for doc, run in pairs}

    old = records(DEFAULT_FRAMES[:-1], 3)
    new = records(DEFAULT_FRAMES[::-1], 5)
    assert len(old) == (len(DEFAULT_FRAMES) - 1) * 3
    assert 'stable-give-000001' in old
    assert {key: new[key] for key in old} == old


def test_custom_pool_split_and_optional_slot_settings_are_honoured():
    source = Generator(seed=3, split='test')
    custom_pool = EntityPool(source.pool)
    generator = Generator(seed=3, pool=custom_pool, split='test',
                          frames=(frame_by_lemma('buy'), frame_by_lemma('visit')),
                          optional_slot_probability=0)
    for realized in generator.generate_per_frame(5):
        situation = realized.situation
        assert set(situation.bindings) == {s.name for s in situation.frame.required_slots}
        assert all(entity in tuple(custom_pool) for entity in situation.bindings.values())


def test_impossible_pool_fails_with_frame_and_quota_instead_of_silently_shortening_output():
    with pytest.raises(RuntimeError, match="frame 'give': generated 0/2"):
        list(Generator(seed=0, pool=EntityPool()).generate_per_frame(2))


def test_repeated_texts_are_resampled_and_exhaustion_is_bounded(monkeypatch):
    from semantic_corpus.semantic_generator import generator as module

    single = next(Generator(seed=0).generate_per_frame(1))
    monkeypatch.setattr(module, 'realize', lambda *args: single)
    stream = Generator(seed=0, frames=DEFAULT_FRAMES[:1]).generate_per_frame(2)
    assert next(stream) == single
    with pytest.raises(RuntimeError, match="generated 1/2"):
        next(stream)


def test_empty_negative_and_duplicate_frame_requests():
    assert list(Generator(seed=0).generate_per_frame(0)) == []
    with pytest.raises(ValueError, match='non-negative'):
        list(Generator(seed=0).generate_per_frame(-1))
    with pytest.raises(ValueError, match='no frames'):
        list(Generator(seed=0, frames=()).generate_per_frame(1))
    with pytest.raises(ValueError, match='unique frame lemmas'):
        list(Generator(seed=0, frames=(DEFAULT_FRAMES[0],) * 2).generate_per_frame(1))


@pytest.mark.parametrize('command', ['generate', 'questions', 'build'])
def test_cli_refuses_nonpositive_quota(command):
    with pytest.raises(SystemExit) as error:
        main([command, '--count', '0'])
    assert error.value.code == 2


def test_cli_generate_and_questions_use_per_frame_counts(capsys):
    assert main(['generate', '--count', '2']) == 0
    assert len(capsys.readouterr().out.splitlines()) == 2 * len(DEFAULT_FRAMES)
    assert main(['questions', '--count', '2']) == 0
    assert f'from {2 * len(DEFAULT_FRAMES)} situations (2 per frame)' in capsys.readouterr().out


def test_cli_build_persists_equal_quotas_and_count_units(tmp_path):
    root = tmp_path / 'store'
    args = ['build', '--store', str(root), '--out', str(tmp_path / 'out'), '--count', '3']
    assert main(args) == 0
    store = CorpusStore(root, create=False)
    assert Counter(d.metadata['frame'] for d in store.documents()) == {f.lemma: 3 for f in DEFAULT_FRAMES}
    assert store.versions()[0]['build_config']['count_scope'] == 'per_frame'


def test_cli_preflight_does_not_write_partial_corpus_when_quota_cannot_be_met(tmp_path, monkeypatch):
    original = Generator.generate_per_frame

    def fail_after_one(self, count):
        yield next(original(self, count))
        raise RuntimeError('quota unavailable')

    monkeypatch.setattr(Generator, 'generate_per_frame', fail_after_one)
    root = tmp_path / 'store'
    with pytest.raises(RuntimeError, match='quota unavailable'):
        main(['build', '--store', str(root), '--out', str(tmp_path / 'out'), '--count', '2'])
    store = CorpusStore(root, create=False)
    assert store.count('documents') == store.count('annotation_runs') == store.count('qa_examples') == 0
    assert not store.versions()
