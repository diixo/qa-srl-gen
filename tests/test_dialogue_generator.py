"""Scenario semantics, provenance, split integrity and chat-export contracts."""

from collections import Counter
from dataclasses import replace
import hashlib
import json

import pytest

from semantic_corpus.cli import main
from semantic_corpus.dialogue_generator import (
    TASKS, USER_ASSISTANT_TASKS, DialogueGenerator, build_dialogue_dataset, dialogue_tasks,
)
from semantic_corpus.dialogue_generator.export import extraction_row, prompt_key, response_row
from semantic_corpus.dialogue_generator.extraction_text import decode_update, parse_prompt
from semantic_corpus.dialogue_generator.models import (
    Delta, Evidence, Fact, Memory, Query, Replacement, Turn, answer_query, apply_delta,
)
from semantic_corpus.semantic_generator.substitutions import EntityPool


@pytest.mark.parametrize("task,user_only", [(task, False) for task in TASKS]
                         + [(task, True) for task in USER_ASSISTANT_TASKS])
def test_scenarios_replay_and_only_quote_visible_turns(task, user_only):
    for episode in DialogueGenerator(seed=19, user_assistant_only=user_only).generate(task, 20, split="train"):
        episode.validate()
        visible = {}
        for turn in episode.turns:
            visible[turn.id] = turn.text
            if turn.role == "user":
                transition = next(t for t in episode.transitions if t.turn == turn)
                assert apply_delta(transition.before, transition.delta,
                                   tuple(t for t in episode.turns if t.id in visible)) == transition.after
                for fact in (*transition.delta.add, *(r.fact for r in transition.delta.replace)):
                    assert all(e.quote in visible[e.turn_id] for e in fact.evidence)


def test_corrections_retire_old_day_and_keep_people_separate():
    generator = DialogueGenerator(seed=8)
    for e in generator.generate("correction", 25, split="test"):
        old, = e.memory.retired
        current = e.memory.find("user", "meeting_day")
        assert old.value != current.value
        assert current.value in e.answer and old.value not in e.answer
    for e in generator.generate("different_people", 25, split="test"):
        feeling = e.memory.find(e.query.subject, "feeling")
        assert feeling.value in e.answer
        others = [f.value for f in e.memory.facts if f.relation == "feeling" and f != feeling]
        assert all(value not in e.answer for value in others)


@pytest.mark.parametrize("task", ["desire_vs_done", "suggestion_vs_done"])
def test_wishes_and_advice_do_not_imply_completion(task):
    outcomes = Counter()
    for e in DialogueGenerator(seed=42).generate(task, 90, split="train"):
        completed = e.memory.find("user", e.query.relation)
        if completed is None or completed.value != e.query.value:
            assert e.expected_action == "unknown"
            assert not e.supporting_ids
            outcomes["unknown"] += 1
        elif completed.polarity == "negative":
            assert e.answer.startswith("No,")
            outcomes["no"] += 1
        else:
            assert e.answer.startswith("Yes,")
            outcomes["yes"] += 1
    assert set(outcomes) == {"unknown", "no", "yes"}
    assert min(outcomes.values()) > 10


def test_short_references_use_both_user_selection_and_assistant_options():
    choices = set()
    for e in DialogueGenerator(seed=3).generate("option_reference", 50, split="train"):
        selection = e.memory.find("user", "selection")
        offer = e.memory.assistant_actions[-1]
        assert selection.value in offer.items and selection.value in e.answer
        assert len(selection.evidence) == 2
        evidence_text = " ".join(ev.quote for ev in selection.evidence if ev.turn_id.startswith("u"))
        position = 0 if "first" in evidence_text else 1
        assert selection.value == offer.items[position]
        choices.add(position)
    assert choices == {0, 1}
    for e in DialogueGenerator().generate("ambiguous_reference", 10, split="validation"):
        assert e.expected_action == "clarify"
        assert e.transitions[-1].delta == Delta()
        assert e.memory.find("user", "selection") is None
        assert all(option in e.answer for option in e.memory.assistant_actions[-1].items)


def test_retraction_unknown_topic_return_and_no_tools():
    generator = DialogueGenerator()
    for task in ("retraction", "unknown", "topic_return", "capability"):
        e = next(generator.generate(task, 1, split="train"))
        if task in ("retraction", "unknown"):
            assert e.memory.find("user", "meeting_day") is None
            assert e.expected_action == "unknown"
            assert bool(e.memory.retired) == (task == "retraction")
        elif task == "topic_return":
            assert e.memory.find("user", "drink").value in e.answer
            assert len(e.turns) >= 9
        else:
            assert e.expected_action == "explain_limitation"
            assert "can't" in e.answer
            assert not e.memory.assistant_actions
        assert e.transitions[-1].delta == Delta()


def test_update_validation_is_atomic_and_preserves_history():
    u1 = Turn("u1", "user", "I am in Paris.")
    u2 = Turn("u2", "user", "I am now in London.")
    f1 = Fact("f1", "user", "location", "Paris", (Evidence("u1", u1.text),))
    f2 = Fact("f2", "user", "location", "London", (Evidence("u2", u2.text),))
    before = apply_delta(Memory(), Delta(add=(f1,)), (u1,))
    after = apply_delta(before, Delta(replace=(Replacement("f1", f2),)), (u1, u2))
    assert before.facts == (f1,) and not before.retired
    assert after.facts == (f2,) and after.retired == (f1,)
    invalid = (
        Delta(add=(f2,)),  # Conflicting active slot.
        Delta(invalidate=("missing",)),
        Delta(invalidate=("f1", "f1")),
        Delta(add=(replace(f2, id="f1"),)),
        Delta(replace=(Replacement("f1", replace(f2, subject="another person")),)),
        Delta(replace=(Replacement("f1", replace(f2, modality="desired")),)),
        Delta(replace=(Replacement("f1", replace(f2, evidence=(Evidence("future", "London"),))),)),
        Delta(replace=(Replacement("f1", replace(f2, evidence=(Evidence("u2", "Berlin"),))),)),
        Delta(replace=(Replacement("f1", replace(f2, evidence=(Evidence("u1", u1.text),))),)),
    )
    for delta in invalid:
        with pytest.raises(ValueError):
            apply_delta(before, delta, (u1, u2))
        assert before.facts == (f1,) and before.retired == ()
    with pytest.raises(ValueError, match="fresh IDs"):
        apply_delta(after, Delta(add=(replace(f2, id="f1", relation="drink"),)), (u1, u2))


def test_question_or_missing_claim_cannot_become_negative_knowledge():
    question = Query("yes_no", relation="visited", value="London")
    assert answer_query(Memory(), question)[1] == "unknown"
    wish = Fact("f1", "user", "visited", "London", (), modality="desired")
    assert answer_query(Memory((wish,)), question)[1] == "unknown"
    elsewhere = replace(wish, value="Paris", modality="asserted")
    assert answer_query(Memory((elsewhere,)), question)[1] == "unknown"


def test_determinism_prefix_stability_and_disjoint_names():
    g = DialogueGenerator(seed=9)
    all_names = {}
    for split in ("train", "validation", "test"):
        episodes = list(g.generate("different_people", 15, split=split))
        assert episodes == list(DialogueGenerator(seed=9).generate("different_people", 15, split=split))
        assert episodes[:5] == list(g.generate("different_people", 5, split=split))
        assert episodes != list(DialogueGenerator(seed=10).generate("different_people", 15, split=split))
        all_names[split] = {f.subject for e in episodes for f in e.memory.facts if f.subject != "user"}
    assert not all_names["train"] & (all_names["validation"] | all_names["test"])
    assert not all_names["validation"] & all_names["test"]
    with pytest.raises(ValueError, match="person names"):
        list(DialogueGenerator(pool=EntityPool()).generate("recall", 1, split="train"))


def test_exports_keep_targets_out_of_prompts_and_have_equal_loss_scope():
    e = next(DialogueGenerator().generate("option_reference", 1, split="test"))
    for view in ("history", "memory"):
        row = response_row(e, view)
        assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
        assert row["messages"][-1]["content"] == e.answer
        assert "expected_action" not in row["messages"][1]["content"]
        assert "supporting_ids" not in row["messages"][1]["content"]
    transition = next(t for t in e.transitions if any(f.relation == "selection" for f in t.delta.add))
    row = extraction_row(e, transition)
    memory, user, previous = parse_prompt(row["messages"][1]["content"])
    assert not any(f.relation == "selection" for f in memory.facts)
    assert previous.text == transition.previous_assistant.text
    assert user.text == transition.turn.text
    assert decode_update(row["messages"][-1]["content"], transition.before,
                         transition.turn, transition.previous_assistant) == transition.delta


@pytest.mark.parametrize("user_only", [False, True])
def test_streamed_build_quotas_checksums_prompt_dedup_and_trainer_contract(tmp_path, user_only):
    contract = tmp_path / "trainer.json"
    contract.write_text(json.dumps({"settings": {"model": "local-test-model", "max_tokens": 1024},
                                    "chat_template_sha256": "a" * 64}), encoding="utf-8")
    dest = tmp_path / "dataset"
    settings = dict(train=3, validation=2, test=2, training_report=contract, user_assistant_only=user_only)
    manifest = build_dialogue_dataset(dest, **settings)
    assert manifest == json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    for file, checksum in manifest["files"].items():
        assert hashlib.sha256((dest / file).read_bytes()).hexdigest() == checksum
    for view in ("memory", "history", "extraction"):
        previous_keys = set()
        for split, quota in (("train", 3), ("validation", 2), ("test", 2)):
            rows = json.loads((dest / view / f"{split}.json").read_text(encoding="utf-8"))
            if view != "extraction":
                assert Counter(row["task"] for row in rows) == dict.fromkeys(dialogue_tasks(user_only), quota)
            keys = {prompt_key(row) for row in rows}
            assert len(keys) == len(rows)
            assert not keys & previous_keys
            previous_keys.update(keys)
        report = json.loads((dest / view / "report.json").read_text(encoding="utf-8"))
        assert report["settings"] == {"model": "local-test-model", "max_tokens": 1024, "seed": 42}
        assert report["token_validation"] == "deferred_to_training_loader"
    again = tmp_path / "repeat"
    assert manifest == build_dialogue_dataset(again, **settings)
    with pytest.raises(FileExistsError):
        build_dialogue_dataset(dest)
    assert manifest == json.loads((dest / "manifest.json").read_text(encoding="utf-8"))


def test_failed_build_does_not_publish_or_overwrite_files(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("simulated failure")
    monkeypatch.setattr(DialogueGenerator, "generate", broken)
    with pytest.raises(ValueError, match="simulated failure"):
        build_dialogue_dataset(tmp_path / "failed", train=1, validation=1, test=1)
    assert list(tmp_path.iterdir()) == []


def test_progress_tracks_all_splits_and_callback_can_abort_publication(tmp_path):
    updates = []
    build_dialogue_dataset(tmp_path / 'complete', train=1, validation=1, test=1,
                           progress=lambda *args: updates.append(args))
    assert updates[0] == ('Indexing mined predicates', 0, 0)
    generation = [row for row in updates if row[2] > 0]
    assert generation[0][1:] == (0, 3 * len(TASKS))
    assert updates[-1] == ('Checking dataset files', 3 * len(TASKS), 3 * len(TASKS))
    assert [row[1] for row in generation] == sorted(row[1] for row in generation)

    def cancel(phase, processed, total):
        if phase == 'Checking dataset files':
            raise ValueError('Cancelled before publication')

    with pytest.raises(ValueError, match='Cancelled before publication'):
        build_dialogue_dataset(tmp_path / 'cancelled', train=1, validation=1, test=1, progress=cancel)
    assert not (tmp_path / 'cancelled').exists()
    assert not list(tmp_path.glob('.dialogue-build-*'))

    def cancel_index(phase, processed, total):
        if phase == 'Indexing mined predicates':
            raise ValueError('Cancelled during indexing')

    with pytest.raises(ValueError, match='Cancelled during indexing'):
        build_dialogue_dataset(tmp_path / 'cancelled-index', train=1, validation=1,
                               test=1, progress=cancel_index)
    assert not (tmp_path / 'cancelled-index').exists()


@pytest.mark.parametrize("user_only", [False, True])
def test_omitting_metadata_preserves_training_content_splits_and_audit(tmp_path, user_only):
    full = tmp_path / 'full'
    minimal = tmp_path / 'messages-only'
    settings = dict(train=1, validation=1, test=1, user_assistant_only=user_only)
    before = build_dialogue_dataset(full, **settings)
    after = build_dialogue_dataset(minimal, **settings, include_metadata=False)
    assert before['include_metadata'] is True and after['include_metadata'] is False
    assert before['splits'] == after['splits']
    for split in ('train', 'validation', 'test'):
        for view in ('history', 'memory', 'extraction'):
            original = json.loads((full / view / f'{split}.json').read_text(encoding='utf-8'))
            exported = json.loads((minimal / view / f'{split}.json').read_text(encoding='utf-8'))
            assert all(set(row) == {'messages'} for row in exported)
            assert exported == [{'messages': row['messages']} for row in original]
            assert len({prompt_key(row) for row in exported}) == len(exported)
        audit = f'{split}.episodes.jsonl'
        assert (full / audit).read_bytes() == (minimal / audit).read_bytes()
    for name, checksum in after['files'].items():
        assert hashlib.sha256((minimal / name).read_bytes()).hexdigest() == checksum


def test_cli_dialogue_build_and_refusal(tmp_path, capsys):
    args = ["dialogue-build", "--out", str(tmp_path / "data"), "--train", "1", "--validation", "1", "--test", "1"]
    assert main(args) == 0
    assert f"{len(TASKS)} paired response examples" in capsys.readouterr().out
    assert main(args) == 2
    assert "new output directory" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(args + ["--train", "0"])
    minimal_args = args.copy()
    minimal_args[2] = str(tmp_path / 'messages-only')
    assert main(minimal_args + ['--no-metadata']) == 0
    rows = json.loads((tmp_path / 'messages-only/memory/train.json').read_text(encoding='utf-8'))
    assert all(set(row) == {'messages'} for row in rows)
    minimal_args[2] = str(tmp_path / 'user-only')
    assert main(minimal_args + ['--no-metadata', '--user-assistant-only']) == 0
    manifest = json.loads((tmp_path / 'user-only/manifest.json').read_text(encoding='utf-8'))
    assert manifest['user_assistant_only'] is True
    assert manifest['tasks'] == list(USER_ASSISTANT_TASKS)


def test_user_only_scenarios_need_no_names_and_keep_context_splits_across_seeds(monkeypatch):
    from semantic_corpus.dialogue_generator import generator as module

    def refuse_pool():
        raise AssertionError('Simple conversations must not load the named entity pool')

    monkeypatch.setattr(module, 'default_pool', refuse_pool)
    context_splits = {}
    for seed in (9, 10):
        generator = DialogueGenerator(seed=seed, user_assistant_only=True)
        assert generator.tasks == USER_ASSISTANT_TASKS
        with pytest.raises(ValueError, match='Unknown task'):
            list(generator.generate('different_people', 1, split='train'))
        for split in ('train', 'validation', 'test'):
            for task in USER_ASSISTANT_TASKS:
                episodes = list(generator.generate(task, 12, split=split))
                assert episodes[:5] == list(generator.generate(task, 5, split=split))
                for episode in episodes:
                    assert episode.query.subject == 'user'
                    assert all(f.subject == 'user' for f in (*episode.memory.facts, *episode.memory.retired))
                    assert all(f.value in module.COMMON_PLACES for f in episode.memory.facts
                               if f.relation in ('location', 'visited'))
                    context = tuple((f.relation, f.value) for f in episode.transitions[0].after.facts)
                    assert context_splits.setdefault(context, split) == split
                    # No loss of the grouping context through corrections/retractions.
                    assert all(f in episode.memory.facts for f in episode.transitions[0].after.facts)


@pytest.mark.parametrize('user_only', [False, True])
def test_default_sizes_have_no_duplicate_or_cross_split_prompts(tmp_path, user_only):
    manifest = build_dialogue_dataset(tmp_path / 'dataset', user_assistant_only=user_only)
    assert 'entities.json' in manifest['resources']
    assert 'semantic_generator/frames.py' in manifest['resources']
    assert manifest['prompt_version'] == 'dialogue-prompts-v8'
    assert manifest['content_formats'] == {'history': 'history-text-v1', 'memory': 'memory-text-v6',
                                            'extraction': 'extraction-text-v5'}
    for view in ('history', 'memory', 'extraction'):
        seen = set()
        for split, count in (('train', 200), ('validation', 20), ('test', 20)):
            rows = json.loads((tmp_path / 'dataset' / view / f'{split}.json').read_text(encoding='utf-8'))
            if view != 'extraction':
                assert Counter(row['task'] for row in rows) == dict.fromkeys(dialogue_tasks(user_only), count)
            keys = {prompt_key(row) for row in rows}
            assert len(keys) == len(rows)
            assert not seen & keys
            seen.update(keys)


def test_user_only_exhaustion_refuses_to_pad_or_publish_partial_data(tmp_path):
    from semantic_corpus.dialogue_generator.generator import COMMON_PLACES, FEELINGS, DRINKS

    impossible = len(COMMON_PLACES) * len(FEELINGS) * len(DRINKS) + 1
    with pytest.raises(ValueError, match='reduce the per-task quota'):
        build_dialogue_dataset(tmp_path / 'too-many', train=impossible, validation=1, test=1,
                               user_assistant_only=True)
    assert list(tmp_path.iterdir()) == []
