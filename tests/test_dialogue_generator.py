"""Scenario semantics, provenance, split integrity and chat-export contracts."""

from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json

import pytest

from semantic_corpus.cli import main
from semantic_corpus.dialogue_generator import TASKS, DialogueGenerator, build_dialogue_dataset
from semantic_corpus.dialogue_generator.export import extraction_row, prompt_key, response_row
from semantic_corpus.dialogue_generator.models import (
    Delta, Evidence, Fact, Memory, Query, Replacement, Turn, answer_query, apply_delta,
)
from semantic_corpus.semantic_generator.substitutions import EntityPool


@pytest.mark.parametrize("task", TASKS)
def test_scenarios_replay_and_only_quote_visible_turns(task):
    for episode in DialogueGenerator(seed=19).generate(task, 20, split="train"):
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


@pytest.mark.parametrize("task,relation", [("desire_vs_done", "visited"),
                                         ("suggestion_vs_done", "completed_break")])
def test_wishes_and_advice_do_not_imply_completion(task, relation):
    outcomes = Counter()
    for e in DialogueGenerator(seed=42).generate(task, 90, split="train"):
        completed = e.memory.find("user", relation)
        if completed is None:
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
    prompt = json.loads(row["messages"][1]["content"])
    assert not any(f["relation"] == "selection" for f in prompt["state_before"]["facts"])
    assert prompt["last_assistant"]["text"] == transition.previous_assistant.text
    assert json.loads(row["messages"][-1]["content"]) == json.loads(json.dumps(asdict(transition.delta)))


def test_streamed_build_quotas_checksums_prompt_dedup_and_trainer_contract(tmp_path):
    contract = tmp_path / "trainer.json"
    contract.write_text(json.dumps({"settings": {"model": "local-test-model", "max_tokens": 1024},
                                    "chat_template_sha256": "a" * 64}), encoding="utf-8")
    dest = tmp_path / "dataset"
    manifest = build_dialogue_dataset(dest, train=3, validation=2, test=2, training_report=contract)
    assert manifest == json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    for file, checksum in manifest["files"].items():
        assert hashlib.sha256((dest / file).read_bytes()).hexdigest() == checksum
    for view in ("memory", "history", "extraction"):
        previous_keys = set()
        for split, quota in (("train", 3), ("validation", 2), ("test", 2)):
            rows = json.loads((dest / view / f"{split}.json").read_text(encoding="utf-8"))
            if view != "extraction":
                assert Counter(row["task"] for row in rows) == dict.fromkeys(TASKS, quota)
            keys = {prompt_key(row) for row in rows}
            assert len(keys) == len(rows)
            assert not keys & previous_keys
            previous_keys.update(keys)
        report = json.loads((dest / view / "report.json").read_text(encoding="utf-8"))
        assert report["settings"] == {"model": "local-test-model", "max_tokens": 1024, "seed": 42}
        assert report["token_validation"] == "deferred_to_training_loader"
    again = tmp_path / "repeat"
    assert manifest == build_dialogue_dataset(again, train=3, validation=2, test=2, training_report=contract)
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
    assert updates[0][1:] == (0, 3 * len(TASKS))
    assert updates[-1] == ('Checking dataset files', 3 * len(TASKS), 3 * len(TASKS))
    assert [row[1] for row in updates] == sorted(row[1] for row in updates)

    def cancel(phase, processed, total):
        if phase == 'Checking dataset files':
            raise ValueError('Cancelled before publication')

    with pytest.raises(ValueError, match='Cancelled before publication'):
        build_dialogue_dataset(tmp_path / 'cancelled', train=1, validation=1, test=1, progress=cancel)
    assert not (tmp_path / 'cancelled').exists()
    assert not list(tmp_path.glob('.dialogue-build-*'))


def test_omitting_metadata_preserves_training_content_splits_and_audit(tmp_path):
    full = tmp_path / 'full'
    minimal = tmp_path / 'messages-only'
    before = build_dialogue_dataset(full, train=1, validation=1, test=1)
    after = build_dialogue_dataset(minimal, train=1, validation=1, test=1, include_metadata=False)
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
    assert "12 paired response examples" in capsys.readouterr().out
    assert main(args) == 2
    assert "new output directory" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(args + ["--train", "0"])
    minimal_args = args.copy()
    minimal_args[2] = str(tmp_path / 'messages-only')
    assert main(minimal_args + ['--no-metadata']) == 0
    rows = json.loads((tmp_path / 'messages-only/memory/train.json').read_text(encoding='utf-8'))
    assert all(set(row) == {'messages'} for row in rows)
