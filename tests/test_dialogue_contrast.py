from collections import Counter, defaultdict
from dataclasses import replace
import json

import pytest

from semantic_corpus.dialogue_generator.actions import ACTIONS, CONTRAST_TASKS, action_statement
from semantic_corpus.dialogue_generator.challenge import build_challenge
from semantic_corpus.dialogue_generator.export import build_dialogue_dataset, response_row
from semantic_corpus.dialogue_generator.generator import DialogueGenerator
from semantic_corpus.dialogue_generator.memory_text import render_memory
from semantic_corpus.dialogue_generator.models import Fact, Memory, Query, answer_query
from semantic_corpus.dialogue_generator.trainer.data import check_disjoint, load_rows
from semantic_corpus.dialogue_generator.trainer.metrics import AnswerMetrics


@pytest.mark.parametrize('task', CONTRAST_TASKS)
def test_matched_groups_balance_outcomes_cover_actions_and_preserve_meaning(task):
    episodes = list(DialogueGenerator(user_assistant_only=True).generate(task, 360, split='train'))
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode.contrast_group].append(episode)
        episode.validate()
        assert episode.answer == answer_query(episode.memory, episode.query)[0]
        prompt = response_row(episode, 'memory')['messages'][1]['content']
        assert episode.contrast_group not in prompt and episode.condition not in prompt
    assert {e.query.relation for e in episodes} == {a.relation for a in ACTIONS}
    assert {e.condition.split(':')[0] for e in episodes} == {'plain', 'other_object', 'other_action', 'accepted'}
    for group in groups.values():
        assert len(group) == 3
        assert {e.condition.split(':')[1] for e in group} == {'yes', 'no', 'unknown'}
        assert len({e.turns[-1].text for e in group}) == 1
        assert len({e.memory_style for e in group}) == 1
        assert len({e.query.value for e in group}) == 1
    if task == 'desire_vs_done':
        assert {'planned', 'desired'} <= {f.modality for e in episodes for f in e.memory.facts}


def test_new_action_forms_preserve_modality_negation_and_objects():
    for action in ACTIONS:
        fact = Fact('f1', 'user', action.relation, action.objects[0], ())
        question = Query('yes_no', relation=action.relation, value=fact.value)
        for modality in ('asserted', 'desired', 'planned'):
            for negative in (False, True):
                changed = replace(fact, modality=modality, polarity='negative' if negative else 'positive')
                texts = {render_memory(Memory((changed,)), style=style) for style in range(5)}
                assert len(texts) >= 3
                assert all(fact.value in text for text in texts)
                if negative:
                    assert all('not ' in text for text in texts)
                expected = 'unknown' if modality != 'asserted' else 'answer'
                assert answer_query(Memory((changed,)), question)[1] == expected
        with pytest.raises(ValueError):
            action_statement(action, fact.value, modality='maybe')
    reading = next(action for action in ACTIONS if action.relation == 'read')
    assert action_statement(reading, 'the book', style=1) == 'You reported having read the book.'
    assert action_statement(reading, 'the book', negative=True, style=1) == 'You reported not having read the book.'


def test_export_multiplier_frozen_challenge_and_group_leakage(tmp_path):
    out = tmp_path / 'data'
    result = build_dialogue_dataset(out, train=6, validation=3, test=3,
                                    user_assistant_only=True, contrast_multiplier=3)
    assert result['splits']['train']['episodes'] == 90
    train, val, test = [load_rows(out / 'memory' / f'{split}.json') for split in ('train', 'validation', 'test')]
    for left, right in ((train, val), (train, test), (val, test)):
        check_disjoint(left, right)
    assert Counter(row.task for row in train)['desire_vs_done'] == 18
    challenge_path = tmp_path / 'challenge'
    report = build_challenge(challenge_path, groups_per_task=6)
    assert report['count'] == 36
    challenge = load_rows(challenge_path / 'test.json')
    check_disjoint(train, challenge)
    check_disjoint(val, challenge)
    assert all('Is there confirmation' in row.messages[1]['content'] or
               'Can you tell whether' in row.messages[1]['content'] for row in challenge)
    build_challenge(tmp_path / 'again', groups_per_task=6)
    assert (challenge_path / 'test.json').read_bytes() == (tmp_path / 'again/test.json').read_bytes()
    with pytest.raises(FileExistsError):
        build_challenge(challenge_path)
    first = next(row for row in train if row.contrast_group)
    with pytest.raises(ValueError, match='Contrast group leakage'):
        check_disjoint([first], [replace(first, prompt_key='different', scenario_key='different')])


def test_metrics_expose_false_completion_and_entire_group_failures():
    metrics = AnswerMetrics()
    for answer in ('Yes, you said you have.', "No, you said you haven't done that.", "I don't know."):
        prediction = 'Yes, you said you have.' if answer == "I don't know." else answer
        metrics.add(answer, prediction, task='desire_vs_done', group='g', condition='plain')
    report = metrics.report()
    assert report['unknown_false_yes'] == 1 and report['unknown_count'] == 1
    assert report['contrast_groups'] == {'complete': 1, 'all_correct': 0, 'partial_or_duplicate': 0, 'accuracy': 0}
    assert report['task_answer_confusion']['desire_vs_done']['confusion']['unknown'] == {'yes': 1}
