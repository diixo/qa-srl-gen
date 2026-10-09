"""Meaning and provenance boundaries of plain-text response memory."""

from dataclasses import replace

import pytest

from semantic_corpus.dialogue_generator import DialogueGenerator
from semantic_corpus.dialogue_generator.export import response_row
from semantic_corpus.dialogue_generator.memory_text import render_memory
from semantic_corpus.dialogue_generator.models import AssistantAction, Evidence, Fact, Memory


def test_text_separates_wishes_negative_reports_completed_actions_and_people():
    memory = Memory(facts=(
        Fact('f1', 'user', 'visited', 'the park', (), modality='desired'),
        Fact('f2', 'user', 'visited', 'the park', (), polarity='negative'),
        Fact('f3', 'user', 'completed_break', 'a short walk', ()),
        Fact('f4', 'user', 'feeling', 'tired', ()),
        Fact('f5', 'Alice', 'feeling', 'happy', ()),
        Fact('f6', 'Alice', 'location', 'Paris', ()),
        Fact('f7', 'Alice', 'drink', 'tea', ()),
        Fact('f8', 'user', 'selection', 'a music break', ()),
    ))
    text = render_memory(memory)
    assert text.splitlines() == [
        'Known facts:',
        '- You want to visit the park.',
        '- You have not visited the park.',
        '- You have taken a short walk.',
        '- You feel tired.',
        '- Alice feels happy.',
        '- Alice is in Paris.',
        '- Alice prefers tea.',
        '- You chose a music break.',
    ]
    assert 'You have visited the park.' not in text
    assert 'You feel happy.' not in text


@pytest.mark.parametrize('user_only', [False, True])
def test_corrections_and_retractions_do_not_reintroduce_retired_days(user_only):
    generator = DialogueGenerator(user_assistant_only=user_only)
    for episode in generator.generate('correction', 12, split='train'):
        prompt = response_row(episode, 'memory')['messages'][1]['content']
        assert episode.memory.retired[0].value not in prompt
        assert f"Your meeting is on {episode.memory.find('user', 'meeting_day').value}." in prompt
        assert response_row(episode, 'memory')['messages'][-1]['content'] == episode.answer
    for episode in generator.generate('retraction', 12, split='train'):
        text = render_memory(episode.memory)
        assert episode.memory.retired[0].value not in text
        assert 'Your meeting is' not in text
        assert episode.expected_action == 'unknown'


def test_option_order_and_suggestions_survive_without_claiming_completion():
    first = AssistantAction('options', ('a short walk', 'a music break'), 'a1')
    second = AssistantAction('suggestion', ('a snack break',), 'a2')
    memory = Memory(assistant_actions=(first, second))
    text = render_memory(memory)
    assert 'No facts recorded.' in text
    assert '  1. a short walk\n  2. a music break' in text
    assert 'The assistant suggested a snack break.' in text
    assert text.index('  2. a music break') < text.index('The assistant suggested')
    assert 'You have taken' not in text
    swapped = replace(memory, assistant_actions=(replace(first, items=tuple(reversed(first.items))), second))
    assert '  1. a music break\n  2. a short walk' in render_memory(swapped)


def test_ids_sources_and_retired_facts_are_not_part_of_the_model_context():
    fact = Fact('private-fact-id', 'user', 'visited', 'the park',
                (Evidence('private-turn-id', 'private evidence text'),))
    action = AssistantAction('options', ('a walk', 'a rest'), 'private-action-source')
    memory = Memory((fact,), assistant_actions=(action,))
    renamed = replace(memory,
                      facts=(replace(fact, id='different-id', evidence=()),),
                      retired=(Fact('old', 'user', 'location', 'private retired location', ()),),
                      assistant_actions=(replace(action, source_turn='different-source'),))
    assert render_memory(memory) == render_memory(renamed)
    assert 'private' not in render_memory(memory)
    # Distinctions that change the meaning must still change the context.
    for change in ({'polarity': 'negative'}, {'modality': 'desired'}, {'subject': 'Alice'}):
        assert render_memory(memory) != render_memory(replace(memory, facts=(replace(fact, **change),)))
    assert render_memory(Memory()) == 'Known facts:\n- No facts recorded.'


def test_unsupported_memory_fails_instead_of_silently_losing_information():
    fact = Fact('f1', 'user', 'visited', 'the park', ())
    for change in ({'relation': 'unsupported'}, {'polarity': 'maybe'}, {'modality': 'hypothetical'},
                   {'relation': 'feeling', 'modality': 'desired'}):
        with pytest.raises(ValueError, match='Unsupported'):
            render_memory(Memory((replace(fact, **change),)))
    with pytest.raises(ValueError, match='Unsupported assistant action'):
        render_memory(Memory(assistant_actions=(AssistantAction('unsupported', ('a walk',), 'a1'),)))


@pytest.mark.parametrize('user_only', [False, True])
def test_every_task_exports_plain_text_context_and_the_same_response_target(user_only):
    generator = DialogueGenerator(user_assistant_only=user_only)
    for task in generator.tasks:
        for episode in generator.generate(task, 8, split='test'):
            row = response_row(episode, 'memory')
            prompt = row['messages'][1]['content']
            assert prompt.startswith('Known facts:\n')
            assert prompt.endswith('Latest user message: ' + episode.turns[-1].text)
            assert all(token not in prompt for token in ('{', '}', 'turn_id', 'source_turn', '"id"', 'evidence'))
            assert [message['role'] for message in row['messages']] == ['system', 'user', 'assistant']
            assert row['messages'][-1]['content'] == response_row(episode, 'history')['messages'][-1]['content']
