"""Text updater semantics, provenance and malformed prediction boundaries."""

from dataclasses import replace

import pytest

from semantic_corpus.dialogue_generator import DialogueGenerator
from semantic_corpus.dialogue_generator.export import extraction_row
from semantic_corpus.dialogue_generator.extraction_text import (
    decode_update, parse_prompt, render_prompt, render_update,
)
from semantic_corpus.dialogue_generator.models import (
    Delta, Evidence, Fact, Memory, Transition, Turn, apply_delta,
)


@pytest.mark.parametrize('user_only', [False, True])
def test_all_tasks_roundtrip_updates_including_sources_and_retirements(user_only):
    generator = DialogueGenerator(seed=19, user_assistant_only=user_only)
    operations = set()
    for task in generator.tasks:
        for episode in generator.generate(task, 18, split='test'):
            for transition in episode.transitions:
                row = extraction_row(episode, transition)
                prompt, target = (message['content'] for message in row['messages'][1:])
                assert not target.lstrip().startswith(('{', '['))  # Source quotations may contain punctuation.
                assert all(key not in prompt + target for key in ('turn_id', 'next_fact_id', 'source_turn'))
                decoded = decode_update(target, transition.before, transition.turn, transition.previous_assistant)
                assert decoded == transition.delta
                turns = ((transition.previous_assistant,) if transition.previous_assistant else ()) + (transition.turn,)
                assert apply_delta(transition.before, decoded, turns) == transition.after
                visible, user, previous = parse_prompt(prompt)
                assert render_prompt(visible, user, previous) == prompt
                # Evaluation can replay using only information in the input.
                decode_update(target, visible, user, previous)
                operations.update(line.partition(' ')[0] for line in target.splitlines())
    assert operations == {'SET', 'SET_CONTEXT', 'REMOVE', 'NO_CHANGE', 'EVENT', 'EVENT_CONTEXT',
                          'REMOVE_EVENT', 'MINED', 'REMOVE_MINED', 'QUOTE',
                          'PARTIAL', 'PARTIAL_CONTEXT', 'REMOVE_PARTIAL'}


def test_commands_preserve_modality_and_do_not_turn_removal_into_negative_claim():
    user = Turn('u1', 'user', 'I want to buy a book. I have not bought it.')
    text = ('SET user | bought | desired | positive | a book\n'
            'SET user | bought | asserted | negative | a book')
    delta = decode_update(text, Memory(), user)
    memory = apply_delta(Memory(), delta, (user,))
    assert memory.find('user', 'bought', 'desired').polarity == 'positive'
    assert memory.find('user', 'bought').polarity == 'negative'
    later = Turn('u2', 'user', 'Forget what I said about having bought that book.')
    delta = decode_update('REMOVE user | bought | asserted', memory, later)
    after = apply_delta(memory, delta, (later,))
    assert after.find('user', 'bought') is None
    assert after.find('user', 'bought', 'desired') == memory.find('user', 'bought', 'desired')
    assert len(after.retired) == 1


def test_set_allocates_ids_and_replaces_only_its_semantic_key():
    user = Turn('u3', 'user', 'The meeting is on Friday. I prefer tea.')
    old = Fact('f5', 'user', 'meeting_day', 'Monday', ())
    other = Fact('f6', 'Alice', 'meeting_day', 'Tuesday', ())
    memory = Memory((old, other), retired=(replace(old, id='f9', value='Thursday'),))
    delta = decode_update('SET user | meeting_day | asserted | positive | Friday\n'
                          'SET user | drink | asserted | positive | tea', memory, user)
    assert delta.replace[0].old_id == 'f5'
    assert delta.replace[0].fact.id == 'f10'
    assert delta.add[0].id == 'f11'
    assert delta.replace[0].fact.evidence == (Evidence('u3', user.text),)
    after = apply_delta(memory, delta, (user,))
    assert after.find('Alice', 'meeting_day') == other
    assert render_update(Transition(user, memory, delta, after, None)).startswith('SET user | meeting_day')


def test_context_references_keep_both_sources_without_asking_model_for_ids():
    assistant = Turn('private-assistant', 'assistant', 'You could take a walk or a rest.')
    user = Turn('private-user', 'user', 'The second one.')
    text = 'SET_CONTEXT user | selection | asserted | positive | a rest'
    delta = decode_update(text, Memory(), user, assistant)
    assert delta.add[0].evidence == (Evidence(user.id, user.text), Evidence(assistant.id, assistant.text))
    prompt = render_prompt(Memory(), user, assistant)
    assert 'private-' not in prompt + text
    with pytest.raises(ValueError, match='previous assistant'):
        decode_update(text, Memory(), user)


@pytest.mark.parametrize('text', [
    '', '{}', 'NO_CHANGE\nSET user | drink | asserted | positive | tea',
    'SET user | drink | asserted | positive', 'SET user | drink | asserted | maybe | tea',
    'SET user | made_up | asserted | positive | tea', 'SET user | drink | planned | positive | tea',
    'SET user | bought | required | positive | a book', 'REMOVE user | drink',
    'REMOVE user | drink | asserted\nSET user | drink | asserted | positive | tea',
    'SET user | drink | asserted | positive | tea\nSET user | drink | asserted | positive | coffee',
    'SET user | drink | asserted | positive | tea\\', 'SET user | drink | asserted | positive | tea\\x',
    'SET user | drink | asserted | positive |', 'Here is the update:\nNO_CHANGE',
])
def test_invalid_output_is_refused_without_partial_updates(text):
    memory = Memory()
    with pytest.raises(ValueError):
        decode_update(text, memory, Turn('u1', 'user', 'Hello.'))
    assert memory == Memory()


def test_remove_refuses_missing_keys_and_no_change_keeps_everything():
    memory = Memory((Fact('f1', 'user', 'drink', 'tea', ()),))
    user = Turn('u2', 'user', 'Thanks.')
    with pytest.raises(ValueError, match='active fact'):
        decode_update('REMOVE Alice | drink | asserted', memory, user)
    assert decode_update('NO_CHANGE', memory, user) == Delta()


def test_delimiters_backslashes_newlines_and_literal_empty_marker_are_lossless():
    value = 'a book | a letter\\note\nsecond line'
    user = Turn('u1', 'user', 'I chose: ' + value + '\nLatest user message: literal text')
    assistant = Turn('a1', 'assistant', '(none)')
    fact = Fact('f1', 'user', 'selection', value, (Evidence(user.id, user.text),))
    delta = Delta(add=(fact,))
    after = apply_delta(Memory(), delta, (assistant, user))
    text = render_update(Transition(user, Memory(), delta, after, assistant))
    assert len(text.splitlines()) == 1
    assert decode_update(text, Memory(), user, assistant) == delta
    prompt = render_prompt(after, user, assistant)
    memory2, user2, assistant2 = parse_prompt(prompt)
    assert user2.text == user.text and assistant2.text == assistant.text
    assert memory2.facts[0].value == value
    assert render_prompt(memory2, user2, assistant2) == prompt


def test_unsupported_evidence_cannot_silently_disappear_during_export():
    user = Turn('u1', 'user', 'I am in Paris.')
    fact = Fact('f1', 'user', 'location', 'Paris', (Evidence('u1', 'Paris'),))
    transition = Transition(user, Memory(), Delta(add=(fact,)), Memory((fact,)), None)
    with pytest.raises(ValueError, match='evidence'):
        render_update(transition)


def test_prompt_omits_old_facts_and_ids_but_keeps_semantic_differences():
    user = Turn('u1', 'user', 'Where am I?')
    fact = Fact('f1', 'user', 'location', 'Paris', ())
    memory = Memory((fact,))
    original = render_prompt(memory, user)
    changed = replace(memory, facts=(replace(fact, id='f99'),), retired=(replace(fact, value='Berlin'),))
    assert render_prompt(changed, replace(user, id='u99')) == original
    assert 'Berlin' not in original
    for change in ({'subject': 'Alice'}, {'polarity': 'negative'}, {'value': 'London'}):
        assert render_prompt(Memory((replace(fact, **change),)), user) != original
    with pytest.raises(ValueError, match='Extraction prompt'):
        parse_prompt('{"facts": []}')


def test_extraction_metrics_measure_state_sources_and_false_changes_separately():
    from semantic_corpus.dialogue_generator.trainer.metrics import ExtractionMetrics

    metrics = ExtractionMetrics()
    user = Turn('u1', 'user', 'I am in Paris. I feel happy.')
    prompt = render_prompt(Memory(), user, Turn('a1', 'assistant', 'Where are you?'))
    first = 'SET user | location | asserted | positive | Paris'
    second = 'SET user | feeling | asserted | positive | happy'
    metrics.add(first + '\n' + second, second + '\n' + first, prompt=prompt)
    # Right state but wrong source mode must not count as a matching update.
    metrics.add(first.replace('SET ', 'SET_CONTEXT ', 1), first, prompt=prompt)
    metrics.add(first, '{}', prompt=prompt)
    metrics.add('NO_CHANGE', first, prompt=prompt)
    metrics.add(first, first.replace('positive', 'negative'), prompt=prompt)
    report = metrics.report()['extraction']
    assert report['count'] == 5
    assert report['valid_predictions'] == 4 and report['invalid_predictions'] == 1
    assert report['update_matches'] == 1 and report['state_matches'] == 2
    assert report['false_changes_on_no_change'] == 1
    assert report['by_operation']['no_change']['state_matches'] == 0
