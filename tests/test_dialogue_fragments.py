"""Partial meaning must survive without promoting questions or wishes to facts."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from semantic_corpus.dialogue_generator import DialogueGenerator
from semantic_corpus.dialogue_generator.events import (
    Event, EventArgument, completion_evidence, state_evidence, event_statement,
    event_question, event_description,
)
from semantic_corpus.dialogue_generator.export import extraction_row, response_row, prompt_key
from semantic_corpus.dialogue_generator.extraction_text import decode_update, parse_prompt
from semantic_corpus.dialogue_generator.fragments import fragment_episode, load_fragment_patterns
from semantic_corpus.dialogue_generator.models import Delta, Memory, Turn


PATTERNS = {p['id']: p for p in load_fragment_patterns()}


def episode(key, user_only=True):
    context = ('the kitchen', 'tired', 'tea') if user_only else ('Anna', 'Rome')
    return fragment_episode(PATTERNS[key], context, split='train', user_only=user_only)


@pytest.mark.parametrize('key', list(PATTERNS))
@pytest.mark.parametrize('user_only', [True, False])
def test_reviewed_fragment_scenarios_roundtrip_in_all_views(key, user_only):
    e = episode(key, user_only)
    e.validate()
    for t in e.transitions:
        row = extraction_row(e, t)
        prompt, target = [m['content'] for m in row['messages'][1:]]
        memory, user, previous = parse_prompt(prompt)
        assert decode_update(target, t.before, t.turn, t.previous_assistant) == t.delta
        decode_update(target, memory, user, previous)
        assert not target.startswith(('{', '['))
    for view in ('history', 'memory'):
        messages = response_row(e, view)['messages']
        assert messages[-1]['content'] == e.answer
        assert not messages[1]['content'].startswith(('{', '['))


def test_choice_is_not_a_preference_and_learning_location_is_not_current_location():
    e = episode('coffee_choice')
    assert e.memory.find('user', 'selection').value == 'white coffee'
    assert e.memory.find('user', 'drink').value == 'tea'
    selection = e.memory.find('user', 'selection')
    assert [s.turn_id for s in selection.evidence] == ['u2', 'a1']
    e = episode('learning_location')
    assert e.memory.find('user', 'learning_location').value == 'at school, in China'
    assert e.memory.find('user', 'location').value == 'the kitchen'
    assert 'China' in e.answer
    assert 'the kitchen' not in e.answer


def test_known_clause_survives_cut_off_continuation():
    e = episode('interrupted_feeling')
    claim = e.memory.find('user', 'feeling')
    assert claim.value == 'tired' and claim.evidence[0].quote.endswith('but...')
    assert len(claim.evidence) == 1  # The clause states the feeling itself.
    e = episode('interrupted_desire_known_object')
    event = e.memory.events[0].event
    assert event.operators == ('want',) and not event.missing_roles
    assert EventArgument('theme', 'a book', 'PHYSICAL_OBJECT') in event.arguments
    assert completion_evidence(event) == 'unknown'
    for text in (event_statement(event), event_question(event), event_description(event)):
        assert 'unspecified' not in text and 'today' not in text


def test_missing_objects_and_destinations_are_explicit_not_fabricated():
    for key in ('want_buy_missing_object', 'need_pick_up_missing_object', 'want_go_missing_destination'):
        e = episode(key)
        event = e.memory.events[0].event
        assert event.missing_roles and event.time == 'unspecified'
        assert event.arguments == (EventArgument('agent', 'user', 'PERSON'),)
        assert completion_evidence(event) == 'unknown' and e.expected_action == 'unknown'
        assert 'unspecified' in event_statement(event).lower()
        assert event_question(event) and event_description(event)
    contextual = episode('contextual_buy_missing_object')
    assert len(contextual.memory.events[0].evidence) == 2
    assert contextual.memory.events[0].event.operators == ('want',)


def test_refinement_retires_only_partial_intention_and_never_confirms_purchase():
    e = episode('resolve_missing_object')
    old, = e.memory.retired_events
    current, = e.memory.events
    assert old.event.missing_roles == ('theme',) and current.event.missing_roles == ()
    assert current.event.operators == old.event.operators == ('want',)
    assert current.event.time == old.event.time == 'unspecified'
    assert len(current.evidence) == 2
    assert e.expected_action == 'unknown'
    target = extraction_row(e, e.transitions[2])['messages'][-1]['content']
    assert 'EVENT_CONTEXT buy' in target and 'REMOVE_PARTIAL buy' in target
    assert e.memory.find('user', 'location').value == 'the kitchen'


def test_ambiguous_and_noninformative_replies_do_not_create_facts():
    e = episode('ambiguous_choice')
    assert e.expected_action == 'clarify' and e.transitions[-1].delta == Delta()
    assert e.memory.find('user', 'selection') is None
    e = episode('acknowledgement')
    assert e.transitions[1].delta == Delta() and e.expected_action == 'unknown'
    e = episode('negative_day')
    day = e.memory.find('user', 'meeting_day')
    assert day.polarity == 'negative' and e.expected_action == 'unknown'
    e = episode('possible_visit')
    assert e.memory.events[0].event.operators == ('might',)
    assert e.expected_action == 'unknown'


@pytest.mark.parametrize('payload', [
    'PARTIAL buy | unspecified | want | completed | present | missing=theme | agent=PERSON:user',
    'PARTIAL buy | unspecified | want | unspecified | present | agent=PERSON:user',
    'PARTIAL buy | unspecified | want | unspecified | present | missing=theme | missing=theme | agent=PERSON:user',
    'PARTIAL buy | unspecified | want | unspecified | present | missing=price | agent=PERSON:user',
    'PARTIAL buy | unspecified | want | unspecified | present | missing=theme,theme | agent=PERSON:user',
    'PARTIAL buy | unspecified | want | unspecified | present | missing=theme | agent=PERSON:user | theme=PHYSICAL_OBJECT:a book',
    'EVENT buy | unspecified | want | unspecified | present | missing=theme | agent=PERSON:user',
])
def test_invalid_partial_contracts_are_rejected_atomically(payload):
    with pytest.raises(ValueError):
        decode_update(payload, Memory(), Turn('u1', 'user', 'I want to buy...'))


def test_partial_past_and_state_are_never_evidence_for_a_complete_query():
    partial = Event('buy', (EventArgument('agent', 'user', 'PERSON'),),
                    missing_roles=('theme',), time='unspecified', tense='past')
    assert completion_evidence(partial) == 'unknown'
    complete = replace(partial, arguments=partial.arguments + (EventArgument('theme', 'a book', 'PHYSICAL_OBJECT'),),
                       missing_roles=(), phase='completed')
    assert partial.identity != complete.identity
    state = Event('have', (EventArgument('possessor', 'user', 'PERSON'),), missing_roles=('possession',))
    assert state_evidence(state) == 'unknown'
    predicate_only = Event('buy', (), time='unspecified', missing_roles=('agent', 'theme'))
    assert predicate_only.agent == 'unspecified participant'
    assert completion_evidence(predicate_only) == 'unknown'
    assert 'none specified' in event_statement(predicate_only)


def test_audit_event_reader_preserves_partial_roles_and_accepts_old_records(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from review_dialogue_roles import read_event
    partial = episode('want_buy_missing_object').memory.events[0].event
    complete = episode('interrupted_desire_known_object').memory.events[0].event
    for event in (partial, complete):
        assert read_event(json.loads(json.dumps(asdict(event)))) == event
    old = asdict(complete)
    old.pop('missing_roles')
    assert read_event(old) == complete


@pytest.mark.parametrize('user_only', [True, False])
def test_deterministic_balanced_generation_and_no_cross_split_prompts(user_only):
    generator = DialogueGenerator(seed=12, user_assistant_only=user_only)
    keys, scenarios = set(), set()
    for split in ('train', 'validation', 'test'):
        rows = list(generator.generate('fragment_meaning', 32, split=split))
        assert rows[:5] == list(generator.generate('fragment_meaning', 5, split=split))
        assert {e.condition for e in rows} == set(PATTERNS)
        current = {prompt_key(extraction_row(e, t)) for e in rows for t in e.transitions}
        assert not keys & current
        keys.update(current)
        assert not scenarios & {e.scenario_key for e in rows}
        scenarios.update(e.scenario_key for e in rows)
