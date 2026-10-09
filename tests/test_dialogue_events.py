"""Event scope, shared slot constraints and end-to-end text update semantics."""

from collections import Counter, defaultdict
from dataclasses import replace

import pytest
from semantic_corpus.ontology import Label

from semantic_corpus.dialogue_generator.events import (
    EVENT_LEXEMES, Event, EventArgument, completion_evidence, event_question, event_statement,
)
from semantic_corpus.dialogue_generator.event_scenarios import EventSampler, EVENT_TASKS, SCOPE_PROFILES
from semantic_corpus.dialogue_generator.generator import DialogueGenerator
from semantic_corpus.dialogue_generator.export import extraction_row, response_row
from semantic_corpus.dialogue_generator.extraction_text import decode_update, parse_prompt, render_prompt, render_update
from semantic_corpus.dialogue_generator.models import (
    Delta, Evidence, EventClaim, EventReplacement, Fact, Memory, Query, Transition, Turn, answer_query, apply_delta,
)
from semantic_corpus.semantic_generator.substitutions import load_pool


def purchase(item='a book', time='today', **kwargs):
    return Event('buy', (EventArgument('agent', 'user', 'PERSON'), EventArgument('theme', item, 'PHYSICAL_OBJECT')),
                 time=time, **kwargs)


@pytest.mark.parametrize('operators,phase,tense,expected', [
    ((), 'completed', 'present', 'yes'), (('not',), 'completed', 'present', 'no'),
    ((), 'ongoing', 'present', 'unknown'), (('not',), 'ongoing', 'present', 'unknown'),
    ((), 'unspecified', 'future', 'unknown'),
    (('want',), 'unspecified', 'present', 'unknown'), (('need',), 'unspecified', 'present', 'unknown'),
    (('not', 'need'), 'unspecified', 'present', 'unknown'), (('need', 'not'), 'unspecified', 'present', 'unknown'),
    (('not', 'want'), 'unspecified', 'present', 'unknown'), (('want', 'not'), 'unspecified', 'present', 'unknown'),
    (('plan', 'try'), 'unspecified', 'present', 'unknown'),
    (('want', 'manage'), 'unspecified', 'present', 'unknown'),
    (('try',), 'unspecified', 'past', 'unknown'), (('start',), 'unspecified', 'past', 'unknown'),
    (('stop',), 'unspecified', 'past', 'unknown'), (('not', 'manage'), 'unspecified', 'past', 'no'),
    (('not', 'finish'), 'unspecified', 'past', 'no'), (('manage', 'not'), 'unspecified', 'past', 'no'),
    (('fail', 'not'), 'unspecified', 'past', 'yes'),
    (('manage',), 'unspecified', 'past', 'yes'), (('finish',), 'unspecified', 'past', 'yes'),
    (('fail',), 'unspecified', 'past', 'no'), (('manage', 'try'), 'unspecified', 'past', 'unknown'),
    (('manage',), 'unspecified', 'present', 'unknown'),
])
def test_completion_is_not_inferred_from_action_label_or_modal_word(operators, phase, tense, expected):
    event = purchase(operators=operators, phase=phase, tense=tense)
    assert completion_evidence(event) == expected
    claim = EventClaim('e1', event, ())
    answer, action, _ = answer_query(Memory(events=(claim,)), Query('event_completion', event=purchase()))
    assert ('unknown' if action == 'unknown' else 'yes' if answer.startswith('Yes') else 'no') == expected


def test_scope_and_particles_survive_surface_alternation():
    event = Event('pick up', purchase().arguments)
    assert event_statement(replace(event, operators=('not', 'want')), perspective='user') == 'I do not want to pick up a book today.'
    assert event_statement(replace(event, operators=('want', 'not')), perspective='user') == 'I want not to pick up a book today.'
    assert event_statement(replace(event, operators=('not', 'need')), perspective='user') == 'I do not need to pick up a book today.'
    assert event_statement(replace(event, operators=('need', 'not')), perspective='user') == 'I need to avoid picking up a book today.'
    assert event_statement(replace(event, operators=('plan', 'try')), perspective='user', style=1) == 'I plan to try to pick a book up today.'
    assert event_statement(replace(event, operators=('need', 'start')), perspective='user') == 'I need to start picking up a book today.'
    assert event_question(event, style=1) == 'Have I picked a book up today?'
    assert event_statement(replace(event, phase='ongoing'), perspective='user') == 'I am picking up a book today.'
    assert event_statement(replace(event, phase='completed', tense='past'), perspective='user') == 'I picked up a book today.'
    pronoun = replace(event, arguments=(event.arguments[0], replace(event.arguments[1], text='it')), phase='completed')
    assert event_statement(pronoun, perspective='user') == 'I have picked it up today.'


def test_prepositional_and_intransitive_frames_do_not_gain_fake_objects():
    care = Event('look after', (EventArgument('agent', 'user', 'PERSON'), EventArgument('theme', 'the child', 'PERSON')))
    assert event_statement(replace(care, operators=('want',)), perspective='user', style=1) == 'I want to look after the child today.'
    assert event_statement(replace(care, phase='completed'), perspective='user') == 'I have finished looking after the child today.'
    get_up = Event('get up', (EventArgument('agent', 'user', 'PERSON'),), phase='completed')
    assert event_statement(get_up, perspective='user') == 'I have got up today.'
    with pytest.raises(ValueError, match='arguments'):
        Event('get up', purchase().arguments)
    with pytest.raises(ValueError, match='incompatible'):
        Event('look after', purchase().arguments)
    with pytest.raises(ValueError, match='incompatible'):
        Event('buy', (EventArgument('agent', 'user', 'PERSON'), EventArgument('theme', 'Paris', 'LOCATION')))


def test_shared_pool_fills_every_required_role_and_keeps_selection_constraints():
    import random
    sampler = EventSampler(load_pool())
    for lexeme in EVENT_LEXEMES:
        for _ in range(8):
            event = sampler.sample(lexeme, random.Random(_))
            assert {arg.role for arg in event.arguments} == lexeme.frame.pattern(lexeme.pattern).mentioned()
            for arg in event.arguments:
                assert lexeme.frame.slot_map[arg.role].accepts((Label(arg.label),))
            for profile in SCOPE_PROFILES:
                candidate = replace(event, operators=profile)
                assert completion_evidence(candidate) == 'unknown'
                assert event_statement(candidate, perspective='user').endswith('today.')


def test_multiple_objects_times_and_intentions_coexist_and_corrections_are_explicit():
    user = Turn('u1', 'user', 'I bought a book today and a ticket yesterday. I want to buy a coat.')
    events = (purchase(phase='completed'), purchase('a ticket', 'yesterday', phase='completed'),
              purchase('a coat', operators=('want',)))
    claims = tuple(EventClaim(f'e{i}', event, (Evidence(user.id, user.text),)) for i, event in enumerate(events, 1))
    memory = apply_delta(Memory(), Delta(event_add=claims), (user,))
    assert len(memory.events) == 3
    assert answer_query(memory, Query('event_completion', event=purchase()))[0].startswith('Yes')
    assert answer_query(memory, Query('event_completion', event=purchase('a ticket')))[1] == 'unknown'
    assert answer_query(memory, Query('event_completion', event=purchase('a coat')))[1] == 'unknown'
    later = Turn('u2', 'user', 'Correction: I have not bought a book today.')
    claim = EventClaim('e4', replace(events[0], operators=('not',)), (Evidence(later.id, later.text),))
    corrected = apply_delta(memory, Delta(event_replace=(EventReplacement('e1', claim),)), (later,))
    assert len(corrected.events) == 3 and corrected.retired_events == (claims[0],)
    assert answer_query(corrected, Query('event_completion', event=purchase()))[0].startswith('No')
    assert corrected.find_event(events[1]) == claims[1]
    forgotten = apply_delta(corrected, Delta(event_invalidate=('e4',)), (later,))
    assert answer_query(forgotten, Query('event_completion', event=purchase()))[1] == 'unknown'
    with pytest.raises(ValueError, match='identity'):
        apply_delta(memory, Delta(event_replace=(EventReplacement('e1', replace(claim, event=events[1])),)), (later,))
    with pytest.raises(ValueError, match='Conflicting'):
        apply_delta(memory, Delta(event_add=(claim,)), (later,))
    with pytest.raises(ValueError, match='fresh IDs'):
        apply_delta(memory, Delta(add=(Fact('e1', 'user', 'drink', 'tea', (Evidence(later.id, later.text),)),)), (later,))


def test_conflicting_report_channels_do_not_create_a_certain_answer():
    yes = EventClaim('e1', purchase(phase='completed'), ())
    no = EventClaim('e2', purchase(operators=('fail',), tense='past'), ())
    assert answer_query(Memory(events=(yes, no)), Query('event_completion', event=purchase()))[1] == 'unknown'


def test_inner_negation_is_a_distinct_claim_and_outer_negation_is_a_correction():
    turn = Turn('u1', 'user', "I don't want to buy a book. I want not to buy a book.")
    without_desire = purchase(operators=('not', 'want'))
    desired_avoidance = purchase(operators=('want', 'not'))
    assert without_desire.key != desired_avoidance.key
    assert without_desire.key == purchase(operators=('want',)).key
    claims = tuple(EventClaim(f'e{i}', event, (Evidence(turn.id, turn.text),))
                   for i, event in enumerate((without_desire, desired_avoidance), 1))
    memory = apply_delta(Memory(), Delta(event_add=claims), (turn,))
    assert len(memory.events) == 2
    assert answer_query(memory, Query('event_completion', event=purchase()))[1] == 'unknown'


@pytest.mark.parametrize('task', EVENT_TASKS)
def test_scenario_groups_roundtrip_events_sources_and_answers(task):
    episodes = list(DialogueGenerator(user_assistant_only=True).generate(task, 192, split='test'))
    groups = defaultdict(list)
    predicates, operators = set(), set()
    for episode in episodes:
        episode.validate()
        groups[episode.contrast_group].append(episode)
        predicates.add(episode.query.event.predicate)
        for claim in episode.memory.events:
            operators.update(claim.event.operators)
        for transition in episode.transitions:
            row = extraction_row(episode, transition)
            prompt, target = [message['content'] for message in row['messages'][1:]]
            assert '{' not in prompt + target
            delta = decode_update(target, transition.before, transition.turn, transition.previous_assistant)
            assert delta == transition.delta
            visible, user, previous = parse_prompt(prompt)
            decode_update(target, visible, user, previous)
        assert 'Reported events' in response_row(episode, 'memory')['messages'][1]['content']
    assert predicates == {lexeme.lemma for lexeme in EVENT_LEXEMES}
    for group in groups.values():
        assert len(group) == 3
        assert {e.condition.rsplit(':', 1)[-1] for e in group} == {'yes', 'no', 'unknown'}
        actual = Counter('unknown' if e.expected_action == 'unknown' else 'yes' if e.answer.startswith('Yes') else 'no' for e in group)
        assert actual == {'yes': 1, 'no': 1, 'unknown': 1}
    if task == 'event_scope':
        assert {'need', 'want', 'plan', 'intend', 'decide', 'not', 'try'} <= operators


def test_event_metrics_do_not_ignore_wrong_scope_or_event_identity():
    from semantic_corpus.dialogue_generator.trainer.metrics import ExtractionMetrics
    user = Turn('u1', 'user', 'I want to pick up a book today.')
    event = Event('pick up', purchase().arguments, operators=('want',))
    claim = EventClaim('e1', event, (Evidence(user.id, user.text),))
    delta = Delta(event_add=(claim,))
    transition = Transition(user, Memory(), delta, apply_delta(Memory(), delta, (user,)), None)
    target = render_update(transition)
    prompt = render_prompt(Memory(), user)
    metrics = ExtractionMetrics()
    metrics.add(target, target, prompt=prompt)
    metrics.add(target, target.replace('want | unspecified', 'none | completed'), prompt=prompt)
    metrics.add(target, target.replace('a book', 'a coat'), prompt=prompt)
    report = metrics.report()['extraction']
    assert report['valid_predictions'] == 3
    assert report['state_matches'] == report['update_matches'] == 1
    assert report['by_operation']['event_add']['count'] == 3


def test_scope_sampling_crosses_predicates_and_operators_instead_of_lockstep_pairing():
    expected = {(lexeme.lemma, profile) for lexeme in EVENT_LEXEMES for profile in SCOPE_PROFILES}
    episodes = DialogueGenerator(user_assistant_only=True).generate('event_scope', len(expected) * 3, split='train')
    observed = set()
    for episode in episodes:
        initial = next(t.delta.event_add[0].event for t in episode.transitions if t.delta.event_add)
        observed.add((initial.predicate, initial.operators))
    assert observed == expected


def test_contextual_event_updates_preserve_both_sources_and_fresh_ids():
    previous = Turn('a1', 'assistant', 'You could pick up a book today.')
    user = Turn('u1', 'user', 'I want to try that.')
    text = 'EVENT_CONTEXT pick up | today | want>try | unspecified | present | agent=PERSON:user | theme=PHYSICAL_OBJECT:a book'
    delta = decode_update(text, Memory(), user, previous)
    claim = delta.event_add[0]
    assert claim.id == 'e1'
    assert claim.evidence == (Evidence(user.id, user.text), Evidence(previous.id, previous.text))
    after = apply_delta(Memory(), delta, (previous, user))
    assert render_update(Transition(user, Memory(), delta, after, previous)) == text
    assert answer_query(after, Query('event_completion', event=Event('pick up', purchase().arguments)))[1] == 'unknown'
    with pytest.raises(ValueError, match='previous assistant'):
        decode_update(text, Memory(), user)
