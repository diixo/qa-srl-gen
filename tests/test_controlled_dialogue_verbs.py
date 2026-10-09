"""Irregular/defective forms, scope, role preservation and reported-state logic."""
from dataclasses import replace

import pytest

from semantic_corpus.dialogue_generator.events import Event, EventArgument, completion_evidence, event_statement, event_question
from semantic_corpus.dialogue_generator.extraction_text import decode_update, render_update
from semantic_corpus.dialogue_generator.models import EventClaim, Memory, Query, Transition, Turn, answer_query, apply_delta


def buy(**kwargs):
    return Event('buy', (EventArgument('agent', 'user', 'PERSON'), EventArgument('theme', 'a book', 'PHYSICAL_OBJECT')), **kwargs)


@pytest.mark.parametrize('scope,tense,sentence', [
    (('can',), 'modal', 'I can buy a book today.'),
    (('not', 'can'), 'modal', 'I cannot buy a book today.'),
    (('could',), 'modal', 'I could buy a book today.'),
    (('not', 'could'), 'modal', 'I could not buy a book today.'),
    (('must',), 'modal', 'I must buy a book today.'),
    (('must', 'not'), 'modal', 'I must not buy a book today.'),
    (('may',), 'modal', 'I may buy a book today.'),
    (('may', 'not'), 'modal', 'I may not buy a book today.'),
    (('might', 'have'), 'modal', 'I might have bought a book today.'),
    (('must', 'have'), 'modal', 'I must have bought a book today.'),
    (('may', 'have', 'be'), 'modal', 'I may have been buying a book today.'),
    (('can', 'be'), 'modal', 'I can be buying a book today.'),
    (('have_to',), 'present', 'I have to buy a book today.'),
    (('not', 'have_to'), 'present', 'I do not have to buy a book today.'),
    (('have_to',), 'past', 'I had to buy a book today.'),
    (('have_to',), 'future', 'I will have to buy a book today.'),
    (('be_able_to',), 'past', 'I was able to buy a book today.'),
    (('be_going_to',), 'present', 'I am going to buy a book today.'),
    (('not', 'be_going_to'), 'past', 'I was not going to buy a book today.'),
    (('have', 'be'), 'past', 'I had been buying a book today.'),
    (('let_us',), 'present', "Let's buy a book today."),
    (('let_us', 'not'), 'present', "Let's not buy a book today."),
])
def test_scoped_forms_do_not_assert_completion(scope, tense, sentence):
    event = buy(operators=scope, tense=tense)
    assert event_statement(event, perspective='user') == sentence
    assert completion_evidence(event) == 'unknown'


@pytest.mark.parametrize('operators,tense', [
    (('must',), 'past'), (('can',), 'present'), (('want', 'can'), 'modal'),
    (('may', 'must'), 'modal'), (('start', 'may'), 'modal'),
    (('not', 'must'), 'modal'), (('not', 'may'), 'modal'),
    (('be',), 'present'), (('have',), 'present'),
    (('start', 'be_going_to'), 'present'), (('want', 'let_us'), 'present'),
    (('can', 'not'), 'modal'), (('could', 'not'), 'modal'),
])
def test_unlicensed_modal_forms_and_duplicate_aspect_encodings_are_rejected(operators, tense):
    with pytest.raises(ValueError):
        buy(operators=operators, tense=tense)


def test_let_keeps_permitter_separate_from_actor_and_roundtrips_without_json():
    event = buy(operators=('let',), tense='past', permitter=EventArgument('permitter', 'assistant', 'PERSON'))
    text = event_statement(event, perspective='user')
    assert text == 'You let me buy a book today.'
    assert event_statement(event) == 'I let you buy a book today.'
    command = 'EVENT buy | today | let | unspecified | past | agent=PERSON:user | theme=PHYSICAL_OBJECT:a book | permitter=PERSON:assistant'
    turn = Turn('u1', 'user', text)
    delta = decode_update(command, Memory(), turn)
    assert delta.event_add[0].event == event
    after = apply_delta(Memory(), delta, (turn,))
    assert render_update(Transition(turn, Memory(), delta, after, None)) == command
    assert answer_query(after, Query('event_completion', event=buy()))[1] == 'unknown'
    assert event.key != replace(event, permitter=EventArgument('permitter', 'Anna', 'PERSON')).key
    named = replace(event, permitter=EventArgument('permitter', 'Anna', 'PERSON'))
    assert event_statement(named, style=3) == 'You reported: "Anna let me buy a book today."'
    with pytest.raises(ValueError, match='permitter'):
        buy(operators=('let',))


@pytest.mark.parametrize('lemma,arguments,positive,negative,past,question', [
    ('be', (EventArgument('entity', 'user', 'PERSON'), EventArgument('location', 'the park', 'LOCATION')),
     'I am in the park today.', 'I am not in the park today.', 'I was in the park today.', 'Am I in the park today?'),
    ('have', (EventArgument('possession', 'a book', 'PHYSICAL_OBJECT'), EventArgument('possessor', 'user', 'PERSON')),
     'I have a book today.', 'I do not have a book today.', 'I had a book today.', 'Do I have a book today?'),
])
def test_lexical_states_are_not_completed_actions(lemma, arguments, positive, negative, past, question):
    base = Event(lemma, arguments)
    assert event_statement(base, perspective='user') == positive
    assert event_statement(replace(base, operators=('not',)), perspective='user') == negative
    assert event_statement(replace(base, tense='past'), perspective='user') == past
    assert event_question(base) == question
    for operators, expected in [((), 'Yes,'), (('not',), 'No,'), (('want',), "I don't know")]:
        event = replace(base, operators=operators)
        memory = Memory(events=(EventClaim('e1', event, ()),))
        assert answer_query(memory, Query('event_state', event=base))[0].startswith(expected)
        assert completion_evidence(event) == 'unknown'
    old = Memory(events=(EventClaim('e1', replace(base, tense='past'), ()),))
    assert answer_query(old, Query('event_state', event=base))[1] == 'unknown'
    with pytest.raises(ValueError):
        replace(base, phase='completed')


def test_going_to_a_place_and_going_to_act_are_distinct():
    motion = Event('go', (EventArgument('agent', 'user', 'PERSON'), EventArgument('destination', 'the park', 'LOCATION')),
                   phase='ongoing')
    assert event_statement(motion, perspective='user') == 'I am going to the park today.'
    assert completion_evidence(motion) == 'unknown'
    past = replace(motion, phase='completed', tense='past')
    assert event_statement(past, perspective='user') == 'I went to the park today.'
    assert completion_evidence(past) == 'yes'
    assert event_statement(replace(past, tense='present'), perspective='user') == 'I have gone to the park today.'
    assert event_statement(buy(operators=('be_going_to',)), perspective='user') == 'I am going to buy a book today.'


def test_modal_contractions_and_named_subject_agreement():
    assert event_statement(buy(operators=('not', 'can'), tense='modal'), perspective='user', style=2) == "I can't buy a book today."
    assert event_statement(buy(operators=('must', 'not'), tense='modal'), perspective='user', style=2) == "I mustn't buy a book today."
    assert event_statement(buy(operators=('not', 'have_to')), perspective='user', style=2) == "I don't have to buy a book today."
    named = replace(buy(), arguments=(EventArgument('agent', 'Anna', 'PERSON'), buy().arguments[1]))
    assert event_statement(replace(named, operators=('can',), tense='modal')) == 'Anna can buy a book today.'
    assert event_statement(replace(named, operators=('have_to',))) == 'Anna has to buy a book today.'
    assert event_statement(replace(named, operators=('be_going_to',))) == 'Anna is going to buy a book today.'
    with pytest.raises(ValueError, match='let_us'):
        replace(named, operators=('let_us',))
    assert event_statement(buy(operators=('be_able_to',), tense='past')) == 'You were able to buy a book today.'
