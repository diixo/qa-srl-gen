"""Participants, passive morphology and independent events survive all views."""
from dataclasses import asdict, replace
import gzip
import json

import pytest

from semantic_corpus.dialogue_generator.events import Event, EventArgument, event_statement, event_question, occurrence_evidence
from semantic_corpus.dialogue_generator.extraction_text import _event_payload, _fields, parse_commands, decode_update
from semantic_corpus.dialogue_generator.mined_realization import active_phrase, lexicon, source_scope
from semantic_corpus.dialogue_generator.mined_arguments import realize_arguments
from semantic_corpus.dialogue_generator.mined_subjects import MinedSubject, realize_subject
from semantic_corpus.dialogue_generator.mined_scope import parse_scope
from semantic_corpus.dialogue_generator.mined_bank import DialogueMinedBank
from semantic_corpus.dialogue_generator.mined_scenarios import _source_event
from semantic_corpus.dialogue_generator.generator import DialogueGenerator
from semantic_corpus.dialogue_generator.export import extraction_row, response_row
from semantic_corpus.dialogue_generator.models import Memory, EventClaim, Query, answer_query
from semantic_corpus.template_bank import fingerprint
from test_template_semantics import template
from test_dialogue_mined import small_bank


def forms(lemma):
    return [asdict(p) for p in lexicon().paradigms(lemma)]


def event_from(row, head='v0', lemma='sell'):
    spec, reason = active_phrase(row, head, lemma, forms(lemma), {'Anna', 'Ben', 'Mary'})
    assert reason is None
    subject, reason = realize_subject(row, spec, row['defaults'], {'Anna', 'Ben', 'Mary'})
    assert reason is None
    (tail, args), reason = realize_arguments(row, spec, row['defaults'], {'Anna', 'Ben', 'Mary'})
    assert reason is None
    return _source_event({**spec, 'subject': subject, 'tail': tail, 'argument_values': args,
                          'source_scope': source_scope(spec, row['defaults'], forms(lemma))})


@pytest.mark.parametrize('voice', ['active', 'passive'])
def test_roles_and_voice_keep_subject_and_argument_distinct(voice):
    row = template(lemma='sell', verb='sold', passive=voice == 'passive', adjective=False)
    event = event_from(row)
    assert event.voice == voice
    assert event.subject.text == ('Anna' if voice == 'active' else 'the book')
    role = 'object' if voice == 'active' else 'by_phrase'
    arg = next(a for a in event.arguments if a.role == role)
    assert arg.text == ('a book' if voice == 'active' else 'Anna')
    kind, fields = _event_payload(event)
    assert parse_commands(kind + ' ' + _fields(fields))[0].event == event
    assert occurrence_evidence(event) == 'yes'
    assert event_question(event) == ('Did Anna sell a book today?' if voice == 'active' else 'Was the book sold by Anna today?')
    assert event.identity != replace(event, voice='active', arguments=()).identity


def test_agentless_passive_does_not_invent_an_agent():
    event = event_from(template(lemma='sell', verb='sold', passive=True, agent=False, adjective=False))
    assert not event.arguments
    assert event_statement(event, perspective='user') == 'The book was sold today.'
    assert event_statement(replace(event, operators=('not',)), perspective='user') == 'The book was not sold today.'
    assert event_question(event) == 'Was the book sold today?'


def test_passive_participle_and_pronoun_by_phrase_are_not_active_agents():
    row = template(lemma='see', verb='seen', passive=True, adjective=False)
    row['defaults']['n0'] = 'me'
    row['slots']['n0'].update(pos='PRON', tag='PRP')
    event = event_from(row, lemma='see')
    assert event_statement(event, perspective='user') == 'The book was seen by me today.'
    assert event_statement(event) == 'The book was seen by you today.'
    row['defaults']['v0'] = 'saw'
    assert active_phrase(row, 'v0', 'see', forms('see'), {'Anna'})[1] == 'unsupported_source_scope'


def test_object_role_keeps_separated_phrasal_particle():
    event = event_from(template(lemma='pick', verb='picked', particle=True, adjective=False), lemma='pick')
    assert event.predicate == 'pick up'
    assert event_statement(event, perspective='user') == 'Anna picked a book up today.'
    assert next(a.text for a in event.arguments if a.role == 'object') == 'a book'


@pytest.mark.parametrize('surface,ops,tense,phase,statement,evidence', [
    ('were sold', (), 'past', 'reported', 'The books were sold today.', 'yes'),
    ('were not sold', ('not',), 'past', 'reported', 'The books were not sold today.', 'no'),
    ('were being sold', (), 'past', 'ongoing', 'The books were being sold today.', 'unknown'),
    ('were not being sold', ('not',), 'past', 'ongoing', 'The books were not being sold today.', 'unknown'),
    ('have been sold', ('have',), 'present', 'unspecified', 'The books have been sold today.', 'yes'),
    ('may have been sold', ('may', 'have'), 'modal', 'unspecified', 'The books may have been sold today.', 'unknown'),
    ('will not be sold', ('not',), 'future', 'unspecified', 'The books will not be sold today.', 'unknown'),
])
def test_passive_scope_and_plural_agreement(surface, ops, tense, phase, statement, evidence):
    scope, reason = parse_scope(surface, 'sell', forms('sell'), voice='passive')
    assert reason is None and (scope.operators, scope.tense, scope.phase) == (ops, tense, phase)
    event = Event('sell', (), subject=MinedSubject('the books', 'PHYSICAL_OBJECT', 'plural'), mined=True,
                  voice='passive', **asdict(scope))
    assert event_statement(event, perspective='user') == statement
    assert occurrence_evidence(event) == evidence


@pytest.mark.parametrize('surface', ['did not be sold', 'does be sold', 'was sell', 'was selling'])
def test_passive_rejects_active_forms_and_do_support(surface):
    assert parse_scope(surface, 'sell', forms('sell'), voice='passive')[0] is None


def test_passive_embedded_under_lexical_controller_keeps_do_support_on_controller():
    scope, reason = parse_scope('did not want to be seen', 'see', forms('see'), voice='passive')
    assert reason is None and scope.operators == ('not', 'want') and scope.tense == 'past'


def test_object_and_possessive_perspective_changes_are_local():
    row = template(lemma='help', verb='helped', adjective=False)
    row['defaults']['n0'] = 'I'
    row['slots']['n0'].update(pos='PRON', tag='PRP')
    row['defaults']['n1'] = 'you'
    row['slots']['n1'].update(pos='PRON', tag='PRP')
    row['parts'][3] = {'literal': ' '}
    row['parts'][4] = {'literal': ''}  # remove noun article
    event = event_from(row, lemma='help')
    assert event_statement(event, perspective='user') == 'I helped you today.'
    assert event_statement(event) == 'You helped me today.'
    assert next(a.text for a in event.arguments if a.role == 'object') == 'assistant'
    row = template(lemma='sell', verb='sold', adjective=False)
    row['parts'][4] = {'literal': 'my '}
    event = event_from(row)
    assert event_statement(event) == 'Anna sold your book today.'


def test_ditransitive_and_prepositional_arguments_preserve_order():
    row = template(lemma='send', verb='sent', adjective=False)
    row['slots']['n2'] = {'kind': 'nominal', 'pos': 'PROPN', 'tag': 'NNP', 'dependency': 'iobj', 'head_slot': 'v0'}
    row['defaults']['n2'] = 'Ben'
    row['parts'][3:3] = [{'literal': ' '}, {'slot': 'n2'}]
    event = event_from(row, lemma='send')
    assert event_statement(event, perspective='user') == 'Anna sent Ben a book today.'
    assert {a.role for a in event.arguments} == {'object', 'indirect_object', 'complement'}
    row['slots']['n2'].update(dependency='pobj', head_slot=None)
    del row['parts'][3:5]
    index = len(row['parts']) - 1
    row['parts'][-1:] = [{'literal': ' to '}, {'slot': 'n2'}, {'literal': '.'}]
    row['predicate_groups'][0]['constructions'] = [{'kind': 'preposition_candidate', 'lemma': 'send to',
                                                   'markers': [{'part': index, 'start': 1, 'end': 3}]}]
    event = event_from(row, lemma='send')
    assert event_statement(event, perspective='user') == 'Anna sent a book to Ben today.'
    assert {a.role for a in event.arguments} == {'object', 'oblique_to', 'complement'}


def coordinated():
    row = template(lemma='sell', verb='sold', adjective=False)
    row['slots'].update(n2={'kind': 'nominal', 'pos': 'PROPN', 'tag': 'NNP', 'dependency': 'nsubj', 'head_slot': 'v2'},
                        v2={'kind': 'verb', 'pos': 'VERB', 'tag': 'VBD', 'lemma': 'leave', 'dependency': 'conj', 'head_slot': 'v0'})
    row['defaults'].update(n2='Mary', v2='left')
    row['parts'][-1:] = [{'literal': ' and '}, {'slot': 'n2'}, {'literal': ' '}, {'slot': 'v2'}, {'literal': '.'}]
    row['predicate_groups'].append({'head_slot': 'v2', 'lemma': 'leave', 'constructions': []})
    return row


def test_independent_clauses_are_all_or_quotation():
    row = coordinated()
    spec, reason = active_phrase(row, 'v0', 'sell', forms('sell'), {'Anna', 'Mary'})
    assert reason is None and len(spec['companions']) == 1
    assert spec['companions'][0]['predicate'] == 'leave'
    # A shared or unresolved subject cannot be inferred from mere adjacency.
    row['slots']['n2']['dependency'] = 'dobj'
    assert active_phrase(row, 'v0', 'sell', forms('sell'), {'Anna', 'Mary'})[0] is None
    row = coordinated()
    row['parts'][-5]['literal'] = ' or '
    assert active_phrase(row, 'v0', 'sell', forms('sell'), {'Anna', 'Mary'})[0] is None
    row = coordinated()
    row['parts'][-3]['literal'] = ' never '
    assert active_phrase(row, 'v0', 'sell', forms('sell'), {'Anna', 'Mary'})[1] == 'unsupported_companion_clause'


def test_companion_identity_cannot_supply_conflicting_answer_to_primary():
    row = coordinated()
    # Two reports about one event require explicit discourse/correction scope.
    row['defaults']['n2'] = 'Anna'
    row['defaults']['v2'] = 'sold'
    row['slots']['v2']['lemma'] = 'sell'
    row['slots']['n3'] = {**row['slots']['n1'], 'head_slot': 'v2'}
    row['defaults']['n3'] = 'book'
    row['parts'][-1:] = [{'literal': ' a '}, {'slot': 'n3'}, {'literal': '.'}]
    row['predicate_groups'][-1]['lemma'] = 'sell'
    assert active_phrase(row, 'v0', 'sell', forms('sell'), {'Anna'})[1] == 'overlapping_companion_events'


@pytest.mark.parametrize('user_only', [True, False])
def test_multi_event_export_and_selective_withdrawal(small_bank, user_only):
    row = coordinated()
    row.update(verbs=['sell', 'leave'], fillers={k: [v] for k, v in row['defaults'].items()})
    row['template_id'] = fingerprint(row)
    bank_path = small_bank.banks[0].path
    with gzip.open(bank_path / 'templates' / (row['template_id'][:2] + '.jsonl.gz'), 'at', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')
    with gzip.open(bank_path / 'verbs.jsonl.gz', 'at', encoding='utf-8') as f:
        for lemma, word in [('sell', 'sold'), ('leave', 'left')]:
            f.write(json.dumps({'lemma': lemma, 'forms': {word: 1}}) + '\n')
    with small_bank.dictionary.open('a', encoding='utf-8') as f:
        f.write('sell\tsells\tselling\tsold\tsold\nleave\tleaves\tleaving\tleft\tleft\n')
    bank = DialogueMinedBank(small_bank.root, cache=False)
    assert bank.report['multi_event_occurrences'] == 2
    episodes = list(DialogueGenerator(user_assistant_only=user_only, mined_bank=bank).generate('mined_predicates', 15, split='train'))
    selected = [e for e in episodes if e.mined_source.get('companion_heads')]
    assert len(selected) == 6
    for episode in selected:
        initial = episode.transitions[1]
        assert len(initial.delta.event_add) == 2
        companion = initial.delta.event_add[1].event
        claim = episode.memory.find_event(companion)
        assert claim and claim.event == companion
        assert answer_query(episode.memory, Query('mined_occurrence', event=companion))[0].startswith('Yes')
        if episode.condition.endswith((':no', ':unknown')):
            assert any('Withdraw only this report:' in t.text for t in episode.turns)
        assert episode.turns[2].text.index('Anna') < episode.turns[2].text.index('Mary')
        for transition in episode.transitions:
            target = extraction_row(episode, transition)['messages'][-1]['content']
            assert decode_update(target, transition.before, transition.turn, transition.previous_assistant) == transition.delta
        assert response_row(episode, 'memory')['messages'][-1]['content'] == episode.answer
        assert response_row(episode, 'history')['messages'][-1]['content'] == episode.answer


def test_contract_rejects_fabricated_or_overlapping_roles():
    subject = MinedSubject('Anna', 'PERSON')
    for args in [(EventArgument('object', 'a book', 'PHYSICAL_OBJECT'),),
                 (EventArgument('agent', 'Anna', 'PERSON'),),
                 (EventArgument('complement', 'a book', 'UNRESOLVED'), EventArgument('object', 'a car', 'PHYSICAL_OBJECT'))]:
        with pytest.raises(ValueError):
            Event('sell', args, mined=True, subject=subject)
