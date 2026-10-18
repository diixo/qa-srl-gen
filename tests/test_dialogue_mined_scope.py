"""Source scope must survive mining, generation, extraction and memory queries."""
from dataclasses import asdict, replace
import gzip
import json

import pytest

from semantic_corpus.dialogue_generator.events import Event, completion_evidence, occurrence_evidence, event_statement
from semantic_corpus.dialogue_generator.extraction_text import _event_payload, _fields, parse_commands, decode_update
from semantic_corpus.dialogue_generator.mined_scope import parse_scope
from semantic_corpus.dialogue_generator.mined_realization import active_phrase, source_scope, lexicon
from semantic_corpus.dialogue_generator.mined_subjects import MinedSubject
from semantic_corpus.dialogue_generator.models import Memory, EventClaim, Query, answer_query
from semantic_corpus.dialogue_generator.generator import DialogueGenerator
from semantic_corpus.dialogue_generator.export import extraction_row
from semantic_corpus.dialogue_generator.mined_bank import DialogueMinedBank
from semantic_corpus.template_bank import fingerprint
from test_dialogue_mined import small_bank


def paradigms(lemma='leave'):
    return [asdict(p) for p in lexicon().paradigms(lemma)]


@pytest.mark.parametrize('text, operators, tense, phase, evidence', [
    ('wanted to leave', ('want',), 'past', 'unspecified', 'unknown'),
    ('did not want to leave', ('not', 'want'), 'past', 'unspecified', 'unknown'),
    ('wanted not to leave', ('want', 'not'), 'past', 'unspecified', 'unknown'),
    ('wanted to not leave', ('want', 'not'), 'past', 'unspecified', 'unknown'),
    ('needs to leave', ('need',), 'present', 'unspecified', 'unknown'),
    ('needed to try to leave', ('need', 'try'), 'past', 'unspecified', 'unknown'),
    ('planned to manage to leave', ('plan', 'manage'), 'past', 'unspecified', 'unknown'),
    ('intended to leave', ('intend',), 'past', 'unspecified', 'unknown'),
    ('decided not to leave', ('decide', 'not'), 'past', 'unspecified', 'unknown'),
    ('tried to leave', ('try',), 'past', 'unspecified', 'unknown'),
    ('managed to leave', ('manage',), 'past', 'unspecified', 'yes'),
    ('managed not to leave', ('manage', 'not'), 'past', 'unspecified', 'no'),
    ('failed to leave', ('fail',), 'past', 'unspecified', 'no'),
    ('failed not to leave', ('fail', 'not'), 'past', 'unspecified', 'yes'),
    ('did not manage to leave', ('not', 'manage'), 'past', 'unspecified', 'unknown'),
    ('started leaving', ('start',), 'past', 'unspecified', 'unknown'),
    ('started to leave', ('start',), 'past', 'unspecified', 'unknown'),
    ('finished leaving', ('finish',), 'past', 'unspecified', 'yes'),
    ('stopped leaving', ('stop',), 'past', 'unspecified', 'unknown'),
    ("can't leave", ('not', 'can'), 'modal', 'unspecified', 'unknown'),
    ('may not leave', ('may', 'not'), 'modal', 'unspecified', 'unknown'),
    ('must not leave', ('must', 'not'), 'modal', 'unspecified', 'unknown'),
    ('should leave', ('should',), 'modal', 'unspecified', 'unknown'),
    ("shouldn't leave", ('should', 'not'), 'modal', 'unspecified', 'unknown'),
    ('should not leave', ('should', 'not'), 'modal', 'unspecified', 'unknown'),
    ('should have left', ('should', 'have'), 'modal', 'unspecified', 'unknown'),
    ('might have managed to leave', ('might', 'have', 'manage'), 'modal', 'unspecified', 'unknown'),
    ('must have left', ('must', 'have'), 'modal', 'unspecified', 'unknown'),
    ('had to leave', ('have_to',), 'past', 'unspecified', 'unknown'),
    ('did not have to leave', ('not', 'have_to'), 'past', 'unspecified', 'unknown'),
    ('has left', ('have',), 'present', 'unspecified', 'yes'),
    ('had wanted to leave', ('have', 'want'), 'past', 'unspecified', 'unknown'),
    ("'s leaving", (), 'present', 'ongoing', 'unknown'),
    ("'s left", ('have',), 'present', 'unspecified', 'yes'),
    ('was leaving', (), 'past', 'ongoing', 'unknown'),
    ('was not leaving', ('not',), 'past', 'ongoing', 'unknown'),
    ('was going to leave', ('be_going_to',), 'past', 'unspecified', 'unknown'),
    ('is able to leave', ('be_able_to',), 'present', 'unspecified', 'unknown'),
    ('will not leave', ('not',), 'future', 'unspecified', 'unknown'),
    ('did not leave', ('not',), 'past', 'reported', 'no'),
])
def test_scope_and_evidence_survive_text_commands(text, operators, tense, phase, evidence):
    scope, reason = parse_scope(text, 'leave', paradigms())
    assert reason is None
    assert (scope.operators, scope.tense, scope.phase) == (operators, tense, phase)
    event = Event('leave', (), mined=True, subject=MinedSubject('Tom', 'PERSON'), **asdict(scope))
    assert occurrence_evidence(event) == evidence
    assert completion_evidence(event) == 'unknown'
    kind, fields = _event_payload(event)
    assert parse_commands(kind + ' ' + _fields(fields))[0].event == event
    assert event_statement(event, perspective='user').startswith('Tom ')
    answer, action, _ = answer_query(Memory(events=(EventClaim('e1', event, ()),)), Query('mined_occurrence', event=event))
    assert (action == 'unknown') == (evidence == 'unknown')
    if evidence != 'unknown':
        assert answer.startswith('Yes' if evidence == 'yes' else 'No')


@pytest.mark.parametrize('text', ['stopped to leave', 'tried leaving', 'would leave',
                                 'must can leave', 'wanted Tom to leave', 'has to left',
                                 'was left', 'wanted probably to leave', 'never left',
                                 'may have been left', 'want want leave',
                                 'wanted to try to plan to manage to fail to leave'])
def test_unlicensed_or_ambiguous_grammar_stays_quoted(text):
    assert parse_scope(text, 'leave', paradigms())[0] is None


def chain_record(*, embedded_subject=False, operator='want', surface='wanted', marker=' to '):
    """Small compiler-format dependency record, with no parser dependency."""
    slots = {'n0': {'pos': 'PROPN', 'kind': 'nominal', 'tag': 'NNP', 'dependency': 'nsubj', 'head_slot': 'v0'},
             'v0': {'pos': 'VERB', 'kind': 'verb', 'lemma': operator, 'tag': 'VBD', 'dependency': 'ROOT', 'head_slot': 'v0'},
             'v1': {'pos': 'VERB', 'kind': 'verb', 'lemma': 'leave', 'tag': 'VB', 'dependency': 'xcomp', 'head_slot': 'v0'}}
    defaults = {'n0': 'Tom', 'v0': surface, 'v1': 'leave'}
    parts = [{'slot': 'n0'}, {'literal': ' '}, {'slot': 'v0'}, {'literal': marker}, {'slot': 'v1'}, {'literal': '.'}]
    if embedded_subject:
        slots['n1'] = {'pos': 'PROPN', 'kind': 'nominal', 'tag': 'NNP', 'dependency': 'nsubj', 'head_slot': 'v1'}
        defaults['n1'] = 'Mary'
        parts[3:4] = [{'literal': ' '}, {'slot': 'n1'}, {'literal': marker}]
    return {'slots': slots, 'defaults': defaults, 'parts': parts, 'attested_bindings': [defaults],
            'predicate_groups': [{'head_slot': key, 'lemma': slot['lemma'], 'constructions': []}
                                 for key, slot in slots.items() if slot['kind'] == 'verb']}


def test_dependency_control_and_negation_retain_shared_subject():
    row = chain_record(marker=' not to ')
    spec, reason = active_phrase(row, 'v1', 'leave', paradigms(), {'Tom', 'Mary'})
    assert reason is None
    scope = source_scope(spec, row['defaults'], paradigms())
    assert scope.operators == ('want', 'not')
    assert spec['subject_slot'] == 'n0'
    assert spec['tail_parts'] == [{'literal': '.'}]
    row = chain_record(embedded_subject=True)
    assert active_phrase(row, 'v1', 'leave', paradigms(), {'Tom', 'Mary'})[1] == 'unresolved_control_subject'
    row = chain_record(operator='promise', surface='promised')
    assert active_phrase(row, 'v1', 'leave', paradigms(), {'Tom'})[1] == 'unreviewed_control_operator'


def test_every_attested_scope_is_validated():
    row = chain_record()
    row['attested_bindings'].append({**row['defaults'], 'v0': 'wants probably'})
    assert active_phrase(row, 'v1', 'leave', paradigms(), {'Tom'})[0] is None


def test_inner_and_outer_negation_are_distinct_memory_keys():
    base = Event('leave', (), mined=True, subject=MinedSubject('Tom', 'PERSON'), operators=('want',), tense='past')
    outer, inner = replace(base, operators=('not', 'want')), replace(base, operators=('want', 'not'))
    assert outer.key == base.key and inner.key != outer.key
    assert event_statement(outer, perspective='user') == 'Tom did not want to leave today.'
    assert event_statement(inner, perspective='user') == 'Tom wanted not to leave today.'


def test_plural_progressive_and_conflicting_evidence():
    event = Event('leave', (), mined=True, subject=MinedSubject('the workers', 'PERSON', 'plural'), phase='ongoing', tense='past')
    assert event_statement(event, perspective='user') == 'The workers were leaving today.'
    denial = replace(event, phase='reported', operators=('not',))
    memory = Memory(events=(EventClaim('e1', replace(event, operators=('manage',), phase='unspecified'), ()), EventClaim('e2', denial, ())))
    assert answer_query(memory, Query('mined_occurrence', event=event))[1] == 'unknown'


@pytest.mark.parametrize('user_only', [False, True])
@pytest.mark.parametrize('operator, surface, marker', [('want', 'wanted', ' not to '),
                                                     ('manage', 'managed', ' to '), ('fail', 'failed', ' to ')])
def test_scoped_source_bank_to_all_three_export_views(small_bank, user_only, operator, surface, marker):
    row = chain_record(operator=operator, surface=surface, marker=marker)
    row.update(verbs=[operator, 'leave'], constructions=[], attributes=[], adjective_lemmas=[],
               fillers={k: [v] for k, v in row['defaults'].items()})
    row['template_id'] = fingerprint(row)
    path = small_bank.banks[0].path
    file = path / 'templates' / (row['template_id'][:2] + '.jsonl.gz')
    with gzip.open(file, 'at', encoding='utf-8') as stream:
        stream.write(json.dumps(row) + '\n')
    with gzip.open(path / 'verbs.jsonl.gz', 'at', encoding='utf-8') as stream:
        stream.write(json.dumps({'lemma': 'leave', 'forms': {'leave': 1}}) + '\n')
    with small_bank.dictionary.open('a', encoding='utf-8') as stream:
        stream.write('leave\tleaves\tleaving\tleft\tleft\n')
    bank = DialogueMinedBank(small_bank.root, cache=False)
    assert len(bank.entries['leave']['scoped_references']) == 1
    episodes = list(DialogueGenerator(user_assistant_only=user_only, mined_bank=bank).generate('mined_predicates', 12, split='train'))
    selected = [e for e in episodes if e.mined_source['raw_lemma'] == 'leave']
    assert len(selected) == 3
    assert {e.condition.rsplit(':', 1)[-1] for e in selected} == {'yes', 'no', 'unknown'}
    for episode in selected:
        assert episode.mined_source['scope_origin'] == 'source'
        initial = episode.transitions[1].delta.event_add[0].event
        assert initial.operators == (operator, 'not') if marker == ' not to ' else initial.operators == (operator,)
        assert initial.subject == MinedSubject('Tom', 'PERSON')
        expected = episode.condition.rsplit(':', 1)[-1]
        assert episode.expected_action == ('unknown' if expected == 'unknown' else 'answer')
        if expected != 'unknown':
            assert episode.answer.startswith('Yes' if expected == 'yes' else 'No')
        episode.validate()
        for transition in episode.transitions:
            target = extraction_row(episode, transition)['messages'][-1]['content']
            assert decode_update(target, transition.before, transition.turn, transition.previous_assistant) == transition.delta


def test_embedded_phrasal_verb_keeps_separated_particle():
    row = chain_record()
    row['slots']['v1']['lemma'] = 'take'
    row['defaults'].update(v1='take', n1='coat')
    row['slots']['n1'] = {'kind': 'nominal', 'pos': 'NOUN', 'tag': 'NN', 'dependency': 'dobj', 'head_slot': 'v1'}
    row['parts'][-1:] = [{'literal': ' the '}, {'slot': 'n1'}, {'literal': ' off.'}]
    row['predicate_groups'][-1].update(lemma='take off', constructions=[{'kind': 'particle', 'lemma': 'take off'}])
    spec, reason = active_phrase(row, 'v1', 'take', paradigms('take'), {'Tom'})
    assert reason is None and spec['predicate'] == 'take off'
    assert source_scope(spec, row['defaults'], paradigms('take')).operators == ('want',)
