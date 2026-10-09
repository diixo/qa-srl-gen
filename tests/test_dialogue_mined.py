"""Mined dialogue morphology, scope, text roundtrips and complete inventory audit."""
from dataclasses import asdict, replace
import gzip
import json

import pytest

from semantic_corpus.dialogue_generator.events import Event, EventArgument, event_statement, completion_evidence
from semantic_corpus.dialogue_generator.generator import DialogueGenerator
from semantic_corpus.dialogue_generator.mined_bank import DialogueMinedBank, normalize_verb
from semantic_corpus.dialogue_generator.models import EventClaim, Memory, Query, answer_query
from semantic_corpus.dialogue_generator.extraction_text import decode_update, parse_commands, parse_prompt
from semantic_corpus.dialogue_generator.export import extraction_row
from semantic_corpus.dialogue_generator.export import build_dialogue_dataset
from semantic_corpus.dialogue_generator.observed_clauses import ObservedClause
from semantic_corpus.qasrl_core.inflections import load_inflections
from semantic_corpus.dialogue_generator.mined_realization import active_phrase, lexicon
from semantic_corpus.template_bank import FORMAT, fingerprint


def record(words, head, lemma, subject=0, dependency='ROOT'):
    parts, slots, defaults = [], {}, {}
    for i, word in enumerate(words):
        if i:
            parts.append({'literal': ' '})
        if i in (subject, head):
            key = f'v{i}' if i == head else f'n{i}'
            parts.append({'slot': key})
            defaults[key] = word
            slots[key] = {'kind': 'verb' if i == head else 'nominal', 'pos': 'VERB' if i == head else 'PRON',
                          'tag': 'VBD' if i == head else 'PRP', 'lemma': lemma if i == head else word.lower(),
                          'dependency': dependency if i == head else 'nsubj', 'head_slot': f'v{head}'}
        else:
            parts.append({'literal': word})
    parts.append({'literal': '.'})
    result = {'parts': parts, 'slots': slots, 'defaults': defaults, 'verbs': [lemma], 'constructions': [],
              'predicate_groups': [{'lemma': lemma, 'head_slot': f'v{head}', 'constructions': []}],
              'attested_bindings': [defaults], 'fillers': {k: [v] for k, v in defaults.items()},
              'attributes': [], 'adjective_lemmas': []}
    result['template_id'] = fingerprint(result)
    return result


@pytest.fixture
def small_bank(tmp_path):
    root = tmp_path / 'repo'
    path = root / 'data/eng-base-templates'
    dictionary = root / 'data/wiktionary/en_verb_inflections.txt'
    dictionary.parent.mkdir(parents=True)
    dictionary.write_text('sleep\tsleeps\tsleeping\tslept\tslept\n'
                          'come\tcomes\tcoming\tcame\tcome\n'
                          'rain\trains\training\trained\trained\n', encoding='utf-8')
    rows = [record(['I', 'slept'], 1, 'sleep'), record(['I', 'came'], 1, 'came'),
            record(['It', 'rained'], 1, 'rain'), record(['I', 'have'], 1, 'have')]
    def write(file, data):
        file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(file, 'wt', encoding='utf-8') as stream:
            for item in data:
                stream.write(json.dumps(item) + '\n')
    for prefix in {r['template_id'][:2] for r in rows}:
        write(path / 'templates' / (prefix + '.jsonl.gz'), [r for r in rows if r['template_id'].startswith(prefix)])
    write(path / 'verbs.jsonl.gz', [{'lemma': lemma, 'forms': {form: 1}} for lemma, form in
                                   [('sleep', 'slept'), ('came', 'came'), ('rain', 'rained'), ('have', 'have')]])
    write(path / 'verb_alternatives.jsonl.gz', [{'lemma': 'candy', 'forms': {'candy': 1}}])
    (path / 'manifest.json').write_text(json.dumps({'format': FORMAT, 'status': 'complete',
                                                  'counts': {'templates': len(rows), 'verb_lemmas': 4}}), encoding='utf-8')
    return DialogueMinedBank(root)


def test_all_inventory_heads_are_accounted_for_without_promoting_alternatives(small_bank):
    bank = small_bank
    assert set(bank.lemmas) == {'sleep', 'came', 'rain'}
    assert bank.entries['came']['canonical'] == 'come'
    assert bank.report['dispositions'] == {'compositional': 2, 'attested_only': 1}
    assert bank.report['alternative_only_lemmas'] == ['candy']
    assert not bank.report['unavailable']
    before = {str(p): p.read_bytes() for p in bank.banks[0].path.rglob('*') if p.is_file()}
    reopened = DialogueMinedBank(bank.root)
    assert reopened.identity == bank.identity and reopened.entries == bank.entries
    assert before == {str(p): p.read_bytes() for p in bank.banks[0].path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('user_only', [False, True])
def test_full_mined_cycle_replays_all_three_outcomes_and_text_commands(small_bank, user_only):
    generator = DialogueGenerator(user_assistant_only=user_only, mined_bank=small_bank)
    episodes = list(generator.generate('mined_predicates', 9, split='train'))
    assert {e.mined_source['raw_lemma'] for e in episodes} == set(small_bank.lemmas)
    for lemma in small_bank.lemmas:
        group = [e for e in episodes if e.mined_source['raw_lemma'] == lemma]
        assert {e.condition.rsplit(':', 1)[-1] for e in group} == {'yes', 'no', 'unknown'}
    for episode in episodes:
        episode.validate()
        for transition in episode.transitions:
            row = extraction_row(episode, transition)
            prompt, target = [m['content'] for m in row['messages'][1:]]
            visible, user, previous = parse_prompt(prompt)
            delta = decode_update(target, transition.before, transition.turn, transition.previous_assistant)
            assert delta == transition.delta
            decode_update(target, visible, user, previous)
        assert completion_evidence(episode.query.event) == 'unknown'


def test_dictionary_forms_and_particles_stay_separate_from_completion():
    event = Event('sell', (EventArgument('agent', 'user', 'PERSON'),
                          EventArgument('complement', 'a ticket', 'UNRESOLVED')),
                  tense='past', phase='reported', mined=True)
    assert event_statement(event, perspective='user') == 'I sold a ticket today.'
    assert event_statement(replace(event, phase='unspecified', operators=('may', 'have'), tense='modal'),
                           perspective='user') == 'I may have sold a ticket today.'
    assert completion_evidence(event) == 'unknown'
    assert answer_query(Memory(events=(EventClaim('e1', event, ()),)), Query('mined_occurrence', event=event))[0].startswith('Yes,')
    phrasal = replace(event, predicate='take off', arguments=(event.arguments[0], EventArgument('complement', 'the coat off', 'UNRESOLVED')))
    assert event_statement(phrasal, perspective='user') == 'I took the coat off today.'
    with pytest.raises(ValueError):
        replace(event, predicate='go')
    with pytest.raises(ValueError):
        replace(event, phase='completed')


def test_quoted_modality_is_never_confirmation_of_embedded_completion():
    event = Event('@clause', (), observation=ObservedClause('Anna might have bought a ticket.', 'reported'))
    assert completion_evidence(event) == 'unknown'
    assert answer_query(Memory(events=(EventClaim('e1', event, ()),)), Query('observed_report', event=event))[0].startswith('Yes,')
    assert answer_query(Memory(events=(EventClaim('e1', event, ()),)), Query('event_completion', event=event))[1] == 'unknown'
    with pytest.raises(ValueError):
        parse_commands('QUOTE invented | text')


def test_full_coverage_quota_does_not_inflate_other_tasks(small_bank, tmp_path):
    manifest = build_dialogue_dataset(tmp_path / 'all', train=1, validation=1, test=1,
                                     user_assistant_only=True, mined_bank=small_bank, mined_train=0)
    assert manifest['splits']['train']['by_task']['mined_predicates'] == 9
    assert manifest['splits']['train']['by_task']['recall'] == 1
    assert manifest['mined_bank']['selected_lemmas']['train'] == 3
    report = json.loads((tmp_path / 'all/mined-coverage.json').read_text())
    assert report['train']['unselected_lemmas'] == []


def test_quoted_intention_survives_an_independent_report():
    event = Event('@clause', (), observation=ObservedClause('It might rain.', 'desired'))
    report = replace(event, observation=replace(event.observation, status='reported'))
    assert report.key != event.key


def test_reverse_inflections_do_not_vote_by_number_of_syncretic_forms(tmp_path):
    path = tmp_path / 'forms.tsv'
    path.write_text('see\tsees\tseeing\tsaw\tseen\n'
                    'saw\tsaws\tsawing\tsawed\tsawed\n', encoding='utf-8')
    info = normalize_verb({'lemma': 'parser_typo', 'forms': {'saw': 10}}, load_inflections(path))
    assert info['inflection_source'] == 'observed_only'
    assert not info['paradigms']


@pytest.mark.parametrize('words, head, subject, reason', [
    (['Down', 'came', 'Tom'], 1, 2, 'inverted_subject'),
    (['I', 'came', 'three', 'nights', 'a', 'week'], 1, 0, 'time_or_adverb_scope'),
    (['I', 'came', 'with', 'anybody'], 1, 0, 'polarity_sensitive_complement'),
    (['I', 'came', 'with', 'it'], 1, 0, 'unresolved_reference'),
])
def test_scope_sensitive_complements_remain_quotations(words, head, subject, reason):
    spec, rejected = active_phrase(record(words, head, 'come', subject=subject), f'v{head}',
                                  'come', [asdict(p) for p in lexicon().paradigms('come')], {'Tom'})
    assert spec is None and rejected == reason


def test_compiled_particle_group_preserves_compound_predicate():
    row = record(['I', 'took', 'the', 'coat', 'off'], 1, 'take')
    # The compiler's group lemma is the whole phrasal verb, the slot is its head.
    row['predicate_groups'][0].update(lemma='take off', constructions=[
        {'lemma': 'take off', 'kind': 'particle', 'separated': True}])
    # A lexical noun is a slot in a real compiled template.
    row['parts'][6] = {'slot': 'n3'}
    row['slots']['n3'] = {'kind': 'nominal', 'pos': 'NOUN', 'dependency': 'dobj', 'head_slot': 'v1'}
    row['defaults']['n3'] = 'coat'
    spec, reason = active_phrase(row, 'v1', 'take', [asdict(p) for p in lexicon().paradigms('take')], set())
    assert reason is None and spec['predicate'] == 'take off'
