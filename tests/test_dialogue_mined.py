"""Mined dialogue morphology, scope, text roundtrips and complete inventory audit."""
from dataclasses import asdict, replace
import gzip
import json

import pytest

from semantic_corpus.dialogue_generator.events import Event, EventArgument, event_statement, event_question, completion_evidence
from semantic_corpus.dialogue_generator.mined_subjects import MinedSubject, realize_subject, POSSIBILITY_PROFILES
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
    path = root / 'data/mined-templates'
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
    assert bank.report['dispositions'] == {'compositional': 3}
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
        if episode.mined_source['raw_lemma'] == 'rain':
            assert episode.query.event.subject == MinedSubject('it', 'IMPERSONAL')
            assert episode.turns[-1].text == 'Did it rain today?'
            assert episode.transitions[1].delta.event_add[0].event.operators in POSSIBILITY_PROFILES


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


@pytest.mark.parametrize('subject, positive, negative, question, memory', [
    (MinedSubject('user', 'PERSON'), 'I slept', 'I did not sleep', 'Did I sleep', 'You slept'),
    (MinedSubject('assistant', 'PERSON'), 'You slept', 'You did not sleep', 'Did you sleep', 'I slept'),
    (MinedSubject('Tom', 'PERSON'), 'Tom slept', 'Tom did not sleep', 'Did Tom sleep', 'Tom slept'),
    (MinedSubject('my dog', 'ANIMAL'), 'My dog slept', 'My dog did not sleep', 'Did my dog sleep', 'Your dog slept'),
    (MinedSubject('the dogs', 'ANIMAL', 'plural'), 'The dogs slept', 'The dogs did not sleep', 'Did the dogs sleep', 'The dogs slept'),
])
def test_subject_is_preserved_in_reports_questions_and_memory(subject, positive, negative, question, memory):
    event = Event('sleep', (), tense='past', phase='reported', mined=True, subject=subject)
    assert event_statement(event, perspective='user') == positive + ' today.'
    assert event_statement(replace(event, operators=('not',)), perspective='user') == negative + ' today.'
    assert event_statement(event) == memory + ' today.'
    assert event_question(event) == question + ' today?'
    assert completion_evidence(event) == 'unknown'


@pytest.mark.parametrize('subject, want, able, going, past, let', [
    (MinedSubject('Tom', 'PERSON'), 'Tom wants', 'Tom is able', 'Tom is going', 'Tom was able', 'You let Tom'),
    (MinedSubject('the workers', 'PERSON', 'plural'), 'The workers want', 'The workers are able', 'The workers are going', 'The workers were able', 'You let the workers'),
    (MinedSubject('assistant', 'PERSON'), 'You want', 'You are able', 'You are going', 'You were able', 'I let you'),
])
def test_named_and_plural_subject_agreement(subject, want, able, going, past, let):
    base = Event('sleep', (), operators=('want',), mined=True, subject=subject)
    assert event_statement(base, perspective='user') == want + ' to sleep today.'
    assert event_statement(replace(base, operators=('be_able_to',)), perspective='user') == able + ' to sleep today.'
    assert event_statement(replace(base, operators=('be_going_to',)), perspective='user') == going + ' to sleep today.'
    assert event_statement(replace(base, operators=('be_able_to',), tense='past'), perspective='user') == past + ' to sleep today.'
    permitter = EventArgument('permitter', 'user' if subject.text == 'assistant' else 'assistant', 'PERSON')
    assert event_statement(replace(base, operators=('let',), permitter=permitter), perspective='user') == let + ' sleep today.'
    with pytest.raises(ValueError):
        replace(base, operators=('let_us',))


def test_subjects_do_not_share_event_memory_and_commands_roundtrip():
    from semantic_corpus.dialogue_generator.extraction_text import _event_payload, _fields
    user = Event('sleep', (), tense='past', phase='reported', mined=True, subject=MinedSubject('user', 'PERSON'))
    tom = replace(user, subject=MinedSubject('Tom', 'PERSON'))
    assert user.identity != tom.identity and user.key != tom.key
    memory = Memory(events=(EventClaim('e1', tom, ()),))
    assert answer_query(memory, Query('mined_occurrence', event=user))[1] == 'unknown'
    assert answer_query(memory, Query('mined_occurrence', event=tom))[0].startswith('Yes,')
    for subject in (MinedSubject('it', 'IMPERSONAL'), MinedSubject('the engines', 'PHYSICAL_OBJECT', 'plural'),
                    MinedSubject('your dog', 'ANIMAL'), MinedSubject('Tom', 'PERSON'), MinedSubject('assistant', 'PERSON')):
        event = replace(user, predicate='rain' if subject.label == 'IMPERSONAL' else 'sleep', subject=subject)
        kind, fields = _event_payload(event)
        for prefix in ('', 'REMOVE_'):
            command = parse_commands(prefix + kind + ' ' + _fields(fields))[0]
            assert command.event == event
    legacy = parse_commands('MINED sleep | today | none | reported | past | agent=PERSON:user')[0].event
    assert legacy.subject is None and event_statement(legacy, perspective='user') == 'I slept today.'
    assert legacy.key == user.key


@pytest.mark.parametrize('fields', [
    'subject=PERSON:Tom', 'number=plural | agent=PERSON:user',
    'subject=PERSON:Tom | number=dual', 'subject=PERSON:Tom | number=singular | number=plural',
    'subject=PERSON:Tom | number=singular | subject=PERSON:Ann',
    'subject=PERSON:Tom | number=singular | agent=PERSON:user',
    'subject=IMPERSONAL:it | number=singular', 'subject=ACTION:the engines | number=plural',
])
def test_malformed_subject_commands_are_rejected(fields):
    with pytest.raises(ValueError):
        parse_commands('MINED sleep | today | none | reported | past | ' + fields)


@pytest.mark.parametrize('subject', [MinedSubject('it', 'IMPERSONAL'), MinedSubject('the engine', 'PHYSICAL_OBJECT'),
                                    MinedSubject('the dog', 'ANIMAL'), MinedSubject('Zorg', 'UNRESOLVED')])
def test_nonperson_subjects_do_not_receive_invented_intentions(subject):
    event = Event('rain' if subject.label == 'IMPERSONAL' else 'run', (),
                  tense='past', phase='reported', mined=True, subject=subject)
    for profile in POSSIBILITY_PROFILES:
        candidate = replace(event, phase='unspecified', tense='modal', operators=profile)
        assert completion_evidence(candidate) == 'unknown'
    for profile in (('want',), ('plan',), ('let_us',), ('be_going_to',)):
        with pytest.raises(ValueError):
            replace(event, phase='unspecified', tense='present', operators=profile)


def test_whole_subject_np_survives_mining_with_possessives_and_number():
    from semantic_corpus.semantic_generator.substitutions import default_pool
    row = record(['My', 'dogs', 'slept'], 2, 'sleep', subject=1)
    row['slots']['n1'].update(pos='NOUN', tag='NNS', morph='Number=Plur')
    row['parts'][0] = {'slot': 'n0'}
    row['slots']['n0'] = {'kind': 'nominal', 'pos': 'PRON', 'tag': 'PRP$', 'dependency': 'poss', 'head_slot': 'n1'}
    row['defaults']['n0'] = 'My'
    spec, reason = active_phrase(row, 'v2', 'sleep', [asdict(p) for p in lexicon().paradigms('sleep')], set())
    assert reason is None
    subject, reason = realize_subject(row, spec, row['defaults'], set(), default_pool())
    assert reason is None and subject.text == 'my dogs' and subject.number == 'plural'
    assert subject.surface('memory', initial=True) == 'Your dogs'
    # A new relative clause cannot be silently dropped from the subject.
    row['slots']['v3'] = {'dependency': 'relcl', 'head_slot': 'n1', 'kind': 'verb'}
    assert active_phrase(row, 'v2', 'sleep', [asdict(p) for p in lexicon().paradigms('sleep')], set())[1] == 'embedded_or_coordinated_predicates'


@pytest.mark.parametrize('word', ['He', 'She', 'They', 'We', 'It'])
def test_unresolved_pronouns_do_not_become_the_user(word):
    row = record([word, 'slept'], 1, 'sleep')
    assert active_phrase(row, 'v1', 'sleep', [asdict(p) for p in lexicon().paradigms('sleep')], set())[1] == 'unresolved_subject_reference'


@pytest.mark.parametrize('auxiliary, word, tag', [('can', 'sleep', 'VB'), ("can't", 'sleep', 'VB'),
                                               ("'ve", 'slept', 'VBN'), ('may have', 'slept', 'VBN'),
                                               ('does not', 'sleep', 'VB')])
def test_source_auxiliaries_require_a_recognized_scope(auxiliary, word, tag):
    row = record(['I', auxiliary, word], 2, 'sleep')
    row['slots']['v2']['tag'] = tag
    spec, reason = active_phrase(row, 'v2', 'sleep', [asdict(p) for p in lexicon().paradigms('sleep')], set())
    assert reason is None
    assert ''.join(p.get('literal', row['defaults'].get(p.get('slot'), '')) for p in spec['subject_parts']) == 'I'
    for adverb in ('obviously', 'maybe'):
        row['parts'][3]['literal'] = ' ' + adverb + ' '
        assert active_phrase(row, 'v2', 'sleep', [asdict(p) for p in lexicon().paradigms('sleep')], set())[1] == 'unsupported_source_scope'


def test_compiled_auxiliary_slot_is_not_confused_with_a_second_subject():
    row = record(['Tom', 'is', 'sleeping'], 2, 'sleep')
    row['slots']['n0'].update(pos='PROPN', tag='NNP')
    row['slots']['v2']['tag'] = 'VBG'
    row['parts'][2] = {'slot': 'v1'}
    row['slots']['v1'] = {'kind': 'verb', 'pos': 'AUX', 'dependency': 'aux', 'head_slot': 'v2', 'lemma': 'be'}
    row['defaults']['v1'] = 'is'
    spec, reason = active_phrase(row, 'v2', 'sleep', [asdict(p) for p in lexicon().paradigms('sleep')], {'Tom'})
    assert reason is None
    assert realize_subject(row, spec, row['defaults'], {'Tom'})[0] == MinedSubject('Tom', 'PERSON')
    row['slots']['v1']['pos'] = 'VERB'
    assert active_phrase(row, 'v2', 'sleep', [asdict(p) for p in lexicon().paradigms('sleep')], {'Tom'})[1] == 'embedded_or_coordinated_predicates'


def test_vocatives_cannot_reassign_the_addressee_or_join_the_verb_complement():
    row = record(['You', 'slept', ',', 'Tom'], 1, 'sleep')
    assert active_phrase(row, 'v1', 'sleep', [asdict(p) for p in lexicon().paradigms('sleep')], {'Tom'})[1] == 'vocative_or_detached_complement'


def test_common_noun_case_survives_sentence_to_question_conversion():
    row = record(['Dogs', 'slept'], 1, 'sleep')
    row['slots']['n0'].update(pos='NOUN', tag='NNS')
    spec, reason = active_phrase(row, 'v1', 'sleep', [asdict(p) for p in lexicon().paradigms('sleep')], set())
    assert reason is None
    subject, reason = realize_subject(row, spec, row['defaults'], set())
    assert reason is None and subject.text == 'dogs'
    event = Event('sleep', (), phase='reported', tense='past', mined=True, subject=subject)
    assert event_statement(event, perspective='user') == 'Dogs slept today.'
    assert event_question(event) == 'Did dogs sleep today?'


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
