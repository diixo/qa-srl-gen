"""Parser errors and inflection ambiguity must not become asserted events."""
from copy import deepcopy
import gzip
import json
import random
from pathlib import Path
import os
import subprocess
import sys

import pytest

from semantic_corpus.dialogue_generator.events import Event, event_statement
from semantic_corpus.dialogue_generator.mined_bank import DialogueMinedBank, normalize_verb
from semantic_corpus.dialogue_generator.mined_realization import active_phrase, lexicon, mined_lexeme, source_scope, refresh_resources
from semantic_corpus.dialogue_generator import mined_bank, mined_review
from semantic_corpus.dialogue_generator.mined_review import review_slot, review_template
from semantic_corpus.dialogue_generator.mined_subjects import MinedSubject
from semantic_corpus.dialogue_generator.export import build_dialogue_dataset
from semantic_corpus.qasrl_core.inflections import load_inflections
from semantic_corpus.template_bank import fingerprint
from test_dialogue_mined import record, small_bank
from test_dialogue_mined_roles import coordinated


def lexical_record(surface, lemma, tag='VBD'):
    row = record(['I', surface], 1, lemma)
    row['slots']['v1']['tag'] = tag
    return row


def test_inventory_recovery_does_not_drop_unmatched_forms_or_case_counts():
    info = normalize_verb({'lemma': 're', 'forms': {'re': 21, 'ring': 1}}, lexicon())
    assert info['canonical'] == 're' and info['inflection_source'] == 'observed_only'
    info = normalize_verb({'lemma': 'came', 'forms': {'Came': 1, 'came': 2}}, lexicon())
    assert info['canonical'] == 'come' and info['forms'] == {'came': 3}


@pytest.mark.parametrize('lemma,surface,canonical', [('came', 'came', 'come'), ('swam', 'swam', 'swim'),
                                                   ('forgot', 'forgot', 'forget')])
def test_finite_alias_and_tag_repair_preserves_text_and_archive(lemma, surface, canonical):
    row = lexical_record(surface, lemma, 'VBP')
    original = deepcopy(row)
    projected, reviews = review_template(row, lexicon())
    decision = reviews['v1']
    assert decision['canonical'] == canonical and decision['tag'] == 'VBD'
    assert decision['reason'] == 'repaired_finite_tag'
    assert projected['slots']['v1']['morph'] == 'Tense=Past|VerbForm=Fin'
    assert projected['parts'] == row['parts'] and projected['defaults'] == row['defaults']
    assert projected['predicate_groups'][0]['lemma'] == canonical
    spec, reason = active_phrase(row, 'v1', lemma, (), set())
    assert reason is None and spec['predicate'] == canonical
    assert source_scope(spec, row['defaults'], spec['source_paradigms']).tense == 'past'
    assert row == original


@pytest.mark.parametrize('change', ['no_subject', 'auxiliary', 'embedded', 'participle', 'mixed_bindings'])
def test_tag_repair_requires_one_finite_reading_for_every_binding(change):
    row = lexical_record('came', 'came', 'VBP')
    if change == 'no_subject':
        row['slots']['n0']['dependency'] = 'dobj'
    elif change == 'auxiliary':
        row['slots']['v2'] = {'kind': 'verb', 'pos': 'AUX', 'tag': 'MD', 'lemma': 'can', 'dependency': 'aux', 'head_slot': 'v1'}
    elif change == 'embedded':
        row['slots']['v1']['dependency'] = 'xcomp'
    elif change == 'participle':
        row['slots']['v1']['tag'] = 'VBN'
    else:
        row['attested_bindings'].append({**row['defaults'], 'v1': 'unknownword'})
    assert review_slot(row, 'v1', lexicon())['status'] == 'quotation_only'


@pytest.mark.parametrize('surface,lemma,tag,reason', [
    ('-', '-', 'VB', 'nonlexical_token'), ('\ufffdve', '\ufffdve', 'VB', 'nonlexical_token'),
    ('catnip', 'catnip', 'VB', 'missing_dictionary_paradigm'),
    ('found', 'found', 'VBP', 'finite_homograph'),
    ('saw', 'saw', 'VBP', 'finite_homograph'),
    ('lying', 'lie', 'VBG', 'sense_ambiguous_inflection'),
    ('lies', 'lie', 'VBZ', 'sense_ambiguous_inflection'),
])
def test_noise_and_sense_ambiguity_stay_quotes(surface, lemma, tag, reason):
    row = lexical_record(surface, lemma, tag)
    assert active_phrase(row, 'v1', lemma, (), set())[1] == reason


def test_lying_cat_never_conjugates_to_lied():
    row = record(['I', 'am', 'lying'], 2, 'lie')
    row['slots']['v2']['tag'] = 'VBG'
    assert active_phrase(row, 'v2', 'lie', (), set())[1] == 'sense_ambiguous_inflection'
    row = lexical_record('lied', 'lie')
    assert active_phrase(row, 'v1', 'lie', (), set())[1] is None


def test_auxiliary_and_lexical_enjoyed_are_distinguished():
    row = lexical_record('enjoyed', 'enjoyed')
    row['slots']['v1'].update(pos='AUX', dependency='aux')
    assert review_slot(row, 'v1', lexicon())['reason'] == 'auxiliary_or_modal'
    row['slots']['v1'].update(pos='VERB', dependency='ROOT')
    assert review_slot(row, 'v1', lexicon())['canonical'] == 'enjoy'


def test_tagged_homographs_keep_lexical_found_separate_from_find():
    for surface, lemma, tag, expected in [('founded', 'found', 'VBD', 'found'),
                                         ('found', 'find', 'VBD', 'find'), ('sawed', 'saw', 'VBD', 'saw')]:
        decision = review_slot(lexical_record(surface, lemma, tag), 'v1', lexicon())
        assert decision['status'] != 'quotation_only' and decision['canonical'] == expected
    # The scrape also contains foind/found. A broken lemma cannot be repaired
    # by simply choosing the most frequent English head.
    assert review_slot(lexical_record('found', 'found'), 'v1', lexicon())['reason'] == 'ambiguous_lemma'


def test_ambiguous_reverse_lookup_and_competing_paradigms_need_review(tmp_path):
    path = tmp_path / 'forms.txt'
    path.write_text('alpha\talphas\talphing\tshared\tshared\n'
                    'beta\tbetas\tbeting\tshared\tshared\n'
                    'other\tothers\tothering\tothered\tothered\n'
                    'other\tothers\tothering\totherd\totherd\n', encoding='utf-8')
    dictionary = load_inflections(path)
    assert review_slot(lexical_record('shared', 'typo'), 'v1', dictionary)['reason'] == 'ambiguous_lemma'
    assert review_slot(lexical_record('othered', 'other'), 'v1', dictionary)['reason'] == 'ambiguous_generation_paradigm'


def test_reviewed_generation_preference_is_used_in_every_realization():
    assert mined_lexeme('awaken').past == 'awakened'
    assert mined_lexeme('plead').past == 'pleaded'
    for lemma, surface in [('awaken', 'awakened'), ('plead', 'pled'), ('plead', 'pleaded')]:
        row = lexical_record(surface, lemma)
        assert active_phrase(row, 'v1', lemma, (), set())[1] is None
    event = Event('awaken', (), mined=True, subject=MinedSubject('user', 'PERSON'), tense='past', phase='reported')
    assert event_statement(event, perspective='user') == 'I awakened today.'


def test_alias_particle_is_repaired_without_reordering():
    row = record(['I', 'swam', 'away'], 1, 'swam')
    row['slots']['v1']['tag'] = 'VBP'
    row['predicate_groups'][0].update(lemma='swam away', constructions=[{'lemma': 'swam away', 'kind': 'particle', 'markers': []}])
    spec, reason = active_phrase(row, 'v1', 'swam', (), set())
    assert reason is None and spec['predicate'] == 'swim away'
    assert row['predicate_groups'][0]['lemma'] == 'swam away'


def test_companion_alias_repair_and_ambiguity_use_the_same_checks():
    row = coordinated()
    row['slots']['v2'].update(lemma='swam', tag='VBP')
    row['defaults']['v2'] = 'swam'
    row['predicate_groups'][-1]['lemma'] = 'swam'
    spec, reason = active_phrase(row, 'v0', 'sell', (), {'Anna', 'Mary'})
    assert reason is None and spec['companions'][0]['predicate'] == 'swim'
    assert spec['companions'][0]['lexical_review']['reason'] == 'repaired_finite_tag'
    row['slots']['v2'].update(lemma='lie', tag='VBZ')
    row['defaults']['v2'] = 'lies'
    row['predicate_groups'][-1]['lemma'] = 'lie'
    assert active_phrase(row, 'v0', 'sell', (), {'Anna', 'Mary'})[1] == 'unsupported_companion_clause'


def test_export_contains_review_evidence_without_changing_archives(small_bank, tmp_path, monkeypatch):
    path = small_bank.banks[0].path
    row = lexical_record('came', 'came', 'VBP')
    row['template_id'] = fingerprint(row)
    with gzip.open(path / 'templates' / (row['template_id'][:2] + '.jsonl.gz'), 'at', encoding='utf-8') as stream:
        stream.write(json.dumps(row) + '\n')
    before = {str(p): p.read_bytes() for p in path.rglob('*') if p.is_file()}
    bank = DialogueMinedBank(small_bank.root, cache=False)
    assert bank.entries['came']['lexical_decisions']['repaired_finite_tag'] == 1
    example = bank.review_report()['predicates']['came']['examples']['repaired_finite_tag']
    assert example['reference'][1] == row['template_id']
    assert example['lexical_review']['canonical'] == 'come'
    out = tmp_path / 'export'
    manifest = build_dialogue_dataset(out, train=1, validation=1, test=1, mined_train=0, mined_bank=bank)
    report = json.loads((out / manifest['mined_bank']['review_file']).read_text(encoding='utf-8'))
    assert report['identity'] == bank.identity and report['predicates']['came']['examples']
    assert before == {str(p): p.read_bytes() for p in path.rglob('*') if p.is_file()}
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from review_dialogue_lexical import check_decisions
    assert check_decisions(out)['inventory_decisions']['repaired_finite_tag'] == 1
    ledger = out / manifest['mined_bank']['review_file']
    original = ledger.read_text(encoding='utf-8')
    report['policy']['version'] = 'unrelated-policy'
    ledger.write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='generation policy'):
        check_decisions(out)
    ledger.write_text(original, encoding='utf-8')
    manifest['resources']['mined_morphology.json'] = 'wrong-hash'
    (out / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='resource differs'):
        check_decisions(out)


def test_new_bank_refreshes_policy_and_renderer_and_old_bank_cannot_publish(small_bank, tmp_path, monkeypatch):
    path = tmp_path / 'policy.json'
    data = json.loads(mined_review.POLICY_PATH.read_text(encoding='utf-8'))
    path.write_text(json.dumps(data), encoding='utf-8')
    try:
        with monkeypatch.context() as context:
            context.setattr(mined_review, 'POLICY_PATH', path)
            context.setattr(mined_bank, 'POLICY_PATH', path)
            before = DialogueMinedBank(small_bank.root)
            assert mined_lexeme('plead').past == 'pleaded'
            data['generation']['plead']['forms'] = ['plead', 'pleads', 'pleading', 'pled', 'pled']
            path.write_text(json.dumps(data), encoding='utf-8')
            with pytest.raises(ValueError, match='resource changed'):
                before.review_report()
            after = DialogueMinedBank(small_bank.root)
            assert after.identity != before.identity
            assert mined_lexeme('plead').past == 'pled'
            assert after.review_report()['policy']['generation']['plead']['forms'][3] == 'pled'
            with pytest.raises(ValueError, match='resource changed'):
                before.review_report()
    finally:
        refresh_resources()  # Do not leak a temporary policy into other tests.


def test_changed_dictionary_is_reloaded_and_must_match_the_renderer(small_bank):
    # Both are complete paradigms; the source still agrees with past "slept".
    # The alternate participle would silently differ in generated modal/perfect.
    old = small_bank.dictionary.read_text(encoding='utf-8')
    lexicon(small_bank.dictionary)  # Populate the lookup cache before the edit.
    small_bank.dictionary.write_text(old.replace('slept\tslept', 'slept\tsleeped'), encoding='utf-8')
    bank = DialogueMinedBank(small_bank.root)
    assert lexicon(small_bank.dictionary).paradigm('sleep').past_participle == 'sleeped'
    assert bank.identity != small_bank.identity
    assert bank.entries['sleep']['native_rejections'] == {'generation_paradigm_mismatch': 1}
    assert bank.sample('sleep', random.Random(0))[1] is None
    with pytest.raises(ValueError, match='resource changed'):
        small_bank.review_report()


def test_clause_diagnostics_and_slot_order_are_stable_across_hash_seeds():
    from semantic_corpus.dialogue_generator.mined_clauses import independent_clauses
    row = coordinated()
    clauses, reason = independent_clauses(row)
    assert reason is None
    for clause in clauses:
        assert list(clause['slots']) == [k for k in row['slots'] if k in clause['slots']]
    # Two different failures in the same record used to win in random order.
    row['slots']['n0']['dependency'] = 'dobj'
    row['slots']['v2']['head_slot'] = 'embedded-head'
    code = ('import json,sys; from semantic_corpus.dialogue_generator.mined_clauses import independent_clauses; '
            'print(independent_clauses(json.load(sys.stdin))[1])')
    outcomes = {subprocess.run([sys.executable, '-c', code], input=json.dumps(row), text=True,
                               capture_output=True, check=True, env={**os.environ, 'PYTHONHASHSEED': str(seed)}).stdout.strip()
                for seed in range(6)}
    assert outcomes == {'shared_or_unresolved_coordination_subject'}
