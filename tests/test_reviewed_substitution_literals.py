"""Finite literal licenses must preserve source pins and grammar boundaries."""
from copy import deepcopy

import pytest

from semantic_corpus.reviewed_substitutions import render_template
from semantic_corpus.template_bank import CorpusTemplate, fingerprint
from semantic_corpus.template_usage import constrain_record
from tests.test_template_semantics import template


def fixture():
    record = template(adjective=False)
    record['parts'][-1] = {'literal': ' for 9 dollars.'}
    part = len(record['parts']) - 1
    record['predicate_groups'][0]['constructions'] = [dict(
        lemma='buy for', kind='preposition_candidate', separated=False,
        markers=[dict(part=part, start=1, end=4)])]
    text = CorpusTemplate(record).render()['text']
    body = dict(idx='test:1:1:1', example=text,
                provenance=dict(origin=dict(turn=1, span=[0, len(text)])))
    constrain_record(record, body, [text], dict(unit='sentence', meaning={'event_status': 'reported'},
                     usage=dict(mode='attested_sentence', reasons=['fixture'])))
    record['usage_constraints']['meaning'].update(
        extraction_adapter=dict(id='adapter', sha256='1' * 64),
        interpretation='Reports buying a book for 9 dollars.', enclosing_turn_text=text)
    rule = dict(id='fixture', templates={record['template_id']: fingerprint(record)},
                adapters={'adapter': '1' * 64}, reason='Change quoted price only.',
                choices=[{'9': '125'}])
    return record, rule, part


def license_parts(record, rule, part, before, after):
    rule['literal_choices'] = {record['template_id']: [dict(
        choice=0, choice_sha256=fingerprint(rule['choices'][0]), reason='Explicit fixture review',
        parts=[dict(part=part, before=before, after=after)])]}


def test_literal_price_requires_explicit_license_and_preserves_original():
    record, rule, part = fixture()
    original = deepcopy(record)
    assert render_template(CorpusTemplate(record), rule, 0) is None
    license_parts(record, rule, part, ' for 9 dollars.', ' for 125 dollars.')
    result = render_template(CorpusTemplate(record), rule, 0)
    assert result['text'] == 'Anna bought a book for 125 dollars.'
    assert result['usage_constraints']['meaning']['interpretation'].endswith('125 dollars.')
    assert result['predicate_groups'][0]['constructions'][0]['markers'][0]['text'] == 'for'
    assert result['bindings']['v0'] == 'bought'
    assert record == original
    assert CorpusTemplate(record).render(result['bindings'])['text'].endswith('9 dollars.')


def test_literal_license_is_pinned_to_choice_and_original_part():
    record, rule, part = fixture()
    license_parts(record, rule, part, ' for 8 dollars.', ' for 125 dollars.')
    with pytest.raises(ValueError, match='Stale reviewed literal part'):
        render_template(CorpusTemplate(record), rule, 0)
    license_parts(record, rule, part, ' for 9 dollars.', ' for 125 dollars.')
    rule['choices'][0]['9'] = '100'
    with pytest.raises(ValueError, match='Stale reviewed literal choice'):
        render_template(CorpusTemplate(record), rule, 0)


def test_literal_license_cannot_change_construction_marker_or_verb_slot():
    record, rule, part = fixture()
    rule['choices'] = [{'for': 'with'}]
    license_parts(record, rule, part, ' for 9 dollars.', ' with 9 dollars.')
    assert render_template(CorpusTemplate(record), rule, 0) is None
    rule['choices'] = [{'bought': 'sold'}]
    license_parts(record, rule, 2, 'bought', 'sold')
    with pytest.raises(ValueError, match='Stale reviewed literal part'):
        render_template(CorpusTemplate(record), rule, 0)


def test_literal_length_change_rebases_later_marker():
    record, rule, part = fixture()
    record['parts'][part] = {'literal': ' 9 days before departure.'}
    record['predicate_groups'][0]['constructions'][0].update(
        lemma='buy before', markers=[dict(part=part, start=8, end=14)])
    policy = record['usage_constraints']
    text = 'Anna bought a book 9 days before departure.'
    policy.update(text=text, context=[text], span=[0, len(text)])
    rule['templates'][record['template_id']] = fingerprint(record)
    license_parts(record, rule, part, ' 9 days before departure.', ' 125 days before departure.')
    result = render_template(CorpusTemplate(record), rule, 0)
    marker = result['predicate_groups'][0]['constructions'][0]['markers'][0]
    assert marker['text'] == 'before'
    assert result['text'][slice(*marker['span'])] == 'before'


def test_unlicensed_extra_literal_edit_does_not_render():
    record, rule, part = fixture()
    license_parts(record, rule, part, ' for 9 dollars.', ' for 125 euros.')
    assert render_template(CorpusTemplate(record), rule, 0) is None


def slot_license(record, rule, bindings):
    rule['slot_choices'] = {record['template_id']: [dict(
        choice=0, choice_sha256=fingerprint(rule['choices'][0]), reason='Explicit alignment', bindings=bindings)]}


def test_explicit_slot_alignment_preserves_source_and_exact_target():
    record, rule, _ = fixture()
    original = deepcopy(record)
    rule['choices'] = [{'Anna': 'Beth', 'book': 'novel'}]
    slot_license(record, rule, {'n0': dict(before='Anna', after='Beth'), 'n1': dict(before='book', after='novel')})
    result = render_template(CorpusTemplate(record), rule, 0)
    assert result['text'] == 'Beth bought a novel for 9 dollars.'
    assert result['bindings']['v0'] == 'bought'
    assert result['predicate_groups'][0]['constructions'][0]['markers'][0]['text'] == 'for'
    assert record == original
    # A licensed alignment cannot supply a different final sentence.
    rule['slot_choices'][record['template_id']][0]['bindings']['n1']['after'] = 'the novel'
    assert render_template(CorpusTemplate(record), rule, 0) is None


@pytest.mark.parametrize('bindings', [
    {'v0': dict(before='bought', after='sold')},
    {'n1': dict(before='magazine', after='novel')},
    {'missing': dict(before='book', after='novel')},
    {'n1': dict(before='book', after='')},
])
def test_slot_alignment_rejects_verb_stale_missing_and_empty_bindings(bindings):
    record, rule, _ = fixture()
    rule['choices'] = [{'book': 'novel'}]
    slot_license(record, rule, bindings)
    with pytest.raises(ValueError, match='reviewed slot binding'):
        render_template(CorpusTemplate(record), rule, 0)


def test_slot_alignment_choice_hash_cannot_be_reused():
    record, rule, _ = fixture()
    rule['choices'] = [{'book': 'novel'}]
    slot_license(record, rule, {'n1': dict(before='book', after='novel')})
    rule['choices'][0]['book'] = 'magazine'
    with pytest.raises(ValueError, match='Stale reviewed slot choice'):
        render_template(CorpusTemplate(record), rule, 0)


def verb_fixture():
    record, rule, _ = fixture()
    record['predicate_groups'][0]['constructions'] = []
    rule['templates'][record['template_id']] = fingerprint(record)
    rule['choices'] = [{'bought': 'purchased'}]
    rule['verb_choices'] = {record['template_id']: [dict(
        choice=0, choice_sha256=fingerprint(rule['choices'][0]), reason='Reviewed same-form lexical pair',
        bindings={'v0': dict(before='bought', after='purchased', lemma='purchase', tag='VBD',
                             slot_sha256=fingerprint(record['slots']['v0']))})]}
    return record, rule


def test_verb_license_updates_lemma_and_spans_without_mutating_source():
    record, rule = verb_fixture()
    original = deepcopy(record)
    row = render_template(CorpusTemplate(record), rule, 0)
    assert row['text'] == 'Anna purchased a book for 9 dollars.'
    assert row['slots']['v0'] == {**record['slots']['v0'], 'lemma': 'purchase'}
    assert row['predicate_groups'][0]['lemma'] == 'purchase'
    assert row['verbs'] == ['purchase']
    assert row['text'][slice(*row['spans']['v0'])] == 'purchased'
    assert row['usage_constraints']['mode'] == 'context_only'
    assert record == original


@pytest.mark.parametrize('field,value', [('tag','VB'), ('before','buys'), ('slot_sha256','0'*64),
                                       ('after','will purchase'), ('lemma','')])
def test_verb_license_rejects_stale_form_and_multiword_rewrite(field, value):
    record, rule = verb_fixture()
    rule['verb_choices'][record['template_id']][0]['bindings']['v0'][field] = value
    with pytest.raises(ValueError, match='reviewed verb binding'):
        render_template(CorpusTemplate(record), rule, 0)


def test_verb_license_rejects_auxiliary_even_with_updated_pins():
    record, rule = verb_fixture()
    record['slots']['v0'].update(pos='AUX', dependency='aux')
    rule['templates'][record['template_id']] = fingerprint(record)
    rule['verb_choices'][record['template_id']][0]['bindings']['v0']['slot_sha256'] = fingerprint(record['slots']['v0'])
    with pytest.raises(ValueError, match='reviewed verb binding'):
        render_template(CorpusTemplate(record), rule, 0)


def test_verb_license_cannot_change_construction_or_reuse_choice():
    record, rule = verb_fixture()
    original, _, _ = fixture()
    record['predicate_groups'] = original['predicate_groups']
    rule['templates'][record['template_id']] = fingerprint(record)
    assert render_template(CorpusTemplate(record), rule, 0) is None
    rule['choices'][0]['bought'] = 'sold'
    with pytest.raises(ValueError, match='Stale reviewed verb choice'):
        render_template(CorpusTemplate(record), rule, 0)


def test_verb_license_must_render_exact_approved_target():
    record, rule = verb_fixture()
    rule['verb_choices'][record['template_id']][0]['bindings']['v0'].update(after='sold', lemma='sell')
    assert render_template(CorpusTemplate(record), rule, 0) is None


def phrase_fixture():
    record, rule, _ = fixture()
    rule['choices'] = [{'bought': 'purchased'}]
    before = CorpusTemplate(record).render()['text']
    rule['phrase_choices'] = {record['template_id']: [dict(
        choice=0, choice_sha256=fingerprint(rule['choices'][0]), reason='Reviewed complete paraphrase',
        before=before, after=before.replace('bought', 'purchased'))]}
    return record, rule


def test_whole_phrase_license_does_not_inherit_an_unreviewed_parse():
    record, rule = phrase_fixture()
    original = deepcopy(record)
    row = render_template(CorpusTemplate(record), rule, 0)
    assert row['text'] == 'Anna purchased a book for 9 dollars.'
    assert row['slots'] == row['bindings'] == row['spans'] == {}
    assert not row['predicate_groups'] and not row['verbs'] and not row['attributes']
    assert row['usage_constraints']['mode'] == 'context_only'
    assert record == original
    del rule['phrase_choices']
    assert render_template(CorpusTemplate(record), rule, 0) is None


@pytest.mark.parametrize('field,value', [('before','Different source'),('after','Anna will purchase a book.'),
                                       ('choice_sha256','0'*64),('reason','')])
def test_whole_phrase_license_rejects_stale_or_unlicensed_surface(field,value):
    record, rule = phrase_fixture()
    rule['phrase_choices'][record['template_id']][0][field] = value
    with pytest.raises(ValueError, match='Stale reviewed phrase'):
        render_template(CorpusTemplate(record), rule, 0)


def test_correction_applies_to_original_and_variant_without_changing_source():
    from semantic_corpus.reviewed_substitutions import expand_patterns, realize_pattern, validate_rules
    record, rule = phrase_fixture()
    p = dict(id='adapter', kind='interpretation', source=dict(split='train',dialogue=1),
             assistant='Context',reply='Anna bought a book.',value='Incorrect meaning',question='Meaning?')
    rule.update(version='reviewed-substitutions-v1',split='train',dialogue=1,adapters={'adapter':fingerprint(p)},
        adapter_revisions={'adapter':dict(source_sha256=fingerprint(p),reason='Direct reread',
            fields={'value':dict(before='Incorrect meaning',after='Reports a purchase.')})})
    original = deepcopy(p)
    effective, variant = expand_patterns([p], [rule])
    assert effective['value'] == realize_pattern(variant,0)['value'] == 'Reports a purchase.'
    assert realize_pattern(variant,0)['reply'] == 'Anna purchased a book.'
    assert p == original
    assert effective['source'] == p['source']
    rule['adapter_revisions']['adapter']['fields']['value']['before'] = 'Stale'
    with pytest.raises(ValueError, match='Stale reviewed adapter'):
        validate_rules([rule],[p])


def test_effective_template_meaning_keeps_archived_record_and_revision_proof():
    record, _, _ = fixture()
    original = deepcopy(record)
    revision = dict(source_sha256='1'*64,reason='Correct a semantic inference',
                    fields={'value':dict(before='Reports buying a book for 9 dollars.',after='Reported purchase, not independently verified.')})
    row = CorpusTemplate(record,adapter_revisions={'adapter':revision}).render()
    assert row['usage_constraints']['meaning']['interpretation'] == 'Reported purchase, not independently verified.'
    assert row['usage_constraints']['meaning']['adapter_revision'] == revision
    assert record == original
    revision['source_sha256'] = '0'*64
    with pytest.raises(ValueError, match='Stale template adapter'):
        CorpusTemplate(record,adapter_revisions={'adapter':revision}).render()
