"""Reviewed alternatives must preserve source evidence and contextual answers."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from semantic_corpus.dialogue_generator.fragments import (
    load_fragment_patterns, fragment_episode, validate_fragment_patterns,
)
from semantic_corpus.dialogue_generator.export import extraction_row, response_row
from semantic_corpus.dialogue_generator import DialogueGenerator
from semantic_corpus.dialogue_generator.extraction_text import decode_update, parse_prompt
from semantic_corpus.reviewed_substitutions import (
    substitute, validate_rules, expand_patterns, realize_pattern, render_template,
)
from semantic_corpus.template_bank import TemplateBank, CorpusTemplate, fingerprint
from semantic_corpus.template_usage import validate_usage
from semantic_corpus.documents import Document
from semantic_corpus.semantic_generator.substitutions import EntityPool
from semantic_corpus.template_semantics import TemplateAnnotator

ROOT = Path(__file__).resolve().parents[1] / 'data/mined-templates'
CATALOG = json.loads((ROOT / 'reviewed/dialogue-fragments.json').read_text(encoding='utf-8'))
RULES = CATALOG['substitutions']
PATTERNS = [p for p in load_fragment_patterns() if p.get('substitution_rule', {}).get('id')
            in {r['id'] for r in RULES}]


def test_simultaneous_replacements_respect_words_and_possessives():
    assert substitute("Ann's aunt Anne met Beth.", {'Ann': 'Beth', 'Beth': 'Sue'}) == \
        "Beth's aunt Anne met Sue."


def test_native_rules_cannot_rewrite_verbs_or_fixed_syntax():
    from tests.test_template_usage import restricted
    record = restricted()
    policy = record['usage_constraints']
    policy['meaning'].update(extraction_adapter={'id': 'adapter', 'sha256': '1' * 64},
                             interpretation='Anna reports a purchase.',
                             enclosing_turn_text=policy['text'])
    rule = dict(id='fixture', templates={record['template_id']: fingerprint(record)},
                adapters={'adapter': '1' * 64}, reason='fixture',
                choices=[{'bought': 'will buy'}, {'a red': 'the red'}, {'Anna': 'Beatrice'}])
    template = CorpusTemplate(record)
    assert render_template(template, rule, 0) is None
    assert render_template(template, rule, 1) is None
    result = render_template(template, rule, 2)
    assert result['text'].startswith('Beatrice bought')
    assert result['usage_constraints']['meaning']['interpretation'] == 'Beatrice reports a purchase.'


def test_rules_pin_original_adapters_and_reject_overlaps():
    validate_rules(RULES, CATALOG['patterns'])
    bad = deepcopy(RULES)
    bad[0]['adapters'][next(iter(bad[0]['adapters']))] = '0' * 64
    with pytest.raises(ValueError, match='Stale'):
        validate_rules(bad, CATALOG['patterns'])
    duplicate = deepcopy(RULES[0])
    duplicate['id'] += '_copy'
    with pytest.raises(ValueError, match='Overlapping'):
        validate_rules([RULES[0], duplicate], CATALOG['patterns'])
    conflicting = deepcopy(next(p for p in CATALOG['patterns'] if p['id'] == PATTERNS[0]['substitution_base_id']))
    conflicting['id'] = PATTERNS[0]['id']
    with pytest.raises(ValueError, match='conflicts'):
        expand_patterns([*CATALOG['patterns'], conflicting], RULES)


@pytest.mark.parametrize('pattern', PATTERNS, ids=lambda p: p['id'])
@pytest.mark.parametrize('user_only', [True, False])
def test_every_approved_choice_roundtrips_without_inventing_events(pattern, user_only):
    snapshot = deepcopy(pattern)
    context = ('the kitchen', 'tired', 'tea') if user_only else ('Anna', 'Rome')
    for choice in range(len(pattern['substitution_rule']['choices'])):
        realized = realize_pattern(pattern, choice)
        e = fragment_episode(pattern, context, split='train', user_only=user_only, variant=choice)
        e.validate()
        assert not e.memory.events
        assert e.memory.find('user', 'utterance_interpretation').value == realized['value']
        assert any(t.text == realized['reply'] for t in e.turns)
        assert any(t.text == realized['assistant'] for t in e.turns)
        for transition in e.transitions:
            row = extraction_row(e, transition)
            prompt, target = [m['content'] for m in row['messages'][1:]]
            memory, user, previous = parse_prompt(prompt)
            assert decode_update(target, transition.before, transition.turn,
                                 transition.previous_assistant) == transition.delta
            decode_update(target, memory, user, previous)
            assert not target.startswith(('{', '['))
        for view in ('history', 'memory'):
            assert response_row(e, view)['messages'][-1]['content'] == e.answer
    assert pattern == snapshot


def test_weekend_and_noncompletion_survive_recombination():
    for d in (1863, 1864):
        rule = next(r for r in RULES if r['dialogue'] == d)
        assert rule['choices'] == [{'Saturday': 'Sunday'}]
    loan = next(p for p in PATTERNS if p['substitution_base_id'] == 'dd_backfill_turn_1847_3')
    changed = realize_pattern(loan, 0)
    assert "didn't pay me back" in changed['reply']
    assert 'has not repaid' in changed['value']
    assert 'Alex' in changed['reply'] and 'Alex' in changed['value']
    disputed = next(p for p in PATTERNS if p['substitution_base_id'] == 'dd_backfill_turn_1881_6')
    assert 'remains disputed' in realize_pattern(disputed, 0)['value']
    proposed = next(p for p in PATTERNS if p['substitution_base_id'] == 'dd_backfill_turn_1904_3')
    assert 'proposes' in realize_pattern(proposed, 0)['value']


def test_expansion_preserves_source_catalog_and_detects_stale_patterns():
    before = deepcopy(CATALOG['patterns'])
    expanded = expand_patterns(before, RULES)
    assert expanded[:len(before)] == CATALOG['patterns']
    validate_fragment_patterns(expanded)
    bad = deepcopy(PATTERNS[0])
    bad['reply'] += ' Changed.'
    with pytest.raises(ValueError, match='Stale'):
        realize_pattern(bad, 0)


def test_generated_variants_keep_splits_disjoint(monkeypatch):
    from semantic_corpus.dialogue_generator import fragments
    monkeypatch.setattr(fragments, 'load_fragment_patterns', lambda: deepcopy(PATTERNS))
    generator = DialogueGenerator(seed=42, user_assistant_only=True)
    seen = {}
    for split in ('train', 'validation', 'test'):
        prompts, scenarios, conditions = set(), set(), set()
        for e in generator.generate('fragment_meaning', len(PATTERNS), split=split):
            scenarios.add(e.scenario_key)
            conditions.add(e.condition)
            for row in [extraction_row(e, t) for t in e.transitions] + [
                    response_row(e, view) for view in ('history', 'memory')]:
                prompts.add(fingerprint(row['messages'][:-1]))
        assert len(conditions) == len(PATTERNS)
        for prior_prompts, prior_scenarios in seen.values():
            assert not prompts.intersection(prior_prompts)
            assert not scenarios.intersection(prior_scenarios)
        seen[split] = prompts, scenarios


@pytest.mark.parametrize('component', ['dailydialog-backfill-sentences-1406-1465',
                                       'dailydialog-backfill-completion-1406-1465',
                                       'dailydialog-backfill-1466-1485',
                                       'dailydialog-backfill-1486-1545',
                                       'dailydialog-backfill-1546-1605',
                                       'dailydialog-backfill-1606-1665',
                                       'dailydialog-backfill-1666-1725',
                                       'dailydialog-backfill-1726-1785',
                                       'dailydialog-backfill-1786-1845',
                                       'dailydialog-backfill-1846-1905',
                                       'dailydialog-backfill-1906-1965',
                                       'dailydialog-backfill-1966-2025',
                                       'dailydialog-backfill-2026-2085',
                                       'dailydialog-backfill-2086-2145',
                                       'dailydialog-backfill-2146-2205',
                                       'dailydialog-backfill-2206-2215'])
def test_native_slots_keep_source_pins_and_block_unreviewed_qa(component):
    bank = next(b for b in TemplateBank(ROOT).components()
                if b.source_id == component)
    changed = 0
    for template in bank.templates():
        before = deepcopy(template.record)
        rule = template.substitution_rule
        if rule is None:
            assert template.sample(recombine=True)['text'] == before['usage_constraints']['text']
            assert template.record == before
            continue
        pinned_record, pinned_rule = deepcopy(template.record), rule
        assert template.sample(recombine=False)['text'] == before['usage_constraints']['text']
        possible = []
        for choice in range(len(rule['choices'])):
            row = render_template(template, rule, choice)
            if row is None:
                continue
            changed += 1
            possible.append(row['text'])
            assert row['text'] != before['usage_constraints']['text']
            assert row['reviewed_substitution']['source_template_sha256'] == fingerprint(before)
            validate_usage(row['usage_constraints'], row['text'], row['slots'])
            for slot, span in row['spans'].items():
                assert row['text'][span[0]:span[1]] == row['bindings'][slot]
                if row['slots'][slot]['kind'] == 'verb':
                    assert row['bindings'][slot] == before['defaults'][slot]
            doc = Document('reviewed-variant', row['text'],
                           metadata={k: v for k, v in row.items() if k != 'text'})
            assert TemplateAnnotator(entities=EntityPool([])).annotate(doc).run is None
            with pytest.raises(ValueError, match='slot replacement'):
                template.render(row['bindings'])
        sampled = template.sample(recombine=True)
        assert sampled['text'] in (possible or [before['usage_constraints']['text']])
        assert template.record == before
    assert changed > 0
    stale = deepcopy(pinned_record)
    stale['count'] += 1
    with pytest.raises(ValueError, match='Stale'):
        render_template(CorpusTemplate(stale), pinned_rule, 0)
    pending = deepcopy(pinned_record)
    pending['usage_constraints']['mode'] = 'pending'
    changed_rule = deepcopy(pinned_rule)
    changed_rule['templates'][pending['template_id']] = fingerprint(pending)
    with pytest.raises(ValueError, match='admitted sentence'):
        render_template(CorpusTemplate(pending), changed_rule, 0)


def test_backfill_preserves_uncertainty_offers_and_incomplete_meaning():
    by_id = {p['id']: p for p in CATALOG['patterns']}
    cases = {
        'dd_backfill_turn_1929_14': 'no reliably recoverable meaning',
        'dd_backfill_turn_1934_3': 'incomplete',
        'dd_backfill_turn_1937_17': 'Does not know',
        'dd_backfill_turn_1955_6': 'if this pace continues',
        'dd_backfill_turn_1955_14': 'not accepting or starting',
        'dd_backfill_turn_1962_6': 'alternative explanation',
        'dd_backfill_turn_1965_3': 'without establishing who turned the lock',
    }
    for identity, meaning in cases.items():
        pattern = by_id[identity]
        assert meaning in pattern['value']
        assert all(t <= pattern['source']['turn'] for t in pattern['review']['evidence_turns'])
    # The earlier compact adaptation used a later reply to assert ignorance.
    assert 'does not know her condition' not in by_id['dd_1937_t13']['value']
    assert not any(r['dialogue'] in {1912, 1921, 1961} for r in RULES)
    geography = next(r for r in RULES if r['dialogue'] == 1959)
    assert geography['choices'] == [{'map': 'wall map'}]


def test_backfill_keeps_proposals_inferences_and_unanswered_confessions_distinct():
    by_id = {p['id']: p for p in CATALOG['patterns']}
    cases = {
        'dd_backfill_turn_1966_4': 'not confirmed oversleeping',
        'dd_backfill_turn_1967_3': 'books are now in the box',
        'dd_backfill_turn_1970_5': 'not a completed donation',
        'dd_backfill_turn_1998_8': 'not verified medical advice',
        'dd_backfill_turn_2002_11': 'no visit is completed',
        'dd_backfill_turn_2013_14': 'that task is not finished yet',
        'dd_backfill_turn_2016_9': 'no rejection has occurred',
        'dd_backfill_turn_2018_7': 'explicitly uncertain',
        'dd_backfill_turn_2024_9': 'neither acceptance nor refusal',
    }
    for identity, required in cases.items():
        assert required in by_id[identity]['value']
    hesitant = by_id['dd_backfill_turn_2024_9']
    assert hesitant['reply'] == 'I... I...'
    assert max(hesitant['review']['evidence_turns']) == 9
    variant = next(p for p in PATTERNS
                   if p['substitution_base_id'] == 'dd_backfill_turn_2018_7')
    realized = realize_pattern(variant, 0)
    assert "Daniel's father" in realized['value']
    assert 'explicitly uncertain' in realized['value']
    for dialogue in (1968, 1971, 1986, 1993, 2003, 2008, 2020):
        assert not any(r['dialogue'] == dialogue for r in RULES)


def test_backfill_preserves_noncompletion_and_does_not_fill_from_future_turns():
    by_id = {p['id']: p for p in CATALOG['patterns']}
    cases = {
        '2053_5': 'was supposed to call last night but did not',
        '2053_13': 'no breakup has occurred',
        '2053_16': 'this remains advice',
        '2066_2': 'apple polisher is figurative',
        '2073_16': 'agreement is not obtained yet',
        '2079_5': 'not alive',
        '2080_6': 'missing medium is not supplied from a future reply',
        '2085_4': 'no spraying is reported',
        '2085_6': 'no acceptance of the apology is established',
    }
    for suffix, meaning in cases.items():
        pattern = by_id['dd_backfill_turn_' + suffix]
        assert meaning in pattern['value']
        assert max(pattern['review']['evidence_turns']) <= pattern['source']['turn']
    missing_medium = by_id['dd_backfill_turn_2080_6']
    assert 'oil' not in missing_medium['value'].lower()
    variant = next(p for p in PATTERNS
                   if p['substitution_base_id'] == 'dd_backfill_turn_2053_5')
    realized = realize_pattern(variant, 0)
    assert 'partner' in realized['reply'] and 'partner' in realized['value']
    assert 'was supposed to call last night but did not' in realized['value']
    for dialogue in (2035, 2047, 2052, 2072, 2074, 2083):
        assert not any(r['dialogue'] == dialogue for r in RULES)


def test_backfill_distinguishes_idioms_claims_and_promises_from_events():
    by_id = {p['id']: p for p in CATALOG['patterns']}
    cases = {
        '2087_8': 'success is not asserted',
        '2088_4': 'the rink visit did not occur',
        '2108_1': 'fallen asleep means numb',
        '2109_6': 'not a verified supernatural event',
        '2112_4': 'Alleges gambling addiction',
        '2114_2': 'arrival home is not reported',
        '2116_2': 'volunteer work has not started',
        '2119_6': 'not reached the parents yet',
        '2126_2': 'ended their relationship',
        '2127_7': 'no retake has occurred',
        '2129_12': 'missing items are not invented',
        '2138_4': 'attribution is not confirmed',
        '2139_14': 'no call or reservation is completed',
        '2142_2': 'no animal violence is reported',
    }
    for suffix, expected in cases.items():
        pattern = by_id['dd_backfill_turn_' + suffix]
        assert expected in pattern['value']
        assert max(pattern['review']['evidence_turns']) <= pattern['source']['turn']
    uncertain = next(p for p in PATTERNS
                     if p['substitution_base_id'] == 'dd_backfill_turn_2138_4')
    changed = realize_pattern(uncertain, 0)
    assert 'Li Ming' in changed['value']
    assert 'attribution is not confirmed' in changed['value']
    assert by_id['dd_backfill_turn_2129_12']['reply'].endswith('single,')
    assert 'planes' in by_id['dd_backfill_turn_2117_2']['reply']
    assert 'specious' in by_id['dd_backfill_turn_2095_6']['reply']
    for dialogue in (2094, 2108, 2114, 2117, 2125, 2129, 2142):
        assert not any(r['dialogue'] == dialogue for r in RULES)


def test_backfill_retains_damaged_names_fragments_and_uncompleted_plans():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    cases = {
        '2149_13': 'tickets are not yet reported bought',
        '2160_4': 'residence is not specified',
        '2162_4': 'suspects',
        '2172_3': 'Mary/May mismatch is unresolved',
        '2176_7': "speaker's advice, not verified treatment",
        '2179_3': 'intended place remains uncertain',
        '2180_7': 'relocation is not completed',
        '2183_13': 'no money has been staked',
        '2190_12': 'completed enrollment is not explicitly reported',
        '2195_4': 'list breaks off',
        '2195_6': 'unfinished and not expanded',
        '2198_14': 'reconciliation is not reported',
        '2200_2': 'current location is not specified',
        '2202_5': 'improvements are not completed',
    }
    for suffix, expected in cases.items():
        p = indexed['dd_backfill_turn_' + suffix]
        assert expected in p['value']
        assert max(p['review']['evidence_turns']) <= p['source']['turn']
    assert 'May' in indexed['dd_backfill_turn_2172_3']['reply']
    assert indexed['dd_backfill_turn_2195_4']['reply'].endswith('music, a')
    assert indexed['dd_backfill_turn_2195_6']['reply'].endswith('radar h')
    subject = next(p for p in PATTERNS if p['substitution_base_id'] == 'dd_backfill_turn_2190_3')
    changed = realize_pattern(subject, 0)
    assert 'History' in changed['reply'] and 'History' in changed['value']
    assert 'psychology' not in changed['value'].lower()
    for d in (2178, 2179, 2182, 2193, 2197, 2199):
        assert not any(r['dialogue'] == d for r in RULES)


def test_legacy_adapter_links_require_reviewed_record_and_meaning():
    linked = next(r for r in RULES if r.get('template_adapters'))
    validate_rules([linked], CATALOG['patterns'])
    for field in ('sha256', 'interpretation'):
        changed = deepcopy(linked)
        ref = next(iter(changed['template_adapters'].values()))
        ref[field] = 'unreviewed'
        with pytest.raises(ValueError, match='Unreviewed'):
            validate_rules([changed], CATALOG['patterns'])
    changed = deepcopy(linked)
    key = next(iter(changed['template_adapters']))
    changed['templates'][key] = '0' * 64
    with pytest.raises(ValueError, match='Unreviewed'):
        validate_rules([changed], CATALOG['patterns'])


def test_last_backfill_keeps_wishes_hypotheses_and_stage_actions_distinct():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    cases = {
        '2206_6': 'differs from the earlier brief answer',
        '2208_5': 'hypothetical, not an established fraud',
        '2209_12': 'consultation is not completed',
        '2211_14': 'not verified safety guidance or completed action',
        '2211_21': 'compliance is not yet reported',
        '2213_5': 'reports making and handing over the cocktail',
        '2214_8': 'acceptance remains undecided',
        '2215_5': 'ironing has not occurred',
        '2215_6': 'a wish and a metaphor',
    }
    for suffix, expected in cases.items():
        p = indexed['dd_backfill_turn_' + suffix]
        assert expected in p['value']
        assert max(p['review']['evidence_turns']) <= p['source']['turn']
    assert 'Leo' not in indexed['dd_backfill_link_1420_2_1']['value']
    assert 'subsequently' not in indexed['dd_backfill_link_1406_4_1']['value']
    for d, key, expected in ((2208, 'dd_backfill_turn_2208_2', 'a ninety-five-dollar'),
                             (1641, 'dd_backfill_turn_1641_3', 'folk-music')):
        p = next(p for p in PATTERNS if p['substitution_base_id'] == key)
        assert expected in realize_pattern(p, 0)['value']


def test_historical_backfill_has_no_unreviewed_substitution_holes():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf-8'))
    state = pending['sources'][0]['review']['backfill_1406_2215']
    review = state['substitution_review']
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train'}
    restricted = set(review['restricted_dialogues'])
    assert not licensed.intersection(restricted)
    assert licensed | restricted == set(range(1406, 2216))
    assert review['remaining_dialogues'] == state['remaining_dialogues'] == 0
    assert state['templates_remaining_occurrences'] == 0
    assert not any(state['linked_lexical_remaining'].values())


def test_article_aware_choices_agree_in_context_and_extraction():
    for identity, field, expected, forbidden in (
        ('dd_backfill_turn_1519_3', 'assistant', 'A glass ashtray', 'An glass'),
        ('dd_backfill_turn_1719_1', 'value', 'a French test', 'an French'),
        ('dd_backfill_turn_1719_6', 'assistant', 'a French test', 'an French'),
    ):
        pattern = next(p for p in PATTERNS if p['substitution_base_id'] == identity)
        text = realize_pattern(pattern, 0)[field]
        assert expected in text and forbidden not in text
