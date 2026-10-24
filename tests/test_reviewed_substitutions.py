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


def test_new_batch_keeps_originals_and_covers_every_turn():
    catalog = {p['id']: p for p in CATALOG['patterns']}
    source = Path(__file__).resolve().parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl'
    rows = source.read_text(encoding='utf8').splitlines()
    checked = 0
    for d in range(2276, 2336):
        for turn, text in enumerate(json.loads(rows[d-1]), 1):
            pattern = catalog[f'dd_review_turn_{d}_{turn}']
            assert pattern['reply'] == text == pattern['source']['text']
            assert all(t <= turn for t in pattern['review']['evidence_turns'])
            checked += 1
    assert checked == 469
    # A denial, confession and promise are separate states, not a single event.
    assert 'Denies the allegation' in catalog['dd_review_turn_2303_6']['value']
    assert 'Confesses taking' in catalog['dd_review_turn_2303_8']['value']
    assert 'not yet been completed' in catalog['dd_review_turn_2303_10']['value']
    # Missing complements and inconsistent names remain bounded by evidence.
    assert 'unspecified' in catalog['dd_review_turn_2327_2']['value']
    assert 'not silently replaced' in catalog['dd_review_turn_2298_14']['value']
    assert 'not completed' in catalog['dd_review_turn_2326_2']['value']
    assert 'guess' not in catalog['dd_review_turn_2326_2']['reply'].lower()
    assert 'route maps' not in catalog['dd_review_turn_2294_9']['assistant']
    assert 'does not establish a precise emotion' in catalog['dd_review_turn_2299_6']['value']


def test_interpretation_prompt_preserves_contract_without_unrelated_event_syntax():
    from semantic_corpus.dialogue_generator.extraction_text import EXTRACT_SYSTEM, INTERPRETATION_EXTRACT_SYSTEM
    assert 'SET subject | relation | modality | polarity | value' in INTERPRETATION_EXTRACT_SYSTEM
    assert 'SET_CONTEXT' in INTERPRETATION_EXTRACT_SYSTEM
    assert 'NO_CHANGE' in INTERPRETATION_EXTRACT_SYSTEM
    assert 'never completion or truth of embedded events' in INTERPRETATION_EXTRACT_SYSTEM
    assert 'EVENT predicate' not in INTERPRETATION_EXTRACT_SYSTEM
    assert 'EVENT predicate' in EXTRACT_SYSTEM


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
                                       'dailydialog-backfill-2206-2215',
                                       'dailydialog-reviewed-2276-2335',
                                       'dailydialog-reviewed-2336-2395',
                                       'dailydialog-reviewed-2396-2455',
                                       'dailydialog-reviewed-2456-2515',
                                       'dailydialog-reviewed-2516-2575',
                                       'dailydialog-reviewed-2576-2635',
                                       'dailydialog-reviewed-2636-2695',
                                       'dailydialog-reviewed-2696-2755',
                                       'dailydialog-reviewed-2756-2815',
                                       'dailydialog-reviewed-2816-2875',
                                       'dailydialog-reviewed-2876-2935',
                                       'dailydialog-reviewed-2936-2995',
                                       'dailydialog-reviewed-2996-3055',
                                       'dailydialog-reviewed-3056-3115',
                                       'dailydialog-reviewed-3116-3175',
                                       'dailydialog-reviewed-3176-3235',
                                       'dailydialog-reviewed-3236-3295',
                                       'dailydialog-reviewed-3296-3355',
                                       'dailydialog-reviewed-3356-3415'])
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
            if row['bindings'] != before['defaults']:
                with pytest.raises(ValueError, match='slot replacement'):
                    template.render(row['bindings'])
            else:
                # A pinned literal variant is not an unrestricted source edit.
                assert template.render(row['bindings'])['text'] == before['usage_constraints']['text']
                assert row['text'] != before['usage_constraints']['text']
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
    for dialogue in (2094, 2108, 2114, 2117, 2129, 2142):
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
    licensed = {r['dialogue'] for r in RULES
                if r['split'] == 'train' and 1406 <= r['dialogue'] <= 2215}
    restricted = set(review['restricted_dialogues'])
    assert not licensed.intersection(restricted)
    assert licensed | restricted == set(range(1406, 2216))
    assert review['remaining_dialogues'] == state['remaining_dialogues'] == 0
    assert state['templates_remaining_occurrences'] == 0
    assert not any(state['linked_lexical_remaining'].values())


def test_new_batch_has_decisions_in_every_resource_track():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf-8'))
    review = pending['sources'][0]['review']
    batch = review['batches']['train_2276_2335']
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 2276 <= r['dialogue'] <= 2335}
    restricted = {r['dialogue'] for r in batch['substitution_restrictions']}
    assert not licensed & restricted
    assert licensed | restricted == set(range(2276, 2336))
    decisions = [r for key in ('sentence_attestations', 'occurrence_adaptations')
                 for r in review[key]['decisions']
                 if r['occurrence_id'].startswith('train:') and 2276 <= int(r['occurrence_id'].split(':')[1]) <= 2335]
    assert len({r['occurrence_id'] for r in decisions}) == len(decisions) == 765
    assert sum(v['added_decisions'] for v in batch['lexical'].values()) == 523
    assert batch['remaining_batch_occurrences'] == batch['remaining_batch_turns'] == 0


def test_article_aware_choices_agree_in_context_and_extraction():
    for identity, field, expected, forbidden in (
        ('dd_backfill_turn_1519_3', 'assistant', 'A glass ashtray', 'An glass'),
        ('dd_backfill_turn_1719_1', 'value', 'a French test', 'an French'),
        ('dd_backfill_turn_1719_6', 'assistant', 'a French test', 'an French'),
    ):
        pattern = next(p for p in PATTERNS if p['substitution_base_id'] == identity)
        text = realize_pattern(pattern, 0)[field]
        assert expected in text and forbidden not in text


def test_batch_2336_2395_preserves_all_source_turns_and_semantic_limits():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    rows = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    count = 0
    for d in range(2336, 2396):
        for t, text in enumerate(json.loads(rows[d-1]), 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert max(p['review']['evidence_turns']) <= t
            count += 1
    assert count == 450
    cases = {
        '2353_6': 'does not establish ability to perform the vault',
        '2359_1': 'gender is not inferred',
        '2360_18': 'neither task is reported completed',
        '2366_3': 'completed marriage is uncertain',
        '2369_2': 'not confirmed forgetting',
        '2373_10': 'not a report of completed divorce',
        '2374_11': 'no programs are invented',
        '2376_1': 'not a resolved person',
        '2377_2': 'not silently converted to a deadline',
        '2380_2': 'not an injured animal',
        '2390_8': 'not verified',
        '2390_12': 'breakup is not completed',
        '2392_5': 'park origin is not established',
        '2394_13': 'not independently verified',
    }
    for key, expected in cases.items():
        assert expected in indexed['dd_review_turn_' + key]['value']
    assert indexed['dd_review_turn_2374_11']['reply'].endswith('programs')
    assert 'shell' in indexed['dd_review_turn_2376_1']['reply']
    assert 'for half an hour' in indexed['dd_review_turn_2377_2']['reply']


def test_batch_2336_2395_covers_candidates_lexicon_and_substitution_decisions():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_2336_2395']
    decisions = [row for key in ('sentence_attestations', 'occurrence_adaptations')
                 for row in review[key]['decisions']
                 if row['occurrence_id'].startswith('train:')
                 and 2336 <= int(row['occurrence_id'].split(':')[1]) <= 2395]
    assert len(decisions) == len({row['occurrence_id'] for row in decisions}) == 751
    assert sum(x['added_decisions'] for x in report['lexical'].values()) == 461
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 2336 <= r['dialogue'] <= 2395}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert not licensed & restricted
    assert licensed | restricted == set(range(2336, 2396))
    assert report['remaining_batch_occurrences'] == report['remaining_batch_turns'] == 0
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for row in decisions:
        pin = row['allowed_adapter']
        assert pin['sha256'] == fingerprint(indexed[pin['id']])


def test_batch_2396_2455_preserves_source_scope_and_uncertainty():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    rows = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    count = 0
    for d in range(2396, 2456):
        for t, text in enumerate(json.loads(rows[d - 1]), 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert max(p['review']['evidence_turns']) <= t
            count += 1
    assert count == 424
    for key, expected in {
        '2413_2': 'whether the breakup has been communicated is not established',
        '2415_2': 'does not independently establish infidelity',
        '2415_3': 'do not resolve the mismatch',
        '2418_11': 'explicitly reports Jimmy bringing May orange juice',
        '2418_12': 'do not establish a different drink recipient',
        '2418_16': 'neither activity is reported completed',
        '2432_4': 'not assigned an invented reference',
        '2434_4': 'shooter after the pass is not unambiguously named',
        '2448_2': 'unverified steroid-use allegation',
        '2448_4': 'no drug use is established',
        '2452_10': 'alternatives, not two certain colors',
        '2452_13': 'no clothing description is supplied',
    }.items():
        assert expected in indexed['dd_review_turn_' + key]['value']
    changed = next(p for p in PATTERNS if p['substitution_base_id'] == 'dd_review_turn_2418_11')
    assert 'Paul bringing Anna orange juice' in realize_pattern(changed, 0)['value']
    # Follow-up licenses independent fillers, not resolution of these ambiguities.
    for d, turn in ((2415, 3), (2432, 4), (2434, 4), (2448, 2), (2452, 10)):
        identity = f'dd_review_turn_{d}_{turn}'
        variants = [p for p in PATTERNS if p['substitution_base_id'] == identity]
        for variant in variants:
            value = realize_pattern(variant, 0)['value']
            assert value.replace('Emma', 'Jenny') == indexed[identity]['value']


def test_batch_2396_2455_covers_every_candidate_and_lexical_track():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_2396_2455']
    decisions = [row for key in ('sentence_attestations', 'occurrence_adaptations')
                 for row in review[key]['decisions']
                 if row['occurrence_id'].startswith('train:')
                 and 2396 <= int(row['occurrence_id'].split(':')[1]) <= 2455]
    assert len(decisions) == len({row['occurrence_id'] for row in decisions}) == 720
    assert sum(x['added_decisions'] for x in report['lexical'].values()) == 429
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 2396 <= r['dialogue'] <= 2455}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert not licensed & restricted
    assert licensed | restricted == set(range(2396, 2456))
    assert report['remaining_batch_occurrences'] == report['remaining_batch_turns'] == 0
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for row in decisions:
        pin = row['allowed_adapter']
        assert pin['sha256'] == fingerprint(indexed[pin['id']])


def test_batch_2456_2515_retains_game_quotation_uncertainty_and_source_damage():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    rows = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    count = 0
    for d in range(2456, 2516):
        for t, text in enumerate(json.loads(rows[d-1]), 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert max(p['review']['evidence_turns']) <= t
            count += 1
    assert count == 444
    for key, expected in {
        '2458_4': 'no actual search or upward glance is established',
        '2460_4': 'intended destination remains unknown',
        '2462_9': 'their cover story, not an established fact',
        '2462_11': 'knowing complicity is not established',
        '2465_7': 'does not mean biological sleep',
        '2468_8': 'chose to go home',
        '2470_14': 'not a legal rule or assurance of safety',
        '2478_6': 'does not itself establish their arrival',
        '2485_1': 'does not mean two separate plans',
        '2487_10': 'actual groping is not reported',
        '2495_8': 'relationship status remains unverified',
        '2502_5': 'no continuation is invented',
        '2505_8': 'confined to the game, not a real death',
        '2507_3': 'no causal connection',
        '2508_4': 'Rejects that assumption',
        '2509_10': 'does not prove absence of regulation',
        '2510_7': "does not make George's tickets free",
        '2511_10': 'full mastery of butterfly stroke is not established',
        '2512_13': 'not treated as literal bending or a ship event',
    }.items():
        assert expected in indexed['dd_review_turn_' + key]['value']
    assert indexed['dd_review_turn_2502_5']['reply'].endswith('Someone wit')
    assert 'rye' in indexed['dd_review_turn_2460_4']['reply']
    assert indexed['dd_review_turn_2499_3']['reply'] == indexed['dd_review_turn_2417_3']['reply']
    assert indexed['dd_review_turn_2499_3']['source']['dialogue'] != indexed['dd_review_turn_2417_3']['source']['dialogue']
    for key, wanted in (('dd_review_turn_2462_9', 'scooter'),
                        ('dd_review_turn_2473_6', 'Italian dressing'),
                        ('dd_review_turn_2510_7', "Paul's tickets free")):
        pattern = next(p for p in PATTERNS if p['substitution_base_id'] == key)
        assert wanted in realize_pattern(pattern, 0)['value']


def test_batch_2456_2515_covers_all_resource_tracks():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_2456_2515']
    decisions = [row for key in ('sentence_attestations', 'occurrence_adaptations')
                 for row in review[key]['decisions']
                 if row['occurrence_id'].startswith('train:')
                 and 2456 <= int(row['occurrence_id'].split(':')[1]) <= 2515]
    assert len(decisions) == len({row['occurrence_id'] for row in decisions}) == 798
    assert sum(x['added_decisions'] for x in report['lexical'].values()) == 443
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 2456 <= r['dialogue'] <= 2515}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert not licensed & restricted
    assert licensed | restricted == set(range(2456, 2516))
    assert report['remaining_batch_occurrences'] == report['remaining_batch_turns'] == 0
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for row in decisions:
        pin = row['allowed_adapter']
        assert pin['sha256'] == fingerprint(indexed[pin['id']])


def test_batch_2456_2515_never_reopens_excluded_truncation():
    from semantic_corpus.dailydialog_quarantine import load_holds, check_admitted_ids, check_fragment_catalog
    catalog_path = ROOT / 'reviewed/dialogue-fragments.json'
    holds = load_holds(ROOT / 'reviewed/dailydialog-quarantine.json')
    held = next(e for e in holds if e['id'] == 'train:2502:5:4')
    assert held['status'] == 'excluded' and held['review_closed'] is True
    assert held['template_admission'] is False and held['automatic_retry'] is False
    original = next(p for p in CATALOG['patterns'] if p['id'] == 'dd_review_turn_2502_5')
    assert held['allowed_adapters'][original['id']] == fingerprint(original)
    check_fragment_catalog(CATALOG['patterns'], catalog_path)
    with pytest.raises(ValueError, match='blocks template admission'):
        check_admitted_ids([held['id']], holds)
    invented = deepcopy(original)
    invented['value'] = 'A qualified person offered the speaker a genetics job.'
    with pytest.raises(ValueError, match='blocks fragment adapter'):
        check_fragment_catalog([invented if p['id'] == original['id'] else p
                                for p in CATALOG['patterns']], catalog_path)


def test_batch_2516_2575_preserves_sources_contradictions_and_planned_actions():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    rows = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    count = 0
    for d in range(2516, 2576):
        for t, text in enumerate(json.loads(rows[d-1]), 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert max(p['review']['evidence_turns']) <= t
            count += 1
    assert count == 463
    for key, premature_name in (('2526_1', 'Sally'), ('2527_1', 'White'), ('2551_1', 'Jared')):
        opening = indexed['dd_review_turn_' + key]
        assert premature_name not in opening['value']
        assert opening['review']['evidence_turns'] == [1]
    for key, expected in {
        '2520_5': 'Jake/Jack variation remains unresolved',
        '2522_7': 'does not establish that chips were eaten',
        '2526_7': 'does not establish completion',
        '2533_5': 'roses were not obtained',
        '2534_4': 'conflicts with the earlier twice-weekly claim',
        '2537_2': 'does not reliably identify their container',
        '2539_6': "Taylor's death is not explicitly stated",
        '2540_6': 'pronunciation repair',
        '2541_8': 'not independently confirmed anatomy',
        '2541_9': 'Successful ring placement is not stated',
        '2542_7': 'does not say he inherited the house',
        '2545_5': 'not a universal visa rule',
        '2548_2': 'completion or publication of an article is unknown',
        '2551_8': "parents' agreement to fund most costs",
        '2551_10': 'not completed services or confirmed bookings',
        '2551_13': 'rather than established mortal danger',
        '2556_3': 'acceptance and starting work are not stated',
        '2557_6': 'not verified medical facts',
        '2558_7': 'returning change or issuing tickets is not stated',
        '2560_5': 'not that each spouse lives alone',
        '2562_11': 'no identity reassignment is invented',
        '2567_5': 'does not deny their online contact',
        '2569_4': 'they come from Canada',
        '2570_3': 'do not establish a divorce decision',
        '2571_3': 'not reconstructed as khaki',
        '2573_7': 'changes his decision, gives the key',
        '2575_6': 'does not identify Jurassic Park as a joint winner',
    }.items():
        assert expected in indexed['dd_review_turn_' + key]['value']
    long = indexed['dd_review_turn_2551_10']
    assert long['review']['evidence_turns'] == [1, 2, 8, 9, 10]
    assert len(long['reply']) > 800  # Keep the whole long source reply.
    for key, expected in (('2516_2', 'Helen'), ('2533_5', 'tulips were not obtained'),
                          ('2569_2', '1994'), ('2574_3', 'Helen Morgan')):
        p = next(p for p in PATTERNS if p['substitution_base_id'] == 'dd_review_turn_' + key)
        assert expected in realize_pattern(p, 0)['value']


def test_batch_2516_2575_covers_every_resource_and_retains_closed_quarantine():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_2516_2575']
    decisions = [row for key in ('sentence_attestations', 'occurrence_adaptations')
                 for row in review[key]['decisions']
                 if row['occurrence_id'].startswith('train:')
                 and 2516 <= int(row['occurrence_id'].split(':')[1]) <= 2575]
    assert len(decisions) == len({row['occurrence_id'] for row in decisions}) == 800
    assert sum(x['added_decisions'] for x in report['lexical'].values()) == 507
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 2516 <= r['dialogue'] <= 2575}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert not licensed & restricted
    assert licensed | restricted == set(range(2516, 2576))
    assert report['remaining_batch_occurrences'] == report['remaining_batch_turns'] == 0
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for row in decisions:
        pin = row['allowed_adapter']
        assert pin['sha256'] == fingerprint(indexed[pin['id']])
    checks = report['checks']
    assert checks['max_tokens'] <= 768 and checks['overlength'] == []
    assert checks['source_reply_truncation'] is False
    assert report['remaining_global']['unresolved_quarantine_warnings'] == 0


def test_restrictions_2516_2575_followup_covers_all_22_without_rewriting_evidence():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    batch = pending['sources'][0]['review']['batches']['train_2516_2575']
    followup = batch['substitution_followup']
    expected = {2517, 2519, 2524, 2528, 2529, 2530, 2534, 2543, 2545, 2546, 2550,
                2553, 2555, 2557, 2558, 2560, 2564, 2565, 2566, 2567, 2571, 2573}
    assert {r['dialogue'] for r in followup['original_restrictions']} == expected
    assert {r['dialogue'] for r in followup['decisions']} == expected
    assert batch['substitution_restrictions'] == []
    assert batch['substitution_rules'] == 60
    rules = {r['dialogue']: r for r in RULES if r['dialogue'] in expected}
    assert set(rules) == expected
    originals = {p['id']: p for p in CATALOG['patterns']}
    source = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    for decision in followup['decisions']:
        rule = rules[decision['dialogue']]
        assert decision['rule_sha256'] == fingerprint(rule)
        assert decision['retained_limits'] == rule['reason']
        for turn, text in enumerate(json.loads(source[rule['dialogue']-1]), 1):
            p = originals[f"dd_review_turn_{rule['dialogue']}_{turn}"]
            assert p['reply'] == p['source']['text'] == text
            assert fingerprint(p) == rule['adapters'][p['id']]
            assert max(p['review']['evidence_turns']) <= turn


def _restriction_variant(dialogue, turn, choice=0):
    p = next(p for p in PATTERNS if p['substitution_base_id'] == f'dd_review_turn_{dialogue}_{turn}')
    return realize_pattern(p, choice)


def test_restrictions_2516_2575_prices_and_time_change_together():
    assert '80 yuan' in _restriction_variant(2558, 4)['reply']
    assert 'eighty yuan' in _restriction_variant(2558, 4)['value']
    assert '50 yuan' in _restriction_variant(2558, 6)['reply']
    assert 'fifty yuan' in _restriction_variant(2558, 6)['value']
    paid = _restriction_variant(2558, 7)
    assert paid['reply'] == 'Here is 70 yuan.'
    assert 'seventy yuan' in paid['value']
    assert 'Twenty yuan would be due in change' in paid['value']
    assert 'returning change or issuing tickets is not stated' in paid['value']
    assert 'eighty yuan' in paid['assistant'] and 'fifty yuan' in paid['assistant']
    # The second choice is independent of the first; prices remain original.
    assert _restriction_variant(2558, 3, 1)['reply'] == '3:30, please.'
    assert 'three-thirty' in _restriction_variant(2558, 7, 1)['assistant']
    assert _restriction_variant(2558, 7, 1)['reply'] == 'Here is 50 yuan.'
    assert 'Ten yuan would be due' in _restriction_variant(2558, 7, 1)['value']
    assert 'before ten' in _restriction_variant(2573, 3)['reply']
    assert 'ends at ten' in _restriction_variant(2573, 4)['value']
    assert 'journey and return have not yet occurred' in _restriction_variant(2573, 7)['value']


def test_restrictions_2516_2575_temperature_and_references_stay_consistent():
    for turn in range(1, 13):
        p = _restriction_variant(2565, turn)
        assert 'cold' not in p['reply'] + p['assistant'] + p['value']
    assert 'warmer' not in _restriction_variant(2565, 7)['reply']
    assert 'coolness' in _restriction_variant(2565, 7)['value']
    assert 'centrally controlled' in _restriction_variant(2565, 10)['value']
    assert 'cannot be adjusted' in _restriction_variant(2565, 10)['value']
    assert _restriction_variant(2565, 12)['reply'] == 'You can wear lighter clothes.'
    assert 'doing so is not reported' in _restriction_variant(2565, 12)['value']
    assert "Dad, have we got any cookies?" in _restriction_variant(2564, 1)['reply']
    assert 'father' in _restriction_variant(2564, 1)['value']
    assert 'possibly chocolate-chip cookies' in _restriction_variant(2564, 5)['value']
    sister = _restriction_variant(2560, 3)
    assert 'younger sister' in sister['reply'] and 'younger sister' in sister['value']
    assert 'an younger' not in sister['reply'] + sister['value']
    assert 'a younger sister' in sister['reply'] and 'a younger sister' in sister['value']
    assert 'only child' in sister['assistant']
    assert 'couple lives separately' in _restriction_variant(2560, 5)['value']


def test_restrictions_2516_2575_preserve_uncertainty_and_damaged_text():
    assert 'without naming a destination' in _restriction_variant(2519, 2)['value']
    assert 'shopping destination' in _restriction_variant(2519, 2)['assistant']
    assert 'as their opinion' in _restriction_variant(2524, 2)['value']
    assert 'poverty' in _restriction_variant(2524, 3)['value']
    assert 'format is not specified' in _restriction_variant(2528, 4)['value']
    assert _restriction_variant(2528, 2)['reply'] == 'It is free on Sundays.'
    assert 'not a completed recovery' in _restriction_variant(2529, 4)['value']
    assert 'two weeks' in _restriction_variant(2530, 11)['value']
    assert 'no photograph has yet been taken' in _restriction_variant(2530, 11)['assistant']
    assert 'conflicts with the earlier twice-weekly claim' in _restriction_variant(2534, 4)['value']
    assert 'forty yuan' in _restriction_variant(2534, 4)['value']
    assert 'uncertain report' in _restriction_variant(2543, 1)['value']
    assert 'regulation is unnamed' in _restriction_variant(2543, 1)['value']
    assert 'not a universal visa rule' in _restriction_variant(2545, 5)['value']
    assert 'Madrid' in _restriction_variant(2545, 5)['assistant']
    assert 'all adults' in _restriction_variant(2546, 1)['reply']
    assert 'literacy for all adults' in _restriction_variant(2546, 2)['assistant']
    assert 'not mean it has been achieved' in _restriction_variant(2546, 2)['value']
    assert 'volleyball fans' in _restriction_variant(2550, 9)['value']
    assert 'no country names' in _restriction_variant(2550, 9)['value']
    assert 'August sixth' in _restriction_variant(2553, 2)['value']
    assert 'not activities already completed' in _restriction_variant(2553, 10)['value']
    assert 'slightly better' in _restriction_variant(2555, 2)['value']
    assert 'rather than proof of recovery' in _restriction_variant(2555, 2)['value']
    assert 'recovery is not established' in _restriction_variant(2555, 3)['value']
    assert 'afternoon coffee' in _restriction_variant(2557, 4)['value']
    assert "aimed 3'including" in _restriction_variant(2557, 6)['reply']
    assert 'not verified medical facts' in _restriction_variant(2557, 6)['value']
    assert 'preparation is not completed' in _restriction_variant(2566, 4)['value']
    assert 'an exam' in _restriction_variant(2566, 2)['reply']
    assert 'after completing an exam' in _restriction_variant(2566, 2)['value']
    assert 'Emma' in _restriction_variant(2567, 10)['value']
    assert 'sincerity is not established' in _restriction_variant(2567, 10)['value']
    assert 'black dress' in _restriction_variant(2571, 2)['reply']
    damaged = _restriction_variant(2571, 3)
    assert 'put on car key' in damaged['reply']
    assert 'unknown and is not reconstructed as khaki' in damaged['value']


def test_batch_2576_2635_preserves_original_turns_and_qualifies_embedded_claims():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    rows = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    count = 0
    for d in range(2576, 2636):
        for t, text in enumerate(json.loads(rows[d-1]), 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert max(p['review']['evidence_turns']) <= t
            count += 1
    assert count == 408
    for key, expected in {
        '2577_4': 'deception or rejection is not established',
        '2578_6': 'handing them over is not shown',
        '2579_2': 'not a universal legal rule',
        '2580_2': 'without redefining the self-description',
        '2581_7': 'assumes the adjoining bathroom is private',
        '2581_9': 'bathroom is shared with Cindy',
        '2584_7': 'credits Zina as the expert',
        '2586_10': 'changes their decision, accepts the lift',
        '2588_5': 'not explicitly completed',
        '2589_8': 'not located in this community explicitly',
        '2590_7': "author's identity and sincerity are unverified",
        '2591_8': 'unfinished continuation does not specify that activity',
        '2592_8': 'costume alternatives, not real occupations',
        '2594_10': 'no binge is reported as occurring tonight',
        '2595_6': "rather than resolving the earlier 'next semester'",
        '2595_10': 'payment is not reported completed',
        '2599_10': "not proof that her condition caused today's absence",
        '2601_4': 'more than 120 units',
        '2602_5': 'rejection has not happened here',
        '2606_7': 'not completed transactions',
        '2609_7': 'ceremony is not reported completed',
        '2609_8': 'no slap occurred',
        '2610_5': 'supposed callback is not confirmed',
        '2613_3': 'neither his location nor safety is confirmed',
        '2616_10': 'assumes she was his daughter',
        '2616_11': 'the woman is his wife',
        '2616_12': 'this is floor 56 while their destination is floor 70',
        '2617_6': "Class Two scored seventy and Jack's team sixty-seven",
        '2619_6': 'not the book being returned',
        '2620_12': 'not an established fact about the generation',
        '2622_2': 'no realized profit is established',
        '2623_8': 'recent delivery alone does not establish freshness',
        '2625_10': 'an application, offer and job change are not completed',
        '2627_2': 'no salary is supplied',
        '2629_4': 'does not establish Olympic-level skill',
        '2632_8': 'student status is not silently replaced',
        '2633_5': 'do not report supplying a credit-card number',
        '2635_7': 'does not specify a reliable minimum number',
        '2635_8': 'likely sarcastic',
    }.items():
        assert expected in indexed['dd_review_turn_' + key]['value']
    assert indexed['dd_review_turn_2632_7']['reply'] == indexed['dd_review_turn_2632_8']['reply']
    assert 'registration is now complete' in indexed['dd_review_turn_2595_10']['value']
    assert indexed['dd_review_turn_2632_7']['value'] != indexed['dd_review_turn_2632_8']['value']
    for key, later_name in (('2582_1', 'Mary'), ('2590_1', 'Zina'), ('2595_1', 'Susan'), ('2611_1', 'Wendy'), ('2616_1', 'Anderson')):
        assert later_name not in indexed['dd_review_turn_' + key]['value']
    for key, expected in (('2592_12', 'at eleven'), ('2603_2', 'six thirty'),
                          ('2605_8', 'Italy'), ('2618_4', 'not a music examination'),
                          ('2621_2', 'prefers coffee'), ('2632_4', 'Helen Morgan')):
        p = next(p for p in PATTERNS if p['substitution_base_id'] == 'dd_review_turn_' + key)
        assert expected in realize_pattern(p, 0)['value']


def test_batch_2576_2635_covers_all_resource_tracks_and_duplicates():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_2576_2635']
    decisions = [row for key in ('sentence_attestations', 'occurrence_adaptations')
                 for row in review[key]['decisions']
                 if row['occurrence_id'].startswith('train:')
                 and 2576 <= int(row['occurrence_id'].split(':')[1]) <= 2635]
    assert len(decisions) == len({row['occurrence_id'] for row in decisions}) == 677
    assert sum(x['added_decisions'] for x in report['lexical'].values()) == 417
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 2576 <= r['dialogue'] <= 2635}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert not licensed & restricted
    assert licensed | restricted == set(range(2576, 2636))
    assert report['remaining_batch_occurrences'] == report['remaining_batch_turns'] == 0
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for row in decisions:
        pin = row['allowed_adapter']
        assert pin['sha256'] == fingerprint(indexed[pin['id']])
    duplicate = next(r for r in decisions if r['occurrence_id'] == 'train:2632:8:1')
    assert duplicate['outcome'] == 'adapted_visible_reply_only'
    lexical = pending['sources'][1]['review']['nominals.jsonl.gz']['decisions']
    student = next(r for r in lexical if r['row_index'] == 17881)
    assert student['readings'][0]['text'] == 'business student'
    assert 'PERSON' in student['readings'][0]['labels']
    fruit = next(r for r in lexical if r['row_index'] == 34453)['readings'][0]
    assert fruit['text'] == 'fruits' and fruit['number'] == 'plural'
    assert report['checks']['max_tokens'] <= 768 and report['checks']['overlength'] == []
    assert report['checks']['source_reply_truncation'] is False
    assert report['remaining_global']['unresolved_quarantine_warnings'] == 0


def test_batch_2636_2695_covers_sources_resources_and_corrected_wording():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_2636_2695']
    source = ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl'
    lines = source.read_text(encoding='utf8').splitlines()
    indexed = {p['id']: p for p in CATALOG['patterns']}
    seen = 0
    for d in range(2636, 2696):
        turns = json.loads(lines[d-1])
        for t, text in enumerate(turns, 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert p['source']['context'] == turns[:t]
            assert max(p['review']['evidence_turns']) <= t
            seen += 1
    assert seen == 510
    decisions = [r for k in ('sentence_attestations', 'occurrence_adaptations')
                 for r in review[k]['decisions'] if r['occurrence_id'].startswith('train:')
                 and 2636 <= int(r['occurrence_id'].split(':')[1]) <= 2695]
    assert len(decisions) == len({r['occurrence_id'] for r in decisions}) == 842
    for r in decisions:
        assert r['allowed_adapter']['sha256'] == fingerprint(indexed[r['allowed_adapter']['id']])
    assert sum(r['added_decisions'] for r in report['lexical'].values()) == 476
    rules = {r['dialogue'] for r in RULES if r['split'] == 'train' and 2636 <= r['dialogue'] <= 2695}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert not rules & restricted and rules | restricted == set(range(2636, 2696))
    assert len(report['source_corrections']) == 26
    for change in report['source_corrections']:
        _, d, t = change['address'].split(':')
        assert json.loads(lines[int(d)-1])[int(t)-1] == change['after']
        assert change['before'] != change['after']
        assert fingerprint(change['before']) == change['before_sha256']
        assert fingerprint(change['after']) == change['after_sha256']
    assert "how proud I am of you" in indexed['dd_review_turn_2637_12']['reply']
    assert "haven't begun" in indexed['dd_review_turn_2666_5']['reply']
    assert "There are a lot of per" in indexed['dd_review_turn_2664_10']['reply']
    assert "because I'm not..." in indexed['dd_review_turn_2650_10']['reply']
    assert report['checks']['max_tokens'] <= 768 and not report['checks']['overlength']
    assert not any(report['checks']['split_prompt_overlap'].values())


def test_batch_2636_2695_preserves_implicit_meaning_without_inventing_actions():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for identity, phrase in {
        '2636_16': 'card issuance are not reported',
        '2638_10': 'presents the money',
        '2638_11': 'confirming payment',
        '2640_4': 'not real-world participation',
        '2641_10': 'historical claim is reported',
        '2646_4': 'no present cohabitation',
        '2650_10': 'unfinished final clause',
        '2651_4': 'not a verified Gregorian date rule',
        '2653_8': 'not a reported killing',
        '2660_5': 'which parent has which nationality is unspecified',
        '2660_7': 'no birth has happened',
        '2661_12': 'successful purchase is not confirmed',
        '2663_11': 'completed reservation is not reported',
        '2664_9': 'installation has not happened',
        '2664_10': 'not completed',
        '2666_9': 'joint studying has not yet happened',
        '2668_3': 'not an actual killing by the son',
        '2675_8': 'attending a meeting is not yet reported',
        '2683_2': 'actual family relationship is not established',
        '2686_9': "not 'couldn't agree more'",
        '2687_1': 'cancellation is unconfirmed',
        '2689_12': 'does not establish that an original score was changed',
        '2691_4': 'not raising a person named Cain',
        '2693_22': 'not a claim that all those dances',
        '2695_7': 'absence is a question',
    }.items():
        assert phrase in indexed['dd_review_turn_' + identity]['value']
    for t in range(1, 9):
        a, b = (indexed[f'dd_review_turn_{d}_{t}'] for d in (2669, 2679))
        assert a['reply'] == b['reply'] and a['value'] == b['value']
        assert a['source']['dialogue'] != b['source']['dialogue']
    for identity, wanted in [('2638_7', 'eight dollars'), ('2652_1', 'Michael Green'),
                             ('2666_8', 'physics'), ('2677_1', 'bicycle')]:
        variant = next(p for p in PATTERNS if p['substitution_base_id'] == 'dd_review_turn_' + identity)
        assert wanted in realize_pattern(variant, 0)['value']


def test_batch_2696_2755_complete_coverage_and_corrected_source_evidence():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_2696_2755']
    lines = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    indexed = {p['id']: p for p in CATALOG['patterns']}
    count = 0
    for d in range(2696, 2756):
        turns = json.loads(lines[d - 1])
        for t, text in enumerate(turns, 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert p['source']['context'] == turns[:t]
            assert max(p['review']['evidence_turns']) <= t
            count += 1
    assert count == 480
    decisions = [r for k in ('sentence_attestations', 'occurrence_adaptations')
                 for r in review[k]['decisions'] if r['occurrence_id'].startswith('train:')
                 and 2696 <= int(r['occurrence_id'].split(':')[1]) <= 2755]
    assert len(decisions) == len({r['occurrence_id'] for r in decisions}) == 788
    for r in decisions:
        assert r['allowed_adapter']['sha256'] == fingerprint(indexed[r['allowed_adapter']['id']])
    assert sum(x['added_decisions'] for x in report['lexical'].values()) == 489
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 2696 <= r['dialogue'] <= 2755}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert not licensed & restricted and licensed | restricted == set(range(2696, 2756))
    assert len(report['source_corrections']) == 51
    for change in report['source_corrections']:
        _, d, t = change['address'].split(':')
        assert json.loads(lines[int(d) - 1])[int(t) - 1] == change['after']
        assert fingerprint(change['before']) == change['before_sha256']
        assert fingerprint(change['after']) == change['after_sha256']
    assert 'Henry is almost broke' in indexed['dd_review_turn_2747_5']['reply']
    assert 'Do you know' in indexed['dd_review_turn_2718_4']['reply']
    assert 'When you finish the book, I' in indexed['dd_review_turn_2718_18']['reply']
    for identity in ('train:2716:1:2', 'train:2716:1:3', 'train:2718:18:2'):
        assert next(r for r in decisions if r['occurrence_id'] == identity)['outcome'] == 'adapted_visible_reply_only'
    assert report['checks']['max_tokens'] <= 768 and not report['checks']['overlength']
    assert not any(report['checks']['split_prompt_overlap'].values())


def test_batch_2696_2755_avoids_fiction_intention_and_identity_leaks():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for identity, phrase in {
        '2696_7': "identity has not yet been established",
        '2697_2': 'finishing the report is not explicitly stated',
        '2699_4': 'motive is speculation',
        '2703_8': 'unanswered rather than accepted',
        '2704_5': 'no objective personality traits',
        '2708_3': 'speaker boundaries inside the turn are unclear',
        '2716_2': 'no attack is reported',
        '2717_12': 'does not confirm the book was checked out',
        '2718_12': 'not a legal rule',
        '2720_4': 'does not report an actual search',
        '2721_8': 'from ownership',
        '2724_11': 'unfinished final clause',
        '2725_12': 'purchasing is completed',
        '2725_13': 'actual disposal or departure is not reported',
        '2728_8': 'does not establish the student has been absent',
        '2730_17': 'not explicitly declined the teaching offer',
        '2731_11': 'fictional events and conditional purposes',
        '2733_8': 'precise venue name is not normalized',
        '2735_5': "individual nationalities are not assigned",
        '2739_4': 'no ingredient is reconstructed',
        '2740_6': 'oil discovery or successful extraction is not established',
        '2745_6': 'not independently observed neglect',
        '2747_1': 'unnamed man',
        '2748_16': 'tickets are not yet booked',
        '2750_23': 'cooking and arrival are future',
        '2751_10': 'no age is invented',
        '2753_4': 'enrollment or graduation from Yale is not completed',
        '2754_11': 'future flight remains separate',
        '2755_7': "feelings remain the speaker's inference",
    }.items():
        assert phrase in indexed['dd_review_turn_' + identity]['value']
    assert 'recorded music' not in indexed['dd_review_turn_2724_11']['value']
    for identity, phrase in [('2703_2', 'David'), ('2717_12', 'novel'),
                             ('2747_5', 'Robert'), ('2751_6', 'dark brown')]:
        variant = next(p for p in PATTERNS if p['substitution_base_id'] == 'dd_review_turn_' + identity)
        assert phrase in realize_pattern(variant, 0)['value']


def test_batch_2756_2815_covers_sources_candidates_and_dictionaries():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_2756_2815']
    lines = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    indexed = {p['id']: p for p in CATALOG['patterns']}
    count = 0
    for d in range(2756, 2816):
        turns = json.loads(lines[d - 1])
        for t, text in enumerate(turns, 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert p['source']['context'] == turns[:t]
            assert max(p['review']['evidence_turns']) <= t
            count += 1
    assert count == 381
    decisions = [r for k in ('sentence_attestations', 'occurrence_adaptations')
                 for r in review[k]['decisions'] if r['occurrence_id'].startswith('train:')
                 and 2756 <= int(r['occurrence_id'].split(':')[1]) <= 2815]
    assert len(decisions) == len({r['occurrence_id'] for r in decisions}) == 644
    for r in decisions:
        assert r['allowed_adapter']['sha256'] == fingerprint(indexed[r['allowed_adapter']['id']])
    assert sum(x['added_decisions'] for x in report['lexical'].values()) == 444
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 2756 <= r['dialogue'] <= 2815}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert not licensed & restricted and licensed | restricted == set(range(2756, 2816))
    assert len(report['source_corrections']) == 44
    for change in report['source_corrections']:
        _, d, t = change['address'].split(':')
        assert json.loads(lines[int(d) - 1])[int(t) - 1] == change['after']
        assert fingerprint(change['before']) == change['before_sha256']
        assert fingerprint(change['after']) == change['after_sha256']
    assert "firecrackers aren't toys. They're dangerous!" in indexed['dd_review_turn_2779_4']['reply']
    assert 'vote in some new leaders' in indexed['dd_review_turn_2772_5']['reply']
    assert indexed['dd_review_turn_2766_1']['reply'].endswith('the battle of')
    assert indexed['dd_review_turn_2766_4']['reply'].endswith('transformed British')
    for identity in ('train:2766:1:4', 'train:2766:4:3', 'train:2788:7:1', 'train:2788:7:2',
                     'train:2812:8:1', 'train:2812:8:2', 'train:2812:8:3'):
        assert next(r for r in decisions if r['occurrence_id'] == identity)['outcome'] == 'adapted_visible_reply_only'
    assert report['checks']['max_tokens'] <= 768 and not report['checks']['overlength']
    assert not any(report['checks']['split_prompt_overlap'].values())


def test_batch_2756_2815_preserves_retractions_idioms_and_completion():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for identity, phrase in {
        '2756_1': 'unverified rumor',
        '2756_2': 'not completed integration',
        '2756_3': 'do not establish job losses',
        '2760_4': 'not the Ritz Carlton',
        '2760_6': 'not a literal identity claim',
        '2761_3': 'inheritance is not reported completed',
        '2763_5': 'established only now',
        '2766_2': 'not a verified burial-location fact',
        '2766_10': 'wax representations',
        '2770_7': 'no ski trip is inferred',
        '2770_9': 'not a completed marriage',
        '2772_5': 'no election or removal has occurred',
        '2774_9': 'not literally receiving twice as many banknotes',
        '2774_13': 'no new reservation, move or payment is completed',
        '2778_3': 'no literal words',
        '2779_7': 'cause remains disputed',
        '2779_11': 'not yet received or completed',
        '2787_8': 'not independently confirmed',
        '2787_14': 'earlier comments as teasing',
        '2787_16': 'retracting the earlier negative characterization',
        '2789_5': 'loan is being issued',
        '2789_7': 'renewal is conditional and has not happened',
        '2793_2': 'no pearls, pigs or actual throwing',
        '2803_9': 'no hiring decision',
        '2805_3': 'not recast as a clear claim',
        '2808_3': 'does not establish a second visit',
        '2808_9': 'not ten completed days of rest',
        '2812_8': 'not verified counts or a computed marriage rate',
        '2812_10': 'wedding is future',
        '2815_3': 'no literal burn',
    }.items():
        assert phrase in indexed['dd_review_turn_' + identity]['value']
    assert 'teasing' not in indexed['dd_review_turn_2787_8']['value']
    for identity, phrase in [('2763_4', 'Michael Miller'), ('2787_16', 'David'),
                             ('2789_5', 'novels'), ('2799_2', 'row thirteen')]:
        variant = next(p for p in PATTERNS if p['substitution_base_id'] == 'dd_review_turn_' + identity)
        assert phrase in realize_pattern(variant, 0)['value']


def test_batch_2816_2875_full_coverage_and_source_repairs():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_2816_2875']
    lines = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    indexed = {p['id']: p for p in CATALOG['patterns']}
    count = 0
    for d in range(2816, 2876):
        turns = json.loads(lines[d - 1])
        for t, text in enumerate(turns, 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert p['source']['context'] == turns[:t]
            assert max(p['review']['evidence_turns']) <= t
            count += 1
    assert count == 471
    decisions = [r for k in ('sentence_attestations', 'occurrence_adaptations')
                 for r in review[k]['decisions'] if r['occurrence_id'].startswith('train:')
                 and 2816 <= int(r['occurrence_id'].split(':')[1]) <= 2875]
    assert len(decisions) == len({r['occurrence_id'] for r in decisions}) == 772
    for r in decisions:
        assert r['allowed_adapter']['sha256'] == fingerprint(indexed[r['allowed_adapter']['id']])
    assert sum(x['added_decisions'] for x in report['lexical'].values()) == 520
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 2816 <= r['dialogue'] <= 2875}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert not licensed & restricted and licensed | restricted == set(range(2816, 2876))
    assert len(report['source_corrections']) == 54
    for change in report['source_corrections']:
        _, d, t = change['address'].split(':')
        assert json.loads(lines[int(d) - 1])[int(t) - 1] == change['after']
        assert fingerprint(change['before']) == change['before_sha256']
        assert fingerprint(change['after']) == change['after_sha256']
    assert "six months' salary" in indexed['dd_review_turn_2818_2']['reply']
    assert "Yes, I have two of her CDs, I'll lend" in indexed['dd_review_turn_2826_8']['reply']
    assert 'more. That' in indexed['dd_review_turn_2875_4']['reply']
    assert indexed['dd_review_turn_2831_8']['reply'].endswith('writing tha')
    assert 'transom' in indexed['dd_review_turn_2841_4']['reply']
    assert 'now just sink' in indexed['dd_review_turn_2868_3']['reply']
    for identity in ('train:2819:6:6', 'train:2822:2:1', 'train:2822:2:2',
                     'train:2831:8:3', 'train:2841:2:3', 'train:2853:6:1', 'train:2868:3:2'):
        assert next(r for r in decisions if r['occurrence_id'] == identity)['outcome'] == 'adapted_visible_reply_only'
    assert report['checks']['max_tokens'] <= 768 and not report['checks']['overlength']
    assert not any(report['checks']['split_prompt_overlap'].values())
    lexical = pending['sources'][1]['review']['nominals.jsonl.gz']['decisions']
    for raw, expected_label in [('Flares', 'PHYSICAL_OBJECT'), ('cholesterol', 'PHYSICAL_OBJECT')]:
        rows = [r for r in lexical if r['text'] == raw and any(a.startswith(('train:2854:', 'train:2869:')) for a in r['evidence'])]
        assert rows and all(expected_label in reading['labels'] for r in rows for reading in r['readings'])


def test_batch_2816_2875_preserves_counterfactuals_denials_and_action_stages():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for identity, phrase in {
        '2816_4': 'not an established property',
        '2816_5': 'rejects that stereotype',
        '2817_2': 'forgiveness are not yet established',
        '2818_6': 'repossession is threatened, not completed',
        '2819_8': 'stopping short of actual finger loss',
        '2821_5': 'attendance and graduation are not established',
        '2822_4': 'opening is completed',
        '2823_13': 'successful gifts or autographs are not established',
        '2825_6': 'not assigned to each of the first three',
        '2826_1': 'playing cards, not rulers',
        '2826_8': 'lending is not completed',
        '2827_6': 'did not make her present',
        '2827_13': 'supernatural hearing is not an established fact',
        '2830_6': 'not independently verified current exam specifications',
        '2831_9': 'no advisor identity',
        '2832_6': 'counterfactual, not completed',
        '2832_8': 'accident is reported as real',
        '2834_5': 'corrects the plan',
        '2835_2': 'no literal heat or cooking',
        '2836_2': 'not his current age',
        '2840_2': 'hands over or presents a pen',
        '2840_5': 'not a specific monetary debt',
        '2842_8': 'completed move or divorce is not established',
        '2844_12': "bachelor's degree is not completed",
        '2847_3': 'no proposal or engagement established',
        '2847_16': 'rejects that broad conclusion',
        '2850_8': 'action underway',
        '2850_9': 'recycling is not yet completed',
        '2854_9': 'clothing, not signal flares',
        '2856_2': 'does not mean she died',
        '2859_15': 'they do want repair',
        '2861_4': 'signing and enrollment are not yet confirmed',
        '2862_8': 'no age is supplied',
        '2865_5': 'physical receipt or purchase is not explicitly confirmed',
        '2866_7': "uncle's ownership is not the speaker's ownership",
        '2867_6': 'activation did not establish successful operation',
        '2868_5': 'no definite date',
        '2869_9': 'not achieved loss or a safe recommended rate',
        '2872_7': 'not accepting the purchase suggestion',
        '2874_4': 'future return remains uncompleted',
        '2875_4': 'sweeping attributed claims',
    }.items():
        assert phrase in indexed['dd_review_turn_' + identity]['value']
    assert 'twenty-nine' not in indexed['dd_review_turn_2836_1']['value']
    for identity, phrase in [('2836_2', 'thirty-one'), ('2840_2', 'pencil'),
                             ('2849_4', 'Robert'), ('2867_4', 'water pump')]:
        variant = next(p for p in PATTERNS if p['substitution_base_id'] == 'dd_review_turn_' + identity)
        assert phrase in realize_pattern(variant, 0)['value']


def test_batch_2876_2935_covers_sources_candidates_and_lexical_evidence():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_2876_2935']
    lines = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    indexed = {p['id']: p for p in CATALOG['patterns']}
    count = 0
    for d in range(2876, 2936):
        turns = json.loads(lines[d - 1])
        for t, text in enumerate(turns, 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert p['source']['context'] == turns[:t]
            assert p['review']['source_sha256'] == fingerprint(text)
            assert p['review']['context_sha256'] == fingerprint(turns[:t])
            assert max(p['review']['evidence_turns']) <= t
            count += 1
    assert count == report['turns'] == 468
    decisions = [r for k in ('sentence_attestations', 'occurrence_adaptations')
                 for r in review[k]['decisions'] if r['occurrence_id'].startswith('train:')
                 and 2876 <= int(r['occurrence_id'].split(':')[1]) <= 2935]
    assert len(decisions) == len({r['occurrence_id'] for r in decisions}) == 771
    assert sum(r['outcome'] == 'attested_sentence_only' for r in decisions) == 614
    for r in decisions:
        assert r['allowed_adapter']['sha256'] == fingerprint(indexed[r['allowed_adapter']['id']])
    assert sum(x['added_decisions'] for x in report['lexical'].values()) == 525
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 2876 <= r['dialogue'] <= 2935}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert len(licensed) == 29 and len(restricted) == 31
    assert not licensed & restricted and licensed | restricted == set(range(2876, 2936))
    for change in report['source_corrections']:
        _, d, t = change['address'].split(':')
        assert json.loads(lines[int(d) - 1])[int(t) - 1] == change['after']
        assert fingerprint(change['before']) == change['before_sha256']
        assert fingerprint(change['after']) == change['after_sha256']
    assert 'The jeweller told' in indexed['dd_review_turn_2910_7']['reply']
    assert 'mountain streams' in indexed['dd_review_turn_2910_7']['reply']
    assert 'idea. She' in indexed['dd_review_turn_2931_12']['reply']
    assert 'set up a company' in indexed['dd_review_turn_2899_4']['reply']
    for identity in ('train:2890:6:1', 'train:2900:6:4', 'train:2900:7:3',
                     'train:2901:5:3', 'train:2904:2:1', 'train:2904:2:2',
                     'train:2912:8:2', 'train:2925:1:2', 'train:2925:1:4'):
        assert next(r for r in decisions if r['occurrence_id'] == identity)['outcome'] == 'adapted_visible_reply_only'
    assert indexed['dd_review_turn_2912_8']['reply'].endswith('we c')
    assert 'N-A-N-J-U' in indexed['dd_review_turn_2915_6']['reply']
    assert 'Nandu' in indexed['dd_review_turn_2915_4']['reply']
    assert report['checks']['max_tokens'] <= 768 and not report['checks']['overlength']
    assert not any(report['checks']['split_prompt_overlap'].values())
    inventories = pending['sources'][1]['review']
    # Punctuation fragments and named-entity mistakes cannot leak into fillers.
    for name, indices in [('nominals.jsonl.gz', (36767, 13230, 37810, 70841)),
                          ('entity_candidates.jsonl.gz', (3187, 6326, 6542, 12759))]:
        rows = {r['row_index']: r for r in inventories[name]['decisions']}
        assert all(rows[i]['outcome'] == 'rejected' for i in indices)


def test_batch_2876_2935_preserves_ambiguity_attribution_and_action_stages():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for identity, phrase in {
        '2876_4': 'final shoe color remains unspecified',
        '2878_10': 'Tickets are not yet transferred',
        '2879_2': 'series title',
        '2880_12': 'not endorsed',
        '2880_13': 'does not make 1 prime',
        '2884_1': 'not report literal heat or an explosion',
        '2886_1': 'electricity has not yet been cut',
        '2890_6': 'does not identify an exact topic',
        '2890_10': 'no withdrawal has happened',
        '2891_2': 'does not identify which',
        '2892_3': 'reported allegation',
        '2893_9': 'forgiveness is not reported completed',
        '2895_5': 'counterfactual',
        '2898_5': 'fulfillment are unknown',
        '2899_4': 'not selling an existing personal company',
        '2900_7': 'damaged final phrase',
        '2901_13': 'booking is complete but travel and the stay are not',
        '2902_6': 'no replacement booking or compensation',
        '2903_6': 'exact identity is not invented',
        '2904_1': 'no job offer or acceptance',
        '2905_10': 'damaged double-negative',
        '2908_4': 'past events in that order',
        '2908_10': 'no payment has yet occurred',
        '2909_10': 'receipt or drinking is not established',
        '2910_5': 'no discovery is not proof of no coal',
        '2911_4': 'father, not the son',
        '2912_8': 'has not accepted the job',
        '2913_8': 'not completed cessation',
        '2915_6': 'conflicting with Nandu',
        '2917_6': 'conditional on missing the class',
        '2919_5': 'no promotion is established',
        '2922_3': 'resignation has not occurred',
        '2925_1': 'do not support definite measurements',
        '2926_6': "does not establish any patron's orientation",
        '2926_21': 'both branches are conditional and uncompleted',
        '2926_22': 'entry has not yet happened',
        '2928_2': 'already marinated',
        '2928_5': 'not delivery',
        '2929_2': 'neither death nor recovery is established',
        '2931_8': 'prior reading of this book remains unknown',
        '2932_2': 'advice is not a requirement',
        '2933_12': 'neither misconduct nor divorce is established',
        '2934_1': 'no animal identity is asserted',
        '2935_9': 'nor completed ride is established',
    }.items():
        assert phrase in indexed['dd_review_turn_' + identity]['value']
    assert 'Sally' not in indexed['dd_review_turn_2890_3']['value']
    assert 'N-A-N-J-U' not in indexed['dd_review_turn_2915_4']['value']
    for identity, phrase in [('2881_1', 'bicycle'), ('2889_2', 'basketball'),
                             ('2909_10', 'Paul'), ('2926_3', "Paul's Bar")]:
        variant = next(p for p in PATTERNS if p['substitution_base_id'] == 'dd_review_turn_' + identity)
        assert phrase in realize_pattern(variant, 0)['value']


def test_batch_2936_2995_full_coverage_source_repairs_and_restrictions():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_2936_2995']
    lines = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    indexed = {p['id']: p for p in CATALOG['patterns']}
    count = 0
    for d in range(2936, 2996):
        turns = json.loads(lines[d - 1])
        for t, text in enumerate(turns, 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert p['source']['context'] == turns[:t]
            assert p['review']['source_sha256'] == fingerprint(text)
            assert p['review']['context_sha256'] == fingerprint(turns[:t])
            assert max(p['review']['evidence_turns']) <= t
            count += 1
    assert count == report['turns'] == 446
    decisions = [r for k in ('sentence_attestations', 'occurrence_adaptations')
                 for r in review[k]['decisions'] if r['occurrence_id'].startswith('train:')
                 and 2936 <= int(r['occurrence_id'].split(':')[1]) <= 2995]
    assert len(decisions) == len({r['occurrence_id'] for r in decisions}) == 720
    assert sum(r['outcome'] == 'attested_sentence_only' for r in decisions) == 556
    for r in decisions:
        assert r['allowed_adapter']['sha256'] == fingerprint(indexed[r['allowed_adapter']['id']])
    assert sum(x['added_decisions'] for x in report['lexical'].values()) == 389
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 2936 <= r['dialogue'] <= 2995}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert len(licensed) == 32 and len(restricted) == 28
    assert not licensed & restricted and licensed | restricted == set(range(2936, 2996))
    assert len(report['source_corrections']) == 51
    for change in report['source_corrections']:
        _, d, t = change['address'].split(':')
        assert json.loads(lines[int(d) - 1])[int(t) - 1] == change['after']
        assert fingerprint(change['before']) == change['before_sha256']
        assert fingerprint(change['after']) == change['after_sha256']
    assert 'Professor Donald. This' in indexed['dd_review_turn_2964_8']['reply']
    assert "It's me, Susan Taylor." == indexed['dd_review_turn_2991_2']['reply']
    assert 'quiet and reserved' in indexed['dd_review_turn_2969_5']['reply']
    assert 'could use some help' in indexed['dd_review_turn_2973_4']['reply']
    assert 'Stars.com' in indexed['dd_review_turn_2943_2']['reply']
    assert indexed['dd_review_turn_2994_6']['reply'].endswith('Ho')
    assert indexed['dd_review_turn_2994_7']['reply'].endswith("Dad's getting")
    for identity in ('train:2943:2:1', 'train:2943:2:2', 'train:2964:3:1',
                     'train:2991:2:1', 'train:2991:2:2', 'train:2994:6:6', 'train:2994:7:4'):
        assert next(r for r in decisions if r['occurrence_id'] == identity)['outcome'] == 'adapted_visible_reply_only'
    assert report['checks']['max_tokens'] <= 768 and not report['checks']['overlength']
    assert not any(report['checks']['split_prompt_overlap'].values())
    inventories = pending['sources'][1]['review']
    for name, indices in [('nominals.jsonl.gz', (4553, 6072, 24936, 42992, 71462)),
                          ('entity_candidates.jsonl.gz', (3724, 4710, 4923, 12841))]:
        rows = {r['row_index']: r for r in inventories[name]['decisions']}
        assert all(rows[i]['outcome'] == 'rejected' for i in indices)


def test_batch_2936_2995_context_does_not_turn_proposals_or_rumors_into_facts():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for identity, phrase in {
        '2936_14': 'gender stereotype',
        '2938_6': 'commencement of work is not stated',
        '2939_2': 'not holding a burning object',
        '2940_4': 'firmly refuses',
        '2941_6': 'not independently confirmed',
        '2942_3': 'not silently converted',
        '2943_4': 'without specifying them',
        '2945_9': 'remains undisclosed',
        '2947_10': 'not necessarily the previously named novel',
        '2947_13': 'separate renewal',
        '2949_15': 'called to cancel',
        '2950_4': 'not permission for this evening',
        '2952_3': 'intoxication are not specified',
        '2952_8': 'arrest within the fictional scene',
        '2955_2': 'not resolved by this reply alone',
        '2955_4': 'no dismissal or job loss is established',
        '2956_8': 'no sale or payment occurs',
        '2957_8': 'without deciding to abandon train travel',
        '2960_4': 'does not report actual death',
        '2962_4': 'potentially ironic',
        '2963_3': 'both are currently nine',
        '2964_7': 'permission is not completed enrollment',
        '2966_6': 'without firmly committing',
        '2967_2': 'not entirely',
        '2968_8': 'enrollment are not established',
        '2969_7': 'does not establish any specific future assistance',
        '2970_6': 'recovery did not occur',
        '2971_7': 'agreed but not completed',
        '2974_4': 'no sale or loss is completed',
        '2975_1': "speaker's payment is complete",
        '2975_2': 'has not paid',
        '2978_3': 'not a conviction or proven guilt',
        '2979_2': 'not a denial',
        '2980_9': 'arrival at the hotel is not established',
        '2982_8': 'forecast, not elapsed',
        '2983_6': 'not full fluency',
        '2984_4': 'not a reported foul',
        '2986_4': 'liking the performance but disliking the story',
        '2987_8': 'do not establish that the harassment report was false',
        '2987_13': 'does not prove the allegation',
        '2988_7': 'receipt is complete but attendance is not',
        '2991_7': 'lamp is not acquired yet',
        '2992_12': 'no award or eligibility is established',
        '2993_6': 'expressing a stereotype',
        '2994_4': 'not public offices',
        '2994_7': 'no reliable additional event',
        '2994_17': 'acceptance remains tentative',
        '2995_5': 'does not deny all gaming',
    }.items():
        assert phrase in indexed['dd_review_turn_' + identity]['value']
    assert 'cold' not in indexed['dd_review_turn_2955_2']['value']
    assert 'Steven' not in indexed['dd_review_turn_2941_7']['value']
    assert 'four' not in indexed['dd_review_turn_2994_10']['value']
    for identity, phrase in [('2938_2', 'Reading'), ('2970_4', 'backpack'),
                             ('2980_7', 'Park Hotel'), ('2981_1', 'Saturday'),
                             ('2991_2', 'Helen Carter')]:
        variant = next(p for p in PATTERNS if p['substitution_base_id'] == 'dd_review_turn_' + identity)
        assert phrase in realize_pattern(variant, 0)['value']


def test_batch_2996_3055_full_coverage_and_preserved_uncertainty():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_2996_3055']
    lines = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    indexed = {p['id']: p for p in CATALOG['patterns']}
    count = 0
    for d in range(2996, 3056):
        turns = json.loads(lines[d - 1])
        for t, text in enumerate(turns, 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert p['source']['context'] == turns[:t]
            assert p['review']['source_sha256'] == fingerprint(text)
            assert p['review']['context_sha256'] == fingerprint(turns[:t])
            assert max(p['review']['evidence_turns']) <= t
            count += 1
    assert count == report['turns'] == 428
    decisions = [r for k in ('sentence_attestations', 'occurrence_adaptations')
                 for r in review[k]['decisions'] if r['occurrence_id'].startswith('train:')
                 and 2996 <= int(r['occurrence_id'].split(':')[1]) <= 3055]
    assert len(decisions) == len({r['occurrence_id'] for r in decisions}) == 693
    assert sum(r['outcome'] == 'attested_sentence_only' for r in decisions) == 570
    for r in decisions:
        assert r['allowed_adapter']['sha256'] == fingerprint(indexed[r['allowed_adapter']['id']])
    assert sum(x['added_decisions'] for x in report['lexical'].values()) == 339
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 2996 <= r['dialogue'] <= 3055}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert len(licensed) == 30 and len(restricted) == 30
    assert not licensed & restricted and licensed | restricted == set(range(2996, 3056))
    assert len(report['source_corrections']) == 42
    for c in report['source_corrections']:
        _, d, t = c['address'].split(':')
        assert json.loads(lines[int(d)-1])[int(t)-1] == c['after']
        assert fingerprint(c['before']) == c['before_sha256']
        assert fingerprint(c['after']) == c['after_sha256']
    assert 'small dessert' in indexed['dd_review_turn_3000_6']['reply']
    assert indexed['dd_review_turn_3000_8']['reply'].endswith('try')
    assert indexed['dd_review_turn_3007_3']['reply'].endswith('loans')
    assert 'some wreckers' in indexed['dd_review_turn_3032_1']['reply']
    assert 'stomium' in indexed['dd_review_turn_3038_13']['reply']
    assert 'Zheng He' in indexed['dd_review_turn_3047_4']['reply']
    assert 'Jim' not in indexed['dd_review_turn_3037_1']['value']
    assert 'father' not in indexed['dd_review_turn_3047_1']['value']
    for identity in ('train:2999:7:2', 'train:3000:8:3', 'train:3007:3:4', 'train:3009:21:2',
                     'train:3014:4:1', 'train:3032:1:2', 'train:3038:13:4', 'train:3050:4:1'):
        assert next(r for r in decisions if r['occurrence_id'] == identity)['outcome'] == 'adapted_visible_reply_only'
    assert report['checks']['max_tokens'] <= 768 and not report['checks']['overlength']
    assert not any(report['checks']['split_prompt_overlap'].values())
    inventories = pending['sources'][1]['review']
    for name, indices in [('nominals.jsonl.gz', (17658, 28136, 32519, 35713, 50055, 54186, 61512, 66471)),
                          ('entity_candidates.jsonl.gz', (5229, 8987, 10292, 10409))]:
        rows = {r['row_index']: r for r in inventories[name]['decisions']}
        assert all(rows[i]['outcome'] == 'rejected' for i in indices)


def test_batch_2996_3055_negation_sarcasm_and_action_stage_survive_variants():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for identity, phrase in {
        '2999_6': 'not a verified count',
        '3000_8': 'missing recommendation is not invented',
        '3001_1': 'enrollment or attendance is not yet established',
        '3002_6': 'undecided, not physical location',
        '3003_9': 'not a report of physical illness',
        '3005_2': 'figurative',
        '3007_6': 'transfer is not established',
        '3008_4': 'intention is not an actual prior request',
        '3009_20': 'no exercise or purchase is completed',
        '3009_21': 'not inferred from a later turn',
        '3010_2': 'indifference',
        '3012_8': 'tasting it is not stated',
        '3013_3': 'no invitation is confirmed',
        '3014_7': 'whether this means father or husband is unresolved',
        '3016_4': 'not four biological children',
        '3017_2': 'no cure has happened',
        '3018_9': 'no exchange or move has occurred',
        '3021_2': 'not a literal request for repetition',
        '3022_3': 'onset of light rain',
        '3024_10': 'leave has not been granted',
        '3025_11': 'collection has not occurred',
        '3026_8': 'future three days have not elapsed',
        '3027_4': 'does not prove the speaker lacks knowledge',
        '3029_2': 'morning or evening is unspecified',
        '3030_10': 'speculation, not established',
        '3032_4': 'object identity remains uncertain',
        '3033_9': "no winner's identity is disclosed",
        '3034_3': 'not that the debt is forgiven',
        '3038_13': 'missing components',
        '3040_3': 'no collision is said to have happened',
        '3041_2': 'not a completed killing',
        '3041_4': 'praise is sarcastic',
        '3041_8': 'refuses to return',
        '3044_3': 'accusation is not proven',
        '3046_7': 'consequences are hypothetical',
        '3047_12': 'not a literal sea voyage',
        '3047_16': 'permission and the time are not yet supplied',
        '3048_8': 'no offer, hiring or job acceptance',
        '3049_2': 'status, not a seasoning location',
        '3050_3': 'per-pursuit breakdown is not given',
        '3051_9': 'reading is not completed',
        '3053_4': 'no personal name is given',
        '3054_6': 'does not by itself prove a romantic affair',
        '3054_8': 'confrontation did not occur',
        '3054_11': 'regret is not confirmed',
        '3055_8': 'simple but functional studio',
    }.items():
        assert phrase in indexed['dd_review_turn_' + identity]['value']
    for identity, phrase in [('3001_1', 'Paul'), ('3012_3', 'chocolate'),
                             ('3015_1', 'Canada'), ('3025_10', 'Li Ming'),
                             ('3026_8', 'two days'), ('3042_1', 'jacket')]:
        variant = next(p for p in PATTERNS if p['substitution_base_id'] == 'dd_review_turn_' + identity)
        assert phrase in realize_pattern(variant, 0)['value']


def test_batch_3056_3115_source_coverage_and_evidence_boundaries():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_3056_3115']
    lines = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    indexed = {p['id']: p for p in CATALOG['patterns']}
    count = 0
    for d in range(3056, 3116):
        turns = json.loads(lines[d - 1])
        for t, text in enumerate(turns, 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert p['source']['context'] == turns[:t]
            assert p['review']['source_sha256'] == fingerprint(text)
            assert p['review']['context_sha256'] == fingerprint(turns[:t])
            assert max(p['review']['evidence_turns']) <= t
            count += 1
    assert count == report['turns'] == 470
    decisions = [r for k in ('sentence_attestations', 'occurrence_adaptations')
                 for r in review[k]['decisions'] if r['occurrence_id'].startswith('train:')
                 and 3056 <= int(r['occurrence_id'].split(':')[1]) <= 3115]
    assert len(decisions) == len({r['occurrence_id'] for r in decisions}) == 782
    assert sum(r['outcome'] == 'attested_sentence_only' for r in decisions) == 635
    for r in decisions:
        assert r['allowed_adapter']['sha256'] == fingerprint(indexed[r['allowed_adapter']['id']])
    assert sum(x['added_decisions'] for x in report['lexical'].values()) == 473
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 3056 <= r['dialogue'] <= 3115}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert len(licensed) == 41 and len(restricted) == 19
    assert not licensed & restricted and licensed | restricted == set(range(3056, 3116))
    assert len(report['source_corrections']) == 65
    for change in report['source_corrections']:
        _, d, t = change['address'].split(':')
        assert json.loads(lines[int(d) - 1])[int(t) - 1] == change['after']
        assert fingerprint(change['before']) == change['before_sha256']
        assert fingerprint(change['after']) == change['after_sha256']
    for identity, phrase in [('3061_4', 'old milk'), ('3076_3', 'big face down the eve'),
                             ('3076_5', 'mystery belief serve'), ('3100_8', 'Howard sounds'),
                             ('3109_2', 'Songhua University')]:
        assert phrase in indexed['dd_review_turn_' + identity]['reply']
    for identity in ('train:3062:16:1', 'train:3065:5:3', 'train:3066:3:2',
                     'train:3072:18:1', 'train:3076:3:1', 'train:3078:4:1',
                     'train:3079:7:2', 'train:3099:2:1', 'train:3100:17:2',
                     'train:3103:7:1', 'train:3111:11:3'):
        assert next(r for r in decisions if r['occurrence_id'] == identity)['outcome'] == 'adapted_visible_reply_only'
    assert report['checks']['max_tokens'] <= 768 and not report['checks']['overlength']
    assert not any(report['checks']['split_prompt_overlap'].values())
    inventories = pending['sources'][1]['review']
    for name, indices in [('adverbs.jsonl.gz', (770,)),
                          ('entity_candidates.jsonl.gz', (2543, 3586, 4822, 7395, 9843, 9934, 11175)),
                          ('nominals.jsonl.gz', (17149, 20544, 20845, 30402, 37155, 39528,
                                                53916, 61327, 64297, 66317, 69324))]:
        rows = {r['row_index']: r for r in inventories[name]['decisions']}
        assert all(rows[i]['outcome'] == 'rejected' for i in indices)


def test_batch_3056_3115_illness_rumors_proposals_and_idioms_are_not_events():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for identity, phrase in {
        '3056_3': 'collection has not occurred',
        '3056_9': 'no actual collapse or match result is asserted',
        '3057_4': 'does not confirm cheaper evening seats',
        '3058_7': 'remaining length, not a specified number of inches to remove',
        '3060_1': 'no submitted resignation is established by this opening alone',
        '3060_3': 'submission is completed, departure is not yet established',
        '3060_5': 'starting work there is future',
        '3061_4': 'unclear ingredient is preserved',
        '3062_14': 'not independently established cures',
        '3063_4': 'return to teaching is conditional',
        '3064_4': "Femi's feelings are an inference",
        '3065_3': 'expresses a wish to provide lifelong care',
        '3065_6': 'explicitly accepts the marriage proposal',
        '3065_9': 'placement is not explicitly completed',
        '3066_5': 'not actual drug use or a literal plan',
        '3067_8': 'spirits are not asserted as real',
        '3068_10': 'no ride has happened',
        '3070_17': 'delay or disappearance has not happened',
        '3071_8': 'not independently verified current policy',
        '3072_12': 'does not report a real death',
        '3073_10': 'reservations have not been made',
        '3074_8': 'no royal identity',
        '3075_5': 'outage is an actual reported change',
        '3076_3': 'not turned into an invented event or menu',
        '3078_4': 'recovery has not happened',
        '3079_15': 'direction of the visit is not yet given',
        '3080_7': 'victory is not established',
        '3081_3': 'not literal food preparation',
        '3083_2': 'hearsay, not a confirmed diagnosis',
        '3083_3': 'neither diagnosis is verified',
        '3084_10': 'no acceptance or enrollment',
        '3087_3': 'not proof he has done no job searching',
        '3088_1': 'loss is not certain',
        '3089_6': 'not a verified publication schedule',
        '3091_7': 'catching is now completed',
        '3091_8': 'does not create an engagement',
        '3092_5': 'no gift has been bought or given',
        '3096_4': 'no resignation has occurred',
        '3099_9': 'origin and current study location differ',
        '3100_4': 'distinct from the interlocutor Bob',
        '3100_12': 'total chapter count is not yet known',
        '3101_1': 'no actual score or whipping is established',
        '3102_12': 'no transfer is planned now',
        '3103_7': 'without claiming success or a job offer',
        '3104_8': 'call has not happened',
        '3105_4': 'idiom does not establish a murder',
        '3108_6': 'does not independently verify the report',
        '3109_2': 'unusual name is preserved',
        '3110_4': 'N-O-I-R-I-N',
        '3111_13': 'do not establish sustained abstinence',
        '3112_4': 'does not prove he never works late',
        '3113_8': 'effectiveness or implementation is not established',
        '3114_2': 'wording says presence',
        '3115_4': 'no award has been won',
        '3115_8': 'tragic outcome is not confirmed',
    }.items():
        assert phrase in indexed['dd_review_turn_' + identity]['value']
    assert 'Diane' not in indexed['dd_review_turn_3107_1']['value']
    for identity, phrase in [('3060_5', 'ABC Company'), ('3068_6', 'Paul'),
                             ('3073_10', 'Paul'), ('3081_3', 'Paul'),
                             ('3097_1', 'German'), ('3109_2', 'Oakfield University')]:
        variant = next(p for p in PATTERNS if p['substitution_base_id'] == 'dd_review_turn_' + identity)
        assert phrase in realize_pattern(variant, 0)['value']


def test_batch_3116_3175_source_coverage_and_preserved_incomplete_replies():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_3116_3175']
    lines = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    indexed = {p['id']: p for p in CATALOG['patterns']}
    count = 0
    for d in range(3116, 3176):
        turns = json.loads(lines[d - 1])
        for t, text in enumerate(turns, 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert p['source']['context'] == turns[:t]
            assert p['review']['source_sha256'] == fingerprint(text)
            assert p['review']['context_sha256'] == fingerprint(turns[:t])
            assert max(p['review']['evidence_turns']) <= t
            count += 1
    assert count == report['turns'] == 428
    decisions = [r for k in ('sentence_attestations', 'occurrence_adaptations')
                 for r in review[k]['decisions'] if r['occurrence_id'].startswith('train:')
                 and 3116 <= int(r['occurrence_id'].split(':')[1]) <= 3175]
    assert len(decisions) == len({r['occurrence_id'] for r in decisions}) == 701
    assert sum(r['outcome'] == 'attested_sentence_only' for r in decisions) == 566
    for r in decisions:
        assert r['allowed_adapter']['sha256'] == fingerprint(indexed[r['allowed_adapter']['id']])
    assert sum(x['added_decisions'] for x in report['lexical'].values()) == 372
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 3116 <= r['dialogue'] <= 3175}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert len(licensed) == 41 and len(restricted) == 19
    assert not licensed & restricted and licensed | restricted == set(range(3116, 3176))
    assert len(report['source_corrections']) == 59
    for c in report['source_corrections']:
        _, d, t = c['address'].split(':')
        assert json.loads(lines[int(d)-1])[int(t)-1] == c['after']
        assert fingerprint(c['before']) == c['before_sha256']
        assert fingerprint(c['after']) == c['after_sha256']
    assert 'vested personnel file system' in indexed['dd_review_turn_3132_6']['reply']
    assert 'a @ agent' in indexed['dd_review_turn_3135_6']['reply']
    assert indexed['dd_review_turn_3151_1']['reply'].endswith('J')
    assert indexed['dd_review_turn_3153_4']['reply'].endswith('improve you')
    assert indexed['dd_review_turn_3153_6']['reply'].endswith('in order to be')
    assert 'or thing' in indexed['dd_review_turn_3156_6']['reply']
    assert 'Stars.com' in indexed['dd_review_turn_3146_3']['reply']
    assert 'go bowling' in indexed['dd_review_turn_3163_2']['reply']
    for identity in ('train:3122:7:1', 'train:3125:1:1', 'train:3132:6:2',
                     'train:3140:13:2', 'train:3140:13:3', 'train:3140:13:11',
                     'train:3140:13:12', 'train:3146:3:1', 'train:3146:3:2',
                     'train:3151:1:2', 'train:3153:4:4', 'train:3153:6:4',
                     'train:3162:5:1', 'train:3162:5:2'):
        assert next(r for r in decisions if r['occurrence_id'] == identity)['outcome'] == 'adapted_visible_reply_only'
    assert report['checks']['max_tokens'] <= 768 and not report['checks']['overlength']
    assert not any(report['checks']['split_prompt_overlap'].values())
    inventories = pending['sources'][1]['review']
    for name, indices in [('adverbs.jsonl.gz', (492, 801)),
                          ('entity_candidates.jsonl.gz', (4878, 7107, 7282, 8642, 8942)),
                          ('nominals.jsonl.gz', (6606, 16732, 20343, 21446, 23151,
                                                40227, 40668, 47353, 48028, 67860))]:
        rows = {r['row_index']: r for r in inventories[name]['decisions']}
        assert all(rows[i]['outcome'] == 'rejected' for i in indices)
    # Inventory evidence uses zero-based flat source indices, not shifted turns.
    rows = {r['row_index']: r for r in inventories['entity_candidates.jsonl.gz']['decisions']}
    assert rows[5353]['evidence'] == ['train:3142:1']
    assert rows[11530]['evidence'] == ['train:3161:1']


def test_batch_3116_3175_offers_reports_and_scores_preserve_uncertainty():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for identity, phrase in {
        '3116_10': "husband's actual response is unknown",
        '3117_2': 'no resignation has occurred',
        '3118_2': 'not an actual death',
        '3119_6': 'having received a promotion',
        '3119_9': 'no actual appointment requirement is established',
        '3119_16': 'figurative',
        '3121_10': 'not verified laws',
        '3122_7': 'not an established future outcome',
        '3124_2': 'permission is not yet obtained',
        '3126_3': 'no second visit is introduced',
        '3126_7': 'completion of bed rest is not established',
        '3128_3': 'not a confirmed change',
        '3129_12': 'ticket purchase are not explicitly completed',
        '3130_4': 'tentatively attributes',
        '3131_4': 'an exact age is not supplied',
        '3132_6': "does not establish Simon's precise responsibility",
        '3132_9': 'introduction has not explicitly occurred',
        '3133_6': 'unsuccessfully tried to get in',
        '3133_18': 'inference is not established fact',
        '3133_19': 'hypothetically predicting an equal grade',
        '3134_3': 'no literal death is reported',
        '3135_6': 'does not establish a precise kind of agent',
        '3136_2': 'exact party time is not provided',
        '3138_2': 'not spatial location',
        '3139_11': 'rejects the naughty label',
        '3140_9': 'actual appointment or start is not established',
        '3140_12': 'does not describe skin color',
        '3141_2': 'did not get the job',
        '3142_4': 'no item has been brought',
        '3143_4': 'actual regret is not established',
        '3144_6': 'implementation has not occurred',
        '3145_2': 'exact task of summing up is unspecified',
        '3146_6': 'do not establish a psychiatric condition',
        '3147_3': 'no stopping amount or completed pour is specified',
        '3149_2': 'no flower purchase is stated',
        '3150_13': 'does not establish play on both days every week',
        '3151_2': 'no financing or doctorate is completed',
        '3152_6': 'no website is named yet',
        '3153_4': 'missing complement is not invented',
        '3153_9': 'no shooting or arrival is completed',
        '3154_2': 'not an animal identity',
        '3155_4': 'no test result exists yet',
        '3156_3': 'no invitation has occurred',
        '3157_3': 'source name is preserved',
        '3158_4': 'not confirmed by independent evidence',
        '3159_5': 'ownership of the brother reference uncertain',
        '3160_2': 'no calendar year is given',
        '3161_8': 'no decision has been made',
        '3162_7': 'invitation does not establish attendance',
        '3163_5': 'camping is not confirmed',
        '3165_4': 'he has no children',
        '3166_4': 'attendance is not established',
        '3168_2': 'completed purchase is not established',
        '3169_7': 'no named borrower or theft is established',
        '3169_11': 'completion of the reservation is not explicitly reported',
        '3170_2': 'attendance has not occurred',
        '3171_12': 'arrival is not established',
        '3172_10': 'no corrected score is established',
        '3173_2': 'recovery has not happened',
        '3174_3': 'estimates his age as almost sixty',
        '3175_6': 'not necessarily a romantic meeting',
    }.items():
        assert phrase in indexed['dd_review_turn_' + identity]['value']
    assert 'Susan' not in indexed['dd_review_turn_3140_3']['value']
    assert 'Melaena' not in indexed['dd_review_turn_3157_1']['value']
    for identity, phrase in [('3119_6', 'Paul'), ('3126_7', 'seven days'),
                             ('3131_2', 'twenty-five'), ('3137_3', 'Park Hotel'),
                             ('3143_1', 'Helen'), ('3164_2', 'Shanghai'),
                             ('3167_5', 'three times a year'), ('3175_3', 'Wednesday')]:
        variant = next(p for p in PATTERNS if p['substitution_base_id'] == 'dd_review_turn_' + identity)
        assert phrase in realize_pattern(variant, 0)['value']


def test_batch_3176_3235_source_coverage_and_preserved_incomplete_replies():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_3176_3235']
    lines = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    indexed = {p['id']: p for p in CATALOG['patterns']}
    count = 0
    for d in range(3176, 3236):
        turns = json.loads(lines[d - 1])
        for t, text in enumerate(turns, 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert p['source']['context'] == turns[:t]
            assert p['review']['source_sha256'] == fingerprint(text)
            assert p['review']['context_sha256'] == fingerprint(turns[:t])
            assert max(p['review']['evidence_turns']) <= t
            count += 1
    assert count == report['turns'] == 427
    decisions = [r for k in ('sentence_attestations', 'occurrence_adaptations')
                 for r in review[k]['decisions'] if r['occurrence_id'].startswith('train:')
                 and 3176 <= int(r['occurrence_id'].split(':')[1]) <= 3235]
    assert len(decisions) == len({r['occurrence_id'] for r in decisions}) == 640
    assert sum(r['outcome'] == 'attested_sentence_only' for r in decisions) == 512
    for r in decisions:
        assert r['allowed_adapter']['sha256'] == fingerprint(indexed[r['allowed_adapter']['id']])
    assert sum(x['added_decisions'] for x in report['lexical'].values()) == 378
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 3176 <= r['dialogue'] <= 3235}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert len(licensed) == 36 and len(restricted) == 24
    assert not licensed & restricted and licensed | restricted == set(range(3176, 3236))
    assert len(report['source_corrections']) == 54
    for c in report['source_corrections']:
        _, d, t = c['address'].split(':')
        assert json.loads(lines[int(d)-1])[int(t)-1] == c['after']
        assert fingerprint(c['before']) == c['before_sha256']
        assert fingerprint(c['after']) == c['after_sha256']


def test_batch_3176_3235_ambiguous_sources_and_contextual_substitutions():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for identity, phrase in {
        '3177_2': 'not an anatomical claim',
        '3183_8': 'successful sustained cessation is not established',
        '3184_3': 'no missing condition or job guarantee is invented',
        '3188_3': 'not another person named Jane',
        '3193_3': 'no specific feast or event is reconstructed',
        '3197_4': 'not inferred from later turns',
        '3198_8': 'the spelling conflict is preserved',
        '3200_4': 'no exact price is given',
        '3202_4': 'a completed purchase remain unconfirmed',
        '3203_4': 'not independently verified motivation',
        '3204_6': 'no travel duration is supplied',
        '3205_7': 'movement outside is not explicitly completed',
        '3206_11': 'hope was not fulfilled',
        '3209_6': 'unusual names are not silently repaired',
        '3210_10': 'exact payment transaction is not described',
        '3213_8': 'neither explanation nor preview is completed',
        '3215_4': 'not a completed wager or promotion',
        '3219_6': 'remains a reported account',
        '3220_6': 'neither purchase nor actual fraud is confirmed',
        '3224_7': 'no signing, raise or final agreement is completed',
        '3226_5': 'award claim is attributed to the quizmaster',
        '3228_4': 'no arrival is established',
        '3229_6': 'not reliably recoverable',
        '3230_9': 'actual bribery intent is not established',
        '3231_9': 'distinct from the vacation ending the following day',
        '3235_8': 'actual additional employment are not established',
        '3235_12': 'drive has not explicitly occurred',
    }.items():
        assert phrase in indexed['dd_review_turn_' + identity]['value']
    for identity, phrase in [('3184_3', 'provide'), ('3193_3', 'big face down the eve'),
                             ('3193_5', 'mystery belief'), ('3198_8', 'Isabella'),
                             ('3204_6', 'It takes about'), ('3209_6', 'Peckham'),
                             ('3229_6', 'beer insist')]:
        assert phrase in indexed['dd_review_turn_' + identity]['reply']
    variants = {p['substitution_base_id']: realize_pattern(p, 0) for p in PATTERNS
                if p['source']['split'] == 'train' and 3176 <= p['source']['dialogue'] <= 3235}
    assert 'sailing model boats' in variants['dd_review_turn_3202_1']['reply']
    assert 'sailing model boats' in variants['dd_review_turn_3202_2']['value']
    assert 'no model boat' in variants['dd_review_turn_3202_3']['reply']
    assert 'completed purchase remain unconfirmed' in variants['dd_review_turn_3202_4']['value']
    assert not any('flying model boats' in p[f] for p in variants.values() for f in ('reply', 'value'))
    assert '13:00' in variants['dd_review_turn_3196_4']['reply']
    assert 'one in the afternoon' in variants['dd_review_turn_3196_4']['value']
    assert 'declines' in variants['dd_review_turn_3196_5']['value']
    assert 'Alex Wilson' in variants['dd_review_turn_3219_1']['reply']
    assert 'Mr. Wilson' in variants['dd_review_turn_3219_4']['reply']
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    rows = {r['occurrence_id']: r for k in ('sentence_attestations', 'occurrence_adaptations')
            for r in review[k]['decisions']}
    for identity in ('train:3183:2:1', 'train:3184:3:3', 'train:3197:4:1',
                     'train:3198:1:4', 'train:3204:6:5', 'train:3205:8:1',
                     'train:3213:8:1', 'train:3219:6:1', 'train:3224:7:2',
                     'train:3224:7:3', 'train:3226:4:2', 'train:3229:6:1'):
        assert rows[identity]['outcome'] == 'adapted_visible_reply_only'
    inventories = pending['sources'][1]['review']
    for name, ids in [('entity_candidates.jsonl.gz', (2674, 3580, 6121, 7500, 12638)),
                      ('nominals.jsonl.gz', (2383, 23844, 24345, 39363, 49818, 61532))]:
        records = {r['row_index']: r for r in inventories[name]['decisions']}
        assert all(records[i]['outcome'] == 'rejected' for i in ids)
    checks = review['batches']['train_3176_3235']['checks']
    assert checks['max_tokens'] <= 768 and not checks['overlength']
    assert not any(checks['split_prompt_overlap'].values())


def test_batch_3296_3355_covers_sources_candidates_and_lexical_entries():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_3296_3355']
    lines = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    indexed = {p['id']: p for p in CATALOG['patterns']}
    count = 0
    for d in range(3296, 3356):
        turns = json.loads(lines[d - 1])
        for t, text in enumerate(turns, 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert p['source']['context'] == turns[:t]
            assert p['review']['source_sha256'] == fingerprint(text)
            assert p['review']['context_sha256'] == fingerprint(turns[:t])
            assert max(p['review']['evidence_turns']) <= t
            count += 1
    assert count == report['turns'] == 474
    decisions = [r for k in ('sentence_attestations', 'occurrence_adaptations')
                 for r in review[k]['decisions'] if r['occurrence_id'].startswith('train:')
                 and 3296 <= int(r['occurrence_id'].split(':')[1]) <= 3355]
    assert len(decisions) == len({r['occurrence_id'] for r in decisions}) == 812
    assert sum(r['outcome'] == 'attested_sentence_only' for r in decisions) == 690
    for r in decisions:
        assert r['allowed_adapter']['sha256'] == fingerprint(indexed[r['allowed_adapter']['id']])
    assert sum(x['added_decisions'] for x in report['lexical'].values()) == 467
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 3296 <= r['dialogue'] <= 3355}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert len(licensed) == 28 and len(restricted) == 32
    assert not licensed & restricted and licensed | restricted == set(range(3296, 3356))
    for c in report['source_corrections']:
        _, d, t = c['address'].split(':')
        assert json.loads(lines[int(d)-1])[int(t)-1] == c['after']
        assert fingerprint(c['before']) == c['before_sha256']
        assert fingerprint(c['after']) == c['after_sha256']


def test_batch_3296_3355_preserves_reported_claims_refusals_and_uncertain_names():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for identity, phrase in {
        '3297_6': 'neither next shot is completed',
        '3298_12': 'not a verified current schedule',
        '3300_5': 'ends at don',
        '3304_6': 'ends at it',
        '3305_5': 'extinction is not established',
        '3307_2': 'no city name or ownership',
        '3310_2': 'not interpreted as a license level',
        '3313_12': 'exercise has not resumed',
        '3314_2': 'Calmed up is unclear',
        '3318_3': 'arrival is not established',
        '3321_4': 'distinct from the listener Bob',
        '3321_12': 'total chapter count is not yet given',
        '3324_3': 'no precise wind scale',
        '3325_19': 'not her consent',
        '3329_4': 'refusal remains',
        '3334_6': 'paramedics',
        '3335_13': 'latest agreement takes precedence',
        '3336_4': 'not verified medical facts',
        '3337_5': 'round describing the day is unclear',
        '3340_5': 'source language is unspecified',
        '3341_6': 'no demonstration or purchase',
        '3343_5': 'unfinished word',
        '3344_12': 'no completed divorce',
        '3345_11': 'not a factual anatomical feature',
        '3346_2': 'speaker-to-name mapping remains uncertain',
        '3346_6': 'film events, not real-world actions',
        '3347_1': 'Greets Walter',
        '3347_2': 'Greets Sterling',
        '3347_4': 'Professor Wood',
        '3348_5': 'spelling conflict is preserved',
        '3350_6': 'match winner are not specified',
        '3351_12': 'first explicit identification of Anna',
        '3352_7': 'not presented as verified knowledge',
        '3353_6': 'increase, not the full monthly food budget',
        '3354_6': 'neither job has been obtained',
        '3355_7': 'no job offer or contact is completed',
    }.items():
        assert phrase in indexed['dd_review_turn_' + identity]['value']
    for t in range(1, 12):
        assert 'Anna' not in indexed[f'dd_review_turn_3351_{t}']['value']
    for identity, phrase in [('3300_5', 'don'), ('3310_2', 'mechanic 5'),
                             ('3314_2', 'calmed up'), ('3337_5', 'round day'),
                             ('3343_5', 'Drin'), ('3343_6', 'Wea'),
                             ('3348_4', 'Jane Brown'), ('3348_5', 'Jan Brown')]:
        assert phrase in indexed['dd_review_turn_' + identity]['reply']
    assert 'Great Wall' in indexed['dd_review_turn_3298_1']['reply']
    assert "haven't bought" in indexed['dd_review_turn_3353_3']['reply']
    variants = {p['substitution_base_id']: realize_pattern(p, 0) for p in PATTERNS
                if p['source']['split'] == 'train' and 3296 <= p['source']['dialogue'] <= 3355}
    assert 'freshwater fish' in variants['dd_review_turn_3296_6']['reply']
    assert 'freshwater fish' in variants['dd_review_turn_3296_6']['value']
    assert 'Spanish' in variants['dd_review_turn_3340_5']['reply']
    assert 'Spanish' in variants['dd_review_turn_3340_5']['value']
    assert 'Greets Peter' in variants['dd_review_turn_3347_1']['value']
    assert 'Professor Miller' in variants['dd_review_turn_3347_4']['value']
    assert 'first explicit identification of Helen' in variants['dd_review_turn_3351_12']['value']
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    decisions = {r['occurrence_id']: r for k in ('sentence_attestations', 'occurrence_adaptations')
                 for r in pending['sources'][0]['review'][k]['decisions']}
    for identity in ('train:3300:5:3', 'train:3304:6:3', 'train:3338:3:1',
                     'train:3338:3:2', 'train:3343:5:4', 'train:3343:6:4',
                     'train:3346:6:5', 'train:3348:3:2'):
        assert decisions[identity]['outcome'] == 'adapted_visible_reply_only'
    for name, ids in [('entity_candidates.jsonl.gz', (3756, 4068, 6097, 8709)),
                      ('nominals.jsonl.gz', (4613, 5069, 15141, 35309, 46432, 67066))]:
        rows = {r['row_index']: r for r in pending['sources'][1]['review'][name]['decisions']}
        assert all(rows[i]['outcome'] == 'rejected' for i in ids)
    checks = pending['sources'][0]['review']['batches']['train_3296_3355']['checks']
    assert checks['max_tokens'] <= 768 and not checks['overlength']
    assert not any(checks['split_prompt_overlap'].values())


def test_batch_3236_3295_source_coverage_and_preserved_incomplete_replies():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_3236_3295']
    lines = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    indexed = {p['id']: p for p in CATALOG['patterns']}
    count = 0
    for d in range(3236, 3296):
        turns = json.loads(lines[d - 1])
        for t, text in enumerate(turns, 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert p['source']['context'] == turns[:t]
            assert p['review']['source_sha256'] == fingerprint(text)
            assert p['review']['context_sha256'] == fingerprint(turns[:t])
            assert max(p['review']['evidence_turns']) <= t
            count += 1
    assert count == report['turns'] == 399
    decisions = [r for k in ('sentence_attestations', 'occurrence_adaptations')
                 for r in review[k]['decisions'] if r['occurrence_id'].startswith('train:')
                 and 3236 <= int(r['occurrence_id'].split(':')[1]) <= 3295]
    assert len(decisions) == len({r['occurrence_id'] for r in decisions}) == 663
    assert sum(r['outcome'] == 'attested_sentence_only' for r in decisions) == 535
    for r in decisions:
        assert r['allowed_adapter']['sha256'] == fingerprint(indexed[r['allowed_adapter']['id']])
    assert sum(x['added_decisions'] for x in report['lexical'].values()) == 362
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 3236 <= r['dialogue'] <= 3295}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert len(licensed) == 37 and len(restricted) == 23
    assert not licensed & restricted and licensed | restricted == set(range(3236, 3296))
    assert len(report['source_corrections']) == 45
    for c in report['source_corrections']:
        _, d, t = c['address'].split(':')
        assert json.loads(lines[int(d)-1])[int(t)-1] == c['after']
        assert fingerprint(c['before']) == c['before_sha256']
        assert fingerprint(c['after']) == c['after_sha256']


def test_batch_3236_3295_reports_counterfactuals_and_title_boundaries():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for identity, phrase in {
        '3236_10': 'no unmentioned time jump',
        '3237_3': 'only if time permits',
        '3238_4': 'drinking or serving is not completed',
        '3239_2': 'definite cancellation is not reported',
        '3240_4': 'late auction bidding, not shooting',
        '3241_8': 'hands over the additional requested material',
        '3242_2': 'not complete abstinence',
        '3243_5': 'booking has not occurred',
        '3244_12': 'new degree is not completed',
        '3245_4': 'acceptance, not completed attendance',
        '3246_4': 'returned to the same area',
        '3247_3': 'without the actual help',
        '3248_2': 'hyperbole, not an actual death',
        '3249_6': 'forecast and past measurement remain distinct',
        '3250_2': 'consulting her, not a physical action',
        '3251_6': 'no spasm is occurring',
        '3252_2': 'does not establish an actual separation',
        '3253_2': 'without specifying morning or evening',
        '3254_3': 'neither swimming nor tanning has occurred',
        '3255_6': 'finding them is not confirmed',
        '3256_2': 'neither team won',
        '3256_4': 'player or a team is not clear',
        '3257_8': 'sit out the next dance',
        '3258_3': 'no year or independent date confirmation',
        '3259_2': 'Rejects the situation being as serious as divorce',
        '3260_9': 'does not establish that the adviser also has a birthday',
        '3261_2': 'sitting is not explicitly completed',
        '3262_6': 'neither arrival nor call has happened',
        '3263_2': 'only the Sichuan leg explicitly has air travel',
        '3263_4': 'minimum duration',
        '3264_3': 'NCP is unexplained',
        '3265_7': 'not a specified number of inches to remove',
        '3265_12': 'haircut is not yet reported as completed',
        '3266_6': 'location is undecided',
        '3266_10': 'neither alternative is purchased',
        '3267_7': 'content is unknown',
        '3268_10': 'Denies having that kind of deposit account',
        '3269_6': 'no toss or choice is completed',
        '3270_6': 'a different person from John',
        '3271_7': 'no visit has occurred',
        '3272_5': 'not completed from later turns',
        '3272_7': 'identity is not yet established',
        '3272_11': 'father in green pajamas',
        '3273_4': 'no resignation, application or new job',
        '3274_1': 'not confidently resolved as a car purchase',
        '3274_7': 'borrowing is reported, shopping is not completed',
        '3275_5': 'Declines quitting for now',
        '3276_4': 'family is very well',
        '3277_2': 'cause is unknown',
        '3278_6': 'no future visit is completed',
        '3279_3': 'trailing objection is not completed',
        '3280_6': 'cutting is not completed',
        '3281_7': 'relocation is not completed',
        '3282_1': 'accepting that position or playing has not yet occurred',
        '3283_4': 'measured fever is not established',
        '3284_4': 'does not establish a completed move',
        '3285_2': 'country is unnamed',
        '3286_12': 'not its chart position',
        '3287_4': 'no subject or satisfaction level is invented',
        '3287_5': 'missing instruction is not invented',
        '3288_1': 'not an actual death or kicking action',
        '3289_6': 'no rehearsal or successful delivery is reported',
        '3290_13': 'no financing is secured',
        '3291_4': 'arrival has not occurred',
        '3292_8': 'cultural claim is attributed',
        '3293_1': 'not literal omniscience',
        '3294_4': 'sole income earner',
        '3295_6': 'artistic career is counterfactual',
        '3295_8': 'alternative of never meeting is counterfactual',
    }.items():
        assert phrase in indexed['dd_review_turn_' + identity]['value']
    for identity, phrase in [('3264_3', 'NCP'), ('3287_4', 'I ended up studying something'),
                             ('3287_5', 'mak'), ('3272_5', 'when...'), ('3272_9', 'until...')]:
        assert phrase in indexed['dd_review_turn_' + identity]['reply']
    for t in range(1, 11):
        assert 'father' not in indexed[f'dd_review_turn_3272_{t}']['value'].lower()
    variants = {p['substitution_base_id']: realize_pattern(p, 0) for p in PATTERNS
                if p['source']['split'] == 'train' and 3236 <= p['source']['dialogue'] <= 3295}
    for identity, reply, meaning in [('3240_4', '$ 350', '$350'),
                                     ('3248_1', '60 hours', 'sixty hours'),
                                     ('3258_3', 'May 6th', 'May sixth'),
                                     ('3263_4', '21 days', 'twenty-one days'),
                                     ('3278_1', '9:00 pm', '9 p.m.'),
                                     ('3291_3', "eight o'clock", "eight o'clock")]:
        p = variants['dd_review_turn_' + identity]
        assert reply in p['reply'] and meaning in p['value']
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    decisions = {r['occurrence_id']: r for k in ('sentence_attestations', 'occurrence_adaptations')
                 for r in review[k]['decisions']}
    for identity in ('train:3264:4:1', 'train:3272:5:3', 'train:3272:9:2',
                     'train:3286:4:1', 'train:3286:4:2', 'train:3286:6:2',
                     'train:3286:6:3', 'train:3286:9:1', 'train:3286:9:2',
                     'train:3287:4:5', 'train:3287:5:4'):
        assert decisions[identity]['outcome'] == 'adapted_visible_reply_only'
    for name, indices in [('entity_candidates.jsonl.gz', (5594, 6756, 7818)),
                          ('nominals.jsonl.gz', (164, 7862, 9036, 54230, 66173, 69228))]:
        rows = {r['row_index']: r for r in pending['sources'][1]['review'][name]['decisions']}
        assert all(rows[i]['outcome'] == 'rejected' for i in indices)
    names = {r['row_index']: r for r in pending['sources'][1]['review']['entity_candidates.jsonl.gz']['decisions']}
    assert names[3326]['evidence'] == ['train:3286:7']
    assert names[6203]['evidence'] == ['train:3263:2']
    checks = review['batches']['train_3236_3295']['checks']
    assert checks['max_tokens'] <= 768 and not checks['overlength']
    assert not any(checks['split_prompt_overlap'].values())


def test_batch_3356_3415_counterexamples_and_consistent_variants():
    indexed = {p['id']: p for p in CATALOG['patterns']}
    for identity, phrase in {
        '3356_3': 'before losing it',
        '3356_7': 'completion is future',
        '3357_4': 'no previous tennis skill',
        '3358_3': 'safety is not independently guaranteed',
        '3359_3': 'does not reliably establish an explosion',
        '3360_9': 'arrival in class is not reported',
        '3361_4': 'not an auction',
        '3362_3': 'exact debt type is unclear',
        '3362_8': 'no repayment has occurred',
        '3363_4': 'hiring is not confirmed',
        '3364_1': 'without naming the occasion',
        '3365_5': 'Has no plan yet',
        '3366_6': 'spells Nugent',
        '3367_5': 'extinction is not established',
        '3368_3': 'repeated he is unclear',
        '3369_8': 'not general ethnic rules',
        '3370_9': 'no identity is invented',
        '3371_4': '3-4 to 5-4',
        '3371_6': 'game is not declared finished',
        '3372_5': 'later call is not completed',
        '3373_3': 'not wagering',
        '3374_6': 'placement is intended',
        '3375_2': '150 pounds to 120 pounds',
        '3376_3': 'no new income or house purchase',
        '3377_5': 'no year is supplied',
        '3378_4': 'conversation has not happened',
        '3379_6': 'not yet described as one',
        '3380_6': 'new job is not completed',
        '3381_4': 'without explaining the cause',
        '3382_5': 'pieces are not the number of people',
        '3383_2': 'completion is not yet confirmed',
        '3383_4': 'Now declares the styling finished',
        '3383_10': 'payment is not reported',
        '3384_12': 'availability is unconfirmed',
        '3384_17': 'childcare is not yet confirmed',
        '3385_3': 'culprit is not identified',
        '3385_5': 'neither has occurred',
        '3386_4': 'not a proven event',
        '3387_6': 'purchase location is not explicitly confirmed',
        '3387_11': 'neither the gym visit nor the tour',
        '3388_12': 'wins have not occurred',
        '3389_5': 'morning or afternoon is unspecified',
        '3390_3': 'age twelve',
        '3391_2': 'not a request for more information',
        '3392_3': 'not a verified treatment claim',
        '3393_6': 'ratios are attributed',
        '3394_3': 'not a jewelry gift',
        '3395_4': 'recovery or death is not established',
        '3396_2': 'neither confirmed nor denied',
        '3397_8': 'neither is confirmed',
        '3397_15': 'not a verified repair limitation',
        '3398_4': 'neither parenthood nor fabrication',
        '3399_11': 'arrival is future',
        '3400_2': 'reconciling as a couple',
        '3401_10': 'neither message nor attendance',
        '3402_9': 'size does not establish legal admissibility',
        '3403_6': 'stones are collected',
        '3404_2': 'not arithmetic',
        '3405_7': 'completion is not yet achieved',
        '3405_9': 'journey has now reached',
        '3406_5': 'cutting is future',
        '3407_1': 'no actual shooting',
        '3408_10': 'none of the suggested tasks',
        '3409_3': 'no exact time',
        '3410_4': 'calling flowers friends is figurative',
        '3411_6': 'contents remain unknown',
        '3411_10': 'diamond brooch',
        '3411_11': 'current anniversary number is not given',
        '3412_2': 'no current meal or actual vomiting',
        '3413_8': 'not completed actions',
        '3414_3': 'emotional',
        '3415_6': 'not reliably interpreted',
        '3415_9': 'unclear reed up',
        '3415_11': 'not verified advice',
    }.items():
        assert phrase in indexed['dd_review_turn_' + identity]['value']
    # Earlier prompts do not leak the eventual object or the future sitter's name.
    assert 'brooch' not in indexed['dd_review_turn_3411_6']['assistant'].lower()
    assert 'Sara' not in indexed['dd_review_turn_3384_7']['assistant']
    for identity, text in [('3362_3', 'mortgage'), ('3371_2', 'Cal Ripen'),
                           ('3411_10', 'A.'), ('3415_6', 'lidos'),
                           ('3415_9', 'reed up'), ('3415_13', 'real blessed')]:
        assert text in indexed['dd_review_turn_' + identity]['reply']
    for identity, text in [('3384_3', 'L.A. Hank'), ('3385_3', 'senior-level lawyers'),
                           ('3389_4', 'do you'), ('3397_6', 'pack of gum'),
                           ('3405_7', 'have been working on it')]:
        assert text in indexed['dd_review_turn_' + identity]['reply']
    variants = {p['substitution_base_id']: realize_pattern(p, 0) for p in PATTERNS
                if p['source']['split'] == 'train' and 3356 <= p['source']['dialogue'] <= 3415}
    for identity, reply, meaning in [
        ('3357_1', 'badminton', 'badminton'),
        ('3360_7', '114', '114'),
        ('3366_6', 'Peter', 'Peter'),
        ('3375_2', '160 pounds to 130', '160 pounds to 130 pounds'),
        ('3383_10', '$ 60', '$60'),
        ('3388_12', 'five more games', 'five remaining games'),
        ('3389_5', 'one twenty', '1:20'),
        ('3390_3', 'Thirteen', 'thirteen'),
        ('3410_2', 'tulips', 'tulips'),
    ]:
        p = variants['dd_review_turn_' + identity]
        assert reply in p['reply'] and meaning in p['value']
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    for name, ids in [('entity_candidates.jsonl.gz', (4324, 5397, 11070)),
                      ('nominals.jsonl.gz', (4611, 7050, 7254, 19506, 37920, 46115, 53762))]:
        rows = {r['row_index']: r for r in pending['sources'][1]['review'][name]['decisions']}
        assert all(rows[i]['outcome'] == 'rejected' for i in ids)
    checks = pending['sources'][0]['review']['batches']['train_3356_3415']['checks']
    assert checks['max_tokens'] <= 768 and not checks['overlength']
    assert not any(checks['split_prompt_overlap'].values())


def test_batch_3356_3415_covers_sources_candidates_and_lexical_entries():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    report = review['batches']['train_3356_3415']
    lines = (ROOT.parents[1] / 'data/dailydialog/filtered/dailydialog-train.jsonl').read_text(encoding='utf8').splitlines()
    indexed = {p['id']: p for p in CATALOG['patterns']}
    count = 0
    for d in range(3356, 3416):
        turns = json.loads(lines[d - 1])
        for t, text in enumerate(turns, 1):
            p = indexed[f'dd_review_turn_{d}_{t}']
            assert p['reply'] == p['source']['text'] == text
            assert p['source']['context'] == turns[:t]
            assert p['review']['source_sha256'] == fingerprint(text)
            assert p['review']['context_sha256'] == fingerprint(turns[:t])
            assert max(p['review']['evidence_turns']) <= t
            count += 1
    assert count == report['turns'] == 440
    decisions = [r for k in ('sentence_attestations', 'occurrence_adaptations')
                 for r in review[k]['decisions'] if r['occurrence_id'].startswith('train:')
                 and 3356 <= int(r['occurrence_id'].split(':')[1]) <= 3415]
    assert len(decisions) == len({r['occurrence_id'] for r in decisions}) == 697
    assert sum(r['outcome'] == 'attested_sentence_only' for r in decisions) == 612
    for r in decisions:
        assert r['allowed_adapter']['sha256'] == fingerprint(indexed[r['allowed_adapter']['id']])
    assert sum(x['added_decisions'] for x in report['lexical'].values()) == 363
    licensed = {r['dialogue'] for r in RULES if r['split'] == 'train' and 3356 <= r['dialogue'] <= 3415}
    restricted = {r['dialogue'] for r in report['substitution_restrictions']}
    assert len(licensed) == 46 and len(restricted) == 14
    assert not licensed & restricted and licensed | restricted == set(range(3356, 3416))
    for c in report['source_corrections']:
        _, d, t = c['address'].split(':')
        assert json.loads(lines[int(d)-1])[int(t)-1] == c['after']
        assert fingerprint(c['before']) == c['before_sha256']
        assert fingerprint(c['after']) == c['after_sha256']


def test_cross_batch_restriction_followup_preserves_coverage_and_pins():
    pending = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))
    review = pending['sources'][0]['review']
    followup = review['substitution_restriction_followup']
    assert followup['restrictions_considered'] == 654
    assert followup['new_finite_rules'] == len(followup['decisions']) == 143
    remaining = set(review['backfill_1406_2215']['substitution_review']['restricted_dialogues'])
    remaining.update(r['dialogue'] for b in review['batches'].values()
                     for r in b.get('substitution_restrictions', []))
    added = {d['dialogue'] for d in followup['decisions']}
    assert len(remaining) == followup['remaining_dialogue_restrictions'] == 511
    assert len(added) == 143 and not added & remaining
    indexed = {r['id']: r for r in RULES}
    for decision in followup['decisions']:
        rule = indexed[decision['rule']]
        assert decision['rule_sha256'] == fingerprint(rule)
        assert decision['previous_reason'] and decision['retained_limits'] == rule['reason']
    assert followup['checks']['max_tokens'] <= 768
    assert not followup['checks']['overlength']
    assert not any(followup['checks']['split_prompt_overlap'].values())
    assert review['next_source_dialogue']['dialogue'] == 3416


@pytest.mark.parametrize('dialogue,turn,reply,value', [
    (2125, 7, '22 dollars', 'twenty-two dollars'),
    (2125, 8, '30 dollars', 'thirty dollars'),
    (2332, 4, 'at 13', 'one in the afternoon'),
    (2424, 6, '250', 'two-hundred-and-fifty-foot'),
    (2672, 4, 'courier', 'courier work'),
    (2818, 2, 'seven months', 'not seven separate payments'),
    (2825, 6, 'fifty percent', 'twenty-five percent each'),
    (2865, 8, 'about nine', 'around nine'),
    (2963, 3, 'eleven', 'both are currently ten'),
    (3044, 1, 'silver watch', 'silver-colored watch'),
    (3082, 2, '9 sharp', 'exactly nine'),
    (3171, 9, 'D261', 'D261'),
    (3186, 4, 'Glasgow', 'Glasgow'),
    (3377, 3, 'ninth of March', 'March ninth'),
    (3398, 3, 'four-year-old', 'report is not verified'),
    (3404, 2, 'Count me out', 'Declines participation'),
])
def test_cross_batch_substitutions_keep_numbers_references_and_scope(dialogue, turn, reply, value):
    matches = [p for p in PATTERNS if p['source']['split'] == 'train'
               and p['source']['dialogue'] == dialogue and p['source']['turn'] == turn]
    assert any(reply in realize_pattern(p, 0)['reply'] and value in realize_pattern(p, 0)['value']
               for p in matches)
    if dialogue == 2125:
        assert 20 * 1.10 == 22 and 30 - 22 == 8
    if dialogue == 2825:
        assert 50 + 25 + 25 == 100


def test_restriction_debt_keeps_unresolved_cases_visible():
    review = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))['sources'][0]['review']
    report = review['substitution_restriction_followup']['passes']['train_2276_2335']
    assert len(report['dialogues_reviewed']) == 21
    assert report['new_rules'] == 16
    assert set(map(int, report['retained_restrictions'])) == {2288, 2290, 2305, 2306, 2310}
    batch = review['batches']['train_2276_2335']
    assert {r['dialogue'] for r in batch['substitution_restrictions']} == {2288, 2290, 2305, 2306, 2310}
    assert batch['substitution_rules'] == 55
    assert report['checks']['native_variants'] == 19
    assert report['native_blocked_choices_added'] == 0
    assert report['checks']['max_tokens'] <= 768
    assert not any(report['checks']['split_prompt_overlap'].values())
    assert not report['checks']['overlength']
    reviewed = [p for p in CATALOG['patterns'] if p['kind'] == 'interpretation'
                and p['source']['split'] == 'train'
                and p['source']['dialogue'] in report['dialogues_reviewed']]
    assert report['reviewed_patterns_sha256'] == fingerprint(reviewed)


@pytest.mark.parametrize('dialogue,turn,reply,value', [
    (2277, 4, 'grinding cocoa beans', 'inference, not an obligation'),
    (2285, 4, 'grandparents', "speaker's grandparents"),
    (2292, 6, "you're wonderful", 'without explicitly answering the love question'),
    (2294, 6, 'four yuan', 'instructions, not completed travel'),
    (2295, 3, 'next Saturday', 'invites the listener next Saturday'),
    (2298, 6, 'clients', 'emailing clients'),
    (2300, 8, 'two weeks', 'two weeks since the birth'),
    (2304, 7, 'at least three weeks', 'still required'),
    (2307, 3, 'the day after tomorrow', 'Proposes climbing'),
    (2314, 10, 'Foxtrot', 'favorite dance'),
    (2319, 19, '10:00', "ten o'clock"),
    (2323, 4, 'Spanish', 'tolerates mistakes'),
    (2326, 2, 'German', 'truncated and not completed'),
    (2328, 5, 'extra writing', 'writing is not yet completed'),
    (2331, 12, 'four days', 'killing is explicitly denied'),
    (2333, 14, 'Four. Three boys and a girl.', 'four children: three boys and one girl'),
])
def test_restriction_debt_variants_preserve_semantic_distinctions(dialogue, turn, reply, value):
    variant = _restriction_variant(dialogue, turn)
    assert reply in variant['reply'] and value in variant['value']
    if dialogue == 2314:
        assert 'current tango dancing' in _restriction_variant(2314, 6)['value']
    if dialogue == 2326:
        unchanged = next(p for p in CATALOG['patterns'] if p['id'] == 'dd_review_turn_2326_7')
        assert 'Chinese and Japanese' in unchanged['reply']
        assert not any(p['substitution_base_id'] == unchanged['id'] for p in PATTERNS)
    if dialogue == 2292:
        unchanged = next(p for p in CATALOG['patterns'] if p['id'] == 'dd_review_turn_2292_15')
        assert 'does not explicitly confirm putting the gun down' in unchanged['value']
        assert not any(p['substitution_base_id'] == unchanged['id'] for p in PATTERNS)


def test_restriction_followup_2336_2395_covers_the_whole_pending_group():
    review = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))['sources'][0]['review']
    f = review['substitution_restriction_followup']
    report = f['passes']['train_2336_2395']
    assert len(report['dialogues_reviewed']) == 26
    assert report['new_rules'] == 22
    assert set(map(int, report['retained_restrictions'])) == {2350, 2365, 2383, 2393}
    batch = review['batches']['train_2336_2395']
    assert batch['substitution_rules'] == 56
    assert {r['dialogue'] for r in batch['substitution_restrictions']} == {2350, 2365, 2383, 2393}
    assert report['checks']['native_variants'] == 28
    assert len(report['native_variant_dialogues']) == 22
    assert not report['adapter_only_rule_ids'] and report['native_blocked_choices_added'] == 0
    assert report['checks']['max_tokens'] <= 768
    assert not report['checks']['overlength'] and not any(report['checks']['split_prompt_overlap'].values())
    reviewed = [p for p in CATALOG['patterns'] if p['kind'] == 'interpretation'
                and p['source']['split'] == 'train' and p['source']['dialogue'] in report['dialogues_reviewed']]
    assert fingerprint(reviewed) == report['reviewed_patterns_sha256']


@pytest.mark.parametrize('dialogue,turn,reply,value', [
    (2338, 4, '25 years', 'twenty-five years'),
    (2339, 3, 'not making late night calls', 'requests'),
    (2342, 4, 'Oscar award', 'not an established award result'),
    (2343, 2, 'Yosemite Valley', 'planned trip'),
    (2346, 4, 'cousin', "speaker's cousin"),
    (2347, 4, 'warm jackets', 'look foolish'),
    (2352, 5, 'darling', 'precise meeting is unspecified'),
    (2353, 2, '2008', 'event is not yet named'),
    (2355, 3, 'council', 'advocated, not completed'),
    (2359, 1, 'Simon', 'gender is not inferred'),
    (2362, 4, 'potatoes', 'personal claim'),
    (2366, 3, 'niece', 'completed marriage is uncertain'),
    (2370, 1, 'the day before yesterday', 'words from the day before yesterday'),
    (2375, 4, 'fairy tales', 'romance now'),
    (2376, 3, 'Italian food', 'Italian food'),
    (2378, 10, 'supervisor', 'sending personal email is office-wide'),
    (2380, 1, 'Thursday', 'complete absence on Thursday'),
    (2382, 5, 'second-hand market', 'suggests visiting'),
    (2386, 6, 'small hint', 'not removal already performed'),
    (2389, 2, 'low mood', 'declines'),
    (2394, 10, '$ 600', 'hypothetical'),
    (2395, 3, 'uncle', 'did not earn'),
])
def test_restriction_followup_2336_2395_preserves_meaning(dialogue, turn, reply, value):
    variant = _restriction_variant(dialogue, turn)
    assert reply in variant['reply'] and value in variant['value']


def test_explicit_adjacent_slot_alignment_and_untouched_historical_claims():
    rule = next(r for r in RULES if r['id'] == 'dd_restriction_followup_2339')
    tid = next(iter(rule['slot_choices']))
    bank = next(b for b in TemplateBank(ROOT).components() if b.source_id == 'dailydialog-reviewed-2336-2395')
    t = next(bank.templates(template_id=tid))
    original = deepcopy(t.record)
    result = render_template(t, rule, 0)
    assert result['bindings']['a0'] == 'late' and result['bindings']['n2'] == 'night'
    assert 'not making late night calls' in result['text']
    assert t.record == original
    for key, slot in t.slots.items():
        if slot['kind'] == 'verb':
            assert result['bindings'][key] == original['defaults'][key]
    unlicensed = deepcopy(rule)
    unlicensed.pop('slot_choices')
    assert render_template(t, unlicensed, 0) is None
    invalid = deepcopy(rule)
    invalid['slot_choices'][tid][0]['bindings']['a0']['after'] = ''
    with pytest.raises(ValueError, match='Invalid reviewed slot binding'):
        validate_rules([invalid], CATALOG['patterns'])
    unchanged = {p['id']: p for p in CATALOG['patterns']}
    assert 'about three years' in unchanged['dd_review_turn_2394_6']['value']
    assert not any(p['substitution_base_id'] == 'dd_review_turn_2394_6' for p in PATTERNS)


def test_restriction_followup_2397_2588_accounts_for_all_fifty():
    review = json.loads((ROOT / 'pending.json').read_text(encoding='utf8'))['sources'][0]['review']
    report = review['substitution_restriction_followup']['passes']['train_2397_2588']
    selected = set(report['dialogues_reviewed'])
    retained = set(map(int, report['retained_restrictions']))
    rules = [r for r in RULES if r['id'].startswith('dd_restriction_followup_')
             and r['dialogue'] in selected]
    licensed = {r['dialogue'] for r in rules}
    assert len(selected) == 50 and len(licensed) == report['new_rules'] == 34
    assert len(retained) == 16 and not retained & licensed
    assert licensed | retained == selected
    assert set(report['native_variant_dialogues']) == licensed
    assert report['checks']['native_variants'] == 44
    assert report['previous_remaining'] - report['remaining_dialogue_restrictions'] == 34
    assert not report['adapter_only_rule_ids'] and not report['native_blocked_choices_added']
    assert report['checks']['episodes'] == 738 and report['checks']['extraction_rows'] == 2214
    assert report['checks']['max_tokens'] <= 768
    assert not report['checks']['overlength'] and not any(report['checks']['split_prompt_overlap'].values())
    remaining = {r['dialogue'] for b in review['batches'].values()
                 for r in b.get('substitution_restrictions', [])}
    assert retained <= remaining and not licensed & remaining
    patterns = [p for p in CATALOG['patterns'] if p['kind'] == 'interpretation'
                and p['source']['split'] == 'train' and p['source']['dialogue'] in selected]
    assert fingerprint(patterns) == report['reviewed_patterns_sha256']
    # A conflicting existing sleep interpretation is explicitly unresolved, not
    # silently licensed by this substitution pass.
    assert report['semantic_followup_required'][0]['dialogue'] == 2496
    assert 2496 in retained


@pytest.mark.parametrize('dialogue,turn,reply,value', [
    (2405, 6, 'Sunday', 'merely as Sunday'),
    (2407, 3, 'A woman', 'claimed financial need'),
    (2409, 2, 'several months', 'several months'),
    (2410, 12, 'a few teas', 'having tea'),
    (2415, 2, 'Susan', 'does not independently establish infidelity'),
    (2416, 3, 'tasty', 'pleasant flavor'),
    (2426, 7, 'van', 'expects distance'),
    (2429, 4, 'physician', 'actually stopping is not reported'),
    (2432, 5, 'David', 'his own beliefs'),
    (2434, 1, 'Mark Fields', 'Bob Copeland'),
    (2445, 4, 'large soccer fields', 'observed during the trip'),
    (2446, 4, 'French', 'minor subjects'),
    (2448, 6, 'thirsty', 'suggests getting a drink'),
    (2450, 4, 'pay a quarter', 'four travelers'),
    (2452, 2, 'five eight five nine', 'not an exact measurement'),
    (2453, 3, 'whole morning', 'entire morning'),
    (2459, 4, 'dissertation', 'no trip is agreed'),
    (2460, 1, 'weightlifter', 'weightlifter or a boxer'),
    (2461, 4, 'relatives', 'Plans to host'),
    (2466, 1, 'French literature', 'interest in French'),
    (2467, 1, 'Uncle', "Uncle's recent condition"),
    (2468, 4, 'two year anniversary', 'failed to attend'),
    (2474, 3, 'store', 'no theft is established'),
    (2477, 4, 'garden to ourselves', 'initially claims'),
    (2485, 13, 'loyalty card', 'loyalty discount card'),
    (2488, 1, 'dinner', 'Suggests it is time'),
    (2500, 2, 'busy tomorrow', 'asks to postpone'),
    (2503, 3, 'local paper', 'not chosen'),
    (2506, 5, "She's", "sister's assessment"),
    (2509, 12, 'history documentary', 'viewing has not yet occurred'),
    (2579, 3, 'Taylor Swift', 'downloading or paying is not completed'),
    (2585, 4, 'Her mother, not me', 'does not establish that she has acted'),
    (2587, 2, 'chef', 'no specific recipe'),
    (2588, 3, 'my cousin', "cousin's claim"),
])
def test_restriction_followup_2397_2588_preserves_scope(dialogue, turn, reply, value):
    variant = _restriction_variant(dialogue, turn)
    assert reply in variant['reply'] and value in variant['value']


def test_restriction_followup_2397_2588_keeps_unknowns_and_negation():
    original = {p['id']: p for p in CATALOG['patterns']}
    assert 'does not independently establish infidelity' in _restriction_variant(2415, 2)['value']
    assert 'do not resolve the mismatch or verify fidelity' in _restriction_variant(2415, 3)['value']
    assert 'denial' in _restriction_variant(2415, 3)['value']
    assert 'not silently replaced with a gym' in _restriction_variant(2460, 4)['value']
    assert 'did not answer' in original['dd_review_turn_2468_8']['value']
    assert 'regret is not established' in original['dd_review_turn_2468_11']['value']
    assert not any(p['substitution_base_id'] in ('dd_review_turn_2468_8', 'dd_review_turn_2468_11')
                   for p in PATTERNS)
    assert 'Corrects that impression' in _restriction_variant(2477, 5)['value']
    assert 'a rain check is not a weather observation' in _restriction_variant(2500, 2)['value']
    assert 'without establishing a new date' in _restriction_variant(2500, 3)['value']
    assert "son's mother" in original['dd_review_turn_2585_4']['value']
    assert "daughter's mother" in _restriction_variant(2585, 4)['value']
    # Only the final drink motivation changes; the allegations remain untouched.
    assert not any(p['substitution_base_id'] == 'dd_review_turn_2448_2' for p in PATTERNS)
    assert 'unverified steroid-use allegation' in original['dd_review_turn_2448_2']['value']
    # The revised anniversary must not also alter 'The first thing'.
    anniversary = _restriction_variant(2468, 4)
    assert 'The first thing' in anniversary['reply']
    assert 'second anniversary' in anniversary['value']
    assert 4 * 0.25 == 1
