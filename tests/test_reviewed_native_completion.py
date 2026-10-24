"""Finite native phrase licenses must preserve context and remain auditable."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from semantic_corpus.template_bank import TemplateBank, CorpusTemplate, fingerprint
from semantic_corpus.reviewed_substitutions import render_template, substitute, realize_pattern
from semantic_corpus.dialogue_generator.fragments import load_fragment_patterns
from semantic_corpus.documents import Document
from semantic_corpus.semantic_generator.substitutions import EntityPool
from semantic_corpus.template_semantics import TemplateAnnotator

ROOT = Path(__file__).resolve().parents[1] / 'data/mined-templates'
CATALOG = json.loads((ROOT/'reviewed/dialogue-fragments.json').read_text(encoding='utf8'))
RULES = {r['id']:r for r in CATALOG['substitutions']}
FOLLOWUP = json.loads((ROOT/'pending.json').read_text(encoding='utf8'))['sources'][0]['review']['substitution_restriction_followup']


@pytest.fixture(scope='module')
def report():
    return FOLLOWUP['native_completion']


@pytest.fixture(scope='module')
def templates(report):
    bank = TemplateBank(ROOT)
    component = next(b for b in bank.components() if b.source_id == report['component'])
    found = {t.template_id:t for t in component.templates()}
    needed = {d['template'] for d in report['blocked_pair_decisions']}
    matched = set()
    for t in bank.templates():
        if t.template_id in needed:
            assert t.template_id not in matched
            found[t.template_id] = t
            matched.add(t.template_id)
    assert matched == needed
    return found


def test_previous_native_debt_is_accounted_for_without_double_counting(report):
    previous = report['previous_limits']
    assert len(previous['blocked_template_choices']) == 46
    assert len(previous['adapter_only_rule_ids']) == 195
    assert len(set(previous['adapter_only_rule_ids']) & {d['rule'] for d in previous['blocked_template_choices']}) == 25
    assert report['rules_reviewed'] == 210
    assert len(report['blocked_pair_decisions']) == 46
    assert len(report['adapter_rule_decisions']) == 195
    assert report['remaining_blocked_pairs'] == report['remaining_adapter_only_rules'] == 0
    assert FOLLOWUP['native_limits']['blocked_template_choices'] == []
    assert FOLLOWUP['native_limits']['adapter_only_rule_ids'] == []
    assert FOLLOWUP['remaining_dialogue_restrictions'] == 0
    assert report['original_patterns_unchanged'] and report['previous_bank_ranges_unchanged']
    assert report['checks']['max_tokens'] <= 768 and not report['checks']['overlength']
    assert not any(report['checks']['split_prompt_overlap'].values())


def test_every_previously_blocked_pair_has_an_exact_license(templates, report):
    for d in report['blocked_pair_decisions']:
        t=templates[d['template']];r=RULES[d['rule']];before=deepcopy(t.record)
        result=render_template(t,r,d['choice'])
        assert result['text']==d['target'] != d['text']
        assert result['usage_constraints']['mode']=='context_only'
        assert not result['slots'] and not result['predicate_groups']
        assert result['usage_constraints']['meaning']['interpretation']==d['interpretation']
        assert t.render()['text']==d['text'] and t.record==before
        # Removing the individual permission restores the original protection.
        unlicensed=deepcopy(r)
        unlicensed['phrase_choices'][t.template_id]=[
            x for x in unlicensed['phrase_choices'][t.template_id] if x['choice']!=d['choice']]
        assert render_template(t,unlicensed,d['choice']) is None


def test_all_adapter_only_rules_now_have_complete_native_turns(templates,report):
    indexed={p['id']:p for p in CATALOG['patterns']}
    covered={};variants=0
    for link in report['new_template_links']:
        t=templates[link['template']];r=RULES[link['rule']];p=indexed[link['adapter']]
        assert t.render()['text']==p['reply']==p['source']['context'][p['source']['turn']-1]
        assert fingerprint(t.record)==r['templates'][t.template_id]
        for i,choice in enumerate(r['choices']):
            if substitute(p['reply'],choice)==p['reply']:continue
            row=render_template(t,r,i)
            assert row['text']==substitute(p['reply'],choice)
            assert row['usage_constraints']['meaning']['interpretation']==substitute(p['value'],choice)
            assert not row['slots'] and not row['predicate_groups']
            doc=Document('finite',row['text'],metadata={k:v for k,v in row.items() if k!='text'})
            assert TemplateAnnotator(entities=EntityPool([])).annotate(doc).run is None
            covered.setdefault(r['id'],set()).add(i);variants+=1
        assert t.sample(recombine=False)['text']==p['reply']
        assert t.sample(recombine=True)['text']!=p['reply']
    assert len(report['new_template_links'])==report['new_templates']==243
    assert variants==report['checks']['native_new_variants']
    assert set(covered)==set(report['previous_limits']['adapter_only_rule_ids'])
    for rid,choices in covered.items():assert choices==set(range(len(RULES[rid]['choices'])))


def test_full_turn_interpretations_preserve_stage_directions_and_incomplete_words(report):
    indexed={p['id']:p for p in load_fragment_patterns()}
    assert len(report['new_complete_patterns'])==7
    gym=indexed['dd_native_complete_1471_2']
    assert 'singsong' in gym['reply'] and 'precise game is unresolved' in gym['value']
    assert 'no exercise or visit is completed' in gym['value']
    stage=realize_pattern(indexed['dd_native_complete_1483_1_substitution'],0)
    assert "Helen's hand" in stage['reply'] and "Helen's hand" in stage['value']
    assert 'other instructions is not established' in stage['value']
    song=realize_pattern(indexed['dd_native_complete_1483_6_substitution'],0)
    assert "I'm fling, Daniel." in song['reply']
    assert 'do not report an actual flight' in song['value']
    date=realize_pattern(indexed['dd_native_complete_1454_1_substitution'],0)
    assert 'Friday' in date['reply'] and 'next Friday' in date['value']
    assert 'acceptance are not established' in date['value']


def test_contextual_changes_keep_alternatives_hyphenation_and_noncompletion():
    indexed={p['id']:p for p in load_fragment_patterns()}
    def variant(key):return realize_pattern(indexed[key+'_substitution'],0)
    q=variant('dd_review_turn_2621_1');a=variant('dd_review_turn_2621_2')
    assert 'coffee or tea' in q['reply'] and 'coffee or tea' in q['value']
    assert 'always be coffee' in a['reply'] and 'prefers coffee' in a['value']
    parking=variant('dd_backfill_turn_1529_5')
    assert 'car park' in parking['reply'] and 'car-park development' in parking['value']
    assert 'not confirmed completed' in parking['value']
    boat=variant('dd_review_turn_3202_3')
    assert 'no model boat' in boat['reply'] and 'no model boat' in boat['value']
    death=variant('dd_review_turn_3219_4')
    assert 'Mr. Wilson' in death['reply'] and 'attributed to the ward' in death['value']


def test_source_synchronization_keeps_reversible_before_and_after(report):
    path=ROOT.parents[1]/'data/dailydialog/filtered/dailydialog-train.jsonl'
    lines=path.read_text(encoding='utf8').splitlines()
    assert len(report['source_corrections'])==11
    for d in report['source_corrections']:
        assert json.loads(lines[d['dialogue']-1])[d['turn']-1]==d['after']
        assert fingerprint(d['before'])==d['before_sha256']
        assert fingerprint(d['after'])==d['after_sha256']
        assert d['before']!=d['after'] and d['reason']
