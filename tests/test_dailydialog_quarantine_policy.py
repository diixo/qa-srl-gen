"""Deferred sources cannot re-enter generation by relabelling an admission."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from semantic_corpus.dailydialog_quarantine import (
    REGISTER_NAME, check_fragment_catalog, check_template, load_holds,
)
from semantic_corpus.dialogue_generator.fragments import load_fragment_patterns
from semantic_corpus.template_bank import fingerprint
from semantic_corpus.template_registry import _check_hybrid_admission, register_bank
from scripts.admit_dailydialog_fragments import admit
from scripts.admit_dailydialog_hybrid import compile_admission
from scripts.audit_dailydialog_quarantine import verify_evidence
from scripts.import_dailydialog_sentences import digest
from tests.test_template_registry import make_bank


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def held():
    text = 'You have to be very c'
    source = dict(split='test', dialogue=364, turn=1, segment=1,
                  span=[0, len(text)], text=text)
    return dict(id='test:364:1:1', status='deferred_missing_evidence',
                reason='Required property is missing.', reopen_when='Obtain the source continuation.',
                unknown=['Required property'], template_admission=False, automatic_retry=False,
                allowed_adapters={}, source=source, original_prefix=[text],
                original_prefix_sha256=fingerprint([text]))


def save_hold(directory, entry):
    path = directory / REGISTER_NAME
    write(path, dict(format='dailydialog-quarantine-v1', occurrences=[entry]))
    return path


def pattern(entry):
    source = dict(entry['source'], context=entry['original_prefix'])
    return dict(id='reviewed-test', kind='interpretation', context=True,
                assistant='What do I have to do?', reply=source['text'],
                value='expressed an obligation with its property unknown',
                question='What did I convey?', source=source,
                review=dict(method='direct_assistant_fragment_adapter',
                            scope='adapted_visible_pair_only', adaptation='Keep unknown details.',
                            evidence_turns=[1], source_sha256=fingerprint(source['text']),
                            context_sha256=fingerprint(source['context'])))


def test_catalog_load_and_admission_reject_held_source_without_mutation(tmp_path, held):
    catalog = tmp_path/'dialogue-fragments.json'
    proposal = pattern(held)
    save_hold(tmp_path, held)
    original = dict(format='reviewed-dialogue-fragments-v1', patterns=[load_fragment_patterns()[0]])
    write(catalog, original)
    before = catalog.read_bytes()
    reviews, contexts = tmp_path/'reviews.json', tmp_path/'contexts.jsonl'
    write(reviews, dict(format='reviewed-fragment-adapters-v1', patterns=[proposal]))
    write(contexts, dict(id='test:364', turns=held['original_prefix']))
    with pytest.raises(ValueError, match='quarantine blocks fragment'):
        admit(reviews, contexts, catalog=catalog, publish=True)
    assert catalog.read_bytes() == before
    write(catalog, dict(format='reviewed-dialogue-fragments-v1', patterns=[proposal]))
    with pytest.raises(ValueError, match='quarantine blocks fragment'):
        load_fragment_patterns(catalog)


def test_only_exact_adjacent_adapter_survives_without_source_files(tmp_path, held):
    allowed = pattern(held)
    held.update(status='adapter_only', allowed_adapters={allowed['id']:fingerprint(allowed)})
    save_hold(tmp_path, held)
    catalog = tmp_path/'dialogue-fragments.json'
    write(catalog, dict(format='reviewed-dialogue-fragments-v1', patterns=[allowed]))
    assert load_fragment_patterns(catalog) == [allowed]
    changed = deepcopy(allowed)
    changed['value'] = 'performed the action'
    with pytest.raises(ValueError, match='quarantine blocks fragment'):
        check_fragment_catalog([changed], catalog)
    changed = deepcopy(allowed)
    changed['id'] = 'copy-with-new-name'
    with pytest.raises(ValueError, match='quarantine blocks fragment'):
        check_fragment_catalog([changed], catalog)
    # An adapter exception never grants permission to publish a whole template.
    with pytest.raises(ValueError, match='quarantine blocks template'):
        check_template({'usage_constraints':{'source_id':held['id']}}, load_holds(tmp_path/REGISTER_NAME))


def test_renumbering_or_joining_does_not_bypass_span_hold(tmp_path, held):
    path = save_hold(tmp_path, held)
    renamed = dict(held['source'], segment=99, span=[0, len(held['source']['text'])+20])
    with pytest.raises(ValueError, match='quarantine blocks source span'):
        check_template({'examples':[{'origin':renamed}]}, load_holds(path))
    proposal = pattern(held)
    proposal['source'] = renamed
    with pytest.raises(ValueError, match='quarantine blocks fragment'):
        check_fragment_catalog([proposal], tmp_path/'catalog.json')
    # Holds are occurrence-specific, not a global blacklist of English text.
    check_template({'examples':[{'origin':dict(renamed, dialogue=999)}]}, load_holds(path))
    check_template({'examples':[{'origin':dict(renamed, span=[100,110])}]}, load_holds(path))


def test_compile_cannot_bypass_hold_by_omitting_quarantine_option(tmp_path, held):
    repo = tmp_path/'repo'
    save_hold(repo/'data/mined-templates/reviewed', held)
    reviews = tmp_path/'reviews.jsonl'
    write(reviews, dict(id=held['id'], usage={'mode':'attested_sentence'}))
    out = tmp_path/'out'
    with pytest.raises(ValueError, match='quarantine blocks template'):
        compile_admission(tmp_path/'candidate', reviews, out, repo)
    assert not out.exists()


def test_registration_rechecks_holds_even_for_previously_compiled_bank(tmp_path, held):
    repo = tmp_path/'repo'
    save_hold(repo/'data/mined-templates/reviewed', held)
    directory, source, _ = make_bank(tmp_path/'source-bank')
    manifest = json.loads((directory/'manifest.json').read_text())
    manifest['hybrid_import'] = dict(review_status='REVIEWED_ATTESTATIONS',
                                    admitted_occurrences=[held['id']])
    write(directory/'manifest.json', manifest)
    before = {p.relative_to(repo):p.read_bytes() for p in repo.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='quarantine blocks template'):
        register_bank(directory, source, repo_root=repo, bank_id='held')
    assert before == {p.relative_to(repo):p.read_bytes() for p in repo.rglob('*') if p.is_file()}


def test_registry_checks_source_spans_after_ids_change(tmp_path, held):
    repo = tmp_path/'repo'
    save_hold(repo/'data/mined-templates/reviewed', held)
    bank = SimpleNamespace(manifest={'hybrid_import':{
        'review_status':'REVIEWED_ATTESTATIONS', 'admitted_occurrences':['test:364:1:99']}},
        templates=lambda: iter([SimpleNamespace(record={
            'examples':[{'origin':dict(held['source'], segment=99)}]})]))
    with pytest.raises(ValueError, match='quarantine blocks source span'):
        _check_hybrid_admission(bank, repo_root=repo)


@pytest.mark.parametrize('defect', ['source', 'prefix', 'automatic_retry', 'admission', 'exception'])
def test_invalid_register_is_not_silently_ignored(tmp_path, held, defect):
    if defect == 'source':
        held['source']['text'] = 'changed source'
    elif defect == 'prefix':
        held['original_prefix_sha256'] = 'stale'
    elif defect == 'automatic_retry':
        held['automatic_retry'] = True
    elif defect == 'admission':
        held['template_admission'] = True
    else:
        held['allowed_adapters'] = {'anything':'a'*64}
    with pytest.raises(ValueError, match='quarantine|reviewed adjacent'):
        load_holds(save_hold(tmp_path, held))


@pytest.mark.parametrize('change', ['append', 'edit', 'prepend', 'remove', 'metadata', 'missing_snapshot', 'other_evidence'])
def test_evidence_allows_catalog_append_but_preserves_historical_content(tmp_path, change):
    name = 'data/mined-templates/reviewed/dialogue-fragments.json'
    path = tmp_path/name
    path.parent.mkdir(parents=True)
    baseline = dict(format='reviewed-dialogue-fragments-v1', patterns=[{'id':'old','value':'unknown'}])
    path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    register = dict(evidence={name:digest(path)}, catalog_evidence=dict(
        path=name, patterns=1, serialization='utf8-json-indent2-lf'))
    current = deepcopy(baseline)
    current['patterns'].append({'id':'new','value':'reviewed'})
    if change == 'edit':current['patterns'][0]['value'] = 'completed'
    elif change == 'prepend':current['patterns'].reverse()
    elif change == 'remove':current['patterns'] = []
    elif change == 'metadata':current['format'] = 'changed'
    elif change == 'missing_snapshot':register.pop('catalog_evidence')
    elif change == 'other_evidence':
        other = tmp_path/'source.txt'
        other.write_text('original', encoding='utf-8')
        register['evidence']['source.txt'] = digest(other)
        other.write_text('changed', encoding='utf-8')
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    if change == 'append':
        verify_evidence(tmp_path, register)
    else:
        with pytest.raises(ValueError, match='Review evidence changed'):
            verify_evidence(tmp_path, register)
