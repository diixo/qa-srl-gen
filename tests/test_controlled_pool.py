"""Working-bank exclusions never rewrite raw data or remove auxiliaries."""
import gzip
import json

import pytest

from semantic_corpus.controlled_verbs import controlled_head, excluded_predicates
from semantic_corpus.resource_pool import BuiltInPool
from semantic_corpus.semantic_generator.substitutions import EntityPool
from semantic_corpus.template_bank import FORMAT, TemplateBank, fingerprint


def template(text, verb_slots):
    parts, slots, defaults = [], {}, {}
    for index, word in enumerate(text.split()):
        if parts:
            parts.append({'literal': ' '})
        if index in verb_slots:
            lemma, dependency = verb_slots[index]
            key = f'v{index}'
            parts.append({'slot': key})
            slots[key] = {'kind': 'verb', 'pos': 'AUX' if dependency == 'aux' else 'VERB',
                          'lemma': lemma, 'dependency': dependency, 'tag': 'VB'}
            defaults[key] = word
        else:
            parts.append({'literal': word})
    row = {'parts': parts, 'slots': slots, 'defaults': defaults, 'predicate_groups': [],
           'verbs': sorted({slot['lemma'] for slot in slots.values()}), 'constructions': [],
           'attributes': [], 'attested_bindings': [defaults], 'fillers': {k: [v] for k, v in defaults.items()}}
    row['template_id'] = fingerprint(row)
    return row


@pytest.fixture
def bank(tmp_path):
    path = tmp_path / 'data/eng-base-templates'
    rows = [
        template('I have a book.', {1: ('have', 'ROOT')}),
        template('I had bought a book.', {1: ('have', 'aux'), 2: ('buy', 'ROOT')}),
        template('She is buying a book.', {1: ('be', 'aux'), 2: ('buy', 'ROOT')}),
        template('I can buy a book.', {1: ('can', 'aux'), 2: ('buy', 'ROOT')}),
        template('I went home.', {1: ('go', 'ROOT')}),
        template('I let him buy it.', {1: ('let', 'ROOT'), 3: ('buy', 'xcomp')}),
        template('I am here.', {1: ('be', 'ROOT')}),
        template('I want to go home.', {1: ('want', 'ROOT'), 3: ('go', 'xcomp')}),
    ]
    def write(file, values):
        file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(file, 'wt', encoding='utf-8') as stream:
            for row in values:
                stream.write(json.dumps(row) + '\n')
    for row in rows:
        write(path / 'templates' / f'{row["template_id"][:2]}.jsonl.gz',
              [item for item in rows if item['template_id'][:2] == row['template_id'][:2]])
    verbs = [{'lemma': word} for word in ('be', 'buy', 'can', 'could', 'go', 'have', 'let', 'may', 'might', 'must', 'want')]
    for name in ('verbs', 'verb_alternatives', 'constructions'):
        write(path / f'{name}.jsonl.gz', verbs)
    (path / 'manifest.json').write_text(json.dumps({'format': FORMAT, 'status': 'complete',
        'counts': {'templates': len(rows), 'verb_lemmas': len(verbs)}, 'source_sha256': 'a' * 64}), encoding='utf-8')
    return BuiltInPool(tmp_path, entities=EntityPool()), rows


def test_pool_filters_lexical_heads_but_keeps_raw_bank_and_auxiliary_uses(bank):
    pool, rows = bank
    before = {str(p): p.read_bytes() for p in pool.bank.path.rglob('*') if p.is_file()}
    assert len(list(TemplateBank(pool.bank.path).templates())) == len(rows)
    expected = {'I had bought a book.', 'She is buying a book.', 'I can buy a book.'}
    assert {row.render()['text'] for row in pool.templates()} == expected
    docs = [doc for split in ('train', 'dev', 'test') for doc in pool.mined_documents(split=split)]
    assert {doc.text for doc in docs} == expected
    assert all(not excluded_predicates(doc.metadata) for doc in docs)
    assert {row['lemma'] for row in pool.verbs()} == {'buy', 'want'}
    assert {row['lemma'] for row in pool.verb_alternatives()} == {'buy', 'want'}
    assert {row['lemma'] for row in pool.constructions()} == {'buy', 'want'}
    assert before == {str(p): p.read_bytes() for p in pool.bank.path.rglob('*') if p.is_file()}
    assert pool.summary()['counts_are_archived'] is True
    assert pool.metadata()['controlled_verbs'] == ['can', 'must', 'may', 'be', 'let', 'have', 'go']


@pytest.mark.parametrize('lemma', ['be', 'was', 'can', 'could', 'may', 'might', 'must', 'have', 'had', 'go', 'went', 'gone', 'go on', 'let'])
def test_explicit_mined_selection_cannot_bypass_controlled_routing(bank, lemma):
    pool, _ = bank
    with pytest.raises(ValueError, match='controlled grammar'):
        list(pool.templates(lemma=lemma))
    with pytest.raises(ValueError, match='controlled grammar'):
        list(pool.mined_documents(lemma=lemma))


def test_routing_is_not_a_substring_filter():
    assert controlled_head('candy') is None
    assert controlled_head('behave') is None
    assert not excluded_predicates(template('We behave.', {1: ('behave', 'ROOT')}))
    assert excluded_predicates(template('She has a book.', {1: ('have', 'ROOT')})) == {'have'}
