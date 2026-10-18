"""Dictionary homographs must survive without becoming unsupported actions."""
from copy import deepcopy
import json

import pytest

from scripts.benchmark_dailydialog_hybrid import risk_flags
from scripts.revise_dailydialog_hybrid_lexical import cached_doc, overlay, revise
from semantic_corpus.template_bank import CorpusTemplate, fingerprint
from semantic_corpus.template_compiler import LexicalAudit, compile_record
from tests.test_template_bank import parsed


@pytest.fixture
def audit(tmp_path):
    path = tmp_path/'lexicon'
    path.mkdir()
    (path/'en_verb_inflections.txt').write_text(''.join(
        f'{v}\t{v}s\t{v}ing\t{v}ed\t{v}ed\n' for v in ('yes','good','taxi','fire','jump','tom','one')))
    (path/'en_postags_withverb.txt').write_text('good\tadj verb\n')
    return LexicalAudit(path)


@pytest.mark.parametrize('word,pos,tag', [('Yes','INTJ','UH'),('Taxi','NOUN','NN'),
    ('Fire','NOUN','NN'),('Tom','PROPN','NNP'),('One','NUM','CD')])
def test_short_replies_and_ambiguous_commands_keep_dictionary_evidence(audit,word,pos,tag):
    doc = parsed([word,'!'], [0,0], ['ROOT','punct'], [pos,'PUNCT'], [tag,'.'], [word.lower(),'!'])
    record, ledger = compile_record(doc, {}, audit)
    assert CorpusTemplate(record).render()['text'] == doc.text
    assert not record['verbs'] and not record['predicate_groups']
    assert ledger['tokens'][0]['verb_lemmas'] == [word.lower()]
    assert ledger['imperative_candidates'][0]['review_status'] == 'UNREVIEWED'
    assert 'dictionary_imperative_candidate' in risk_flags(doc, record, ledger)


def test_greeting_adjective_keeps_its_nominal_target(audit):
    doc = parsed(['Good','morning','.'], [1,1,1], ['amod','ROOT','punct'],
                 ['ADJ','NOUN','PUNCT'], ['JJ','NN','.'], ['good','morning','.'])
    record, ledger = compile_record(doc, {}, audit)
    rendered = CorpusTemplate(record).render()
    assert record['verbs'] == []
    assert rendered['attributes'][0]['target_span'] == rendered['spans']['n0']
    assert record['slots']['a0']['kind'] == 'adjective'
    assert ledger['imperative_candidates'][0]['lemma'] == 'good'


@pytest.mark.parametrize('word', ['Taxi','Jump','Yes'])
def test_syntactically_verbal_readings_are_not_blacklisted(audit, word):
    doc = parsed([word,'!'], [0,0], ['ROOT','punct'], ['VERB','PUNCT'], ['VB','.'], [word.lower(),'!'])
    record, ledger = compile_record(doc, {}, audit)
    assert record['verbs'] == [word.lower()]
    assert record['slots']['v0']['tag'] == 'VB'
    assert not ledger['imperative_candidates']


def old_payload(audit):
    doc = parsed(['Yes','!'], [0,0], ['ROOT','punct'], ['INTJ','PUNCT'], ['UH','.'], ['yes','!'])
    record, ledger = compile_record(doc, {}, audit)
    ledger['tokens'][0].update(kind='verb',analysis='imperative_alternative')
    ledger.pop('imperative_candidates')
    ledger['analysis_status'] = 'imperative_alternative_added'
    record['verbs'] = ['yes']
    return doc, dict(record=record, ledger=ledger, entities=[], flags=['dictionary_recovery'],
        method='local_spacy_syntax', review_status='UNREVIEWED')


def test_cached_revision_preserves_text_syntax_alternatives_and_old_payload(audit):
    doc, payload = old_payload(audit)
    saved = deepcopy(payload)
    result = revise(payload, audit, doc.vocab)
    assert payload == saved
    assert result['record']['verbs'] == []
    assert result['ledger']['tokens'][0]['verb_lemmas'] == ['yes']
    assert CorpusTemplate(result['record']).render()['text'] == doc.text
    restored = cached_doc(result['ledger'], doc.vocab)
    assert [(t.text,t.pos_,t.dep_,t.head.i) for t in restored] == [(t.text,t.pos_,t.dep_,t.head.i) for t in doc]
    assert result['review_status'] == 'UNREVIEWED'
    payload['review_status'] = 'REVIEWED_REUSE'
    with pytest.raises(ValueError,match='unreviewed'):
        revise(payload, audit, doc.vocab)


def test_overlay_never_updates_original_cache(tmp_path):
    from scripts.process_dailydialog_hybrid import connect
    source = tmp_path/'original.sqlite3'
    db = connect(source)
    db.execute('INSERT INTO parses VALUES (?,?,?)', ('key','Yes.','old'))
    db.execute('INSERT INTO occurrences VALUES (?,?,?)', ('train:1:1:1','key','{}'))
    db.commit(); db.close()
    before = source.read_bytes()
    revised = overlay(source, tmp_path/'revision.sqlite3')
    revised.execute('INSERT INTO replacements VALUES (?,?,?)', ('key','new',fingerprint('old')))
    assert revised.execute('SELECT payload FROM parses').fetchone()[0] == 'new'
    assert revised.execute('SELECT rowid FROM occurrences').fetchone()[0] == 1
    import sqlite3
    with pytest.raises(sqlite3.OperationalError,match='readonly'):
        revised.execute("UPDATE original.parses SET payload='bad'")
    revised.commit(); revised.close()
    assert source.read_bytes() == before
