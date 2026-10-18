"""Bulk rules must preserve scope, passive voice, evidence and reviewed records."""
from copy import deepcopy
import gzip
import json

import pytest

from scripts.audit_dailydialog_hybrid_bank import WholeBankAudit, issues, proposals, repair, repair_adjective_coordination, open_overlay
from scripts.benchmark_dailydialog_hybrid import risk_flags
from scripts.process_dailydialog_hybrid import variant_signature
from semantic_corpus.template_bank import CorpusTemplate
from semantic_corpus.template_compiler import LexicalAudit, compile_record
from tests.test_template_bank import parsed


@pytest.fixture
def audit(tmp_path):
    directory=tmp_path/'lexicon';directory.mkdir()
    (directory/'en_verb_inflections.txt').write_text('say\tsays\tsaying\tsaid\tsaid\nhave\thas\thaving\thad\thad\n')
    (directory/'en_postags_withverb.txt').write_text('')
    return LexicalAudit(directory)


def payload(doc,audit):
    record,ledger=compile_record(doc,{},audit)
    return dict(record=record,ledger=ledger,flags=risk_flags(doc,record,ledger),entities=[],
        method='local_spacy_syntax',review_status='UNREVIEWED',variant=variant_signature(record,ledger))


def active(audit):
    doc=parsed(['He',"'s",'never','been','here','.'],[3,3,3,3,3,3],
        ['nsubjpass','auxpass','neg','ROOT','advmod','punct'],
        ['PRON','AUX','ADV','AUX','ADV','PUNCT'],['PRP','VBZ','RB','VBN','RB','.'],
        ['he','be','never','be','here','.'])
    return doc,payload(doc,audit)


def test_contracted_perfect_keeps_negation_and_is_idempotent(audit):
    doc,before=active(audit);saved=deepcopy(before)
    after=repair(before,audit,doc.vocab)
    assert before==saved
    assert after['ledger']['tokens'][0]['dep']=='nsubj'
    assert after['ledger']['tokens'][1]['lemma']=='have'
    assert after['ledger']['tokens'][1]['dep']=='aux'
    assert after['ledger']['tokens'][2]==before['ledger']['tokens'][2]
    assert 'negation' in after['flags'] and after['review_status']=='UNREVIEWED'
    assert not after['ledger']['automatic_revision']['semantic_approval']
    assert CorpusTemplate(after['record']).render()['text']==doc.text
    assert repair(after,audit,doc.vocab) is None


def test_perfect_passive_retains_passive_participle_and_subject(audit):
    doc=parsed(['He',"'s",'been','elected','.'],[3,3,3,3,3],
        ['nsubjpass','aux','auxpass','ROOT','punct'],['PRON','AUX','AUX','VERB','PUNCT'],
        ['PRP','VBZ','VBN','VBN','.'],['he','be','be','elect','.'])
    after=repair(payload(doc,audit),audit,doc.vocab)
    ts=after['ledger']['tokens']
    assert ts[1]['lemma']=='have' and ts[2]['dep']=='auxpass' and ts[0]['dep']=='nsubjpass'


def test_progressive_lemma_does_not_assert_completion(audit):
    doc=parsed(['Jack',"'s",'working','.'],[2,2,2,2],['nsubj','aux','ROOT','punct'],
        ['PROPN','AUX','VERB','PUNCT'],['NNP','POS','VBG','.'],['jack',"'s",'work','.'])
    after=repair(payload(doc,audit),audit,doc.vocab)
    assert after['ledger']['tokens'][1]['lemma']=='be'
    assert after['ledger']['tokens'][2]['tag']=='VBG'
    assert 'meaning' not in after['ledger']['automatic_revision']


@pytest.mark.parametrize('lemma', ['be','have'])
def test_ambiguous_contracted_participle_is_not_guessed(audit,lemma):
    doc=parsed(['It',"'s",'broken','.'],[2,2,2,2],['nsubjpass','auxpass','ROOT','punct'],
        ['PRON','AUX','VERB','PUNCT'],['PRP','VBZ','VBN','.'],['it',lemma,'break','.'])
    assert repair(payload(doc,audit),audit,doc.vocab) is None


def test_possessive_s_is_not_changed(audit):
    doc=parsed(['Jack',"'s",'painting','.'],[2,0,2,2],['poss','case','ROOT','punct'],
        ['PROPN','PART','NOUN','PUNCT'],['NNP','POS','NN','.'],['jack',"'s",'painting','.'])
    assert repair(payload(doc,audit),audit,doc.vocab) is None


def test_contracted_copula_attached_to_another_clause_is_not_an_is_has_choice(audit):
    doc=parsed(['It',"'s",'important',',','I','heard','.'],[1,5,1,5,5,5,5],
        ['nsubj','ccomp','acomp','punct','nsubj','ROOT','punct'],
        ['PRON','AUX','ADJ','PUNCT','PRON','VERB','PUNCT'],
        ['PRP','VBZ','JJ',',','PRP','VBN','.'],['it','be','important',',','I','hear','.'])
    before=payload(doc,audit)
    saved=deepcopy(before)
    reasons=issues(before,{})
    assert 'participle_state_or_perfect' not in reasons
    assert 'contraction_nonlocal_head' in reasons
    assert before==saved and before['review_status']=='UNREVIEWED'


def test_direct_passive_still_requires_review(audit):
    doc=parsed(['It',"'s",'broken','.'],[2,2,2,2],['nsubjpass','auxpass','ROOT','punct'],
        ['PRON','AUX','VERB','PUNCT'],['PRP','VBZ','VBN','.'],['it','be','break','.'])
    reasons=issues(payload(doc,audit),{})
    assert 'participle_state_or_perfect' in reasons
    assert 'contraction_nonlocal_head' not in reasons


def test_progressive_passive_is_preserved(audit):
    doc=parsed(['It',"'s",'being','processed','.'],[3,3,3,3,3],
        ['nsubjpass','aux','auxpass','ROOT','punct'],['PRON','AUX','AUX','VERB','PUNCT'],
        ['PRP','VBZ','VBG','VBN','.'],['it','be','be','process','.'])
    assert repair(payload(doc,audit),audit,doc.vocab) is None


def test_full_form_have_uses_dictionary_to_verify_participle(audit):
    doc=parsed(['I','have','said','no','.'],[2,2,2,2,2],['nsubj','aux','ROOT','dobj','punct'],
        ['PRON','AUX','VERB','INTJ','PUNCT'],['PRP','VBP','VBD','UH','.'],['i','have','say','no','.'])
    after=repair(payload(doc,audit),audit,doc.vocab)
    assert after['ledger']['tokens'][2]['tag']=='VBN'
    assert after['ledger']['tokens'][2]['verb_lemmas']==['say']


def test_causative_have_does_not_make_finite_past_a_participle(audit):
    doc=parsed(['I','had','Jack','work','.'],[1,1,3,1,1],['nsubj','ROOT','nsubj','ccomp','punct'],
        ['PRON','VERB','PROPN','VERB','PUNCT'],['PRP','VBD','NNP','VB','.'],['i','have','jack','work','.'])
    assert repair(payload(doc,audit),audit,doc.vocab) is None


def test_non_nominal_adjective_target_becomes_unresolved_not_invented(audit):
    doc=parsed(['Walking','is','nice','.'],[1,1,1,1],['nsubj','ROOT','acomp','punct'],
        ['VERB','AUX','ADJ','PUNCT'],['VBG','VBZ','JJ','.'],['walk','be','nice','.'])
    before=payload(doc,audit)
    # An old compiled bank could attach nice to verbal Walking. Construct that
    # legacy defect explicitly; the current compiler no longer creates it.
    for collection in (before['record']['attributes'], before['ledger']['attributes']):
        collection[0]['target_slot']='v0'
    assert before['record']['attributes'][0]['target_slot']=='v0'
    after=repair(before,audit,doc.vocab)
    assert after['record']['attributes'][0]['target_slot'] is None
    assert after['ledger']['tokens']==before['ledger']['tokens']
    assert after['ledger']['automatic_revision']['unresolved_adjective_targets'][0]['previous_target']=='v0'
    assert 'adjective_target_unresolved' in issues(after,{})
    assert repair(after,audit,doc.vocab) is None


def test_context_reviews_and_reused_manual_syntax_are_protected(audit):
    doc,before=active(audit)
    assert repair(before,audit,doc.vocab,protected=True) is None
    before['review_status']='REVIEWED_REUSE'
    assert repair(before,audit,doc.vocab) is None


def test_uncertain_source_and_quotes_do_not_trigger_grammar_rules(audit):
    _,before=active(audit)
    for marker in ('\ufffd','"','\u201c'):
        ledger=deepcopy(before['ledger']);ledger['text']+=marker
        assert proposals(ledger,audit)==[]


def prepare_audit(tmp_path,text):
    source=tmp_path/'source';source.mkdir()
    out=tmp_path/'out';out.mkdir()
    with gzip.open(source/'contexts.jsonl.gz','wt',encoding='utf-8') as f:
        f.write(json.dumps(dict(id='train:1',turns=[text]))+'\n')
    with gzip.open(source/'coverage.jsonl.gz','wt',encoding='utf-8') as f:
        f.write(json.dumps(dict(id='train:1:1',original=text,segments=[dict(id='train:1:1:1',span=[0,len(text)],text=text)]))+'\n')
    row=dict(example=text,provenance=dict(origin=dict(turn=1,span=[0,len(text)]),context_id='train:1',original=text))
    return WholeBankAudit(source,out),row


def test_full_audit_rejects_missing_and_duplicate_occurrences(audit,tmp_path):
    doc,data=active(audit)
    checker,row=prepare_audit(tmp_path,doc.text)
    with pytest.raises(ValueError,match='unprocessed'):
        checker.finish()
    # Reopen only the output stream for the valid audit; source remains unchanged.
    checker.stream=gzip.open(checker.out/'syntax-audit.jsonl.gz','wt',encoding='utf-8')
    checker.observe('train:1:1:1',row,data)
    with pytest.raises(ValueError,match='duplicate'):
        checker.observe('train:1:1:1',row,data)
    result=checker.finish()
    assert result['counts']['occurrences']==1 and result['dialogues']==1
    assert result['semantic_approval'] is False


def test_full_audit_rejects_changed_context_address(audit,tmp_path):
    doc,data=active(audit)
    checker,row=prepare_audit(tmp_path,doc.text)
    row['provenance']['original']='unrelated text'
    try:
        with pytest.raises(ValueError,match='address'):
            checker.observe('train:1:1:1',row,data)
    finally:
        checker.stream.close()


def coordination(audit, subject='room', conjunction='and'):
    doc=parsed([subject,'is','quiet',conjunction,'comfortable','.'],[1,1,1,2,2,1],
        ['nsubj','ROOT','acomp','cc','conj','punct'],['NOUN','AUX','ADJ','CCONJ','ADJ','PUNCT'],
        ['NN','VBZ','JJ','CC','JJ','.'],[subject.lower(),'be','quiet',conjunction,'comfortable','.'])
    result=payload(doc,audit)
    # Preserve a fixture from the older compiler's missing-conjunct linkage.
    for collection in (result['record']['attributes'], result['ledger']['attributes']):
        collection[1]['target_slot']=None
    return result


def test_adjective_coordination_adds_only_target_and_retains_prior_audit(audit):
    before=coordination(audit)
    before['ledger']['automatic_revision']={'version':'prior','rules':['earlier_rule']}
    saved=deepcopy(before)
    after=repair_adjective_coordination(before)
    assert before==saved
    assert after['record']['attributes'][1]['target_slot']=='n0'
    assert after['ledger']['tokens']==before['ledger']['tokens']
    assert after['ledger']['automatic_revision']==before['ledger']['automatic_revision']
    assert after['ledger']['attributes'][0]==before['ledger']['attributes'][0]
    assert after['ledger']['attribute_revision']['attribute_changes'][0]['previous_target'] is None
    assert not after['ledger']['attribute_revision']['semantic_approval']
    assert repair_adjective_coordination(after) is None
    assert CorpusTemplate(after['record']).render()['text']==before['ledger']['text']


@pytest.mark.parametrize('subject,conjunction',[('It','and'),('room','or'),('room','nor')])
def test_ambiguous_it_and_disjunctive_scope_are_not_propagated(audit,subject,conjunction):
    assert repair_adjective_coordination(coordination(audit,subject,conjunction)) is None


def test_pre_nominal_coordination_preserves_negation_and_modifiers(audit):
    doc=parsed(['a','nice','but','not','expensive','room'],[5,5,1,4,1,5],
        ['det','amod','cc','neg','conj','ROOT'],['DET','ADJ','CCONJ','PART','ADJ','NOUN'],
        ['DT','JJ','CC','RB','JJ','NN'],['a','nice','but','not','expensive','room'])
    before=payload(doc,audit)
    for collection in (before['record']['attributes'], before['ledger']['attributes']):
        collection[1]['target_slot']=None
    after=repair_adjective_coordination(before)
    assert after['ledger']['attributes'][1]['target_slot']=='n0'
    assert after['ledger']['attributes'][1]['modifiers']==before['ledger']['attributes'][1]['modifiers']
    assert after['ledger']['tokens']==before['ledger']['tokens']
    assert after['ledger']['attribute_revision']['attribute_changes'][0]['coordinators'][0]['text']=='but'


def test_separate_subject_and_complement_block_coordination(audit):
    before=coordination(audit)
    before['ledger']['tokens'][0]['head']=4
    assert repair_adjective_coordination(before) is None
    before=coordination(audit)
    before['ledger']['tokens'][0]['head']=4
    before['ledger']['tokens'][0]['dep']='xcomp'
    assert repair_adjective_coordination(before) is None


def test_conflicting_existing_target_and_manual_review_are_preserved(audit):
    before=coordination(audit)
    before['record']['attributes'][1]['target_slot']='a0'
    assert repair_adjective_coordination(before) is None
    assert repair_adjective_coordination(coordination(audit),protected=True) is None
    before=coordination(audit);before['review_status']='REVIEWED_REUSE'
    assert repair_adjective_coordination(before) is None


def test_non_copular_resultative_does_not_inherit_subject(audit):
    before=coordination(audit)
    before['ledger']['tokens'][1]['parser_lemma']='make'
    assert repair_adjective_coordination(before) is None


def test_repeated_comparative_is_not_two_properties(audit):
    doc=parsed(['more','and','more','time'],[3,0,0,3],['amod','cc','conj','ROOT'],
        ['ADJ','CCONJ','ADJ','NOUN'],['JJR','CC','JJR','NN'],['more','and','more','time'])
    assert repair_adjective_coordination(payload(doc,audit)) is None


def test_revision_overlay_carries_prior_repairs_and_context_decisions(tmp_path):
    import sqlite3
    def plan(folder, **fields):
        folder.mkdir()
        (folder/'plan.json').write_text(json.dumps(fields))
    original=tmp_path/'original';original.mkdir()
    with sqlite3.connect(original/'cache.sqlite3') as db:
        db.execute('CREATE TABLE parses (key TEXT PRIMARY KEY,text TEXT,payload TEXT)')
        db.executemany('INSERT INTO parses VALUES (?,?,?)',[('a','A','original-a'),('b','B','original-b')])
        db.execute('CREATE TABLE occurrences (id TEXT PRIMARY KEY,key TEXT,body TEXT)')
        db.executemany('INSERT INTO occurrences VALUES (?,?,?)',[('1','a','body-a'),('2','b','body-b')])
    lexical=tmp_path/'lexical';plan(lexical,source=str(original),lexical_resources={})
    with sqlite3.connect(lexical/'revision.sqlite3') as db:
        db.execute('CREATE TABLE replacements (key TEXT PRIMARY KEY,payload TEXT)')
        db.execute("INSERT INTO replacements VALUES ('a','lexical-a')")
    context=tmp_path/'context';plan(context,source=str(lexical))
    source=tmp_path/'source';plan(source,source=str(context))
    with sqlite3.connect(source/'revision.sqlite3') as db:
        db.execute('CREATE TABLE corrections (id TEXT PRIMARY KEY,key TEXT UNIQUE,text TEXT,body TEXT,payload TEXT,review TEXT)')
        db.execute("INSERT INTO corrections VALUES ('2','c','B','reviewed-body','reviewed-payload','review')")
        db.execute('CREATE TABLE automatic_repairs (key TEXT PRIMARY KEY,payload TEXT)')
        db.execute("INSERT INTO automatic_repairs VALUES ('a','bulk-a')")
        db.execute('CREATE TABLE retired_occurrences (id TEXT PRIMARY KEY,replacement TEXT NOT NULL,snapshot TEXT NOT NULL)')
        db.execute("INSERT INTO retired_occurrences VALUES ('1','2','original-snapshot')")
    out=tmp_path/'out';out.mkdir()
    db,_,_,_=open_overlay(source,out)
    try:
        assert db.execute("SELECT payload FROM parses WHERE key='a'").fetchone()==('bulk-a',)
        assert db.execute("SELECT id FROM occurrences ORDER BY id").fetchall()==[('2',)]
        assert db.execute("SELECT * FROM retired_occurrences").fetchall()==[('1','2','original-snapshot')]
        assert db.execute("SELECT key,body FROM occurrences WHERE id='2'").fetchone()==('c','reviewed-body')
        db.execute("INSERT OR REPLACE INTO automatic_repairs VALUES ('a','next-a')")
        assert db.execute("SELECT payload FROM source_parses WHERE key='a'").fetchone()==('bulk-a',)
        assert db.execute("SELECT payload FROM parses WHERE key='a'").fetchone()==('next-a',)
        assert db.execute("SELECT payload FROM parses WHERE key='c'").fetchone()==('reviewed-payload',)
        assert db.execute("SELECT payload FROM previous.automatic_repairs WHERE key='a'").fetchone()==('bulk-a',)
    finally:
        db.close()
