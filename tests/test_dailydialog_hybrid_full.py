"""Coverage, context isolation and whole-sentence construction regression tests."""
import json
from copy import deepcopy

import pytest

from scripts.process_dailydialog_hybrid import (
    connect, export, materialize, prepare, reference_groups, split_sentences, variant_signature,
)
from semantic_corpus.template_bank import read_rows
from tests.test_dailydialog_hybrid import native


@pytest.mark.parametrize('text,expected',[
    ("Well, I didn't clean the house and I didn't study.", ["Well, I didn't clean the house and I didn't study."]),
    ("I want your advice but I don't want you to tell me what to do.", ["I want your advice but I don't want you to tell me what to do."]),
    ('No. I used to have one though.', ['No.','I used to have one though.']),
    ('Well, so long, Mary. Have fun.', ['Well, so long, Mary.','Have fun.']),
    ('Meet Prof. Smith at 5 PM. Saturday.', ['Meet Prof. Smith at 5 PM. Saturday.']),
    ('It is 3.5 miles. Go there.', ['It is 3.5 miles.','Go there.']),
    ('What a storm.it was terrible.', ['What a storm.','it was terrible.']),
    ('She said, "Go away!" but I stayed.', ['She said, "Go away!" but I stayed.']),
    ('Well... I might.', ['Well... I might.']),
    ('(A few minutes later.) We have locked your card.', ['(A few minutes later.)','We have locked your card.']),
    ('He smiles (as expected). Then leaves.', ['He smiles (as expected).','Then leaves.']),
])
def test_boundaries_preserve_whole_clauses_and_source(text,expected):
    spans,_=split_sentences(text)
    assert [text[a:b] for a,b in spans]==expected
    assert ''.join(text[a:b] for a,b in spans).replace(' ','')==text.replace(' ','')


def reference(dialogue=1):
    text='Anna saw Beth.'
    return dict(split='train',dialogue=dialogue,turn=1,original=text,names=[],segments=[dict(
        status='sentence',original=text,text=text,correction=None,record=native(),
        origin=dict(split='train',dialogue=dialogue,turn=1,segment=1,span=[0,len(text)]))])


def test_conflicting_review_is_not_reused():
    first,second=reference(),reference(2)
    second['segments'][0]['record']['slots']['n0']['dependency']='dobj'
    _,reusable,conflicts=reference_groups([first,second])
    assert not reusable and len(conflicts)==1


def test_known_reference_defect_is_not_propagated():
    _,reusable,conflicts=reference_groups([reference(6)])
    assert not reusable and len(conflicts)==1


def test_full_coverage_and_context_safe_reuse(tmp_path):
    directory=tmp_path/'data/dailydialog/filtered'; directory.mkdir(parents=True)
    inputs={'train':[['Anna saw Beth.'],['Where are you?','At school.'],
                     ['Where do you study?','At school.'],['Anna saw Beth.']],
            'val':[['Hello!']], 'test':[['I’m home.']]}
    for split,rows in inputs.items():
        (directory/f'dailydialog-{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    out=tmp_path/'out'; out.mkdir()
    db=connect(out/'cache.sqlite3')
    counts=prepare(tmp_path,out,db,[reference()])
    assert counts['remaining_dialogues']==5 and counts['remaining_turns']==7
    assert counts['preserved_reviewed_dialogues']==1 and counts['reviewed_dialogue_reuses']==1
    rows={i:json.loads(body) for i,body in db.execute('SELECT id,body FROM occurrences')}
    a,b=rows['train:2:2:1'],rows['train:3:2:1']
    assert a['provenance']['cache_key']==b['provenance']['cache_key']
    assert a['provenance']['context_id']!=b['provenance']['context_id']
    assert a['provenance']['reused_from'] is None and b['provenance']['reused_from'] is None
    assert a['provenance']['semantic_review']==b['provenance']['semantic_review']=='UNREVIEWED'
    reused=rows['train:4:1:1']['provenance']
    assert reused['reused_from']=='train:1:1:1' and reused['origin']['dialogue']==4
    normalized=rows['test:1:1:1']
    assert normalized['example']=="I'm home."
    assert normalized['provenance']['original']=='I’m home.'
    assert normalized['provenance']['correction']=='normalize_apostrophes'
    coverage=list(read_rows(out/'coverage.jsonl.gz'))
    assert len(coverage)==7
    assert len(list(read_rows(out/'contexts.jsonl.gz')))==5
    db.close()


@pytest.mark.parametrize('apostrophe',['\u2019','\u2018','\u201b','\u02bc','\u02bb','\uff07','`'])
def test_apostrophes_normalize_without_changing_raw_text_or_offsets(tmp_path,apostrophe):
    text = 'I'+apostrophe+'m home.'
    directory=tmp_path/'data/dailydialog/filtered'; directory.mkdir(parents=True)
    for split in ('train','val','test'):
        (directory/f'dailydialog-{split}.jsonl').write_text(json.dumps([text])+'\n',encoding='utf-8')
    out=tmp_path/'out';out.mkdir()
    db=connect(out/'cache.sqlite3')
    prepare(tmp_path,out,db,[])
    for raw, in db.execute('SELECT body FROM occurrences'):
        row=json.loads(raw)
        assert row['example']=="I'm home."
        assert row['provenance']['original']==text
        assert row['provenance']['origin']['span']==[0,len(text)]
        assert row['provenance']['correction']=='normalize_apostrophes'
    db.close()


def test_noncanonical_reviewed_copy_is_not_silently_rewritten(tmp_path):
    ref=reference()
    ref['segments'][0]['text']='Anna\u2019s home.'
    before=deepcopy(ref)
    _,reusable,conflicts=reference_groups([ref])
    assert not reusable and len(conflicts)==1
    assert ref==before


def test_tense_negation_intention_are_distinct_groups():
    def signature(text,lemma,tag,neg=False):
        tokens=[dict(text=text,kind='verb',lemma=lemma,tag=tag,dep='ROOT',head=0)]
        if neg:
            tokens.append(dict(text='not',lemma='not',kind=None,dep='neg',head=0))
        return variant_signature({'predicate_groups':[]},{'tokens':tokens,'text':text})
    variants=[signature('picked','pick','VBD'),signature('pick','pick','VB',True),signature('wanted','want','VBD')]
    assert len({json.dumps(v,sort_keys=True) for v in variants})==3


def test_native_bank_reuse_retains_new_address_and_original_context(tmp_path):
    directory=tmp_path/'data/dailydialog/filtered'; directory.mkdir(parents=True)
    for split in ('train','val','test'):
        (directory/f'dailydialog-{split}.jsonl').write_text('["Anna saw Beth."]\n',encoding='utf-8')
    out=tmp_path/'out'; out.mkdir()
    lexicon=tmp_path/'lexicon'; lexicon.mkdir()
    (lexicon/'en_verb_inflections.txt').write_text('see\tsees\tseeing\tsaw\tseen\n')
    (lexicon/'en_postags_withverb.txt').write_text('see\tverb\n')
    db=connect(out/'cache.sqlite3')
    prepare(tmp_path,out,db,[reference()])
    summary=export(db,out)
    assert summary['compiled_occurrences']==2
    manifest,validation=materialize(db,out,{'name':'test-reviewed-cache'},lexicon)
    assert manifest['review_status']=='UNREVIEWED'
    assert not manifest['hybrid_import']['published']
    assert validation['status']=='passed' and validation['source_records']==2
    occurrence=list(read_rows(out/'bank/occurrences/00000.jsonl.gz'))[0]
    assert occurrence['idx']=='val:1:1:1'
    assert occurrence['origin']['split']=='val'
    assert occurrence['reused_from']=='train:1:1:1'
    assert occurrence['semantic_review']=='UNREVIEWED'
    contexts=list(read_rows(out/'bank/contexts.jsonl.gz'))
    assert {r['id'] for r in contexts}=={'val:1','test:1'}
    db.close()
