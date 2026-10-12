"""Mechanical validation of model-authored reviews; no NLP model required."""
from copy import deepcopy
import json
import pytest
from scripts.import_dailydialog_sentences import check_turn, compile_reviews, partition, publish, reviews
from semantic_corpus.template_bank import TemplateBank, fingerprint, read_rows
from semantic_corpus.template_validation import validate_bank


def reviewed_turn():
    text='Anna arrived.'
    shape=dict(parts=[{'slot':'n0'},{'literal':' '},{'slot':'v0'},{'literal':'.'}],
               slots={'n0':dict(kind='nominal',pos='PROPN',tag='NNP',dependency='nsubj',head_slot='v0'),
                      'v0':dict(kind='verb',pos='VERB',tag='VBD',lemma='arrive',dependency='ROOT',head_slot='v0')},
               verbs=['arrive'],constructions=[],predicate_groups=[dict(head_slot='v0',lemma='arrive',constructions=[])],attributes=[])
    record=dict(**shape,template_id=fingerprint(shape),defaults={'n0':'Anna','v0':'arrived'},adjective_lemmas=[])
    return dict(split='train',dialogue=1,turn=1,original=text,segments=[dict(
        status='sentence',text=text,original=text,origin=dict(split='train',dialogue=1,turn=1,segment=1,span=[0,len(text)]),
        note='',correction=None,record=record)])


def test_native_review_compiles_and_renders_without_nlp(tmp_path):
    row=reviewed_turn()
    check_turn(row,row['original'])
    lexicon=tmp_path/'dictionary'
    lexicon.mkdir()
    (lexicon/'en_verb_inflections.txt').write_text('arrive\tarrives\tarriving\tarrived\tarrived\n')
    (lexicon/'en_postags_withverb.txt').write_text('arrive\tverb\n')
    source,manifest,report=compile_reviews([row],tmp_path/'bank',tmp_path/'work',lexicon)
    bank=TemplateBank(tmp_path/'bank')
    assert validate_bank(bank,source)==report
    assert 'spacy_version' not in manifest
    source.unlink()
    assert next(bank.generate())['text']=='Anna arrived.'
    assert next(bank.occurrences())['origin']['turn']==1


@pytest.mark.parametrize('damage',['gap','span','text','identity'])
def test_incomplete_or_inconsistent_review_is_rejected(damage):
    row=deepcopy(reviewed_turn())
    if damage=='gap':
        row['original']+=' More.'
    elif damage=='span':
        row['segments'][0]['origin']['span'][0]=2
    elif damage=='text':
        row['segments'][0]['text']='Anna left.'
    else:
        row['segments'][0]['origin']['turn']=2
    with pytest.raises(ValueError):
        check_turn(row,row['original'])


def test_partition_preserves_all_dialogues_and_origin(tmp_path):
    source=tmp_path/'source'
    source.mkdir()
    for split in ('train','val','test'):
        (source/f'dailydialog-{split}.jsonl').write_text(''.join(json.dumps([f'{split} {i}','Hello!'])+'\n' for i in range(3)))
    parts=tmp_path/'parts'
    manifest=partition(source,parts,number=4)
    assert [p['dialogues'] for p in manifest['parts']]==[3,2,2,2]
    assert manifest['turns']==18
    records=[r for p in manifest['parts'] for r in read_rows(parts/p['file'])]
    assert len({(r['split'],r['dialogue']) for r in records})==9
    assert partition(source,parts,number=4)==manifest
    _,progress=reviews(parts)
    assert progress['reviewed_turns']==0 and not progress['complete']


def test_review_publication_deduplicates_turns_and_rejects_changed_review(tmp_path):
    source=tmp_path/'data/dailydialog/filtered'
    source.mkdir(parents=True)
    row=reviewed_turn()
    for split in ('train','val','test'):
        (source/f'dailydialog-{split}.jsonl').write_text(json.dumps([row['original']])+'\n')
    parts=tmp_path/'data/dailydialog/review-parts'
    partition(source,parts,number=3)
    annotation=parts/'annotations/part-001.jsonl'
    annotation.write_text(json.dumps(row)+'\n')
    lexicon=tmp_path/'data/wiktionary'
    lexicon.mkdir()
    (lexicon/'en_verb_inflections.txt').write_text('arrive\tarrives\tarriving\tarrived\tarrived\n')
    (lexicon/'en_postags_withverb.txt').write_text('arrive\tverb\n')
    assert publish(parts,tmp_path)['newly_published_turns']==1
    bank=tmp_path/'data/mined-templates'
    before=(bank/'manifest.json').read_bytes()
    assert publish(parts,tmp_path)['newly_published_turns']==0
    assert (bank/'manifest.json').read_bytes()==before
    assert not (bank/'sources').exists()
    row['segments'][0]['note']='Changed analysis.'
    annotation.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='published review changed'):
        publish(parts,tmp_path)
