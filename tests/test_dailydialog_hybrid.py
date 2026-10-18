"""Quality comparison must neither certify surface matches nor lose context."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts.benchmark_dailydialog_hybrid import (
    aligned_nominals, compare, corpus_index, features, load_reference,
    risk_flags, select_review, sentence_spans,
)
from scripts.import_dailydialog_sentences import compile_reviews
from semantic_corpus.template_bank import fingerprint


def native():
    shape = dict(
        parts=[{'slot':'n0'},{'literal':' '},{'slot':'v0'},{'literal':' '},{'slot':'n1'},{'literal':'.'}],
        slots={
            'n0':dict(kind='nominal',pos='PROPN',tag='NNP',dependency='nsubj',head_slot='v0'),
            'v0':dict(kind='verb',pos='VERB',tag='VBD',dependency='ROOT',head_slot='v0',lemma='see'),
            'n1':dict(kind='nominal',pos='PROPN',tag='NNP',dependency='dobj',head_slot='v0'),
        },verbs=['see'],constructions=[],predicate_groups=[dict(head_slot='v0',lemma='see',constructions=[])],attributes=[])
    return dict(**shape,template_id=fingerprint(shape),defaults={'n0':'Anna','v0':'saw','n1':'Beth'},adjective_lemmas=[])


def test_same_surface_different_roles_is_not_agreement():
    reference = native()
    automatic = deepcopy(reference)
    automatic['slots']['n0']['dependency']='dobj'
    automatic['slots']['n1']['dependency']='nsubj'
    result=compare(reference,automatic)
    assert result['verbs'] and result['verb_scope']
    assert not result['nominal_roles'] and not result['nominal_roles_containment']


def test_slot_renaming_does_not_create_difference():
    reference=native()
    renamed=deepcopy(reference)
    mapping={'n0':'subject','n1':'object','v0':'predicate'}
    renamed['slots']={mapping[k]:dict(v,head_slot=mapping[v['head_slot']]) for k,v in renamed['slots'].items()}
    renamed['defaults']={mapping[k]:v for k,v in renamed['defaults'].items()}
    renamed['parts']=[{'slot':mapping[p['slot']]} if 'slot' in p else p for p in renamed['parts']]
    renamed['predicate_groups'][0]['head_slot']='predicate'
    assert all(compare(reference,renamed).values())


def test_nominal_width_relaxation_preserves_dependency_and_rejects_ambiguity():
    manual=[[[0,12],'nsubj',[13,17],None]]
    automatic=[[[4,12],'nsubj',[13,17],None]]
    assert aligned_nominals(manual,automatic)
    assert not aligned_nominals(manual,[[[4,12],'dobj',[13,17],None]])
    assert not aligned_nominals(manual,automatic+[[[0,3],'nsubj',[13,17],None]])
    assert not aligned_nominals(manual,[[[4,12],'nsubj',[22,26],None]])


def test_penn_tense_difference_remains_visible():
    reference=native(); proposed=deepcopy(reference)
    proposed['slots']['v0']['tag']='VBP'
    assert not compare(reference,proposed)['verbs']


def test_sentence_boundaries_account_for_all_non_whitespace():
    spacy=pytest.importorskip('spacy')
    nlp=spacy.blank('en'); nlp.add_pipe('sentencizer')
    text='Anna arrived, but Beth left.  Did she call?'
    spans=sentence_spans(nlp(text))
    assert [text[a:b] for a,b in spans]==['Anna arrived, but Beth left.','Did she call?']
    broken=SimpleNamespace(text='Hello! Bye.',sents=[SimpleNamespace(start_char=0,end_char=6)])
    with pytest.raises(ValueError,match='remainder'):
        sentence_spans(broken)


def test_negative_and_intent_examples_are_routed_even_when_they_render():
    row=native()
    row['slots']['v0']['lemma']='want'
    flags=risk_flags(SimpleNamespace(text="Anna doesn't want to leave."),row,{'analysis_status':'parsed_verbal'})
    assert {'negation','modality_or_intention'} <= set(flags)


def test_complete_dialogue_duplicates_share_cohort_but_short_replies_do_not_reuse(tmp_path):
    source=tmp_path/'source'; source.mkdir()
    rows=[['Did you leave?','I did.'],['Did you buy it?','I did.'],['Did you leave?','I did.']]
    for split in ('train','val','test'):
        (source/f'dailydialog-{split}.jsonl').write_text(
            ''.join(json.dumps(r)+'\n' for r in (rows if split=='train' else [['Hello.']])),encoding='utf-8')
    references=[dict(split='train',dialogue=d,turn=t,original=text)
                for d in (1,3) for t,text in enumerate(rows[d-1],1)]
    counts,cohorts,_=corpus_index(source,references,tmp_path)
    assert counts['reviewed_dialogues']==2
    assert counts['remaining_dialogues']==3
    assert counts['remaining_novel_dialogue_texts']==2
    assert counts['remaining_whole_dialogue_reuse_candidates']==1
    assert cohorts['train:1:1']==cohorts['train:3:1']
    entries=[json.loads(s) for s in (tmp_path/'coverage.jsonl').read_text().splitlines()]
    assert entries[1]['exact_dialogue_duplicate_of'] is None
    assert entries[2]['exact_dialogue_duplicate_of']=='train:1'


def test_stale_published_source_is_rejected(tmp_path):
    for split in ('train','val','test'):
        (tmp_path/f'dailydialog-{split}.jsonl').write_text('["Hello."]\n')
    references=[dict(split='train',dialogue=1,turn=1,original='Goodbye.')]
    with pytest.raises(ValueError,match='differs'):
        corpus_index(tmp_path,references,tmp_path)


def test_reference_is_recoverable_from_published_bank_without_review_parts(tmp_path):
    text='Anna saw Beth.'
    row=dict(split='train',dialogue=1,turn=1,original=text,names=[],segments=[dict(
        status='sentence',original=text,text=text,correction=None,note='',record=native(),
        origin=dict(split='train',dialogue=1,turn=1,segment=1,span=[0,len(text)]))])
    lexicon=tmp_path/'lexicon'; lexicon.mkdir()
    (lexicon/'en_verb_inflections.txt').write_text('see\tsees\tseeing\tsaw\tseen\n')
    (lexicon/'en_postags_withverb.txt').write_text('see\tverb\n')
    source,_,_=compile_reviews([row],tmp_path/'bank',tmp_path/'work',lexicon)
    source.unlink()
    recovered=load_reference(tmp_path/'bank')
    assert len(recovered)==1 and recovered[0]['original']==text
    assert features(recovered[0]['segments'][0]['record'])==features(native())


def test_sample_includes_apparent_successes_and_is_deterministic():
    rows=[]
    for flags in ([],['negation']):
        for equal in (True,False):
            for i in range(25):
                rows.append(dict(id=f'{flags}:{equal}:{i}',kind='syntax',flags=flags,equal={'verbs':equal}))
    rows.extend(dict(id=f'b:{i}',kind='boundary',equal=False) for i in range(15))
    result=select_review(rows)
    assert len(result)==60
    assert len({r['id'] for r in result})==60
    assert select_review(list(reversed(rows)))==result
    assert {r['selection_reason'] for r in result}=={
        'boundary_difference','unflagged_agreement','unflagged_difference','flagged_agreement','flagged_difference'}
