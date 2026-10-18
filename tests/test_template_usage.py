"""Reviewed context must survive rendering and must not become asserted facts."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from semantic_corpus.documents import Document
from semantic_corpus.semantic_generator.substitutions import EntityPool
from semantic_corpus.template_bank import CorpusTemplate
from semantic_corpus.template_semantics import TemplateAnnotator
from semantic_corpus.template_usage import constrain_record
from semantic_corpus.template_registry import _check_hybrid_admission
from semantic_corpus.dialogue_generator.mined_realization import active_phrase
from tests.test_template_semantics import template
from tests.test_template_bank import parsed
from semantic_corpus.template_compiler import LexicalAudit, compile_record


def restricted(mode='attested_sentence'):
    record = template()
    text = CorpusTemplate(record).render()['text']
    body = dict(idx='test:1:2:1', example=text, provenance=dict(origin=dict(turn=2, span=[0,len(text)])))
    review = dict(unit='sentence' if mode=='attested_sentence' else 'fragment',
                  meaning={'event_status': 'reported'}, usage=dict(mode=mode, reasons=['reported_scope']))
    constrain_record(record, body, ['What did Anna do?', text], review)
    return record


def test_recombination_and_direct_replacement_cannot_escape_context_review():
    record = restricted()
    original = deepcopy(record)
    # Foreign fillers cannot turn an attestation into newly "reviewed" data.
    record['fillers']['n0'].append('Ben')
    candidate = CorpusTemplate(record)
    for seed in range(4):
        rendered = candidate.sample(seed=seed, recombine=True)
        assert rendered['text'] == original['usage_constraints']['text']
        assert rendered['usage_constraints']['context'][0] == 'What did Anna do?'
    with pytest.raises(ValueError, match='slot replacement'):
        candidate.render({'n0':'Ben'})
    rendered['usage_constraints']['meaning'].clear()
    assert record['usage_constraints']['meaning'] == original['usage_constraints']['meaning']


def test_context_review_blocks_predicate_and_property_qa_and_action_realization():
    record = restricted()
    row = CorpusTemplate(record).render()
    doc = Document('context-test', row['text'], metadata={k:v for k,v in row.items() if k!='text'})
    result = TemplateAnnotator(entities=EntityPool([])).annotate(doc)
    assert result.run is None  # Including the otherwise extractable adjective.
    assert result.document.metadata['template_semantics']['skipped'][0]['reason']=='reviewed_context_only'
    assert active_phrase(record, 'v0', 'buy', (), set()) == (None, 'reviewed_context_only')


def test_review_does_not_merge_with_same_surface_in_another_context():
    first, second = restricted(), restricted()
    text = second['usage_constraints']['text']
    body = dict(idx='test:2:1:1',example=text,provenance=dict(origin=dict(turn=1,span=[0,len(text)])))
    constrain_record(second,body,[text],dict(unit='sentence',meaning={'event_status':'unverified'},
                                           usage=dict(mode='attested_sentence',reasons=['context_differs'])))
    assert first['template_id'] != second['template_id']


@pytest.mark.parametrize('mode', ['context_only','pending'])
def test_fragments_and_pending_cases_cannot_be_published(mode):
    record = restricted(mode)
    record['count']=1
    bank = SimpleNamespace(manifest={'hybrid_import':{'review_status':'REVIEWED_ATTESTATIONS',
                            'admitted_occurrences':['test:1:2:1']},'counts':{'compiled_records':1}},
                           templates=lambda:iter([CorpusTemplate(record)]))
    with pytest.raises(ValueError,match='individually reviewed'):
        _check_hybrid_admission(bank)


def test_quarantine_cannot_be_published_just_because_roundtrips_pass():
    bank=SimpleNamespace(manifest={'hybrid_import':{'review_status':'UNREVIEWED'}})
    with pytest.raises(ValueError,match='quarantine'):
        _check_hybrid_admission(bank)


def test_embedded_context_must_still_match_observed_text():
    record=restricted()
    record['usage_constraints']['context'][1]='A different sentence.'
    with pytest.raises(ValueError,match='embedded context'):
        CorpusTemplate(record).render()


def test_disjunction_gets_target_without_becoming_two_asserted_properties(tmp_path):
    lexicon=tmp_path/'lexicon';lexicon.mkdir()
    (lexicon/'en_verb_inflections.txt').write_text('')
    (lexicon/'en_postags_withverb.txt').write_text('')
    doc=parsed(['Are','you','identical','or','fraternal','?'],[0,0,0,2,2,0],
        ['ROOT','nsubj','acomp','cc','conj','punct'],
        ['AUX','PRON','ADJ','CCONJ','ADJ','PUNCT'],['VBP','PRP','JJ','CC','JJ','.'],
        ['be','you','identical','or','fraternal','?'])
    record,_=compile_record(doc,{},LexicalAudit(lexicon))
    body=dict(idx='test:1:1:1',example=doc.text,provenance=dict(origin=dict(turn=1,span=[0,len(doc.text)])))
    review=dict(unit='sentence',meaning={'speech_act':'alternative_question'},
        usage=dict(mode='attested_sentence',reasons=['questioned_alternatives'],
            alternative_properties=[dict(slots=['a0','a1'],target_slot='n0',operator='or',scope='question')]))
    rendered=constrain_record(record,body,[doc.text],review)
    assert all(a['target_slot']=='n0' and a['scope']=='question_alternative' for a in rendered['attributes'])
    assert rendered['text']==doc.text
    assert TemplateAnnotator(entities=EntityPool([])).annotate(
        Document('alternatives',doc.text,metadata=rendered)).run is None


def test_idiom_span_cannot_be_silently_rebound():
    record=restricted()
    record['usage_constraints']['fixed_expressions']=[dict(text='under the weather',span=[0,5],meaning='unwell')]
    with pytest.raises(ValueError,match='fixed expression'):
        CorpusTemplate(record).render()


@pytest.mark.parametrize('kind,target', [('clause','v0'), ('nominal','n0'),
                                      ('free_relative','n0'), ('implicit_participant',None)])
def test_reviewed_property_target_preserves_scope_and_blocks_factual_qa(kind,target):
    record = restricted()
    text = record['usage_constraints']['text']
    adjective = record['attributes'][0]['slot']
    binding = dict(slot=adjective,target_slot=target,target_kind=kind,scope='hypothetical_evaluation')
    usage = dict(mode='attested_sentence',reasons=['contextual_property'],property_bindings=[binding])
    body = dict(idx='test:1:2:1',example=text,provenance=dict(origin=dict(turn=2,span=[0,len(text)])))
    rendered = constrain_record(record,body,['Context',text],dict(unit='sentence',meaning={'scope':'hypothetical'},usage=usage))
    attribute = next(a for a in rendered['attributes'] if a['slot']==adjective)
    assert all(attribute[k]==v for k,v in binding.items())
    assert attribute['target_span']==rendered['spans'].get(target)
    assert CorpusTemplate(record).sample(recombine=True)['text']==text
    assert TemplateAnnotator(entities=EntityPool([])).annotate(Document('p',text,metadata=rendered)).run is None


@pytest.mark.parametrize('kind,target', [('nominal','v0'), ('clause','n0'),
                                      ('implicit_participant','n0'), ('unknown',None)])
def test_inconsistent_property_target_kind_is_rejected(kind,target):
    record = restricted()
    record['usage_constraints']['property_bindings'] = [dict(slot=record['attributes'][0]['slot'],
        target_slot=target,target_kind=kind,scope='question')]
    with pytest.raises(ValueError,match='property'):
        CorpusTemplate(record).render()
