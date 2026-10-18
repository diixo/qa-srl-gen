from copy import deepcopy

from scripts.dailydialog_attribute_targets import repair
from semantic_corpus.template_bank import CorpusTemplate
from tests.test_dailydialog_bank_audit import audit, payload
from tests.test_template_bank import parsed


def test_only_the_unsupported_property_link_is_removed(audit):
    doc=parsed(['I','know','much'],[1,1,1],['nsubj','ROOT','dobj'],
               ['PRON','VERB','ADJ'],['PRP','VBP','JJ'],['i','know','much'])
    data=payload(doc,audit)
    for attrs in (data['record']['attributes'],data['ledger']['attributes']):attrs[0]['target_slot']='n0'
    original=deepcopy(data)
    revised=repair(data,audit,doc.vocab)
    assert data==original
    assert revised['ledger']['tokens']==data['ledger']['tokens']
    assert revised['record']['attributes'][0]['target_slot'] is None
    assert revised['ledger']['attribute_target_revision']['preimage']==original
    assert revised['review_status']=='UNREVIEWED'
    assert CorpusTemplate(revised['record']).render()['text']==doc.text
    assert repair(revised,audit,doc.vocab) is None
    assert repair(data,audit,doc.vocab,protected=True) is None
