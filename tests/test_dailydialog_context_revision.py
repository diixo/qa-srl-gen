"""Context reviews must never spread to identical text in another dialogue."""
from copy import deepcopy
import gzip
import json
import sqlite3

import pytest

from scripts import process_dailydialog_hybrid as full
from scripts.revise_dailydialog_hybrid_context import apply_review, overlay, read_reviews, verify_conservation
from semantic_corpus.template_bank import CorpusTemplate, fingerprint, read_rows
from semantic_corpus.template_compiler import LexicalAudit, compile_record
from tests.test_template_bank import parsed


@pytest.fixture
def sample(tmp_path):
    lexicon = tmp_path/'lexicon'
    lexicon.mkdir()
    (lexicon/'en_verb_inflections.txt').write_text('fire\tfires\tfiring\tfired\tfired\n')
    (lexicon/'en_postags_withverb.txt').write_text('fire\tnoun verb\n')
    audit = LexicalAudit(lexicon)
    doc = parsed(['Fire', '!'], [0, 0], ['ROOT', 'punct'],
                 ['NOUN', 'PUNCT'], ['NN', '.'], ['fire', '!'])
    record, ledger = compile_record(doc, {}, audit)
    payload = dict(record=record, ledger=ledger, flags=['no_verb_or_fragment'],
        variant=full.variant_signature(record, ledger), entities=[],
        review_status='UNREVIEWED', method='local_spacy_syntax')
    body = dict(idx='test:1:2:1', example=doc.text, provenance=dict(
        origin=dict(split='test', dialogue=1, turn=2, segment=1, span=[0,len(doc.text)]),
        original=doc.text, context_id='test:1', cache_key='shared', boundary_flags=[]))
    context = ['Ready to shoot?', doc.text]
    review = dict(id=body['idx'], source_sha256=fingerprint(doc.text),
        context_sha256=fingerprint(context), payload_sha256=fingerprint(payload),
        method='direct_assistant_review', unit='sentence', evidence_turns=[1,2],
        meaning=dict(speech_act='command', event_status='directive', content={'action':'shoot'},
                     completed_action_asserted=False, unknown=['whether anyone shoots']),
        note='Command in a shooting context, not a report of a fire.',
        token_edits=[dict(token=0, set=dict(pos='VERB',tag='VB',morph='VerbForm=Inf'))])
    return body, payload, context, review, audit, doc.vocab, lexicon


def test_correction_preserves_source_alternatives_and_records_partial_scope(sample):
    body, payload, context, review, audit, vocab, _ = sample
    before = deepcopy((body, payload, review))
    changed, result = apply_review(body, payload, context, review, audit, vocab)
    assert (body, payload, review) == before
    assert CorpusTemplate(result['record']).render()['text'] == body['example']
    assert result['record']['verbs'] == ['fire']
    assert result['record']['slots']['v0']['analysis'] == 'model_context_review'
    assert result['ledger']['tokens'][0]['previous_syntax']['pos'] == 'NOUN'
    assert result['ledger']['tokens'][0]['verb_lemmas'] == ['fire']
    assert len(result['ledger']['tokens']) == len(payload['ledger']['tokens'])
    assert result['review_status'] == 'UNREVIEWED'
    annotation = changed['provenance']['context_review']
    assert not annotation['meaning']['completed_action_asserted']
    assert not annotation['template_semantics_approved']
    assert not annotation['unseen_group_members_approved']
    assert changed['provenance']['cache_key'] != body['provenance']['cache_key']


def test_scope_restrictions_survive_occurrence_review_without_approving_family(sample):
    body, payload, context, review, audit, vocab, _ = sample
    review['usage'] = dict(mode='attested_sentence', reasons=['command_not_completed_action'])
    changed, result = apply_review(body, payload, context, review, audit, vocab)
    usage = result['record']['usage_constraints']
    assert usage['context'] == context and usage['meaning']['completed_action_asserted'] is False
    assert result['record']['template_id'] == result['ledger']['template_id']
    assert changed['provenance']['context_review']['usage'] == review['usage']
    assert not changed['provenance']['context_review']['template_semantics_approved']
    assert 'usage_constraints' not in payload['record']


@pytest.mark.parametrize('field', ['source_sha256','context_sha256','payload_sha256'])
def test_stale_preimage_is_rejected(sample, field):
    body, payload, context, review, audit, vocab, _ = sample
    review[field] = 'stale'
    with pytest.raises(ValueError, match='preimage'):
        apply_review(body, payload, context, review, audit, vocab)


def test_same_text_in_another_context_cannot_reuse_decision(sample):
    body, payload, context, review, audit, vocab, _ = sample
    with pytest.raises(ValueError, match='preimage'):
        apply_review(body, payload, ['What is burning?',body['example']], review, audit, vocab)


@pytest.mark.parametrize('edit', [
    {'token':0,'set':{'head':99}}, {'token':0,'set':{'text':'Shoot'}},
    {'token':0,'set':{'head':1}}, {'token':-1,'set':{'tag':'VB'}},
])
def test_invalid_syntax_edits_fail_closed(sample, edit):
    body, payload, context, review, audit, vocab, _ = sample
    review['token_edits'] = [edit]
    with pytest.raises(ValueError):
        apply_review(body, payload, context, review, audit, vocab)


def test_existing_reviewed_reference_cannot_be_changed(sample):
    body, payload, context, review, audit, vocab, _ = sample
    payload['review_status'] = 'REVIEWED_REUSE'
    review['payload_sha256'] = fingerprint(payload)
    with pytest.raises(ValueError, match='unreviewed'):
        apply_review(body, payload, context, review, audit, vocab)


def test_compiler_normalization_cannot_silently_change_requested_syntax(sample):
    body, payload, context, review, audit, vocab, _ = sample
    # The compiler stores lowercase lemmas, including for proper names/titles.
    # Report this mismatch before rebuilding the whole bank.
    review['token_edits'][0]['set']['parser_lemma'] = 'Fire'
    with pytest.raises(ValueError, match='compiled syntax differs'):
        apply_review(body, payload, context, review, audit, vocab)


def test_meaning_only_review_retains_syntax_and_fragment(sample):
    body, payload, context, review, audit, vocab, _ = sample
    review.update(unit='fragment', token_edits=[])
    review['meaning'].update(speech_act='warning', event_status='reported_hazard', content={'hazard':'fire'})
    changed, result = apply_review(body, payload, context, review, audit, vocab)
    assert result['record'] == payload['record'] and result['ledger'] == payload['ledger']
    assert changed['provenance']['context_review']['unit'] == 'fragment'


def test_existing_context_review_cannot_be_replaced(sample):
    body, payload, context, review, audit, vocab, _ = sample
    body['provenance']['context_review'] = {'note':'previous protected decision'}
    with pytest.raises(ValueError, match='overwritten'):
        apply_review(body,payload,context,review,audit,vocab)


def test_reference_overlay_preserves_reviewed_tokens_and_cannot_license_publication(sample):
    body,payload,context,review,audit,vocab,_ = sample
    payload.update(method='reviewed_whole_dialogue_reuse',review_status='REVIEWED_REUSE')
    review.update(payload_sha256=fingerprint(payload),reference_overlay=True,token_edits=[],
                  usage=dict(mode='context_only',reasons=['candidate_local_refinement']))
    before = deepcopy(payload)
    changed,result = apply_review(body,payload,context,review,audit,vocab)
    assert payload==before
    assert result['ledger']['tokens']==payload['ledger']['tokens']
    assert result['review_status']=='REVIEWED_REUSE'
    assert changed['provenance']['context_review']['reference_overlay'] is True
    review['usage']['mode']='attested_sentence'
    with pytest.raises(ValueError,match='unreviewed'):
        apply_review(body,payload,context,review,audit,vocab)
    review['usage']['mode']='context_only'
    review['token_edits']=[dict(token=0,set=dict(pos='VERB'))]
    with pytest.raises(ValueError,match='unreviewed'):
        apply_review(body,payload,context,review,audit,vocab)


@pytest.mark.parametrize('field',['automatic_revision','attribute_revision'])
def test_prior_repair_cannot_disappear_during_recompilation(sample,field):
    body, payload, context, review, audit, vocab, _ = sample
    payload['ledger'][field] = {'rules':['earlier repair']}
    review['payload_sha256'] = fingerprint(payload)
    with pytest.raises(ValueError, match='reconciliation'):
        apply_review(body,payload,context,review,audit,vocab)
    review['token_edits'] = []
    _, result = apply_review(body,payload,context,review,audit,vocab)
    assert result['ledger'] == payload['ledger']


def test_cumulative_batches_preserve_previous_decisions_and_reject_duplicates(tmp_path):
    first, second = tmp_path/'first.jsonl', tmp_path/'second.jsonl'
    previous = dict(id='train:1:1:1', token_edits=[{'token':0,'set':{'pos':'VERB'}}])
    additional = dict(id='train:2:1:1', token_edits=[])
    first.write_text(json.dumps(previous)+'\n', encoding='utf-8')
    second.write_text(json.dumps(additional)+'\n', encoding='utf-8')
    assert read_reviews([first, second]) == [previous, additional]
    with pytest.raises(ValueError, match='duplicate'):
        read_reviews([first, second, first])
    with pytest.raises(ValueError, match='empty'):
        read_reviews([])


def test_pending_meaning_is_recorded_without_inventing_a_syntax_repair(sample):
    body, payload, context, review, audit, vocab, _ = sample
    review.update(unit='pending', token_edits=[])
    review['meaning']['unknown'] = ['unresolved source']
    changed, result = apply_review(body, payload, context, review, audit, vocab)
    assert changed['provenance']['context_review']['unit'] == 'pending'
    assert 'context_pending' in result['flags']
    assert result['record'] == payload['record']
    review['token_edits'] = [dict(token=0, set=dict(pos='VERB'))]
    with pytest.raises(ValueError, match='guessed'):
        apply_review(body, payload, context, review, audit, vocab)


def test_perfect_auxiliary_repair_keeps_negation_and_active_roles(sample):
    body, _, _, review, audit, _, _ = sample
    from spacy.tokens import Doc
    import spacy
    doc = Doc(spacy.blank('en').vocab, words=['He', "'s", 'never', 'asked', '.'],
        spaces=[False,True,True,False,False], heads=[3,3,3,3,3],
        deps=['nsubjpass','auxpass','neg','ROOT','punct'],
        pos=['PRON','AUX','ADV','VERB','PUNCT'], tags=['PRP','VBZ','RB','VBN','.'],
        lemmas=['he','be','never','ask','.'])
    record, ledger = compile_record(doc, {}, audit)
    payload = dict(record=record,ledger=ledger,entities=[],flags=[],
        variant=full.variant_signature(record,ledger),method='local_spacy_syntax',review_status='UNREVIEWED')
    body['example'] = body['provenance']['original'] = doc.text
    body['provenance']['origin']['span'] = [0,len(doc.text)]
    context = ['Has he asked?', doc.text]
    review.update(source_sha256=fingerprint(doc.text),context_sha256=fingerprint(context),
        payload_sha256=fingerprint(payload),token_edits=[dict(token=0,set=dict(dep='nsubj')),
        dict(token=1,set=dict(dep='aux',parser_lemma='have'))])
    review['meaning'] = dict(speech_act='answer',event_status='negated_action',
        content={'predicate':'ask','polarity':'never'},completed_action_asserted=False,unknown=[])
    changed, result = apply_review(body,payload,context,review,audit,doc.vocab)
    assert CorpusTemplate(result['record']).render()['text'] == "He's never asked."
    assert result['record']['verbs'] == ['ask','have']
    assert result['record']['slots']['n0']['dependency'] == 'nsubj'
    assert result['record']['slots']['v0']['dependency'] == 'aux'
    assert result['ledger']['tokens'][1]['previous_syntax']['parser_lemma'] == 'be'
    assert result['ledger']['tokens'][2] == ledger['tokens'][2]
    assert result['variant']['operators'][1]['lemma'] == 'never'
    assert changed['provenance']['context_review']['meaning']['completed_action_asserted'] is False


def test_correct_passive_is_not_changed_by_a_meaning_only_review(sample):
    body, _, _, review, audit, _, _ = sample
    doc = parsed(['It', "'s", 'being', 'processed', '.'], [3,3,3,3,3],
        ['nsubjpass','aux','auxpass','ROOT','punct'], ['PRON','AUX','AUX','VERB','PUNCT'],
        ['PRP','VBZ','VBG','VBN','.'], ['it','be','be','process','.'])
    record, ledger = compile_record(doc, {}, audit)
    payload = dict(record=record,ledger=ledger,entities=[],flags=[],
        variant=full.variant_signature(record,ledger),method='local_spacy_syntax',review_status='UNREVIEWED')
    body['example'] = body['provenance']['original'] = doc.text
    body['provenance']['origin']['span'] = [0,len(doc.text)]
    context = ['Is my application ready?', doc.text]
    review.update(source_sha256=fingerprint(doc.text),context_sha256=fingerprint(context),
        payload_sha256=fingerprint(payload),token_edits=[])
    review['meaning'].update(event_status='ongoing_passive_process',completed_action_asserted=False)
    _, result = apply_review(body,payload,context,review,audit,doc.vocab)
    assert result['record'] == record and result['ledger'] == ledger
    assert result['record']['slots']['n0']['dependency'] == 'nsubjpass'


def test_bank_materialization_keeps_context_local_and_inputs_readonly(sample, tmp_path):
    body, payload, context, review, audit, vocab, lexicon = sample
    original, lexical = tmp_path/'original.sqlite3', tmp_path/'lexical.sqlite3'
    db = full.connect(original)
    db.execute('INSERT INTO parses VALUES (?,?,?)', ('shared',body['example'],json.dumps(payload)))
    other = deepcopy(body)
    other['idx'] = 'test:2:2:1'
    other['provenance']['origin']['dialogue'] = 2
    other['provenance']['context_id'] = 'test:2'
    for row in (body, other):
        db.execute('INSERT INTO occurrences VALUES (?,?,?)', (row['idx'],'shared',json.dumps(row)))
    db.commit(); db.close()
    db = sqlite3.connect(lexical)
    db.execute('CREATE TABLE replacements (key TEXT PRIMARY KEY, payload TEXT)')
    db.commit(); db.close()
    hashes = [original.read_bytes(), lexical.read_bytes()]
    before = tmp_path/'before'; before.mkdir()
    baseline = overlay(original,lexical,before/'revision.sqlite3')
    out = tmp_path/'out'; out.mkdir()
    revised = overlay(original, lexical, out/'revision.sqlite3')
    changed, result = apply_review(body, payload, context, review, audit, vocab)
    revised.execute('INSERT INTO corrections VALUES (?,?,?,?,?,?)', (review['id'],
        changed['provenance']['cache_key'],body['example'],json.dumps(changed),json.dumps(result),json.dumps(review)))
    for table in ('original.parses', 'lexical.replacements'):
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            revised.execute(f"UPDATE {table} SET payload='bad'")
    with gzip.open(out/'contexts.jsonl.gz','wt',encoding='utf-8') as stream:
        for identity,turns in [('test:1',context),('test:2',['What is burning?',body['example']])]:
            stream.write(json.dumps(dict(id=identity,turns=turns))+'\n')
    with gzip.open(out/'coverage.jsonl.gz','wt',encoding='utf-8') as stream:
        stream.write('{}\n')
    for name in ('contexts.jsonl.gz','coverage.jsonl.gz'):
        (before/name).write_bytes((out/name).read_bytes())
    full.export(baseline,before)
    full.materialize(baseline,before,{'name':'test'},lexicon)
    baseline.close()
    counts = full.export(revised, out)
    assert counts['context_reviewed_occurrences'] == counts['context_syntax_corrections'] == 1
    manifest, validation = full.materialize(revised, out, {'name':'test'}, lexicon)
    assert validation['status'] == 'passed' and validation['source_records'] == 2
    rows = list(read_rows(out/'bank/occurrences/00000.jsonl.gz'))
    assert rows[0]['tokens'][0]['kind'] == 'verb'
    assert rows[0]['context_review']['meaning']['event_status'] == 'directive'
    assert rows[1]['tokens'][0]['kind'] == 'nominal'
    assert 'context_review' not in rows[1]
    assert all(r['semantic_review'] == 'UNREVIEWED' for r in rows)
    assert not manifest['hybrid_import']['published']
    conservation = verify_conservation(before,out,[review])
    assert conservation['counts']['identical_occurrences'] == 1
    assert conservation['counts']['syntax_corrections'] == 1
    # A syntactically plausible change outside the reviewed occurrence must fail.
    rows[1]['tokens'][0]['head'] = 1
    with gzip.open(out/'bank/occurrences/00000.jsonl.gz','wt',encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='unaddressed occurrence'):
        verify_conservation(before,out,[review])
    revised.close()
    assert [original.read_bytes(), lexical.read_bytes()] == hashes
