"""Mined dependency trees become auditable, non-gold semantic QA offline."""

from dataclasses import replace
import gzip
import json

import pytest

from semantic_corpus.documents import AnnotationRun, Document
from semantic_corpus.ontology import Label, LabelSet, Relation, ReviewStatus
from semantic_corpus.question_generator import QuestionGenerator
from semantic_corpus.resource_pool import BuiltInPool, mined_split
from semantic_corpus.semantic_generator import DEFAULT_FRAMES
from semantic_corpus.semantic_generator.generator import DEFAULT_PARADIGMS
from semantic_corpus.semantic_generator.substitutions import Entity, EntityPool
from semantic_corpus.template_bank import CorpusTemplate, FORMAT, fingerprint
from semantic_corpus.template_semantics import TemplateAnnotator, SOURCE, VERSION, needs_annotation


@pytest.fixture
def annotator():
    entities = EntityPool([Entity('Anna', LabelSet.of(Label.PERSON)),
                           Entity('Ben', LabelSet.of(Label.PERSON)),
                           Entity('book', LabelSet.of(Label.PHYSICAL_OBJECT)),
                           Entity('telescope', LabelSet.of(Label.PHYSICAL_OBJECT))])
    return TemplateAnnotator(entities=entities, frames=DEFAULT_FRAMES, paradigms=DEFAULT_PARADIGMS)


def template(*, passive=False, agent=True, particle=False, lemma='buy', verb='bought', adjective=True):
    slots = {
        'n0': {'kind': 'nominal', 'pos': 'PROPN', 'tag': 'NNP', 'dependency': 'nsubj', 'head_slot': 'v0'},
        'v0': {'kind': 'verb', 'pos': 'VERB', 'tag': 'VBD', 'lemma': lemma, 'dependency': 'ROOT', 'head_slot': 'v0'},
        'n1': {'kind': 'nominal', 'pos': 'NOUN', 'tag': 'NN', 'dependency': 'dobj', 'head_slot': 'v0'},
    }
    bindings = {'n0': 'Anna', 'v0': verb, 'n1': 'book'}
    group = {'head_slot': 'v0', 'lemma': lemma, 'constructions': []}
    noun = [{'literal': 'a ' if not passive else 'The '}]
    attributes = []
    if adjective:
        slots['a0'] = {'kind': 'adjective', 'pos': 'ADJ', 'tag': 'JJ', 'dependency': 'amod', 'head_slot': 'n1'}
        bindings['a0'] = 'red'
        noun.extend([{'slot': 'a0'}, {'literal': ' '}])
        attributes.append({'slot': 'a0', 'target_slot': 'n1', 'degree': ['Pos']})
    noun.append({'slot': 'n1'})
    if passive:
        slots['v0']['tag'] = 'VBN'
        slots['n1']['dependency'] = 'nsubjpass'
        slots['v1'] = {'kind': 'verb', 'pos': 'AUX', 'tag': 'VBD', 'lemma': 'be',
                       'dependency': 'auxpass', 'head_slot': 'v0'}
        bindings['v1'] = 'was'
        parts = [*noun, {'literal': ' '}, {'slot': 'v1'}, {'literal': ' '}, {'slot': 'v0'}]
        if agent:
            slots['n0'].update(dependency='pobj', head_slot=None,
                               preposition={'text': 'by', 'head_slot': 'v0'})
            parts.extend([{'literal': ' by '}, {'slot': 'n0'}])
        else:
            del slots['n0'], bindings['n0']
    else:
        parts = [{'slot': 'n0'}, {'literal': ' '}, {'slot': 'v0'}, {'literal': ' '}, *noun]
    if particle:
        part = len(parts)
        parts.append({'literal': ' up'})
        group.update(lemma='pick up', constructions=[{'lemma': 'pick up', 'kind': 'particle',
                     'separated': True, 'markers': [{'part': part, 'start': 1, 'end': 3}]}])
    parts.append({'literal': '.'})
    record = dict(parts=parts, slots=slots, predicate_groups=[group], attributes=attributes,
                  verbs=[lemma], constructions=['pick up'] if particle else [])
    record['template_id'] = fingerprint(record)
    record.update(defaults=bindings, attested_bindings=[bindings], fillers={k: [v] for k, v in bindings.items()},
                  count=1, adjective_lemmas=['red'] if adjective else [])
    return record


def annotate(annotator, record, bindings=None):
    row = CorpusTemplate(record).render(bindings)
    doc = Document('mined-test', row['text'], metadata={k: v for k, v in row.items() if k != 'text'})
    result = annotator.annotate(doc)
    qa = QuestionGenerator().for_document(result.document, result.run) if result.run else []
    assert all(not q.check() for q in qa)
    return result, {q.question: q.answer for q in qa}, qa


def test_active_roles_properties_evidence_and_non_gold_provenance(annotator):
    result, qa, examples = annotate(annotator, template())
    assert qa['Who bought a red book?'] == 'Anna.'
    assert qa['What did Anna buy?'] == 'A red book.'
    assert qa['Which property is attributed to the book?'] == 'Red.'
    assert {e.relation for e in result.run.relations} == {Relation.AGENT_OF, Relation.THEME_OF, Relation.PROPERTY_OF}
    assert not result.run.validate_against(result.document)
    assert all(item.review_status == ReviewStatus.UNREVIEWED
               for item in (*result.run.mentions, *result.run.predicates, *result.run.properties))
    assert not result.run.synthetic
    assert all(q.metadata['review_status'] == 'UNREVIEWED' and q.metadata['semantic_method'] == VERSION for q in examples)
    assert not needs_annotation(result.document, [result.run])


@pytest.mark.parametrize('agent', [True, False])
def test_passive_maps_subject_to_theme_and_does_not_invent_an_agent(annotator, agent):
    result, qa, _ = annotate(annotator, template(passive=True, agent=agent))
    assert result.run.predicates[0].voice == 'passive'
    themes = [m for m in result.run.mentions if any(e.source_id == m.mention_id and e.relation == Relation.THEME_OF
                                                  for e in result.run.relations)]
    assert themes[0].exact_text == 'The red book'
    if agent:
        assert qa['Who bought the red book?'] == 'Anna.'
        assert qa['What did Anna buy?'] == 'The red book.'
    else:
        assert qa['What was bought?'] == 'The red book.'
        assert not any(e.relation == Relation.AGENT_OF for e in result.run.relations)


def test_separated_particles_and_replacements_retain_exact_offsets(annotator):
    result, qa, _ = annotate(annotator, template(particle=True, lemma='pick', verb='picked'),
                              {'n0': 'Ben', 'n1': 'telescope', 'a0': 'beautiful'})
    assert qa['What did Ben pick up?'] == 'A beautiful telescope.'
    predicate, = result.run.predicates
    assert predicate.lemma == 'pick up'
    assert result.document.slice(predicate.parts[0].span) == 'up'
    assert result.document.slice(predicate.span) == 'picked'
    assert not result.run.validate_against(result.document)


@pytest.mark.parametrize('compound', [False, True])
def test_governed_prepositions_remain_in_predicate_and_question(annotator, compound):
    head, verb = ('put', 'put') if compound else ('look', 'looked')
    record = template(lemma=head, verb=verb, adjective=False)
    record['slots']['n1'].update(dependency='pobj', head_slot=None)
    record['parts'][3] = {'literal': ' up with ' if compound else ' after '}
    markers = ([{'part': 3, 'start': 1, 'end': 3}, {'part': 3, 'start': 4, 'end': 8}]
               if compound else [{'part': 3, 'start': 1, 'end': 6}])
    lemma = 'put up with' if compound else 'look after'
    record['predicate_groups'][0]['constructions'] = [{
        'lemma': lemma, 'kind': 'particle_preposition' if compound else 'preposition_candidate',
        'separated': False, 'markers': markers}]
    result, qa, _ = annotate(annotator, record)
    assert qa[f'What did Anna {lemma}?'] == 'A book.'
    predicate, = result.run.predicates
    assert predicate.lemma == lemma
    assert [part.text for part in predicate.parts] == (['up', 'with'] if compound else ['after'])
    assert not result.run.validate_against(result.document)


def test_verb_outside_handwritten_frames_and_unknown_name_are_supported(annotator):
    result, qa, _ = annotate(annotator, template(lemma='polish', verb='polished', adjective=False), {'n0': 'Zorblax'})
    assert qa['Who or what polished a book?'] == 'Zorblax.'
    assert qa['What did Zorblax polish?'] == 'A book.'
    assert result.run.predicates[0].predicate_type is None
    assert not next(m for m in result.run.mentions if m.exact_text == 'Zorblax').labels


@pytest.mark.parametrize('mutation,reason', [
    ('no_dependencies', 'missing_slot_dependencies'),
    ('embedded', 'embedded_or_nonfinite_clause'),
    ('imperative', 'missing_finite_tense'),
    ('conditional', 'question_or_conditional_scope'),
    ('uncertain', 'uncertainty_or_exception_scope'),
    ('irregular', 'unresolved_verb_inflection'),
])
def test_unsupported_clauses_are_preserved_and_reported(annotator, mutation, reason):
    record = template(adjective=False)
    if mutation == 'no_dependencies':
        record['slots']['v0'].pop('dependency')
    elif mutation == 'embedded':
        record['slots']['v0']['dependency'] = 'xcomp'
    elif mutation == 'imperative':
        record['slots']['v0']['tag'] = 'VB'
    elif mutation == 'conditional':
        record['parts'].insert(0, {'literal': 'If '})
    elif mutation == 'uncertain':
        record['parts'].insert(0, {'literal': 'Perhaps '})
    else:
        record['slots']['v0']['lemma'] = 'unknown'
    result, qa, _ = annotate(annotator, record)
    assert result.run is None and not qa
    assert result.document.metadata['template_semantics']['skipped'][0]['reason'] == reason
    assert needs_annotation(result.document, [])


def test_partial_annotations_still_need_a_teacher_until_replaced(annotator):
    record = template()
    record['slots']['v0']['dependency'] = 'xcomp'
    result, qa, _ = annotate(annotator, record)
    assert qa and result.run.properties and not result.run.predicates
    assert needs_annotation(result.document, [result.run])
    assert not needs_annotation(result.document, [result.run, AnnotationRun('teacher', model_name='http')])


@pytest.mark.parametrize('embedded', [False, True])
@pytest.mark.parametrize('nested', [False, True])
def test_coordinated_predicate_inherits_scope_of_its_ancestors(annotator, embedded, nested):
    record = template(adjective=False)
    record['slots']['v0'].update(dependency='conj', head_slot='v1')
    record['slots']['v1'] = dict(kind='verb', pos='VERB', tag='VBD', lemma='smile',
                                dependency='ccomp' if embedded else 'ROOT', head_slot='v2' if embedded else 'v1')
    record['defaults']['v1'] = 'smiled'
    prefix = [{'literal': 'Ben '}, {'slot': 'v1'}, {'literal': ' and '}]
    if embedded:
        record['slots']['v2'] = dict(kind='verb', pos='VERB', tag='VBD', lemma='say',
                                    dependency='ROOT', head_slot='v2')
        record['defaults']['v2'] = 'said'
        prefix = [{'literal': 'They '}, {'slot': 'v2'}, {'literal': ' '}, *prefix]
    if nested:
        record['slots']['v3'] = dict(kind='verb', pos='VERB', tag='VBD', lemma='laugh',
                                    dependency='conj', head_slot='v1')
        record['defaults']['v3'] = 'laughed'
        record['slots']['v0']['head_slot'] = 'v3'
        prefix += [{'literal': 'he '}, {'slot': 'v3'}, {'literal': ' and '}]
    record['parts'] = prefix + record['parts']
    result, qa, _ = annotate(annotator, record)
    if embedded:
        assert result.run is None and not qa
        assert result.document.metadata['template_semantics']['skipped'][0]['reason'] == 'embedded_or_nonfinite_clause'
    else:
        assert qa['What did Anna buy?'] == 'A book.'


@pytest.mark.parametrize('condition', ['As long as Ben agrees, ', 'Provided that Ben agrees, '])
def test_conditional_markers_do_not_become_unconditional_claims(annotator, condition):
    record = template(adjective=False)
    record['parts'].insert(0, {'literal': condition})
    result, qa, _ = annotate(annotator, record)
    assert result.run is None and not qa
    assert result.document.metadata['template_semantics']['skipped'][0]['reason'] == 'question_or_conditional_scope'


def test_cyclic_coordination_is_rejected_without_looping(annotator):
    record = template(adjective=False)
    record['slots']['v0'].update(dependency='conj', head_slot='v0')
    result, qa, _ = annotate(annotator, record)
    assert result.run is None and not qa


@pytest.mark.parametrize('perfect', [True, False])
def test_auxiliaries_negation_and_plural_subject_agreement(annotator, perfect):
    record = template(verb='bought' if perfect else 'buying', adjective=False)
    record['slots']['v0']['tag'] = 'VBN' if perfect else 'VBG'
    record['slots']['n0'].update(pos='PRON', tag='PRP')
    record['slots']['v1'] = {'kind': 'verb', 'pos': 'AUX', 'tag': 'VBZ' if perfect else 'VBP',
                             'lemma': 'have' if perfect else 'be', 'dependency': 'aux', 'head_slot': 'v0'}
    record['defaults'].update(n0='She' if perfect else 'They', v1='has' if perfect else 'are')
    record['parts'][2:2] = [{'slot': 'v1'}, {'literal': ' not ' if perfect else ' '}]
    result, qa, _ = annotate(annotator, record)
    question = "What hasn't she bought?" if perfect else 'What are they buying?'
    assert qa[question] == 'A book.'
    assert result.run.predicates[0].aspect == ('perfect' if perfect else 'progressive')


@pytest.mark.parametrize('prep,lemma,verb', [(False, 'give', 'gave'), (True, 'give', 'gave'),
                                           (False, 'ask', 'asked'), (False, 'buy', 'bought')])
def test_direct_and_prepositional_recipient_word_order(annotator, prep, lemma, verb):
    record = template(lemma=lemma, verb=verb, adjective=False)
    record['defaults']['n2'] = 'Ben'
    record['slots']['n2'] = {'kind': 'nominal', 'pos': 'PROPN', 'tag': 'NNP',
                             'dependency': 'pobj' if prep else 'dative', 'head_slot': None if prep else 'v0'}
    if prep:
        part = len(record['parts']) - 1
        record['parts'][-1:-1] = [{'literal': ' to '}, {'slot': 'n2'}]
        record['predicate_groups'][0]['constructions'] = [{'lemma': 'give to', 'kind': 'preposition_candidate',
            'separated': False, 'markers': [{'part': part, 'start': 1, 'end': 3}]}]
    else:
        record['parts'][3:3] = [{'literal': ' '}, {'slot': 'n2'}]
    result, qa, _ = annotate(annotator, record)
    if lemma == 'give':
        assert qa['Who did Anna give a book to?'] == 'Ben.'
        assert qa['What did Anna give to Ben?'] == 'A book.'
    elif lemma == 'buy':
        assert qa['Who did Anna buy a book for?'] == 'Ben.'
        assert qa['What did Anna buy for Ben?'] == 'A book.'
    else:
        assert qa['What did Anna ask Ben?'] == 'A book.'
        assert not any(' to ' in question or question.endswith(' to?') for question in qa)
    assert any(edge.relation == Relation.RECIPIENT_OF for edge in result.run.relations)


def test_negative_quantifiers_do_not_become_positive_event_claims(annotator):
    record = template(adjective=False)
    record['parts'].insert(0, {'literal': 'No '})
    result, qa, _ = annotate(annotator, record, {'n0': 'one'})
    assert not qa and result.run is None
    assert result.document.metadata['template_semantics']['skipped'][0]['reason'] == 'negative_quantifier_scope'


def test_literal_number_modifiers_are_included_in_answer_spans(annotator):
    record = template(adjective=False)
    record['parts'][4] = {'literal': 'twenty '}
    record['slots']['n1']['tag'] = 'NNS'
    result, qa, _ = annotate(annotator, record, {'n1': 'books'})
    assert 'Twenty books.' in qa.values()
    assert any(m.exact_text == 'twenty books' for m in result.run.mentions)


def test_modified_adjective_stays_in_noun_phrase_without_false_property(annotator):
    record = template()
    record['parts'][4] = {'literal': 'a very '}
    result, qa, _ = annotate(annotator, record)
    assert qa['What did Anna buy?'] == 'A very red book.'
    assert not result.run.properties


def test_dropping_adjunct_does_not_change_negative_scope(annotator):
    record = template(adjective=False)
    record['slots']['v0']['tag'] = 'VB'
    record['slots']['v1'] = {'kind': 'verb', 'pos': 'AUX', 'tag': 'VBD',
                             'lemma': 'do', 'dependency': 'aux', 'head_slot': 'v0'}
    record['defaults'].update(v0='buy', v1='did')
    record['parts'][2:2] = [{'slot': 'v1'}, {'literal': ' not '}]
    result, qa, _ = annotate(annotator, record)
    assert qa["What didn't Anna buy?"] == 'A book.'
    record['parts'][-1] = {'literal': ' yesterday.'}
    result, qa, _ = annotate(annotator, record)
    assert not qa and result.run is None
    assert result.document.metadata['template_semantics']['skipped'][0]['reason'] == 'negative_complement_scope'


@pytest.mark.parametrize('past', [False, True])
def test_first_person_progressive_agreement(annotator, past):
    record = template(verb='buying', adjective=False)
    record['slots']['v0']['tag'] = 'VBG'
    record['slots']['n0'].update(pos='PRON', tag='PRP')
    record['slots']['v1'] = {'kind': 'verb', 'pos': 'AUX', 'tag': 'VBD' if past else 'VBP',
                             'lemma': 'be', 'dependency': 'aux', 'head_slot': 'v0'}
    record['defaults'].update(n0='I', v1='was' if past else 'am')
    record['parts'][2:2] = [{'slot': 'v1'}, {'literal': ' '}]
    _, qa, _ = annotate(annotator, record)
    assert qa['What was I buying?' if past else 'What am I buying?'] == 'A book.'


def test_reflexive_does_not_produce_ungrammatical_subject_question(annotator):
    record = template(lemma='injure', verb='injured', adjective=False)
    record['slots']['n0'].update(pos='PRON', tag='PRP')
    record['slots']['n1'].update(pos='PRON', tag='PRP')
    record['parts'][4] = {'literal': ''}
    _, qa, _ = annotate(annotator, record, {'n0': 'I', 'n1': 'myself'})
    assert qa == {'Who or what did I injure?': 'Myself.'}


@pytest.mark.parametrize('adverb', ['never', 'seldom', 'hardly'])
def test_preverbal_scope_adverbs_do_not_become_positive_claims(annotator, adverb):
    record = template(adjective=False)
    record['parts'][1] = {'literal': f' {adverb} '}
    result, qa, _ = annotate(annotator, record)
    assert not qa and result.run is None
    assert needs_annotation(result.document, [])


def test_determiner_extension_does_not_absorb_the_recipient_pronoun(annotator):
    record = template(lemma='give', verb='gave', adjective=False)
    record['defaults']['n2'] = 'her'
    record['slots']['n2'] = {'kind': 'nominal', 'pos': 'PRON', 'tag': 'PRP',
                             'dependency': 'dative', 'head_slot': 'v0'}
    record['parts'][3:3] = [{'literal': ' '}, {'slot': 'n2'}]
    result, qa, _ = annotate(annotator, record)
    assert qa['What did Anna give to her?'] == 'A book.'
    assert qa['Who gave a book to her?'] == 'Anna.'
    assert all(m.exact_text != 'her a book' for m in result.run.mentions)


def test_passive_pronouns_have_object_case_in_active_questions(annotator):
    record = template(passive=True, adjective=False, lemma='injure', verb='injured')
    record['parts'][0] = {'literal': ''}
    for key in ('n0', 'n1'):
        record['slots'][key].update(pos='PRON', tag='PRP')
    _, qa, _ = annotate(annotator, record, {'n0': 'him', 'n1': 'She'})
    assert qa['Who injured her?'] == 'Him.'
    assert qa['Who did he injure?'] == 'She.'


def test_complex_prepositional_argument_in_another_sentence_does_not_hide_qa(annotator):
    record = template(adjective=False)
    record['parts'] += [{'literal': ' Ben walks with a '}, {'slot': 'n2'},
                        {'literal': ' who '}, {'slot': 'v1'}, {'literal': '.'}]
    record['slots']['n2'] = {'kind': 'nominal', 'pos': 'NOUN', 'tag': 'NN',
                             'dependency': 'pobj', 'head_slot': None}
    record['slots']['v1'] = {'kind': 'verb', 'pos': 'VERB', 'tag': 'VBZ',
                             'lemma': 'work', 'dependency': 'relcl', 'head_slot': 'n2'}
    record['defaults'].update(n2='man', v1='works')
    record['predicate_groups'].append({'head_slot': 'v1', 'lemma': 'work', 'constructions': []})
    result, qa, _ = annotate(annotator, record)
    assert qa['What did Anna buy?'] == 'A book.'
    assert needs_annotation(result.document, [result.run])


def write_bank(root, record):
    bank = root / 'data/mined-templates'
    (bank / 'templates').mkdir(parents=True)
    with gzip.open(bank / 'templates' / (record['template_id'][:2] + '.jsonl.gz'), 'wt', encoding='utf-8') as stream:
        stream.write(json.dumps(record) + '\n')
    (bank / 'manifest.json').write_text(json.dumps({'format': FORMAT, 'status': 'complete',
        'source_sha256': 'a' * 64, 'counts': {'templates': 1, 'compiled_records': 1}}), encoding='utf-8')


def test_cli_build_exports_mined_qa_and_is_idempotent(annotator, tmp_path, monkeypatch):
    from semantic_corpus import resource_pool
    from semantic_corpus.cli import main
    from semantic_corpus.semantic_generator import Generator
    from semantic_corpus.storage import CorpusStore
    record = template()
    write_bank(tmp_path, record)
    pool = BuiltInPool(tmp_path, entities=annotator.entities)
    monkeypatch.setattr(resource_pool, 'BuiltInPool', lambda: pool)
    monkeypatch.setattr(Generator, 'generate_per_frame', lambda self, count: iter(()))
    split = next(s for s in ('train', 'dev', 'test') if list(pool.mined_documents(split=s)))
    args = ['build', '--store', str(tmp_path / 'store'), '--out', str(tmp_path / 'out'),
            '--count', '1', '--split', split]
    assert main(args) == 0
    store = CorpusStore(tmp_path / 'store', create=False)
    assert store.count('annotation_runs') == 1
    assert store.count('qa_examples') == 3
    summary = json.loads((tmp_path / 'out/generation-summary.json').read_text())
    assert summary['mined']['qa_examples'] == 3 and summary['pending_annotation'] == 0
    assert (tmp_path / 'out/sft.jsonl').read_text().strip()
    assert not (tmp_path / 'out/bio.jsonl').read_text().strip()
    before = {p.name: p.read_bytes() for p in (tmp_path / 'store').iterdir() if p.is_file()}
    assert main(args) == 0
    assert before == {p.name: p.read_bytes() for p in (tmp_path / 'store').iterdir() if p.is_file()}
    # Adding/changing the optional inflection dictionary changes the semantic
    # build identity even if the selected template uses a curated paradigm.
    from semantic_corpus.storage import StoreError
    dictionary = tmp_path / 'data/wiktionary/en_verb_inflections.txt'
    dictionary.parent.mkdir()
    dictionary.write_text('polish\tpolishes\tpolishing\tpolished\tpolished\n', encoding='utf-8')
    monkeypatch.setattr(resource_pool, 'BuiltInPool', lambda: BuiltInPool(tmp_path, entities=annotator.entities))
    with pytest.raises(StoreError, match='version'):
        main(args)
    assert before == {p.name: p.read_bytes() for p in (tmp_path / 'store').iterdir() if p.is_file()}


def test_mined_qa_candidates_do_not_become_false_bio_outside_labels(annotator, tmp_path):
    from semantic_corpus.exporters.bio import export_bio
    result, _, qa = annotate(annotator, template(lemma='polish', verb='polished', adjective=False),
                              {'n0': 'Zorblax'})
    assert qa
    report = export_bio([(result.document, result.run)], tmp_path / 'bio.jsonl')
    assert report.documents == 0 and report.skipped_mined_candidates == 1
    assert not report.is_lossless
    assert (tmp_path / 'bio.jsonl').read_text() == ''
    teacher = replace(result.run, run_id='teacher-reviewed', model_name='http')
    report = export_bio([(result.document, teacher)], tmp_path / 'bio.jsonl')
    assert report.documents == 1 and report.skipped_mined_candidates == 0
