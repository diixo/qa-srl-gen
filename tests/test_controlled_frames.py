"""The new lexical frames must also yield grammatical standalone QA."""
from dataclasses import replace

import pytest

from semantic_corpus.ontology import Label, LabelSet
from semantic_corpus.semantic_generator import Generator
from semantic_corpus.semantic_generator.frames import BE, HAVE, GO
from semantic_corpus.semantic_generator.realization import BindingError, Features, realize
from semantic_corpus.semantic_generator.substitutions import Entity
from semantic_corpus.semantic_generator.canonical import to_canonical
from semantic_corpus.question_generator import QuestionGenerator


@pytest.mark.parametrize('frame,bindings,present,past,question', [
    (BE, {'entity': Entity('Anna', LabelSet.of(Label.PERSON)), 'location': Entity('the park', LabelSet.of(Label.LOCATION))},
     'Anna is in the park.', 'Anna was in the park.', 'Is Anna in the park?'),
    (HAVE, {'possessor': Entity('Anna', LabelSet.of(Label.PERSON)), 'possession': Entity('a book', LabelSet.of(Label.PHYSICAL_OBJECT))},
     'Anna has a book.', 'Anna had a book.', 'Does Anna have a book?'),
    (GO, {'agent': Entity('Anna', LabelSet.of(Label.PERSON)), 'destination': Entity('the park', LabelSet.of(Label.LOCATION))},
     'Anna goes to the park.', 'Anna went to the park.', 'Does Anna go to the park?'),
])
def test_lexical_irregulars_and_questions(frame, bindings, present, past, question):
    generator = Generator(seed=0, frames=(frame,))
    situation = replace(generator.sample_situation(frame), bindings=bindings, features=Features(tense='present'))
    result = realize(situation)
    assert result.text == present and not result.check()
    assert realize(replace(situation, features=Features(tense='past'))).text == past
    doc, run = to_canonical(result, document_id='controlled')
    questions = {qa.question for qa in QuestionGenerator(seed=0).for_document(doc, run)}
    assert question in questions
    assert not any(word in q for q in questions for word in ('haves', 'haved', 'gos', 'goed', 'Is Anna?', 'Who is?'))
    if frame in (BE, HAVE):
        with pytest.raises(BindingError, match='progressive'):
            realize(replace(situation, features=Features(is_progressive=True)))
    if frame == BE:
        assert realize(replace(situation, features=Features(tense='present', is_negated=True))).text == "Anna isn't in the park."
        plural = {**bindings, 'entity': Entity('the children', LabelSet.of(Label.PERSON), number='plural')}
        assert realize(replace(situation, bindings=plural)).text == 'The children are in the park.'
