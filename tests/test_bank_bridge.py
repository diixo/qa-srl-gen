"""QA-SRL Bank as canonical annotated text, plus question templates."""

from __future__ import annotations

import itertools

import pytest

from semantic_corpus.ontology import Label, Relation
from semantic_corpus.qasrl_bridge import (
    BANK_SOURCE,
    bank_qa_examples,
    iter_bank_canonical,
    sentence_to_canonical,
    token_spans,
)
from semantic_corpus.qasrl_core.bank_reader import read_bank
from semantic_corpus.qasrl_core.question_template import (
    QuestionTemplate,
    normalize_adverbials,
    normalize_to_active,
)
from semantic_corpus.question_generator import QAKind
from semantic_corpus.storage import CorpusStore


@pytest.fixture(scope="module")
def sentences(fixtures_dir):
    return list(read_bank(fixtures_dir / "bank_sample.jsonl"))


# -- token offsets ---------------------------------------------------------


def test_token_spans_partition_the_joined_text():
    tokens = ["You", "can", "see", "it", "."]
    text = " ".join(tokens)
    spans = token_spans(tokens)
    assert len(spans) == len(tokens)
    for token, span in zip(tokens, spans):
        assert span.text_in(text) == token


# -- the bridge ------------------------------------------------------------


def test_every_bank_sentence_converts_cleanly(sentences):
    for sentence in sentences:
        document, run = sentence_to_canonical(sentence)
        assert run.validate_against(document) == []
        assert document.source == BANK_SOURCE
        assert not run.synthetic, "the bank is real text, not generated"


def test_predicates_keep_their_lemma_and_position(sentences):
    document, run = sentence_to_canonical(sentences[0])
    assert run.predicates
    for predicate in run.predicates:
        assert predicate.lemma
        assert predicate.span.text_in(document.text) == predicate.exact_text
        assert predicate.predicate_type is None
        assert predicate.polarity is None


def test_mentions_carry_no_invented_entity_type(sentences):
    """The bank never says what kind of thing an argument is."""
    for sentence in sentences:
        _document, run = sentence_to_canonical(sentence)
        for mention in run.mentions:
            assert mention.labels.entity_labels == ()


def test_the_question_is_kept_on_the_relation(sentences):
    """The handoff makes the question primary and the role derived."""
    found = False
    for sentence in sentences:
        _document, run = sentence_to_canonical(sentence)
        for edge in run.relations:
            assert edge.question, "a bank relation without its question"
            assert edge.question.endswith("?")
            found = True
    assert found


def test_passive_syntax_does_not_invent_a_semantic_role():
    """The question survives even when its role needs semantic annotation."""
    import gzip
    import json

    from semantic_corpus.qasrl_core.models import Sentence

    sentence = Sentence.from_json(
        {
            "sentenceId": "t:1",
            "sentenceTokens": ["You", "can", "see", "an", "example", "."],
            "verbEntries": {
                "2": {
                    "verbIndex": 2,
                    "verbInflectedForms": {
                        "stem": "see",
                        "presentSingular3rd": "sees",
                        "presentParticiple": "seeing",
                        "past": "saw",
                        "pastParticiple": "seen",
                    },
                    "questionLabels": {
                        "What can be seen?": {
                            "questionString": "What can be seen?",
                            "questionSlots": {
                                "wh": "what", "aux": "can", "subj": "_",
                                "verb": "be pastParticiple", "obj": "_",
                                "prep": "_", "obj2": "_",
                            },
                            "tense": "can",
                            "isPerfect": False,
                            "isProgressive": False,
                            "isNegated": False,
                            "isPassive": True,
                            "answerJudgments": [
                                {"sourceId": "a", "isValid": True, "spans": [[3, 5]]},
                                {"sourceId": "b", "isValid": True, "spans": [[3, 5]]},
                            ],
                        }
                    },
                }
            },
        }
    )
    _document, run = sentence_to_canonical(sentence)
    assert [e.relation for e in run.relations] == [None]
    assert run.relations[0].question == "What can be seen?"


def test_a_span_needs_agreement_to_be_kept(sentences):
    """One annotator's highlight is a claim, not a fact."""
    sentence = sentences[0]
    strict = sentence_to_canonical(sentence, min_votes=3)[1]
    loose = sentence_to_canonical(sentence, min_votes=1)[1]
    assert len(strict.mentions) <= len(loose.mentions)


# -- QA examples -----------------------------------------------------------


def test_the_bank_yields_qa_examples_written_by_people(sentences):
    total = 0
    for sentence in sentences:
        document, _run = sentence_to_canonical(sentence)
        for example in bank_qa_examples(sentence, document):
            assert example.check() == []
            assert example.kind is QAKind.ATOMIC
            assert example.context == document.text
            assert example.metadata["votes"] >= 2
            # The answer must be a span of the sentence, not a paraphrase.
            assert example.answer.rstrip(".").lower() in document.text.lower()
            total += 1
    assert total > 50


@pytest.mark.corpus
def test_alternative_answers_are_recorded(bank_dev):
    """Annotators often highlight more than one acceptable span.

    The 20-sentence fixture happens to contain no such question, so this
    one needs the real bank.
    """
    found = 0
    for sentence in itertools.islice(read_bank(bank_dev), 200):
        for example in bank_qa_examples(sentence):
            if example.metadata.get("alternative_answers"):
                found += 1
    assert found


def test_bank_documents_go_into_the_store(sentences, tmp_path):
    store = CorpusStore(tmp_path / "s")
    for index, sentence in enumerate(sentences[:5]):
        document, run = sentence_to_canonical(sentence, split="train")
        store.add_document(document)
        store.add_run(run, document)
        store.add_examples(bank_qa_examples(sentence, document))
    assert store.count("documents") == 5
    assert store.count("qa_examples") > 0
    for document in store.documents():
        for run in store.runs(document_id=document.document_id):
            assert run.validate_against(document) == []


@pytest.mark.corpus
def test_streaming_a_real_bank_file(bank_dev):
    pairs = list(itertools.islice(iter_bank_canonical(bank_dev), 100))
    assert len(pairs) == 100
    for document, run in pairs:
        assert run.validate_against(document) == []


# -- question templates ----------------------------------------------------


def template(question_slots) -> QuestionTemplate:
    return QuestionTemplate.from_slots(question_slots)


def test_a_template_strips_tense_and_lexis(sentences):
    document, run = sentence_to_canonical(sentences[0])
    shapes = set()
    for sentence in sentences:
        for _entry, label in sentence.question_labels():
            shapes.add(template(label.question_slots).template_string)
    assert shapes
    assert all("verb" in shape for shape in shapes)


def test_animacy_is_collapsed():
    from semantic_corpus.qasrl_core.models import QuestionSlots

    who = QuestionSlots("who", "_", "_", "past", "someone", "_", "_")
    what = QuestionSlots("what", "_", "_", "past", "something", "_", "_")
    assert template(who) == template(what)


def test_the_passive_is_recognised():
    from semantic_corpus.qasrl_core.models import QuestionSlots

    passive = QuestionSlots("what", "is", "_", "pastParticiple", "_", "_", "_")
    active = QuestionSlots("what", "does", "something", "stem", "_", "_", "_")
    assert template(passive).is_passive
    assert not template(active).is_passive


def test_normalising_to_active_folds_a_voice_alternation():
    from semantic_corpus.qasrl_core.models import QuestionSlots

    passive = QuestionSlots("what", "is", "_", "pastParticiple", "_", "_", "_")
    folded = normalize_to_active(template(passive))
    assert not folded.is_passive
    assert folded.has_subj


def test_normalising_adverbials_collapses_the_clause():
    from semantic_corpus.qasrl_core.models import QuestionSlots

    first = QuestionSlots("where", "does", "something", "stem", "something", "_", "_")
    second = QuestionSlots("where", "did", "something", "stem", "_", "to", "someone")
    assert normalize_adverbials(template(first)) == normalize_adverbials(
        template(second)
    )


def test_a_template_renders_and_round_trips(sentences):
    for sentence in sentences:
        for _entry, label in sentence.question_labels():
            shape = template(label.question_slots)
            assert shape.question_string.endswith("?")
            assert QuestionTemplate.from_json(shape.to_json()) == shape


@pytest.mark.corpus
def test_templates_compress_the_bank(bank_dev):
    """Counting shapes says something counting questions cannot."""
    raw, active, adverbial = set(), set(), set()
    questions = 0
    for sentence in itertools.islice(read_bank(bank_dev), 800):
        for _entry, label in sentence.question_labels():
            questions += 1
            shape = template(label.question_slots)
            raw.add(shape.template_string)
            folded = normalize_to_active(shape)
            active.add(folded.template_string)
            adverbial.add(normalize_adverbials(folded).template_string)
    assert questions > 1000
    assert len(raw) < questions / 10
    assert len(active) < len(raw)
    assert len(adverbial) < len(active)
