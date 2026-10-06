"""Multiword predicate morphology, order and discontinuous annotations."""

from dataclasses import replace

import pytest

from semantic_corpus.documents import Passage, TextSpan
from semantic_corpus.exporters.bio import BioReport, tag_document
from semantic_corpus.ontology import Label, LabelSet
from semantic_corpus.question_generator import QuestionGenerator
from semantic_corpus.semantic_generator import (
    DEFAULT_PARADIGMS, Entity, Features, Generator, Situation, frame_by_lemma, realize,
)
from semantic_corpus.semantic_generator.canonical import combine, to_canonical
from semantic_corpus.storage import CorpusStore


def situation(lemma="pick up", *, theme="the book", features=Features(), plural=False):
    frame = frame_by_lemma(lemma)
    bindings = {"agent": Entity("the teachers" if plural else "Anna", LabelSet.of(Label.PERSON),
                                number="plural" if plural else "singular")}
    if "theme" in frame.slot_map:
        label = Label.PERSON if lemma in ("look after", "put up with") else Label.PHYSICAL_OBJECT
        bindings["theme"] = Entity(theme, LabelSet.of(label))
    return Situation(frame, bindings, DEFAULT_PARADIGMS[frame.head_lemma], features)


@pytest.mark.parametrize("lemma, pattern, text, markers", [
    ("pick up", "joined", "Anna picked up the book.", ["up"]),
    ("pick up", "separated", "Anna picked the book up.", ["up"]),
    ("pick up", "passive", "The book was picked up by Anna.", ["up"]),
    ("pick up", "passive_agentless", "The book was picked up.", ["up"]),
    ("put down", "separated", "Anna put the book down.", ["down"]),
    ("give up", "intransitive", "Anna gave up.", ["up"]),
    ("get up", "intransitive", "Anna got up.", ["up"]),
    ("look after", "prepositional", "Anna looked after the child.", ["after"]),
    ("look after", "passive", "The child was looked after by Anna.", ["after"]),
    ("put up with", "prepositional", "Anna put up with the child.", ["up", "with"]),
])
def test_realization_and_questions(lemma, pattern, text, markers):
    value = situation(lemma, theme="the child" if lemma in ("look after", "put up with") else "the book")
    realized = realize(value, pattern)
    assert realized.text == text
    assert not realized.check()
    doc, run = to_canonical(realized, document_id="phrase")
    assert not run.validate_against(doc)
    predicate, = run.predicates
    assert predicate.lemma == lemma
    assert [part.span.text_in(text) for part in predicate.parts] == markers
    assert predicate.span.text_in(text) in {"picked", "put", "gave", "got", "looked"}
    examples = QuestionGenerator(paradigms=DEFAULT_PARADIGMS).for_document(doc, run)
    qa = {e.question: e.answer for e in examples}
    assert all(not e.check() for e in examples)
    if pattern != "passive_agentless":
        assert "What did Anna do?" in qa
        assert " up" not in next(q for q in qa if q == "What did Anna do?")
        if "theme" in value.bindings:
            wh = "Who" if lemma in ("look after", "put up with") else "What"
            assert f"{wh} did Anna {lemma}?" in qa
    else:
        assert any(e.question == "Who picked up the book?" and not e.answerable for e in examples)
    if lemma == "put up with":
        assert not any("was put" in q or "were put" in q for q in qa)


@pytest.mark.parametrize("features, plural, expected", [
    (Features(tense="present"), False, "Anna picks up the book."),
    (Features(tense="present"), True, "The teachers pick up the book."),
    (Features(is_negated=True), False, "Anna didn't pick up the book."),
    (Features(tense="present", is_perfect=True), False, "Anna has picked up the book."),
    (Features(tense="present", is_progressive=True), True, "The teachers are picking up the book."),
])
def test_only_head_inflects(features, plural, expected):
    result = realize(situation(features=features, plural=plural), "joined")
    assert result.text == expected
    doc, run = to_canonical(result, document_id="aspect")
    examples = QuestionGenerator(paradigms=DEFAULT_PARADIGMS).for_document(doc, run)
    assert all(not e.check() for e in examples)
    if features.is_negated:
        assert any(e.question == "Did Anna pick up the book?" and e.answer.startswith("No.") for e in examples)


def test_pronoun_separation_in_text_and_questions():
    result = realize(situation(theme="it"), "joined")
    assert result.text == "Anna picked it up."
    doc, run = to_canonical(result, document_id="pronoun")
    qa = {e.question: e.answer for e in QuestionGenerator(paradigms=DEFAULT_PARADIGMS).for_document(doc, run)}
    assert qa["Who picked it up?"] == "Anna."
    assert qa["Did Anna pick it up?"] == "Yes. Anna picked it up."
    assert qa["What did Anna pick up?"] == "It."


def test_roundtrip_combine_passages_and_bio(tmp_path):
    doc, run = to_canonical(realize(situation(), "separated"), document_id="original")
    store = CorpusStore(tmp_path / "store")
    store.add_document(doc)
    store.add_run(run)
    restored, = store.runs(document_id=doc.document_id)
    assert restored.predicates == run.predicates
    combined, joined = combine([(doc, run), (doc, run)], document_id="joined")
    assert not joined.validate_against(combined)
    offset = len(doc.text) + 1
    assert joined.predicates[1].parts[0].span == run.predicates[0].parts[0].span.shifted(offset)
    passage = Passage("second", "joined", TextSpan(offset, len(combined.text)))
    local = replace(combined, passages=(passage,))
    examples = QuestionGenerator(paradigms=DEFAULT_PARADIGMS).for_passages(local, joined)
    assert examples and all(not e.check() for e in examples)
    report = BioReport()
    tagging = tag_document(doc, run, report=report)
    tags = dict(zip(tagging.tokens, tagging.tags))
    assert tags["picked"] == "B-ACTION" and tags["up"] == "B-ACTION"
    assert tags["book"].endswith("PHYSICAL_OBJECT")
    assert report.lost_predicate_links == 1 and not report.is_lossless


def test_all_new_frames_generate_in_each_split():
    from semantic_corpus.semantic_generator.frames import PHRASAL_FRAMES
    for split in ("train", "dev", "test"):
        generated = list(Generator(seed=7, frames=PHRASAL_FRAMES, split=split).generate_per_frame(3))
        assert len(generated) == len(PHRASAL_FRAMES) * 3
        assert all(not item.check() for item in generated)


def test_invalid_parts_are_rejected():
    doc, run = to_canonical(realize(situation(), "separated"), document_id="invalid")
    pred, = run.predicates
    bad = replace(pred.parts[0], span=TextSpan(0, 2))
    with pytest.raises(ValueError):
        replace(pred, parts=(bad,)).validate_against(doc)
