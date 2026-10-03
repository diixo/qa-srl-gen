"""Regressions for the implementation audit, independent of Scala porting."""

from dataclasses import replace
import json

import pytest

from semantic_corpus.documents import (
    AnnotationRun, DialogueAnnotation, Document, EntityMention, Passage,
    Predicate, Property, RelationEdge, SpanError, TextSpan, Utterance,
)
from semantic_corpus.ontology import (
    Label, LabelSet, Mood, Polarity, Relation, ReviewStatus, SpeechAct,
)
from semantic_corpus.question_generator import QuestionGenerator, QAExample, QAKind
from semantic_corpus.question_generator.templates import (
    ParadigmResolver, atomic_questions, compound_questions, polarity_questions,
)
from semantic_corpus.semantic_generator import Generator, Features, realize
from semantic_corpus.semantic_generator.canonical import combine, to_canonical
from semantic_corpus.semantic_generator.frames import frame_by_lemma
from semantic_corpus.semantic_generator.generator import DEFAULT_PARADIGMS
from semantic_corpus.semantic_annotator import (
    RuleBasedTeacher, annotate_document, classify_turn, segment_dialogue, tokenize,
)
from semantic_corpus.semantic_annotator.alignment import align_response
from semantic_corpus.semantic_annotator.teacher import ProposedAnnotation, TeacherResponse
from semantic_corpus.semantic_annotator.verifier import promote_agreed, verify_independently
from semantic_corpus.storage import CorpusStore, StoreError, SplitConflict


def generated(lemma="give", pattern="ditransitive", **features):
    generator = Generator(seed=42)
    situation = generator.sample_situation(frame_by_lemma(lemma), features=Features(**features))
    bindings = dict(situation.bindings)
    forms = {"agent": "Anna", "recipient": "Rex", "theme": "a red ball"}
    for slot, form in forms.items():
        if slot in bindings:
            bindings[slot] = next(e for e in generator.pool.readings(form) if not e.appositive)
    situation = replace(situation, bindings=bindings)
    return to_canonical(realize(situation, pattern), document_id=lemma)


def question_generator():
    return QuestionGenerator(paradigms=DEFAULT_PARADIGMS, ambiguous_forms=("Rex",))


def dialogue_document():
    turns = ("Oh, I see.", "Oh, really?")
    text = "\n".join(turns)
    first = TextSpan(0, len(turns[0]))
    second = TextSpan(len(turns[0]) + 1, len(text))
    return segment_dialogue(Document("d", text, utterances=(
        Utterance("u0", "d", 0, "A", first), Utterance("u1", "d", 1, "B", second),
    )))


def test_span_cannot_extend_beyond_document_even_if_slice_matches():
    with pytest.raises(SpanError):
        TextSpan(0, 999).check("Anna", "Anna")


def test_validation_checks_document_identity_and_property_target():
    doc = Document("d", "red ball")
    run = AnnotationRun("r", properties=(
        Property("p", "other", TextSpan(0, 3), "red", "red", target_id="absent"),
    ))
    problems = run.validate_against(doc)
    assert any("document" in p for p in problems)
    assert any("absent" in p for p in problems)


def test_combining_remaps_document_and_property_references():
    one = generated()
    doc, run = combine((one, one), document_id="both")
    assert run.validate_against(doc) == []
    assert run.properties
    assert all(i.document_id == "both" for i in (*run.mentions, *run.predicates, *run.properties))
    assert all(p.target_id in run.ids() for p in run.properties)
    assert run.properties[0].target_id != run.properties[1].target_id


def test_an_omitted_agent_does_not_make_another_events_agent_unknown():
    first = generated(pattern="passive_agentless")
    text = "Anna visited Kyiv."
    doc = Document("visit", text)
    run = AnnotationRun("visit-r", synthetic=True,
        mentions=(EntityMention("a", "visit", TextSpan(0, 4), "Anna", LabelSet.of(Label.PERSON)),
                  EntityMention("k", "visit", TextSpan(13, 17), "Kyiv", LabelSet.of(Label.LOCATION))),
        predicates=(Predicate("v", "visit", TextSpan(5, 12), "visited", "visit", tense="past"),),
        relations=(RelationEdge("a", Relation.AGENT_OF, "v"), RelationEdge("k", Relation.LOCATION_OF, "v")))
    combined = combine((first, (doc, run)), document_id="both")
    answers = [e for e in question_generator().for_document(*combined) if e.question == "Who visited Kyiv?"]
    assert len(answers) == 1 and answers[0].answer == "Anna."
    assert all(e.answerable for e in answers)


def test_bare_rex_does_not_reveal_hidden_type():
    doc, run = generated()
    rex = next(m for m in run.mentions if m.exact_text == "Rex")
    assert not rex.type_grounded
    examples = question_generator().for_document(doc, run)
    typed = [e for e in examples if e.question == "What kind of entity is Rex?"]
    assert typed and all(not e.answerable for e in typed)
    assert not any(e.kind is QAKind.ONTOLOGY and "Rex" in e.question for e in examples)


@pytest.mark.parametrize("features,expected", [
    ({"tense": "present", "is_perfect": True}, "What has Anna done?"),
    ({"tense": "present", "is_progressive": True}, "What is Anna doing?"),
    ({"tense": "will", "is_perfect": True}, "What will Anna have done?"),
])
def test_event_questions_keep_the_entire_verb_chain(features, expected):
    examples = atomic_questions(*generated(**features), resolver=ParadigmResolver(known=DEFAULT_PARADIGMS))
    assert expected in {e.question for e in examples}


def _with_kyiv(doc, run):
    """Add a grounded locative adjunct, independent of random optional slots."""
    doc = replace(doc, text=doc.text[:-1] + " in Kyiv.")
    start = doc.text.rindex("Kyiv")
    kyiv = EntityMention("kyiv", doc.document_id, TextSpan(start, start + 4), "Kyiv", LabelSet.of(Label.LOCATION))
    run = replace(run, mentions=run.mentions + (kyiv,),
                  relations=run.relations + (RelationEdge("kyiv", Relation.LOCATION_OF, run.predicates[0].predicate_id),))
    return doc, run


def test_compound_question_keeps_the_verb_chain():
    doc, run = _with_kyiv(*generated(tense="will"))
    examples = compound_questions(doc, run, resolver=ParadigmResolver(known=DEFAULT_PARADIGMS))
    assert examples and all("will give" in e.question for e in examples)


def test_no_compound_question_about_a_negated_event():
    """It would ask where a non-event happened, under ambiguous negation scope."""
    doc, run = _with_kyiv(*generated(tense="past", is_negated=True))
    assert compound_questions(doc, run, resolver=ParadigmResolver(known=DEFAULT_PARADIGMS)) == []


def test_future_polarity_does_not_assert_a_past_event():
    examples = polarity_questions(*generated(tense="will"), resolver=ParadigmResolver(known=DEFAULT_PARADIGMS))
    assert examples
    assert all("happened" not in e.question + e.answer and "will give" in e.answer for e in examples)


def test_later_passage_evidence_uses_local_coordinates():
    text = "First sentence. Anna visited Kyiv."
    start = text.index("Anna")
    doc = Document("d", text, passages=(Passage("p", "d", TextSpan(start, len(text))),))
    run = AnnotationRun("r", mentions=(EntityMention("a", "d", TextSpan(start, start+4), "Anna", LabelSet.of(Label.PERSON)),),
                        predicates=(Predicate("v", "d", TextSpan(start+5, start+12), "visited", "visit", tense="past"),),
                        relations=(RelationEdge("a", Relation.AGENT_OF, "v"),))
    examples = question_generator().for_passages(doc, run)
    assert examples and all(e.context == "Anna visited Kyiv." and not e.check() for e in examples)
    who = next(e for e in examples if e.answer == "Anna.")
    assert who.evidence[0].text_in(who.context) == "Anna"


@pytest.mark.parametrize("text,absent", [
    ("No cars arrived.", SpeechAct.REJECTION),
    ("Good cars are expensive.", SpeechAct.APPROVAL),
    ("I am not sorry.", SpeechAct.APOLOGY),
    ("No problem.", SpeechAct.REJECTION),
])
def test_rule_teacher_does_not_confuse_content_with_speech_act(text, absent):
    assert absent not in classify_turn(text).speech_acts


def test_imperative_and_curly_apostrophe_negation():
    assert classify_turn("Stop.").mood is Mood.IMPERATIVE
    text = "I don’t know."
    assert classify_turn(text).polarity is Polarity.NEGATIVE
    assert all(text[t.start_char:t.end_char] == t.text for t in tokenize(text))
    assert SpeechAct.REASSURANCE in classify_turn("No problem.").speech_acts


def test_alignment_keeps_predicate_labels_negation_and_property_target():
    doc = Document("d", "Anna was sad. Anna was very tall.")
    response = TeacherResponse(annotations=(
        ProposedAnnotation("Anna", ("PERSON",)),
        ProposedAnnotation("Anna", ("PERSON",), occurrence=1),
        ProposedAnnotation("sad", ("STATE", "EMOTION"), lemma="sad", negated=True, tense="past"),
        ProposedAnnotation("very tall", ("PROPERTY",), head="tall", degree="very",
                           relation="PROPERTY_OF", target_text="Anna", target_occurrence=1),
    ))
    result = align_response(response, doc, run_id="r")
    assert not result.rejected
    pred = result.predicates[0]
    assert pred.labels == LabelSet.of(Label.STATE, Label.EMOTION)
    assert pred.polarity is Polarity.NEGATIVE and pred.tense == "past"
    assert result.properties[0].target_id == result.mentions[1].mention_id


@pytest.mark.parametrize("change", [
    {"lemma": "take"}, {"polarity": Polarity.NEGATIVE}, {"tense": "will"},
    {"aspect": "perfect"}, {"voice": "passive"}, {"modality": "might"},
])
def test_verifier_does_not_promote_semantic_disagreement(change):
    _, first = generated()
    first = replace(first, predicates=tuple(replace(p, review_status=ReviewStatus.UNREVIEWED) for p in first.predicates))
    second = replace(first, run_id="second", predicates=(replace(first.predicates[0], **change),))
    assert promote_agreed(first, second).predicates[0].review_status is ReviewStatus.UNREVIEWED


def test_verifier_compares_relations_and_dialogue():
    _, first = generated()
    first = replace(first, mentions=tuple(replace(m, review_status=ReviewStatus.UNREVIEWED) for m in first.mentions))
    edge = first.relations[0]
    second = replace(first, relations=(replace(edge, relation=Relation.THEME_OF),) + first.relations[1:])
    if edge.relation is Relation.THEME_OF:
        second = replace(second, relations=(replace(edge, relation=Relation.AGENT_OF),) + first.relations[1:])
    result = promote_agreed(first, second)
    assert next(m for m in result.mentions if m.mention_id == edge.source_id).review_status is ReviewStatus.UNREVIEWED
    first = AnnotationRun("a", dialogue=(DialogueAnnotation("u", speech_acts=(SpeechAct.AGREEMENT,)),))
    second = AnnotationRun("b", dialogue=(DialogueAnnotation("u", speech_acts=(SpeechAct.REJECTION,)),))
    assert promote_agreed(first, second).dialogue[0].review_status is ReviewStatus.UNREVIEWED


def test_independent_verification_preserves_dialogue_focus_and_annotations():
    doc = dialogue_document()
    first = annotate_document(doc, RuleBasedTeacher(), run_id="a").run
    second, agreement = verify_independently(first, doc, RuleBasedTeacher())
    assert len(second.dialogue) == 2
    assert agreement.agreement_rate == 1.0
    assert second.validate_against(doc) == []
    assert len({m.span for m in second.mentions}) == len(second.mentions)
    assert all(d.review_status is ReviewStatus.VERIFIED for d in promote_agreed(first, second).dialogue)


def test_storage_identity_and_split_fail_before_writes(tmp_path):
    store = CorpusStore(tmp_path)
    doc = Document("d", "Anna left.", split="train")
    store.add_document(doc)
    before = store.content_hash()
    for bad, error in [(replace(doc, text="Rex left."), StoreError),
                       (replace(doc, split="test"), SplitConflict),
                       (replace(doc, document_id="copy", split="test"), SplitConflict)]:
        with pytest.raises(error):
            store.add_document(bad)
        assert store.content_hash() == before


def test_storage_rejects_orphan_runs_and_examples(tmp_path):
    store = CorpusStore(tmp_path)
    doc, run = generated()
    with pytest.raises(StoreError):
        store.add_run(run)
    with pytest.raises(StoreError):
        store.add_examples([QAExample("Anna left.", "Who left?", "Anna.", document_id="missing")])
    assert store.count("annotation_runs") == store.count("spans") == store.count("qa_examples") == 0
    store.add_document(doc)
    with pytest.raises(StoreError):
        store.add_examples([QAExample(doc.text, "Who?", "Anna.", document_id=doc.document_id, run_id="missing")])


def test_dialogue_only_run_is_readable_by_document(tmp_path):
    doc = dialogue_document()
    run = AnnotationRun("r", dialogue=(DialogueAnnotation("u1", speech_acts=(SpeechAct.REACTION,)),))
    store = CorpusStore(tmp_path)
    store.add_document(doc)
    store.add_run(run)
    assert list(CorpusStore(tmp_path).runs(document_id="d")) == [run]


def test_new_semantic_fields_survive_storage(tmp_path):
    doc, run = generated(pattern="passive_agentless")
    pred = replace(run.predicates[0], predicate_type=Label.STATE, extra_labels=LabelSet.of(Label.EMOTION))
    run = replace(run, predicates=(pred,))
    store = CorpusStore(tmp_path)
    store.add_document(doc)
    store.add_run(run, doc)
    assert next(CorpusStore(tmp_path).runs()) == run


def test_bio_includes_properties_and_counts_nested_losses():
    from semantic_corpus.exporters.bio import BioReport, tag_document
    doc = Document("d", "red ball")
    prop = Property("p", "d", TextSpan(0, 3), "red", "red")
    run = AnnotationRun("r", properties=(prop,))
    report = BioReport()
    assert tag_document(doc, run, report=report).tags == ("B-PROPERTY", "O")
    assert report.tagged_spans == 1
    run = replace(run, mentions=(EntityMention("m", "d", TextSpan(0, 8), "red ball", LabelSet.of(Label.PHYSICAL_OBJECT)),))
    report = BioReport()
    tag_document(doc, run, report=report)
    assert report.dropped_overlapping == 1 and not report.is_lossless


def test_bio_reports_predicate_extra_labels_and_unknown_labels():
    from semantic_corpus.exporters.bio import BioReport, tag_document
    doc = Document("d", "sad")
    pred = Predicate("p", "d", TextSpan(0, 3), "sad", "sad", predicate_type=Label.STATE, extra_labels=LabelSet.of(Label.EMOTION))
    report = BioReport()
    tag_document(doc, AnnotationRun("r", predicates=(pred,)), report=report)
    assert report.dropped_labels == 1
    pred = replace(pred, predicate_type=None, extra_labels=LabelSet())
    report = BioReport()
    tag_document(doc, AnnotationRun("r", predicates=(pred,)), report=report)
    assert report.dropped_unrepresentable == 1


def test_build_deduplicates_atomically_and_can_repeat(tmp_path):
    from semantic_corpus.cli import main
    root, out = tmp_path / "store", tmp_path / "export"
    args = ["build", "--store", str(root), "--out", str(out), "--count", "100", "--seed", "42"]
    assert main(args) == 0
    store = CorpusStore(root)
    count = store.count("documents")
    assert count < 100  # This seed reproduces the content duplicate.
    assert store.count("annotation_runs") == count
    documents = {d.document_id for d in store.documents()}
    assert all(r["document_id"] in documents for r in store.read("spans"))
    with (out / "bio.jsonl").open(encoding="utf-8") as stream:
        assert sum(1 for _ in stream) == count
    with (out / "sft.jsonl").open(encoding="utf-8") as stream:
        assert sum(1 for _ in stream) == store.count("qa_examples")
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    assert main(args) == 0
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(StoreError):
        main(args[:-1] + ["43"])
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


def test_hashing_does_not_read_whole_files(tmp_path, monkeypatch):
    from pathlib import Path
    store = CorpusStore(tmp_path)
    store.add_document(Document("d", "Anna."))
    expected = store.content_hash()
    def forbidden(*args, **kwargs):
        raise AssertionError("read_bytes loads an entire table")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert store.content_hash() == expected


def bank_destination():
    from semantic_corpus.qasrl_core.models import (
        AnswerJudgment, InflectedForms, QuestionLabel, QuestionSlots, Sentence, Span, VerbEntry,
    )
    label = QuestionLabel("What did someone go to?",
        QuestionSlots("what", "did", "someone", "stem", "_", "to", "_"),
        "past", False, False, False, False,
        (AnswerJudgment("a", True, (Span(3, 4),)), AnswerJudgment("b", True, (Span(3, 4),))))
    entry = VerbEntry(1, InflectedForms("go", "goes", "going", "went", "gone"), {label.question_string: label})
    return Sentence("bank", ("Anna", "went", "to", "Kyiv", "."), {"1": entry})


def test_bank_does_not_label_to_kyiv_as_recipient_and_keeps_question(tmp_path):
    from semantic_corpus.qasrl_bridge import sentence_to_canonical
    doc, run = sentence_to_canonical(bank_destination())
    assert run.relations and run.relations[0].relation is None
    assert run.predicates[0].predicate_type is run.predicates[0].polarity is None
    examples = question_generator().for_document(doc, run)
    assert any(e.question == "What did someone go to?" and e.answer == "Kyiv." for e in examples)
    store = CorpusStore(tmp_path)
    store.add_document(doc)
    store.add_run(run, doc)
    assert next(CorpusStore(tmp_path).runs()) == run


def test_repeated_judgment_from_one_annotator_is_one_vote():
    from semantic_corpus.qasrl_bridge import sentence_to_canonical
    sentence = bank_destination()
    entry = sentence.verb_entries["1"]
    label = next(iter(entry.question_labels.values()))
    judgment = label.answer_judgments[0]
    label = replace(label, answer_judgments=(judgment, judgment))
    assert list(label.span_votes().values()) == [1]
    entry = replace(entry, question_labels={label.question_string: label})
    sentence = replace(sentence, verb_entries={"1": entry})
    assert not sentence_to_canonical(sentence, min_votes=2)[1].mentions


def test_report_detects_legacy_orphans(tmp_path):
    from semantic_corpus.exporters import build_report
    store = CorpusStore(tmp_path)
    # Simulate a store produced before reference validation was added.
    row = QAExample("Anna left.", "Who left?", "Anna.", document_id="absent").to_json()
    store.path_for("qa_examples").write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = build_report(store)
    assert not report.is_clean
    assert any("missing documents" in issue for issue in report.integrity_issues)


def test_run_index_handles_appends_and_legacy_dialogue_headers(tmp_path):
    doc = dialogue_document()
    first = AnnotationRun("first", dialogue=(DialogueAnnotation("u0", speech_acts=(SpeechAct.INFORM,)),))
    second = replace(first, run_id="second")
    store = CorpusStore(tmp_path)
    store.add_document(doc)
    store.add_run(first, doc)
    assert list(store.runs(document_id="d")) == [first]
    store.add_run(second, doc)
    assert list(store.runs(document_id="d")) == [first, second]
    headers = list(store.read("annotation_runs"))
    for header in headers:
        header.pop("document_ids")
    store.path_for("annotation_runs").write_text(
        "".join(json.dumps(h) + "\n" for h in headers), encoding="utf-8")
    assert list(CorpusStore(tmp_path).runs(document_id="d")) == [first, second]


def test_property_disagreement_is_not_verified_but_renamed_agreement_is():
    _, run = generated()
    run = replace(run, properties=tuple(replace(p, review_status=ReviewStatus.UNREVIEWED) for p in run.properties))
    assert run.properties
    second = replace(run, properties=(replace(run.properties[0], degree="very"),))
    assert promote_agreed(run, second).properties[0].review_status is ReviewStatus.UNREVIEWED
    renamed = {ident: ident + "-second" for ident in run.ids()}
    second = replace(run,
        mentions=tuple(replace(m, mention_id=renamed[m.mention_id]) for m in run.mentions),
        predicates=tuple(replace(p, predicate_id=renamed[p.predicate_id]) for p in run.predicates),
        properties=tuple(replace(p, property_id=renamed[p.property_id], target_id=renamed[p.target_id]) for p in run.properties),
        relations=tuple(replace(e, source_id=renamed[e.source_id], target_id=renamed[e.target_id]) for e in run.relations))
    assert promote_agreed(run, second).properties[0].review_status is ReviewStatus.VERIFIED


def test_rejected_second_annotation_does_not_verify_first():
    _, run = generated()
    run = replace(run, predicates=tuple(replace(p, review_status=ReviewStatus.UNREVIEWED) for p in run.predicates))
    second = replace(run, predicates=tuple(replace(p, review_status=ReviewStatus.REJECTED) for p in run.predicates))
    assert promote_agreed(run, second).predicates[0].review_status is ReviewStatus.UNREVIEWED


def test_native_bank_question_survives_alternative_answer_spans(tmp_path):
    from semantic_corpus.qasrl_core.models import Span
    from semantic_corpus.qasrl_bridge import bank_qa_examples, sentence_to_canonical
    sentence = bank_destination()
    entry = sentence.verb_entries["1"]
    label = next(iter(entry.question_labels.values()))
    label = replace(label, answer_judgments=tuple(
        replace(j, spans=(Span(3, 4), Span(2, 4))) for j in label.answer_judgments))
    sentence = replace(sentence, verb_entries={"1": replace(entry, question_labels={label.question_string: label})})
    doc, run = sentence_to_canonical(sentence)
    # Cover the shared canonical pipeline, including reloading from storage.
    store = CorpusStore(tmp_path)
    store.add_document(doc)
    store.add_run(run, doc)
    expected = bank_qa_examples(sentence)[0]
    found = [e for e in question_generator().for_document(doc, next(store.runs()))
             if e.question == expected.question]
    assert len(found) == 1
    assert found[0].answer == expected.answer
    assert found[0].metadata["alternative_answers"] == expected.metadata["alternative_answers"]


def test_document_identity_uses_json_representation_for_metadata(tmp_path):
    doc = Document("d", "Anna left.", metadata={"sources": ("a", "b")})
    store = CorpusStore(tmp_path)
    assert store.add_document(doc)
    assert not store.add_document(doc)
    store.add_run(AnnotationRun("r"), doc)
    assert next(store.runs()).run_id == "r"


def test_run_rejects_duplicate_ids_across_documents_before_writing(tmp_path):
    store = CorpusStore(tmp_path)
    docs = (Document("a", "Anna left."), Document("b", "Rex left."))
    for doc in docs:
        store.add_document(doc)
    run = AnnotationRun("r", mentions=tuple(
        EntityMention("same", doc.document_id, TextSpan(0, len(doc.text.split()[0])),
                      doc.text.split()[0], LabelSet.of(Label.PERSON)) for doc in docs))
    before = store.content_hash()
    with pytest.raises(StoreError, match="duplicate"):
        store.add_run(run)
    assert store.content_hash() == before


def test_legacy_run_membership_is_checked_when_adding_qa(tmp_path):
    doc, run = generated()
    store = CorpusStore(tmp_path)
    store.add_document(doc)
    store.add_document(Document("other", "Anna left."))
    store.add_run(run, doc)
    header = next(store.read("annotation_runs"))
    header.pop("document_ids")
    store.path_for("annotation_runs").write_text(json.dumps(header) + "\n", encoding="utf-8")
    store = CorpusStore(tmp_path)
    before = store.content_hash()
    with pytest.raises(StoreError, match="does not annotate"):
        store.add_examples([QAExample("Anna left.", "Who left?", "Anna.",
                                     document_id="other", run_id=run.run_id)])
    assert store.content_hash() == before
    assert store.add_examples([QAExample(doc.text, "Who?", "Anna.",
                                        document_id=doc.document_id, run_id=run.run_id)]) == 1


def test_property_target_comes_only_from_property_of():
    doc = Document("d", "sad Anna")
    result = align_response(TeacherResponse(annotations=(
        ProposedAnnotation("Anna", ("PERSON",)),
        ProposedAnnotation("sad", ("PROPERTY", "EMOTION"), head="sad",
                           relation="STATE_OF", target_text="Anna"),
    )), doc, run_id="r")
    assert result.properties[0].target_id is None


def test_verifier_does_not_confuse_span_ids_with_dialogue_keys():
    mention = EntityMention("dialogue:u", "d", TextSpan(0, 4), "Anna", LabelSet.of(Label.PERSON))
    dialogue = DialogueAnnotation("u", speech_acts=(SpeechAct.INFORM,))
    first = AnnotationRun("a", mentions=(mention,), dialogue=(dialogue,))
    second = replace(first, run_id="b", mentions=(replace(mention, labels=LabelSet.of(Label.ANIMAL)),))
    reviewed = promote_agreed(first, second)
    assert reviewed.mentions[0].review_status is ReviewStatus.UNREVIEWED
    assert reviewed.dialogue[0].review_status is ReviewStatus.VERIFIED


def test_build_checks_generator_version_before_appending_qa(tmp_path, monkeypatch):
    from semantic_corpus.cli import main
    from semantic_corpus.storage.repository import GENERATOR_VERSION
    root = tmp_path / "store"
    args = ["build", "--store", str(root), "--out", str(tmp_path / "out"), "--count", "1"]
    assert main(args) == 0
    store = CorpusStore(root)
    manifest = store.versions()[0]
    manifest["generator_version"] = GENERATOR_VERSION + "-old"
    store.path_for("dataset_versions").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    original = QuestionGenerator.for_document
    def changed_questions(self, document, run, **kwargs):
        return original(self, document, run, **kwargs) + [QAExample(
            document.text, "A newly added template?", "An answer.",
            document_id=document.document_id, run_id=run.run_id)]
    monkeypatch.setattr(QuestionGenerator, "for_document", changed_questions)
    before = store.content_hash()
    with pytest.raises(StoreError):
        main(args)
    assert store.content_hash() == before


def test_published_version_rejects_changed_configuration(tmp_path):
    store = CorpusStore(tmp_path)
    store.add_document(Document("d", "Anna left."))
    store.publish_version("v1", build_config={"seed": 1})
    with pytest.raises(StoreError):
        store.publish_version("v1", build_config={"seed": 2})


def test_a_rejected_example_writes_none_of_its_batch(tmp_path):
    store = CorpusStore(tmp_path)
    store.add_document(Document("d", "Anna left."))
    good = QAExample("Anna left.", "Who left?", "Anna.", document_id="d")
    bad = QAExample("Anna left.", "Who left?", "Anna.", document_id="missing")
    with pytest.raises(StoreError, match="unknown document"):
        store.add_examples([good, bad])
    assert store.count("qa_examples") == 0
    assert store.add_examples([good, good]) == 1


def test_independent_verification_sees_the_same_resources(monkeypatch):
    from semantic_corpus.semantic_annotator import pipeline
    from semantic_corpus.semantic_annotator.candidates import CandidateResources
    document = dialogue_document()
    resources = CandidateResources()
    run = annotate_document(document, RuleBasedTeacher(), run_id="r1", resources=resources).run
    seen = {}
    original = pipeline.annotate_document
    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, "annotate_document", spy)
    verify_independently(run, document, RuleBasedTeacher(), resources=resources)
    assert seen["resources"] is resources


@pytest.mark.parametrize("text, act, expected", [
    ("Good morning, sir.", SpeechAct.GREETING, True),
    ("Afternoon tea is ready.", SpeechAct.GREETING, False),
    ("I'm so sorry.", SpeechAct.APOLOGY, True),
    ("I am not sorry.", SpeechAct.APOLOGY, False),
    ("She said sorry.", SpeechAct.APOLOGY, False),
    ("Thank you so much.", SpeechAct.THANKING, True),
    ("No problem.", SpeechAct.REASSURANCE, True),
    ("No problem was found.", SpeechAct.REASSURANCE, False),
    ("Fine. Bye!", SpeechAct.FAREWELL, True),
])
def test_speech_act_cues_come_from_the_lexicon(text, act, expected):
    assert (act in classify_turn(text).speech_acts) is expected


def test_candidates_and_teacher_agree_on_sentence_openers():
    from semantic_corpus.semantic_annotator.candidates import extract_candidates
    text = "Fine. Well, let's go."
    proposed = {c.exact_text for c in extract_candidates(Document("d", text))
                if Label.DISCOURSE_MARKER in c.proposed_labels}
    taught = {marker[0] for marker in classify_turn(text).markers}
    assert "Well" in proposed
    assert proposed == taught


def test_the_shell_entry_reports_a_refused_write_without_a_traceback(monkeypatch, capsys):
    from semantic_corpus import cli
    def refuse(argv=None):
        raise StoreError("version 'v1' exists; use a new version")
    monkeypatch.setattr(cli, "main", refuse)
    assert cli._run_from_shell() == 2
    assert capsys.readouterr().err == "error: version 'v1' exists; use a new version\n"


@pytest.mark.parametrize("invalid", ["path", "cycle", "surrogate"])
def test_unserializable_example_leaves_batch_and_dedup_index_unchanged(tmp_path, invalid):
    store = CorpusStore(tmp_path)
    store.add_document(Document("d", "Anna left."))
    good = QAExample("Anna left.", "Who left?", "Anna.", document_id="d")
    value = {"path": tmp_path, "surrogate": "\ud800", "cycle": {}}[invalid]
    if invalid == "cycle":
        value["self"] = value
    bad = replace(good, question="What happened?", metadata={"value": value})
    before = store.content_hash()
    with pytest.raises(StoreError, match="cannot serialize JSON"):
        store.add_examples(iter([good, bad]))
    assert store.content_hash() == before
    assert store.count("qa_examples") == 0
    assert store.add_examples([good]) == 1
    assert store.add_examples([good]) == 0
    assert CorpusStore(tmp_path).add_examples([good]) == 0


def test_example_iterator_failure_does_not_write_its_prefix(tmp_path):
    store = CorpusStore(tmp_path)
    store.add_document(Document("d", "Anna left."))
    good = QAExample("Anna left.", "Who left?", "Anna.", document_id="d")

    def broken_input():
        yield good
        raise ValueError("upstream failed")

    before = store.content_hash()
    with pytest.raises(ValueError, match="upstream failed"):
        store.add_examples(broken_input())
    assert store.content_hash() == before
    assert store.add_examples([good]) == 1


def test_example_batch_releases_payloads_before_consuming_the_whole_input(tmp_path):
    import gc
    import weakref

    class Payload(str):
        pass

    store = CorpusStore(tmp_path)
    store.add_document(Document("d", "Anna left."))
    references = []

    def examples():
        for index in range(40):
            payload = Payload("x" * 65536)
            references.append(weakref.ref(payload))
            yield QAExample("Anna left.", f"Who left in case {index}?", "Anna.",
                            document_id="d", metadata={"payload": payload})
            gc.collect()
            # At most the current and previous record may still be live.
            assert sum(ref() is not None for ref in references) <= 2

    assert store.add_examples(examples()) == 40
    assert store.count("qa_examples") == 40
    assert all(len(e.metadata["payload"]) == 65536 for e in store.examples())


def test_batch_caches_run_headers_but_checks_each_documents_membership(tmp_path, monkeypatch):
    store = CorpusStore(tmp_path)
    doc = Document("d", "Anna left.")
    store.add_document(doc)
    store.add_document(Document("other", "Rex left."))
    store.add_run(AnnotationRun("r"), doc)
    original = store._rows_for
    calls = []

    def counted(table, key, value):
        if table == "annotation_runs":
            calls.append(value)
        return original(table, key, value)

    monkeypatch.setattr(store, "_rows_for", counted)
    good = QAExample(doc.text, "Who left?", "Anna.", document_id="d", run_id="r")
    assert store.add_examples([good, good], generation_run_id="r") == 1
    assert calls == ["r"]
    calls.clear()
    bad = replace(good, document_id="other")
    before = store.content_hash()
    with pytest.raises(StoreError, match="does not annotate"):
        store.add_examples([good, bad], generation_run_id="r")
    assert calls == ["r"]
    assert store.content_hash() == before


@pytest.mark.parametrize("agreement", ["legacy", "explicit", "partial"])
def test_native_answers_keep_vote_order_across_store_versions(tmp_path, agreement):
    from semantic_corpus.qasrl_core.models import Span
    from semantic_corpus.qasrl_bridge import sentence_to_canonical
    from semantic_corpus.question_generator.templates import native_questions

    sentence = bank_destination()
    entry = sentence.verb_entries["1"]
    label = next(iter(entry.question_labels.values()))
    judges = tuple(replace(j, spans=(Span(3, 4), Span(2, 4))) for j in label.answer_judgments)
    judges += (replace(judges[0], source_id="third", spans=(Span(3, 4),)),)
    label = replace(label, answer_judgments=judges)
    sentence = replace(sentence, verb_entries={"1": replace(entry, question_labels={label.question_string: label})})
    doc, run = sentence_to_canonical(sentence)
    if agreement == "legacy":
        run = replace(run, relations=tuple(replace(e, confidence=None) for e in run.relations))
    elif agreement == "partial":
        run = replace(run, relations=(replace(run.relations[0], confidence=None), *run.relations[1:]))
    else:
        # Explicit agreement must win even when record order is reversed.
        run = replace(run, relations=tuple(reversed(run.relations)))
    store = CorpusStore(tmp_path)
    store.add_document(doc)
    store.add_run(run, doc)
    example, = native_questions(doc, next(store.runs()))
    assert example.answer == "Kyiv."
    assert example.metadata["alternative_answers"] == ["to Kyiv"]
