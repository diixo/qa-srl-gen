"""Bank metadata, provenance and optional legacy conversion contracts."""

from dataclasses import replace
import gzip
import json

import pytest

from semantic_corpus.qasrl_core.bank_index import (
    DataIndex, DatasetPartition, Document, DocumentId, DocumentMetadata, Domain, SentenceId, read_index,
)
from semantic_corpus.qasrl_core.bank_reader import QaTextQuestion, read_consolidated_bank
from semantic_corpus.qasrl_core.bank_sources import AnnotationRound, AnswerSource, QuestionSource, filter_expanded_to_orig
from semantic_corpus.qasrl_core.dataset import ConsolidatedSentence, Dataset
from semantic_corpus.qasrl_core.models import AnswerJudgment, InflectedForms, QuestionLabel, QuestionSlots, Sentence, Span, VerbEntry


@pytest.mark.parametrize("value,domain,doc,paragraph,sentence", [
    ("Wiki1k:wikipedia:123:2:10", Domain.WIKIPEDIA, "123", 2, 10),
    ("Wiki1k:wikinews:42:0:2", Domain.WIKINEWS, "42", 0, 2),
    ("TQA:T_0359_12", Domain.TQA, "T_0359", 0, 12),
])
def test_sentence_id_codec_and_domains(value, domain, doc, paragraph, sentence):
    sid = SentenceId.from_string(value)
    assert sid == SentenceId(DocumentId(domain, doc), paragraph, sentence)
    assert sid.to_json() == value
    assert sid.domain == domain
    assert DocumentId.from_json(sid.document_id.to_json()) == sid.document_id


@pytest.mark.parametrize("value", ["TQA:x", "TQA:x_-1", "Wiki1k:tqa:x:0:1", "Wiki1k:wikipedia:x:0:a", "Wiki1k:wikipedia:x:0:1:2"])
def test_invalid_sentence_ids_raise(value):
    with pytest.raises(ValueError, match="sentence ID"):
        SentenceId.from_string(value)


def test_sentence_id_natural_ordering_and_lossless_tqa_constraints():
    values = ["TQA:T_1_1", "Wiki1k:wikinews:1:0:1", "Wiki1k:wikipedia:1:0:10", "Wiki1k:wikipedia:1:0:2"]
    assert [str(value) for value in sorted(map(SentenceId.from_string, values))] == [values[3], values[2], values[1], values[0]]
    with pytest.raises(ValueError, match="paragraph"):
        SentenceId(DocumentId(Domain.TQA, "x"), 1, 1)


def index_json():
    return {
        "documents": {"train": [{"part": "train", "idString": "TQA:T_0359", "domain": "tqa", "id": "T_0359", "title": "A topic"}], "dev": []},
        "denseIds": ["TQA:T_0359_10", "TQA:T_0359_2"],
    }


def test_index_round_trip_defaults_and_gzip(tmp_path):
    data = index_json()
    path = tmp_path / "index.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
    index = read_index(path)
    assert index.num_documents == 1
    assert index.qa_nom_ids == index.qasrl_gs_ids == ()
    assert index.get_part(DocumentId(Domain.TQA, "T_0359")) is DatasetPartition.TRAIN
    assert tuple(map(str, index.dense_ids)) == ("TQA:T_0359_2", "TQA:T_0359_10")
    assert DataIndex.from_json(index.to_json()) == index
    assert DataIndex().all_document_metas == ()
    with pytest.raises(KeyError):
        index.get_part(DocumentId(Domain.WIKIPEDIA, "unknown"))


@pytest.mark.parametrize("field,value", [("qaNomIds", ["bad"]), ("qasrlGsIds", None), ("denseIds", "TQA:T_0359_0")])
def test_malformed_optional_index_fields_are_not_silently_discarded(field, value):
    data = index_json() | {field: value}
    with pytest.raises(ValueError):
        DataIndex.from_json(data)


def test_index_rejects_conflicting_partitions_and_names_bad_file(tmp_path):
    data = index_json()
    data["documents"]["train"][0]["part"] = "test"
    with pytest.raises(ValueError, match="partition"):
        DataIndex.from_json(data)
    path = tmp_path / "bad.json"
    path.write_text("broken", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.json"):
        read_index(path)


def test_document_codec_orders_sentences_numerically_and_checks_ownership():
    meta = DocumentMetadata(DocumentId(Domain.TQA, "T_0359"), DatasetPartition.TRAIN, "Topic")
    s10 = ConsolidatedSentence("TQA:T_0359_10", ("ten",))
    s2 = ConsolidatedSentence("TQA:T_0359_2", ("two",), non_predicates={"0": "two"})
    document = Document(meta, (s10, s2, s2))
    assert document.sentences == (s2, s10)
    assert Document.from_json(document.to_json()) == document
    with pytest.raises(ValueError, match="different document"):
        Document(meta, (replace(s2, sentence_id="TQA:other_2"),))
    with pytest.raises(ValueError, match="duplicate"):
        Document(meta, (s2, replace(s2, sentence_tokens=("conflict",))))


def test_read_consolidated_is_lazy_and_preserves_nominal_extra(tmp_path):
    path = tmp_path / "consolidated.jsonl"
    record = {"sentenceId": "TQA:T_0359_0", "sentenceTokens": ["a"], "verbEntries": {}, "nonPredicates": {"0": "a"}, "nomEntries": {"0": {"raw": 1}}}
    path.write_text(json.dumps(record) + "\nmalformed\n", encoding="utf-8")
    stream = read_consolidated_bank(path)
    assert next(stream).to_json() == record
    with pytest.raises(ValueError, match="2"):
        next(stream)


@pytest.mark.parametrize("source,round_", [
    ("turk-qasrl2.0-17", AnnotationRound.ORIGINAL),
    ("turk-qasrl2.0-17-expansion", AnnotationRound.EXPANSION),
    ("turk-qasrl2.0-17-eval", AnnotationRound.EVAL),
    ("turk-qanom-17", AnnotationRound.QANOM),
])
def test_answer_source_rounds(source, round_):
    parsed = AnswerSource.from_string(source)
    assert parsed.turker_id == "17" and parsed.round is round_
    assert str(parsed) == source


@pytest.mark.parametrize("source", ["model-qasrl2.0-v2", "turk-qasrl2.0-17", "turk-qanom-17"])
def test_question_source_round_trip(source):
    assert str(QuestionSource.from_string(source)) == source


@pytest.mark.parametrize("parse,value", [(AnswerSource.from_string, "turk-qasrl2.0-17-unknown"), (AnswerSource.from_string, "turk-qasrl2.0-17-eval-"), (QuestionSource.from_string, "turk-qasrl2.0-17-eval")])
def test_unknown_provenance_is_not_invented(parse, value):
    with pytest.raises(ValueError):
        parse(value)


def test_expanded_to_orig_filters_sources_and_answers_independently(give_forms):
    slots = QuestionSlots("who", "_", "_", "past", "something", "_", "_")
    judgments = tuple(AnswerJudgment(source, True, (Span(0, 1),)) for source in ("turk-qasrl2.0-1", "turk-qasrl2.0-2-expansion", "turk-qanom-3"))
    question = QuestionLabel("Who gave something?", slots, "past", False, False, False, False, judgments, ("model-qasrl2.0-v1", "turk-qasrl2.0-1"))
    model_only = replace(question, question_string="What gave something?", question_sources=("model-qasrl2.0-v1",))
    entry = VerbEntry(1, give_forms, {question.question_string: question, model_only.question_string: model_only})
    original = Dataset.from_sentences([Sentence("s", ("Anna", "gave", "rice"), {"1": entry})])
    filtered = filter_expanded_to_orig(original)
    labels = filtered.sentences["s"].verb_entries["1"].question_labels
    assert list(labels) == [question.question_string]
    assert labels[question.question_string].question_sources == ("turk-qasrl2.0-1",)
    assert labels[question.question_string].answer_judgments == (judgments[0],)
    assert len(original.sentences["s"].verb_entries["1"].question_labels) == 2


def test_legacy_slots_convert_without_reparsing_boundaries(give_forms):
    raw = QaTextQuestion(("what", "did", "someone", "give", "something", "to", "_"), ("Anna",))
    slots = raw.to_question_slots(give_forms)
    assert slots.verb == "stem" and slots.prep == "to" and slots.obj == "something"
    assert raw.answers == ("Anna",)
    with pytest.raises(ValueError, match="No valid"):
        replace(raw, slots=("what", "did", "someone", "unknown", "something", "to", "_")).to_question_slots(give_forms)


def test_legacy_silent_preposition_and_ambiguous_forms():
    help_forms = InflectedForms("help", "helps", "helping", "helped", "helped")
    raw = QaTextQuestion(("what", "did", "someone", "help", "someone", "_", "do"), ())
    assert raw.to_question_slots(help_forms).prep == ""
    put_forms = InflectedForms("put", "puts", "putting", "put", "put")
    raw = QaTextQuestion(("who", "_", "_", "put", "something", "_", "_"), ())
    analyses = raw.to_question_slots_all(put_forms)
    assert {slots.verb for slots in analyses} == {"past"}
    # A caller-supplied syncretic paradigm must not silently choose a tense.
    syncretic = replace(put_forms, present_singular_3rd="put")
    assert {slots.verb for slots in raw.to_question_slots_all(syncretic)} == {"presentSingular3rd", "past"}
    with pytest.raises(ValueError, match="Ambiguous"):
        raw.to_question_slots(syncretic)


@pytest.mark.corpus
def test_real_bank_index(repo_root):
    path = repo_root / "data" / "qasrl-v2" / "index.json.gz"
    if not path.exists():
        pytest.skip("Bank metadata not available")
    index = read_index(path)
    assert index.num_documents > 100
    assert index.dense_ids
    assert DataIndex.from_json(index.to_json()) == index
