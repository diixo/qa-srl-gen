"""Raw QANom conversion and interchange regression checks."""

from copy import deepcopy
import gzip
import json

import pytest

from semantic_corpus.qasrl_core.inflections import InflectionLexicon
from semantic_corpus.qasrl_core.models import InflectedForms, VerbForm
from semantic_corpus.qasrl_core.qanom import (
    QANomSentence, fix_up_question, modify_source, read_qanom,
    read_reformatted_qanom_data, reprocess_qanom, reprocess_qanom_sentence,
)


REBOUND = InflectedForms("rebound", "rebounds", "rebounding", "rebounded", "rebounded")


def lexicon(*paradigms):
    forward, reverse = {}, {}
    for forms in paradigms:
        forward.setdefault(forms.stem, []).append(forms)
        for form in VerbForm:
            reverse.setdefault(forms.get(form), []).append((forms, form))
    return InflectionLexicon({k: tuple(v) for k, v in forward.items()},
                             {k: tuple(v) for k, v in reverse.items()})


def label(worker="Worker-7", span=(3, 4)):
    return {"questionSources": [worker], "answerJudgments": [
        {"sourceId": worker, "isValid": True, "spans": [list(span)]},
        {"sourceId": "Worker-8", "isValid": False},
    ]}


def raw_sentence():
    return {
        "sentenceId": "TQA:test_0",
        "sentenceTokens": ["The", "rebound", "surprised", "Alice", "."],
        "verbEntries": {
            "0": {"isVerbal": False, "verbForm": "THE"},
            "1": {"isVerbal": True, "verbForm": "REBOUND", "questionLabels": {
                "Who rebound something?": label(),
            }},
        },
        "license": "example-license",
        "metadata": {"nested": ["retained"]},
    }


def test_qanom_reprocess_corrects_question_sources_and_non_predicates():
    raw = raw_sentence()
    before = deepcopy(raw)
    sentence = reprocess_qanom_sentence(raw, lexicon(REBOUND))
    assert raw == before
    assert isinstance(sentence, QANomSentence)
    assert sentence.non_predicates == {"0": "the"}
    assert sentence.extra == {"license": "example-license", "metadata": {"nested": ["retained"]}}
    entry = sentence.verb_entries["1"]
    assert entry.verb_inflected_forms == REBOUND
    assert set(entry.question_labels) == {"Who rebounded something?"}
    question = entry.question_labels["Who rebounded something?"]
    assert question.question_sources == ("turk-qanom-7",)
    assert {answer.source_id for answer in question.answer_judgments} == {"turk-qanom-7", "turk-qanom-8"}
    assert question.tense == "past"
    assert question.question_slots.verb == "past"
    assert QANomSentence.from_json(sentence.to_json()) == sentence


def test_qanom_corrected_question_collision_merges_all_annotations():
    raw = raw_sentence()
    raw["verbEntries"]["1"]["questionLabels"]["Who rebounded something?"] = label("Worker-9", (0, 1))
    sentence = reprocess_qanom_sentence(raw, lexicon(REBOUND))
    question = sentence.verb_entries["1"].question_labels["Who rebounded something?"]
    assert set(question.question_sources) == {"turk-qanom-7", "turk-qanom-9"}
    assert len(question.answer_judgments) == 3  # duplicate rejection merged once
    assert {tuple(span.to_json()) for judgment in question.valid_judgments for span in judgment.spans} == {(0, 1), (3, 4)}


def test_qanom_reverse_forms_and_all_questions_disambiguate_paradigm():
    lay = InflectedForms("lay", "lays", "laying", "laid", "laid")
    lie = InflectedForms("lie", "lies", "lying", "lay", "lain")
    raw = raw_sentence()
    raw["verbEntries"]["1"].update(verbForm="lay", questionLabels={
        "Who lay something?": label(), "Who is lying?": label(),
    })
    sentence = reprocess_qanom_sentence(raw, lexicon(lay, lie))
    assert sentence.verb_entries["1"].verb_inflected_forms == lie
    # Individually parsable questions do not justify mixing paradigms.
    raw["verbEntries"]["1"]["questionLabels"]["Who lays something?"] = label()
    with pytest.raises(ValueError, match="parse all questions"):
        reprocess_qanom_sentence(raw, lexicon(lay, lie))


def test_qanom_candidate_order_is_stable():
    first = InflectedForms("broadcast", "broadcasts", "broadcasting", "broadcast", "broadcast")
    second = InflectedForms("broadcast", "broadcasts", "broadcasting", "broadcasted", "broadcasted")
    raw = raw_sentence()
    raw["verbEntries"]["1"].update(verbForm="broadcast", questionLabels={"Who broadcasts?": label()})
    for entries in ((first, second), (second, first)):
        sentence = reprocess_qanom_sentence(raw, lexicon(*entries))
        assert sentence.verb_entries["1"].verb_inflected_forms == first


@pytest.mark.parametrize("source", ["worker-7", "Worker-x", "turk-qanom-7", "Worker-7 trailing"])
def test_qanom_rejects_unknown_raw_sources(source):
    with pytest.raises(ValueError, match="Unknown raw QANom source"):
        modify_source(source)


def test_qanom_unknown_forms_and_invalid_questions_raise():
    with pytest.raises(ValueError, match="Inflections not found"):
        reprocess_qanom_sentence(raw_sentence(), lexicon())
    raw = raw_sentence()
    raw["verbEntries"]["1"]["questionLabels"] = {"Why why why?": label()}
    with pytest.raises(ValueError, match="parse all questions"):
        reprocess_qanom_sentence(raw, lexicon(REBOUND))


@pytest.mark.parametrize("field,value,message", [
    ("isVerbal", "false", "must be a boolean"),
    ("verbForm", None, "must be a nonempty string"),
    ("questionLabels", [], "must be an object"),
])
def test_qanom_malformed_raw_entry_is_rejected(field, value, message):
    raw = raw_sentence()
    raw["verbEntries"]["1"][field] = value
    with pytest.raises(ValueError, match=message):
        reprocess_qanom_sentence(raw, lexicon(REBOUND))


@pytest.mark.parametrize("span", [[3, 99], ["3", 4], [3, 3]])
def test_qanom_invalid_answer_spans_are_rejected(span):
    raw = raw_sentence()
    raw["verbEntries"]["1"]["questionLabels"]["Who rebound something?"]["answerJudgments"][0]["spans"] = [span]
    with pytest.raises(ValueError):
        reprocess_qanom_sentence(raw, lexicon(REBOUND))


def test_qanom_non_predicate_overlap_is_rejected():
    raw = raw_sentence()
    raw["nonPredicates"] = {"1": "rebound"}
    with pytest.raises(ValueError, match="both a predicate and non-predicate"):
        reprocess_qanom_sentence(raw, lexicon(REBOUND))


@pytest.mark.parametrize("compressed", [False, True])
def test_qanom_stream_readers_handle_plain_and_gzip(tmp_path, compressed):
    path = tmp_path / ("data.jsonl.gz" if compressed else "data.jsonl")
    opener = gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(raw_sentence()) + "\n")
    sentence, = reprocess_qanom(path, lexicon(REBOUND))
    with opener(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(sentence.to_json()) + "\n")
    assert list(read_qanom(path)) == [sentence]


def test_qanom_reader_is_lazy_and_reports_record_context(tmp_path):
    path = tmp_path / "data.jsonl"
    sentence = reprocess_qanom_sentence(raw_sentence(), lexicon(REBOUND))
    path.write_text(json.dumps(sentence.to_json()) + "\n{}\n", encoding="utf-8")
    iterator = read_qanom(path)
    assert next(iterator) == sentence
    with pytest.raises(ValueError, match="record 2.*sentenceId"):
        next(iterator)


def test_qanom_partition_index_and_conflicting_duplicates(tmp_path):
    sentence = reprocess_qanom_sentence(raw_sentence(), lexicon(REBOUND))
    for partition in ("train", "dev", "test"):
        (tmp_path / f"{partition}.jsonl").write_text(json.dumps(sentence.to_json()) + "\n", encoding="utf-8")
    dataset = read_reformatted_qanom_data(tmp_path)
    assert sum(len(group) for group in dataset.qa_nom_sentences_by_id.values()) == 1
    different = sentence.to_json()
    different["sentenceTokens"][0] = "A"
    (tmp_path / "test.jsonl").write_text(json.dumps(different) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        read_reformatted_qanom_data(tmp_path)


def test_qanom_exact_upstream_correction_only():
    assert fix_up_question("Who rebound something?") == "Who rebounded something?"
    assert fix_up_question("Who does something rebound?") == "Who does something rebound?"
    assert modify_source("Worker-007") == "turk-qanom-007"


@pytest.mark.parametrize("field,value,message", [
    ("tense", "garbage", "Unknown QANom tense"),
    ("verb", "not a form", "Unknown QANom verb form"),
    ("verb", "pastparticiple", "Unknown QANom verb form"),
    ("verb", "garbage past", "Unknown QANom verb prefix"),
])
def test_qanom_reformatted_rejects_unknown_tense_and_verb_keys(field, value, message):
    data = reprocess_qanom_sentence(raw_sentence(), lexicon(REBOUND)).to_json()
    question = data["verbEntries"]["1"]["questionLabels"]["Who rebounded something?"]
    if field == "verb":
        question["questionSlots"]["verb"] = value
    else:
        question[field] = value
    with pytest.raises(ValueError, match=message):
        QANomSentence.from_json(data)


@pytest.mark.parametrize("reformatted", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_qanom_positive_judgment_requires_answers(reformatted, missing):
    data = raw_sentence()
    if reformatted:
        data = reprocess_qanom_sentence(data, lexicon(REBOUND)).to_json()
    question = next(iter(data["verbEntries"]["1"]["questionLabels"].values()))
    judgment = question["answerJudgments"][0]
    if missing:
        judgment.pop("spans")
    else:
        judgment["spans"] = []
    with pytest.raises(ValueError, match="valid answer judgment requires at least one answer span"):
        if reformatted:
            QANomSentence.from_json(data)
        else:
            reprocess_qanom_sentence(data, lexicon(REBOUND))


@pytest.mark.parametrize("reformatted", [False, True])
def test_qanom_semantic_judgment_sets_ignore_span_order(reformatted):
    data = raw_sentence()
    if reformatted:
        data = reprocess_qanom_sentence(data, lexicon(REBOUND)).to_json()
    source_prefix = "turk-qanom-" if reformatted else "Worker-"
    question = next(iter(data["verbEntries"]["1"]["questionLabels"].values()))
    question["answerJudgments"] = [
        {"sourceId": source_prefix + "1", "isValid": True, "spans": [[0, 1], [2, 3]]},
        {"sourceId": source_prefix + "1", "isValid": True, "spans": [[2, 3], [0, 1], [0, 1]]},
        {"sourceId": source_prefix + "2", "isValid": False},
        {"sourceId": source_prefix + "2", "isValid": False, "spans": []},
    ]
    sentence = QANomSentence.from_json(data) if reformatted else reprocess_qanom_sentence(data, lexicon(REBOUND))
    result = next(iter(sentence.verb_entries["1"].question_labels.values()))
    assert len(result.answer_judgments) == 2
    assert result.valid_ratio == 0.5
    assert [span.to_json() for span in result.valid_judgments[0].spans] == [[0, 1], [2, 3]]
    assert QANomSentence.from_json(sentence.to_json()) == sentence
