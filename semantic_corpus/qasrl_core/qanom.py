"""QANom interchange and raw-annotation reprocessing.

Ported from ``apps/qanom-reformat`` at julianmichael/qasrl 16ab4949 (MIT;
see THIRD_PARTY_NOTICES.md). File readers stream JSONL/gzip; QANomData is
the explicitly materialized document index. Unknown sentence fields survive
reprocessing, and colliding corrected questions merge their annotations.
Malformed positive judgments with no answers are rejected explicitly, rather
than silently being changed into negative judgments as in upstream's decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import re
from typing import Any, Iterator, Mapping

from .bank_index import DocumentId, SentenceId
from .bank_reader import read_jsonl
from .dataset import ConsolidatedSentence
from .inflections import InflectionLexicon
from .models import AnswerJudgment, InflectedForms, QuestionLabel, Span, VerbEntry, VerbForm
from .slot_based_label import read_preferred_state
from .state_machine import VERB_PREFIX_WORDS
from .tense import Tense

__all__ = [
    "QANomSentence", "QANomData", "read_qanom", "read_reformatted_qanom_data",
    "modify_source", "fix_up_question", "reprocess_qanom_sentence", "reprocess_qanom",
]


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _array(value: Any, name: str) -> list | tuple:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    return value


def _index(value: Any, size: int, name: str) -> int:
    if isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
        value = int(value)
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < size:
        raise ValueError(f"{name} must be a token index in [0, {size}), got {value!r}")
    return value


def _sentence_fields(data: Any):
    data = _object(data, "QANom sentence")
    sid = _string(data.get("sentenceId"), "sentenceId")
    tokens = tuple(_string(token, "sentenceTokens element") for token in
                   _array(data.get("sentenceTokens"), "sentenceTokens"))
    entries = _object(data.get("verbEntries"), "verbEntries")
    return sid, tokens, entries


def _non_predicates(value: Any, size: int) -> dict[str, str]:
    result = {}
    for key, form in _object(value, "nonPredicates").items():
        key = str(_index(key, size, "nonPredicates key"))
        result[key] = _string(form, "nonPredicates form").lower()
    return result


def _read_judgment(data: Any, size: int, *, raw: bool) -> AnswerJudgment:
    data = _object(data, "answerJudgment")
    source = _string(data.get("sourceId"), "answerJudgment.sourceId")
    if raw:
        source = modify_source(source)
    valid = _boolean(data.get("isValid"), "answerJudgment.isValid")
    spans = []
    for value in _array(data.get("spans", []), "answerJudgment.spans"):
        pair = _array(value, "answer span")
        if len(pair) != 2 or any(type(index) is not int for index in pair):
            raise ValueError("answer span must contain two integer token offsets")
        span = Span(*pair)
        if span.end > size:
            raise ValueError(f"answer span {pair!r} exceeds sentence length {size}")
        if span not in spans:
            spans.append(span)
    if not valid and spans:
        raise ValueError("an invalid answer judgment cannot carry answer spans")
    if valid and not spans:
        raise ValueError("a valid answer judgment requires at least one answer span")
    # Scala uses a set of spans inside each judgment and a set of judgments.
    # Canonical order makes equality/deduplication insensitive to input order.
    return AnswerJudgment(source, valid, tuple(sorted(spans)))


@dataclass(frozen=True)
class QANomSentence(ConsolidatedSentence):
    """The same typed schema as upstream QANomSentence, with extra fields."""

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> QANomSentence:
        _, tokens, entries = _sentence_fields(data)
        non_predicates = _non_predicates(data.get("nonPredicates"), len(tokens))
        normalized_judgments = {}
        for key, entry in entries.items():
            index = _index(key, len(tokens), "verbEntries key")
            entry = _object(entry, f"verbEntries[{key}]")
            if type(entry.get("verbIndex")) is not int or entry["verbIndex"] != index:
                raise ValueError(f"verbEntries[{key}].verbIndex must match its key")
            if str(index) in non_predicates:
                raise ValueError(f"token {index} is both a predicate and non-predicate")
            forms = _object(entry.get("verbInflectedForms"), "verbInflectedForms")
            for name in ("stem", "presentSingular3rd", "presentParticiple", "past", "pastParticiple"):
                _string(forms.get(name), f"verbInflectedForms.{name}")
            for question, label in _object(entry.get("questionLabels"), "questionLabels").items():
                _string(question, "questionLabels key")
                label = _object(label, f"questionLabels[{question}]")
                if label.get("questionString") != question:
                    raise ValueError("questionString must match the questionLabels key")
                for name in ("isPerfect", "isProgressive", "isNegated", "isPassive"):
                    _boolean(label.get(name), name)
                tense = _string(label.get("tense"), "tense")
                try:
                    Tense(tense)
                except ValueError as error:
                    raise ValueError(f"Unknown QANom tense: {tense!r}") from error
                slots = _object(label.get("questionSlots"), "questionSlots")
                for name in ("wh", "aux", "subj", "verb", "obj", "prep", "obj2"):
                    if not isinstance(slots.get(name), str):
                        raise ValueError(f"questionSlots.{name} must be a string")
                verb_words = slots["verb"].split(" ")
                try:
                    VerbForm(verb_words[-1])
                except ValueError as error:
                    raise ValueError(f"Unknown QANom verb form: {slots['verb']!r}") from error
                if any(word not in VERB_PREFIX_WORDS for word in verb_words[:-1]):
                    raise ValueError(f"Unknown QANom verb prefix: {slots['verb']!r}")
                for source in _array(label.get("questionSources", []), "questionSources"):
                    _string(source, "question source")
                normalized_judgments[key, question] = tuple(dict.fromkeys(
                    _read_judgment(judgment, len(tokens), raw=False)
                    for judgment in _array(label.get("answerJudgments"), "answerJudgments")
                ))
        try:
            result = super().from_json(data)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed QANom sentence: {error}") from error
        # Use the checked, set-normalized judgments, not the unnormalized
        # generic Bank decoder's tuples. Preserve all sentence extra fields.
        normalized_entries = {
            key: replace(entry, question_labels={
                question: replace(label,
                                  answer_judgments=normalized_judgments[key, question],
                                  question_sources=tuple(dict.fromkeys(label.question_sources)))
                for question, label in entry.question_labels.items()
            })
            for key, entry in result.verb_entries.items()
        }
        return cls(result.sentence_id, result.sentence_tokens, normalized_entries,
                   non_predicates, result.extra)


def modify_source(source: str) -> str:
    """Convert upstream raw Worker-N provenance into the Bank namespace."""
    match = re.fullmatch(r"Worker-([0-9]+)", _string(source, "source"))
    if match is None:
        raise ValueError(f"Unknown raw QANom source: {source!r}")
    return f"turk-qanom-{match[1]}"


def fix_up_question(question: str) -> str:
    """Apply the one corpus correction from upstream Reprocess.scala."""
    return "Who rebounded something?" if question == "Who rebound something?" else question


def _candidate_paradigms(form: str, lexicon: InflectionLexicon) -> tuple[InflectedForms, ...]:
    # Prefer forward lookup, then reverse candidates; sort both groups so the
    # choice does not depend on dictionary construction or hash iteration order.
    forward = sorted(set(lexicon.paradigms(form)), key=lambda forms: forms.all_forms)
    reverse = sorted({forms for forms, _ in lexicon.analyses(form)}, key=lambda forms: forms.all_forms)
    return tuple(dict.fromkeys((*forward, *reverse)))


def reprocess_qanom_sentence(data: Mapping[str, Any], lexicon: InflectionLexicon) -> QANomSentence:
    """Reformat one raw QANom record, choosing a paradigm fitting every QA.

    ``isVerbal=false`` entries become ``nonPredicates``; true entries use the
    associated verbal paradigm to analyze questions about nominal predicates.
    This performs no model inference or dictionary downloads.
    """
    sid, tokens, raw_entries = _sentence_fields(data)
    non_predicates = _non_predicates(data.get("nonPredicates", {}), len(tokens))
    entries: dict[str, VerbEntry] = {}
    for key, value in raw_entries.items():
        index = _index(key, len(tokens), "verbEntries key")
        key = str(index)
        raw = _object(value, f"verbEntries[{key}]")
        verbal = _boolean(raw.get("isVerbal"), f"verbEntries[{key}].isVerbal")
        form = _string(raw.get("verbForm"), f"verbEntries[{key}].verbForm").lower()
        if not verbal:
            if key in non_predicates and non_predicates[key] != form:
                raise ValueError(f"Conflicting non-predicate forms at token {index}")
            non_predicates[key] = form
            continue
        if key in non_predicates:
            raise ValueError(f"token {index} is both a predicate and non-predicate")
        candidates = _candidate_paradigms(form, lexicon)
        if not candidates:
            raise ValueError(f"Inflections not found for QANom verb form {form!r}")
        questions = []
        for original, question_data in _object(raw.get("questionLabels"), "questionLabels").items():
            original = _string(original, "questionLabels key")
            question_data = _object(question_data, f"questionLabels[{original}]")
            if "questionString" in question_data and question_data["questionString"] != original:
                raise ValueError("questionString must match the questionLabels key")
            question = fix_up_question(original)
            sources = tuple(sorted({modify_source(source) for source in
                                    _array(question_data.get("questionSources"), "questionSources")}))
            judgments = tuple(dict.fromkeys(
                _read_judgment(judgment, len(tokens), raw=True)
                for judgment in _array(question_data.get("answerJudgments"), "answerJudgments")
            ))
            questions.append((question, sources, judgments))
        states = None
        for forms in candidates:
            trial = [read_preferred_state(tokens, forms, question) for question, _, _ in questions]
            if all(state is not None for state in trial):
                states = trial
                break
        if states is None:
            raise ValueError(f"No inflected forms for {form!r} parse all questions: "
                             + ", ".join(repr(question) for question, _, _ in questions))
        labels = {}
        for (question, sources, judgments), state in zip(questions, states):
            assert state is not None
            frame = state.frame
            label = QuestionLabel(
                question, frame.to_slots_upstream(state.answer_slot), frame.tense,
                frame.is_perfect, frame.is_progressive, frame.is_negated, frame.is_passive,
                judgments, sources,
            )
            labels[question] = labels[question].combine_with_like(label) if question in labels else label
        entries[key] = VerbEntry(index, forms, dict(sorted(labels.items())))
    extra = {key: value for key, value in data.items()
             if key not in {"sentenceId", "sentenceTokens", "verbEntries", "nonPredicates"}}
    return QANomSentence(sid, tokens, entries, non_predicates, extra)


def read_qanom(path: Path | str) -> Iterator[QANomSentence]:
    """Stream already reformatted QANom JSONL or JSONL.gz."""
    for record_number, record in enumerate(read_jsonl(path), 1):
        try:
            yield QANomSentence.from_json(record)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}, record {record_number}: {error}") from error


def reprocess_qanom(path: Path | str, lexicon: InflectionLexicon) -> Iterator[QANomSentence]:
    """Stream and reprocess raw QANom JSONL or JSONL.gz."""
    for record_number, record in enumerate(read_jsonl(path), 1):
        try:
            yield reprocess_qanom_sentence(record, lexicon)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}, record {record_number}: {error}") from error


@dataclass(frozen=True)
class QANomData:
    qa_nom_sentences_by_id: dict[DocumentId, dict[SentenceId, QANomSentence]] = field(default_factory=dict)


def read_reformatted_qanom_data(path: Path | str) -> QANomData:
    """Read train/dev/test into upstream's document/sentence index.

    Each partition may be ``.jsonl`` or ``.jsonl.gz``. Conflicting duplicate
    sentence IDs raise ValueError instead of silently replacing annotations.
    """
    root = Path(path)
    documents: dict[DocumentId, dict[SentenceId, QANomSentence]] = {}
    for partition in ("train", "dev", "test"):
        source = root / f"{partition}.jsonl"
        if not source.exists():
            source = root / f"{partition}.jsonl.gz"
        for sentence in read_qanom(source):
            sid = SentenceId.from_string(sentence.sentence_id)
            group = documents.setdefault(sid.document_id, {})
            if sid in group and group[sid] != sentence:
                raise ValueError(f"Conflicting duplicate QANom sentence: {sid}")
            group[sid] = sentence
    return QANomData({doc: dict(sorted(sentences.items())) for doc, sentences in sorted(documents.items())})
