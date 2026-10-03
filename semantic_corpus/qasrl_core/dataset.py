"""In-memory QA-SRL dataset algebra, ported from julianmichael/qasrl (MIT).

Source: qasrl/data/Dataset.scala and qasrl/bank/Consolidated*.scala at
16ab4949. See THIRD_PARTY_NOTICES.md. Bank file readers remain streaming;
constructing a Dataset is an explicit request to materialize its contents.

Merges are deterministic and left-biased on conflicts, with diagnostics
containing both records. Unlike upstream, mismatched tokens are rejected
before combining token-indexed annotations, and unknown JSON data is retained.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Iterator, Mapping

from .models import QuestionLabel, Sentence, VerbEntry

__all__ = [
    "Dataset", "MergeResult", "DataMergeFailure", "SentenceMergeFailure",
    "VerbMergeFailure", "QuestionMergeFailure", "ConsolidatedSentence",
    "ConsolidatedDataset", "ConsolidatedMergeResult",
]


@dataclass(frozen=True)
class DataMergeFailure:
    sentence_id: str
    sentence_tokens: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class SentenceMergeFailure(DataMergeFailure):
    s1: Sentence
    s2: Sentence


@dataclass(frozen=True)
class VerbMergeFailure(DataMergeFailure):
    v1: VerbEntry
    v2: VerbEntry


@dataclass(frozen=True)
class QuestionMergeFailure(DataMergeFailure):
    v1: VerbEntry
    v2: VerbEntry
    q1: QuestionLabel
    q2: QuestionLabel


@dataclass(frozen=True)
class MergeResult:
    dataset: "Dataset"
    failures: tuple[DataMergeFailure, ...] = ()


def _merge_extra(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Union unknown fields; incompatible data must not disappear silently."""
    conflicts = [key for key in left.keys() & right.keys() if left[key] != right[key]]
    if conflicts:
        raise ValueError("Conflicting extra fields: " + ", ".join(sorted(conflicts)))
    return dict(left) | dict(right)


@dataclass(frozen=True)
class Dataset:
    sentences: dict[str, Sentence] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, sentence in self.sentences.items():
            if key != sentence.sentence_id:
                raise ValueError(f"Sentence key {key!r} differs from ID {sentence.sentence_id!r}")
        object.__setattr__(self, "sentences", dict(sorted(self.sentences.items())))

    def __iter__(self) -> Iterator[Sentence]:
        return iter(self.sentences.values())

    @classmethod
    def from_sentences(cls, sentences: Iterable[Sentence]) -> "Dataset":
        """Materialize unique sentences; duplicate IDs require explicit merge."""
        values: dict[str, Sentence] = {}
        for sentence in sentences:
            if sentence.sentence_id in values:
                raise ValueError(f"Duplicate sentence ID: {sentence.sentence_id!r}")
            values[sentence.sentence_id] = sentence
        return cls(values)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Dataset":
        return cls({key: Sentence.from_json(value) for key, value in data["sentences"].items()})

    def to_json(self) -> dict[str, Any]:
        return {"sentences": {key: value.to_json() for key, value in self.sentences.items()}}

    def filter_sentence_ids(self, predicate: Callable[[str], bool]) -> "Dataset":
        return Dataset({key: sentence for key, sentence in self.sentences.items() if predicate(key)})

    def filter_sentences(self, predicate: Callable[[Sentence], bool]) -> "Dataset":
        return Dataset({key: sentence for key, sentence in self.sentences.items() if predicate(sentence)})

    def cull_verbless_sentences(self) -> "Dataset":
        return self.filter_sentences(lambda sentence: bool(sentence.verb_entries))

    def cull_questionless_sentences(self) -> "Dataset":
        return self.filter_sentences(lambda sentence: any(entry.question_labels for entry in sentence))

    def cull_questionless_verbs(self) -> "Dataset":
        return Dataset({
            key: replace(sentence, verb_entries={k: v for k, v in sentence.verb_entries.items() if v.question_labels})
            for key, sentence in self.sentences.items()
        })

    def map_question_labels(self, transform: Callable[[QuestionLabel], QuestionLabel | None]) -> "Dataset":
        """Transform each label once, dropping None; preserve empty containers."""
        sentences = {}
        for sid, sentence in self.sentences.items():
            entries = {}
            for key, entry in sentence.verb_entries.items():
                labels = {}
                for question, label in entry.question_labels.items():
                    changed = transform(label)
                    if changed is not None:
                        labels[question] = changed
                entries[key] = replace(entry, question_labels=labels)
            sentences[sid] = replace(sentence, verb_entries=entries)
        return Dataset(sentences)

    def filter_question_labels(self, predicate: Callable[[QuestionLabel], bool]) -> "Dataset":
        return self.map_question_labels(lambda label: label if predicate(label) else None)

    def filter_question_sources(self, predicate: Callable[[str], bool]) -> "Dataset":
        def filter_label(label: QuestionLabel) -> QuestionLabel | None:
            sources = tuple(source for source in label.question_sources if predicate(source))
            return replace(label, question_sources=sources) if sources else None
        return self.map_question_labels(filter_label)

    def merge(self, other: "Dataset") -> MergeResult:
        """Merge all compatible annotations and report each retained conflict.

        A question conflict retains the left question; a verb conflict retains
        the entire left verb. A sentence-token or extra-field conflict retains
        the entire left sentence, so right-hand token offsets are never mixed
        with a different tokenization. Other records continue merging.
        """
        sentences: dict[str, Sentence] = {}
        failures: list[DataMergeFailure] = []
        for sid in sorted(self.sentences.keys() | other.sentences.keys()):
            if sid not in self.sentences or sid not in other.sentences:
                sentences[sid] = self.sentences[sid] if sid in self.sentences else other.sentences[sid]
                continue
            left, right = self.sentences[sid], other.sentences[sid]
            try:
                if left.sentence_tokens != right.sentence_tokens:
                    raise ValueError("Cannot combine different sentence tokens")
                extra = _merge_extra(left.extra, right.extra)
            except ValueError as error:
                failures.append(SentenceMergeFailure(sid, left.sentence_tokens, str(error), left, right))
                sentences[sid] = left
                continue
            entries = {}
            for key in sorted(left.verb_entries.keys() | right.verb_entries.keys(), key=int):
                if key not in left.verb_entries or key not in right.verb_entries:
                    entries[key] = left.verb_entries[key] if key in left.verb_entries else right.verb_entries[key]
                    continue
                lv, rv = left.verb_entries[key], right.verb_entries[key]
                if (lv.verb_index, lv.verb_inflected_forms) != (rv.verb_index, rv.verb_inflected_forms):
                    failures.append(VerbMergeFailure(sid, left.sentence_tokens, "Cannot combine different verb indices or forms", lv, rv))
                    entries[key] = lv
                    continue
                labels = {}
                for question in sorted(lv.question_labels.keys() | rv.question_labels.keys()):
                    if question not in lv.question_labels or question not in rv.question_labels:
                        labels[question] = lv.question_labels[question] if question in lv.question_labels else rv.question_labels[question]
                        continue
                    lq, rq = lv.question_labels[question], rv.question_labels[question]
                    try:
                        labels[question] = lq.combine_with_like(rq)
                    except ValueError as error:
                        failures.append(QuestionMergeFailure(sid, left.sentence_tokens, str(error), lv, rv, lq, rq))
                        labels[question] = lq
                entries[key] = replace(lv, question_labels=labels)
            sentences[sid] = replace(left, verb_entries=entries, extra=extra)
        return MergeResult(Dataset(sentences), tuple(failures))

    def combine(self, other: "Dataset", process_merge_failures: Callable[[tuple[DataMergeFailure, ...]], None]) -> "Dataset":
        """Scala's monoid combine: report diagnostics through a required sink."""
        result = self.merge(other)
        process_merge_failures(result.failures)
        return result.dataset


@dataclass(frozen=True)
class ConsolidatedSentence:
    """Bank consolidated / QANom interchange sentence with non-predicates.

    Upstream QANomSentence has this same wire schema. Nominal predicates use
    verbEntries; arbitrary Bank 2.1 nomEntries remain unmodelled in extra.
    """

    sentence_id: str
    sentence_tokens: tuple[str, ...]
    verb_entries: dict[str, VerbEntry] = field(default_factory=dict)
    non_predicates: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_sentence(self) -> Sentence:
        return Sentence(self.sentence_id, self.sentence_tokens, dict(self.verb_entries), dict(self.extra))

    @classmethod
    def from_sentence(cls, sentence: Sentence) -> "ConsolidatedSentence":
        extra = dict(sentence.extra)
        non_predicates = extra.pop("nonPredicates", {})
        return cls(sentence.sentence_id, sentence.sentence_tokens, dict(sentence.verb_entries), dict(non_predicates), extra)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ConsolidatedSentence":
        return cls.from_sentence(Sentence.from_json(data))

    def to_json(self) -> dict[str, Any]:
        return self.to_sentence().to_json() | {"nonPredicates": dict(self.non_predicates)}


@dataclass(frozen=True)
class ConsolidatedMergeResult:
    dataset: "ConsolidatedDataset"
    failures: tuple[DataMergeFailure, ...] = ()


@dataclass(frozen=True)
class ConsolidatedDataset:
    sentences: dict[str, ConsolidatedSentence] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, sentence in self.sentences.items():
            if key != sentence.sentence_id:
                raise ValueError(f"Sentence key {key!r} differs from ID {sentence.sentence_id!r}")
        object.__setattr__(self, "sentences", dict(sorted(self.sentences.items())))

    def to_dataset(self) -> Dataset:
        """Project onto verb annotations, as in upstream (drops nonPredicates)."""
        return Dataset({key: value.to_sentence() for key, value in self.sentences.items()})

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> "ConsolidatedDataset":
        return cls({key: ConsolidatedSentence.from_sentence(value) for key, value in dataset.sentences.items()})

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ConsolidatedDataset":
        return cls({key: ConsolidatedSentence.from_json(value) for key, value in data["sentences"].items()})

    def to_json(self) -> dict[str, Any]:
        return {"sentences": {key: value.to_json() for key, value in self.sentences.items()}}

    def merge(self, other: "ConsolidatedDataset") -> ConsolidatedMergeResult:
        result = self.to_dataset().merge(other.to_dataset())
        failures = list(result.failures)
        failed_sentences = {failure.sentence_id for failure in failures if isinstance(failure, SentenceMergeFailure)}
        sentences = {}
        for sid, value in result.dataset.sentences.items():
            left, right = self.sentences.get(sid), other.sentences.get(sid)
            if left is None or right is None:
                sentences[sid] = left if left is not None else right
            elif sid in failed_sentences:
                sentences[sid] = left
            else:
                try:
                    non_predicates = _merge_extra(left.non_predicates, right.non_predicates)
                except ValueError as error:
                    # Upstream concatenates conflicting strings; reject those
                    # conflicts and retain both original records diagnostically.
                    failures.append(SentenceMergeFailure(
                        sid, left.sentence_tokens, "Conflicting non-predicates: " + str(error),
                        Sentence.from_json(left.to_json()), Sentence.from_json(right.to_json()),
                    ))
                    sentences[sid] = left
                else:
                    sentences[sid] = replace(ConsolidatedSentence.from_sentence(value), non_predicates=non_predicates)
        return ConsolidatedMergeResult(ConsolidatedDataset(sentences), tuple(failures))

    def combine(self, other: "ConsolidatedDataset", process_merge_failures: Callable[[tuple[DataMergeFailure, ...]], None]) -> "ConsolidatedDataset":
        result = self.merge(other)
        process_merge_failures(result.failures)
        return result.dataset
