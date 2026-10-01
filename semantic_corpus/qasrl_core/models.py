"""Data models for the QA-SRL Bank.

Ported from the Scala project ``julianmichael/qasrl`` (MIT). The JSON field
names of QA-SRL Bank 2.0/2.1 are preserved exactly on (de)serialisation; the
Python attribute names are snake_case. See THIRD_PARTY_NOTICES.md.

Bank format reference: https://github.com/uwnlp/qasrl-bank/blob/master/FORMAT.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Mapping, Sequence

__all__ = [
    "VerbForm",
    "InflectedForms",
    "Span",
    "AnswerJudgment",
    "QuestionSlots",
    "QuestionLabel",
    "VerbEntry",
    "Sentence",
    "SLOT_ORDER",
]


class VerbForm(str, Enum):
    """The five English verb forms QA-SRL inflects a predicate into.

    The values are the JSON keys used in ``verbInflectedForms`` and, at the
    same time, the placeholder tokens that appear inside the ``verb`` slot of
    a question (e.g. ``"being pastParticiple"``).
    """

    STEM = "stem"
    PRESENT_SINGULAR_3RD = "presentSingular3rd"
    PRESENT_PARTICIPLE = "presentParticiple"
    PAST = "past"
    PAST_PARTICIPLE = "pastParticiple"


#: Order in which slots are concatenated to form the surface question.
SLOT_ORDER: tuple[str, ...] = ("wh", "aux", "subj", "verb", "obj", "prep", "obj2")


@dataclass(frozen=True, slots=True)
class InflectedForms:
    """A verb paradigm: the five forms QA-SRL needs to realise a question."""

    stem: str
    present_singular_3rd: str
    present_participle: str
    past: str
    past_participle: str

    def __getitem__(self, form: VerbForm | str) -> str:
        return self.get(form)

    def get(self, form: VerbForm | str) -> str:
        """Return the surface string for *form* (a :class:`VerbForm` or its JSON key)."""
        key = form.value if isinstance(form, VerbForm) else form
        try:
            return getattr(self, _FORM_ATTR[key])
        except KeyError:
            raise KeyError(f"unknown verb form: {key!r}") from None

    def forms_for(self, surface: str) -> tuple[VerbForm, ...]:
        """Return every form whose surface string equals *surface*.

        A paradigm such as ``put/puts/putting/put/put`` maps one surface string
        onto three different forms, so this deliberately returns a tuple.
        """
        return tuple(f for f in VerbForm if self.get(f) == surface)

    @property
    def all_forms(self) -> tuple[str, ...]:
        return (
            self.stem,
            self.present_singular_3rd,
            self.present_participle,
            self.past,
            self.past_participle,
        )

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "InflectedForms":
        return cls(
            stem=data["stem"],
            present_singular_3rd=data["presentSingular3rd"],
            present_participle=data["presentParticiple"],
            past=data["past"],
            past_participle=data["pastParticiple"],
        )

    def to_json(self) -> dict[str, str]:
        return {
            "stem": self.stem,
            "presentSingular3rd": self.present_singular_3rd,
            "presentParticiple": self.present_participle,
            "past": self.past,
            "pastParticiple": self.past_participle,
        }


_FORM_ATTR: dict[str, str] = {
    VerbForm.STEM.value: "stem",
    VerbForm.PRESENT_SINGULAR_3RD.value: "present_singular_3rd",
    VerbForm.PRESENT_PARTICIPLE.value: "present_participle",
    VerbForm.PAST.value: "past",
    VerbForm.PAST_PARTICIPLE.value: "past_participle",
}


@dataclass(frozen=True, order=True, slots=True)
class Span:
    """A half-open token span ``[start, end)`` over ``Sentence.sentence_tokens``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"span start must be non-negative, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"span end must exceed start, got [{self.start}, {self.end})")

    def __len__(self) -> int:
        return self.end - self.start

    def tokens(self, sentence_tokens: Sequence[str]) -> list[str]:
        return list(sentence_tokens[self.start : self.end])

    def text(self, sentence_tokens: Sequence[str]) -> str:
        return " ".join(self.tokens(sentence_tokens))

    def fits(self, sentence_tokens: Sequence[str]) -> bool:
        return self.end <= len(sentence_tokens)

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end

    @classmethod
    def from_json(cls, data: Sequence[int]) -> "Span":
        start, end = data
        return cls(int(start), int(end))

    def to_json(self) -> list[int]:
        return [self.start, self.end]


@dataclass(frozen=True, slots=True)
class AnswerJudgment:
    """One annotator's verdict on a question, with the spans they highlighted.

    ``spans`` is empty whenever ``is_valid`` is false: an annotator who rejects
    the question does not answer it.
    """

    source_id: str
    is_valid: bool
    spans: tuple[Span, ...] = ()

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "AnswerJudgment":
        return cls(
            source_id=data["sourceId"],
            is_valid=bool(data["isValid"]),
            spans=tuple(Span.from_json(s) for s in (data.get("spans") or ())),
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"sourceId": self.source_id, "isValid": self.is_valid}
        if self.is_valid:
            out["spans"] = [s.to_json() for s in self.spans]
        return out


@dataclass(frozen=True, slots=True)
class QuestionSlots:
    """The seven-slot representation of a QA-SRL question.

    An unfilled slot holds ``"_"``. ``prep`` additionally distinguishes the
    empty string ``""`` — a *filled but silent* preposition that marks a bare
    complement such as ``help something do`` — from ``"_"``, which means the
    question has no prepositional position at all.

    ``verb`` is not a surface string: it holds a :class:`VerbForm` key,
    optionally preceded by literal auxiliary words (``not``, ``be``, ``been``,
    ``being``, ``have``), e.g. ``"have been pastParticiple"``.
    """

    wh: str
    aux: str
    subj: str
    verb: str
    obj: str
    prep: str
    obj2: str

    EMPTY = "_"

    def __iter__(self) -> Iterator[tuple[str, str]]:
        for name in SLOT_ORDER:
            yield name, getattr(self, name)

    def is_filled(self, name: str) -> bool:
        return getattr(self, name) != self.EMPTY

    @property
    def verb_form(self) -> VerbForm:
        """The :class:`VerbForm` carried by the ``verb`` slot (its last word)."""
        return VerbForm(self.verb.split(" ")[-1])

    @property
    def verb_prefix(self) -> tuple[str, ...]:
        """Literal auxiliary words preceding the inflected form in ``verb``."""
        return tuple(self.verb.split(" ")[:-1])

    @classmethod
    def from_json(cls, data: Mapping[str, str]) -> "QuestionSlots":
        return cls(**{name: data[name] for name in SLOT_ORDER})

    def to_json(self) -> dict[str, str]:
        return {name: value for name, value in self}


@dataclass(frozen=True, slots=True)
class QuestionLabel:
    """A single QA-SRL question together with every judgment collected on it."""

    question_string: str
    question_slots: QuestionSlots
    tense: str
    is_perfect: bool
    is_progressive: bool
    is_negated: bool
    is_passive: bool
    answer_judgments: tuple[AnswerJudgment, ...] = ()
    question_sources: tuple[str, ...] = ()

    @property
    def valid_judgments(self) -> tuple[AnswerJudgment, ...]:
        return tuple(j for j in self.answer_judgments if j.is_valid)

    @property
    def valid_ratio(self) -> float:
        """Share of annotators who accepted the question; ``0.0`` if unjudged."""
        if not self.answer_judgments:
            return 0.0
        return len(self.valid_judgments) / len(self.answer_judgments)

    def span_votes(self) -> dict[Span, int]:
        """How many accepting annotators highlighted each span."""
        votes: dict[Span, int] = {}
        for judgment in self.valid_judgments:
            for span in judgment.spans:
                votes[span] = votes.get(span, 0) + 1
        return votes

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "QuestionLabel":
        return cls(
            question_string=data["questionString"],
            question_slots=QuestionSlots.from_json(data["questionSlots"]),
            tense=data["tense"],
            is_perfect=bool(data["isPerfect"]),
            is_progressive=bool(data["isProgressive"]),
            is_negated=bool(data["isNegated"]),
            is_passive=bool(data["isPassive"]),
            answer_judgments=tuple(
                AnswerJudgment.from_json(j) for j in data.get("answerJudgments", ())
            ),
            question_sources=tuple(data.get("questionSources", ())),
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "questionString": self.question_string,
            "questionSlots": self.question_slots.to_json(),
            "tense": self.tense,
            "isPerfect": self.is_perfect,
            "isProgressive": self.is_progressive,
            "isNegated": self.is_negated,
            "isPassive": self.is_passive,
            "answerJudgments": [j.to_json() for j in self.answer_judgments],
        }
        if self.question_sources:
            out["questionSources"] = list(self.question_sources)
        return out


@dataclass(frozen=True, slots=True)
class VerbEntry:
    """All questions asked about one predicate token of a sentence."""

    verb_index: int
    verb_inflected_forms: InflectedForms
    question_labels: dict[str, QuestionLabel] = field(default_factory=dict)

    def __iter__(self) -> Iterator[QuestionLabel]:
        return iter(self.question_labels.values())

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "VerbEntry":
        return cls(
            verb_index=int(data["verbIndex"]),
            verb_inflected_forms=InflectedForms.from_json(data["verbInflectedForms"]),
            question_labels={
                key: QuestionLabel.from_json(value)
                for key, value in data.get("questionLabels", {}).items()
            },
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "verbIndex": self.verb_index,
            "verbInflectedForms": self.verb_inflected_forms.to_json(),
            "questionLabels": {k: v.to_json() for k, v in self.question_labels.items()},
        }


@dataclass(frozen=True, slots=True)
class Sentence:
    """One tokenised sentence with every annotated predicate.

    ``extra`` keeps any field this reader does not model (QA-SRL Bank 2.1 adds
    entries such as ``nomEntries``), so a parsed sentence can be written back
    without silently dropping data.
    """

    sentence_id: str
    sentence_tokens: tuple[str, ...]
    verb_entries: dict[str, VerbEntry] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(self.sentence_tokens)

    def __iter__(self) -> Iterator[VerbEntry]:
        return iter(self.verb_entries.values())

    def question_labels(self) -> Iterator[tuple[VerbEntry, QuestionLabel]]:
        for entry in self.verb_entries.values():
            for label in entry.question_labels.values():
                yield entry, label

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Sentence":
        known = {"sentenceId", "sentenceTokens", "verbEntries"}
        return cls(
            sentence_id=data["sentenceId"],
            sentence_tokens=tuple(data["sentenceTokens"]),
            verb_entries={
                key: VerbEntry.from_json(value)
                for key, value in data.get("verbEntries", {}).items()
            },
            extra={k: v for k, v in data.items() if k not in known},
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "sentenceId": self.sentence_id,
            "sentenceTokens": list(self.sentence_tokens),
            "verbEntries": {k: v.to_json() for k, v in self.verb_entries.items()},
        }
        out.update(self.extra)
        return out
