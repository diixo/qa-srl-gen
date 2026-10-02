"""The canonical internal representation: documents, spans and annotations.

This is the shape every later stage reads and writes, so two properties are
enforced rather than assumed.

**Offsets are global and exact.** Every span carries character offsets into
the *document's* text, not into a passage. A passage knows where it starts, so
a local offset can always be converted, but nothing downstream has to track
which window an offset belongs to. :meth:`TextSpan.check` verifies that a span
really covers the text it claims to, and the ingestion and alignment layers
call it on everything they produce — a span that does not match its source is
the single most damaging kind of corruption here, because it survives every
later stage silently.

**Annotations are append-only.** Annotating a document does not change it: a
new :class:`AnnotationRun` is created, carrying its own ontology version,
model, prompt version and seed. Two runs over the same document coexist, which
is what makes a verification pass independent and makes a disagreement
between runs visible instead of overwritten.

The handoff names this representation but assigns it to no module; it lives at
the top level because the annotator, the storage layer and the exporters all
need it, and putting it inside any one of them would invert that dependency.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Iterable, Iterator, Mapping, Sequence

from .ontology import (
    Label,
    LabelSet,
    MarkerForm,
    MarkerFunction,
    Mood,
    Polarity,
    Relation,
    ReviewStatus,
    SpeechAct,
    Stance,
)

__all__ = [
    "TextSpan",
    "Token",
    "Passage",
    "Utterance",
    "Document",
    "EntityMention",
    "Predicate",
    "Property",
    "RelationEdge",
    "DialogueAnnotation",
    "AnnotationRun",
    "SpanError",
    "sha256_of",
    "ONTOLOGY_VERSION",
]

#: Bumped whenever the label inventory changes meaning. Stored on every run so
#: that annotations made under different ontologies cannot be merged silently.
ONTOLOGY_VERSION = "v1"


class SpanError(ValueError):
    """Raised when a span does not match the text it points into."""


def sha256_of(text: str) -> str:
    """A stable content hash, used for deduplication and provenance."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True, slots=True)
class TextSpan:
    """A half-open character range ``[start_char, end_char)``.

    Token offsets are optional because they are only known once a tokenisation
    has been chosen, and the character offsets are the authoritative ones.
    """

    start_char: int
    end_char: int
    start_token: int | None = None
    end_token: int | None = None

    def __post_init__(self) -> None:
        if self.start_char < 0:
            raise ValueError(f"span start must be non-negative, got {self.start_char}")
        if self.end_char <= self.start_char:
            raise ValueError(
                f"span end must exceed start, got [{self.start_char}, {self.end_char})"
            )

    def __len__(self) -> int:
        return self.end_char - self.start_char

    def text_in(self, text: str) -> str:
        return text[self.start_char : self.end_char]

    def check(self, text: str, expected: str) -> None:
        """Raise :class:`SpanError` unless this span covers *expected*."""
        if self.end_char > len(text):
            raise SpanError(f"span end {self.end_char} exceeds text length {len(text)}")
        found = self.text_in(text)
        if found != expected:
            raise SpanError(
                f"span [{self.start_char}, {self.end_char}) covers {found!r}, "
                f"expected {expected!r}"
            )

    def shifted(self, offset: int) -> "TextSpan":
        """The same span measured from a different origin."""
        return TextSpan(
            self.start_char + offset,
            self.end_char + offset,
            self.start_token,
            self.end_token,
        )

    def overlaps(self, other: "TextSpan") -> bool:
        return self.start_char < other.end_char and other.start_char < self.end_char

    def contains(self, other: "TextSpan") -> bool:
        return (
            self.start_char <= other.start_char and other.end_char <= self.end_char
        )

    def to_json(self) -> dict[str, int]:
        out = {"start_char": self.start_char, "end_char": self.end_char}
        if self.start_token is not None:
            out["start_token"] = self.start_token
        if self.end_token is not None:
            out["end_token"] = self.end_token
        return out


@dataclass(frozen=True, slots=True)
class Token:
    """One token, located in the document by character offsets."""

    text: str
    start_char: int
    end_char: int
    index: int

    @property
    def span(self) -> TextSpan:
        return TextSpan(self.start_char, self.end_char, self.index, self.index + 1)


@dataclass(frozen=True, slots=True)
class Passage:
    """A window of a document, with its position preserved.

    Passages overlap in practice: the handoff asks for a few neighbouring
    sentences of context, so the same sentence appears in more than one
    window. ``sentence_spans`` keeps the individual sentences addressable.
    """

    passage_id: str
    document_id: str
    span: TextSpan
    sentence_spans: tuple[TextSpan, ...] = ()

    def text_in(self, document: "Document") -> str:
        return self.span.text_in(document.text)

    @property
    def start_char(self) -> int:
        return self.span.start_char


@dataclass(frozen=True, slots=True)
class Utterance:
    """One turn of a dialogue.

    Order and speaker are kept because a short reaction cannot be annotated
    without them: ``Wow`` means nothing until you know what it answers.
    """

    utterance_id: str
    document_id: str
    turn_index: int
    speaker: str
    span: TextSpan

    def text_in(self, document: "Document") -> str:
        return self.span.text_in(document.text)


@dataclass(frozen=True, slots=True)
class Document:
    """A source text, stored verbatim.

    The text is never normalised: offsets recorded against it must stay valid
    for the life of the corpus, and a later cleanup pass would invalidate
    every annotation made before it.
    """

    document_id: str
    text: str
    source: str = "unknown"
    license: str = "unknown"
    split: str | None = None
    passages: tuple[Passage, ...] = ()
    utterances: tuple[Utterance, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return sha256_of(self.text)

    @property
    def is_dialogue(self) -> bool:
        return bool(self.utterances)

    def slice(self, span: TextSpan) -> str:
        return span.text_in(self.text)

    def check_span(self, span: TextSpan, expected: str) -> None:
        span.check(self.text, expected)

    def with_passages(self, passages: Iterable[Passage]) -> "Document":
        return replace(self, passages=tuple(passages))

    def to_json(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "source": self.source,
            "license": self.license,
            "split": self.split,
            "sha256": self.sha256,
            **({"metadata": dict(self.metadata)} if self.metadata else {}),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "Document":
        known = {"document_id", "text", "source", "license", "split", "sha256", "metadata"}
        extra = {k: v for k, v in data.items() if k not in known}
        metadata = dict(data.get("metadata") or {})
        metadata.update(extra)
        return cls(
            document_id=str(data["document_id"]),
            text=str(data["text"]),
            source=str(data.get("source", "unknown")),
            license=str(data.get("license", "unknown")),
            split=data.get("split"),  # type: ignore[arg-type]
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EntityMention:
    """A labelled span of a document.

    Carries the provenance fields the handoff requires, because a corpus
    mixing generated and annotated material must be able to say which is
    which: ``source`` names the producer, ``confidence`` what it claimed, and
    ``review_status`` what checking has concluded.
    """

    mention_id: str
    document_id: str
    span: TextSpan
    exact_text: str
    labels: LabelSet
    head_token: int | None = None
    normalized_form: str | None = None
    confidence: float | None = None
    source: str = "unknown"
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    # Synthetic knowledge is not necessarily stated in the surface text.
    type_grounded: bool = True

    def __post_init__(self) -> None:
        problems = self.labels.problems()
        if problems:
            raise ValueError(f"{self.exact_text!r}: {'; '.join(problems)}")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def validate_against(self, document: Document) -> None:
        document.check_span(self.span, self.exact_text)

    def with_status(self, status: ReviewStatus) -> "EntityMention":
        return replace(self, review_status=status)


@dataclass(frozen=True, slots=True)
class Predicate:
    """A predicate mention with its grammatical features.

    The features are kept on the predicate rather than inferred later: a
    reader of the corpus should not have to re-parse the sentence to learn
    that the clause was passive and negated.
    """

    predicate_id: str
    document_id: str
    span: TextSpan
    exact_text: str
    lemma: str
    predicate_type: Label | None = Label.ACTION
    tense: str | None = None
    aspect: str | None = None
    voice: str | None = None
    polarity: Polarity | None = Polarity.POSITIVE
    modality: str | None = None
    confidence: float | None = None
    source: str = "unknown"
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    extra_labels: LabelSet = field(default_factory=LabelSet)
    omitted_slots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.predicate_type not in (None, Label.ACTION, Label.STATE):
            raise ValueError(
                f"predicate_type must be ACTION or STATE, got {self.predicate_type}"
            )
        if self.labels.problems():
            raise ValueError("; ".join(self.labels.problems()))

    @property
    def labels(self) -> LabelSet:
        return self.extra_labels | (() if self.predicate_type is None else (self.predicate_type,))

    def validate_against(self, document: Document) -> None:
        document.check_span(self.span, self.exact_text)


@dataclass(frozen=True, slots=True)
class Property:
    """A descriptive quality attributed to a mention.

    Stored decomposed, as the handoff requires: ``not very tall`` is a head
    (``tall``), a degree (``very``) and a negation, not an opaque string.
    Comparing or querying properties is impossible otherwise.
    """

    property_id: str
    document_id: str
    span: TextSpan
    exact_text: str
    head: str
    target_id: str | None = None
    degree: str | None = None
    negated: bool = False
    labels: LabelSet = field(default_factory=lambda: LabelSet.of(Label.PROPERTY))
    confidence: float | None = None
    source: str = "unknown"
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED

    def validate_against(self, document: Document) -> None:
        document.check_span(self.span, self.exact_text)


@dataclass(frozen=True, slots=True)
class RelationEdge:
    """A typed link between two annotated elements.

    ``question`` keeps the QA-SRL wording of the same link. The handoff is
    explicit that the natural-language question is the primary record and the
    normalised role is the derived one, so both are stored.
    """

    source_id: str
    relation: Relation | None
    target_id: str
    question: str | None = None
    confidence: float | None = None
    source: str = "unknown"


@dataclass(frozen=True, slots=True)
class DialogueAnnotation:
    """Utterance-level annotation: what the turn does, not what it mentions."""

    utterance_id: str
    speech_acts: tuple[SpeechAct, ...] = ()
    polarity: Polarity | None = None
    stance: Stance | None = None
    mood: Mood | None = None
    marker_form: MarkerForm | None = None
    marker_functions: tuple[MarkerFunction, ...] = ()
    confidence: float | None = None
    source: str = "unknown"
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED


@dataclass(frozen=True, slots=True)
class AnnotationRun:
    """One pass of annotation over one or more documents.

    A run is immutable and self-describing. Re-annotating produces another
    run; nothing is edited in place, so the history of what was claimed, by
    which model, under which prompt and ontology, stays intact.
    """

    run_id: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    ontology_version: str = ONTOLOGY_VERSION
    model_name: str = "unknown"
    prompt_version: str = "unknown"
    random_seed: int | None = None
    synthetic: bool = False
    mentions: tuple[EntityMention, ...] = ()
    predicates: tuple[Predicate, ...] = ()
    properties: tuple[Property, ...] = ()
    relations: tuple[RelationEdge, ...] = ()
    dialogue: tuple[DialogueAnnotation, ...] = ()

    def extended(
        self,
        *,
        mentions: Iterable[EntityMention] = (),
        predicates: Iterable[Predicate] = (),
        properties: Iterable[Property] = (),
        relations: Iterable[RelationEdge] = (),
        dialogue: Iterable[DialogueAnnotation] = (),
    ) -> "AnnotationRun":
        """A copy with more annotations; the original is untouched."""
        return replace(
            self,
            mentions=self.mentions + tuple(mentions),
            predicates=self.predicates + tuple(predicates),
            properties=self.properties + tuple(properties),
            relations=self.relations + tuple(relations),
            dialogue=self.dialogue + tuple(dialogue),
        )

    def __len__(self) -> int:
        return (
            len(self.mentions)
            + len(self.predicates)
            + len(self.properties)
            + len(self.relations)
            + len(self.dialogue)
        )

    def ids(self) -> set[str]:
        return (
            {m.mention_id for m in self.mentions}
            | {p.predicate_id for p in self.predicates}
            | {p.property_id for p in self.properties}
        )

    def validate_against(self, document: Document) -> list[str]:
        """Every way this run fails to describe *document*.

        Returns messages instead of raising so a whole run can be assessed at
        once; :mod:`.semantic_annotator.verifier` turns them into decisions.
        """
        problems: list[str] = []
        seen: set[str] = set()
        for item in (*self.mentions, *self.predicates, *self.properties):
            identifier = getattr(
                item, "mention_id", getattr(item, "predicate_id", None)
            ) or getattr(item, "property_id")
            if identifier in seen:
                problems.append(f"duplicate annotation id {identifier!r}")
            seen.add(identifier)
            if item.document_id != document.document_id:
                problems.append(f"{identifier}: belongs to document {item.document_id!r}, not {document.document_id!r}")
            try:
                item.validate_against(document)
            except SpanError as error:
                problems.append(f"{identifier}: {error}")

        known = self.ids()
        for prop in self.properties:
            if prop.target_id is not None and prop.target_id not in known:
                problems.append(f"property target {prop.target_id!r} is not annotated")
        for edge in self.relations:
            if edge.source_id not in known:
                problems.append(f"relation source {edge.source_id!r} is not annotated")
            if edge.target_id not in known:
                problems.append(f"relation target {edge.target_id!r} is not annotated")

        utterance_ids = {u.utterance_id for u in document.utterances}
        for annotation in self.dialogue:
            if annotation.utterance_id not in utterance_ids:
                problems.append(
                    f"dialogue annotation refers to unknown utterance "
                    f"{annotation.utterance_id!r}"
                )
        return problems


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

#: Deliberately simple: a boundary is terminal punctuation followed by space.
#: This is not a sentence splitter for arbitrary prose; the exceptions below
#: are the ones that actually break on real dialogue.
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc", "e.g", "i.e", "no", "fig"}
)
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*\s+")
#: A letter-dot run: ``U.``, ``p.``, ``O.K.``, ``B.A.`` — the shape an
#: initialism leaves behind when a naive splitter cuts it in half.
_INITIALISM = re.compile(r"(?:[A-Za-z]\.)+$")


def _continues_an_initialism(left_tail: str, right: str) -> bool:
    """Whether a period between *left_tail* and *right* is inside an initialism.

    A lone letter before a period is ambiguous: ``So do I.`` ends a sentence
    while ``the U.`` does not. What settles it is the other side. The period
    is internal when the next token continues the run (``U.`` + ``S.``) or
    starts lower-case (``7 p.`` + ``m.``); a capitalised ordinary word after
    it means the sentence really did end — ``Vitamin C.`` + ``It is good``.
    """
    if not _INITIALISM.search(left_tail):
        return False
    following = right.lstrip()
    if not following:
        return False
    if _INITIALISM.match(following.split()[0].rstrip(",;:")):
        return True
    return following[0].islower()


def iter_sentence_spans(text: str, offset: int = 0) -> Iterator[TextSpan]:
    """Yield sentence spans of *text*, shifted by *offset*.

    Spans exclude the trailing whitespace, so the text they cover is exactly
    the sentence.
    """
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.end()
        candidate = text[start:end].strip()
        if not candidate:
            start = end
            continue
        words = candidate.rstrip(".!?\"')]").split()
        if not words:
            # Punctuation alone — a leading "... " in "... Okay, I'm done." —
            # is not a sentence. Keep it with whatever follows.
            continue
        if words[-1].lower() in _ABBREVIATIONS:
            continue
        if _continues_an_initialism(candidate.split()[-1], text[end:]):
            continue
        stripped_end = start + len(text[start:end].rstrip())
        yield TextSpan(start + offset, stripped_end + offset)
        start = end
    tail = text[start:].strip()
    if tail:
        stripped_end = start + len(text[start:].rstrip())
        yield TextSpan(start + offset, stripped_end + offset)
