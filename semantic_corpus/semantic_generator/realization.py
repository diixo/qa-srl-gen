"""Turning a bound frame into a sentence, with spans and relations.

The output is not a string. It is a semantic record — the sentence plus, for
every argument, the exact character span it occupies and the relation it
bears to the predicate. That record is what later stages export as QA, NER or
SRL data, so the offsets have to be right by construction rather than
recovered by searching for substrings afterwards.

English morphology is not reimplemented here. The verb sequence comes from
:class:`~..qasrl_core.frame.Frame`, the same chain builder that reproduces
every verb form in QA-SRL Bank 2.0, so tense, aspect, voice and negation
behave identically whether a sentence is read from the bank or invented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..ontology import NAMEDNESS_LABELS, LabelSet, Relation
from ..qasrl_core.frame import ArgStructure, Frame as VerbFrame
from ..qasrl_core.models import InflectedForms
from .frames import SemanticFrame, SurfacePattern
from .substitutions import Entity

__all__ = [
    "Features",
    "Situation",
    "MentionSpan",
    "RelationEdge",
    "RealizedSituation",
    "realize",
    "BindingError",
]


class BindingError(ValueError):
    """Raised when bindings do not satisfy the frame's typed slots."""


@dataclass(frozen=True, slots=True)
class Features:
    """Grammatical features of the clause to be produced."""

    tense: str = "past"
    is_perfect: bool = False
    is_progressive: bool = False
    is_negated: bool = False

    def describe(self) -> str:
        parts = [self.tense]
        if self.is_perfect:
            parts.append("perfect")
        if self.is_progressive:
            parts.append("progressive")
        if self.is_negated:
            parts.append("negated")
        return "+".join(parts)


@dataclass(frozen=True, slots=True)
class Situation:
    """A frame with its slots bound to entities, before any wording."""

    frame: SemanticFrame
    bindings: Mapping[str, Entity]
    forms: InflectedForms
    features: Features = field(default_factory=Features)

    def validate(self) -> list[str]:
        """Type errors in the bindings, as messages."""
        problems: list[str] = []
        slots = self.frame.slot_map
        for name, entity in self.bindings.items():
            slot = slots.get(name)
            if slot is None:
                problems.append(f"{self.frame.lemma} has no slot {name!r}")
                continue
            if not slot.accepts(entity.labels.labels):
                allowed = ", ".join(sorted(l.value for l in slot.types))
                problems.append(
                    f"slot {name!r} accepts {allowed}, but {entity.text!r} is "
                    f"{entity.labels}"
                )
        for slot in self.frame.required_slots:
            if slot.name not in self.bindings:
                problems.append(f"required slot {slot.name!r} is unbound")
        return problems


@dataclass(frozen=True, slots=True)
class MentionSpan:
    """An entity mention located in the produced sentence."""

    slot: str
    text: str
    start_char: int
    end_char: int
    labels: LabelSet

    def check(self, sentence: str) -> bool:
        return sentence[self.start_char : self.end_char] == self.text


@dataclass(frozen=True, slots=True)
class RelationEdge:
    """A typed link from a mention to the predicate."""

    source_slot: str
    relation: Relation
    target: str


@dataclass(frozen=True, slots=True)
class RealizedSituation:
    """A sentence together with everything known to be true about it."""

    text: str
    situation: Situation
    pattern: SurfacePattern
    mentions: tuple[MentionSpan, ...]
    predicate: MentionSpan
    relations: tuple[RelationEdge, ...]
    omitted_slots: frozenset[str] = field(default_factory=frozenset)

    def mention(self, slot: str) -> MentionSpan | None:
        for span in self.mentions:
            if span.slot == slot:
                return span
        return None

    def check(self) -> list[str]:
        """Verify every recorded span really covers what it claims to."""
        problems = []
        for span in (*self.mentions, self.predicate):
            if not span.check(self.text):
                problems.append(
                    f"span [{span.start_char}, {span.end_char}) should be "
                    f"{span.text!r} but the sentence has "
                    f"{self.text[span.start_char:span.end_char]!r}"
                )
        return problems

    def __str__(self) -> str:
        return self.text


class _Builder:
    """Accumulates words while tracking where each one lands."""

    def __init__(self) -> None:
        self._words: list[str] = []
        self._length = 0

    def add(self, text: str) -> tuple[str, int, int]:
        """Append *text*, returning it as written plus the span it occupies.

        The first word of a sentence is capitalised here rather than
        afterwards, so that what a mention records is literally what the
        sentence contains.
        """
        if not self._words and text:
            text = text[0].upper() + text[1:]
        start = self._length + (1 if self._words else 0)
        self._words.append(text)
        self._length = start + len(text)
        return text, start, self._length

    def attach(self, suffix: str) -> None:
        """Append punctuation to the last word, with no space before it."""
        if not self._words:
            raise ValueError("nothing to attach punctuation to")
        self._words[-1] += suffix
        self._length += len(suffix)

    def strip_trailing(self, suffix: str) -> None:
        """Undo an :meth:`attach` that turned out to be sentence-final."""
        if self._words and self._words[-1].endswith(suffix):
            self._words[-1] = self._words[-1][: -len(suffix)]
            self._length -= len(suffix)

    @property
    def text(self) -> str:
        return " ".join(self._words)


def realize(
    situation: Situation, pattern: SurfacePattern | str | None = None
) -> RealizedSituation:
    """Produce the sentence for *situation* under *pattern*.

    Raises :class:`BindingError` if the bindings do not type-check against the
    frame, rather than emitting a sentence that the semantic record would
    describe incorrectly.
    """
    problems = situation.validate()
    if problems:
        raise BindingError(
            f"{situation.frame.lemma}: " + "; ".join(problems)
        )

    frame = situation.frame
    if pattern is None:
        chosen = frame.patterns[0]
    elif isinstance(pattern, str):
        chosen = frame.pattern(pattern)
    else:
        chosen = pattern

    missing = {
        c.slot
        for c in chosen.complements
        if c.slot not in situation.bindings and c.slot not in frame.adjunct_slots
    }
    if chosen.subject not in situation.bindings:
        missing.add(chosen.subject)
    if missing:
        raise BindingError(
            f"{frame.lemma}: pattern {chosen.name!r} needs unbound slots "
            f"{sorted(missing)}"
        )

    # ArgStructure carries the voice; the chain builder reads it from there.
    verb_frame = VerbFrame(
        verb_inflected_forms=situation.forms,
        structure=ArgStructure(args={}, is_passive=chosen.is_passive),
        tense=situation.features.tense,
        is_perfect=situation.features.is_perfect,
        is_progressive=situation.features.is_progressive,
        is_negated=situation.features.is_negated,
    )

    builder = _Builder()
    mentions: list[MentionSpan] = []
    relations: list[RelationEdge] = []
    slots = frame.slot_map

    def place(slot_name: str, preposition: str | None = None) -> None:
        entity = situation.bindings[slot_name]
        if preposition:
            builder.add(preposition)
        written, start, end = builder.add(entity.text)
        mentions.append(
            MentionSpan(slot_name, written, start, end, entity.labels)
        )
        relations.append(
            RelationEdge(slot_name, slots[slot_name].role, frame.lemma)
        )
        if entity.appositive:
            # ``Rex, a German shepherd,`` — the gloss is what makes the
            # reading recoverable from the sentence rather than from the
            # frame's type constraints, which a reader cannot see.
            builder.attach(",")
            gloss, gloss_start, gloss_end = builder.add(entity.appositive)
            builder.attach(",")
            # The gloss says what kind of thing the entity is, so it carries
            # the type but not the namedness: "the new mechanic" is not a name.
            mentions.append(
                MentionSpan(
                    f"{slot_name}:appositive",
                    gloss,
                    gloss_start,
                    gloss_end,
                    LabelSet(entity.labels.labels - NAMEDNESS_LABELS),
                )
            )
            relations.append(
                RelationEdge(slot_name, Relation.INSTANCE_OF, entity.appositive)
            )

    place(chosen.subject)

    # The predicate is the last word of the chain; the auxiliaries in front
    # of it are grammatical, not referential, so only the last gets a span.
    stack = verb_frame.get_verb_stack()
    head_text, head_start, head_end = "", 0, 0
    for word in stack:
        head_text, head_start, head_end = builder.add(word)

    adjunct_names = frame.adjunct_slots
    for complement in chosen.complements:
        if complement.slot in adjunct_names:
            continue
        place(complement.slot, complement.preposition)

    for name, default_preposition in frame.adjuncts:
        if name not in situation.bindings or name in chosen.omits:
            continue
        entity = situation.bindings[name]
        place(name, entity.adjunct_preposition or default_preposition)

    # A sentence-final appositive would otherwise leave ``shepherd,.``
    builder.strip_trailing(",")
    predicate = MentionSpan(
        slot="__predicate__",
        text=head_text,
        start_char=head_start,
        end_char=head_end,
        labels=LabelSet.of(frame.predicate_type),
    )

    realized = RealizedSituation(
        text=builder.text + ".",
        situation=situation,
        pattern=chosen,
        mentions=tuple(mentions),
        predicate=predicate,
        relations=tuple(relations),
        omitted_slots=frozenset(chosen.omits),
    )
    # These offsets are load-bearing downstream, so the invariant is checked
    # on every sentence rather than trusted.
    broken = realized.check()
    if broken:
        raise AssertionError(
            f"realisation produced inconsistent offsets: {'; '.join(broken)}"
        )
    return realized
