"""Ontology v1: the label inventory and what a labelled span looks like.

The decisions encoded here come from the handoff and are not up for
re-litigation: ``ACTION`` and ``STATE`` are separate; ``PROPERTY`` exists;
``EMOTION`` is a crosscutting feature that combines with ``STATE``,
``PROPERTY`` and ``ABSTRACT_ENTITY``; ``NAMED_ENTITY`` is an extra label
rather than a type of its own.

The central consequence is that **labels are not mutually exclusive**. A span
carries a *set* of labels, so ``afraid`` can be ``STATE + PROPERTY + EMOTION``
at once. Nothing here stores a flat BIO tag; BIO is a lossy export produced
later, never the working representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

__all__ = [
    "Label",
    "PREDICATE_LABELS",
    "ENTITY_LABELS",
    "CHARACTERISTIC_LABELS",
    "NAMEDNESS_LABELS",
    "MarkerForm",
    "MarkerFunction",
    "SpeechAct",
    "Relation",
    "Polarity",
    "LabelSet",
]


class Label(str, Enum):
    """Span labels. Several may apply to one span."""

    # Predicates
    ACTION = "ACTION"
    STATE = "STATE"

    # Entities
    PERSON = "PERSON"
    ANIMAL = "ANIMAL"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    PHYSICAL_OBJECT = "PHYSICAL_OBJECT"
    ABSTRACT_ENTITY = "ABSTRACT_ENTITY"
    DATE = "DATE"
    TIME = "TIME"

    # Characteristics
    PROPERTY = "PROPERTY"
    EMOTION = "EMOTION"

    # Namedness — attributes, not types
    NAMED_ENTITY = "NAMED_ENTITY"
    NAMED_OBJECT = "NAMED_OBJECT"

    # Discourse
    DISCOURSE_MARKER = "DISCOURSE_MARKER"

    def __str__(self) -> str:
        return self.value


PREDICATE_LABELS: frozenset[Label] = frozenset({Label.ACTION, Label.STATE})

ENTITY_LABELS: frozenset[Label] = frozenset(
    {
        Label.PERSON,
        Label.ANIMAL,
        Label.ORGANIZATION,
        Label.LOCATION,
        Label.PHYSICAL_OBJECT,
        Label.ABSTRACT_ENTITY,
        Label.DATE,
        Label.TIME,
    }
)

CHARACTERISTIC_LABELS: frozenset[Label] = frozenset({Label.PROPERTY, Label.EMOTION})

NAMEDNESS_LABELS: frozenset[Label] = frozenset({Label.NAMED_ENTITY, Label.NAMED_OBJECT})


class MarkerForm(str, Enum):
    """The form a discourse marker takes."""

    INTERJECTION = "INTERJECTION"
    FILLED_PAUSE = "FILLED_PAUSE"
    BACKCHANNEL = "BACKCHANNEL"
    RESPONSE_PARTICLE = "RESPONSE_PARTICLE"
    EVALUATIVE_RESPONSE = "EVALUATIVE_RESPONSE"

    def __str__(self) -> str:
        return self.value


class MarkerFunction(str, Enum):
    """What a discourse marker is doing. Context decides, never the word."""

    REACTION = "REACTION"
    SURPRISE = "SURPRISE"
    HESITATION = "HESITATION"
    THINKING = "THINKING"
    UNCERTAINTY = "UNCERTAINTY"
    ACKNOWLEDGEMENT = "ACKNOWLEDGEMENT"
    AGREEMENT = "AGREEMENT"
    ACCEPTANCE = "ACCEPTANCE"
    CONFIRMATION = "CONFIRMATION"
    DISAGREEMENT = "DISAGREEMENT"
    REJECTION = "REJECTION"
    APPROVAL = "APPROVAL"
    DISAPPROVAL = "DISAPPROVAL"

    def __str__(self) -> str:
        return self.value


class SpeechAct(str, Enum):
    """What an utterance does. One utterance may perform several."""

    INFORM = "INFORM"
    QUESTION = "QUESTION"
    ANSWER = "ANSWER"
    REQUEST = "REQUEST"
    COMMAND = "COMMAND"
    AGREEMENT = "AGREEMENT"
    DISAGREEMENT = "DISAGREEMENT"
    CONFIRMATION = "CONFIRMATION"
    REJECTION = "REJECTION"
    ACCEPTANCE = "ACCEPTANCE"
    ACKNOWLEDGEMENT = "ACKNOWLEDGEMENT"
    APPROVAL = "APPROVAL"
    DISAPPROVAL = "DISAPPROVAL"
    REACTION = "REACTION"
    HESITATION = "HESITATION"
    UNCERTAINTY = "UNCERTAINTY"
    BACKCHANNEL = "BACKCHANNEL"
    GREETING = "GREETING"
    FAREWELL = "FAREWELL"
    THANKING = "THANKING"
    APOLOGY = "APOLOGY"
    REASSURANCE = "REASSURANCE"

    def __str__(self) -> str:
        return self.value


class Relation(str, Enum):
    """Edges between spans.

    A QA-SRL question is kept alongside these as a natural-language
    description of the same link: the normalised role is a convenience, not a
    replacement for the question.
    """

    AGENT_OF = "AGENT_OF"
    PATIENT_OF = "PATIENT_OF"
    THEME_OF = "THEME_OF"
    RECIPIENT_OF = "RECIPIENT_OF"
    EXPERIENCER_OF = "EXPERIENCER_OF"
    LOCATION_OF = "LOCATION_OF"
    TIME_OF = "TIME_OF"
    PROPERTY_OF = "PROPERTY_OF"
    STATE_OF = "STATE_OF"
    MEMBER_OF = "MEMBER_OF"
    PART_OF = "PART_OF"
    INSTANCE_OF = "INSTANCE_OF"
    SUBTYPE_OF = "SUBTYPE_OF"

    def __str__(self) -> str:
        return self.value


class Polarity(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class LabelSet:
    """An unordered, deduplicated set of labels with a stable printed order.

    Stable ordering matters because these end up in exported corpora, where a
    set's iteration order would make runs irreproducible.
    """

    labels: frozenset[Label] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not all(isinstance(label, Label) for label in self.labels):
            raise TypeError("LabelSet accepts Label members only")

    @classmethod
    def of(cls, *labels: Label | str) -> "LabelSet":
        return cls(frozenset(Label(label) for label in labels))

    def __contains__(self, label: Label | str) -> bool:
        return Label(label) in self.labels

    def __iter__(self):
        return iter(self.ordered)

    def __len__(self) -> int:
        return len(self.labels)

    def __or__(self, other: "LabelSet | Iterable[Label]") -> "LabelSet":
        extra = other.labels if isinstance(other, LabelSet) else frozenset(other)
        return LabelSet(self.labels | extra)

    @property
    def ordered(self) -> tuple[Label, ...]:
        """Labels in declaration order, so output is reproducible."""
        return tuple(label for label in Label if label in self.labels)

    @property
    def is_predicate(self) -> bool:
        return bool(self.labels & PREDICATE_LABELS)

    @property
    def entity_labels(self) -> tuple[Label, ...]:
        return tuple(label for label in self.ordered if label in ENTITY_LABELS)

    def problems(self) -> list[str]:
        """Combinations the ontology forbids.

        Deliberately short. Most overlaps are legal by design — the one thing
        a span cannot be is both a dynamic event and a static situation.
        """
        issues: list[str] = []
        if Label.ACTION in self.labels and Label.STATE in self.labels:
            issues.append("ACTION and STATE are mutually exclusive")
        if Label.NAMED_OBJECT in self.labels and Label.NAMED_ENTITY not in self.labels:
            issues.append("NAMED_OBJECT implies NAMED_ENTITY")
        if len(self.entity_labels) > 1:
            issues.append(
                "a span has at most one entity type, got "
                + ", ".join(str(label) for label in self.entity_labels)
            )
        return issues

    def __str__(self) -> str:
        return " + ".join(str(label) for label in self.ordered)
