"""Typed frames: what a predicate means and which arguments it licenses.

A frame is the thing that stops the generator substituting any word for any
other. ``visit`` has a destination and no recipient; ``own`` has a possession
and no location; ``feel`` takes an emotion and nothing else in object
position. Each slot declares the ontology labels that may fill it, so a
binding that does not type-check is rejected before anything is realised.

Surface patterns are listed per frame rather than inferred, because the same
meaning has several spellings and they are not interchangeable: *Anna gave Rex
a ball* and *Anna gave a ball to Rex* differ in which argument is adjacent to
the verb, and the passive promotes a different one again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from ..ontology import Label, Relation

__all__ = [
    "FrameSlot",
    "Complement",
    "SurfacePattern",
    "SemanticFrame",
    "DEFAULT_FRAMES",
    "frame_by_lemma",
]


@dataclass(frozen=True, slots=True)
class FrameSlot:
    """One argument position of a frame."""

    name: str
    role: Relation
    types: frozenset[Label]
    optional: bool = False
    #: Labels that disqualify a filler even when ``types`` would admit it.
    #: ``give`` takes an abstract theme (*gave an idea*) but not an emotional
    #: one (*gave sadness*), and emotions are abstract entities.
    excludes: frozenset[Label] = frozenset()

    def accepts(self, labels: Iterable[Label]) -> bool:
        present = frozenset(labels)
        if present & self.excludes:
            return False
        return bool(self.types & present)

    def __str__(self) -> str:
        kinds = " | ".join(sorted(label.value for label in self.types))
        text = f"{self.name} = {kinds}"
        if self.excludes:
            text += " except " + " | ".join(sorted(l.value for l in self.excludes))
        return text + (", optional" if self.optional else "")


@dataclass(frozen=True, slots=True)
class Complement:
    """A post-verbal argument: which slot fills it and what introduces it."""

    slot: str
    preposition: str | None = None


@dataclass(frozen=True, slots=True)
class SurfacePattern:
    """One way of laying a frame's arguments out around the verb."""

    name: str
    subject: str
    complements: tuple[Complement, ...] = ()
    is_passive: bool = False
    #: Slots this pattern leaves unexpressed. The passive of ``give`` may drop
    #: its agent entirely, which is the only way to generate a sentence whose
    #: agent genuinely cannot be recovered.
    omits: frozenset[str] = field(default_factory=frozenset)

    def mentioned(self) -> frozenset[str]:
        return frozenset({self.subject} | {c.slot for c in self.complements})


@dataclass(frozen=True, slots=True)
class SemanticFrame:
    """A predicate with typed argument slots and its surface patterns."""

    lemma: str
    predicate_type: Label
    slots: tuple[FrameSlot, ...]
    patterns: tuple[SurfacePattern, ...]
    #: Adjunct slots are optional by construction and realise as trailing
    #: prepositional phrases, independent of the chosen pattern.
    adjuncts: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.predicate_type not in (Label.ACTION, Label.STATE):
            raise ValueError(
                f"{self.lemma}: predicate_type must be ACTION or STATE, "
                f"got {self.predicate_type}"
            )
        names = [slot.name for slot in self.slots]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.lemma}: duplicate slot names in {names}")
        known = set(names)
        for pattern in self.patterns:
            missing = pattern.mentioned() - known
            if missing:
                raise ValueError(
                    f"{self.lemma}: pattern {pattern.name!r} refers to unknown "
                    f"slots {sorted(missing)}"
                )
        for name, _ in self.adjuncts:
            if name not in known:
                raise ValueError(f"{self.lemma}: unknown adjunct slot {name!r}")

    @property
    def slot_map(self) -> Mapping[str, FrameSlot]:
        return {slot.name: slot for slot in self.slots}

    @property
    def required_slots(self) -> tuple[FrameSlot, ...]:
        return tuple(slot for slot in self.slots if not slot.optional)

    @property
    def adjunct_slots(self) -> frozenset[str]:
        return frozenset(name for name, _ in self.adjuncts)

    def pattern(self, name: str) -> SurfacePattern:
        for pattern in self.patterns:
            if pattern.name == name:
                return pattern
        raise KeyError(f"{self.lemma} has no pattern {name!r}")

    def __str__(self) -> str:
        lines = [f"{self.lemma}: predicate_type = {self.predicate_type}"]
        lines.extend(f"  {slot}" for slot in self.slots)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The starter frames named in the handoff
# ---------------------------------------------------------------------------

_ANIMATE = frozenset({Label.PERSON, Label.ANIMAL})
_AGENTIVE = frozenset({Label.PERSON, Label.ORGANIZATION})
_THING = frozenset({Label.PHYSICAL_OBJECT, Label.ABSTRACT_ENTITY})
_PLACE = frozenset({Label.LOCATION})
_WHEN = frozenset({Label.DATE, Label.TIME})
_NO_EMOTION = frozenset({Label.EMOTION})

_LOCATION_ADJUNCT = FrameSlot("location", Relation.LOCATION_OF, _PLACE, optional=True)
_TIME_ADJUNCT = FrameSlot("time", Relation.TIME_OF, _WHEN, optional=True)


GIVE = SemanticFrame(
    lemma="give",
    predicate_type=Label.ACTION,
    slots=(
        FrameSlot("agent", Relation.AGENT_OF, _AGENTIVE),
        FrameSlot(
            "recipient",
            Relation.RECIPIENT_OF,
            frozenset({Label.PERSON, Label.ANIMAL, Label.ORGANIZATION}),
        ),
        FrameSlot("theme", Relation.THEME_OF, _THING, excludes=_NO_EMOTION),
        _LOCATION_ADJUNCT,
        _TIME_ADJUNCT,
    ),
    patterns=(
        SurfacePattern(
            "ditransitive",
            subject="agent",
            complements=(Complement("recipient"), Complement("theme")),
        ),
        SurfacePattern(
            "prepositional",
            subject="agent",
            complements=(Complement("theme"), Complement("recipient", "to")),
        ),
        SurfacePattern(
            "passive",
            subject="theme",
            complements=(Complement("recipient", "to"), Complement("agent", "by")),
            is_passive=True,
        ),
        SurfacePattern(
            "passive_agentless",
            subject="theme",
            complements=(Complement("recipient", "to"),),
            is_passive=True,
            omits=frozenset({"agent"}),
        ),
    ),
    adjuncts=(("location", "in"), ("time", "on")),
)


VISIT = SemanticFrame(
    lemma="visit",
    predicate_type=Label.ACTION,
    slots=(
        FrameSlot("visitor", Relation.AGENT_OF, _ANIMATE),
        FrameSlot(
            "destination",
            Relation.LOCATION_OF,
            frozenset({Label.LOCATION, Label.ORGANIZATION}),
        ),
        _TIME_ADJUNCT,
    ),
    patterns=(
        SurfacePattern(
            "transitive", subject="visitor", complements=(Complement("destination"),)
        ),
        SurfacePattern(
            "passive",
            subject="destination",
            complements=(Complement("visitor", "by"),),
            is_passive=True,
        ),
    ),
    adjuncts=(("time", "on"),),
)


OWN = SemanticFrame(
    lemma="own",
    predicate_type=Label.STATE,
    slots=(
        FrameSlot("owner", Relation.EXPERIENCER_OF, _AGENTIVE),
        FrameSlot("possession", Relation.THEME_OF, _THING, excludes=_NO_EMOTION),
    ),
    patterns=(
        SurfacePattern(
            "transitive", subject="owner", complements=(Complement("possession"),)
        ),
        SurfacePattern(
            "passive",
            subject="possession",
            complements=(Complement("owner", "by"),),
            is_passive=True,
        ),
    ),
)


FEEL = SemanticFrame(
    lemma="feel",
    predicate_type=Label.STATE,
    slots=(
        FrameSlot("experiencer", Relation.EXPERIENCER_OF, _ANIMATE),
        FrameSlot("emotion", Relation.STATE_OF, frozenset({Label.EMOTION})),
        FrameSlot("cause", Relation.THEME_OF, _THING, optional=True),
    ),
    patterns=(
        SurfacePattern(
            "transitive",
            subject="experiencer",
            complements=(Complement("emotion"),),
        ),
        SurfacePattern(
            "caused",
            subject="experiencer",
            complements=(Complement("emotion"), Complement("cause", "about")),
        ),
    ),
)


MOVE = SemanticFrame(
    lemma="move",
    predicate_type=Label.ACTION,
    slots=(
        FrameSlot("agent", Relation.AGENT_OF, _AGENTIVE | _ANIMATE),
        FrameSlot(
            "theme", Relation.THEME_OF, _THING, optional=True, excludes=_NO_EMOTION
        ),
        FrameSlot("destination", Relation.LOCATION_OF, _PLACE, optional=True),
        _TIME_ADJUNCT,
    ),
    patterns=(
        SurfacePattern(
            "transitive",
            subject="agent",
            complements=(Complement("theme"), Complement("destination", "to")),
        ),
        # Not an omission: in *Austin moves to Paris* the mover is also what
        # moves, so the theme is expressed — as the subject. Declaring it
        # omitted licensed "What was moved? -> The text does not say", which
        # the sentence itself answers.
        SurfacePattern(
            "intransitive",
            subject="agent",
            complements=(Complement("destination", "to"),),
        ),
    ),
    adjuncts=(("time", "on"),),
)


SEE = SemanticFrame(
    lemma="see",
    predicate_type=Label.STATE,
    slots=(
        FrameSlot("experiencer", Relation.EXPERIENCER_OF, _ANIMATE),
        FrameSlot(
            "stimulus",
            Relation.THEME_OF,
            _THING | _ANIMATE | _PLACE,
            excludes=frozenset({Label.EMOTION, Label.ABSTRACT_ENTITY}),
        ),
        _LOCATION_ADJUNCT,
    ),
    patterns=(
        SurfacePattern(
            "transitive", subject="experiencer", complements=(Complement("stimulus"),)
        ),
        SurfacePattern(
            "passive",
            subject="stimulus",
            complements=(Complement("experiencer", "by"),),
            is_passive=True,
        ),
    ),
    adjuncts=(("location", "in"),),
)


SAY = SemanticFrame(
    lemma="say",
    predicate_type=Label.ACTION,
    slots=(
        FrameSlot("speaker", Relation.AGENT_OF, _AGENTIVE),
        FrameSlot(
            "message",
            Relation.THEME_OF,
            frozenset({Label.ABSTRACT_ENTITY}),
            excludes=_NO_EMOTION,
        ),
        FrameSlot("addressee", Relation.RECIPIENT_OF, _ANIMATE, optional=True),
        _TIME_ADJUNCT,
    ),
    patterns=(
        SurfacePattern(
            "transitive", subject="speaker", complements=(Complement("message"),)
        ),
        SurfacePattern(
            "addressed",
            subject="speaker",
            complements=(Complement("message"), Complement("addressee", "to")),
        ),
    ),
    adjuncts=(("time", "on"),),
)


DEFAULT_FRAMES: tuple[SemanticFrame, ...] = (GIVE, VISIT, OWN, FEEL, MOVE, SEE, SAY)


def frame_by_lemma(lemma: str, frames: Iterable[SemanticFrame] = DEFAULT_FRAMES):
    """Look a frame up by its lemma."""
    for frame in frames:
        if frame.lemma == lemma:
            return frame
    raise KeyError(f"no frame for lemma {lemma!r}")
