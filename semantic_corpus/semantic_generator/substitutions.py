"""Entities, name pools and seeded sampling.

Three handoff requirements shape this module.

**Fictional names alongside real ones.** ``Zelora`` and ``Narev`` exist so
that a model cannot answer from world knowledge. If every location in the
corpus were a real city, the test for "did it learn to read context" would be
answerable from memorisation alone.

**No name belongs to one class.** ``Rex`` is an animal in one sentence and a
mechanic in the next; ``Paris`` is a city or a person. Pools therefore attach
several readings to the same surface form, and :class:`EntityPool` can draw
deliberately ambiguous entities.

**Held-out splits.** ``unseen_names_test`` is only meaningful if its names
never appear in training. :meth:`EntityPool.split` partitions the pool by a
hash of the surface form, so the same name lands in the same split on every
run and on every machine, independent of pool order or insertion history.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable

from ..ontology import Label, LabelSet, TypeHierarchy

__all__ = [
    "Entity",
    "EntityPool",
    "Split",
    "default_pool",
    "SPLITS",
]

#: Split names used throughout. ``train`` is everything not held out.
SPLITS: tuple[str, ...] = ("train", "dev", "test")
Split = str


@dataclass(frozen=True, slots=True)
class Entity:
    """Something that can fill an argument slot.

    ``text`` is the surface form exactly as it must appear in the sentence —
    including any determiner — so that realisation never has to guess at
    articles and span offsets stay exact.
    """

    text: str
    labels: LabelSet
    type_name: str | None = None
    is_named: bool = False
    #: The preposition this entity takes when it appears as an adjunct
    #: (``at noon`` but ``on Monday``). Kept out of ``text`` so that the
    #: recorded span covers the entity itself and nothing else.
    adjunct_preposition: str | None = None
    #: A parenthetical gloss that makes the reading recoverable from the
    #: sentence itself: ``Rex, a German shepherd,``. Without one, a name whose
    #: type is settled only by a frame's type constraints is not actually
    #: disambiguated for a reader, and an ambiguity test built on it would be
    #: testing nothing.
    appositive: str | None = None

    def __post_init__(self) -> None:
        problems = self.labels.problems()
        if problems:
            raise ValueError(f"{self.text!r}: {'; '.join(problems)}")
        if self.is_named and Label.NAMED_ENTITY not in self.labels:
            raise ValueError(f"{self.text!r} is named but lacks NAMED_ENTITY")

    @property
    def entity_label(self) -> Label | None:
        found = self.labels.entity_labels
        return found[0] if found else None

    def matches(self, types: Iterable[Label]) -> bool:
        return bool(self.labels.labels & frozenset(types))

    def __str__(self) -> str:
        return self.text


def _named(
    text: str,
    label: Label,
    type_name: str | None = None,
    appositive: str | None = None,
) -> Entity:
    labels = frozenset({label, Label.NAMED_ENTITY})
    return Entity(
        text, LabelSet(labels), type_name, is_named=True, appositive=appositive
    )


def _common(
    text: str,
    label: Label,
    type_name: str | None = None,
    preposition: str | None = None,
) -> Entity:
    return Entity(
        text,
        LabelSet(frozenset({label})),
        type_name,
        is_named=False,
        adjunct_preposition=preposition,
    )


class EntityPool:
    """A collection of entities, drawn from deterministically."""

    def __init__(self, entities: Iterable[Entity] = ()) -> None:
        self._entities: list[Entity] = []
        self._seen: set[Entity] = set()
        for entity in entities:
            self.add(entity)

    def add(self, entity: Entity) -> None:
        """Add *entity*, ignoring an exact duplicate.

        Duplicates would silently weight sampling towards whichever name
        happened to be listed twice, which is the kind of bias that is very
        hard to notice in generated data.
        """
        if entity in self._seen:
            return
        self._seen.add(entity)
        self._entities.append(entity)

    def extend(self, entities: Iterable[Entity]) -> None:
        for entity in entities:
            self.add(entity)

    def __len__(self) -> int:
        return len(self._entities)

    def __iter__(self):
        return iter(self._entities)

    @property
    def entities(self) -> tuple[Entity, ...]:
        return tuple(self._entities)

    @property
    def surface_forms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(e.text for e in self._entities))

    # -- selection ---------------------------------------------------------

    def matching(
        self, types: Iterable[Label], exclude_labels: Iterable[Label] = ()
    ) -> tuple[Entity, ...]:
        """Every entity that can fill a slot accepting *types*.

        *exclude_labels* mirrors a frame slot's exclusions, so that filtering
        happens while drawing rather than after: sampling first and rejecting
        later would waste draws and, worse, make the stream of random numbers
        depend on how many rejects happened to come up.
        """
        wanted = frozenset(types)
        banned = frozenset(exclude_labels)
        return tuple(
            e
            for e in self._entities
            if e.matches(wanted) and not (e.labels.labels & banned)
        )

    def readings(self, text: str) -> tuple[Entity, ...]:
        """All entries sharing a surface form — the ambiguity of that name."""
        return tuple(e for e in self._entities if e.text == text)

    def ambiguous_forms(self) -> tuple[str, ...]:
        """Surface forms with readings under more than one entity label."""
        out = []
        for text in self.surface_forms:
            labels = {e.entity_label for e in self.readings(text)}
            if len(labels) > 1:
                out.append(text)
        return tuple(out)

    def sample(
        self,
        types: Iterable[Label],
        rng: random.Random,
        *,
        exclude: Iterable[str] = (),
        exclude_labels: Iterable[Label] = (),
    ) -> Entity:
        """Draw one entity that fits *types*.

        Raises :class:`LookupError` rather than silently relaxing the type
        constraint: a frame that cannot be filled is a bug in the frame or the
        pool, and papering over it would put ungrammatical or wrongly-typed
        sentences into the corpus.
        """
        banned = set(exclude)
        choices = [
            e for e in self.matching(types, exclude_labels) if e.text not in banned
        ]
        if not choices:
            wanted = ", ".join(sorted(label.value for label in frozenset(types)))
            raise LookupError(f"no entity in the pool matches {wanted}")
        return rng.choice(choices)

    # -- splitting ---------------------------------------------------------

    def split(
        self,
        name: Split,
        *,
        dev_share: float = 0.1,
        test_share: float = 0.1,
        salt: str = "",
    ) -> "EntityPool":
        """The sub-pool for *name*, partitioned by surface form.

        Partitioning by form rather than by entity keeps every reading of an
        ambiguous name together: if ``Rex`` is held out, both its animal and
        its person reading are held out, so a held-out name cannot leak back
        in through one of its senses.
        """
        if name not in SPLITS:
            raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
        if not 0 <= dev_share + test_share < 1:
            raise ValueError("dev_share + test_share must be in [0, 1)")
        chosen = [e for e in self._entities if _split_of(e.text, dev_share, test_share, salt) == name]
        return EntityPool(chosen)

    def split_of(
        self,
        text: str,
        *,
        dev_share: float = 0.1,
        test_share: float = 0.1,
        salt: str = "",
    ) -> Split:
        return _split_of(text, dev_share, test_share, salt)


def _split_of(text: str, dev_share: float, test_share: float, salt: str) -> Split:
    """Assign a surface form to a split by a stable hash.

    ``hash()`` is salted per process in Python, so it cannot be used here:
    the assignment has to be the same in every run for a held-out set to mean
    anything.
    """
    digest = hashlib.sha256(f"{salt}\x00{text}".encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    if position < test_share:
        return "test"
    if position < test_share + dev_share:
        return "dev"
    return "train"


# ---------------------------------------------------------------------------
# The default pool
# ---------------------------------------------------------------------------

_REAL_PLACES = ("Paris", "London", "Kyiv", "Lisbon", "Oslo", "Toronto", "Nairobi")
#: Invented place names: no world knowledge can answer a question about them.
_FICTIONAL_PLACES = ("Zelora", "Narev", "Taldin", "Orvash", "Kemdara", "Vustal")

_REAL_PEOPLE = ("Anna", "Helen", "Marcus", "Priya", "Tomas", "Leila", "Ivan")
_FICTIONAL_PEOPLE = ("Sorel", "Yavin", "Delkath", "Mira", "Tovan", "Esra")

_ORGANIZATIONS = ("Acme", "Belmont", "Orilex", "Kadrin Group", "Halverson")
_ANIMAL_NAMES = ("Rex", "Luna", "Bracken", "Pippin")
_NAMED_OBJECTS = ("Titanic", "Endeavour", "Kestrel")

#: Forms deliberately given more than one reading, so that type assignment
#: cannot be learned from the name alone.
_AMBIGUOUS: tuple[tuple[str, tuple[Label, str], tuple[Label, str]], ...] = (
    ("Rex", (Label.ANIMAL, "a German shepherd"), (Label.PERSON, "the new mechanic")),
    ("Paris", (Label.LOCATION, "the French capital"), (Label.PERSON, "Helen's sister")),
    ("Rose", (Label.PERSON, "the gardener"), (Label.PHYSICAL_OBJECT, "a wooden boat")),
    ("Jaguar", (Label.ANIMAL, "a big cat"), (Label.PHYSICAL_OBJECT, "an old car")),
    ("Austin", (Label.LOCATION, "a city in Texas"), (Label.PERSON, "the new driver")),
)

_COMMON_NOUNS: tuple[tuple[str, Label, str], ...] = (
    ("the dog", Label.ANIMAL, "dog"),
    ("the cat", Label.ANIMAL, "cat"),
    ("the horse", Label.ANIMAL, "horse"),
    ("the woman", Label.PERSON, "woman"),
    ("the mechanic", Label.PERSON, "profession"),
    ("the teacher", Label.PERSON, "profession"),
    ("a red ball", Label.PHYSICAL_OBJECT, "toy"),
    ("a book", Label.PHYSICAL_OBJECT, "document"),
    ("an old car", Label.PHYSICAL_OBJECT, "car"),
    ("the table", Label.PHYSICAL_OBJECT, "furniture"),
    ("the city", Label.LOCATION, "city"),
    ("the village", Label.LOCATION, "village"),
    ("the company", Label.ORGANIZATION, "company"),
    ("the university", Label.ORGANIZATION, "university"),
    ("a plan", Label.ABSTRACT_ENTITY, "notion"),
    ("an idea", Label.ABSTRACT_ENTITY, "notion"),
)

_EMOTIONS: tuple[tuple[str, str], ...] = (
    ("joy", "feeling"),
    ("fear", "feeling"),
    ("anger", "feeling"),
    ("sadness", "feeling"),
)

#: Adjunct fillers, stored bare with the preposition they take alongside.
_DATES: tuple[tuple[str, str], ...] = (
    ("Monday", "on"),
    ("March", "in"),
    ("the first of May", "on"),
)
_TIMES: tuple[tuple[str, str], ...] = (
    ("noon", "at"),
    ("dawn", "before"),
    ("midnight", "after"),
)


def default_pool(hierarchy: TypeHierarchy | None = None) -> EntityPool:
    """The entity pool shipped with the package.

    *hierarchy* is used only to look labels up for common nouns, so that the
    pool and the ontology cannot disagree about what ``the dog`` is.
    """
    pool = EntityPool()
    # Names with a curated set of readings are added only from _AMBIGUOUS, so
    # that a plain single-label entry cannot shadow them in sampling.
    ambiguous_forms = {form for form, _, _ in _AMBIGUOUS}

    for text in _REAL_PLACES + _FICTIONAL_PLACES:
        if text in ambiguous_forms:
            continue
        pool.add(_named(text, Label.LOCATION, "city"))
    for text in _REAL_PEOPLE + _FICTIONAL_PEOPLE:
        if text in ambiguous_forms:
            continue
        pool.add(_named(text, Label.PERSON, "human"))
    for text in _ORGANIZATIONS:
        pool.add(_named(text, Label.ORGANIZATION, "institution"))
    for text in _ANIMAL_NAMES:
        if text in ambiguous_forms:
            continue
        pool.add(_named(text, Label.ANIMAL, "creature"))
    for text in _NAMED_OBJECTS:
        pool.add(
            Entity(
                text,
                LabelSet.of(Label.PHYSICAL_OBJECT, Label.NAMED_OBJECT, Label.NAMED_ENTITY),
                "ship",
                is_named=True,
            )
        )

    for text, first, second in _AMBIGUOUS:
        for label, appositive in (first, second):
            # Both the bare name and the glossed one: the bare form creates
            # the ambiguity, the glossed form resolves it.
            pool.add(_named(text, label))
            pool.add(_named(text, label, appositive=appositive))

    for text, label, type_name in _COMMON_NOUNS:
        if hierarchy is not None and type_name in hierarchy:
            labels = hierarchy.labels_of(type_name)
            if label not in labels:
                raise ValueError(
                    f"{text!r} is declared {label} but the ontology says "
                    f"{type_name} is {labels}"
                )
        pool.add(_common(text, label, type_name))

    for text, type_name in _EMOTIONS:
        pool.add(
            Entity(text, LabelSet.of(Label.ABSTRACT_ENTITY, Label.EMOTION), type_name)
        )

    for text, preposition in _DATES:
        pool.add(_common(text, Label.DATE, "DATE", preposition))
    for text, preposition in _TIMES:
        pool.add(_common(text, Label.TIME, "TIME", preposition))

    return pool
