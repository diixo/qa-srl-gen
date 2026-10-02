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
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..ontology import Label, LabelSet, TypeHierarchy

__all__ = [
    "Entity",
    "EntityPool",
    "Split",
    "default_pool",
    "load_pool",
    "save_pool",
    "entity_from_json",
    "entity_to_json",
    "ENTITIES_PATH",
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
    #: The descriptive head inside this entity's own text, decomposed.
    #: ``a red ball`` carries ``red``; storing it apart from the surface form
    #: is what lets a question ask about the degree separately, which an
    #: opaque string could not support.
    property_head: str | None = None
    property_degree: str | None = None
    property_negated: bool = False

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
    property_head: str | None = None,
    property_degree: str | None = None,
) -> Entity:
    return Entity(
        text,
        LabelSet(frozenset({label})),
        type_name,
        is_named=False,
        adjunct_preposition=preposition,
        property_head=property_head,
        property_degree=property_degree,
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

    def split_key(self, text: str) -> str:
        """The form whose split *text* follows.

        A held-out name must not reappear inside a longer one: holding out
        ``America`` while training on ``North America`` puts the string in
        the training text anyway, and the unseen-name test silently measures
        less than it claims. So a form that contains a shorter pool form as
        a whole word inherits that form's split, and the shortest member of
        such a group decides for all of them.
        """
        words = text.split()
        best = text
        for other in self.surface_forms:
            if other == text or len(other) >= len(best):
                continue
            other_words = other.split()
            span = len(other_words)
            if any(
                words[i : i + span] == other_words
                for i in range(len(words) - span + 1)
            ):
                best = other
        return best

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
        chosen = [
            e
            for e in self._entities
            if self.split_of(e.text, dev_share=dev_share, test_share=test_share, salt=salt)
            == name
        ]
        return EntityPool(chosen)

    def split_of(
        self,
        text: str,
        *,
        dev_share: float = 0.1,
        test_share: float = 0.1,
        salt: str = "",
    ) -> Split:
        return _split_of(self.split_key(text), dev_share, test_share, salt)


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

#: The pool lives in a data file rather than in code. It is the part of the
#: project most likely to grow by an order of magnitude, and growing it
#: should not mean editing Python.
ENTITIES_PATH = Path(__file__).with_name("entities.json")


def entity_from_json(row: Mapping[str, object]) -> Entity:
    """Build one entity from a pool row."""
    labels = LabelSet(frozenset(Label(name) for name in row.get("labels", ())))
    return Entity(
        text=str(row["text"]),
        labels=labels,
        type_name=row.get("type"),  # type: ignore[arg-type]
        is_named=bool(row.get("named", False)),
        adjunct_preposition=row.get("preposition"),  # type: ignore[arg-type]
        appositive=row.get("appositive"),  # type: ignore[arg-type]
        property_head=row.get("property_head"),  # type: ignore[arg-type]
        property_degree=row.get("property_degree"),  # type: ignore[arg-type]
    )


def entity_to_json(entity: Entity) -> dict[str, object]:
    """Serialise an entity back into a pool row."""
    row: dict[str, object] = {
        "text": entity.text,
        "labels": [str(label) for label in entity.labels.ordered],
    }
    for key, value in (
        ("type", entity.type_name),
        ("named", entity.is_named or None),
        ("preposition", entity.adjunct_preposition),
        ("appositive", entity.appositive),
        ("property_head", entity.property_head),
        ("property_degree", entity.property_degree),
    ):
        if value:
            row[key] = value
    return row


def load_pool(
    path: Path | str = ENTITIES_PATH, hierarchy: TypeHierarchy | None = None
) -> EntityPool:
    """Read an entity pool from JSON.

    *hierarchy* is checked against, not consulted: a row claiming a label
    the ontology does not give its type is a contradiction between two
    files that would otherwise sit there unnoticed.
    """
    with Path(path).open("rt", encoding="utf-8") as stream:
        data = json.load(stream)
    pool = EntityPool()
    for index, row in enumerate(data.get("entities", ())):
        try:
            entity = entity_from_json(row)
        except (KeyError, ValueError) as error:
            raise ValueError(f"{path}: row {index}: {error}") from error
        if hierarchy is not None and entity.type_name and entity.type_name in hierarchy:
            inherited = hierarchy.labels_of(entity.type_name)
            label = entity.entity_label
            if label is not None and inherited.entity_labels and label not in inherited.labels:
                raise ValueError(
                    f"{path}: row {index}: {entity.text!r} is declared {label} "
                    f"but the ontology says {entity.type_name} is {inherited}"
                )
        pool.add(entity)
    return pool


def save_pool(pool: EntityPool, path: Path | str, *, comment: Sequence[str] = ()) -> None:
    """Write a pool back out, preserving the file's shape."""
    payload = {
        "_comment": list(comment),
        "entities": [entity_to_json(entity) for entity in pool],
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def default_pool(hierarchy: TypeHierarchy | None = None) -> EntityPool:
    """The entity pool shipped with the package."""
    return load_pool(ENTITIES_PATH, hierarchy)
