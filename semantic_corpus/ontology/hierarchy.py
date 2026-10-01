"""A type hierarchy over the ontology, with inheritance.

The hierarchy answers the question the handoff poses directly::

    Paris is a city.
    Every city is a location.
    Therefore Paris is a location.

Two layers, kept apart on purpose:

*types*
    ``city``, ``dog``, ``woman`` and so on, each a subtype of another type and
    ultimately of one :class:`~.models.Label`. This layer is closed and
    curated; inheritance through it is sound.

*lexicon*
    surface words mapped to *candidate* types. This layer only proposes. A
    word being in it is never enough to assign a label — ``jaguar`` is an
    animal or a car depending on the sentence, and the handoff is explicit
    that context decides. :meth:`TypeHierarchy.candidates` therefore returns
    every reading, and sorting them out is the annotator's job, not the
    dictionary's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from .models import Label, LabelSet

__all__ = ["TypeNode", "TypeHierarchy", "load_default_hierarchy", "DEFAULTS_PATH"]

DEFAULTS_PATH = Path(__file__).with_name("defaults.json")


@dataclass(frozen=True, slots=True)
class TypeNode:
    """One type: its parent, and any labels it contributes beyond the parent's.

    ``extra_labels`` is what lets ``joy`` be ``ABSTRACT_ENTITY + EMOTION``
    while still inheriting cleanly from ``feeling``.
    """

    name: str
    parent: str | None = None
    extra_labels: frozenset[Label] = field(default_factory=frozenset)


class TypeHierarchy:
    """A tree of types rooted in the ontology's entity labels."""

    def __init__(self) -> None:
        self._nodes: dict[str, TypeNode] = {}
        self._lexicon: dict[str, tuple[str, ...]] = {}
        for label in Label:
            self._nodes[label.value] = TypeNode(label.value, None, frozenset({label}))

    # -- construction ------------------------------------------------------

    def add_type(
        self,
        name: str,
        parent: str,
        extra_labels: Iterable[Label | str] = (),
    ) -> None:
        """Register *name* as a subtype of *parent*."""
        if parent not in self._nodes:
            raise KeyError(f"unknown parent type {parent!r} for {name!r}")
        if name in self._nodes and self._nodes[name].parent is None:
            raise ValueError(f"{name!r} is an ontology label and cannot be a subtype")
        node = TypeNode(name, parent, frozenset(Label(x) for x in extra_labels))
        self._nodes[name] = node
        if self._would_cycle(name):
            del self._nodes[name]
            raise ValueError(f"adding {name!r} under {parent!r} would create a cycle")

    def add_word(self, word: str, types: Iterable[str]) -> None:
        """Record that *word* may denote any of *types*."""
        names = tuple(dict.fromkeys(types))
        for name in names:
            if name not in self._nodes:
                raise KeyError(f"unknown type {name!r} for word {word!r}")
        existing = self._lexicon.get(word, ())
        self._lexicon[word] = tuple(dict.fromkeys(existing + names))

    def _would_cycle(self, name: str) -> bool:
        seen: set[str] = set()
        current: str | None = name
        while current is not None:
            if current in seen:
                return True
            seen.add(current)
            current = self._nodes[current].parent
        return False

    # -- queries -----------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._nodes

    @property
    def types(self) -> tuple[str, ...]:
        return tuple(sorted(self._nodes))

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(sorted(self._lexicon))

    def node(self, name: str) -> TypeNode:
        try:
            return self._nodes[name]
        except KeyError:
            raise KeyError(f"unknown type {name!r}") from None

    def ancestors(self, name: str) -> tuple[str, ...]:
        """*name* and every type above it, nearest first."""
        chain: list[str] = []
        current: str | None = name
        while current is not None:
            if current not in self._nodes:
                raise KeyError(f"unknown type {current!r}")
            chain.append(current)
            current = self._nodes[current].parent
        return tuple(chain)

    def children(self, name: str) -> tuple[str, ...]:
        self.node(name)
        return tuple(
            sorted(n.name for n in self._nodes.values() if n.parent == name)
        )

    def descendants(self, name: str) -> tuple[str, ...]:
        out: list[str] = []
        stack = list(self.children(name))
        while stack:
            current = stack.pop()
            out.append(current)
            stack.extend(self.children(current))
        return tuple(sorted(out))

    def labels_of(self, name: str) -> LabelSet:
        """Every label *name* carries, inherited ones included."""
        labels: set[Label] = set()
        for ancestor in self.ancestors(name):
            labels |= self._nodes[ancestor].extra_labels
        return LabelSet(frozenset(labels))

    def is_a(self, name: str, other: str | Label) -> bool:
        """Whether *name* is *other*, directly or by inheritance.

        This is the inference the handoff asks for: ``is_a("city",
        Label.LOCATION)`` is true because ``city`` sits under ``LOCATION``.
        """
        if isinstance(other, Label):
            return other in self.labels_of(name)
        return other in self.ancestors(name)

    def candidates(self, word: str) -> tuple[str, ...]:
        """Types *word* could denote. Empty when the word is unknown.

        More than one reading is the normal case and is not an error: the
        caller must decide from context, and a one-element result is not a
        licence to skip that step.
        """
        return self._lexicon.get(word, ())

    def candidate_labels(self, word: str) -> tuple[LabelSet, ...]:
        """One :class:`~.models.LabelSet` per reading of *word*."""
        return tuple(self.labels_of(name) for name in self.candidates(word))

    def is_ambiguous(self, word: str) -> bool:
        """Whether *word* has readings under more than one entity label."""
        seen = {labels.entity_labels for labels in self.candidate_labels(word)}
        return len(seen) > 1

    # -- serialisation -----------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TypeHierarchy":
        hierarchy = cls()
        raw_types = data.get("types", {})
        assert isinstance(raw_types, dict)
        # Insert parents before children, however the file is ordered.
        pending = dict(raw_types)
        while pending:
            progressed = False
            for name in list(pending):
                spec = pending[name]
                parent = spec["parent"] if isinstance(spec, dict) else str(spec)
                if parent not in hierarchy._nodes:
                    continue
                extra = spec.get("labels", ()) if isinstance(spec, dict) else ()
                hierarchy.add_type(name, parent, extra)
                del pending[name]
                progressed = True
            if not progressed:
                raise ValueError(
                    "unresolvable parents in the type table: " + ", ".join(sorted(pending))
                )
        for word, types in (data.get("lexicon") or {}).items():
            hierarchy.add_word(word, types)
        return hierarchy

    def to_dict(self) -> dict[str, object]:
        types: dict[str, object] = {}
        for name, node in sorted(self._nodes.items()):
            if node.parent is None:
                continue  # an ontology label, implied
            spec: dict[str, object] = {"parent": node.parent}
            if node.extra_labels:
                spec["labels"] = sorted(label.value for label in node.extra_labels)
            types[name] = spec
        return {
            "types": types,
            "lexicon": {word: list(t) for word, t in sorted(self._lexicon.items())},
        }

    @classmethod
    def load(cls, path: Path | str = DEFAULTS_PATH) -> "TypeHierarchy":
        with Path(path).open("rt", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def save(self, path: Path | str) -> None:
        with Path(path).open("wt", encoding="utf-8", newline="\n") as stream:
            json.dump(self.to_dict(), stream, indent=2, ensure_ascii=False)
            stream.write("\n")


def load_default_hierarchy() -> TypeHierarchy:
    """The ontology shipped with the package."""
    return TypeHierarchy.load(DEFAULTS_PATH)
