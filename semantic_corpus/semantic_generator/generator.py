"""Deterministic sampling of situations from frames and entity pools.

Everything random here goes through one :class:`random.Random`, seeded by the
caller, so a run is reproducible from ``(seed, split, frames, pool)`` alone.
That is a hard requirement from the handoff and it is also what makes a
held-out test set trustworthy: if generation drifted between runs, the claim
that a name was never trained on could not be checked.

This module is not in the handoff's file list. It exists because sampling
needs frames, pools and realisation at once, and putting it in any of those
three would make them import each other in a cycle.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Iterable, Iterator, Sequence

from ..ontology import Label, TypeHierarchy, load_default_hierarchy
from ..qasrl_core.models import InflectedForms
from .frames import DEFAULT_FRAMES, SemanticFrame
from .realization import (
    BindingError,
    Features,
    RealizedSituation,
    Situation,
    realize,
)
from .substitutions import Entity, EntityPool, Split, default_pool

__all__ = ["Generator", "AmbiguityPair", "DEFAULT_PARADIGMS"]

#: Paradigms for the starter frames, spelled out so that generation does not
#: depend on what the Wiktionary scrape happens to contain.
DEFAULT_PARADIGMS: dict[str, InflectedForms] = {
    "give": InflectedForms("give", "gives", "giving", "gave", "given"),
    "visit": InflectedForms("visit", "visits", "visiting", "visited", "visited"),
    "own": InflectedForms("own", "owns", "owning", "owned", "owned"),
    "feel": InflectedForms("feel", "feels", "feeling", "felt", "felt"),
    "move": InflectedForms("move", "moves", "moving", "moved", "moved"),
    "see": InflectedForms("see", "sees", "seeing", "saw", "seen"),
    "say": InflectedForms("say", "says", "saying", "said", "said"),
}


@dataclass(frozen=True, slots=True)
class AmbiguityPair:
    """Two sentences in which one name denotes two different kinds of thing."""

    surface_form: str
    left: RealizedSituation
    right: RealizedSituation
    left_label: Label
    right_label: Label

    def question(self) -> str:
        return f"What kind of entity is {self.surface_form}?"


class Generator:
    """Draws situations from a frame set and an entity pool."""

    def __init__(
        self,
        *,
        seed: int,
        frames: Sequence[SemanticFrame] = DEFAULT_FRAMES,
        pool: EntityPool | None = None,
        hierarchy: TypeHierarchy | None = None,
        split: Split | None = None,
        paradigms: dict[str, InflectedForms] | None = None,
        optional_slot_probability: float = 0.5,
    ) -> None:
        self.seed = seed
        self.frames = tuple(frames)
        self.hierarchy = hierarchy or load_default_hierarchy()
        full_pool = pool if pool is not None else default_pool(self.hierarchy)
        self.split = split
        self.pool = full_pool.split(split) if split else full_pool
        self.paradigms = dict(paradigms or DEFAULT_PARADIGMS)
        self.optional_slot_probability = optional_slot_probability
        self._rng = random.Random(seed)

    def reset(self) -> None:
        """Rewind the stream, so the next draw repeats the first one."""
        self._rng = random.Random(self.seed)

    # -- drawing -----------------------------------------------------------

    def forms_for(self, frame: SemanticFrame) -> InflectedForms:
        try:
            return self.paradigms[frame.lemma]
        except KeyError:
            raise KeyError(
                f"no paradigm registered for {frame.lemma!r}; pass one in "
                "`paradigms`"
            ) from None

    def sample_situation(
        self,
        frame: SemanticFrame | None = None,
        *,
        features: Features | None = None,
        distinct: bool = True,
        exclude_forms: Iterable[str] = (),
    ) -> Situation:
        """Bind every required slot, and optional ones by coin flip.

        With *distinct*, one entity cannot fill two slots of the same
        sentence: *Anna gave Anna a ball* is grammatical but useless as
        training data, and it makes the answers to several questions
        indistinguishable.
        """
        frame = frame or self._rng.choice(self.frames)
        bindings: dict[str, Entity] = {}
        used: set[str] = set(exclude_forms)
        for slot in frame.slots:
            if slot.optional and self._rng.random() >= self.optional_slot_probability:
                continue
            try:
                entity = self.pool.sample(
                    slot.types,
                    self._rng,
                    exclude=used if distinct else (),
                    exclude_labels=slot.excludes,
                )
            except LookupError:
                if slot.optional:
                    continue
                raise
            bindings[slot.name] = entity
            if distinct:
                used.add(entity.text)
        return Situation(
            frame=frame,
            bindings=bindings,
            forms=self.forms_for(frame),
            features=features or Features(tense=self._rng.choice(("past", "present"))),
        )

    def generate(self, count: int) -> Iterator[RealizedSituation]:
        """Yield *count* realised sentences, one pattern each."""
        produced = 0
        attempts = 0
        while produced < count:
            attempts += 1
            if attempts > count * 20 + 100:
                raise RuntimeError(
                    f"gave up after {attempts} attempts with {produced}/{count} "
                    "sentences; the pool probably cannot fill these frames"
                )
            situation = self.sample_situation()
            from .transforms import pattern_variants

            patterns = pattern_variants(situation)
            if not patterns:
                continue
            pattern = self._rng.choice(patterns)
            try:
                yield realize(situation, pattern)
            except BindingError:
                continue
            produced += 1

    # -- deliberate ambiguity ---------------------------------------------

    def ambiguity_pairs(self) -> Iterator[AmbiguityPair]:
        """Sentences where context, not the name, settles the entity type.

        For each surface form with several readings, find two frames whose
        typed slots force different readings, and realise one sentence for
        each. A model that has memorised ``Rex -> ANIMAL`` gets the second
        one wrong, which is exactly what ``ambiguous_names_test`` is for.
        """
        for form in self.pool.ambiguous_forms():
            # Prefer the glossed reading: an appositive is what lets a reader
            # settle the type, so a pair built on bare names would not test
            # context at all.
            readings = sorted(
                self.pool.readings(form), key=lambda e: e.appositive is None
            )
            placements: dict[Label, tuple[SemanticFrame, str, Entity]] = {}
            for entity in readings:
                label = entity.entity_label
                if label is None or label in placements:
                    continue
                found = self._slot_accepting(entity)
                if found is not None:
                    placements[label] = (*found, entity)
            if len(placements) < 2:
                continue
            (left_label, left), (right_label, right) = sorted(
                placements.items(), key=lambda kv: kv[0].value
            )[:2]
            left_realized = self._realize_with(*left)
            right_realized = self._realize_with(*right)
            if left_realized is None or right_realized is None:
                continue
            yield AmbiguityPair(
                surface_form=form,
                left=left_realized,
                right=right_realized,
                left_label=left_label,
                right_label=right_label,
            )

    def _slot_accepting(self, entity: Entity) -> tuple[SemanticFrame, str] | None:
        """A frame slot whose type constraint only this reading satisfies."""
        for frame in self.frames:
            for slot in frame.slots:
                if slot.optional or slot.name in frame.adjunct_slots:
                    continue
                if slot.accepts(entity.labels.labels):
                    return frame, slot.name
        return None

    def _realize_with(
        self, frame: SemanticFrame, slot_name: str, entity: Entity
    ) -> RealizedSituation | None:
        # Keep the surface form out of the other slots: a sentence in which
        # one name fills two positions with two different readings makes the
        # very point the pair is meant to isolate unreadable.
        situation = self.sample_situation(frame, exclude_forms={entity.text})
        bindings = dict(situation.bindings)
        bindings[slot_name] = entity
        situation = replace(situation, bindings=bindings)
        from .transforms import pattern_variants

        for pattern in pattern_variants(situation):
            if slot_name in pattern.mentioned():
                try:
                    return realize(situation, pattern)
                except BindingError:
                    continue
        return None

    # -- unanswerable questions -------------------------------------------

    def omitted_argument_examples(self, count: int) -> Iterator[RealizedSituation]:
        """Sentences that leave a known argument unexpressed.

        The record still says there was an agent; the sentence does not name
        it. That is the honest basis for a *the text does not say* example —
        the answer is genuinely absent rather than merely hard to find.
        """
        produced = 0
        for frame in self._cycle(self.frames):
            if produced >= count:
                return
            patterns = [p for p in frame.patterns if p.omits]
            if not patterns:
                continue
            situation = self.sample_situation(frame)
            pattern = self._rng.choice(patterns)
            if not (pattern.mentioned() - frame.adjunct_slots) <= set(situation.bindings):
                continue
            try:
                yield realize(situation, pattern)
            except BindingError:
                continue
            produced += 1

    def _cycle(self, items: Iterable[SemanticFrame]) -> Iterator[SemanticFrame]:
        pool = list(items)
        if not pool:
            return
        while True:
            for item in pool:
                yield item
