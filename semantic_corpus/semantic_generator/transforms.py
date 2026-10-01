"""Surface variation over one semantic record.

A single situation should yield many sentences — the handoff lists tense,
voice, negation, modality and word order — because the point of the corpus is
that the same meaning survives rewording. The record does not change; only
its realisation does, so every variant carries the same relations and the
same answers.

Licensing is delegated to the QA-SRL grammar rather than re-decided here: a
progressive passive under a modal is excluded because the template that
produced the whole bank excludes it, not because of a rule invented for the
generator.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Iterator

from ..qasrl_core.state_machine import TENSES, TenseFeatures
from .frames import SurfacePattern
from .realization import Features, RealizedSituation, Situation, realize

__all__ = [
    "enumerate_features",
    "feature_variants",
    "pattern_variants",
    "all_realizations",
    "negate",
    "set_tense",
    "DEFAULT_TENSES",
]

#: A readable subset of the tense inventory; the full set is :data:`TENSES`.
DEFAULT_TENSES: tuple[str, ...] = ("present", "past", "will", "can", "might")


def enumerate_features(
    *,
    tenses: Iterable[str] = DEFAULT_TENSES,
    perfect: Iterable[bool] = (False, True),
    progressive: Iterable[bool] = (False, True),
    negated: Iterable[bool] = (False, True),
    is_passive: bool = False,
) -> tuple[Features, ...]:
    """Every feature bundle the grammar licenses, in a stable order.

    *is_passive* is a property of the chosen surface pattern, not of the
    features, but it has to be known here: it is what rules out the
    progressive passives the template never offers.
    """
    out: list[Features] = []
    for tense in tenses:
        if tense not in TENSES:
            raise ValueError(f"unknown tense {tense!r}; expected one of {TENSES}")
        for is_perfect in perfect:
            for is_progressive in progressive:
                for is_negated in negated:
                    licensed = TenseFeatures(
                        tense=tense,
                        is_perfect=is_perfect,
                        is_progressive=is_progressive,
                        is_passive=is_passive,
                        is_negated=is_negated,
                    ).is_licensed
                    if licensed:
                        out.append(
                            Features(
                                tense=tense,
                                is_perfect=is_perfect,
                                is_progressive=is_progressive,
                                is_negated=is_negated,
                            )
                        )
    return tuple(out)


def set_tense(situation: Situation, tense: str) -> Situation:
    return replace(situation, features=replace(situation.features, tense=tense))


def negate(situation: Situation, negated: bool = True) -> Situation:
    return replace(situation, features=replace(situation.features, is_negated=negated))


def feature_variants(
    situation: Situation,
    *,
    tenses: Iterable[str] = DEFAULT_TENSES,
    is_passive: bool = False,
    **kwargs,
) -> tuple[Situation, ...]:
    """The same bindings under every licensed feature bundle."""
    return tuple(
        replace(situation, features=features)
        for features in enumerate_features(
            tenses=tenses, is_passive=is_passive, **kwargs
        )
    )


def pattern_variants(situation: Situation) -> tuple[SurfacePattern, ...]:
    """Patterns whose required slots this situation actually binds."""
    bound = set(situation.bindings)
    adjuncts = situation.frame.adjunct_slots
    usable = []
    for pattern in situation.frame.patterns:
        needed = pattern.mentioned() - adjuncts
        if needed <= bound:
            usable.append(pattern)
    return tuple(usable)


def all_realizations(
    situation: Situation,
    *,
    tenses: Iterable[str] = DEFAULT_TENSES,
    perfect: Iterable[bool] = (False,),
    progressive: Iterable[bool] = (False,),
    negated: Iterable[bool] = (False, True),
) -> Iterator[RealizedSituation]:
    """Every sentence this situation can produce, across patterns and features.

    Defaults are deliberately narrow: the full cross-product runs to hundreds
    of sentences per situation, which floods a corpus with near-duplicates.
    Widen it explicitly when that is what you want.
    """
    for pattern in pattern_variants(situation):
        for features in enumerate_features(
            tenses=tenses,
            perfect=perfect,
            progressive=progressive,
            negated=negated,
            is_passive=pattern.is_passive,
        ):
            yield realize(replace(situation, features=features), pattern)
