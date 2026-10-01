"""Controlled generation of semantic situations and their surface forms."""

from .frames import (
    DEFAULT_FRAMES,
    Complement,
    FrameSlot,
    SemanticFrame,
    SurfacePattern,
    frame_by_lemma,
)
from .generator import DEFAULT_PARADIGMS, AmbiguityPair, Generator
from .realization import (
    BindingError,
    Features,
    MentionSpan,
    RealizedSituation,
    RelationEdge,
    Situation,
    realize,
)
from .substitutions import SPLITS, Entity, EntityPool, default_pool
from .transforms import (
    DEFAULT_TENSES,
    all_realizations,
    enumerate_features,
    feature_variants,
    negate,
    pattern_variants,
    set_tense,
)

__all__ = [
    "DEFAULT_FRAMES",
    "Complement",
    "FrameSlot",
    "SemanticFrame",
    "SurfacePattern",
    "frame_by_lemma",
    "Generator",
    "AmbiguityPair",
    "DEFAULT_PARADIGMS",
    "BindingError",
    "Features",
    "MentionSpan",
    "RealizedSituation",
    "RelationEdge",
    "Situation",
    "realize",
    "Entity",
    "EntityPool",
    "default_pool",
    "SPLITS",
    "DEFAULT_TENSES",
    "all_realizations",
    "enumerate_features",
    "feature_variants",
    "negate",
    "pattern_variants",
    "set_tense",
]
