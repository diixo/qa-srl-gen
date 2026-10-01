"""Ontology v1: labels, the type hierarchy and its defaults."""

from .hierarchy import DEFAULTS_PATH, TypeHierarchy, TypeNode, load_default_hierarchy
from .models import (
    CHARACTERISTIC_LABELS,
    ENTITY_LABELS,
    NAMEDNESS_LABELS,
    PREDICATE_LABELS,
    Label,
    LabelSet,
    MarkerForm,
    MarkerFunction,
    Polarity,
    Relation,
    SpeechAct,
)

__all__ = [
    "DEFAULTS_PATH",
    "TypeHierarchy",
    "TypeNode",
    "load_default_hierarchy",
    "Label",
    "LabelSet",
    "MarkerForm",
    "MarkerFunction",
    "Polarity",
    "Relation",
    "SpeechAct",
    "PREDICATE_LABELS",
    "ENTITY_LABELS",
    "CHARACTERISTIC_LABELS",
    "NAMEDNESS_LABELS",
]
