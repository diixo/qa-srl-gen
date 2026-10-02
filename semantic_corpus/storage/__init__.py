"""Append-only corpus storage on the filesystem, as JSONL.

The handoff specifies SQLite; SQLite and Parquet are both ruled out, so the
nine tables are nine JSONL files in a directory with a manifest. Nothing is
ever rewritten: a second opinion is a new run, and a published dataset is a
named snapshot of what the store held at that moment.
"""

from .repository import (
    GENERATOR_VERSION,
    CorpusStore,
    DuplicateDocument,
    SplitConflict,
    StoreError,
)
from .schema import STORE_FORMAT, TABLES, OntologyVersionMismatch

__all__ = [
    "CorpusStore",
    "StoreError",
    "DuplicateDocument",
    "SplitConflict",
    "OntologyVersionMismatch",
    "TABLES",
    "STORE_FORMAT",
    "GENERATOR_VERSION",
]
