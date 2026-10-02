"""Turning a store into the formats a trainer or evaluator consumes.

Exports are derived views. The store stays the record of what is true; an
export may be lossy, and when it is — as BIO necessarily is over a
multi-label, nestable ontology — it says by how much.
"""

from .bio import BioReport, Tagging, export_bio, tag_document
from .jsonl import ExportCounts, read_jsonl, write_jsonl, write_records
from .report import CorpusReport, LeakageFinding, build_report, write_report
from .sft import check_record, export_sft, export_sft_splits, to_record

__all__ = [
    "export_sft",
    "export_sft_splits",
    "to_record",
    "check_record",
    "export_bio",
    "tag_document",
    "BioReport",
    "Tagging",
    "write_jsonl",
    "read_jsonl",
    "write_records",
    "ExportCounts",
    "build_report",
    "write_report",
    "CorpusReport",
    "LeakageFinding",
]
