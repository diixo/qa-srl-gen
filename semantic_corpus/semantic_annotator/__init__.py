"""Annotating real text: ingestion, candidates, teacher, alignment, verification.

The pipeline is deliberately split so that each step can be checked on its
own::

    ingest ──> extract_candidates ──> teacher ──> align_response ──> verify

Only one step involves a model, and its output is treated as a set of claims
to be located and checked, never as positions to be trusted.
"""

from .alignment import (
    AlignmentResult,
    LocationError,
    Rejection,
    align_response,
    locate,
)
from .candidates import Candidate, CandidateResources, extract_candidates, tokenize
from .ingestion import (
    assign_split,
    deduplicate,
    ingest,
    read_dialogue_jsonl,
    read_jsonl_documents,
    read_text_file,
    segment_dialogue,
    segment_document,
)
from .pipeline import AnnotationOutcome, annotate_document
from .teacher import (
    PROMPT_VERSION,
    HttpTeacher,
    ProposedAnnotation,
    ScriptedTeacher,
    Teacher,
    TeacherError,
    TeacherRequest,
    TeacherResponse,
    build_prompt,
    request_for,
)
from .verifier import (
    AgreementReport,
    StructuralReport,
    promote_agreed,
    verify_independently,
    verify_structure,
)

__all__ = [
    "AlignmentResult",
    "LocationError",
    "Rejection",
    "align_response",
    "locate",
    "Candidate",
    "CandidateResources",
    "extract_candidates",
    "tokenize",
    "assign_split",
    "deduplicate",
    "ingest",
    "read_dialogue_jsonl",
    "read_jsonl_documents",
    "read_text_file",
    "segment_dialogue",
    "segment_document",
    "AnnotationOutcome",
    "annotate_document",
    "PROMPT_VERSION",
    "HttpTeacher",
    "ProposedAnnotation",
    "ScriptedTeacher",
    "Teacher",
    "TeacherError",
    "TeacherRequest",
    "TeacherResponse",
    "build_prompt",
    "request_for",
    "AgreementReport",
    "StructuralReport",
    "promote_agreed",
    "verify_independently",
    "verify_structure",
]
