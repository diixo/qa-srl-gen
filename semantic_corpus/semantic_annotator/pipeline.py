"""Running the whole annotator over one document.

Holds no logic of its own beyond wiring: candidates are proposed per passage,
the teacher is asked once per passage, the answers are located, and the
result is collected into a single append-only run. Keeping it thin is what
lets each step be tested without the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..documents import AnnotationRun, Document, ONTOLOGY_VERSION
from ..ontology import ReviewStatus
from .alignment import AlignmentResult, Rejection, align_response
from .candidates import Candidate, CandidateResources, extract_candidates
from .teacher import PROMPT_VERSION, Teacher, request_for
from .verifier import StructuralReport, verify_structure

__all__ = ["AnnotationOutcome", "annotate_document"]


@dataclass(frozen=True, slots=True)
class AnnotationOutcome:
    """Everything one annotation pass produced, including what it discarded."""

    run: AnnotationRun
    document: Document
    candidates: tuple[Candidate, ...] = ()
    rejected: tuple[Rejection, ...] = ()
    structure: StructuralReport = field(default_factory=StructuralReport)

    @property
    def ok(self) -> bool:
        return self.structure.ok

    def summary(self) -> str:
        return (
            f"{self.document.document_id}: {len(self.run)} annotations from "
            f"{len(self.candidates)} candidates, {len(self.rejected)} rejected"
        )


def annotate_document(
    document: Document,
    teacher: Teacher,
    *,
    run_id: str,
    resources: CandidateResources | None = None,
    random_seed: int | None = None,
    status: ReviewStatus = ReviewStatus.AUTO_VALIDATED,
) -> AnnotationOutcome:
    """Annotate *document* with *teacher*, returning a new run.

    The document is not modified. If it has no passages, the whole text is
    treated as one, so an unsegmented document still works.
    """
    resources = resources or CandidateResources()
    passages: Sequence[object] = document.passages or (None,)

    run = AnnotationRun(
        run_id=run_id,
        ontology_version=ONTOLOGY_VERSION,
        model_name=getattr(teacher, "name", "teacher"),
        prompt_version=PROMPT_VERSION,
        random_seed=random_seed,
        synthetic=False,
    )

    all_candidates: list[Candidate] = []
    all_rejected: list[Rejection] = []

    for passage in passages:
        candidates = extract_candidates(document, passage, resources)  # type: ignore[arg-type]
        all_candidates.extend(candidates)
        response = teacher.annotate(request_for(document, passage, candidates))  # type: ignore[arg-type]
        identifier = getattr(passage, "passage_id", None) or document.document_id
        result: AlignmentResult = align_response(
            response,
            document,
            passage,  # type: ignore[arg-type]
            run_id=f"{run_id}#{identifier}",
            source=getattr(teacher, "name", "teacher"),
            status=status,
        )
        all_rejected.extend(result.rejected)
        run = run.extended(
            mentions=result.mentions,
            predicates=result.predicates,
            properties=result.properties,
            relations=result.relations,
        )

    return AnnotationOutcome(
        run=run,
        document=document,
        candidates=tuple(all_candidates),
        rejected=tuple(all_rejected),
        structure=verify_structure(run, document),
    )
