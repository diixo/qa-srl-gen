"""Running the whole annotator over one document.

Holds no logic of its own beyond wiring: candidates are proposed per passage,
the teacher is asked once per passage, the answers are located, and the
result is collected into a single append-only run. Keeping it thin is what
lets each step be tested without the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..documents import (
    AnnotationRun,
    DialogueAnnotation,
    Document,
    ONTOLOGY_VERSION,
    Utterance,
)
from ..ontology import (
    MarkerFunction,
    Mood,
    Polarity,
    ReviewStatus,
    SpeechAct,
    Stance,
)
from .alignment import AlignmentResult, Rejection, align_response
from .candidates import Candidate, CandidateResources, extract_candidates
from .teacher import PROMPT_VERSION, Teacher, TeacherResponse, request_for
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

    # For a dialogue, each passage is one turn plus the turns before it, so
    # a response can be attached to the turn it was about. Pairing them here
    # is what lets a teacher's utterance-level verdict become a
    # DialogueAnnotation instead of being discarded.
    utterance_for = _utterance_by_passage(document)

    for passage in passages:
        candidates = extract_candidates(document, passage, resources)  # type: ignore[arg-type]
        all_candidates.extend(candidates)
        utterance = utterance_for.get(getattr(passage, "passage_id", None))
        request = request_for(
            document,
            passage,  # type: ignore[arg-type]
            candidates,
            focus=utterance.text_in(document) if utterance is not None else None,
        )
        response = teacher.annotate(request)
        identifier = getattr(passage, "passage_id", None) or document.document_id
        result: AlignmentResult = align_response(
            response,
            document,
            passage,  # type: ignore[arg-type]
            run_id=f"{run_id}#{identifier}",
            source=getattr(teacher, "name", "teacher"),
            status=status,
            # The teacher was asked about one turn; its quotes must be found
            # in that turn, not anywhere in the context around it.
            window=utterance.span if utterance is not None else None,
        )
        all_rejected.extend(result.rejected)
        run = run.extended(
            mentions=result.mentions,
            predicates=result.predicates,
            properties=result.properties,
            relations=result.relations,
            dialogue=_dialogue_annotations(
                response, utterance, getattr(teacher, "name", "teacher"), status
            ),
        )

    return AnnotationOutcome(
        run=run,
        document=document,
        candidates=tuple(all_candidates),
        rejected=tuple(all_rejected),
        structure=verify_structure(run, document),
    )


def _utterance_by_passage(document: Document) -> dict[str, Utterance]:
    """Pair each passage with the turn it was built around.

    ``segment_dialogue`` ends every passage at the turn being annotated, so
    the last utterance inside a passage is that turn.
    """
    if not document.is_dialogue:
        return {}
    pairs: dict[str, Utterance] = {}
    for passage in document.passages:
        inside = [u for u in document.utterances if passage.span.contains(u.span)]
        if inside:
            pairs[passage.passage_id] = inside[-1]
    return pairs


def _dialogue_annotations(
    response: TeacherResponse,
    utterance: Utterance | None,
    source: str,
    status: ReviewStatus,
) -> list[DialogueAnnotation]:
    """Build the utterance-level annotation a teacher reported, if any."""
    if utterance is None:
        return []
    speech_acts = tuple(SpeechAct(a) for a in response.speech_acts)
    if not (speech_acts or response.polarity or response.stance or response.mood):
        return []
    marker_functions: tuple[MarkerFunction, ...] = ()
    raw_functions = response.raw.get("marker_functions") if response.raw else None
    if raw_functions:
        marker_functions = tuple(MarkerFunction(f) for f in raw_functions)
    return [
        DialogueAnnotation(
            utterance_id=utterance.utterance_id,
            speech_acts=speech_acts,
            polarity=Polarity(response.polarity) if response.polarity else None,
            stance=Stance(response.stance) if response.stance else None,
            mood=Mood(response.mood) if response.mood else None,
            marker_functions=marker_functions,
            source=source,
            review_status=status,
        )
    ]
