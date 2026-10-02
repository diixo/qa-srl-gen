"""The teacher adapter protocol.

The core must not know which model is annotating. The handoff names four
intended backends — a local HTTP service, a llama.cpp worker, the OpenAI API,
Bedrock — and forbids Hugging Face outright, so what is defined here is a
protocol plus the request and response shapes, not a client.

Two implementations ship:

:class:`HttpTeacher`
    speaks to any HTTP endpoint that accepts and returns JSON, using only
    :mod:`urllib` so the package stays dependency-free. The payload shape is
    injectable, which is what makes one class cover a local FastAPI service
    and a hosted chat API.

:class:`ScriptedTeacher`
    returns answers prepared in advance. Tests and the verification pass need
    a teacher that is deterministic and offline; without one, nothing about
    the annotator could be tested at all.

A teacher returns *text*, not offsets. It is asked to quote spans verbatim,
and :mod:`.alignment` is what turns those quotes back into positions — the
model is never trusted to count characters.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from ..documents import Document, Passage
from .candidates import Candidate

__all__ = [
    "TeacherRequest",
    "ProposedAnnotation",
    "TeacherResponse",
    "Teacher",
    "ScriptedTeacher",
    "HttpTeacher",
    "TeacherError",
    "build_prompt",
    "PROMPT_VERSION",
    "request_for",
]

#: Bumped whenever :func:`build_prompt` changes. Recorded on every annotation
#: run so that output made under different prompts is never pooled blindly.
PROMPT_VERSION = "annotator-v1"


class TeacherError(RuntimeError):
    """Raised when a teacher cannot be reached or returns unusable output."""


@dataclass(frozen=True, slots=True)
class TeacherRequest:
    """What the teacher is shown.

    ``context`` is the passage text and ``focus`` the part being annotated;
    they differ for dialogue, where the preceding turns are needed to read a
    short reply but are not themselves the target.
    """

    document_id: str
    passage_id: str | None
    context: str
    focus: str
    candidates: tuple[Candidate, ...] = ()
    task: str = "annotate"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "passage_id": self.passage_id,
            "context": self.context,
            "focus": self.focus,
            "task": self.task,
            "candidates": [
                {
                    "text": c.exact_text,
                    "proposed_labels": [str(label) for label in c.proposed_labels],
                    "evidence": list(c.evidence),
                    "lemmas": list(c.lemmas),
                }
                for c in self.candidates
            ],
            **({"metadata": dict(self.metadata)} if self.metadata else {}),
        }


@dataclass(frozen=True, slots=True)
class ProposedAnnotation:
    """One span the teacher claims, quoted rather than located.

    ``text`` must appear verbatim in the passage. ``occurrence`` disambiguates
    repeated strings: the second ``the dog`` in a passage is
    ``occurrence=1``.
    """

    text: str
    labels: tuple[str, ...]
    occurrence: int = 0
    lemma: str | None = None
    head: str | None = None
    degree: str | None = None
    negated: bool = False
    relation: str | None = None
    target_text: str | None = None
    question: str | None = None
    confidence: float | None = None
    target_occurrence: int = 0
    tense: str | None = None
    aspect: str | None = None
    voice: str | None = None
    modality: str | None = None

    def __post_init__(self):
        if self.occurrence < 0 or self.target_occurrence < 0:
            raise ValueError("occurrences must be nonnegative")

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ProposedAnnotation":
        if "text" not in data:
            raise TeacherError(f"annotation has no 'text' field: {data!r}")
        labels = data.get("labels") or ()
        if isinstance(labels, str):
            labels = [labels]
        return cls(
            text=str(data["text"]),
            labels=tuple(str(label) for label in labels),
            occurrence=int(data.get("occurrence", 0)),
            lemma=data.get("lemma"),
            head=data.get("head"),
            degree=data.get("degree"),
            negated=bool(data.get("negated", False)),
            relation=data.get("relation"),
            target_text=data.get("target_text"),
            question=data.get("question"),
            confidence=data.get("confidence"),
            target_occurrence=int(data.get("target_occurrence", 0)),
            tense=data.get("tense"), aspect=data.get("aspect"),
            voice=data.get("voice"), modality=data.get("modality"),
        )


@dataclass(frozen=True, slots=True)
class TeacherResponse:
    """What came back, before anything has been checked."""

    annotations: tuple[ProposedAnnotation, ...] = ()
    speech_acts: tuple[str, ...] = ()
    polarity: str | None = None
    stance: str | None = None
    mood: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "TeacherResponse":
        if not isinstance(data, Mapping):
            raise TeacherError(f"expected a JSON object, got {type(data).__name__}")
        raw_annotations = data.get("annotations") or ()
        if not isinstance(raw_annotations, Sequence) or isinstance(raw_annotations, str):
            raise TeacherError("'annotations' must be a list")
        return cls(
            annotations=tuple(
                ProposedAnnotation.from_json(item) for item in raw_annotations
            ),
            speech_acts=tuple(str(a) for a in (data.get("speech_acts") or ())),
            polarity=data.get("polarity"),
            stance=data.get("stance"),
            mood=data.get("mood"),
            raw=dict(data),
        )


@runtime_checkable
class Teacher(Protocol):
    """Anything that can annotate a request.

    Implementations must not mutate the request and must be safe to call
    repeatedly: the verification pass calls a teacher a second time over the
    same material.
    """

    name: str

    def annotate(self, request: TeacherRequest) -> TeacherResponse:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

_INSTRUCTIONS = """\
You are annotating English text with a fixed label set.

Rules:
- Quote every span exactly as it appears in the passage, character for
  character. Do not normalise, re-case or trim.
- If the same string appears more than once, set "occurrence" to its
  zero-based index.
- A span may carry several labels.
- Decide from the context. The candidate list is a set of suggestions
  produced by a dictionary; it is often wrong and never binding.
- If the passage does not support a label, leave it out. Do not guess.

Reply with JSON of the form:
{"annotations": [{"text": ..., "labels": [...], "occurrence": 0}]}
"""


def build_prompt(request: TeacherRequest, labels: Iterable[str]) -> str:
    """Render *request* as a prompt. Version it with :data:`PROMPT_VERSION`."""
    lines = [_INSTRUCTIONS, "", "Labels: " + ", ".join(sorted(labels)), ""]
    if request.context != request.focus:
        lines += ["Context:", request.context, ""]
    lines += ["Passage:", request.focus, ""]
    if request.candidates:
        lines.append("Candidates (suggestions only):")
        for candidate in request.candidates:
            lines.append(
                f"- {candidate.exact_text!r}: "
                f"{', '.join(str(l) for l in candidate.proposed_labels)}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


class ScriptedTeacher:
    """A teacher that replays prepared answers.

    Keyed by passage id when available and by focus text otherwise, so a test
    can prepare answers without knowing how passages will be numbered.
    """

    def __init__(
        self,
        answers: Mapping[str, TeacherResponse] | None = None,
        *,
        default: TeacherResponse | None = None,
        name: str = "scripted",
    ) -> None:
        self.answers = dict(answers or {})
        self.default = default
        self.name = name
        self.calls: list[TeacherRequest] = []

    def annotate(self, request: TeacherRequest) -> TeacherResponse:
        self.calls.append(request)
        for key in (request.passage_id, request.focus):
            if key is not None and key in self.answers:
                return self.answers[key]
        if self.default is not None:
            return self.default
        return TeacherResponse()


class HttpTeacher:
    """A teacher behind an HTTP+JSON endpoint.

    ``encode`` builds the request body and ``decode`` extracts the JSON
    object the annotator expects. Both are injectable because a local FastAPI
    service and a hosted chat API differ only in those two functions — the
    transport, the error handling and the retry behaviour are the same.
    """

    def __init__(
        self,
        url: str,
        *,
        name: str = "http",
        headers: Mapping[str, str] | None = None,
        timeout: float = 60.0,
        encode: Callable[[TeacherRequest], Mapping[str, Any]] | None = None,
        decode: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        labels: Iterable[str] = (),
    ) -> None:
        self.url = url
        self.name = name
        self.headers = {"Content-Type": "application/json", **dict(headers or {})}
        self.timeout = timeout
        self.labels = tuple(labels)
        self._encode = encode or self._default_encode
        self._decode = decode or (lambda payload: payload)

    def _default_encode(self, request: TeacherRequest) -> Mapping[str, Any]:
        return {
            "prompt": build_prompt(request, self.labels),
            "prompt_version": PROMPT_VERSION,
            **request.to_json(),
        }

    def annotate(self, request: TeacherRequest) -> TeacherResponse:
        body = json.dumps(self._encode(request)).encode("utf-8")
        http_request = urllib.request.Request(
            self.url, data=body, headers=dict(self.headers), method="POST"
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise TeacherError(f"{self.name}: cannot reach {self.url}: {error}") from error
        except json.JSONDecodeError as error:
            raise TeacherError(f"{self.name}: response was not JSON: {error}") from error
        return TeacherResponse.from_json(self._decode(payload))


def request_for(
    document: Document,
    passage: Passage | None,
    candidates: Sequence[Candidate] = (),
    *,
    focus: str | None = None,
    task: str = "annotate",
) -> TeacherRequest:
    """Build the request for a passage of a document."""
    context = passage.span.text_in(document.text) if passage else document.text
    return TeacherRequest(
        document_id=document.document_id,
        passage_id=passage.passage_id if passage else None,
        context=context,
        focus=focus if focus is not None else context,
        candidates=tuple(candidates),
        task=task,
    )
