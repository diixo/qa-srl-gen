"""QA examples and how their answers are worded.

The model being trained answers in ordinary language, so the answers stored
here are ordinary language too: ``Anna.``, ``In Kyiv.``, ``The text does not
say.`` The structure that produced them stays in the metadata, where it is
useful for filtering and auditing, and out of the target string, which is
what the loss is computed over.

One rule governs everything in this package and is worth stating plainly:
**questions may be split, context may not.** A single sentence yields many
questions, and that is good. But each example carries all the context its
answer needs, because an example whose answer is ambiguous without
surrounding text teaches the model to guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

from ..documents import TextSpan

__all__ = [
    "QAKind",
    "QAExample",
    "NO_ANSWER_REPLIES",
    "phrase_answer",
    "sentence_case",
]


class QAKind(str, Enum):
    """The example types the handoff enumerates."""

    ATOMIC = "atomic"
    COMPOUND = "compound"
    CONTEXTUAL = "contextual"
    ONTOLOGY = "ontology"
    ENTITY_TYPE = "entity_type"
    PROPERTY = "property"
    SPEECH_ACT = "speech_act"
    NO_ANSWER = "no_answer"
    YES_NO = "yes_no"
    PARAPHRASE = "paraphrase"

    def __str__(self) -> str:
        return self.value


#: Acceptable ways of saying the text does not contain the answer. Several,
#: so the model learns the move rather than one memorised string.
NO_ANSWER_REPLIES: tuple[str, ...] = (
    "The text does not say.",
    "The text does not provide enough information.",
    "It is impossible to determine from the context.",
)


def sentence_case(text: str) -> str:
    """Capitalise the first letter, leaving the rest alone.

    ``str.capitalize`` would lower-case the remainder and destroy names.
    """
    return text[:1].upper() + text[1:] if text else text


def phrase_answer(text: str, preposition: str | None = None) -> str:
    """Word a span as a short answer: ``Anna.``, ``In Kyiv.``

    Adjuncts keep their preposition because *Kyiv.* is not an answer to
    *Where...?* in the way *In Kyiv.* is.
    """
    body = f"{preposition} {text}" if preposition else text
    body = body.strip()
    if not body:
        return ""
    if body[-1] in ".!?":
        return sentence_case(body)
    return sentence_case(body) + "."


@dataclass(frozen=True, slots=True)
class QAExample:
    """One training example: the context shown, the question, the answer.

    ``answerable`` is explicit rather than inferred from the answer text, so
    a corpus can be balanced and audited on it. ``evidence`` records the
    spans the answer came from, which makes an example checkable against its
    source long after it was produced.
    """

    context: str
    question: str
    answer: str
    kind: QAKind = QAKind.ATOMIC
    answerable: bool = True
    document_id: str = ""
    run_id: str | None = None
    evidence: tuple[TextSpan, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("a QA example needs a question")
        if not self.answer.strip():
            raise ValueError("a QA example needs an answer")
        if not self.context.strip():
            raise ValueError("a QA example needs context; answers must be checkable")

    def check(self) -> list[str]:
        """Problems that would make this example misleading to train on."""
        problems: list[str] = []
        if not self.question.rstrip().endswith("?"):
            problems.append(f"question does not end with '?': {self.question!r}")
        for span in self.evidence:
            if span.end_char > len(self.context):
                problems.append(
                    f"evidence span [{span.start_char}, {span.end_char}) "
                    "runs past the context"
                )
        if self.answerable and self.answer in NO_ANSWER_REPLIES:
            problems.append("marked answerable but answers with a refusal")
        if not self.answerable and self.answer not in NO_ANSWER_REPLIES:
            problems.append(
                f"marked unanswerable but gives a substantive answer: {self.answer!r}"
            )
        return problems

    def to_prompt(self) -> str:
        """The decoder-only training layout from the handoff."""
        return (
            f"<context>\n{self.context}\n</context>\n"
            f"<question>\n{self.question}\n</question>\n"
            f"<answer>\n{self.answer}\n</answer>"
        )

    def to_json(self) -> dict[str, object]:
        return {
            "context": self.context,
            "question": self.question,
            "answer": self.answer,
            "kind": str(self.kind),
            "answerable": self.answerable,
            "document_id": self.document_id,
            "run_id": self.run_id,
            "evidence": [span.to_json() for span in self.evidence],
            **({"metadata": dict(self.metadata)} if self.metadata else {}),
        }

    def __str__(self) -> str:
        return f"Q: {self.question}\nA: {self.answer}"
