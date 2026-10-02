"""Questions the text genuinely cannot answer.

Negative examples exist so the model learns to decline instead of inventing.
That only works if the questions are *really* unanswerable, so every one here
is grounded in the record rather than invented:

* an argument the record knows about but the sentence does not express — the
  passive that drops its agent is the clearest case;
* a role the predicate never had, so there is nothing to find;
* a name used in a sense the context does not settle.

A question that is merely hard, or whose answer sits in another sentence of
the same document, is not a negative example. Training on those teaches the
model to refuse when it should answer, which is the opposite failure and
harder to notice.

**Absence from the record is not absence from the text.** For an annotated
document, a role missing from the run may simply be a role the annotator did
not capture — the sentence *On Monday, Anna gave Rex a ball* has a time
whether or not anything recorded it, and asking *When did this happen?* would
produce a refusal that the context contradicts. So the inferences that read
something into silence apply only to runs marked ``synthetic``, where the
record is complete by construction. For annotated runs they return nothing,
and negatives must come from explicit evidence instead.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

from ..documents import AnnotationRun, Document
from ..ontology import Label, Relation
from .answers import NO_ANSWER_REPLIES, QAExample, QAKind
from .templates import (
    ParadigmResolver,
    _inline,
    active_verb_words,
    build_views,
)

__all__ = [
    "no_answer_questions",
    "omitted_argument_questions",
    "missing_role_questions",
    "underdetermined_type_questions",
]

#: The roles worth asking about when they are absent, and how to ask.
_MISSING_ROLE_QUESTIONS = {
    Relation.LOCATION_OF: "Where did this happen?",
    Relation.TIME_OF: "When did this happen?",
}

_SLOT_QUESTION = {
    "agent": "Who {verb} {theme}?",
    "theme": "What was {participle}?",
    "recipient": "Who received {theme}?",
}


def _reply(rng: random.Random | None) -> str:
    """One of the accepted refusals; varied so the wording is not memorised."""
    if rng is None:
        return NO_ANSWER_REPLIES[0]
    return rng.choice(NO_ANSWER_REPLIES)


def omitted_argument_questions(
    document: Document,
    run: AnnotationRun,
    *,
    resolver: ParadigmResolver | None = None,
    rng: random.Random | None = None,
    context: str | None = None,
) -> list[QAExample]:
    """Ask about an argument the sentence deliberately left out.

    The generator records which slots a surface pattern omits, so this is the
    one case where unanswerability is certain: the frame had an agent, the
    passive dropped it, and no amount of reading will recover it.
    """
    omitted = document.metadata.get("omitted_slots") or ()
    if not omitted:
        return []
    resolver = resolver or ParadigmResolver()
    body = context if context is not None else document.text
    examples: list[QAExample] = []
    for view in build_views(run, document):
        theme = view.theme
        active = " ".join(active_verb_words(view.predicate, resolver))
        for slot in omitted:
            template = _SLOT_QUESTION.get(str(slot))
            if template is None:
                continue
            question = template.format(
                verb=active,
                participle=view.predicate.exact_text,
                theme=_inline(theme),
            )
            question = " ".join(question.split())
            examples.append(
                QAExample(
                    context=body,
                    question=question.replace(" ?", "?"),
                    answer=_reply(rng),
                    kind=QAKind.NO_ANSWER,
                    answerable=False,
                    document_id=document.document_id,
                    run_id=run.run_id,
                    metadata={"omitted_slot": str(slot), "reason": "omitted_by_pattern"},
                )
            )
    return examples


def missing_role_questions(
    document: Document,
    run: AnnotationRun,
    *,
    rng: random.Random | None = None,
    context: str | None = None,
) -> list[QAExample]:
    """Ask about a role the sentence never had.

    Only for synthetic runs. In an annotated run the role may be present in
    the text and merely unannotated, and a refusal would then be flatly
    wrong — see the module docstring.
    """
    if not run.synthetic:
        return []
    body = context if context is not None else document.text
    examples: list[QAExample] = []
    for view in build_views(run, document):
        for relation, question in _MISSING_ROLE_QUESTIONS.items():
            if view.by_relation.get(relation):
                continue
            examples.append(
                QAExample(
                    context=body,
                    question=question,
                    answer=_reply(rng),
                    kind=QAKind.NO_ANSWER,
                    answerable=False,
                    document_id=document.document_id,
                    run_id=run.run_id,
                    metadata={"missing_role": str(relation), "reason": "role_absent"},
                )
            )
    return examples


def underdetermined_type_questions(
    document: Document,
    run: AnnotationRun,
    *,
    ambiguous_forms: Iterable[str] = (),
    rng: random.Random | None = None,
    context: str | None = None,
) -> list[QAExample]:
    """Ask the type of a name the context does not settle.

    The handoff's example: *Anna spoke to Rex. Is Rex a person or an animal?*
    Only names the corpus actually uses in more than one sense qualify, and
    only when this sentence gives no gloss — otherwise the question has an
    answer and refusing would be wrong.

    Synthetic runs only, for the same reason as
    :func:`missing_role_questions`: in real text the surrounding words may
    settle a type without any annotation saying so.
    """
    forms = {form for form in ambiguous_forms}
    if not forms or not run.synthetic:
        return []
    body = context if context is not None else document.text
    glossed = {
        mention.exact_text
        for mention in run.mentions
        if mention.mention_id.endswith(":appositive")
    }
    examples: list[QAExample] = []
    for mention in run.mentions:
        if mention.exact_text not in forms or mention.mention_id.endswith(":appositive"):
            continue
        if _has_gloss(run, mention.mention_id):
            continue
        examples.append(
            QAExample(
                context=body,
                question=f"Is {mention.exact_text} a person or an animal?",
                answer=_reply(rng),
                kind=QAKind.NO_ANSWER,
                answerable=False,
                document_id=document.document_id,
                run_id=run.run_id,
                metadata={"form": mention.exact_text, "reason": "type_underdetermined"},
            )
        )
    return examples


def _has_gloss(run: AnnotationRun, mention_id: str) -> bool:
    """Whether an appositive in the same sentence settles this mention's type."""
    return any(
        edge.source_id == mention_id and edge.relation is Relation.INSTANCE_OF
        for edge in run.relations
    )


def no_answer_questions(
    document: Document,
    run: AnnotationRun,
    *,
    resolver: ParadigmResolver | None = None,
    ambiguous_forms: Iterable[str] = (),
    rng: random.Random | None = None,
    context: str | None = None,
) -> list[QAExample]:
    """Every unanswerable question this record supports."""
    return [
        *omitted_argument_questions(
            document, run, resolver=resolver, rng=rng, context=context
        ),
        *missing_role_questions(document, run, rng=rng, context=context),
        *underdetermined_type_questions(
            document, run, ambiguous_forms=ambiguous_forms, rng=rng, context=context
        ),
    ]
