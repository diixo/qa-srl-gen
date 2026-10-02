"""Driving the question generator over a document and its annotations.

Like the semantic generator's driver, this module exists because producing a
balanced set needs every template family at once, and putting the driver
inside any one of them would make them import each other. It is not in the
handoff's file list.

Two properties are enforced here rather than left to the caller:

**Context travels with the question.** Splitting a sentence into eight
questions is wanted; detaching them from the text that makes their answers
unambiguous is not. Every example carries its own context, and for dialogue
that context reaches back far enough to include the turns the answer depends
on.

**Nothing is emitted that cannot be checked.** Each example goes through
:meth:`~.answers.QAExample.check` before it is yielded, so a template bug
surfaces at generation time rather than inside a training run.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from ..documents import AnnotationRun, Document, Passage
from ..ontology import MarkerFunction, SpeechAct, TypeHierarchy, load_default_hierarchy
from ..qasrl_core.inflections import InflectionLexicon
from ..qasrl_core.models import InflectedForms
from .answers import QAExample, QAKind, phrase_answer, sentence_case
from .negatives import no_answer_questions
from .paraphrases import paraphrase
from .templates import (
    ParadigmResolver,
    atomic_questions,
    compound_questions,
    entity_type_questions,
    ontology_questions,
    property_questions,
    yes_no_questions,
)

__all__ = ["QuestionGenerator", "dialogue_act_questions"]


def dialogue_act_questions(
    document: Document,
    run: AnnotationRun,
    *,
    context_turns: int = 2,
) -> list[QAExample]:
    """Ask what a turn is doing, with the turns it responds to in view.

    A short reaction is meaningless alone: *Wow* is not interpretable until
    you can see what it answers. So the context is the preceding turns plus
    this one, never the turn by itself.
    """
    utterances = {u.utterance_id: u for u in document.utterances}
    order = {u.utterance_id: i for i, u in enumerate(document.utterances)}
    examples: list[QAExample] = []

    for annotation in run.dialogue:
        utterance = utterances.get(annotation.utterance_id)
        if utterance is None:
            continue
        index = order[utterance.utterance_id]
        first = document.utterances[max(0, index - context_turns)]
        context = document.text[first.span.start_char : utterance.span.end_char]
        turn_text = utterance.text_in(document)

        if annotation.speech_acts:
            examples.append(
                QAExample(
                    context=context,
                    question=f'What is the speaker doing when they say "{turn_text}"?',
                    answer=phrase_answer(
                        _list_phrase([_readable(a) for a in annotation.speech_acts])
                    ),
                    kind=QAKind.SPEECH_ACT,
                    document_id=document.document_id,
                    run_id=run.run_id,
                    evidence=(utterance.span,),
                    metadata={"utterance_id": utterance.utterance_id},
                )
            )
        if annotation.marker_functions:
            examples.append(
                QAExample(
                    context=context,
                    question=f'What does "{turn_text}" express?',
                    answer=phrase_answer(
                        _list_phrase(
                            [_readable(f) for f in annotation.marker_functions]
                        )
                    ),
                    kind=QAKind.SPEECH_ACT,
                    document_id=document.document_id,
                    run_id=run.run_id,
                    evidence=(utterance.span,),
                    metadata={"marker": True},
                )
            )
    return examples


def _readable(value) -> str:
    return str(value).lower().replace("_", " ")


def _list_phrase(items: Sequence[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


@dataclass(frozen=True, slots=True)
class QuestionGenerator:
    """Produces QA examples from documents and their annotation runs."""

    paradigms: dict[str, InflectedForms] = field(default_factory=dict)
    lexicon: InflectionLexicon | None = None
    hierarchy: TypeHierarchy | None = None
    ambiguous_forms: tuple[str, ...] = ()
    seed: int = 0
    include_paraphrases: bool = True
    include_negatives: bool = True

    @property
    def resolver(self) -> ParadigmResolver:
        return ParadigmResolver(known=self.paradigms, lexicon=self.lexicon)

    def _hierarchy(self) -> TypeHierarchy:
        return self.hierarchy or load_default_hierarchy()

    def for_document(
        self,
        document: Document,
        run: AnnotationRun,
        *,
        context: str | None = None,
    ) -> list[QAExample]:
        """Every example this document and run support, deduplicated."""
        rng = random.Random(f"{self.seed}:{document.document_id}")
        hierarchy = self._hierarchy()
        resolver = self.resolver

        produced: list[QAExample] = []
        produced += atomic_questions(document, run, resolver=resolver, context=context)
        produced += compound_questions(document, run, context=context)
        produced += entity_type_questions(document, run, context=context)
        produced += ontology_questions(document, run, hierarchy, context=context)
        produced += property_questions(document, run, hierarchy, context=context)
        produced += yes_no_questions(document, run, resolver=resolver, context=context)
        produced += dialogue_act_questions(document, run)

        if self.include_negatives:
            produced += no_answer_questions(
                document,
                run,
                resolver=resolver,
                ambiguous_forms=self.ambiguous_forms,
                rng=rng,
                context=context,
            )
        if self.include_paraphrases:
            produced += [
                variant for example in list(produced) for variant in paraphrase(example)
            ]

        return self._finish(produced)

    def for_passages(
        self, document: Document, run: AnnotationRun
    ) -> list[QAExample]:
        """Generate per passage, so each example carries only its own window.

        Used for long documents, where showing the whole text as context
        would bury the answer. Annotations outside a passage are skipped
        rather than asked about with the wrong context.
        """
        if not document.passages:
            return self.for_document(document, run)
        produced: list[QAExample] = []
        for passage in document.passages:
            narrowed = _restrict(run, passage)
            if len(narrowed) == 0:
                continue
            produced += self.for_document(
                document, narrowed, context=passage.span.text_in(document.text)
            )
        return self._finish(produced)

    def _finish(self, produced: Iterable[QAExample]) -> list[QAExample]:
        seen: set[tuple[str, str, str]] = set()
        out: list[QAExample] = []
        for example in produced:
            problems = example.check()
            if problems:
                raise AssertionError(
                    f"generated an unusable example: {'; '.join(problems)}\n"
                    f"  {example.question!r} -> {example.answer!r}"
                )
            key = (example.context, example.question, example.answer)
            if key in seen:
                continue
            seen.add(key)
            out.append(example)
        return out


def _restrict(run: AnnotationRun, passage: Passage) -> AnnotationRun:
    """The part of a run that falls inside one passage."""
    from dataclasses import replace as _replace

    def inside(item) -> bool:
        return passage.span.contains(item.span)

    mentions = tuple(m for m in run.mentions if inside(m))
    predicates = tuple(p for p in run.predicates if inside(p))
    properties = tuple(p for p in run.properties if inside(p))
    kept = (
        {m.mention_id for m in mentions}
        | {p.predicate_id for p in predicates}
        | {p.property_id for p in properties}
    )
    relations = tuple(
        edge
        for edge in run.relations
        if edge.source_id in kept and edge.target_id in kept
    )
    return _replace(
        run,
        mentions=mentions,
        predicates=predicates,
        properties=properties,
        relations=relations,
    )
