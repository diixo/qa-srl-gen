"""Driving the question generator over a document and its annotations.

Like the semantic generator's driver, this module exists because producing a
balanced set needs every template family at once, and putting the driver
inside any one of them would make them import each other. It is not in the
handoff's file list.

Three properties are enforced here rather than left to the caller:

**Context travels with the question.** Splitting a sentence into eight
questions is wanted; detaching them from the text that makes their answers
unambiguous is not. Every example carries its own context, and for dialogue
that context reaches back far enough to include the turns the answer depends
on.

**Nothing is emitted that cannot be checked.** Each example goes through
:meth:`~.answers.QAExample.check` before it is yielded, so a template bug
surfaces at generation time rather than inside a training run.

**The mix is controllable.** Left alone, the templates produce whatever the
record happens to support, which skews heavily towards whichever family fires
most often. :meth:`QuestionGenerator.balance` imposes a target mix
deterministically, because a corpus whose composition drifts between runs
cannot be compared with itself.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence

from ..documents import AnnotationRun, Document, Passage
from ..ontology import TypeHierarchy, load_default_hierarchy
from ..qasrl_core.inflections import InflectionLexicon
from ..qasrl_core.models import InflectedForms
from .answers import QAExample, QAKind, phrase_answer
from .negatives import no_answer_questions
from .paraphrases import paraphrase
from .templates import (
    ParadigmResolver,
    atomic_questions,
    compound_questions,
    contextual_questions,
    entity_type_questions,
    ontology_questions,
    passive_paraphrase_questions,
    polarity_questions,
    property_questions,
    yes_no_questions,
)

__all__ = ["QuestionGenerator", "dialogue_act_questions", "balance"]


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
        turn = utterance.text_in(document)
        # The context is a slice, so a document-global span would point past
        # its end. Evidence is recorded against what the example shows.
        local = utterance.span.shifted(-first.span.start_char)

        # A turn that already ends in "?" or "!" must not be quoted into a
        # question that appends another: `...dinner?"?` reads as a typo and
        # would be learned as one.
        quoted = turn.rstrip()
        if quoted and quoted[-1] in "?!.":
            quoted = quoted[:-1].rstrip()

        def example(question: str, answer: str, **meta) -> QAExample:
            return QAExample(
                context=context,
                question=question,
                answer=phrase_answer(answer),
                kind=QAKind.SPEECH_ACT,
                document_id=document.document_id,
                run_id=run.run_id,
                evidence=(local,),
                metadata={"utterance_id": utterance.utterance_id, **meta},
            )

        if annotation.speech_acts:
            examples.append(
                example(
                    f'What is the speaker doing when they say "{quoted}"?',
                    _list_phrase([_readable(a) for a in annotation.speech_acts]),
                    aspect="speech_act",
                )
            )
        if annotation.marker_functions:
            examples.append(
                example(
                    f'What does "{quoted}" express?',
                    _list_phrase([_readable(f) for f in annotation.marker_functions]),
                    aspect="marker",
                )
            )
        if annotation.stance is not None:
            examples.append(
                example(
                    f'What stance does the speaker take in "{quoted}"?',
                    _readable(annotation.stance),
                    aspect="stance",
                )
            )
        if annotation.polarity is not None:
            examples.append(
                example(
                    f'Is "{quoted}" affirmative or negative?',
                    _readable(annotation.polarity),
                    aspect="polarity",
                )
            )
        if annotation.mood is not None:
            examples.append(
                example(
                    f'Is "{quoted}" a statement, a question, or a command?',
                    _readable(annotation.mood),
                    aspect="mood",
                )
            )
    return examples


def _readable(value) -> str:
    return str(value).lower().replace("_", " ")


def _list_phrase(items: Sequence[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def balance(
    examples: Sequence[QAExample],
    *,
    seed: int = 0,
    max_per_kind: Mapping[QAKind, int] | int | None = None,
    no_answer_share: float | None = None,
    max_total: int | None = None,
) -> list[QAExample]:
    """Impose a target mix on a set of examples, deterministically.

    Selection is a seeded shuffle within each kind, so the same input and
    seed always give the same subset — a corpus whose composition moved
    between runs could not be compared with an earlier version of itself.

    ``no_answer_share`` caps unanswerable examples at that fraction of the
    result. Too few and the model never learns to decline; too many and it
    learns to decline by default, which is the more damaging of the two
    because it looks like caution.
    """
    if no_answer_share is not None and not 0.0 <= no_answer_share <= 1.0:
        raise ValueError("no_answer_share must be a fraction")

    rng = random.Random(seed)
    by_kind: dict[QAKind, list[QAExample]] = {}
    for example in examples:
        by_kind.setdefault(example.kind, []).append(example)
    for group in by_kind.values():
        rng.shuffle(group)

    def cap_for(kind: QAKind) -> int | None:
        if max_per_kind is None:
            return None
        if isinstance(max_per_kind, int):
            return max_per_kind
        return max_per_kind.get(kind)

    kept: list[QAExample] = []
    for kind in sorted(by_kind, key=lambda k: k.value):
        cap = cap_for(kind)
        group = by_kind[kind]
        kept.extend(group if cap is None else group[:cap])

    if no_answer_share is not None:
        answerable = [e for e in kept if e.answerable]
        refusals = [e for e in kept if not e.answerable]
        if no_answer_share == 0:
            allowed = 0
        elif answerable:
            # n / (n + a) = share  ->  n = a * share / (1 - share)
            allowed = (
                len(refusals)
                if no_answer_share >= 1.0
                else int(len(answerable) * no_answer_share / (1 - no_answer_share))
            )
        else:
            allowed = len(refusals)
        kept = answerable + refusals[:allowed]

    kept.sort(key=lambda e: (e.document_id, e.kind.value, e.question))
    if max_total is not None:
        kept = kept[:max_total]
    return kept


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
        produced += compound_questions(document, run, resolver=resolver, context=context)
        # Preserve native bank questions when semantic roles are unknown.
        untyped = {p.predicate_id for p in run.predicates if p.predicate_type is None}
        mentions = {m.mention_id: m for m in run.mentions}
        native_answers = {}
        for edge in run.relations:
            mention = mentions.get(edge.source_id)
            if edge.target_id in untyped and edge.question and mention is not None:
                native_answers.setdefault((edge.target_id, edge.question), {})[mention.span] = mention
        # The bridge stores alternatives in descending vote order. Select the
        # same primary answer as bank_qa_examples, retaining the others as
        # alternatives rather than treating them as contradictory examples.
        for (predicate_id, question), by_span in native_answers.items():
            primary, *alternatives = by_span.values()
            produced.append(QAExample(
                context=context if context is not None else document.text,
                question=question, answer=phrase_answer(primary.exact_text),
                kind=QAKind.ATOMIC, document_id=document.document_id,
                run_id=run.run_id, evidence=(primary.span,),
                metadata={"predicate_id": predicate_id,
                          "alternative_answers": [m.exact_text for m in alternatives]},
            ))
        produced += entity_type_questions(document, run, context=context)
        produced += ontology_questions(document, run, hierarchy, context=context)
        produced += property_questions(document, run, hierarchy, context=context)
        produced += yes_no_questions(document, run, resolver=resolver, context=context)
        produced += polarity_questions(document, run, resolver=resolver, context=context)
        produced += dialogue_act_questions(document, run)
        if context is None:
            # Cross-sentence examples carry their own, wider context, so they
            # make no sense inside a single-passage window.
            produced += contextual_questions(document, run)

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
            produced += passive_paraphrase_questions(
                document, run, resolver=resolver, context=context
            )
            produced += [
                variant for example in list(produced) for variant in paraphrase(example)
            ]

        return self._finish(produced)

    def for_passages(self, document: Document, run: AnnotationRun) -> list[QAExample]:
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
            offset = -passage.span.start_char
            utterances = tuple(replace(u, span=u.span.shifted(offset))
                               for u in document.utterances if passage.span.contains(u.span))
            local_document = replace(document, text=passage.span.text_in(document.text),
                                     passages=(), utterances=utterances)
            utterance_ids = {u.utterance_id for u in utterances}
            narrowed = replace(
                narrowed,
                mentions=tuple(replace(m, span=m.span.shifted(offset)) for m in narrowed.mentions),
                predicates=tuple(replace(p, span=p.span.shifted(offset)) for p in narrowed.predicates),
                properties=tuple(replace(p, span=p.span.shifted(offset)) for p in narrowed.properties),
                dialogue=tuple(d for d in narrowed.dialogue if d.utterance_id in utterance_ids),
            )
            if len(narrowed) == 0:
                continue
            produced += self.for_document(
                local_document, narrowed, context=local_document.text
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
        # Identical wording can refer to different events in a document.
        # Do not turn that ambiguity into contradictory single-answer targets.
        answers: dict[tuple[str, str], set[tuple[bool, str]]] = {}
        for e in out:
            answers.setdefault((e.context, e.question), set()).add(
                (e.answerable, e.answer if e.answerable else "")
            )
        return [e for e in out if len(answers[e.context, e.question]) == 1]


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
    properties = tuple(p for p in properties if p.target_id is None or p.target_id in kept)
    kept = {m.mention_id for m in mentions} | {p.predicate_id for p in predicates} | {p.property_id for p in properties}
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
