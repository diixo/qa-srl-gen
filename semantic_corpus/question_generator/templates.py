"""Building questions from an annotation run.

Input is always a :class:`~..documents.Document` plus an
:class:`~..documents.AnnotationRun`, whether the facts were generated or
annotated. The generator never invents ground truth: every answer here is a
span somebody already recorded, which is the separation the handoff insists
on between the question generator and the semantic generator.

Verb forms are not improvised. A question like *What did Anna give Rex?*
needs do-support and a bare stem, while *Who gave Rex a ball?* needs a finite
verb and no auxiliary at all — and which applies depends on whether the
questioned argument is the subject. That is exactly what
:class:`~..qasrl_core.frame.Frame` computes, and it is reused here rather
than approximated, so a question about a passive perfect comes out right for
the same reason the bank's questions do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..documents import (
    AnnotationRun,
    Document,
    EntityMention,
    Predicate,
    Property,
    RelationEdge,
    TextSpan,
)
from ..ontology import Label, Relation, TypeHierarchy, load_default_hierarchy
from ..qasrl_core.frame import ArgStructure, Frame as VerbFrame
from ..qasrl_core.inflections import InflectionLexicon
from ..qasrl_core.models import InflectedForms
from .answers import QAExample, QAKind, phrase_answer, sentence_case

__all__ = [
    "ArgumentView",
    "ParadigmResolver",
    "regular_forms",
    "atomic_questions",
    "entity_type_questions",
    "ontology_questions",
    "property_questions",
    "yes_no_questions",
    "compound_questions",
    "AGENT_RELATIONS",
    "THEME_RELATIONS",
]

#: Relations that put their mention in subject position.
AGENT_RELATIONS: frozenset[Relation] = frozenset(
    {Relation.AGENT_OF, Relation.EXPERIENCER_OF}
)
#: Relations that put their mention in direct-object position. ``STATE_OF``
#: belongs here because the state a subject is in is realised as its object:
#: *the cat felt sadness*.
THEME_RELATIONS: frozenset[Relation] = frozenset(
    {Relation.THEME_OF, Relation.PATIENT_OF, Relation.STATE_OF}
)

_ADJUNCT_PREPOSITION = {Relation.LOCATION_OF: "in", Relation.TIME_OF: "on"}
_WH_FOR_ADJUNCT = {Relation.LOCATION_OF: "Where", Relation.TIME_OF: "When"}
#: Labels whose mentions are asked about with "who" rather than "what".
_ANIMATE_LABELS = frozenset({Label.PERSON, Label.ANIMAL, Label.ORGANIZATION})


def _inline(mention: EntityMention | None) -> str:
    """A mention as it should read inside a sentence.

    A span lifted from the start of a sentence carries a capital that does
    not belong mid-question: *What did The woman give?* A name keeps its
    capital; a common noun loses it.
    """
    if mention is None:
        return ""
    text = mention.exact_text
    if Label.NAMED_ENTITY in mention.labels or Label.NAMED_OBJECT in mention.labels:
        return text
    return text[:1].lower() + text[1:]


def _article(word: str) -> str:
    """``a`` or ``an``, by the sound the word starts with."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def _wh_for(mention: EntityMention) -> str:
    return "Who" if mention.labels.labels & _ANIMATE_LABELS else "What"


def _preposition_before(document: Document, span: TextSpan) -> str | None:
    """The preposition the text itself puts in front of a span.

    Reading it off the source beats guessing from the relation: the sentence
    says *before dawn*, and a template that assumed *on* would answer
    *When...?* with *On dawn.*
    """
    from ..qasrl_core.state_machine import PREPOSITIONS

    before = document.text[: span.start_char].rstrip()
    if not before:
        return None
    last = before.split()[-1].strip(",;:").lower()
    return last if last in PREPOSITIONS else None


def regular_forms(lemma: str) -> InflectedForms:
    """Regular English inflection, used when no paradigm is known.

    A fallback, not a model of English morphology: it exists so that an
    unknown lemma produces a slightly wrong question rather than no question
    and a crash. Irregular verbs must come from a real paradigm.
    """
    stem = lemma
    if stem.endswith(("s", "x", "z", "ch", "sh")):
        third = stem + "es"
    elif stem.endswith("y") and len(stem) > 1 and stem[-2] not in "aeiou":
        third = stem[:-1] + "ies"
    else:
        third = stem + "s"

    if stem.endswith("e") and not stem.endswith(("ee", "oe", "ye")):
        progressive = stem[:-1] + "ing"
        past = stem + "d"
    elif stem.endswith("y") and len(stem) > 1 and stem[-2] not in "aeiou":
        progressive = stem + "ing"
        past = stem[:-1] + "ied"
    else:
        progressive = stem + "ing"
        past = stem + "ed"
    return InflectedForms(stem, third, progressive, past, past)


@dataclass(frozen=True, slots=True)
class ParadigmResolver:
    """Finds the inflected forms of a lemma, in order of trustworthiness."""

    known: Mapping[str, InflectedForms] = field(default_factory=dict)
    lexicon: InflectionLexicon | None = None

    def __call__(self, lemma: str) -> InflectedForms:
        if lemma in self.known:
            return self.known[lemma]
        if self.lexicon is not None:
            paradigm = self.lexicon.paradigm(lemma)
            if paradigm is not None:
                return paradigm
        return regular_forms(lemma)


@dataclass(frozen=True, slots=True)
class ArgumentView:
    """The arguments of one predicate, grouped by the role they play."""

    predicate: Predicate
    by_relation: Mapping[Relation, tuple[EntityMention, ...]] = field(
        default_factory=dict
    )
    properties: tuple[Property, ...] = ()

    def first(self, *relations: Relation) -> EntityMention | None:
        for relation in relations:
            found = self.by_relation.get(relation)
            if found:
                return found[0]
        return None

    @property
    def agent(self) -> EntityMention | None:
        return self.first(*sorted(AGENT_RELATIONS, key=lambda r: r.value))

    @property
    def theme(self) -> EntityMention | None:
        return self.first(*sorted(THEME_RELATIONS, key=lambda r: r.value))

    @property
    def recipient(self) -> EntityMention | None:
        return self.first(Relation.RECIPIENT_OF)

    def adjuncts(self) -> tuple[tuple[Relation, EntityMention], ...]:
        out = []
        for relation in (Relation.LOCATION_OF, Relation.TIME_OF):
            for mention in self.by_relation.get(relation, ()):
                out.append((relation, mention))
        return tuple(out)


def build_views(run: AnnotationRun) -> tuple[ArgumentView, ...]:
    """Group a run's relations around the predicates they point at."""
    mentions = {m.mention_id: m for m in run.mentions}
    grouped: dict[str, dict[Relation, list[EntityMention]]] = {}
    for edge in run.relations:
        if edge.relation is Relation.INSTANCE_OF:
            continue
        mention = mentions.get(edge.source_id)
        if mention is None:
            continue
        grouped.setdefault(edge.target_id, {}).setdefault(edge.relation, []).append(
            mention
        )
    views = []
    for predicate in run.predicates:
        by_relation = {
            relation: tuple(found)
            for relation, found in grouped.get(predicate.predicate_id, {}).items()
        }
        views.append(
            ArgumentView(
                predicate=predicate,
                by_relation=by_relation,
                properties=tuple(run.properties),
            )
        )
    return tuple(views)


def _verb_words(
    predicate: Predicate, resolver: ParadigmResolver, *, subject_present: bool
) -> list[str]:
    """The verb sequence for a question, with do-support where English needs it."""
    aspect = predicate.aspect or ""
    frame = VerbFrame(
        verb_inflected_forms=resolver(predicate.lemma),
        structure=ArgStructure(args={}, is_passive=predicate.voice == "passive"),
        tense=predicate.tense if predicate.tense in _TENSES else "past",
        is_perfect="perfect" in aspect,
        is_progressive="progressive" in aspect,
        is_negated=str(predicate.polarity) == "NEGATIVE",
    )
    stack = frame.get_verb_stack()
    if subject_present:
        stack = frame.split_verb_stack_if_necessary(stack)
    return stack


_TENSES = frozenset(
    {"present", "past", "can", "will", "might", "should", "would"}
)


def active_verb_words(
    predicate: Predicate, resolver: ParadigmResolver
) -> list[str]:
    """The verb as it would read in the active voice, subject questioned.

    Needed to ask about an agent the passive dropped: the record says
    ``given``, but the question has to say *Who gave ...?*
    """
    from dataclasses import replace as _replace

    return _verb_words(
        _replace(predicate, voice="active"), resolver, subject_present=False
    )


def _join(parts: Iterable[str]) -> str:
    return " ".join(part for part in parts if part)


def _question(parts: Iterable[str]) -> str:
    return sentence_case(_join(parts).strip()) + "?"


def atomic_questions(
    document: Document,
    run: AnnotationRun,
    *,
    resolver: ParadigmResolver | None = None,
    context: str | None = None,
) -> list[QAExample]:
    """One question per argument of each predicate.

    The questioned argument is left out of the question and every other
    known argument is spelled into it, so the question identifies which
    event it is about. Splitting one sentence into six questions is the
    point; dropping the context that disambiguates them is not.
    """
    resolver = resolver or ParadigmResolver()
    body = context if context is not None else document.text
    examples: list[QAExample] = []

    for view in build_views(run):
        predicate = view.predicate
        agent, theme, recipient = view.agent, view.theme, view.recipient

        def example(question: str, mention: EntityMention, kind=QAKind.ATOMIC, **extra):
            return QAExample(
                context=body,
                question=question,
                answer=phrase_answer(mention.exact_text, extra.pop("preposition", None)),
                kind=kind,
                document_id=document.document_id,
                run_id=run.run_id,
                evidence=(mention.span,),
                metadata={
                    "predicate": predicate.lemma,
                    "relation": extra.pop("relation", None),
                    **extra,
                },
            )

        # The subject is the gap: no auxiliary is fronted, the verb stays finite.
        if agent is not None:
            words = _verb_words(predicate, resolver, subject_present=False)
            wh = _wh_for(agent)
            tail = [_inline(theme), _to_phrase(recipient)]
            examples.append(
                example(
                    _question([wh, *words, *tail]),
                    agent,
                    relation="AGENT_OF",
                )
            )

        # Any other argument: the subject is spelled out, so do-support appears.
        if theme is not None and agent is not None:
            words = _verb_words(predicate, resolver, subject_present=True)
            aux, rest = words[0], words[1:]
            wh = _wh_for(theme)
            examples.append(
                example(
                    _question([wh, aux, _inline(agent), *rest, _to_phrase(recipient)]),
                    theme,
                    relation="THEME_OF",
                )
            )

        if recipient is not None and agent is not None:
            words = _verb_words(predicate, resolver, subject_present=True)
            aux, rest = words[0], words[1:]
            wh = _wh_for(recipient)
            examples.append(
                example(
                    _question([wh, aux, _inline(agent), *rest, _inline(theme), "to"]),
                    recipient,
                    relation="RECIPIENT_OF",
                )
            )

        for relation, mention in view.adjuncts():
            wh = _WH_FOR_ADJUNCT[relation]
            if agent is not None:
                words = _verb_words(predicate, resolver, subject_present=True)
                aux, rest = words[0], words[1:]
                question = _question([wh, aux, _inline(agent), *rest, _inline(theme)])
            else:
                question = f"{wh} did this happen?"
            examples.append(
                example(
                    question,
                    mention,
                    relation=str(relation),
                    preposition=_preposition_before(document, mention.span)
                    or _ADJUNCT_PREPOSITION[relation],
                )
            )

        # "What did Anna do?" — the whole event, answered by the clause.
        if agent is not None:
            words = _verb_words(predicate, resolver, subject_present=True)
            aux = words[0]
            answer_parts = [
                _inline(agent),
                predicate.exact_text,
                _inline(theme),
                _to_phrase(recipient),
            ]
            examples.append(
                QAExample(
                    context=body,
                    question=_question(["What", aux, _inline(agent), "do"]),
                    answer=phrase_answer(_join(answer_parts)),
                    kind=QAKind.ATOMIC,
                    document_id=document.document_id,
                    run_id=run.run_id,
                    evidence=(predicate.span,),
                    metadata={"predicate": predicate.lemma, "relation": "EVENT"},
                )
            )

    return examples


def _to_phrase(recipient: EntityMention | None) -> str:
    return f"to {_inline(recipient)}" if recipient is not None else ""


def entity_type_questions(
    document: Document, run: AnnotationRun, *, context: str | None = None
) -> list[QAExample]:
    """Ask what kind of thing a mention is.

    The answer comes from the recorded labels, so a name the corpus uses in
    two senses gets two different correct answers in two different contexts.
    That is the whole point of ``ambiguous_names_test``.
    """
    body = context if context is not None else document.text
    examples: list[QAExample] = []
    for mention in run.mentions:
        entity_label = next(iter(mention.labels.entity_labels), None)
        if entity_label is None:
            continue
        examples.append(
            QAExample(
                context=body,
                question=f"What kind of entity is {_inline(mention)}?",
                answer=phrase_answer(_readable(entity_label)),
                kind=QAKind.ENTITY_TYPE,
                document_id=document.document_id,
                run_id=run.run_id,
                evidence=(mention.span,),
                metadata={"label": str(entity_label)},
            )
        )
    return examples


def ontology_questions(
    document: Document,
    run: AnnotationRun,
    hierarchy: TypeHierarchy | None = None,
    *,
    context: str | None = None,
) -> list[QAExample]:
    """Ask questions whose answer follows from the type hierarchy.

    *Paris is a city; every city is a location; therefore Paris is a
    location.* The answer states the inference rather than just asserting the
    conclusion, so the example teaches the step and not the fact.
    """
    hierarchy = hierarchy or load_default_hierarchy()
    body = context if context is not None else document.text
    examples: list[QAExample] = []
    for mention in run.mentions:
        type_name = mention.normalized_form
        if not type_name or type_name not in hierarchy:
            continue
        ancestors = hierarchy.ancestors(type_name)
        if len(ancestors) < 2:
            continue
        parent = ancestors[1]
        top = ancestors[-1]
        if parent == top:
            continue
        top_words = _readable(top)
        subject = sentence_case(_inline(mention))
        examples.append(
            QAExample(
                context=body,
                question=f"Is {_inline(mention)} {_article(top_words)} {top_words}?",
                answer=(
                    f"Yes. {subject} is {_article(type_name)} {type_name}, "
                    f"and every {type_name} is {_article(top_words)} {top_words}."
                ),
                kind=QAKind.ONTOLOGY,
                document_id=document.document_id,
                run_id=run.run_id,
                evidence=(mention.span,),
                metadata={"type": type_name, "supertype": str(top)},
            )
        )
    return examples


def property_questions(
    document: Document,
    run: AnnotationRun,
    hierarchy: TypeHierarchy | None = None,
    *,
    context: str | None = None,
) -> list[QAExample]:
    """Ask about recorded properties, states and emotions."""
    hierarchy = hierarchy or load_default_hierarchy()
    body = context if context is not None else document.text
    mentions = {m.mention_id: m for m in run.mentions}
    examples: list[QAExample] = []
    for prop in run.properties:
        target = mentions.get(prop.target_id) if prop.target_id else None
        kinds = hierarchy.candidates(prop.head.lower())
        kind_word = kinds[0] if kinds else "property"
        subject = _inline(target) if target else "it"
        examples.append(
            QAExample(
                context=body,
                question=f"What {kind_word} was {subject}?",
                answer=phrase_answer(prop.exact_text),
                kind=QAKind.PROPERTY,
                document_id=document.document_id,
                run_id=run.run_id,
                evidence=(prop.span,),
                metadata={"head": prop.head, "degree": prop.degree, "negated": prop.negated},
            )
        )
    return examples


def yes_no_questions(
    document: Document,
    run: AnnotationRun,
    *,
    resolver: ParadigmResolver | None = None,
    context: str | None = None,
) -> list[QAExample]:
    """Polar questions whose answer carries its justification.

    A bare *Yes.* teaches nothing checkable, so the answer repeats the fact
    it rests on.
    """
    resolver = resolver or ParadigmResolver()
    body = context if context is not None else document.text
    examples: list[QAExample] = []
    for view in build_views(run):
        agent, theme = view.agent, view.theme
        if agent is None:
            continue
        words = _verb_words(view.predicate, resolver, subject_present=True)
        aux, rest = words[0], words[1:]
        question = _question([aux, _inline(agent), *rest, _inline(theme)])
        clause = _join([_inline(agent), view.predicate.exact_text, _inline(theme)])
        answer = f"Yes. {sentence_case(clause)}."
        examples.append(
            QAExample(
                context=body,
                question=question,
                answer=sentence_case(answer),
                kind=QAKind.YES_NO,
                document_id=document.document_id,
                run_id=run.run_id,
                evidence=(view.predicate.span,),
                metadata={"predicate": view.predicate.lemma},
            )
        )
    return examples


def compound_questions(
    document: Document,
    run: AnnotationRun,
    *,
    context: str | None = None,
) -> list[QAExample]:
    """One question covering several arguments at once."""
    body = context if context is not None else document.text
    examples: list[QAExample] = []
    for view in build_views(run):
        agent, theme = view.agent, view.theme
        adjuncts = view.adjuncts()
        if agent is None or theme is None or not adjuncts:
            continue
        relation, adjunct = adjuncts[0]
        wh = _WH_FOR_ADJUNCT[relation].lower()
        examples.append(
            QAExample(
                context=body,
                question=(
                    f"Who {view.predicate.exact_text} {_inline(theme)}, "
                    f"and {wh} did it happen?"
                ),
                answer=phrase_answer(
                    f"{sentence_case(_inline(agent))}, "
                    f"{_preposition_before(document, adjunct.span) or _ADJUNCT_PREPOSITION[relation]} "
                    f"{_inline(adjunct)}"
                ),
                kind=QAKind.COMPOUND,
                document_id=document.document_id,
                run_id=run.run_id,
                evidence=(agent.span, adjunct.span),
                metadata={"predicate": view.predicate.lemma},
            )
        )
    return examples


def _readable(label: Label) -> str:
    """``PHYSICAL_OBJECT`` as a phrase a question can contain."""
    return str(label).lower().replace("_", " ")
