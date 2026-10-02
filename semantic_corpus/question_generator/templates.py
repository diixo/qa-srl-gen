"""Building questions from an annotation run.

Input is always a :class:`~..documents.Document` plus an
:class:`~..documents.AnnotationRun`, whether the facts were generated or
annotated. The generator never invents ground truth: every answer here is a
span somebody already recorded, which is the separation the handoff insists
on between the question generator and the semantic generator.

Three things are read off the text rather than guessed, because guessing
them produces questions whose stored answers are wrong:

*the verb chain*
    *What did Anna give Rex?* needs do-support and a bare stem, while *Who
    gave Rex a ball?* needs a finite verb and no auxiliary. Which applies
    depends on whether the questioned argument is the subject, and that is
    what :class:`~..qasrl_core.frame.Frame` already computes.

*whether an argument is an adjunct*
    A locative role realised as a bare object (*visited Vustal*) is not an
    adjunct, and answering *Where...?* with *In Vustal.* would invent a
    preposition the text does not contain. The preposition in front of a
    span decides this, not the name of the relation.

*polarity*
    A negated clause cannot be answered *Yes.* The polar question is asked
    in the affirmative and answered *No.*, with the negated clause as its
    justification. For the same reason a clause is rebuilt from the verb
    chain rather than from the predicate's surface token, which in *Yavin
    didn't visit Vustal* is only the bare stem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from ..documents import (
    AnnotationRun,
    Document,
    EntityMention,
    Predicate,
    Property,
    TextSpan,
)
from ..ontology import Label, Relation, TypeHierarchy, load_default_hierarchy
from ..qasrl_core.frame import ArgStructure, Frame as VerbFrame
from ..qasrl_core.inflections import InflectionLexicon
from ..qasrl_core.models import InflectedForms
from ..qasrl_core.state_machine import PREPOSITIONS
from .answers import QAExample, QAKind, phrase_answer, sentence_case

__all__ = [
    "ArgumentView",
    "ParadigmResolver",
    "regular_forms",
    "build_views",
    "active_verb_words",
    "preposition_before",
    "atomic_questions",
    "entity_type_questions",
    "contextual_questions",
    "ontology_questions",
    "property_questions",
    "yes_no_questions",
    "polarity_questions",
    "compound_questions",
    "passive_paraphrase_questions",
    "AGENT_RELATIONS",
    "THEME_RELATIONS",
    "ADJUNCT_RELATIONS",
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
#: Relations that *may* be adjuncts. Whether they are is decided per mention
#: from the text, not from the relation.
ADJUNCT_RELATIONS: frozenset[Relation] = frozenset(
    {Relation.LOCATION_OF, Relation.TIME_OF}
)

_WH_FOR_ADJUNCT = {Relation.LOCATION_OF: "Where", Relation.TIME_OF: "When"}
#: Labels whose mentions are asked about with "who" rather than "what".
_ANIMATE_LABELS = frozenset({Label.PERSON, Label.ANIMAL, Label.ORGANIZATION})
_TENSES = frozenset({"present", "past", "can", "will", "might", "should", "would"})


# ---------------------------------------------------------------------------
# Wording helpers
# ---------------------------------------------------------------------------


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


def _readable(label: Label) -> str:
    """``PHYSICAL_OBJECT`` as a phrase a question can contain."""
    return str(label).lower().replace("_", " ")


def preposition_before(document: Document, span: TextSpan) -> str | None:
    """The preposition the text puts in front of a span, if any.

    Reading it off the source beats guessing from the relation twice over:
    the sentence says *before dawn*, not *on dawn*, and *visited Vustal* has
    no preposition at all, so nothing should be invented for it.
    """
    before = document.text[: span.start_char].rstrip()
    if not before:
        return None
    last = before.split()[-1].strip(",;:").lower()
    return last if last in PREPOSITIONS else None


def _join(parts: Iterable[str]) -> str:
    return " ".join(part for part in parts if part)


def _question(parts: Iterable[str]) -> str:
    return sentence_case(_join(parts).strip()) + "?"


# ---------------------------------------------------------------------------
# Verb forms
# ---------------------------------------------------------------------------


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


def _verb_words(
    predicate: Predicate,
    resolver: ParadigmResolver,
    *,
    subject_present: bool,
    negated: bool | None = None,
    passive: bool | None = None,
) -> list[str]:
    """The verb sequence, with do-support where English needs it.

    *negated* and *passive* override what the record says, which is how a
    polar question about a negated clause gets asked in the affirmative, and
    how a passive paraphrase is built from an active record.
    """
    aspect = predicate.aspect or ""
    is_passive = predicate.voice == "passive" if passive is None else passive
    is_negated = str(predicate.polarity) == "NEGATIVE" if negated is None else negated
    frame = VerbFrame(
        verb_inflected_forms=resolver(predicate.lemma),
        structure=ArgStructure(args={}, is_passive=is_passive),
        tense=predicate.tense if predicate.tense in _TENSES else "past",
        is_perfect="perfect" in aspect,
        is_progressive="progressive" in aspect,
        is_negated=is_negated,
    )
    stack = frame.get_verb_stack()
    if subject_present:
        stack = frame.split_verb_stack_if_necessary(stack)
    return stack


def active_verb_words(predicate: Predicate, resolver: ParadigmResolver) -> list[str]:
    """The verb as it would read in the active voice, subject questioned.

    Needed to ask about an agent the passive dropped: the record says
    ``given``, but the question has to say *Who gave ...?*
    """
    return _verb_words(predicate, resolver, subject_present=False, passive=False)


# ---------------------------------------------------------------------------
# Grouping arguments around a predicate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArgumentView:
    """The arguments of one predicate, grouped by the role they play.

    ``adjunct_prepositions`` records what the text puts in front of each
    candidate adjunct. A role in :data:`ADJUNCT_RELATIONS` with no
    preposition is a core object, not an adjunct — compare *visited Vustal*
    with *moved to Vustal*.
    """

    predicate: Predicate
    by_relation: Mapping[Relation, tuple[EntityMention, ...]] = field(
        default_factory=dict
    )
    adjunct_prepositions: Mapping[str, str] = field(default_factory=dict)
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
        direct = self.first(*sorted(THEME_RELATIONS, key=lambda r: r.value))
        if direct is not None:
            return direct
        # A bare locative object (*visited Vustal*) behaves as the object.
        for relation in sorted(ADJUNCT_RELATIONS, key=lambda r: r.value):
            for mention in self.by_relation.get(relation, ()):
                if mention.mention_id not in self.adjunct_prepositions:
                    return mention
        return None

    @property
    def recipient(self) -> EntityMention | None:
        return self.first(Relation.RECIPIENT_OF)

    def adjuncts(self) -> tuple[tuple[Relation, EntityMention, str], ...]:
        """Adjuncts proper, each with the preposition the text uses."""
        out = []
        for relation in (Relation.LOCATION_OF, Relation.TIME_OF):
            for mention in self.by_relation.get(relation, ()):
                preposition = self.adjunct_prepositions.get(mention.mention_id)
                if preposition is not None:
                    out.append((relation, mention, preposition))
        return tuple(out)

    @property
    def is_negated(self) -> bool:
        return str(self.predicate.polarity) == "NEGATIVE"


def build_views(
    run: AnnotationRun, document: Document | None = None
) -> tuple[ArgumentView, ...]:
    """Group a run's relations around the predicates they point at.

    *document* is needed to tell an adjunct from a bare object. Without it
    every candidate adjunct is treated as one, which is only right when the
    text really does use prepositions.
    """
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
        prepositions: dict[str, str] = {}
        for relation in ADJUNCT_RELATIONS:
            for mention in by_relation.get(relation, ()):
                found = (
                    preposition_before(document, mention.span)
                    if document is not None
                    else "in"
                )
                if found:
                    prepositions[mention.mention_id] = found
        views.append(
            ArgumentView(
                predicate=predicate,
                by_relation=by_relation,
                adjunct_prepositions=prepositions,
                properties=tuple(run.properties),
            )
        )
    return tuple(views)


def _to_phrase(recipient: EntityMention | None) -> str:
    return f"to {_inline(recipient)}" if recipient is not None else ""


def _clause(view: ArgumentView, resolver: ParadigmResolver) -> str:
    """The declarative clause, rebuilt from the verb chain.

    The predicate's surface token is not enough: in *Yavin didn't visit
    Vustal* it is the bare stem ``visit``, and reusing it would produce the
    answer *Yavin visit Vustal.*
    """
    words = _verb_words(view.predicate, resolver, subject_present=False)
    parts = [
        _inline(view.agent),
        *words,
        _inline(view.theme),
        _to_phrase(view.recipient),
    ]
    for _, mention, preposition in view.adjuncts():
        parts.append(f"{preposition} {_inline(mention)}")
    return _join(parts)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


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

    for view in build_views(run, document):
        predicate = view.predicate
        agent, theme, recipient = view.agent, view.theme, view.recipient

        def example(question: str, mention: EntityMention, **extra) -> QAExample:
            return QAExample(
                context=body,
                question=question,
                answer=phrase_answer(mention.exact_text, extra.pop("preposition", None)),
                kind=QAKind.ATOMIC,
                document_id=document.document_id,
                run_id=run.run_id,
                evidence=(mention.span,),
                metadata={"predicate": predicate.lemma, **extra},
            )

        # The subject is the gap: no auxiliary is fronted, the verb stays finite.
        if agent is not None:
            words = _verb_words(predicate, resolver, subject_present=False)
            examples.append(
                example(
                    _question(
                        [_wh_for(agent), *words, _inline(theme), _to_phrase(recipient)]
                    ),
                    agent,
                    relation="AGENT_OF",
                )
            )

        # Any other argument: the subject is spelled out, so do-support appears.
        if theme is not None and agent is not None:
            aux, *rest = _verb_words(predicate, resolver, subject_present=True)
            examples.append(
                example(
                    _question(
                        [
                            _wh_for(theme),
                            aux,
                            _inline(agent),
                            *rest,
                            _to_phrase(recipient),
                        ]
                    ),
                    theme,
                    relation="THEME_OF",
                )
            )

        if recipient is not None and agent is not None:
            aux, *rest = _verb_words(predicate, resolver, subject_present=True)
            examples.append(
                example(
                    _question(
                        [
                            _wh_for(recipient),
                            aux,
                            _inline(agent),
                            *rest,
                            _inline(theme),
                            "to",
                        ]
                    ),
                    recipient,
                    relation="RECIPIENT_OF",
                )
            )

        for relation, mention, preposition in view.adjuncts():
            wh = _WH_FOR_ADJUNCT[relation]
            if agent is not None:
                aux, *rest = _verb_words(predicate, resolver, subject_present=True)
                question = _question([wh, aux, _inline(agent), *rest, _inline(theme)])
            else:
                question = f"{wh} did this happen?"
            examples.append(
                example(
                    question, mention, relation=str(relation), preposition=preposition
                )
            )

        # "What did Anna do?" — the whole event, answered by the clause.
        if agent is not None:
            aux = _verb_words(predicate, resolver, subject_present=True)[0]
            examples.append(
                QAExample(
                    context=body,
                    question=_question(["What", aux, _inline(agent), "do"]),
                    answer=phrase_answer(_clause(view, resolver)),
                    kind=QAKind.ATOMIC,
                    document_id=document.document_id,
                    run_id=run.run_id,
                    evidence=(predicate.span,),
                    metadata={"predicate": predicate.lemma, "relation": "EVENT"},
                )
            )

    return examples


def passive_paraphrase_questions(
    document: Document,
    run: AnnotationRun,
    *,
    resolver: ParadigmResolver | None = None,
    context: str | None = None,
) -> list[QAExample]:
    """Ask the same relation in the other voice.

    This is the paraphrase a regular expression cannot produce: turning
    *What did Anna give to Rex?* into *What was given to Rex by Anna?* means
    rebuilding the verb chain, not rewriting words. The answer is unchanged,
    which is what makes it a paraphrase rather than a different question.
    """
    resolver = resolver or ParadigmResolver()
    body = context if context is not None else document.text
    examples: list[QAExample] = []

    for view in build_views(run, document):
        agent, theme, recipient = view.agent, view.theme, view.recipient
        if agent is None or theme is None or view.predicate.voice == "passive":
            continue

        # Theme questioned: it is the passive subject, so it is the gap.
        words = _verb_words(view.predicate, resolver, subject_present=False, passive=True)
        examples.append(
            QAExample(
                context=body,
                question=_question(
                    [_wh_for(theme), *words, _to_phrase(recipient), f"by {_inline(agent)}"]
                ),
                answer=phrase_answer(theme.exact_text),
                kind=QAKind.PARAPHRASE,
                document_id=document.document_id,
                run_id=run.run_id,
                evidence=(theme.span,),
                metadata={"predicate": view.predicate.lemma, "voice": "passive"},
            )
        )

        # Agent questioned: the passive subject is spelled out.
        aux, *rest = _verb_words(
            view.predicate, resolver, subject_present=True, passive=True
        )
        examples.append(
            QAExample(
                context=body,
                question=_question(
                    ["By whom", aux, _inline(theme), *rest, _to_phrase(recipient)]
                ),
                answer=phrase_answer(agent.exact_text),
                kind=QAKind.PARAPHRASE,
                document_id=document.document_id,
                run_id=run.run_id,
                evidence=(agent.span,),
                metadata={"predicate": view.predicate.lemma, "voice": "passive"},
            )
        )
    return examples


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


def contextual_questions(document: Document, run: AnnotationRun) -> list[QAExample]:
    """Questions whose answer needs an earlier sentence.

    A name introduced with a gloss in one sentence and used bare in a later
    one can only be typed by reading both. The context therefore runs from
    the gloss to the later mention — the one case where *not* splitting the
    context is the entire point of the example.
    """
    mentions_by_id = {m.mention_id: m for m in run.mentions}
    has_gloss = {
        edge.source_id
        for edge in run.relations
        if edge.relation is Relation.INSTANCE_OF
    }
    introductions: dict[str, EntityMention] = {}
    for mention_id in sorted(has_gloss):
        subject = mentions_by_id.get(mention_id)
        if subject is not None:
            current = introductions.get(subject.exact_text)
            if current is None or subject.span.start_char < current.span.start_char:
                introductions[subject.exact_text] = subject

    examples: list[QAExample] = []
    for mention in run.mentions:
        introduced = introductions.get(mention.exact_text)
        if introduced is None or introduced.mention_id == mention.mention_id:
            continue
        if mention.span.start_char <= introduced.span.start_char:
            continue
        if mention.mention_id in has_gloss:
            # This mention explains itself, so the earlier sentence is not
            # needed and the example would not be testing context at all.
            continue
        label = next(iter(introduced.labels.entity_labels), None)
        later_label = next(iter(mention.labels.entity_labels), None)
        if label is None or later_label != label:
            # The record disagrees with itself about this name; asking would
            # produce an answer the text contradicts.
            continue
        start = document.text.rfind(".", 0, introduced.span.start_char) + 1
        end = document.text.find(".", mention.span.end_char)
        end = len(document.text) if end < 0 else end + 1
        context = document.text[start:end].strip()
        if context.count(".") < 2:
            continue  # it does not actually span more than one sentence
        examples.append(
            QAExample(
                context=context,
                question=f"What kind of entity is {_inline(mention)}?",
                answer=phrase_answer(_readable(label)),
                kind=QAKind.CONTEXTUAL,
                document_id=document.document_id,
                run_id=run.run_id,
                metadata={"label": str(label), "introduced_at": introduced.mention_id},
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
    location.* The answer states the inference rather than just asserting
    the conclusion, so the example teaches the step and not the fact.
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
        top = ancestors[-1]
        if ancestors[1] == top:
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


def _without_property(text: str, head: str, degree: str | None) -> str:
    """The target phrase with its own adjective removed.

    Otherwise the question gives its answer away: *What size was a very
    large ball?* already says *large*. Asking about *the ball* does not.
    """
    words = [
        word
        for word in text.split()
        if word.lower() not in {head.lower(), (degree or "").lower()}
    ]
    if not words:
        return text
    if words[0].lower() in {"a", "an"}:
        words[0] = "the"
    return " ".join(words)


def _degree_answer(head: str, degree: str | None, negated: bool) -> str:
    if negated and degree:
        return f"Not {degree} {head}"
    if negated:
        return f"Not {head}"
    return f"{degree} {head}" if degree else head


def property_questions(
    document: Document,
    run: AnnotationRun,
    hierarchy: TypeHierarchy | None = None,
    *,
    context: str | None = None,
) -> list[QAExample]:
    """Ask about recorded properties, states and emotions.

    A property is stored decomposed, so the degree can be asked about
    separately: *not very tall* answers *To what degree...?* with *Not very
    tall*, which an opaque string could not support.
    """
    hierarchy = hierarchy or load_default_hierarchy()
    body = context if context is not None else document.text
    mentions = {m.mention_id: m for m in run.mentions}
    examples: list[QAExample] = []
    for prop in run.properties:
        target = mentions.get(prop.target_id) if prop.target_id else None
        kinds = hierarchy.candidates(prop.head.lower())
        kind_word = kinds[0] if kinds else "property"
        subject = _inline(target) if target else "it"
        if target is not None and prop.exact_text in target.exact_text:
            subject = _without_property(subject, prop.head, prop.degree)
        examples.append(
            QAExample(
                context=body,
                question=f"What {kind_word} was {subject}?",
                answer=phrase_answer(prop.exact_text),
                kind=QAKind.PROPERTY,
                document_id=document.document_id,
                run_id=run.run_id,
                evidence=(prop.span,),
                metadata={
                    "head": prop.head,
                    "degree": prop.degree,
                    "negated": prop.negated,
                },
            )
        )
        if prop.degree or prop.negated:
            examples.append(
                QAExample(
                    context=body,
                    question=f"To what degree was {subject} {prop.head}?",
                    answer=phrase_answer(
                        _degree_answer(prop.head, prop.degree, prop.negated)
                    ),
                    kind=QAKind.PROPERTY,
                    document_id=document.document_id,
                    run_id=run.run_id,
                    evidence=(prop.span,),
                    metadata={"head": prop.head, "degree": prop.degree},
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

    Always asked in the affirmative, so that a negated clause produces a
    *No.* instead of the nonsense of confirming a negation. A bare *Yes.*
    teaches nothing checkable, so the answer repeats the fact it rests on.
    """
    resolver = resolver or ParadigmResolver()
    body = context if context is not None else document.text
    examples: list[QAExample] = []
    for view in build_views(run, document):
        agent, theme = view.agent, view.theme
        if agent is None:
            continue
        aux, *rest = _verb_words(
            view.predicate, resolver, subject_present=True, negated=False
        )
        examples.append(
            QAExample(
                context=body,
                question=_question([aux, _inline(agent), *rest, _inline(theme)]),
                answer=(
                    ("No. " if view.is_negated else "Yes. ")
                    + sentence_case(_clause(view, resolver))
                    + "."
                ),
                kind=QAKind.YES_NO,
                document_id=document.document_id,
                run_id=run.run_id,
                evidence=(view.predicate.span,),
                metadata={
                    "predicate": view.predicate.lemma,
                    "polarity": str(view.predicate.polarity),
                },
            )
        )
    return examples


def polarity_questions(
    document: Document,
    run: AnnotationRun,
    *,
    resolver: ParadigmResolver | None = None,
    context: str | None = None,
) -> list[QAExample]:
    """Ask whether the text asserts the event or denies it.

    The handoff lists polarity alongside speech act and stance as something
    questions should target. It is recorded on the predicate, so asking
    about it needs no model.
    """
    resolver = resolver or ParadigmResolver()
    body = context if context is not None else document.text
    examples: list[QAExample] = []
    for view in build_views(run, document):
        if view.agent is None:
            continue
        verdict = (
            "It denies that it happened."
            if view.is_negated
            else "It states that it happened."
        )
        examples.append(
            QAExample(
                context=body,
                question="Does the text say that this happened, or that it did not?",
                answer=f"{verdict} {sentence_case(_clause(view, resolver))}.",
                kind=QAKind.SPEECH_ACT,
                document_id=document.document_id,
                run_id=run.run_id,
                evidence=(view.predicate.span,),
                metadata={
                    "polarity": str(view.predicate.polarity),
                    "predicate": view.predicate.lemma,
                },
            )
        )
    return examples


def compound_questions(
    document: Document, run: AnnotationRun, *, context: str | None = None
) -> list[QAExample]:
    """One question covering several arguments at once."""
    body = context if context is not None else document.text
    examples: list[QAExample] = []
    for view in build_views(run, document):
        agent, theme = view.agent, view.theme
        adjuncts = view.adjuncts()
        if agent is None or theme is None or not adjuncts:
            continue
        relation, adjunct, preposition = adjuncts[0]
        wh = _WH_FOR_ADJUNCT[relation].lower()
        examples.append(
            QAExample(
                context=body,
                question=(
                    f"Who {view.predicate.exact_text} {_inline(theme)}, "
                    f"and {wh} did it happen?"
                ),
                answer=phrase_answer(
                    f"{sentence_case(_inline(agent))}, {preposition} {_inline(adjunct)}"
                ),
                kind=QAKind.COMPOUND,
                document_id=document.document_id,
                run_id=run.run_id,
                evidence=(agent.span, adjunct.span),
                metadata={"predicate": view.predicate.lemma},
            )
        )
    return examples
