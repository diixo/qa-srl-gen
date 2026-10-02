"""Proposing what might be annotated, and why.

A candidate is a *proposal with evidence*, never a decision. The handoff is
explicit about this: finding a word in the verb dictionary is not sufficient
grounds to call it an ``ACTION``, because the dictionary is a scrape full of
rare and ambiguous entries and because English words routinely belong to
several classes. ``cat`` is listed as a verb. ``found`` is attributed to a
lemma that does not exist.

So every candidate records which source proposed it and how strongly, and a
span may carry several competing candidates. Resolving them is the teacher's
job, with the context in front of it — this module's job is to make sure the
teacher is asked about the right spans, and to keep the answer checkable.

Resources used, all local, none of them authoritative on their own:

* ``en_verb_inflections.txt`` — which surface forms could be verbs, and under
  which lemma and paradigm;
* ``en_postags_withverb.txt`` — the part-of-speech distribution of a word,
  with repetition preserved, so ``cat`` can be seen to be a noun far more
  often than a verb;
* ``verb_phrases.txt`` — multi-word verbal expressions;
* the ontology lexicon — candidate entity types for common nouns;
* orthography — capitalisation, which proposes names and nothing more.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..documents import Document, Passage, TextSpan, Token
from ..ontology import (
    ENTITY_LABELS,
    Label,
    LabelSet,
    MarkerForm,
    MarkerFunction,
    TypeHierarchy,
    load_default_hierarchy,
)
from .lexicons import CONTRACTIONS, NEGATION_TOKENS, STATIVE_VERBS, marker_for
from ..qasrl_core.inflections import InflectionLexicon
from ..qasrl_core.models import InflectedForms, VerbForm

__all__ = [
    "Candidate",
    "CandidateResources",
    "tokenize",
    "extract_candidates",
    "rank_lemmas",
    "FUNCTION_WORDS",
]

#: Tokens that are never annotation candidates on their own. Short, and
#: limited to the closed classes that would otherwise dominate the output.
FUNCTION_WORDS: frozenset[str] = frozenset(
    """a an the this that these those my your his her its our their
    i you he she it we they me him them us
    is am are was were be been being
    do does did done have has had having
    will would can could shall should may might must
    and or but nor so yet for of to in on at by with from into
    not no nor if then than as such very too also just only""".split()
)

#: Matches words (with internal apostrophes and hyphens) and nothing else.
_TOKEN = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:[.,]\d+)*")


@dataclass(frozen=True, slots=True)
class Candidate:
    """A span that might deserve a label, with the reason it was proposed.

    ``proposed_labels`` is a set of *possibilities*, not a prediction. A
    candidate with ``{ACTION, PHYSICAL_OBJECT}`` says the evidence is
    compatible with either, which is exactly the situation the teacher is
    there to resolve.
    """

    span: TextSpan
    exact_text: str
    proposed_labels: LabelSet
    evidence: tuple[str, ...]
    #: Every lemma the surface form could belong to, in dictionary order.
    #: Plural on purpose: the scrape gives ``ran`` both ``run`` and the junk
    #: entry ``rin``, and picking one would present a guess as a fact.
    lemmas: tuple[str, ...] = ()
    paradigms: tuple[InflectedForms, ...] = ()
    verb_forms: tuple[VerbForm, ...] = ()
    #: How often the word is tagged this way in the POS resource, when known.
    tag_counts: Mapping[str, int] = field(default_factory=dict)
    #: Set when the span reads as a discourse marker in this position.
    marker_form: MarkerForm | None = None
    marker_functions: tuple[MarkerFunction, ...] = ()
    #: True for the clitic of a contraction that negates its clause.
    negates: bool = False

    def validate_against(self, document: Document) -> None:
        document.check_span(self.span, self.exact_text)

    @property
    def is_predicate_candidate(self) -> bool:
        return bool(self.proposed_labels.labels & {Label.ACTION, Label.STATE})

    @property
    def is_marker(self) -> bool:
        return self.marker_form is not None

    def __str__(self) -> str:
        return f"{self.exact_text!r} -> {self.proposed_labels} ({', '.join(self.evidence)})"


@dataclass(frozen=True, slots=True)
class CandidateResources:
    """The local resources candidate extraction draws on."""

    inflections: InflectionLexicon | None = None
    postags: Mapping[str, Sequence[str]] = field(default_factory=dict)
    verb_phrases: Sequence[Sequence[str]] = ()
    hierarchy: TypeHierarchy | None = None

    @classmethod
    def load(
        cls,
        *,
        wiktionary_dir: str = "data/wiktionary",
        hierarchy: TypeHierarchy | None = None,
    ) -> "CandidateResources":
        """Load everything from the shipped Wiktionary scrape."""
        from pathlib import Path

        from ..qasrl_core.inflections import (
            load_inflections,
            load_postags,
            load_verb_phrases,
        )

        base = Path(wiktionary_dir)
        return cls(
            inflections=load_inflections(base / "en_verb_inflections.txt"),
            postags=load_postags(base / "en_postags_withverb.txt"),
            verb_phrases=load_verb_phrases(base / "verb_phrases.txt"),
            hierarchy=hierarchy or load_default_hierarchy(),
        )

    @property
    def phrase_index(self) -> Mapping[str, tuple[tuple[str, ...], ...]]:
        """Multi-word verbal expressions, keyed by their first word."""
        index: dict[str, list[tuple[str, ...]]] = {}
        for phrase in self.verb_phrases:
            if len(phrase) > 1:
                index.setdefault(phrase[0].lower(), []).append(tuple(phrase))
        return {k: tuple(v) for k, v in index.items()}


def tokenize(
    text: str, offset: int = 0, *, split_contractions: bool = True
) -> tuple[Token, ...]:
    """Split *text* into word tokens that carry their character offsets.

    Contractions are split by default, so that ``don't`` becomes ``do`` plus
    ``n't`` rather than one unknown word. Without this the negation is
    invisible to everything downstream — and a corpus that cannot see
    negation teaches that *I don't want it* and *I want it* mean the same.
    Offsets stay exact: the two pieces partition the original span.
    """
    tokens: list[Token] = []
    for match in _TOKEN.finditer(text):
        word = match.group()
        start = match.start() + offset
        parts = CONTRACTIONS.get(word.lower().replace("’", "'")) if split_contractions else None
        if parts and len(parts[0]) + len(parts[1]) == len(word):
            head_end = start + len(parts[0])
            tokens.append(Token(word[: len(parts[0])], start, head_end, len(tokens)))
            tokens.append(
                Token(word[len(parts[0]) :], head_end, match.end() + offset, len(tokens))
            )
            continue
        tokens.append(Token(word, start, match.end() + offset, len(tokens)))
    return tuple(tokens)


def rank_lemmas(
    lemmas: Sequence[str], resources: "CandidateResources"
) -> tuple[str, ...]:
    """Order candidate lemmas so the plausible ones come first.

    The scrape offers ``thought -> thinck, think, thinke`` and
    ``went -> gan, go``, in no useful order, so taking the first was taking
    a guess. Mere presence in the part-of-speech resource does not separate
    them — it is the same scrape, and ``thinck`` is in it too. What does
    separate them is how many senses each lemma has recorded: a real word
    accumulates several, scrape debris has exactly one. ``think`` carries
    two tags against ``thinck``'s one, ``go`` three against ``gan``'s one,
    ``make`` four against ``myek``'s one.
    """
    postags = resources.postags

    def key(lemma: str) -> tuple[int, int, str]:
        tags = postags.get(lemma.lower(), ())
        return (-len(tags), len(lemma), lemma)

    return tuple(sorted(dict.fromkeys(lemmas), key=key))


def _capital_is_positional(document: Document, span: TextSpan) -> bool:
    """Whether a capital letter is explained by sentence position alone.

    Half the names proposed on DailyDialog were the first word of a second
    sentence inside a turn — *Remember*, *Let's*, *Which*. A capital that
    the position already explains is no evidence of a name.
    """
    before = document.text[: span.start_char].rstrip()
    return not before or before[-1] in '.!?"\u2019\u201d'


def _tag_counts(word: str, resources: CandidateResources) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tag in resources.postags.get(word.lower(), ()):
        counts[tag] = counts.get(tag, 0) + 1
    return counts


def extract_candidates(
    document: Document,
    passage: Passage | None = None,
    resources: CandidateResources | None = None,
) -> list[Candidate]:
    """Propose annotatable spans in *passage* (or the whole document).

    Every returned candidate is checked against the document before being
    handed back: a proposal whose offsets do not match its text would send
    the teacher a span that does not exist.
    """
    resources = resources or CandidateResources()
    if passage is None:
        text, offset = document.text, 0
    else:
        text, offset = passage.span.text_in(document.text), passage.span.start_char

    tokens = tokenize(text, offset)
    candidates: list[Candidate] = []
    phrases = resources.phrase_index
    content_tokens = [token for token in tokens if token.text.isalpha()]
    alone = len(content_tokens) == 1

    consumed: set[int] = set()

    # Multi-word verbal expressions first: they must win over their own first
    # word being proposed separately.
    for index, token in enumerate(tokens):
        for phrase in phrases.get(token.text.lower(), ()):
            window = tokens[index : index + len(phrase)]
            if len(window) != len(phrase):
                continue
            if [t.text.lower() for t in window] != [w.lower() for w in phrase]:
                continue
            span = TextSpan(
                window[0].start_char, window[-1].end_char, index, index + len(phrase)
            )
            candidates.append(
                Candidate(
                    span=span,
                    exact_text=span.text_in(document.text),
                    proposed_labels=LabelSet.of(Label.ACTION),
                    evidence=("verb_phrases",),
                    lemmas=(" ".join(w.lower() for w in phrase),),
                )
            )
            consumed.update(range(index, index + len(phrase)))

    for index, token in enumerate(tokens):
        if index in consumed:
            continue
        lowered = token.text.lower().replace("’", "'")
        span = TextSpan(token.start_char, token.end_char, index, index + 1)

        # Discourse markers are checked before the function-word filter:
        # "no", "well" and "right" are function words in most positions and
        # markers in a few, and the position is what tells them apart.
        after = document.text[token.end_char : token.end_char + 1]
        marker = marker_for(
            lowered,
            turn_initial=index == 0,
            followed_by_comma=bool(after) and after in ",;:!.?",
            stands_alone=alone,
        )
        if marker is not None:
            candidates.append(
                Candidate(
                    span=span,
                    exact_text=span.text_in(document.text),
                    proposed_labels=LabelSet.of(Label.DISCOURSE_MARKER),
                    evidence=("discourse_marker",),
                    marker_form=marker.form,
                    marker_functions=marker.functions,
                )
            )
            continue

        if lowered in NEGATION_TOKENS and not token.text.isalpha():
            # The clitic of a contraction: not a candidate in its own right,
            # but the clause it negates needs to know it is there.
            candidates.append(
                Candidate(
                    span=span,
                    exact_text=span.text_in(document.text),
                    proposed_labels=LabelSet(frozenset()),
                    evidence=("negation",),
                    negates=True,
                )
            )
            continue

        if lowered in FUNCTION_WORDS:
            continue

        counts = _tag_counts(token.text, resources)

        labels: set[Label] = set()
        evidence: list[str] = []
        lemmas: tuple[str, ...] = ()
        paradigms: tuple[InflectedForms, ...] = ()
        forms: tuple[VerbForm, ...] = ()

        # Morphology: could this be an inflected verb?
        if resources.inflections is not None:
            analyses = resources.inflections.analyses(lowered)
            if analyses:
                paradigms = tuple(dict.fromkeys(p for p, _ in analyses))
                lemmas = rank_lemmas(
                    resources.inflections.lemmas_for(lowered), resources
                )
                forms = resources.inflections.forms_of(lowered)
                # Morphology cannot tell a dynamic event from a static one,
                # so the closed list of stative verbs decides; everything
                # else stays an ACTION proposal.
                if any(lemma in STATIVE_VERBS for lemma in lemmas):
                    labels.add(Label.STATE)
                else:
                    labels.add(Label.ACTION)
                evidence.append("inflections")

        if counts:
            evidence.append("postags")
            if "verb" not in counts:
                # Attested, but never as a verb: withdraw the predicate reading.
                labels -= {Label.ACTION, Label.STATE}
            if counts.get("noun"):
                labels.add(Label.ABSTRACT_ENTITY)
            if counts.get("adj"):
                labels.add(Label.PROPERTY)

        # Ontology lexicon: candidate entity types for known nouns.
        if resources.hierarchy is not None:
            for label_set in resources.hierarchy.candidate_labels(lowered):
                labels |= label_set.labels
            if resources.hierarchy.candidates(lowered):
                evidence.append("ontology")

        # Orthography: a capital proposes a name only when position does not
        # already explain it.
        if token.text[:1].isupper() and not _capital_is_positional(document, span):
            labels.add(Label.NAMED_ENTITY)
            evidence.append("capitalisation")

        if not labels:
            continue

        candidates.append(
            Candidate(
                span=span,
                exact_text=span.text_in(document.text),
                proposed_labels=_well_formed(labels),
                evidence=tuple(evidence),
                lemmas=lemmas,
                paradigms=paradigms,
                verb_forms=forms,
                tag_counts=counts,
            )
        )

    for candidate in candidates:
        candidate.validate_against(document)
    return candidates


def _well_formed(labels: Iterable[Label]) -> LabelSet:
    """Drop combinations the ontology forbids, keeping the proposal usable.

    ``ACTION`` and ``STATE`` cannot both hold, and morphology proposes both.
    Rather than emit an invalid label set, the predicate reading collapses to
    ``ACTION``, which the teacher may still overturn — the point of the
    candidate is that the span is *verbal*, not which of the two it is.
    """
    found = set(labels)
    if Label.ACTION in found and Label.STATE in found:
        found.discard(Label.STATE)
    entity_labels = [label for label in Label if label in found and label in ENTITY_LABELS]
    if len(entity_labels) > 1:
        # Keep the most specific reading available, in declaration order.
        for label in entity_labels[1:]:
            found.discard(label)
    return LabelSet(frozenset(found))
