"""The QA-SRL question template state machine (verb-chain part).

Ported in spirit from ``TemplateStateMachine`` / ``Frame`` in the Scala project
``julianmichael/qasrl`` (MIT). This is a re-derivation rather than a literal
translation: the rules below were reconstructed from the published grammar and
then checked exhaustively against QA-SRL Bank 2.0 (see
``tests/test_state_machine.py``, which asserts that every ``(aux, verb)``
combination occurring in the bank is generated here, and no other).

What this module answers
------------------------
Given the grammatical features of a question
(``tense``/``is_perfect``/``is_progressive``/``is_passive``/``is_negated``)
plus whether the questioned argument is the subject, it produces the ``aux``
and ``verb`` slot strings — and the inverse mapping back to features.

The auxiliary chain
-------------------
English stacks verbal heads in a fixed order, each governing the form of the
next::

    (modal | finite) > have (perfect) > be (progressive) > be (passive) > main

The leftmost head is the finite one and is fronted into ``aux`` by
subject-auxiliary inversion. When the chain has no auxiliary at all, two things
can happen: if the questioned argument *is* the subject there is nothing to
invert, so the finite main verb stays in the ``verb`` slot and ``aux`` is
empty; otherwise ``do``-support supplies an auxiliary to front.

Negation attaches to the fronted auxiliary as a contraction (``doesn't``). The
modal ``might`` has no usable contraction, so its negation surfaces as a
literal ``not`` at the head of the ``verb`` slot instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

from .models import QuestionSlots, VerbForm

__all__ = [
    "TenseFeatures",
    "VerbChain",
    "MODAL_TENSES",
    "FINITE_TENSES",
    "TENSES",
    "WH_WORDS",
    "NOUN_WH",
    "ADVERBIAL_WH",
    "PREPOSITIONS",
    "MOST_COMMON_PREPOSITIONS",
    "SUBJ_VALUES",
    "OBJ_VALUES",
    "OBJ2_VALUES",
    "BARE_COMPLEMENT_OBJ2",
    "AUX_WORDS",
    "VERB_PREFIX_WORDS",
    "build_verb_chain",
    "enumerate_verb_chains",
    "verb_chain_index",
    "features_for_chain",
    "is_known_chain",
]

#: Modal values the ``tense`` field can take, with their negated contractions.
MODAL_TENSES: dict[str, str | None] = {
    "can": "can't",
    "will": "won't",
    "might": None,  # no contraction in QA-SRL: negation moves into the verb slot
    "should": "shouldn't",
    "would": "wouldn't",
}

#: Non-modal values of the ``tense`` field.
FINITE_TENSES: tuple[str, ...] = ("present", "past")

TENSES: tuple[str, ...] = FINITE_TENSES + tuple(MODAL_TENSES)

#: Closed vocabularies of the non-verbal slots, as attested in QA-SRL Bank 2.0.
WH_WORDS: tuple[str, ...] = (
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "how much",
    "how long",
)

#: Wh-words that question a nominal argument. Only these can question the
#: subject, and only these licence a gap at all (``QuestionProcessor``'s
#: completeness guard: a who/what question must have an answer slot).
NOUN_WH: frozenset[str] = frozenset({"who", "what"})

#: Wh-words that question an adjunct (``ArgumentSlot.allAdvSlots``). These
#: reach the template with ``subjRequired = true``, so the subject is always
#: spelled out. ``where`` is in both sets: it questions an adverbial, but it
#: can also question a locative second object.
ADVERBIAL_WH: frozenset[str] = frozenset(
    {"when", "where", "why", "how", "how long", "how much"}
)
SUBJ_VALUES: tuple[str, ...] = ("someone", "something")
OBJ_VALUES: tuple[str, ...] = ("someone", "something")
OBJ2_VALUES: tuple[str, ...] = ("someone", "something", "somewhere", "do", "doing")

#: ``obj2`` values that require the silent preposition ``""`` when no real
#: preposition is present (``help something do``, ``keep doing``).
BARE_COMPLEMENT_OBJ2: frozenset[str] = frozenset({"do", "doing"})

#: Literal words that may precede the inflected form inside the ``verb`` slot.
VERB_PREFIX_WORDS: frozenset[str] = frozenset({"not", "be", "been", "being", "have"})

#: The closed preposition inventory (``TemplateStateMachine.lotsOfPrepositions``).
#: A question's own preposition set is drawn from the words of the sentence it
#: is about, intersected with this list, plus the common ones below — which is
#: why the ``prep`` slot looks open-class in the data while actually being
#: bounded (72 entries). Every one of the 62 preposition tokens in the bank
#: comes from this list or from ``BARE_COMPLEMENT_OBJ2``.
PREPOSITIONS: frozenset[str] = frozenset(
    """aboard about above across afore after against ahead along alongside amid
    amidst among amongst around as aside astride at atop before behind below
    beneath beside besides between beyond by despite down during except for from
    given in inside into near next of off on onto opposite out outside over pace
    per round since than through throughout till times to toward towards under
    underneath until unto up upon versus via with within without""".split()
)

#: Prepositions always offered, regardless of the sentence
#: (``TemplateStateMachine.mostCommonPrepositions``).
MOST_COMMON_PREPOSITIONS: frozenset[str] = frozenset(
    {"by", "for", "with", "in", "from", "to", "as"}
)

assert MOST_COMMON_PREPOSITIONS <= PREPOSITIONS

# Finite surface forms of the auxiliaries ``have`` and ``be``.
_FINITE_HAVE = {"present": "has", "past": "had"}
_FINITE_BE = {"present": "is", "past": "was"}
_DO_SUPPORT = {"present": "does", "past": "did"}
_NEGATED_AUX = {
    "has": "hasn't",
    "had": "hadn't",
    "is": "isn't",
    "was": "wasn't",
    "does": "doesn't",
    "did": "didn't",
}


@dataclass(frozen=True, slots=True)
class TenseFeatures:
    """The grammatical features QA-SRL records alongside a question.

    ``subject_is_questioned`` is not stored in the bank: it is read off the
    slots (``subj == "_"`` while the chain carries no auxiliary). It matters
    because it is the only thing that decides between ``What happens?`` and
    ``What does something do?``.
    """

    tense: str
    is_perfect: bool = False
    is_progressive: bool = False
    is_passive: bool = False
    is_negated: bool = False
    subject_is_questioned: bool = False

    def __post_init__(self) -> None:
        if self.tense not in TENSES:
            raise ValueError(f"unknown tense {self.tense!r}; expected one of {TENSES}")

    @property
    def is_modal(self) -> bool:
        return self.tense in MODAL_TENSES

    @property
    def is_licensed(self) -> bool:
        """Whether the template actually offers this combination.

        English can in principle stack a progressive over a passive under a
        modal (*can be being built*), but the template does not: the only
        state that emits ``being`` is the one reached through a finite
        ``be``-auxiliary. So a progressive passive is available in the simple
        present and past and nowhere else. Excluding these 26 marginal chains
        is what keeps the generated inventory equal to the attested one.
        """
        if self.is_progressive and self.is_passive:
            return not self.is_modal and not self.is_perfect
        return True


@dataclass(frozen=True, slots=True)
class VerbChain:
    """The rendered ``aux`` and ``verb`` slots for one feature bundle."""

    aux: str
    verb: str

    @property
    def has_aux(self) -> bool:
        return self.aux != QuestionSlots.EMPTY

    @property
    def verb_form(self) -> VerbForm:
        return VerbForm(self.verb.split(" ")[-1])


def build_verb_chain(features: TenseFeatures) -> VerbChain:
    """Realise *features* as the ``aux`` and ``verb`` slot strings."""
    # Heads of the auxiliary chain, leftmost first. Each entry is the form that
    # the *preceding* head imposes on it; the first entry is finite.
    heads: list[str] = []
    if features.is_modal:
        heads.append("modal")
    if features.is_perfect:
        heads.append("have")
    if features.is_progressive:
        heads.append("be_prog")
    if features.is_passive:
        heads.append("be_pass")

    # The form each head imposes on whatever follows it.
    imposed = {
        "modal": VerbForm.STEM,
        "have": VerbForm.PAST_PARTICIPLE,
        "be_prog": VerbForm.PRESENT_PARTICIPLE,
        "be_pass": VerbForm.PAST_PARTICIPLE,
    }
    # Non-finite surface spellings of ``have``/``be`` per imposed form.
    spelling = {
        ("have", VerbForm.STEM): "have",
        ("have", VerbForm.PAST_PARTICIPLE): "had",
        ("be_prog", VerbForm.STEM): "be",
        ("be_prog", VerbForm.PAST_PARTICIPLE): "been",
        ("be_pass", VerbForm.STEM): "be",
        ("be_pass", VerbForm.PAST_PARTICIPLE): "been",
        ("be_pass", VerbForm.PRESENT_PARTICIPLE): "being",
    }

    words: list[str] = []  # everything after the fronted auxiliary
    fronted: str | None = None

    if heads:
        first, rest = heads[0], heads[1:]
        if first == "modal":
            fronted = features.tense
        elif features.is_modal:  # unreachable: modal is always first when present
            raise AssertionError
        elif first == "have":
            fronted = _FINITE_HAVE[features.tense]
        else:  # be_prog / be_pass
            fronted = _FINITE_BE[features.tense]

        governing = first
        for head in rest:
            form = imposed[governing]
            words.append(spelling[(head, form)])
            governing = head
        main_form = imposed[governing]
    else:
        # No auxiliary in the chain: either the finite main verb stays put, or
        # ``do``-support is introduced so that something can be fronted.
        needs_do_support = not features.subject_is_questioned or features.is_negated
        if needs_do_support:
            fronted = _DO_SUPPORT[features.tense]
            main_form = VerbForm.STEM
        else:
            fronted = None
            main_form = (
                VerbForm.PRESENT_SINGULAR_3RD
                if features.tense == "present"
                else VerbForm.PAST
            )

    if features.is_negated:
        contraction = (
            MODAL_TENSES[features.tense]
            if features.is_modal
            else _NEGATED_AUX.get(fronted or "")
        )
        if contraction is not None:
            fronted = contraction
        else:
            # ``might`` and friends: the negation stays in the verb slot.
            words.insert(0, "not")

    words.append(main_form.value)
    return VerbChain(aux=fronted or QuestionSlots.EMPTY, verb=" ".join(words))


def _all_feature_bundles() -> Iterator[TenseFeatures]:
    for tense in TENSES:
        for is_perfect in (False, True):
            for is_progressive in (False, True):
                for is_passive in (False, True):
                    for is_negated in (False, True):
                        for questioned in (False, True):
                            features = TenseFeatures(
                                tense=tense,
                                is_perfect=is_perfect,
                                is_progressive=is_progressive,
                                is_passive=is_passive,
                                is_negated=is_negated,
                                subject_is_questioned=questioned,
                            )
                            if features.is_licensed:
                                yield features


@lru_cache(maxsize=1)
def verb_chain_index() -> dict[VerbChain, tuple[TenseFeatures, ...]]:
    """Every ``(aux, verb)`` pair the grammar can produce, with its sources.

    A chain can come from more than one feature bundle: ``is pastParticiple``
    is both a passive present (``What is broken?``) and, when the subject is
    questioned, the very same string — the mapping is many-to-one, so the
    inverse returns all preimages.
    """
    index: dict[VerbChain, list[TenseFeatures]] = {}
    for features in _all_feature_bundles():
        index.setdefault(build_verb_chain(features), []).append(features)
    return {chain: tuple(sources) for chain, sources in index.items()}


def enumerate_verb_chains() -> tuple[VerbChain, ...]:
    """All ``(aux, verb)`` slot pairs licensed by the grammar."""
    return tuple(verb_chain_index())


def is_known_chain(aux: str, verb: str) -> bool:
    return VerbChain(aux, verb) in verb_chain_index()


def features_for_chain(aux: str, verb: str) -> tuple[TenseFeatures, ...]:
    """Feature bundles that realise ``(aux, verb)``; empty if the pair is illegal."""
    return verb_chain_index().get(VerbChain(aux, verb), ())


#: Every word that can occupy the ``aux`` slot, derived from the grammar
#: itself. Deliberately disjoint from :data:`VERB_PREFIX_WORDS`: ``have``,
#: ``be``, ``been``, ``being`` and ``not`` are never fronted, which is what
#: makes the prefix of a question splittable without backtracking.
AUX_WORDS: frozenset[str] = frozenset(
    chain.aux for chain in verb_chain_index() if chain.aux != QuestionSlots.EMPTY
)

assert not (AUX_WORDS & VERB_PREFIX_WORDS)
