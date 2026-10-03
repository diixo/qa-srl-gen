"""Typed argument structure: the object the seven slots are a projection of.

Ported from ``Frame.scala``, ``Argument.scala``, ``ArgumentSlot.scala`` and
``ArgStructure.scala`` of ``julianmichael/qasrl`` (MIT); the slot projection
follows ``SlotBasedLabel.getSlotsForQuestionStructure``. See
THIRD_PARTY_NOTICES.md and SCALA_TO_PYTHON.md.

Why this exists
---------------
:class:`~.models.QuestionSlots` is a *rendering*. It cannot say whether
``What does something give something?`` asks about the first or the second
object, because both readings spell the same way. A :class:`Frame` says it
outright: it holds which argument positions the predicate has, what kind of
thing fills each, and — separately — which one is being questioned.

Generation needs this. A frame knows that ``visit`` takes a destination and
not a recipient, so a generator built on frames cannot emit an argument the
predicate does not license, which slot-level code has no way to prevent.

The projection to slots follows the convention of the **released** QA-SRL
Bank 2.0, which differs from current upstream ``master`` in how a bare
complement is written; see SCALA_TO_PYTHON.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Generic, Mapping, TypeVar

from .inflections import BE_FORMS, DO_FORMS, HAVE_FORMS
from .models import InflectedForms, QuestionSlots, VerbForm
from .question_slots import EMPTY, SILENT_PREP
from .state_machine import ADVERBIAL_WH, BARE_COMPLEMENT_OBJ2, MODAL_TENSES
from .tense import NonFinite, Tense

__all__ = [
    "Argument",
    "Noun",
    "Prep",
    "Locative",
    "LOCATIVE",
    "ArgumentSlot",
    "SUBJ",
    "OBJ",
    "OBJ2",
    "adv",
    "ArgStructure",
    "Frame",
    "MAIN_AUX_VERBS",
    "frame_from_slots",
    "frames_are_equivalent",
    "ALL_ADV_SLOTS",
    "ArgumentValue",
]


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


class Argument:
    """What fills an argument position.

    Three behaviours matter downstream:

    ``placeholder``
        the words written when this argument is *not* the one being asked
        about (``someone``, ``somewhere``);
    ``gap``
        the words that stay behind when it *is* (a stranded preposition);
    ``wh``
        the question word that can ask about it.
    """

    __slots__ = ()

    @property
    def placeholder(self) -> list[str]:
        raise NotImplementedError

    @property
    def gap(self) -> str | None:
        raise NotImplementedError

    @property
    def wh(self) -> str | None:
        raise NotImplementedError

    @property
    def is_noun(self) -> bool:
        return isinstance(self, Noun)

    @property
    def is_prep(self) -> bool:
        return isinstance(self, Prep)

    @property
    def is_locative(self) -> bool:
        return isinstance(self, Locative)

    def to_json(self) -> dict[str, Any]:
        """Circe's tagged ``Argument`` encoding (rather than bare ``Noun``)."""
        if isinstance(self, Noun):
            return {"Noun": {"isAnimate": self.is_animate}}
        if isinstance(self, Prep):
            return {"Prep": {"preposition": self.preposition,
                             "objOpt": ({"isAnimate": self.obj.is_animate}
                                        if self.obj is not None else None)}}
        if isinstance(self, Locative):
            return {"Locative": {}}
        raise TypeError(f"unknown argument type: {type(self).__name__}")

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> "Argument":
        if len(data) != 1:
            raise ValueError(f"expected one tagged argument, got {data!r}")
        if "Noun" in data:
            return Noun(data["Noun"]["isAnimate"])
        if "Prep" in data:
            value = data["Prep"]
            obj = value.get("objOpt")
            return Prep(value["preposition"], None if obj is None else Noun(obj["isAnimate"]))
        if "Locative" in data:
            return LOCATIVE
        raise ValueError(f"unknown argument tag: {next(iter(data))!r}")


@dataclass(frozen=True, slots=True)
class Noun(Argument):
    is_animate: bool

    @property
    def placeholder(self) -> list[str]:
        return ["someone" if self.is_animate else "something"]

    @property
    def gap(self) -> str | None:
        return None

    @property
    def wh(self) -> str | None:
        return "who" if self.is_animate else "what"


@dataclass(frozen=True, slots=True)
class Prep(Argument):
    """A prepositional argument; ``obj`` is ``None`` for a bare preposition.

    ``preposition`` may be several words (``out of``) and may end in ``do`` or
    ``doing`` for a non-finite complement (``to do``, ``from doing``).
    """

    preposition: str
    obj: Noun | None = None

    @property
    def placeholder(self) -> list[str]:
        return list(self.obj.placeholder) if self.obj is not None else []

    @property
    def gap(self) -> str | None:
        return self.preposition

    @property
    def wh(self) -> str | None:
        return self.obj.wh if self.obj is not None else None


@dataclass(frozen=True, slots=True)
class Locative(Argument):
    @property
    def placeholder(self) -> list[str]:
        return ["somewhere"]

    @property
    def gap(self) -> str | None:
        return None

    @property
    def wh(self) -> str | None:
        return "where"


#: The sole locative value; ``Locative`` carries no data.
LOCATIVE = Locative()


# ---------------------------------------------------------------------------
# Argument slots
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True, slots=True)
class ArgumentSlot:
    """A position in the argument structure.

    ``kind`` is ``subj``, ``obj``, ``obj2`` or ``adv``; only an adverbial slot
    carries a ``wh``, because there is one adverbial slot per question word.
    """

    kind: str
    wh: str | None = None

    def __str__(self) -> str:
        return self.wh if self.kind == "adv" and self.wh else self.kind

    @property
    def is_adverbial(self) -> bool:
        return self.kind == "adv"

    @classmethod
    def from_string(cls, value: str) -> "ArgumentSlot | None":
        if value in ("subj", "obj", "obj2"):
            return cls(value)
        if value.lower() in ADVERBIAL_WH:
            return cls("adv", value.lower())
        return None

    @classmethod
    def from_json(cls, value: str) -> "ArgumentSlot":
        slot = cls.from_string(value)
        if slot is None:
            raise ValueError(f"unknown argument slot: {value!r}")
        return slot

    def to_json(self) -> str:
        return str(self)


SUBJ = ArgumentSlot("subj")
OBJ = ArgumentSlot("obj")
OBJ2 = ArgumentSlot("obj2")


def adv(wh: str) -> ArgumentSlot:
    """The adverbial slot questioned by *wh* (``ArgumentSlot.allAdvSlots``)."""
    return ArgumentSlot("adv", wh)


ALL_ADV_SLOTS = tuple(adv(wh) for wh in ("when", "where", "why", "how", "how long", "how much"))


# ---------------------------------------------------------------------------
# Argument structure and frame
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArgStructure:
    """Which positions the predicate has, and whether the clause is passive."""

    args: Mapping[ArgumentSlot, Argument | None] = field(default_factory=dict)
    is_passive: bool = False

    def __post_init__(self) -> None:
        # Scala's DependentMap is immutable. Copy before wrapping so mutations
        # of a caller's input dictionary cannot invalidate a Frame's hash.
        object.__setattr__(self, "args", MappingProxyType(dict(self.args)))

    def __hash__(self) -> int:
        return hash((frozenset(self.args.items()), self.is_passive))

    def with_arg(self, slot: ArgumentSlot, argument: Argument | None) -> "ArgStructure":
        return replace(self, args={**self.args, slot: argument})

    def get(self, slot: ArgumentSlot) -> Argument | None:
        return self.args.get(slot)

    def forget_animacy(self) -> "ArgStructure":
        """Collapse ``someone``/``something`` so structures can be compared."""
        new: dict[ArgumentSlot, Argument | None] = {}
        for slot, argument in self.args.items():
            if isinstance(argument, Noun):
                new[slot] = Noun(False)
            elif isinstance(argument, Prep) and argument.obj is not None:
                new[slot] = Prep(argument.preposition, Noun(False))
            else:
                new[slot] = argument
        return replace(self, args=new)

    def valid_answer_slots(self) -> set[ArgumentSlot]:
        """Slots that can be questioned; a bare preposition cannot be."""
        out = set()
        for slot, argument in self.args.items():
            if slot == OBJ2 and isinstance(argument, Prep) and argument.obj is None:
                continue
            out.add(slot)
        return out

    def to_json(self) -> dict[str, Any]:
        args: dict[str, Any] = {}
        for slot, argument in self.args.items():
            if slot.is_adverbial:
                args[str(slot)] = None  # Scala Unit is JSON null.
            elif slot in (SUBJ, OBJ):
                assert isinstance(argument, Noun)
                args[str(slot)] = {"isAnimate": argument.is_animate}
            else:
                assert argument is not None
                args[str(slot)] = argument.to_json()
        return {"args": args, "isPassive": self.is_passive}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ArgStructure":
        args: dict[ArgumentSlot, Argument | None] = {}
        for key, value in data["args"].items():
            slot = ArgumentSlot.from_json(key)
            if slot.is_adverbial:
                args[slot] = None
            elif slot in (SUBJ, OBJ):
                args[slot] = Noun(value["isAnimate"])
            else:
                args[slot] = Argument.from_json(value)
        return cls(args, data["isPassive"])

    def __str__(self) -> str:
        from .question_template import GENERIC_FORMS

        return Frame(GENERIC_FORMS, self, tense="present").clauses()[0]


#: Words that may be fronted out of the verb stack into ``aux``
#: (``SlotBasedLabel.mainAuxVerbs``). ``mightn't`` is listed upstream but the
#: template never produces it, so it occurs nowhere in the bank.
_NEG_CONTRACTIBLE = ("has", "had", "might", "would", "should", "does", "did", "is", "was")
MAIN_AUX_VERBS: frozenset[str] = frozenset(
    list(_NEG_CONTRACTIBLE)
    + [w + "n't" for w in _NEG_CONTRACTIBLE]
    + ["can", "will", "can't", "won't"]
)


_A = TypeVar("_A")


@dataclass(frozen=True, slots=True)
class ArgumentValue(Generic[_A]):
    """An inserted value in ``gen_clauses_with_args`` (Scala ``Right``).

    Literal grammar tokens are plain strings. Wrapping inserted values keeps
    them distinct even when an argument value itself is a string.
    """

    value: _A


@dataclass(frozen=True, slots=True)
class Frame:
    """A predicate with its argument structure and its grammatical features."""

    verb_inflected_forms: InflectedForms
    structure: ArgStructure = field(default_factory=ArgStructure)
    tense: str = "past"
    is_perfect: bool = False
    is_progressive: bool = False
    is_negated: bool = False

    def __post_init__(self) -> None:
        try:
            tense = Tense(self.tense)
        except ValueError:
            raise ValueError(f"unknown tense {self.tense!r}") from None
        object.__setattr__(self, "tense", tense.value)

    @property
    def args(self) -> Mapping[ArgumentSlot, Argument | None]:
        return self.structure.args

    @property
    def is_passive(self) -> bool:
        return self.structure.is_passive

    @property
    def is_modal(self) -> bool:
        return self.tense in MODAL_TENSES

    @property
    def is_finite(self) -> bool:
        return NonFinite.from_string(self.tense) is None

    def with_arg(self, slot: ArgumentSlot, argument: Argument | None) -> "Frame":
        return replace(self, structure=self.structure.with_arg(slot, argument))

    @classmethod
    def empty(cls, forms: InflectedForms) -> "Frame":
        return cls(forms)

    def to_json(self) -> dict[str, Any]:
        return {
            "verbInflectedForms": self.verb_inflected_forms.to_json(),
            "structure": self.structure.to_json(),
            "tense": self.tense,
            "isPerfect": self.is_perfect,
            "isProgressive": self.is_progressive,
            "isNegated": self.is_negated,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Frame":
        return cls(
            InflectedForms.from_json(data["verbInflectedForms"]),
            ArgStructure.from_json(data["structure"]),
            tense=data["tense"],
            is_perfect=data["isPerfect"],
            is_progressive=data["isProgressive"],
            is_negated=data["isNegated"],
        )

    # -- the verb chain ----------------------------------------------------

    def _modal_tokens(self) -> list[str]:
        modal = self.tense
        if not self.is_negated:
            return [modal]
        if modal == "will":
            return ["won't"]
        if modal == "can":
            return ["can't"]
        if modal == "might":
            # No usable contraction, so the negation stays a separate word.
            return ["might", "not"]
        return [modal + "n't"]

    @staticmethod
    def _paradigm_for(word: str, verb_forms: InflectedForms) -> InflectedForms | None:
        for paradigm in (verb_forms, BE_FORMS, DO_FORMS, HAVE_FORMS):
            if word in paradigm.all_forms:
                return paradigm
        return None

    def get_verb_stack(self) -> list[str]:
        """The full verb sequence, leftmost first (``Frame.getVerbStack``).

        Each auxiliary is pushed in front of the stack and re-inflects the
        word it displaces, which is what turns ``be`` into ``being`` when a
        progressive stacks over a passive.
        """
        stack = [self.verb_inflected_forms.stem]

        def mod_form(form: VerbForm) -> None:
            paradigm = self._paradigm_for(stack[0], self.verb_inflected_forms)
            if paradigm is not None:
                stack[0] = paradigm.get(form)

        if self.tense == "gerund":
            # Scala intentionally suppresses a progressive above a passive,
            # and adds a progressive auxiliary only under a perfect gerund.
            if self.is_passive:
                mod_form(VerbForm.PAST_PARTICIPLE)
                stack.insert(0, "be")
            if self.is_progressive and not self.is_passive and self.is_perfect:
                mod_form(VerbForm.PRESENT_PARTICIPLE)
                stack.insert(0, "be")
            if self.is_perfect:
                mod_form(VerbForm.PAST_PARTICIPLE)
                stack.insert(0, "have")
            mod_form(VerbForm.PRESENT_PARTICIPLE)
            if self.is_negated:
                stack.insert(0, "not")
            return stack

        if self.is_passive:
            mod_form(VerbForm.PAST_PARTICIPLE)
            stack.insert(0, "be")
        if self.is_progressive:
            mod_form(VerbForm.PRESENT_PARTICIPLE)
            stack.insert(0, "be")
        if self.is_perfect:
            mod_form(VerbForm.PAST_PARTICIPLE)
            stack.insert(0, "have")

        bare = len(stack) == 1  # nothing auxiliary yet: do-support territory

        if self.is_modal:
            stack[:0] = self._modal_tokens()
        elif self.is_finite:
            finite = (
                VerbForm.PAST if self.tense == "past" else VerbForm.PRESENT_SINGULAR_3RD
            )
            if self.is_negated:
                if bare:
                    stack.insert(0, "didn't" if self.tense == "past" else "doesn't")
                else:
                    mod_form(finite)
                    stack[0] = stack[0] + "n't"
            else:
                mod_form(finite)
        else:
            if self.tense == "to":
                stack.insert(0, "to")
            if self.is_negated:
                stack.insert(0, "not")
        return stack

    def split_verb_stack_if_necessary(self, stack: list[str]) -> list[str]:
        """Introduce ``do``-support so that something can be fronted."""
        if len(stack) > 1 or self.is_modal or not self.is_finite:
            return list(stack)
        paradigm = self._paradigm_for(stack[0], self.verb_inflected_forms)
        head = "did" if self.tense == "past" else "does"
        stem = paradigm.get(VerbForm.STEM) if paradigm else stack[0]
        return [head, stem]

    def get_verb_conjugation(self, subject_present: bool) -> VerbForm:
        """The form of the main verb (``Frame.getVerbConjugation``)."""
        if self.is_passive:
            return VerbForm.PAST_PARTICIPLE
        if self.is_progressive:
            return VerbForm.PRESENT_PARTICIPLE
        if self.is_perfect:
            return VerbForm.PAST_PARTICIPLE
        if self.is_modal or self.is_negated or subject_present:
            return VerbForm.STEM
        if self.tense in ("bare", "to"):
            return VerbForm.STEM
        if self.tense == "gerund":
            return VerbForm.PRESENT_PARTICIPLE
        return VerbForm.PAST if self.tense == "past" else VerbForm.PRESENT_SINGULAR_3RD

    # -- projection to slots -----------------------------------------------

    def wh_for(self, answer_slot: ArgumentSlot) -> str:
        """The question word that asks about *answer_slot* in this frame."""
        if answer_slot.is_adverbial:
            assert answer_slot.wh is not None
            return answer_slot.wh
        argument = self.args.get(answer_slot)
        if argument is None:
            raise ValueError(f"{answer_slot} is not an argument of this frame")
        wh = argument.wh
        if wh is None:
            raise ValueError(f"{answer_slot} holds {argument!r}, which cannot be questioned")
        return wh

    def to_slots(self, answer_slot: ArgumentSlot) -> QuestionSlots:
        """Project onto the seven slots, asking about *answer_slot*.

        Follows ``getSlotsForQuestionStructure`` in the convention used by the
        released bank: a stranded ``do``/``doing`` is split off the
        preposition into ``obj2``, leaving ``prep`` empty-but-present when
        nothing else remains.
        """
        wh = self.wh_for(answer_slot)

        subject = self.args.get(SUBJ)
        if answer_slot == SUBJ or subject is None:
            subj = EMPTY
        else:
            subj = " ".join(subject.placeholder)

        stack = self.get_verb_stack()
        if subj != EMPTY:
            stack = self.split_verb_stack_if_necessary(stack)
        if len(stack) > 1 and stack[0] in MAIN_AUX_VERBS:
            aux, verb_words = stack[0], stack[1:]
        else:
            aux, verb_words = EMPTY, list(stack)
        verb = " ".join(
            verb_words[:-1] + [self.get_verb_conjugation(subj != EMPTY).value]
        )

        obj_argument = self.args.get(OBJ)
        if answer_slot == OBJ or obj_argument is None:
            obj = EMPTY
        else:
            obj = " ".join(obj_argument.placeholder)

        prep, obj2 = self._project_obj2(answer_slot)
        return QuestionSlots(
            wh=wh, aux=aux, subj=subj, verb=verb, obj=obj, prep=prep, obj2=obj2
        )

    def _project_obj2(self, answer_slot: ArgumentSlot) -> tuple[str, str]:
        argument = self.args.get(OBJ2)
        if argument is None:
            return EMPTY, EMPTY

        if answer_slot == OBJ2:
            if isinstance(argument, Prep):
                return _split_bare_complement(argument.preposition)
            # A gapped noun or locative leaves nothing behind.
            return EMPTY, EMPTY

        if isinstance(argument, Noun):
            return EMPTY, " ".join(argument.placeholder)
        if isinstance(argument, Locative):
            return EMPTY, "somewhere"
        assert isinstance(argument, Prep)
        if argument.obj is None:
            return argument.preposition, EMPTY
        return argument.preposition, " ".join(argument.placeholder)

    def to_slots_upstream(self, answer_slot: ArgumentSlot) -> QuestionSlots:
        """Upstream 16ab4949's complement split, with an abstract verb form.

        ``to_slots`` preserves the released Bank 2.0 convention. The upstream
        labeling API instead places ``to do``/``to doing`` in ``obj2`` and
        joins it to a following placeholder (e.g. ``to do something``).
        """
        slots = self.to_slots(answer_slot)
        argument = self.args.get(OBJ2)
        if not isinstance(argument, Prep) or (argument.obj is None and answer_slot != OBJ2):
            return slots
        words = argument.preposition.split()
        if not words or words[-1] not in BARE_COMPLEMENT_OBJ2:
            return slots
        count = 2 if len(words) > 1 and words[-2] == "to" else 1
        prep = " ".join(words[:-count]) or EMPTY
        obj2 = " ".join(words[-count:])
        if answer_slot != OBJ2:
            obj2 += " " + " ".join(argument.placeholder)
        return replace(slots, prep=prep, obj2=obj2)

    # -- rendering ---------------------------------------------------------

    def question(self, answer_slot: ArgumentSlot) -> str:
        """The surface question asking about *answer_slot*."""
        from .question_renderer import render_question

        return render_question(self.to_slots(answer_slot), self.verb_inflected_forms)

    def questions(self) -> dict[ArgumentSlot, str]:
        """One question per questionable argument slot."""
        return {
            slot: self.question(slot)
            for slot in sorted(self.structure.valid_answer_slots())
        }

    def clause(self) -> str:
        """The declarative clause, with placeholders for every argument."""
        words: list[str] = []
        subject = self.args.get(SUBJ)
        if subject is not None:
            words.extend(subject.placeholder)
        words.extend(self.get_verb_stack())
        for slot in (OBJ, OBJ2):
            argument = self.args.get(slot)
            if argument is None:
                continue
            if argument.gap:
                words.extend(argument.gap.split())
            words.extend(argument.placeholder)
        return " ".join(words)

    # -- upstream's nondeterministic rendering -----------------------------

    def _necessary_noun(
        self, slot: ArgumentSlot, arg_values: Mapping[ArgumentSlot, _A]
    ) -> list[list[str | ArgumentValue[_A]]]:
        argument = self.args.get(slot)
        if argument is None:
            return [["someone"], ["something"]]
        if slot in arg_values:
            return [[ArgumentValue(arg_values[slot])]]
        return [list(argument.placeholder)]

    @staticmethod
    def get_ungap(gap: str | None) -> list[str]:
        """Remove complement ``do``/``doing`` when inserting a real value."""
        return [] if gap is None else [w for w in gap.split() if w not in ("do", "doing")]

    def _render_arg(
        self, slot: ArgumentSlot, arg_values: Mapping[ArgumentSlot, _A]
    ) -> list[str | ArgumentValue[_A]]:
        argument = self.args.get(slot)
        if argument is None:
            return []
        if slot in arg_values:
            return self.get_ungap(argument.gap) + [ArgumentValue(arg_values[slot])]
        return ([argument.gap] if argument.gap is not None else []) + argument.placeholder

    def gen_clauses_with_args(
        self, arg_values: Mapping[ArgumentSlot, _A]
    ) -> list[list[str | ArgumentValue[_A]]]:
        """All clauses, retaining the boundaries and types of inserted values.

        An unspecified subject branches into ``someone`` and ``something``;
        the upstream routine does not infer its animacy from other arguments.
        """
        tail = (self.get_verb_stack() + self._render_arg(OBJ, arg_values)
                + self._render_arg(OBJ2, arg_values))
        return [subject + tail for subject in self._necessary_noun(SUBJ, arg_values)]

    def clauses_with_args(self, arg_values: Mapping[ArgumentSlot, str]) -> list[str]:
        return [" ".join(token.value if isinstance(token, ArgumentValue) else token
                         for token in tokens)
                for tokens in self.gen_clauses_with_args(arg_values)]

    def clauses(self) -> list[str]:
        """Upstream ``clauses``; unlike ``clause``, fills a missing subject."""
        return self.clauses_with_args({})

    def clauses_with_arg_markers(self) -> list[list[str | ArgumentValue[ArgumentSlot]]]:
        return self.gen_clauses_with_args({slot: slot for slot in self.args})

    def questions_for_slot(self, slot: ArgumentSlot) -> list[str]:
        """All upstream renderings, including missing-noun alternatives."""
        return self.questions_for_slot_with_args(slot, {})

    def questions_for_slot_with_args(
        self, slot: ArgumentSlot | None, arg_values: Mapping[ArgumentSlot, str]
    ) -> list[str]:
        """Render a wh-question, or an inverted question when slot is ``None``.

        The optional answer slot and inserted arguments follow Scala's
        ``questionsForSlotWithArgs``. An absent subject or queried object
        branches across the two animate/inanimate placeholders; querying an
        absent or objectless second argument produces no questions.
        """
        if slot is None:
            prefixes = [[]]
        elif slot in (SUBJ, OBJ):
            argument = self.args.get(slot)
            whs = ["Who", "What"] if argument is None else [argument.wh.capitalize()]
            prefixes = [[wh] for wh in whs]
        elif slot == OBJ2:
            argument = self.args.get(OBJ2)
            if argument is None or argument.wh is None:
                return []
            prefixes = [[argument.wh.capitalize()]]
        elif slot.is_adverbial:
            prefixes = [[slot.wh.capitalize()]]
        else:
            raise ValueError(f"unknown argument slot: {slot!r}")

        if slot == SUBJ:
            cores = [self.get_verb_stack()]
        else:
            stack = self.split_verb_stack_if_necessary(self.get_verb_stack())
            cores = [[stack[0]] + subj + stack[1:]
                     for subj in self._necessary_noun(SUBJ, arg_values)]

        tail: list[str | ArgumentValue[str]] = []
        for arg_slot in (OBJ, OBJ2):
            if slot == arg_slot:
                argument = self.args.get(arg_slot)
                if argument is not None and argument.gap is not None:
                    tail.append(argument.gap)
            else:
                tail.extend(self._render_arg(arg_slot, arg_values))
        return [" ".join(token.value if isinstance(token, ArgumentValue) else token
                         for token in prefix + core + tail) + "?"
                for prefix in prefixes for core in cores]


def _split_bare_complement(preposition: str) -> tuple[str, str]:
    """Split a stranded preposition into its ``prep`` and ``obj2`` halves.

    ``to do`` strands as ``prep="to"`` plus ``obj2="do"``; a bare ``do``
    leaves ``prep`` present but silent, which is the released bank's way of
    writing ``What does something help do?``.
    """
    words = preposition.split()
    if words and words[-1] in BARE_COMPLEMENT_OBJ2:
        head, complement = words[:-1], words[-1]
        return (" ".join(head) if head else SILENT_PREP), complement
    return preposition, EMPTY


def frames_are_equivalent(left: Frame, right: Frame) -> bool:
    """Whether two frames differ only in the animacy of their arguments."""
    return (
        left.structure.forget_animacy() == right.structure.forget_animacy()
        and (left.tense, left.is_perfect, left.is_progressive, left.is_negated)
        == (right.tense, right.is_perfect, right.is_progressive, right.is_negated)
    )


# ---------------------------------------------------------------------------
# Recovering a frame from slots
# ---------------------------------------------------------------------------

_ANIMATE_WH = {"who": True, "what": False}
_ANIMATE_PLACEHOLDER = {"someone": True, "something": False}


def frame_from_slots(
    slots: QuestionSlots, forms: InflectedForms
) -> tuple[Frame, ArgumentSlot]:
    """Recover ``(frame, answer_slot)`` from a slot bundle.

    Upstream does this by running the automaton and keeping every complete
    state, then picking one with corpus statistics. Here the inverse is
    computed directly from the slot conventions, which is deterministic but
    has to break the same ties upstream does. Two are genuinely undecidable
    from the slots alone and are resolved the way ``ClauseResolution``'s
    fallbacks resolve them:

    * a stranded preposition is read as questioning the object of the
      preposition rather than the direct object (its ``"prepositional"``
      class);
    * a bare ``Where`` question is read as adverbial rather than as a gapped
      locative second object (its ``"where"`` class).

    Projecting the result back with :meth:`Frame.to_slots` reproduces the
    input exactly for every question in QA-SRL Bank 2.0.
    """
    from .state_machine import ADVERBIAL_WH, features_for_chain

    candidates = [
        f
        for f in features_for_chain(slots.aux, slots.verb)
        if f.subject_is_questioned == (slots.subj == EMPTY)
    ]
    if not candidates:
        raise ValueError(
            f"aux={slots.aux!r} with verb={slots.verb!r} is not a chain the "
            "grammar can produce"
        )
    features = candidates[0]

    # Which slot is being asked about.
    if slots.subj == EMPTY:
        answer_slot = SUBJ
    elif slots.obj2 in BARE_COMPLEMENT_OBJ2:
        answer_slot = OBJ2
    elif slots.wh in ADVERBIAL_WH and slots.wh not in _ANIMATE_WH:
        answer_slot = adv(slots.wh)
    elif slots.prep not in (EMPTY, SILENT_PREP) and slots.obj2 == EMPTY:
        answer_slot = OBJ2
    elif slots.obj == EMPTY:
        answer_slot = OBJ
    else:
        answer_slot = OBJ2

    args: dict[ArgumentSlot, Argument] = {}

    if answer_slot == SUBJ:
        args[SUBJ] = Noun(_ANIMATE_WH[slots.wh])
    else:
        args[SUBJ] = Noun(_ANIMATE_PLACEHOLDER[slots.subj])

    if answer_slot == OBJ:
        args[OBJ] = Noun(_ANIMATE_WH[slots.wh])
    elif slots.obj != EMPTY:
        args[OBJ] = Noun(_ANIMATE_PLACEHOLDER[slots.obj])

    obj2 = _obj2_from_slots(slots, answer_slot)
    if obj2 is not None:
        args[OBJ2] = obj2

    # A bare nominal second object is only licensed alongside a first object.
    if (
        OBJ not in args
        and isinstance(args.get(OBJ2), Noun)
        and answer_slot not in (OBJ, SUBJ)
    ):
        args[OBJ] = Noun(False)

    frame = Frame(
        verb_inflected_forms=forms,
        structure=ArgStructure(args=args, is_passive=features.is_passive),
        tense=features.tense,
        is_perfect=features.is_perfect,
        is_progressive=features.is_progressive,
        is_negated=features.is_negated,
    )
    return frame, answer_slot


def _obj2_from_slots(slots: QuestionSlots, answer_slot: ArgumentSlot) -> Argument | None:
    """Rebuild the second-object argument from ``prep`` and ``obj2``."""
    if answer_slot == OBJ2:
        head = "" if slots.prep in (EMPTY, SILENT_PREP) else slots.prep
        tail = "" if slots.obj2 == EMPTY else slots.obj2
        preposition = " ".join(w for w in (head, tail) if w)
        if preposition:
            return Prep(preposition, Noun(_ANIMATE_WH[slots.wh]))
        if slots.wh == "where":
            return LOCATIVE
        return Noun(_ANIMATE_WH[slots.wh])

    if slots.prep not in (EMPTY, SILENT_PREP):
        obj = (
            Noun(_ANIMATE_PLACEHOLDER[slots.obj2])
            if slots.obj2 in _ANIMATE_PLACEHOLDER
            else None
        )
        return Prep(slots.prep, obj)
    if slots.obj2 == "somewhere":
        return LOCATIVE
    if slots.obj2 in _ANIMATE_PLACEHOLDER:
        return Noun(_ANIMATE_PLACEHOLDER[slots.obj2])
    return None
