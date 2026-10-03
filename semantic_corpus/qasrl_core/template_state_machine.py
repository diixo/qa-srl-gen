"""Incremental QA-SRL template grammar, ported from upstream 16ab4949.

This is the transition graph from ``TemplateStateMachine.scala`` in
``julianmichael/qasrl`` (MIT; see THIRD_PARTY_NOTICES.md). Transitions consume
literal text and update a partial Frame. Empty transitions implement grammar
alternatives; no complete questions are enumerated to build this graph.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

from .frame import OBJ, OBJ2, SUBJ, LOCATIVE, Argument, ArgumentSlot, Frame, Noun, Prep
from .models import InflectedForms
from .state_machine import MOST_COMMON_PREPOSITIONS, PREPOSITIONS

__all__ = [
    "FrameState", "TemplateState", "TemplateComplete", "TEMPLATE_COMPLETE",
    "TemplateProgress", "TemplateTransition", "TemplateStateMachine",
]


@dataclass(frozen=True, slots=True)
class FrameState:
    wh_word: str | None
    preposition: str | None
    answer_slot: ArgumentSlot | None
    frame: Frame

    @classmethod
    def initial(cls, forms: InflectedForms) -> FrameState:
        return cls(None, None, None, Frame(forms))


class TemplateState:
    """A node in the template transition graph."""


@dataclass(frozen=True, slots=True)
class TemplateComplete(TemplateState):
    pass


TEMPLATE_COMPLETE = TemplateComplete()
Operation = Callable[[FrameState], FrameState | None]


@dataclass(frozen=True, slots=True)
class TemplateTransition:
    text: str
    target: TemplateState
    operation: Operation | None = None

    def apply(self, frame_state: FrameState) -> tuple[FrameState, TemplateState] | None:
        updated = self.operation(frame_state) if self.operation else frame_state
        return None if updated is None else (updated, self.target)


@dataclass(frozen=True, slots=True)
class TemplateProgress(TemplateState):
    transitions: tuple[TemplateTransition, ...]

    def __post_init__(self) -> None:
        if not self.transitions:
            raise ValueError("a progress state needs at least one transition")


def _progress(*edges: tuple) -> TemplateProgress:
    return TemplateProgress(tuple(TemplateTransition(*edge) for edge in edges))


def _features(**features) -> Operation:
    def update(fs: FrameState) -> FrameState:
        passive = features.get("is_passive")
        values = {k: v for k, v in features.items() if k != "is_passive"}
        if passive is not None:
            values["structure"] = replace(fs.frame.structure, is_passive=passive)
        return replace(fs, frame=replace(fs.frame, **values))
    return update


def _placeholder(slot: ArgumentSlot, make_arg: Callable[[FrameState], Argument | None]) -> Operation:
    def update(fs: FrameState) -> FrameState | None:
        arg = make_arg(fs)
        if slot in fs.frame.args or arg is None:
            return None
        return replace(fs, frame=fs.frame.with_arg(slot, arg))
    return update


def _answer(slot: ArgumentSlot, make_arg: Callable[[FrameState], Argument | None]) -> Operation:
    def update(fs: FrameState) -> FrameState | None:
        if fs.answer_slot is not None or fs.wh_word is None:
            return None
        result = _placeholder(slot, make_arg)(fs)
        return None if result is None else replace(result, answer_slot=slot)
    return update


def _noun_for_wh(fs: FrameState) -> Noun | None:
    return Noun(fs.wh_word == "who") if fs.wh_word in ("who", "what") else None


def _obj2_for_wh(fs: FrameState) -> Argument | None:
    return LOCATIVE if fs.wh_word == "where" else _noun_for_wh(fs)


def _prep_for_wh(fs: FrameState) -> Prep | None:
    noun = _noun_for_wh(fs)
    return Prep(fs.preposition, noun) if fs.preposition is not None and noun is not None else None


def _prep_arg(noun: Noun | None, suffix: str = "") -> Callable[[FrameState], Prep | None]:
    return lambda fs: None if fs.preposition is None else Prep(fs.preposition + suffix, noun)


def _set_prep(preposition: str) -> Operation:
    return lambda fs: replace(fs, preposition=preposition) if fs.preposition is None else None


class TemplateStateMachine:
    """A sentence-conditioned grammar for one verb's incremental questions.

    ``prepositions`` overrides the sentence-derived inventory. As upstream,
    ``to`` and one common non-``to`` preposition remain available even with an
    empty override. Upstream chooses the latter from an unordered Scala Set;
    this port chooses ``by`` deterministically.
    """

    def __init__(
        self,
        tokens: Sequence[str],
        verb_inflected_forms: InflectedForms,
        prepositions: Iterable[str] | None = None,
        *,
        override_prepositions: Iterable[str] | None = None,
    ) -> None:
        if prepositions is not None and override_prepositions is not None:
            raise ValueError("supply prepositions or override_prepositions, not both")
        if override_prepositions is not None:
            prepositions = override_prepositions
        self.tokens = tuple(tokens)
        self.verb_inflected_forms = verb_inflected_forms
        self.initial_frame_state = FrameState.initial(verb_inflected_forms)
        lower = tuple(token.lower() for token in tokens)
        chosen = {token for token in lower if token in PREPOSITIONS}
        chosen.update(
            f"{left} {right}" for left, right in zip(lower, lower[1:])
            if left in PREPOSITIONS and right in PREPOSITIONS
        )
        chosen.update(MOST_COMMON_PREPOSITIONS)
        if prepositions is not None:
            chosen = {prep.lower() for prep in prepositions}
        self.all_chosen_prepositions = frozenset(chosen)
        self.non_to_ending_prepositions = tuple(sorted(
            {"by"} | {prep for prep in chosen if not prep.endswith(" to")}
        ))
        self.to_ending_prepositions = tuple(sorted(
            {"to"} | {prep for prep in chosen if prep.endswith(" to")}
        ))
        self._build()

    def _build(self) -> None:
        forms = self.verb_inflected_forms
        q_mark = _progress(("?", TEMPLATE_COMPLETE))
        self.q_mark = q_mark
        no_prep_obj = _progress(
            ("", q_mark, _answer(OBJ2, _obj2_for_wh)),
            (" someone", q_mark, _placeholder(OBJ2, lambda _: Noun(True))),
            (" something", q_mark, _placeholder(OBJ2, lambda _: Noun(False))),
            (" somewhere", q_mark, _placeholder(OBJ2, lambda _: LOCATIVE)),
        )
        prep_obj_opt = _progress(
            ("", q_mark, _placeholder(OBJ2, _prep_arg(None))),
            ("", q_mark, _answer(OBJ2, _prep_for_wh)),
            (" someone", q_mark, _placeholder(OBJ2, _prep_arg(Noun(True)))),
            (" something", q_mark, _placeholder(OBJ2, _prep_arg(Noun(False)))),
            (" doing", q_mark, _answer(OBJ2, _prep_arg(Noun(False), " doing"))),
            (" doing something", q_mark, _placeholder(OBJ2, _prep_arg(Noun(False), " doing"))),
        )
        post_to_obj = _progress(
            ("", q_mark, _answer(OBJ2, _prep_for_wh)),
            (" someone", q_mark, _placeholder(OBJ2, _prep_arg(Noun(True)))),
            (" something", q_mark, _placeholder(OBJ2, _prep_arg(Noun(False)))),
            (" do", q_mark, _answer(OBJ2, _prep_arg(Noun(False), " do"))),
            (" do something", q_mark, _placeholder(OBJ2, _prep_arg(Noun(False), " do"))),
        )
        post_do_obj = _progress(
            ("", q_mark, _answer(OBJ2, _prep_for_wh)),
            (" something", q_mark, _placeholder(OBJ2, _prep_arg(Noun(False)))),
        )
        non_to = _progress(*(
            (" " + prep, prep_obj_opt, _set_prep(prep))
            for prep in self.non_to_ending_prepositions
        ))
        to = _progress(*(
            (" " + prep, post_to_obj, _set_prep(prep))
            for prep in self.to_ending_prepositions
        ))
        do_prep = _progress(*(
            (" " + prep, post_do_obj, _set_prep(prep)) for prep in ("do", "doing")
        ))
        prep = _progress(*(("", node) for node in (to, do_prep, non_to, no_prep_obj, q_mark)))
        obj = _progress(
            ("", prep),
            ("", prep, _answer(OBJ, _noun_for_wh)),
            (" someone", prep, _placeholder(OBJ, lambda _: Noun(True))),
            (" something", prep, _placeholder(OBJ, lambda _: Noun(False))),
        )
        pp = forms.past_participle
        ing = forms.present_participle
        past_participle_verb = _progress(
            (f" been {ing}", obj, _features(is_progressive=True)),
            (f" been {pp}", obj, _features(is_passive=True)),
            (f" {pp}", obj),
        )
        infinitive_verb = _progress(
            (f" {forms.stem}", obj),
            (f" be {ing}", obj, _features(is_progressive=True)),
            (f" have been {ing}", obj, _features(is_perfect=True, is_progressive=True)),
            (f" be {pp}", obj, _features(is_passive=True)),
            (f" have {pp}", obj, _features(is_perfect=True)),
            (f" have been {pp}", obj, _features(is_perfect=True, is_passive=True)),
        )
        stem_verb = _progress((f" {forms.stem}", obj))
        participle_or_passive = _progress(
            (f" {ing}", obj, _features(is_progressive=True)),
            (f" being {pp}", obj, _features(is_progressive=True, is_passive=True)),
            (f" {pp}", obj, _features(is_passive=True)),
        )
        tensed_verb = _progress(
            (f" {forms.present_singular_3rd}", obj, _features(tense="present")),
            (f" {forms.past}", obj, _features(tense="past")),
        )

        def post_subject_negation(target: TemplateState) -> TemplateState:
            return _progress(("", target), (" not", target, _features(is_negated=True)))

        def subject(target: TemplateState, negated: bool) -> TemplateState:
            destination = target if negated else post_subject_negation(target)
            return _progress(
                (" someone", destination, _placeholder(SUBJ, lambda _: Noun(True))),
                (" something", destination, _placeholder(SUBJ, lambda _: Noun(False))),
                (" it", destination, _placeholder(SUBJ, lambda _: Noun(False))),
            )

        def optional_subject(target: TemplateState, negated: bool) -> TemplateState:
            destination = target if negated else post_subject_negation(target)
            return _progress(
                ("", subject(target, negated)),
                ("", destination, _answer(SUBJ, _noun_for_wh)),
            )

        def subj_target(required: bool, target: TemplateState, negated: bool) -> TemplateState:
            return (subject if required else optional_subject)(target, negated)

        def contraction(required: bool, target: TemplateState) -> TemplateState:
            return _progress(
                ("", subj_target(required, target, False)),
                ("n't", subj_target(required, target, True), _features(is_negated=True)),
            )

        def pre_aux(required: bool) -> TemplateState:
            be_target = contraction(required, participle_or_passive)
            be_aux = _progress(
                (" is", be_target, _features(tense="present")),
                (" was", be_target, _features(tense="past")),
            )
            do_target = contraction(required, stem_verb)
            do_aux = _progress(
                (" does", do_target, _features(tense="present")),
                (" did", do_target, _features(tense="past")),
            )
            have_target = contraction(required, past_participle_verb)
            have_aux = _progress(
                (" has", have_target, _features(tense="present", is_perfect=True)),
                (" had", have_target, _features(tense="past", is_perfect=True)),
            )
            modal_positive = subj_target(required, infinitive_verb, False)
            modal_negative = subj_target(required, infinitive_verb, True)
            modal_contraction = contraction(required, infinitive_verb)
            modal_aux = _progress(
                (" can't", modal_negative, _features(tense="can", is_negated=True)),
                (" can", modal_positive, _features(tense="can")),
                (" won't", modal_negative, _features(tense="will", is_negated=True)),
                (" will", modal_positive, _features(tense="will")),
                (" might", modal_positive, _features(tense="might")),
                (" would", modal_contraction, _features(tense="would")),
                (" should", modal_contraction, _features(tense="should")),
            )
            edges = [("", node) for node in (be_aux, modal_aux, do_aux, have_aux)]
            if not required:
                edges.insert(0, ("", tensed_verb, _answer(SUBJ, _noun_for_wh)))
            return _progress(*edges)

        optional = pre_aux(False)
        required = pre_aux(True)
        self.start = _progress(*(
            (wh.capitalize(), optional if wh in ("who", "what") else required,
             lambda fs, wh=wh: replace(fs, wh_word=wh))
            for wh in ("who", "what", "when", "where", "why", "how", "how much", "how long")
        ))
