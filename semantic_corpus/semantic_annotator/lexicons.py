"""Closed lexicons the candidate layer needs but a scraped dictionary lacks.

Everything here is a hand-written, finite list. That is the point: the
Wiktionary scrape is wide and unreliable — it lists ``cat`` as a verb and
attributes ``found`` to a non-word ``foind`` — so the places where English
really is a closed class are better served by naming the members than by
looking them up.

Measured on DailyDialog's 81 869 turns, these four lists address the bulk of
what the dictionary alone got wrong:

* ``STATIVE_VERBS`` — morphology cannot tell a dynamic event from a static
  situation, so every verb was proposed as ``ACTION``; ``STATE`` appeared on
  0.3% of candidates even though *know*, *think* and *want* are everywhere;
* ``DISCOURSE_MARKERS`` — a quarter of turns open with one, and not a single
  ``DISCOURSE_MARKER`` was ever proposed: *yes* came out as
  ``ACTION + ABSTRACT_ENTITY`` and *ok* got nothing at all;
* ``CONTRACTIONS`` — ``don't``, ``it's`` and ``I'm`` were single unknown
  tokens, so the negation inside them was invisible;
* ``GREETINGS`` / ``THANKS`` / ``APOLOGIES`` — surface forms that really do
  determine a speech act.
"""

from __future__ import annotations

from ..ontology import MarkerForm, MarkerFunction, SpeechAct

__all__ = [
    "STATIVE_VERBS",
    "DISCOURSE_MARKERS",
    "MarkerEntry",
    "CONTRACTIONS",
    "NEGATION_TOKENS",
    "GREETINGS",
    "FAREWELLS",
    "THANKS",
    "APOLOGIES",
    "marker_for",
    "SPEECH_ACT_CUES",
]

#: Verbs denoting a situation rather than an event. The handoff's rule is
#: that ``ACTION`` and ``STATE`` are separate; morphology cannot separate
#: them, so the stative side is listed.
STATIVE_VERBS: frozenset[str] = frozenset(
    """know own contain remain belong exist think believe like love hate
    want need prefer understand remember forget seem appear resemble consist
    deserve matter mean cost weigh measure equal involve depend concern
    suppose imagine doubt recognise recognize realise realize wish hope
    admire dislike envy fear mind owe possess lack include comprise
    see hear smell taste feel notice perceive sound look
    be have""".split()
)


class MarkerEntry(tuple):
    """A discourse marker's form and the functions it can carry.

    A tuple subclass so the table stays readable as literal data while the
    fields still have names.
    """

    __slots__ = ()

    def __new__(cls, form: MarkerForm, functions: tuple[MarkerFunction, ...]):
        return super().__new__(cls, (form, functions))

    @property
    def form(self) -> MarkerForm:
        return self[0]

    @property
    def functions(self) -> tuple[MarkerFunction, ...]:
        return self[1]


def _marker(form: MarkerForm, *functions: MarkerFunction) -> MarkerEntry:
    return MarkerEntry(form, functions)


#: Discourse markers, with the functions each *can* serve. Several are listed
#: because the word never settles it: ``Good.`` on its own is approval, *a
#: good car* is a property, and only position and context tell them apart —
#: which is why :func:`marker_for` takes that context rather than just a word.
DISCOURSE_MARKERS: dict[str, MarkerEntry] = {
    # Interjections
    "oh": _marker(MarkerForm.INTERJECTION, MarkerFunction.REACTION, MarkerFunction.SURPRISE),
    "ah": _marker(MarkerForm.INTERJECTION, MarkerFunction.REACTION),
    "wow": _marker(MarkerForm.INTERJECTION, MarkerFunction.SURPRISE, MarkerFunction.REACTION),
    "oops": _marker(MarkerForm.INTERJECTION, MarkerFunction.REACTION),
    "aha": _marker(MarkerForm.INTERJECTION, MarkerFunction.ACKNOWLEDGEMENT),
    # Filled pauses
    "um": _marker(MarkerForm.FILLED_PAUSE, MarkerFunction.HESITATION),
    "umm": _marker(MarkerForm.FILLED_PAUSE, MarkerFunction.HESITATION),
    "uh": _marker(MarkerForm.FILLED_PAUSE, MarkerFunction.HESITATION),
    "er": _marker(MarkerForm.FILLED_PAUSE, MarkerFunction.HESITATION),
    "well": _marker(MarkerForm.FILLED_PAUSE, MarkerFunction.HESITATION, MarkerFunction.THINKING),
    # Backchannels
    "hmm": _marker(MarkerForm.BACKCHANNEL, MarkerFunction.THINKING, MarkerFunction.UNCERTAINTY),
    "mhm": _marker(MarkerForm.BACKCHANNEL, MarkerFunction.ACKNOWLEDGEMENT),
    "i see": _marker(MarkerForm.BACKCHANNEL, MarkerFunction.ACKNOWLEDGEMENT),
    # Response particles
    "yes": _marker(MarkerForm.RESPONSE_PARTICLE, MarkerFunction.AGREEMENT, MarkerFunction.CONFIRMATION),
    "yeah": _marker(MarkerForm.RESPONSE_PARTICLE, MarkerFunction.AGREEMENT),
    "yep": _marker(MarkerForm.RESPONSE_PARTICLE, MarkerFunction.AGREEMENT),
    "no": _marker(MarkerForm.RESPONSE_PARTICLE, MarkerFunction.DISAGREEMENT, MarkerFunction.REJECTION),
    "nope": _marker(MarkerForm.RESPONSE_PARTICLE, MarkerFunction.REJECTION),
    "ok": _marker(MarkerForm.RESPONSE_PARTICLE, MarkerFunction.ACCEPTANCE, MarkerFunction.ACKNOWLEDGEMENT),
    "okay": _marker(MarkerForm.RESPONSE_PARTICLE, MarkerFunction.ACCEPTANCE, MarkerFunction.ACKNOWLEDGEMENT),
    "alright": _marker(MarkerForm.RESPONSE_PARTICLE, MarkerFunction.ACCEPTANCE),
    "sure": _marker(MarkerForm.RESPONSE_PARTICLE, MarkerFunction.AGREEMENT, MarkerFunction.ACCEPTANCE),
    "certainly": _marker(MarkerForm.RESPONSE_PARTICLE, MarkerFunction.AGREEMENT),
    "right": _marker(MarkerForm.RESPONSE_PARTICLE, MarkerFunction.ACKNOWLEDGEMENT, MarkerFunction.AGREEMENT),
    "exactly": _marker(MarkerForm.RESPONSE_PARTICLE, MarkerFunction.AGREEMENT),
    "really": _marker(MarkerForm.RESPONSE_PARTICLE, MarkerFunction.SURPRISE, MarkerFunction.UNCERTAINTY),
    # Evaluative responses
    "good": _marker(MarkerForm.EVALUATIVE_RESPONSE, MarkerFunction.APPROVAL),
    "great": _marker(MarkerForm.EVALUATIVE_RESPONSE, MarkerFunction.APPROVAL),
    "fine": _marker(MarkerForm.EVALUATIVE_RESPONSE, MarkerFunction.ACCEPTANCE, MarkerFunction.APPROVAL),
    "excellent": _marker(MarkerForm.EVALUATIVE_RESPONSE, MarkerFunction.APPROVAL),
    "perfect": _marker(MarkerForm.EVALUATIVE_RESPONSE, MarkerFunction.APPROVAL),
    "terrible": _marker(MarkerForm.EVALUATIVE_RESPONSE, MarkerFunction.DISAPPROVAL),
    "awful": _marker(MarkerForm.EVALUATIVE_RESPONSE, MarkerFunction.DISAPPROVAL),
}


def marker_for(
    word: str, *, turn_initial: bool, followed_by_comma: bool, stands_alone: bool
) -> MarkerEntry | None:
    """The marker reading of *word*, if the context licenses one.

    The handoff is explicit that context decides: ``Good.`` is approval,
    *It is a good car* is a property. A word in the table is read as a
    marker only when it opens the turn, is set off by a comma, or is the
    whole turn — the three positions where it cannot be modifying anything.
    """
    entry = DISCOURSE_MARKERS.get(word.lower())
    if entry is None:
        return None
    if stands_alone or turn_initial or followed_by_comma:
        return entry
    return None


#: Contracted forms, split into the part that carries meaning and the clitic.
#: Keeping ``n't`` visible is the point: without it, negation disappears from
#: 4 002 ``don't`` tokens, and a corpus that cannot see negation teaches a
#: model that *I don't want it* and *I want it* say the same thing.
CONTRACTIONS: dict[str, tuple[str, str]] = {
    "don't": ("do", "n't"),
    "doesn't": ("does", "n't"),
    "didn't": ("did", "n't"),
    "isn't": ("is", "n't"),
    "aren't": ("are", "n't"),
    "wasn't": ("was", "n't"),
    "weren't": ("were", "n't"),
    "hasn't": ("has", "n't"),
    "haven't": ("have", "n't"),
    "hadn't": ("had", "n't"),
    "won't": ("wo", "n't"),
    "wouldn't": ("would", "n't"),
    "can't": ("ca", "n't"),
    "cannot": ("can", "not"),
    "couldn't": ("could", "n't"),
    "shouldn't": ("should", "n't"),
    "mustn't": ("must", "n't"),
    "it's": ("it", "'s"),
    "that's": ("that", "'s"),
    "there's": ("there", "'s"),
    "here's": ("here", "'s"),
    "he's": ("he", "'s"),
    "she's": ("she", "'s"),
    "what's": ("what", "'s"),
    "who's": ("who", "'s"),
    "let's": ("let", "'s"),
    "i'm": ("i", "'m"),
    "i've": ("i", "'ve"),
    "i'll": ("i", "'ll"),
    "i'd": ("i", "'d"),
    "you're": ("you", "'re"),
    "you've": ("you", "'ve"),
    "you'll": ("you", "'ll"),
    "you'd": ("you", "'d"),
    "we're": ("we", "'re"),
    "we've": ("we", "'ve"),
    "we'll": ("we", "'ll"),
    "they're": ("they", "'re"),
    "they've": ("they", "'ve"),
    "they'll": ("they", "'ll"),
}

#: Anything here makes a clause negative.
NEGATION_TOKENS: frozenset[str] = frozenset(
    {"not", "n't", "never", "no", "none", "nothing", "nobody", "nowhere", "neither"}
)

GREETINGS: frozenset[str] = frozenset(
    {"hello", "hi", "hey", "morning", "afternoon", "evening", "greetings"}
)
FAREWELLS: frozenset[str] = frozenset(
    {"bye", "goodbye", "farewell", "cheerio"}
)
THANKS: frozenset[str] = frozenset({"thanks", "thank", "thankyou"})
APOLOGIES: frozenset[str] = frozenset({"sorry", "apologies", "apologise", "apologize"})

#: Surface cues that settle a speech act on their own.
SPEECH_ACT_CUES: dict[str, SpeechAct] = {
    **{word: SpeechAct.GREETING for word in GREETINGS},
    **{word: SpeechAct.FAREWELL for word in FAREWELLS},
    **{word: SpeechAct.THANKING for word in THANKS},
    **{word: SpeechAct.APOLOGY for word in APOLOGIES},
}