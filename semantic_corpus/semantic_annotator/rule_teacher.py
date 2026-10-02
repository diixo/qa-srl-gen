"""A teacher that decides from surface form alone, with no model.

The annotator is built around a teacher because deciding between candidates
needs to read context, and a dictionary cannot. But not every decision needs
a language model: some are settled by punctuation, word order and a closed
list, and those can be made here — offline, deterministically, and for free.

**What this teacher claims.** Dialogue-level facts whose evidence is on the
surface:

* *mood* — a turn ending in ``?`` is interrogative, in ``!`` exclamative;
* *polarity* — a clause containing ``n't``, ``not`` or ``never`` is negative;
* *speech acts* — a question asks, an imperative requests, ``thanks`` thanks,
  ``sorry`` apologises, ``hello`` greets;
* *discourse markers* — the closed inventory in :mod:`.lexicons`, read as a
  marker only in the positions where it cannot be modifying anything.

**What it refuses to claim.** Entity types and predicate senses. The
candidate layer proposes ``ACTION`` for 86% of tokens and ``ABSTRACT_ENTITY``
for most nouns, because morphology genuinely cannot tell; a teacher that
rubber-stamped those proposals would convert "we do not know" into recorded
fact, which is worse than leaving the corpus empty. So it returns no entity
mentions at all, and :attr:`covers` says so out loud.

The consequence is deliberate and visible in the output: run this over
dialogue and you get dialogue-act annotations and nothing else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..ontology import MarkerFunction, Mood, Polarity, SpeechAct, Stance
from .candidates import FUNCTION_WORDS, tokenize
from .lexicons import (
    APOLOGIES,
    DISCOURSE_MARKERS,
    FAREWELLS,
    GREETINGS,
    NEGATION_TOKENS,
    SPEECH_ACT_CUES,
    THANKS,
    marker_for,
)
from .teacher import ProposedAnnotation, TeacherRequest, TeacherResponse

__all__ = ["RuleBasedTeacher", "classify_turn", "TurnReading"]

#: Verbs that open an imperative often enough to be worth listing. A bare
#: base-form verb at the start of a turn is the real signal; these just
#: catch the most common openers unambiguously.
_IMPERATIVE_OPENERS = frozenset(
    """please tell give show let take put come go stop wait look listen
    call send bring send help follow try keep leave hold""".split()
)

_MODAL_REQUEST = re.compile(
    r"^(could|would|can|will|may)\s+(you|i|we)\b", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class TurnReading:
    """What surface form says about one turn."""

    speech_acts: tuple[SpeechAct, ...] = ()
    polarity: Polarity = Polarity.POSITIVE
    mood: Mood = Mood.DECLARATIVE
    stance: Stance | None = None
    markers: tuple[tuple[str, int, int], ...] = ()
    marker_functions: tuple[MarkerFunction, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.speech_acts and not self.markers


def _stance_from(functions: Iterable[MarkerFunction]) -> Stance | None:
    """Map a marker's function onto the speaker's position, where it does."""
    found = set(functions)
    if found & {MarkerFunction.AGREEMENT, MarkerFunction.APPROVAL, MarkerFunction.ACCEPTANCE}:
        return Stance.SUPPORTIVE
    if found & {MarkerFunction.DISAGREEMENT, MarkerFunction.REJECTION, MarkerFunction.DISAPPROVAL}:
        return Stance.OPPOSED
    if found & {MarkerFunction.UNCERTAINTY, MarkerFunction.HESITATION, MarkerFunction.THINKING}:
        return Stance.UNCERTAIN
    return None


def classify_turn(text: str, *, is_reply: bool = False) -> TurnReading:
    """Read one turn's dialogue-level features off its surface form.

    *is_reply* marks a turn that follows another; only then can a bare
    response particle be an answer rather than an opening.
    """
    stripped = text.strip()
    if not stripped:
        return TurnReading()

    tokens = tokenize(stripped)
    words = [t.text.lower().replace("’", "'") for t in tokens]
    content = [w for w in words if w.isalpha()]

    mood = Mood.DECLARATIVE
    if stripped.endswith("?"):
        mood = Mood.INTERROGATIVE
    elif stripped.endswith("!"):
        mood = Mood.EXCLAMATIVE

    polarity = (
        Polarity.NEGATIVE
        if any(word in NEGATION_TOKENS for word in words)
        else Polarity.POSITIVE
    )

    # Discourse markers, in the positions where they cannot be modifiers.
    markers: list[tuple[str, int, int]] = []
    functions: list[MarkerFunction] = []
    alone = len(content) == 1
    for index, token in enumerate(tokens):
        after = stripped[token.end_char : token.end_char + 1]
        entry = marker_for(
            token.text.lower(),
            turn_initial=index == 0 or bool(re.search(r"[.!?]", stripped[tokens[index-1].end_char:token.start_char])),
            followed_by_comma=bool(after) and after in ",;:!.?",
            stands_alone=alone,
        )
        if entry is not None:
            markers.append((token.text, token.start_char, token.end_char))
            functions.extend(entry.functions)

    acts: list[SpeechAct] = []
    # A turn can ask and then assert: "What do you mean? It will help."
    # ends in a full stop but is still a question, so every sentence counts.
    if "?" in stripped:
        acts.append(SpeechAct.QUESTION)
        if _MODAL_REQUEST.match(stripped):
            acts.append(SpeechAct.REQUEST)
    # Cues must be acts of this speaker, not reported or negated words.
    normalized = stripped.lower().replace("’", "'")
    for clause in re.split(r"[.!?]+\s*", normalized):
        clause = clause.strip()
        if re.match(r"^(hello|hi|hey|greetings)\b|^good (morning|afternoon|evening)\b", clause):
            acts.append(SpeechAct.GREETING)
        if re.match(r"^(bye|goodbye|farewell|cheerio)\b", clause):
            acts.append(SpeechAct.FAREWELL)
        if re.match(r"^(thanks\b|thank you\b|thankyou\b|i thank you\b)", clause):
            acts.append(SpeechAct.THANKING)
        if re.match(r"^(sorry\b|apologies\b|i(?: am|'m) (?:so |very )?sorry\b|i apologi[sz]e\b)", clause):
            acts.append(SpeechAct.APOLOGY)
        if re.match(r"^no problem(?:[,.!]|$)", clause):
            acts.append(SpeechAct.REASSURANCE)
    if mood is not Mood.INTERROGATIVE and content:
        if content[0] in _IMPERATIVE_OPENERS:
            mood = Mood.IMPERATIVE
            acts.append(
                SpeechAct.REQUEST if content[0] == "please" else SpeechAct.COMMAND
            )

    # A marker's function is also an act: "Yes." agrees, "No." rejects.
    function_acts = {
        MarkerFunction.AGREEMENT: SpeechAct.AGREEMENT,
        MarkerFunction.DISAGREEMENT: SpeechAct.DISAGREEMENT,
        MarkerFunction.ACCEPTANCE: SpeechAct.ACCEPTANCE,
        MarkerFunction.REJECTION: SpeechAct.REJECTION,
        MarkerFunction.ACKNOWLEDGEMENT: SpeechAct.ACKNOWLEDGEMENT,
        MarkerFunction.APPROVAL: SpeechAct.APPROVAL,
        MarkerFunction.DISAPPROVAL: SpeechAct.DISAPPROVAL,
        MarkerFunction.SURPRISE: SpeechAct.REACTION,
        MarkerFunction.REACTION: SpeechAct.REACTION,
        MarkerFunction.HESITATION: SpeechAct.HESITATION,
        MarkerFunction.UNCERTAINTY: SpeechAct.UNCERTAINTY,
        MarkerFunction.THINKING: SpeechAct.HESITATION,
        MarkerFunction.CONFIRMATION: SpeechAct.CONFIRMATION,
    }
    for function in functions:
        act = function_acts.get(function)
        if act is not None and act not in acts:
            acts.append(act)

    if is_reply and mood is not Mood.INTERROGATIVE and SpeechAct.QUESTION not in acts:
        if markers and len(content) <= 3:
            # A short turn that is only a response particle is an answer.
            if SpeechAct.ANSWER not in acts:
                acts.insert(0, SpeechAct.ANSWER)

    if not acts:
        acts.append(SpeechAct.INFORM)

    return TurnReading(
        speech_acts=tuple(dict.fromkeys(acts)),
        polarity=polarity,
        mood=mood,
        stance=_stance_from(functions),
        markers=tuple(markers),
        marker_functions=tuple(dict.fromkeys(functions)),
    )


class RuleBasedTeacher:
    """Offline teacher for the dialogue-level layer.

    Implements the :class:`~.teacher.Teacher` protocol, so it drops into the
    pipeline wherever an HTTP teacher would go — and spends nothing.
    """

    #: What this teacher is willing to decide. Read it before trusting an
    #: empty result: no entity mentions means "not attempted", not "none".
    covers: tuple[str, ...] = (
        "discourse_marker",
        "speech_act",
        "polarity",
        "stance",
        "mood",
    )

    def __init__(self, name: str = "rules-v2") -> None:
        self.name = name
        self.calls = 0

    def annotate(self, request: TeacherRequest) -> TeacherResponse:
        self.calls += 1
        focus = request.focus
        is_reply = request.context.strip() != focus.strip()
        reading = classify_turn(focus, is_reply=is_reply)

        annotations = []
        for text, _start, _end in reading.markers:
            # Alignment counts literal occurrences, including non-markers.
            leading = len(focus) - len(focus.lstrip())
            occurrence = sum(1 for m in re.finditer(re.escape(text), focus)
                             if m.start() < _start + leading)
            annotations.append(
                ProposedAnnotation(
                    text=text,
                    labels=("DISCOURSE_MARKER",),
                    occurrence=occurrence,
                    confidence=None,
                )
            )

        return TeacherResponse(
            annotations=tuple(annotations),
            speech_acts=tuple(str(a) for a in reading.speech_acts),
            polarity=str(reading.polarity),
            stance=str(reading.stance) if reading.stance else None,
            mood=str(reading.mood),
            raw={
                "teacher": self.name,
                "covers": list(self.covers),
                # The pipeline reads these back into the DialogueAnnotation;
                # without them "What does 'Oh' express?" has no answer.
                "marker_functions": [str(f) for f in reading.marker_functions],
            },
        )
