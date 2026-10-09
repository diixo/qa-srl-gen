"""Typed events and scoped operators shared by dialogue generation and updates.

An ACTION label is a predicate category, never evidence of completion. Claims
describe what the speaker reported, not independently verified world events.
"""

from dataclasses import dataclass

from ..ontology import Label
from ..semantic_generator.frames import frame_by_lemma


@dataclass(frozen=True)
class EventLexeme:
    lemma: str
    past: str
    third: str
    gerund: str
    participle: str
    pattern: str
    bounded: bool = True

    @property
    def frame(self):
        return frame_by_lemma(self.lemma)


# Forms inflect the head only; particles/prepositions come from the shared
# SemanticFrame. These are reviewed senses, not every reading of a spelling.
EVENT_LEXEMES = (
    EventLexeme('buy', 'bought', 'buys', 'buying', 'bought', 'transitive'),
    EventLexeme('visit', 'visited', 'visits', 'visiting', 'visited', 'transitive', False),
    EventLexeme('give', 'gave', 'gives', 'giving', 'given', 'prepositional'),
    EventLexeme('move', 'moved', 'moves', 'moving', 'moved', 'transitive'),
    EventLexeme('pick up', 'picked', 'picks', 'picking', 'picked', 'joined'),
    EventLexeme('put down', 'put', 'puts', 'putting', 'put', 'joined'),
    EventLexeme('get up', 'got', 'gets', 'getting', 'got', 'intransitive'),
    EventLexeme('look after', 'looked', 'looks', 'looking', 'looked', 'prepositional', False),
)
EVENT_BY_LEMMA = {item.lemma: item for item in EVENT_LEXEMES}


@dataclass(frozen=True)
class Operator:
    lemma: str
    past: str
    third: str
    gerund: str
    complement: str
    category: str


OPERATORS = {item.lemma: item for item in (
    Operator('want', 'wanted', 'wants', 'wanting', 'to', 'desire'),
    Operator('need', 'needed', 'needs', 'needing', 'to', 'necessity'),
    Operator('plan', 'planned', 'plans', 'planning', 'to', 'intention'),
    Operator('intend', 'intended', 'intends', 'intending', 'to', 'intention'),
    Operator('decide', 'decided', 'decides', 'deciding', 'to', 'decision'),
    Operator('try', 'tried', 'tries', 'trying', 'to', 'attempt'),
    Operator('manage', 'managed', 'manages', 'managing', 'to', 'success'),
    Operator('fail', 'failed', 'fails', 'failing', 'to', 'failure'),
    Operator('start', 'started', 'starts', 'starting', 'gerund', 'onset'),
    Operator('finish', 'finished', 'finishes', 'finishing', 'gerund', 'completion'),
    Operator('stop', 'stopped', 'stops', 'stopping', 'gerund', 'cessation'),
)}


@dataclass(frozen=True, order=True)
class EventArgument:
    role: str
    text: str
    label: str


@dataclass(frozen=True)
class Event:
    predicate: str
    arguments: tuple[EventArgument, ...]
    time: str = 'today'
    operators: tuple[str, ...] = ()  # Outer to inner; not negates its child.
    phase: str = 'unspecified'
    tense: str = 'present'

    def __post_init__(self):
        if self.predicate not in EVENT_BY_LEMMA:
            raise ValueError(f'Unsupported event predicate: {self.predicate}')
        if not self.time or self.time != self.time.strip():
            raise ValueError('An event needs an explicit time context')
        if self.phase not in ('unspecified', 'ongoing', 'completed') or self.tense not in ('present', 'past', 'future'):
            raise ValueError('Unsupported event phase or tense')
        if len(self.operators) > 4 or any(op not in OPERATORS and op != 'not' for op in self.operators):
            raise ValueError('Unsupported event operator chain')
        if any(a == b == 'not' for a, b in zip(self.operators, self.operators[1:])):
            raise ValueError('Repeated adjacent negation is not supported')
        if any(op != 'not' for op in self.operators) and self.phase != 'unspecified':
            raise ValueError('Phase belongs to the bare action; operator constructions use unspecified phase')
        if self.tense == 'future' and (self.operators or self.phase != 'unspecified'):
            raise ValueError('Only bare future actions are currently supported')
        if (self.tense == 'past' and self.phase == 'unspecified' and self.operators in ((), ('not',))
                and EVENT_BY_LEMMA[self.predicate].bounded):
            raise ValueError('A bare past bounded action requires an explicit completed phase')
        frame = EVENT_BY_LEMMA[self.predicate].frame
        if frame.predicate_type != Label.ACTION:
            raise ValueError('Typed event claims require an ACTION frame')
        roles = [arg.role for arg in self.arguments]
        if roles != sorted(set(roles)):
            raise ValueError('Event arguments must have distinct, sorted roles')
        pattern = frame.pattern(EVENT_BY_LEMMA[self.predicate].pattern)
        if set(roles) != set(pattern.mentioned()):
            raise ValueError('Event arguments must match the selected shared frame pattern')
        for arg in self.arguments:
            if not arg.text or arg.text != arg.text.strip():
                raise ValueError('Event argument text must be nonempty')
            label = Label(arg.label)
            # GROUP requires a subtype, which this compact contract does not
            # encode yet; refuse rather than bypass the shared group constraints.
            if label == Label.GROUP or not frame.slot_map[arg.role].accepts((label,)):
                raise ValueError(f'{arg.role}: incompatible entity type {arg.label}')

    @property
    def identity(self):
        return self.predicate, self.arguments, self.time

    @property
    def key(self):
        # One current claim per event and operator path. A correction may change
        # outer polarity/phase/tense. Inner negation changes what is wanted or
        # needed: not(want(X)) and want(not(X)) must coexist, not overwrite.
        path = self.operators[1:] if self.operators[:1] == ('not',) else self.operators
        return self.identity, path

    @property
    def agent(self):
        frame = EVENT_BY_LEMMA[self.predicate].frame
        role = frame.pattern(EVENT_BY_LEMMA[self.predicate].pattern).subject
        return next(arg.text for arg in self.arguments if arg.role == role)


def completion_evidence(event: Event) -> str:
    """Conservative yes/no/unknown for completion of this precise event.

    A negated modal is not a negated action. Starting, stopping and attempting
    do not establish completion. Nested success under a wish remains unknown.
    """
    if event.tense == 'future':
        return 'unknown'
    if event.operators in ((), ('not',)) and event.phase == 'completed':
        return 'no' if event.operators else 'yes'
    if event.tense == 'past' and EVENT_BY_LEMMA[event.predicate].bounded:
        if event.operators in (('manage',), ('finish',)):
            return 'yes'
        if event.operators in (('fail',), ('not', 'manage'), ('manage', 'not')):
            return 'no'
        if event.operators == ('fail', 'not'):
            return 'yes'
    # Finishing an activity explicitly establishes its end even when its plain
    # past form (visited/looked after) alone does not guarantee completion.
    if event.tense == 'past' and event.operators == ('finish',):
        return 'yes'
    if event.tense == 'past' and event.operators == ('not', 'finish'):
        return 'no'
    return 'unknown'


def _core(event: Event, form: str, style: int) -> str:
    lexeme = EVENT_BY_LEMMA[event.predicate]
    frame = lexeme.frame
    pattern = frame.pattern(lexeme.pattern)
    pronoun = any(arg.role == 'theme' and arg.text.lower() in ('it', 'them', 'him', 'her', 'us', 'you', 'me')
                  for arg in event.arguments)
    if frame.particles and lexeme.pattern == 'joined' and (style % 2 or pronoun):
        pattern = frame.pattern('separated')
    head = (frame.head_lemma or frame.lemma) if form == 'base' else getattr(lexeme, form)
    parts = [head]
    if pattern.particle_after is None:
        parts.extend(frame.particles)
    arguments = {arg.role: arg.text for arg in event.arguments}
    for complement in pattern.complements:
        if complement.preposition:
            parts.append(complement.preposition)
        parts.append(arguments[complement.slot])
        if pattern.particle_after == complement.slot:
            parts.extend(frame.particles)
    return ' '.join(parts)


def _phrase(event: Event, operators: tuple[str, ...], form: str, style: int) -> str:
    if not operators:
        return _core(event, form, style)
    head, *tail = operators
    tail = tuple(tail)
    if head == 'not':
        if form in ('past', 'third', 'present'):
            helper = 'did' if form == 'past' else 'does' if form == 'third' else 'do'
            return helper + ' not ' + _phrase(event, tail, 'base', style)
        return 'not ' + _phrase(event, tail, form, style)
    operator = OPERATORS[head]
    word = operator.lemma if form in ('base', 'present') else getattr(operator, form)
    if operator.complement == 'to':
        if tail and tail[0] == 'not':
            if head == 'need':
                # Avoid the ambiguous modal reading of "need not": this scope
                # requires non-performance, not absence of necessity.
                return word + ' to avoid ' + _phrase(event, tail[1:], 'gerund', style)
            return word + ' not to ' + _phrase(event, tail[1:], 'base', style)
        return word + ' to ' + _phrase(event, tail, 'base', style)
    return word + ' ' + _phrase(event, tail, 'gerund', style)


def event_statement(event: Event, *, perspective='memory', style=0) -> str:
    if perspective not in ('memory', 'user') or style not in range(5):
        raise ValueError('Unsupported event rendering')
    subject = ('I' if perspective == 'user' else 'You') if event.agent == 'user' else event.agent
    first_or_second = subject in ('I', 'You')
    when = ' ' + event.time
    if event.tense == 'future':
        phrase = 'will ' + _core(event, 'base', style)
    elif event.phase == 'ongoing':
        be = ('was' if subject != 'You' else 'were') if event.tense == 'past' else (
            'am' if subject == 'I' else 'are' if subject == 'You' else 'is')
        phrase = be + (' not ' if event.operators else ' ') + _core(event, 'gerund', style)
    elif event.phase == 'completed':
        negative = bool(event.operators)
        if event.tense == 'past':
            if EVENT_BY_LEMMA[event.predicate].bounded:
                phrase = ('did not ' + _core(event, 'base', style)) if negative else _core(event, 'past', style)
            else:
                phrase = ('did not finish ' if negative else 'finished ') + _core(event, 'gerund', style)
        elif EVENT_BY_LEMMA[event.predicate].bounded:
            have = 'have' if first_or_second else 'has'
            phrase = have + (' not ' if negative else ' ') + _core(event, 'participle', style)
        else:
            have = 'have' if first_or_second else 'has'
            phrase = have + (' not finished ' if negative else ' finished ') + _core(event, 'gerund', style)
    else:
        form = 'past' if event.tense == 'past' else 'present' if first_or_second else 'third'
        # The base form is the finite present for first/second person.
        phrase = _phrase(event, event.operators, form if event.operators else ('base' if form == 'present' else form), style)
    if style >= 3 and perspective == 'memory':
        # Hold-out paraphrases change reporting language, not semantic operators.
        original = event_statement(event, perspective='user', style=style - 3)
        return f'{subject} reported: "{original}"' if style == 3 else f'{subject} stated: "{original}"'
    return subject + ' ' + phrase + when + '.'


def event_question(event: Event, *, style=0) -> str:
    if event.agent != 'user':
        subject, have = event.agent, 'Has'
    else:
        subject, have = 'I', 'Have'
    lexeme = EVENT_BY_LEMMA[event.predicate]
    phrase = _core(event, 'participle', style) if lexeme.bounded else 'finished ' + _core(event, 'gerund', style)
    if style >= 3:
        prefix = 'Is there confirmation that' if style == 3 else 'Can you tell whether'
        return f'{prefix} {subject} {have.lower()} {phrase} {event.time}?'
    return f'{have} {subject} {phrase} {event.time}?'


def event_description(event: Event, *, style=0) -> str:
    return _core(event, 'gerund', style) + ' ' + event.time
