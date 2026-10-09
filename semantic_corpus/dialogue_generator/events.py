"""Typed events and scoped operators shared by dialogue generation and updates.

An ACTION label is a predicate category, never evidence of completion. Claims
describe what the speaker reported, not independently verified world events.
"""

from dataclasses import dataclass

from ..ontology import Label
from ..semantic_generator.frames import frame_by_lemma
from .verb_operators import MODALS, OPERATORS, Operator, be_form, validate_chain
from .observed_clauses import ObservedClause
from .mined_realization import mined_lexeme


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
    EventLexeme('go', 'went', 'goes', 'going', 'gone', 'directional'),
    EventLexeme('have', 'had', 'has', 'having', 'had', 'possession', False),
    EventLexeme('be', 'was', 'is', 'being', 'been', 'location', False),
)
EVENT_BY_LEMMA = {item.lemma: item for item in EVENT_LEXEMES}
ACTION_LEXEMES = tuple(item for item in EVENT_LEXEMES if item.frame.predicate_type == Label.ACTION)
STATE_LEXEMES = tuple(item for item in EVENT_LEXEMES if item.frame.predicate_type == Label.STATE)


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
    permitter: EventArgument | None = None
    observation: ObservedClause | None = None
    mined: bool = False

    def __post_init__(self):
        if self.observation is not None:
            if (not isinstance(self.observation, ObservedClause) or self.predicate != '@clause'
                    or self.arguments or self.operators or self.permitter or self.time != 'today'
                    or self.phase != 'unspecified' or self.tense != 'present' or self.mined):
                raise ValueError('Observed clauses use their own status, without invented roles/tense/aspect')
            return
        if self.predicate not in EVENT_BY_LEMMA and not self.mined:
            raise ValueError(f'Unsupported event predicate: {self.predicate}')
        if not self.time or self.time != self.time.strip():
            raise ValueError('An event needs an explicit time context')
        if self.phase not in (('unspecified', 'reported') if self.mined else ('unspecified', 'ongoing', 'completed')) or self.tense not in ('present', 'past', 'future', 'modal'):
            raise ValueError('Unsupported event phase or tense')
        validate_chain(self.operators, self.tense, is_state=self.is_state)
        if any(op != 'not' for op in self.operators) and self.phase != 'unspecified':
            raise ValueError('Phase belongs to the bare action; operator constructions use unspecified phase')
        if self.tense == 'future' and self.phase != 'unspecified':
            raise ValueError('Future does not establish ongoing/completed phase')
        if self.is_state and self.phase != 'unspecified':
            raise ValueError('Location and possession are states, not completed/ongoing actions')
        if (self.lexeme.bounded and self.operators in (('have',), ('not', 'have'))):
            raise ValueError('Use completed phase for a bare perfect bounded action')
        if ('let' in self.operators) != (self.permitter is not None):
            raise ValueError('Permissive let requires a separate permitter; other chains cannot carry one')
        if self.permitter is not None:
            if (self.permitter.role != 'permitter' or self.permitter.label != 'PERSON'
                    or not self.permitter.text or self.permitter.text != self.permitter.text.strip()):
                raise ValueError('Let requires a named PERSON permitter argument')
        if (self.tense == 'past' and self.phase == 'unspecified' and self.operators in ((), ('not',))
                and self.lexeme.bounded):
            raise ValueError('A bare past bounded action requires an explicit completed phase')
        if self.mined:
            self.lexeme  # A real dictionary paradigm, never guessed regular forms.
            if self.phase == 'reported' and (self.tense != 'past' or self.operators not in ((), ('not',))):
                raise ValueError('A reported mined predication requires bare positive/negative past')
            if self.phase == 'unspecified' and self.operators in ((), ('not',)):
                raise ValueError('A bare mined predication must explicitly use reported phase')
            if tuple(a.role for a in self.arguments) not in (('agent',), ('agent', 'complement')):
                raise ValueError('Mined phrases need an agent and optional intact complement')
            if self.arguments[0].label != 'PERSON' or self.arguments[0].text != 'user':
                raise ValueError('The reviewed mined realization requires the user as subject')
            if len(self.arguments) == 2 and (self.arguments[1].label != 'UNRESOLVED' or not self.arguments[1].text.strip()):
                raise ValueError('A mined complement keeps its unresolved semantic type')
            if any(arg.text != arg.text.strip() for arg in self.arguments):
                raise ValueError('Mined argument text must be trimmed')
            return
        frame = EVENT_BY_LEMMA[self.predicate].frame
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
        if 'let_us' in self.operators and self.agent != 'user':
            raise ValueError('The current let_us proposal requires the user as its initiating actor')

    @property
    def lexeme(self):
        return mined_lexeme(self.predicate) if self.mined else EVENT_BY_LEMMA[self.predicate]

    @property
    def identity(self):
        if self.observation is not None:
            return self.predicate, self.observation.text
        return (self.predicate, self.arguments, self.time, 'mined') if self.mined else (self.predicate, self.arguments, self.time)

    @property
    def key(self):
        if self.observation is not None:
            status = self.observation.status
            return self.identity, 'quotation', ('asserted' if status in ('reported', 'denied') else status)
        # One current claim per event and operator path. A correction may change
        # outer polarity/phase/tense. Inner negation changes what is wanted or
        # needed: not(want(X)) and want(not(X)) must coexist, not overwrite.
        path = self.operators[1:] if self.operators[:1] == ('not',) else self.operators
        return self.identity, path, self.permitter

    @property
    def is_state(self):
        if self.observation is not None or self.mined:
            return False
        return EVENT_BY_LEMMA[self.predicate].frame.predicate_type == Label.STATE

    @property
    def agent(self):
        if self.observation is not None or self.mined:
            return 'user'
        frame = EVENT_BY_LEMMA[self.predicate].frame
        role = frame.pattern(EVENT_BY_LEMMA[self.predicate].pattern).subject
        return next(arg.text for arg in self.arguments if arg.role == role)


def completion_evidence(event: Event) -> str:
    """Conservative yes/no/unknown for completion of this precise event.

    A negated modal is not a negated action. Starting, stopping and attempting
    do not establish completion. Nested success under a wish remains unknown.
    """
    if event.observation is not None or event.mined or event.is_state or event.tense in ('future', 'modal'):
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


def state_evidence(event: Event) -> str:
    if event.is_state and event.tense in ('present', 'past') and event.operators in ((), ('not',)):
        return 'no' if event.operators else 'yes'
    return 'unknown'


def _core(event: Event, form: str, style: int, subject='I') -> str:
    if event.mined:
        lexeme = event.lexeme
        word = lexeme.lemma if form in ('base', 'present') else getattr(lexeme, form)
        tail = next((a.text for a in event.arguments if a.role == 'complement'), '')
        return word + (' ' + tail if tail else '')
    lexeme = EVENT_BY_LEMMA[event.predicate]
    frame = lexeme.frame
    pattern = frame.pattern(lexeme.pattern)
    pronoun = any(arg.role == 'theme' and arg.text.lower() in ('it', 'them', 'him', 'her', 'us', 'you', 'me')
                  for arg in event.arguments)
    if frame.particles and lexeme.pattern == 'joined' and (style % 2 or pronoun):
        pattern = frame.pattern('separated')
    if event.predicate == 'be':
        head = be_form(form, subject)
    else:
        head = (frame.head_lemma or frame.lemma) if form in ('base', 'present') else getattr(lexeme, form)
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


def _person(text, perspective, *, object_case=False):
    if text == 'user':
        return ('me' if object_case else 'I') if perspective == 'user' else ('you' if object_case else 'You')
    if text == 'assistant':
        return ('you' if object_case else 'You') if perspective == 'user' else ('me' if object_case else 'I')
    return text


def _phrase(event: Event, operators: tuple[str, ...], form: str, style: int, subject='I', perspective='user') -> str:
    if not operators:
        return _core(event, form, style, subject)
    head, *tail = operators
    tail = tuple(tail)
    if head == 'not':
        child = tail[0] if tail else event.predicate
        auxiliary = child in MODALS or child in ('be', 'be_able_to', 'be_going_to') or child == 'have' and bool(tail)
        if auxiliary and form in ('past', 'third', 'present', 'modal'):
            phrase = _phrase(event, tail, form, style, subject, perspective)
            first, separator, rest = phrase.partition(' ')
            return ('cannot' if first == 'can' else first + ' not') + (separator + rest if separator else '')
        if form in ('past', 'third', 'present'):
            helper = 'did' if form == 'past' else 'does' if form == 'third' else 'do'
            return helper + ' not ' + _phrase(event, tail, 'base', style, subject, perspective)
        return 'not ' + _phrase(event, tail, form, style, subject, perspective)
    operator = OPERATORS[head]
    if head in MODALS:
        return head + ' ' + _phrase(event, tail, 'base', style, subject, perspective)
    if head == 'let_us':
        return "Let's " + _phrase(event, tail, 'base', style, 'we', perspective)
    if head in ('be', 'be_able_to', 'be_going_to'):
        word = be_form(form, subject)
    elif head == 'have_to':
        word = 'have' if form in ('base', 'present') else getattr(operator, form)
    else:
        word = head if form in ('base', 'present') else (operator.participle or operator.past) if form == 'participle' else getattr(operator, form)
    if word is None:
        raise ValueError(f'Unsupported {head} form: {form}')
    if head == 'let':
        causee = _person(event.agent, perspective, object_case=True)
        return word + ' ' + causee + ' ' + _phrase(event, tail, 'base', style, _person(event.agent, perspective), perspective)
    if head in ('be_able_to', 'be_going_to'):
        word += ' able' if head == 'be_able_to' else ' going'
    if operator.complement in ('to', 'have_to', 'be_able_to', 'be_going_to'):
        if tail and tail[0] == 'not':
            if head == 'need':
                # Avoid the ambiguous modal reading of "need not": this scope
                # requires non-performance, not absence of necessity.
                return word + ' to avoid ' + _phrase(event, tail[1:], 'gerund', style, subject, perspective)
            if head in ('have_to', 'be_able_to', 'be_going_to'):
                return word + ' to ' + _phrase(event, tail, 'base', style, subject, perspective)
            return word + ' not to ' + _phrase(event, tail[1:], 'base', style, subject, perspective)
        return word + ' to ' + _phrase(event, tail, 'base', style, subject, perspective)
    complement_form = 'participle' if operator.complement == 'participle' else 'gerund'
    return word + ' ' + _phrase(event, tail, complement_form, style, subject, perspective)


def event_statement(event: Event, *, perspective='memory', style=0) -> str:
    if perspective not in ('memory', 'user') or style not in range(5):
        raise ValueError('Unsupported event rendering')
    if event.observation is not None:
        return event.observation.statement(perspective=perspective, style=style)
    subject = _person(event.permitter.text if event.permitter else event.agent, perspective)
    first_or_second = subject in ('I', 'You')
    when = ' ' + event.time
    if event.tense == 'future':
        phrase = 'will ' + _phrase(event, event.operators, 'base', style, subject, perspective)
    elif event.phase == 'reported':
        phrase = ('did not ' + _core(event, 'base', style)) if event.operators else _core(event, 'past', style)
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
        form = 'modal' if event.tense == 'modal' else 'past' if event.tense == 'past' else 'present' if first_or_second else 'third'
        # The base form is the finite present for first/second person.
        phrase = _phrase(event, event.operators, form, style, subject, perspective)
    if event.operators[:1] == ('let_us',):
        quoted = phrase + when + '.'
        return quoted if perspective == 'user' else 'You proposed: "' + quoted + '"'
    if style >= 3 and perspective == 'memory':
        # Hold-out paraphrases change reporting language, not semantic operators.
        original = event_statement(event, perspective='user', style=style - 3)
        return f'You reported: "{original}"' if style == 3 else f'You stated: "{original}"'
    if style == 2:
        # Unambiguous contractions only: 'd and 's could hide had/would/is/has.
        # May not has no ordinary contracted form and stays explicit.
        contractions = {'cannot': "can't", 'could not': "couldn't", 'must not': "mustn't",
                        'is not': "isn't", 'are not': "aren't", 'was not': "wasn't", 'were not': "weren't",
                        'do not': "don't", 'does not': "doesn't", 'did not': "didn't",
                        'have not': "haven't", 'has not': "hasn't", 'had not': "hadn't", 'will not': "won't"}
        for full, short in contractions.items():
            if phrase == full or phrase.startswith(full + ' '):
                phrase = short + phrase[len(full):]
                break
    return subject + ' ' + phrase + when + '.'


def event_question(event: Event, *, style=0) -> str:
    if event.observation is not None:
        return event.observation.question()
    if event.mined:
        return f'Did I {_core(event, "base", style)} {event.time}?'
    if event.is_state:
        subject = _person(event.agent, 'user')
        if style >= 3:
            form = 'past' if event.tense == 'past' else 'present' if subject in ('I', 'You') else 'third'
            prefix = 'Is there confirmation that' if style == 3 else 'Can you tell whether'
            return f'{prefix} {subject} {_core(event, form, style, subject)} {event.time}?'
        if event.predicate == 'be':
            be = be_form('past' if event.tense == 'past' else 'present', subject)
            location = next(arg.text for arg in event.arguments if arg.role == 'location')
            return f'{be.capitalize()} {subject} in {location} {event.time}?'
        helper = 'Did' if event.tense == 'past' else 'Do' if subject in ('I', 'You') else 'Does'
        return f'{helper} {subject} {_core(event, "base", style, subject)} {event.time}?'
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
