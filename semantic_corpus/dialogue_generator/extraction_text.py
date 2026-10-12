"""Text commands for the pilot memory updater; IDs/evidence belong to code.

This is a small structured language, not unrestricted natural language. Parsing
checks the command contract and state transitions, not the truth of a claim.
"""

from dataclasses import dataclass
import re

from .actions import ACTION_BY_RELATION
from .models import AssistantAction, Delta, Evidence, Fact, Memory, Replacement, Turn, apply_delta, EventClaim, EventReplacement
from .events import Event, EventArgument
from .observed_clauses import ObservedClause
from .mined_subjects import MinedSubject

EXTRACTION_TEXT_VERSION = 'extraction-text-v8'
PROMPT_PREFIX = 'Current memory:\n'
STATE_RELATIONS = frozenset(('location', 'learning_location', 'meeting_day', 'feeling', 'selection', 'drink'))
ESCAPES = {'\\': '\\', '|': '|', 'n': '\n', 'r': '\r', 't': '\t'}
ENCODE_ESCAPES = {value: '\\' + key for key, value in ESCAPES.items()}

EXTRACT_SYSTEM = (
    'Update memory from the latest user message, one command per line; no JSON, IDs or evidence quotations. '
    'SET subject | relation | modality | polarity | value adds/replaces a fact. '
    'REMOVE subject | relation | modality withdraws it. Otherwise NO_CHANGE alone. '
    'Suffix _CONTEXT cites the last assistant for contextual replies. '
    'Relations: location, learning_location, meeting_day, feeling, selection, drink, visited, bought, called, sent, read, completed_break. '
    'Modality: asserted, desired/planned(actions). Polarity: positive/negative. Speaker=user. '
    'learning_location concerns the current study topic, not current location. '
    'Questions, thanks, unresolved references add nothing. Preserve known fragment meaning without guessing. '
    'EVENT predicate | time | scope | phase | tense | role=TYPE:value ... '
    'Scope: outer>inner or none; not negates its child. Phase: unspecified/ongoing/completed. '
    'Tense: present/past/future/modal; can/could/must/may/might use modal. let needs permitter=PERSON:value. '
    'Lexical be/have are states; auxiliaries are operators. Wishes/plans/suggestions/attempts/starts/stops never prove completion. '
    'PARTIAL uses EVENT fields plus missing=role,role; only known arguments, unspecified phase, absent time=unspecified. '
    'MINED uses EVENT fields plus subject=TYPE:text | number=singular/plural; optional voice=passive(default active), complement=UNRESOLVED:text. '
    'Grammatical roles object/indirect_object/oblique_PREPOSITION/by_phrase locate unique complement phrases. '
    'Resolve I/me/you to user/assistant; retain user perspective. Types: pool labels/UNRESOLVED; weather it=IMPERSONAL. '
    'MINED reported phase means bare past, not completion. Separate independent events. '
    'QUOTE status | text: reported/denied/desired/planned/suggested/possible/required/permitted wording. '
    'Quotes never confirm embedded events; partial events never prove completion. '
    'REMOVE_ prefixes EVENT/MINED/QUOTE/PARTIAL to withdraw an exact report.'
)


def _escape(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError('Text fields must be nonempty strings without surrounding whitespace')
    return ''.join(ENCODE_ESCAPES.get(char, char) for char in value)


def _fields(values) -> str:
    return ' | '.join(_escape(value) for value in values)


def _read_fields(text: str) -> tuple[str, ...]:
    fields, chars = [], []
    escaped = False
    for char in text:
        if escaped:
            if char not in ESCAPES:
                raise ValueError('Unknown text-field escape')
            chars.append(ESCAPES[char])
            escaped = False
        elif char == '\\':
            escaped = True
        elif char == '|':
            fields.append(''.join(chars).strip())
            chars = []
        else:
            chars.append(char)
    if escaped:
        raise ValueError('Incomplete text-field escape')
    fields.append(''.join(chars).strip())
    for field in fields:
        _escape(field)
    return tuple(fields)


def _check_key(subject: str, relation: str, modality: str) -> None:
    _escape(subject)
    if relation not in STATE_RELATIONS and relation not in ACTION_BY_RELATION:
        raise ValueError(f'Unsupported fact relation: {relation}')
    allowed = ('asserted', 'desired', 'planned') if relation in ACTION_BY_RELATION else ('asserted',)
    if modality not in allowed:
        raise ValueError(f'Unsupported fact modality: {modality}')


def _fact_fields(fact: Fact) -> tuple[str, ...]:
    _check_key(*fact.key)
    if fact.polarity not in ('positive', 'negative'):
        raise ValueError(f'Unsupported fact polarity: {fact.polarity}')
    return fact.subject, fact.relation, fact.modality, fact.polarity, fact.value


@dataclass(frozen=True)
class Command:
    operation: str
    subject: str
    relation: str
    modality: str
    polarity: str | None = None
    value: str | None = None

    @property
    def key(self):
        return 'fact', (self.subject, self.relation, self.modality)


@dataclass(frozen=True)
class EventCommand:
    operation: str
    event: Event

    @property
    def key(self):
        return 'event', self.event.key


def _event_fields(event: Event) -> tuple[str, ...]:
    return (event.predicate, event.time, '>'.join(event.operators) or 'none', event.phase, event.tense,
            *((f'subject={event.subject.label}:{event.subject.text}', f'number={event.subject.number}') if event.subject else ()),
            *((f'voice={event.voice}',) if event.voice != 'active' else ()),
            *((f'missing={",".join(event.missing_roles)}',) if event.missing_roles else ()),
            *(f'{arg.role}={arg.label}:{arg.text}' for arg in event.arguments),
            *((f'permitter={event.permitter.label}:{event.permitter.text}',) if event.permitter else ()))


def _read_event(fields, *, partial=False) -> Event:
    if len(fields) < 6:
        raise ValueError('An event needs predicate, time, scope, phase, tense and typed arguments')
    predicate, time, scope, phase, tense, *arguments = fields
    roles = []
    permitter = None
    missing = None
    for argument in arguments:
        role, separator, value = argument.partition('=')
        if role == 'missing':
            if not partial or missing is not None or not separator or not value:
                raise ValueError('PARTIAL requires one nonempty missing=role,role field')
            missing = tuple(value.split(','))
            continue
        label, colon, text = value.partition(':')
        if not separator or not colon:
            raise ValueError('Expected role=TYPE:value')
        argument = EventArgument(role, text, label)
        if role == 'permitter':
            if permitter is not None:
                raise ValueError('Repeated permitter')
            permitter = argument
        else:
            roles.append(argument)
    if partial and missing is None:
        raise ValueError('PARTIAL requires explicit missing roles')
    return Event(predicate, tuple(sorted(roles)), time, () if scope == 'none' else tuple(scope.split('>')),
                 phase, tense, permitter, missing_roles=missing or ())


def _event_payload(event):
    if event.missing_roles:
        return 'PARTIAL', _event_fields(event)
    if event.observation is not None:
        return 'QUOTE', (event.observation.status, event.observation.text)
    return ('MINED' if event.mined else 'EVENT'), _event_fields(event)


def _parse_event_payload(operation, fields):
    if operation == 'PARTIAL':
        return _read_event(fields, partial=True)
    if operation == 'QUOTE':
        if len(fields) != 2:
            raise ValueError('QUOTE needs status and text')
        return Event('@clause', (), observation=ObservedClause(fields[1], fields[0]))
    if operation == 'MINED':
        if len(fields) < 6:
            raise ValueError('MINED needs predicate, time, scope, phase, tense and subject')
        predicate, time, scope, phase, tense, *args = fields
        roles, permitter, subject, number, voice = [], None, None, None, None
        for value in args:
            role, eq, tail = value.partition('=')
            if role == 'voice' and eq:
                if voice is not None:
                    raise ValueError('Repeated voice')
                voice = tail
                continue
            if role == 'number' and eq:
                if number is not None:
                    raise ValueError('Repeated subject number')
                number = tail
                continue
            label, colon, text = tail.partition(':')
            if not eq or not colon:
                raise ValueError('Expected role=TYPE:value')
            arg = EventArgument(role, text, label)
            if role == 'subject':
                if subject is not None:
                    raise ValueError('Repeated subject')
                subject = (text, label)
            elif role == 'permitter':
                if permitter:
                    raise ValueError('Repeated permitter')
                permitter = arg
            else:
                roles.append(arg)
        if (subject is None) != (number is None):
            raise ValueError('A MINED subject and its number must be supplied together')
        return Event(predicate, tuple(sorted(roles)), time, () if scope == 'none' else tuple(scope.split('>')),
                     phase, tense, permitter, mined=True,
                     subject=MinedSubject(*subject, number) if subject else None, voice=voice if voice is not None else 'active')
    return _read_event(fields)


def parse_commands(text: str) -> tuple[Command | EventCommand, ...]:
    """Reject malformed/contradictory output in full before applying anything."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Expected text update commands')
    if text.strip() == 'NO_CHANGE':
        return ()
    commands, seen = [], set()
    for line in text.strip().split('\n'):
        operation, separator, payload = line.strip().partition(' ')
        if not separator or operation not in ('SET', 'SET_CONTEXT', 'REMOVE', 'EVENT', 'EVENT_CONTEXT', 'REMOVE_EVENT',
                                             'MINED', 'MINED_CONTEXT', 'REMOVE_MINED', 'QUOTE', 'QUOTE_CONTEXT', 'REMOVE_QUOTE',
                                             'PARTIAL', 'PARTIAL_CONTEXT', 'REMOVE_PARTIAL'):
            raise ValueError('Expected SET, EVENT, REMOVE, REMOVE_EVENT or NO_CHANGE alone')
        fields = _read_fields(payload)
        if operation.startswith(('EVENT', 'MINED', 'QUOTE', 'PARTIAL', 'REMOVE_EVENT', 'REMOVE_MINED', 'REMOVE_QUOTE', 'REMOVE_PARTIAL')):
            kind = operation.removeprefix('REMOVE_').removesuffix('_CONTEXT')
            command = EventCommand(operation, _parse_event_payload(kind, fields))
        else:
            if len(fields) != (3 if operation == 'REMOVE' else 5):
                raise ValueError(f'Wrong number of fields for {operation}')
            command = Command(operation, *fields)
            _check_key(command.subject, command.relation, command.modality)
            if operation != 'REMOVE' and command.polarity not in ('positive', 'negative'):
                raise ValueError('Unsupported fact polarity')
        if command.key in seen:
            raise ValueError('Repeated or conflicting update key')
        seen.add(command.key)
        commands.append(command)
    return tuple(commands)


def render_prompt(memory: Memory, user_turn: Turn, previous_assistant: Turn | None = None) -> str:
    if user_turn.role != 'user' or (previous_assistant and previous_assistant.role != 'assistant'):
        raise ValueError('Expected a user turn and optional previous assistant turn')
    lines = ['Current memory:']
    lines.extend(_fields(_fact_fields(fact)) for fact in memory.facts)
    if not memory.facts:
        lines.append('(none)')
    if memory.events:
        lines.extend(('', 'Event memory:'))
        for claim in memory.events:
            kind, fields = _event_payload(claim.event)
            lines.append(('' if kind == 'EVENT' else kind + ' ') + _fields(fields))
    lines.extend(('', 'Assistant actions (earliest first):'))
    for action in memory.assistant_actions:
        if action.kind not in ('options', 'suggestion') or not action.items:
            raise ValueError('Unsupported assistant action')
        lines.append(_fields((action.kind, *action.items)))
    if not memory.assistant_actions:
        lines.append('(none)')
    # Each utterance occupies exactly one escaped line, so its text cannot
    # introduce a second structural heading or command into this input format.
    lines.extend(('', 'Last assistant message: ' + ('text: ' + _escape(previous_assistant.text) if previous_assistant else '(none)'),
                  'Latest user message: ' + _escape(user_turn.text)))
    return '\n'.join(lines)


def _number(fact_id: str) -> int:
    if not re.fullmatch(r'f[1-9][0-9]*', fact_id):
        raise ValueError('The pilot updater requires internal fact IDs f1, f2, ...')
    return int(fact_id[1:])


def decode_update(text: str, memory: Memory, user_turn: Turn,
                  previous_assistant: Turn | None = None) -> Delta:
    """Allocate fresh IDs and exact source quotes, then validate atomically.

    SET_CONTEXT explicitly requests both visible utterances as evidence. This
    records provenance; it cannot by itself verify a model's interpretation.
    """
    render_prompt(memory, user_turn, previous_assistant)
    next_id = max((_number(f.id) for f in (*memory.facts, *memory.retired)), default=0) + 1
    additions, replacements, removals = [], [], []
    event_ids = [claim.id for claim in (*memory.events, *memory.retired_events)]
    if any(not re.fullmatch(r'e[1-9][0-9]*', key) for key in event_ids):
        raise ValueError('The pilot updater requires internal event IDs e1, e2, ...')
    next_event_id = max((int(key[1:]) for key in event_ids), default=0) + 1
    event_add, event_replace, event_remove = [], [], []
    for command in parse_commands(text):
        if isinstance(command, EventCommand):
            old = memory.find_event(command.event)
            if command.operation.startswith('REMOVE_'):
                if old is None or old.event != command.event:
                    raise ValueError('REMOVE_EVENT must match an active event exactly')
                event_remove.append(old.id)
                continue
            evidence = (Evidence(user_turn.id, user_turn.text),)
            if command.operation.endswith('_CONTEXT'):
                if previous_assistant is None:
                    raise ValueError('EVENT_CONTEXT requires a previous assistant message')
                evidence += (Evidence(previous_assistant.id, previous_assistant.text),)
            claim = EventClaim(f'e{next_event_id}', command.event, evidence)
            next_event_id += 1
            if old is None:
                event_add.append(claim)
            else:
                event_replace.append(EventReplacement(old.id, claim))
            continue
        old = memory.find(command.subject, command.relation, command.modality)
        if command.operation == 'REMOVE':
            if old is None:
                raise ValueError('REMOVE must refer to an active fact')
            removals.append(old.id)
            continue
        evidence = (Evidence(user_turn.id, user_turn.text),)
        if command.operation == 'SET_CONTEXT':
            if previous_assistant is None:
                raise ValueError('SET_CONTEXT requires a previous assistant message')
            evidence += (Evidence(previous_assistant.id, previous_assistant.text),)
        fact = Fact(f'f{next_id}', command.subject, command.relation, command.value, evidence,
                    command.polarity, command.modality)
        next_id += 1
        if old is None:
            additions.append(fact)
        else:
            replacements.append(Replacement(old.id, fact))
    delta = Delta(tuple(additions), tuple(replacements), tuple(removals),
                  tuple(event_add), tuple(event_replace), tuple(event_remove))
    turns = ((previous_assistant,) if previous_assistant else ()) + (user_turn,)
    apply_delta(memory, delta, turns)
    return delta


def render_update(transition) -> str:
    """Refuse to export an update if text decoding would lose any audit detail."""
    lines = []
    current = (Evidence(transition.turn.id, transition.turn.text),)
    previous = transition.previous_assistant
    contextual = current + ((Evidence(previous.id, previous.text),) if previous else ())
    new_facts = (*transition.delta.add, *(r.fact for r in transition.delta.replace))
    # IDs need not appear in the text, but their allocation order must be kept.
    for fact in sorted(new_facts, key=lambda fact: _number(fact.id)):
        if fact.evidence == current:
            operation = 'SET'
        elif previous and fact.evidence == contextual:
            operation = 'SET_CONTEXT'
        else:
            raise ValueError('Text updates support full current/previous-turn evidence only')
        lines.append(operation + ' ' + _fields(_fact_fields(fact)))
    active = {fact.id: fact for fact in transition.before.facts}
    for fact_id in transition.delta.invalidate:
        if fact_id not in active:
            raise ValueError('Cannot remove an inactive fact')
        lines.append('REMOVE ' + _fields(active[fact_id].key))
    new_events = (*transition.delta.event_add, *(r.claim for r in transition.delta.event_replace))
    for claim in sorted(new_events, key=lambda claim: int(claim.id[1:])):
        if claim.evidence == current:
            operation = 'EVENT'
        elif previous and claim.evidence == contextual:
            operation = 'EVENT_CONTEXT'
        else:
            raise ValueError('Text events support full current/previous-turn evidence only')
        kind, fields = _event_payload(claim.event)
        lines.append(operation.replace('EVENT', kind) + ' ' + _fields(fields))
    active_events = {claim.id: claim for claim in transition.before.events}
    for claim_id in transition.delta.event_invalidate:
        if claim_id not in active_events:
            raise ValueError('Cannot remove an inactive event')
        kind, fields = _event_payload(active_events[claim_id].event)
        lines.append('REMOVE_' + kind + ' ' + _fields(fields))
    text = '\n'.join(lines) if lines else 'NO_CHANGE'
    if decode_update(text, transition.before, transition.turn, previous) != transition.delta:
        raise ValueError('Text update cannot preserve the original delta')
    return text


def parse_prompt(text: str) -> tuple[Memory, Turn, Turn | None]:
    """Reconstruct visible state for evaluation, assigning local internal IDs.

    Original IDs/retired facts/evidence are deliberately absent from the model
    input. Use the real Memory with decode_update when applying live updates.
    """
    if not text.startswith(PROMPT_PREFIX):
        raise ValueError('Expected a text Extraction prompt')
    try:
        facts_text, tail = text[len(PROMPT_PREFIX):].split('\n\nAssistant actions (earliest first):\n', 1)
        actions_text, tail = tail.split('\n\nLast assistant message: ', 1)
        assistant_text, user_text = tail.split('\nLatest user message: ', 1)
    except ValueError as error:
        raise ValueError('Malformed text Extraction prompt') from error
    facts = []
    events = []
    if '\n\nEvent memory:\n' in facts_text:
        facts_text, events_text = facts_text.split('\n\nEvent memory:\n', 1)
        for index, line in enumerate(events_text.split('\n'), 1):
            kind, _, tail = line.partition(' ')
            if kind not in ('QUOTE', 'MINED', 'PARTIAL'):
                kind, tail = 'EVENT', line
            events.append(EventClaim(f'e{index}', _parse_event_payload(kind, _read_fields(tail)), ()))
    if len({claim.key for claim in events}) != len(events):
        raise ValueError('Conflicting active events in prompt')
    if facts_text != '(none)':
        for index, line in enumerate(facts_text.split('\n'), 1):
            fields = _read_fields(line)
            if len(fields) != 5:
                raise ValueError('Malformed memory fact')
            subject, relation, modality, polarity, value = fields
            fact = Fact(f'f{index}', subject, relation, value, (), polarity, modality)
            _fact_fields(fact)
            facts.append(fact)
    if len({fact.key for fact in facts}) != len(facts):
        raise ValueError('Conflicting active facts in prompt')
    actions = []
    if actions_text != '(none)':
        for index, line in enumerate(actions_text.split('\n'), 1):
            fields = _read_fields(line)
            if len(fields) < 2 or fields[0] not in ('options', 'suggestion'):
                raise ValueError('Malformed assistant action')
            actions.append(AssistantAction(fields[0], fields[1:], f'context{index}'))
    def utterance(value):
        fields = _read_fields(value)
        if len(fields) != 1:
            raise ValueError('Malformed utterance')
        return fields[0]
    if assistant_text != '(none)' and not assistant_text.startswith('text: '):
        raise ValueError('Malformed previous assistant message')
    previous = None if assistant_text == '(none)' else Turn('a1', 'assistant', utterance(assistant_text[6:]))
    user = Turn('u1', 'user', utterance(user_text))
    memory = Memory(tuple(facts), assistant_actions=tuple(actions), events=tuple(events))
    if render_prompt(memory, user, previous) != text:
        raise ValueError('Noncanonical text Extraction prompt')
    return memory, user, previous
