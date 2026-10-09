"""Small, explicit dialogue memory for controlled synthetic experiments.

Facts are claims made in a conversation, not independently verified world facts.
The checker validates references and updates, not arbitrary English semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from .events import Event, completion_evidence, state_evidence

SCHEMA_VERSION = "dialogue-memory-v6"


@dataclass(frozen=True)
class Turn:
    id: str
    role: str
    text: str


@dataclass(frozen=True)
class Evidence:
    turn_id: str
    quote: str


@dataclass(frozen=True)
class Fact:
    id: str
    subject: str
    relation: str
    value: str
    evidence: tuple[Evidence, ...]
    polarity: str = "positive"
    modality: str = "asserted"

    @property
    def key(self) -> tuple[str, str, str]:
        # Pilot relations are single-valued within a modality and subject.
        return self.subject, self.relation, self.modality


@dataclass(frozen=True)
class Replacement:
    old_id: str
    fact: Fact


@dataclass(frozen=True)
class EventClaim:
    id: str
    event: Event
    evidence: tuple[Evidence, ...]

    @property
    def key(self):
        return self.event.key


@dataclass(frozen=True)
class EventReplacement:
    old_id: str
    claim: EventClaim


@dataclass(frozen=True)
class Delta:
    add: tuple[Fact, ...] = ()
    replace: tuple[Replacement, ...] = ()
    invalidate: tuple[str, ...] = ()
    event_add: tuple[EventClaim, ...] = ()
    event_replace: tuple[EventReplacement, ...] = ()
    event_invalidate: tuple[str, ...] = ()


@dataclass(frozen=True)
class AssistantAction:
    kind: str
    items: tuple[str, ...]
    source_turn: str


@dataclass(frozen=True)
class Memory:
    facts: tuple[Fact, ...] = ()
    retired: tuple[Fact, ...] = ()
    assistant_actions: tuple[AssistantAction, ...] = ()
    events: tuple[EventClaim, ...] = ()
    retired_events: tuple[EventClaim, ...] = ()

    def find(self, subject: str, relation: str, modality: str = "asserted") -> Fact | None:
        return next((f for f in self.facts if f.key == (subject, relation, modality)), None)

    def prompt_data(self) -> dict:
        # Retired claims stay in the audit record, not in current-state prompts.
        return {"facts": [asdict(f) for f in self.facts],
                "assistant_actions": [asdict(a) for a in self.assistant_actions],
                "events": [asdict(event) for event in self.events]}

    def find_event(self, event: Event) -> EventClaim | None:
        return next((claim for claim in self.events if claim.key == event.key), None)


def apply_delta(memory: Memory, delta: Delta, turns: tuple[Turn, ...]) -> Memory:
    """Apply one user's update atomically; refuse stale IDs or invented evidence."""
    if not turns or turns[-1].role != "user":
        raise ValueError("A memory delta must accompany a user turn")
    by_turn = {turn.id: turn for turn in turns}
    if len(by_turn) != len(turns):
        raise ValueError("Duplicate turn IDs")
    active = {f.id: f for f in memory.facts}
    historical = {f.id for f in memory.retired}
    if len(active) != len(memory.facts) or len(historical) != len(memory.retired) or active.keys() & historical:
        raise ValueError("Duplicate fact IDs in memory")
    removed = (*delta.invalidate, *(r.old_id for r in delta.replace))
    if len(set(removed)) != len(removed) or any(fid not in active for fid in removed):
        raise ValueError("An update must refer to distinct active fact IDs")
    for replacement in delta.replace:
        if active[replacement.old_id].key != replacement.fact.key:
            raise ValueError("Replacement cannot change subject, relation or modality")
    additions = (*delta.add, *(r.fact for r in delta.replace))
    seen = set(active) | historical | {claim.id for claim in (*memory.events, *memory.retired_events)}
    for fact in additions:
        if not all(isinstance(s, str) and s.strip() for s in
                   (fact.id, fact.subject, fact.relation, fact.value)) or fact.id in seen:
            raise ValueError("New facts need nonempty fields and fresh IDs")
        seen.add(fact.id)
        if fact.polarity not in ("positive", "negative") or fact.modality not in ("asserted", "desired", "planned"):
            raise ValueError("Unsupported polarity or modality")
        if not fact.evidence or not any(e.turn_id == turns[-1].id for e in fact.evidence):
            raise ValueError("A new fact needs evidence from the current user turn")
        for evidence in fact.evidence:
            source = by_turn.get(evidence.turn_id)
            if source is None or not evidence.quote or evidence.quote not in source.text:
                raise ValueError("Evidence must quote an existing turn exactly")
    kept = tuple(f for f in memory.facts if f.id not in removed)
    facts = kept + additions
    if len({f.key for f in facts}) != len(facts):
        raise ValueError("Conflicting active facts: use an explicit replacement")
    event_active = {claim.id: claim for claim in memory.events}
    event_history = {claim.id for claim in memory.retired_events}
    if (len(event_active) != len(memory.events) or len(event_history) != len(memory.retired_events)
            or event_active.keys() & event_history or (set(active) | historical) & (set(event_active) | event_history)):
        raise ValueError('Duplicate event IDs in memory')
    event_removed = (*delta.event_invalidate, *(r.old_id for r in delta.event_replace))
    if len(set(event_removed)) != len(event_removed) or any(key not in event_active for key in event_removed):
        raise ValueError('Event updates must refer to distinct active IDs')
    for replacement in delta.event_replace:
        if event_active[replacement.old_id].key != replacement.claim.key:
            raise ValueError('Event replacement cannot change identity or operator path')
    event_added = (*delta.event_add, *(r.claim for r in delta.event_replace))
    seen.update(event_active)
    seen.update(event_history)
    for claim in event_added:
        if not claim.id or claim.id in seen:
            raise ValueError('New events need fresh IDs')
        seen.add(claim.id)
        if not isinstance(claim.event, Event):
            raise ValueError('Event claims need a validated Event')
        if not claim.evidence or not any(e.turn_id == turns[-1].id for e in claim.evidence):
            raise ValueError('An event needs evidence from the current user turn')
        for evidence in claim.evidence:
            source = by_turn.get(evidence.turn_id)
            if source is None or not evidence.quote or evidence.quote not in source.text:
                raise ValueError('Event evidence must quote an existing turn exactly')
    events = tuple(claim for claim in memory.events if claim.id not in event_removed) + event_added
    if len({claim.key for claim in events}) != len(events):
        raise ValueError('Conflicting active event claims: use an explicit replacement')
    return Memory(facts, memory.retired + tuple(active[fid] for fid in removed), memory.assistant_actions,
                  events, memory.retired_events + tuple(event_active[key] for key in event_removed))


@dataclass(frozen=True)
class Transition:
    turn: Turn
    before: Memory
    delta: Delta
    after: Memory
    previous_assistant: Turn | None


@dataclass(frozen=True)
class Query:
    kind: str
    subject: str = "user"
    relation: str = ""
    value: str = ""
    event: Event | None = None


def answer_query(memory: Memory, query: Query) -> tuple[str, str, tuple[str, ...]]:
    """Reference answer, dialogue action and supporting IDs from visible state."""
    if query.kind not in ("lookup", "yes_no", "clarify_option", "capability", "event_completion", "event_state", "observed_report", "mined_occurrence"):
        raise ValueError(f"Unsupported query kind: {query.kind}")
    if query.kind == 'observed_report':
        if query.event is None or query.event.observation is None:
            raise ValueError('An observed-report query requires a quoted clause')
        claim = memory.find_event(query.event)
        status = claim.event.observation.status if claim else None
        if status == 'reported':
            return 'Yes, you reported saying those words.', 'answer', (claim.id,)
        if status == 'denied':
            return 'No, you denied saying those words.', 'answer', (claim.id,)
        return "I don't know; saying those words has not been confirmed.", 'unknown', ()
    if query.kind == 'mined_occurrence':
        if query.event is None or not query.event.mined:
            raise ValueError('A mined-occurrence query requires a mined event')
        claims = [c for c in memory.events if c.event.identity == query.event.identity
                  and c.event.phase == 'reported']
        kinds = {bool(c.event.operators) for c in claims}
        if len(kinds) != 1:
            return "I don't know; this has not been confirmed.", 'unknown', ()
        negative = next(iter(kinds))
        return ('No, you explicitly denied it.' if negative else 'Yes, you explicitly reported it.'), 'answer', tuple(c.id for c in claims)
    if query.kind == "clarify_option":
        options = next(a for a in reversed(memory.assistant_actions) if a.kind == "options")
        return f"Do you mean {options.items[0]} or {options.items[1]}?", "clarify", (options.source_turn,)
    if query.kind == "capability":
        return "I can't send messages or set reminders in this chat.", "explain_limitation", ()
    if query.kind in ('event_completion', 'event_state'):
        if query.event is None:
            raise ValueError('Event completion queries require a typed event')
        is_state = query.kind == 'event_state'
        evidence = state_evidence if is_state else completion_evidence
        relevant = [(claim, evidence(claim.event)) for claim in memory.events
                    if claim.event.identity == query.event.identity
                    and (not is_state or claim.event.tense == query.event.tense)]
        known = {result for _, result in relevant if result != 'unknown'}
        if len(known) != 1:
            text = "I don't know; that state has not been established." if is_state else "I don't know; completion has not been established."
            return text, 'unknown', ()
        result = next(iter(known))
        supports = tuple(claim.id for claim, value in relevant if value == result)
        if is_state:
            return ('Yes, that state was reported.' if result == 'yes' else 'No, that state was explicitly denied.'), 'answer', supports
        return ('Yes, completion was reported.' if result == 'yes' else 'No, non-completion was reported.'), 'answer', supports
    fact = memory.find(query.subject, query.relation)
    subject = "You" if query.subject == "user" else query.subject
    if query.kind == "yes_no":
        if fact is None or fact.value != query.value:
            return "I don't know; you haven't said whether you did that.", "unknown", ()
        if fact.polarity == "negative":
            return "No, you said you haven't done that.", "answer", (fact.id,)
        return "Yes, you said you have.", "answer", (fact.id,)
    if fact is None or fact.polarity == "negative":
        return "I don't know from the current information.", "unknown", ()
    if query.relation == "location":
        answer = f"{subject} {'are' if query.subject == 'user' else 'is'} in {fact.value}."
    elif query.relation == "feeling":
        answer = f"{subject} {'feel' if query.subject == 'user' else 'feels'} {fact.value}."
    elif query.relation == "meeting_day":
        answer = f"Your meeting is on {fact.value}."
    elif query.relation == "drink":
        answer = f"{subject} {'prefer' if query.subject == 'user' else 'prefers'} {fact.value}."
    elif query.relation == "selection":
        answer = f"You chose {fact.value}."
    else:
        raise ValueError(f"Unsupported query: {query}")
    return answer, "answer", (fact.id,)


@dataclass(frozen=True)
class Episode:
    id: str
    task: str
    split: str
    scenario_key: str
    turns: tuple[Turn, ...]  # Ends with the user's request, before the target.
    transitions: tuple[Transition, ...]
    memory: Memory
    query: Query
    answer: str
    expected_action: str
    supporting_ids: tuple[str, ...]
    contrast_group: str | None = None
    memory_style: int = 0
    condition: str | None = None
    mined_source: dict | None = None

    def validate(self) -> None:
        state = Memory()
        prefix: list[Turn] = []
        transitions = iter(self.transitions)
        for index, turn in enumerate(self.turns):
            if turn.role != ("user" if index % 2 == 0 else "assistant") or not turn.text.strip():
                raise ValueError("Malformed dialogue")
            prefix.append(turn)
            if turn.role == "user":
                transition = next(transitions, None)
                if transition is None or transition.turn != turn or transition.before != state:
                    raise ValueError("Transition does not match dialogue state")
                previous = self.turns[index - 1] if index else None
                if transition.previous_assistant != previous:
                    raise ValueError("Incorrect previous assistant context")
                state = apply_delta(state, transition.delta, tuple(prefix))
                if state != transition.after:
                    raise ValueError("Incorrect state after delta")
            else:
                actions = tuple(a for a in self.memory.assistant_actions if a.source_turn == turn.id)
                if any(a.kind not in ("options", "suggestion") or not a.items
                       or any(item not in turn.text for item in a.items) for a in actions):
                    raise ValueError("Assistant action is not grounded in its utterance")
                state = Memory(state.facts, state.retired, state.assistant_actions + actions,
                               state.events, state.retired_events)
        if not self.turns or self.turns[-1].role != "user" or next(transitions, None) is not None:
            raise ValueError("Incomplete dialogue or extra transitions")
        if state != self.memory:
            raise ValueError("Incorrect final memory")
        if answer_query(state, self.query) != (self.answer, self.expected_action, self.supporting_ids):
            raise ValueError("Answer disagrees with scenario state")
