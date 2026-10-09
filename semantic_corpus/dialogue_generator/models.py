"""Small, explicit dialogue memory for controlled synthetic experiments.

Facts are claims made in a conversation, not independently verified world facts.
The checker validates references and updates, not arbitrary English semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SCHEMA_VERSION = "dialogue-memory-v2"


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
class Delta:
    add: tuple[Fact, ...] = ()
    replace: tuple[Replacement, ...] = ()
    invalidate: tuple[str, ...] = ()


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

    def find(self, subject: str, relation: str, modality: str = "asserted") -> Fact | None:
        return next((f for f in self.facts if f.key == (subject, relation, modality)), None)

    def prompt_data(self) -> dict:
        # Retired claims stay in the audit record, not in current-state prompts.
        return {"facts": [asdict(f) for f in self.facts],
                "assistant_actions": [asdict(a) for a in self.assistant_actions]}


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
    seen = set(active) | historical
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
    return Memory(facts, memory.retired + tuple(active[fid] for fid in removed), memory.assistant_actions)


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


def answer_query(memory: Memory, query: Query) -> tuple[str, str, tuple[str, ...]]:
    """Reference answer, dialogue action and supporting IDs from visible state."""
    if query.kind not in ("lookup", "yes_no", "clarify_option", "capability"):
        raise ValueError(f"Unsupported query kind: {query.kind}")
    if query.kind == "clarify_option":
        options = next(a for a in reversed(memory.assistant_actions) if a.kind == "options")
        return f"Do you mean {options.items[0]} or {options.items[1]}?", "clarify", (options.source_turn,)
    if query.kind == "capability":
        return "I can't send messages or set reminders in this chat.", "explain_limitation", ()
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
                state = Memory(state.facts, state.retired, state.assistant_actions + actions)
        if not self.turns or self.turns[-1].role != "user" or next(transitions, None) is not None:
            raise ValueError("Incomplete dialogue or extra transitions")
        if state != self.memory:
            raise ValueError("Incorrect final memory")
        if answer_query(state, self.query) != (self.answer, self.expected_action, self.supporting_ids):
            raise ValueError("Answer disagrees with scenario state")
