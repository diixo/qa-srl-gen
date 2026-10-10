"""Seeded dialogue scenarios; English surfaces and memory share one event plan."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from itertools import product
import json
import random
import re
from functools import cached_property
from typing import Iterator

from ..ontology import Label
from ..semantic_generator.substitutions import EntityPool, default_pool
from .models import (
    AssistantAction, Delta, Episode, Evidence, Fact, Memory, Query,
    Replacement, Transition, Turn, answer_query, apply_delta, EventClaim, EventReplacement,
)

GENERATOR_VERSION = "dialogue-babi-0.9.0"
TASKS = (
    "recall", "correction", "different_people", "negation", "desire_vs_done",
    "option_reference", "ambiguous_reference", "unknown", "suggestion_vs_done",
    "topic_return", "retraction", "capability",
    "event_scope", "event_progress", "event_updates", "event_states",
    "mined_predicates",
)
USER_ASSISTANT_TASKS = tuple(task for task in TASKS if task != "different_people")
COMMON_PLACES = (
    "the kitchen", "the living room", "the bedroom", "the garden",
    "the office", "the library", "the park", "the cafe",
    "the gym", "the classroom", "the supermarket", "the museum",
    "the station", "the restaurant", "the bookstore", "the hallway",
)
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
FEELINGS = ("tired", "happy", "bored", "excited", "nervous", "relaxed")
DRINKS = ("tea", "coffee", "water", "orange juice", "hot chocolate", "lemonade")
ACTIVITIES = ("a short walk", "a music break", "a snack break", "a rest break", "a coffee break")


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def dialogue_tasks(user_assistant_only: bool = False) -> tuple[str, ...]:
    if type(user_assistant_only) is not bool:
        raise ValueError("user_assistant_only must be a boolean")
    return USER_ASSISTANT_TASKS if user_assistant_only else TASKS


def _context_split(context: tuple[str, str, str]) -> str:
    # Keep an entire starting context in one split, including its extraction
    # prefix and all later tasks/surface variants. Independent of seed/quotas.
    bucket = int(fingerprint(["user-assistant-context-v1", context])[:8], 16) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


class _Conversation:
    def __init__(self) -> None:
        self.turns: list[Turn] = []
        self.transitions: list[Transition] = []
        self.memory = Memory()
        self.next_fact = 1
        self.next_event = 1

    def user(self, text: str, *, facts: tuple[tuple[str, str, str, str, str], ...] = (),
             invalidate: tuple[str, ...] = (), reference: Turn | None = None,
             events=(), event_invalidate: tuple[str, ...] = ()) -> None:
        turn = Turn(f"u{len(self.transitions) + 1}", "user", text)
        previous = self.turns[-1] if self.turns else None
        self.turns.append(turn)
        additions: list[Fact] = []
        replacements: list[Replacement] = []
        for subject, relation, value, polarity, modality in facts:
            evidence = (Evidence(turn.id, text),)
            if reference is not None:
                evidence += (Evidence(reference.id, reference.text),)
            fact = Fact(f"f{self.next_fact}", subject, relation, value, evidence, polarity, modality)
            self.next_fact += 1
            old = self.memory.find(subject, relation, modality)
            if old is None:
                additions.append(fact)
            else:
                replacements.append(Replacement(old.id, fact))
        event_add, event_replace = [], []
        for event in events:
            evidence = (Evidence(turn.id, text),)
            if reference is not None:
                evidence += (Evidence(reference.id, reference.text),)
            claim = EventClaim(f'e{self.next_event}', event, evidence)
            self.next_event += 1
            old = self.memory.find_event(event)
            if old is None:
                event_add.append(claim)
            else:
                event_replace.append(EventReplacement(old.id, claim))
        delta = Delta(tuple(additions), tuple(replacements), invalidate,
                      tuple(event_add), tuple(event_replace), event_invalidate)
        after = apply_delta(self.memory, delta, tuple(self.turns))
        self.transitions.append(Transition(turn, self.memory, delta, after, previous))
        self.memory = after

    def assistant(self, text: str, *, action: str | None = None, items: tuple[str, ...] = ()) -> Turn:
        turn = Turn(f"a{len(self.transitions)}", "assistant", text)
        self.turns.append(turn)
        if action is not None:
            self.memory = Memory(self.memory.facts, self.memory.retired,
                                 self.memory.assistant_actions + (AssistantAction(action, items, turn.id),),
                                 self.memory.events, self.memory.retired_events)
        return turn

    def pair(self, text: str, **kwargs) -> None:
        self.user(text, **kwargs)
        self.assistant("Got it.")


def _fact(subject: str, relation: str, value: str, *, negative: bool = False,
          desired: bool = False) -> tuple[str, str, str, str, str]:
    return subject, relation, value, "negative" if negative else "positive", "desired" if desired else "asserted"


def _scenario(task: str, split: str, rng: random.Random,
              names: tuple[str, ...], places: tuple[str, ...],
              context: tuple[str, str, str] | None = None) -> Episode:
    if context is None:
        name, other = rng.sample(names, 2)
        # A name may be both PERSON and LOCATION in the shared pool.
        place, destination = rng.sample([place for place in places if place not in (name, other)], 2)
    else:
        place = context[0]
        destination = rng.choice([value for value in COMMON_PLACES if value != place])
    day, new_day = rng.sample(DAYS, 2)
    feeling, other_feeling = rng.sample(FEELINGS, 2)
    drink, other_drink = rng.sample(DRINKS, 2)
    first, second = rng.sample(ACTIVITIES, 2)
    variant = rng.randrange(2)
    choose = lambda a, b: (a, b)[variant]
    c = _Conversation()
    if context is None:
        # Named scenarios use one grounded distractor and held-out vocabulary.
        c.pair(choose(f"My friend {name} is in {place}.", f"{name}, my friend, is currently in {place}."),
               facts=(_fact(name, "location", place),))
    else:
        place, feeling, drink = context
        # These user facts stay active throughout the episode. Grouping them
        # together protects both response views and extraction from split leaks.
        c.pair(choose(f"I'm in {place}. I feel {feeling}, and I prefer {drink}.",
                      f"I prefer {drink}. Right now I'm in {place} and feel {feeling}."),
               facts=(_fact("user", "location", place), _fact("user", "feeling", feeling),
                      _fact("user", "drink", drink)))

    if task == "recall":
        if context is None:
            c.pair(choose(f"I'm in {destination} now.", f"Right now I am in {destination}."),
                   facts=(_fact("user", "location", destination),))
        c.user(choose("Where am I now?", "Remind me where I am."))
        query = Query("lookup", relation="location")
    elif task == "correction":
        meeting = "My meeting" if context is not None else f"My meeting with {other}"
        c.pair(f"{meeting} is on {day}.", facts=(_fact("user", "meeting_day", day),))
        c.pair(choose(f"Correction: it's on {new_day}, not {day}.", f"I got the day wrong. It's {new_day}."),
               facts=(_fact("user", "meeting_day", new_day),))
        c.user(choose("Which day is my meeting?", "Remind me of the corrected meeting day."))
        query = Query("lookup", relation="meeting_day")
    elif task == "different_people":
        c.pair(choose(f"I feel {feeling}, but {other} feels {other_feeling}.",
                      f"{other} feels {other_feeling}. I feel {feeling}."),
               facts=(_fact("user", "feeling", feeling), _fact(other, "feeling", other_feeling)))
        subject = rng.choice(("user", other))
        c.user("How do I feel?" if subject == "user" else f"How does {other} feel?")
        query = Query("lookup", subject=subject, relation="feeling")
    elif task == "negation":
        done = rng.choice((True, False))
        c.pair(f"I want to visit {destination}.", facts=(_fact("user", "visited", destination, desired=True),))
        if done is not None:
            text = (choose(f"I've visited {destination} now.", f"I have now been to {destination}.") if done
                    else choose(f"I haven't visited {destination} yet.", f"I have not been to {destination}."))
            c.pair(text, facts=(_fact("user", "visited", destination, negative=not done),))
        c.user(choose(f"Have I visited {destination}?", f"Have I already been to {destination}?"))
        query = Query("yes_no", relation="visited", value=destination)
    elif task in ("option_reference", "ambiguous_reference"):
        if context is None:
            c.user(f"I'm visiting {destination} and would like a break. Give me two options.",
                   facts=(_fact("user", "location", destination),))
        else:
            c.user("I'd like a break. Give me two options.")
        offer = c.assistant(f"You could take {first} or {second}, in that order.",
                            action="options", items=(first, second))
        if task == "ambiguous_reference":
            c.user(choose("That one.", "I'd like that option."))
            query = Query("clarify_option")
        else:
            position = rng.randrange(2)
            ordinal = ("first", "second")[position]
            selected = (first, second)[position]
            c.pair(choose(f"The {ordinal} one.", f"I'll take the {ordinal} option."),
                   facts=(_fact("user", "selection", selected),), reference=offer)
            c.user(choose("What did I choose?", "Which option did I pick?"))
            query = Query("lookup", relation="selection")
    elif task == "unknown":
        c.pair("Can you help with my meeting?" if context is not None
               else f"Can you help with my meeting with {other}?")
        c.user(choose("On which day is my meeting?", "What day is that meeting scheduled for?"))
        query = Query("lookup", relation="meeting_day")
    elif task == "topic_return":
        if context is None:
            c.pair(f"I prefer {drink}.", facts=(_fact("user", "drink", drink),))
            c.pair(f"Changing the subject: {other} prefers {other_drink}.", facts=(_fact(other, "drink", other_drink),))
            c.pair(f"Also, I'm in {destination} today.", facts=(_fact("user", "location", destination),))
        else:
            c.pair(f"Changing the subject: my meeting is on {day}.",
                   facts=(_fact("user", "meeting_day", day),))
            c.pair("That's all about the meeting for now.")
        c.user(choose("Back to my drink: what do I prefer?", "Returning to drinks, what was my preference?"))
        query = Query("lookup", relation="drink")
    elif task == "retraction":
        meeting = "My meeting" if context is not None else f"My meeting with {other}"
        c.pair(f"{meeting} is on {day}.", facts=(_fact("user", "meeting_day", day),))
        old = c.memory.find("user", "meeting_day")
        c.pair(choose("Forget that day. The meeting date hasn't been decided yet.",
                      "That meeting day was wrong; the date is now undecided."), invalidate=(old.id,))
        c.user(choose("When is my meeting?", "Do you know my meeting day?"))
        query = Query("lookup", relation="meeting_day")
    elif task == "capability":
        if context is None:
            c.pair(f"Can we talk about contacting {other} on {day}?")
            c.user(choose(f"Send {other} a message for me.", f"Set a reminder to contact {other} on {day}."))
        else:
            c.pair("Can you help me plan my day?")
            c.user(choose("Set a reminder for me to take a break.", "Remind me to drink some water in an hour."))
        query = Query("capability")
    else:
        raise ValueError(f"Unknown dialogue task: {task}")

    answer, action, supports = answer_query(c.memory, query)
    # Surface choices and seed never determine a scenario's split. All variants
    # of this event plan have the same key, including replaced/retired values.
    plan = {"task": task, "query": asdict(query), "updates": [],
            "actions": [asdict(a) for a in c.memory.assistant_actions]}
    for transition in c.transitions:
        delta = asdict(transition.delta)
        for fact in delta["add"]:
            fact.pop("evidence")
        for replacement in delta["replace"]:
            replacement["fact"].pop("evidence")
        plan["updates"].append(delta)
    key = fingerprint(plan)
    episode = Episode(f"dialogue-{key[:24]}-v{variant}", task, split, key, tuple(c.turns),
                      tuple(c.transitions), c.memory, query, answer, action, supports)
    episode.validate()
    return episode


class DialogueGenerator:
    """Draw named-pool scenarios or conversations about the user alone.

    Task structures and surface families are shared between splits. Named
    entities or user-context combinations and event plans are held out; this
    is not an unseen-task or free-chat benchmark.
    """

    def __init__(self, *, seed: int = 42, pool: EntityPool | None = None,
                 user_assistant_only: bool = False, mined_bank=None) -> None:
        self.seed = seed
        self.tasks = dialogue_tasks(user_assistant_only)
        self.user_assistant_only = user_assistant_only
        self.pool = default_pool() if pool is None and not user_assistant_only else pool
        self._mined_bank = mined_bank

    @cached_property
    def mined_bank(self):
        if self._mined_bank is not None:
            return self._mined_bank
        from .mined_bank import DialogueMinedBank
        return DialogueMinedBank()

    def generate(self, task: str, count: int, *, split: str) -> Iterator[Episode]:
        if task not in self.tasks or split not in ("train", "validation", "test"):
            raise ValueError("Unknown task or split")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("Count must be a nonnegative integer")
        if not count:
            return
        if task == 'mined_predicates':
            from .mined_scenarios import mined_episodes
            yield from mined_episodes(self, count, split)
            return
        if task in ('event_scope', 'event_progress', 'event_updates', 'event_states'):
            from .event_scenarios import event_episodes
            yield from event_episodes(self, task, count, split)
            return
        from .actions import CONTRAST_TASKS
        if task in CONTRAST_TASKS:
            from .contrast import contrast_episodes
            yield from contrast_episodes(self, task, count, split)
            return
        if self.user_assistant_only:
            yield from self._generate_user_assistant(task, count, split=split)
            return
        pool = self.pool.split("dev" if split == "validation" else split)
        # Conservative grammar subset: no titles, determiners or plural names.
        safe = [e for e in pool if e.is_named and e.number == "singular"
                and re.fullmatch(r"[A-Z][a-z]{2,}", e.text)]
        names = tuple(sorted({e.text for e in safe if Label.PERSON in e.labels}))
        places = tuple(sorted({e.text for e in safe if Label.LOCATION in e.labels}))
        if len(names) < 2 or len(places) < 4:
            raise ValueError(f"{split} needs at least two person names and four place names")
        rng = random.Random(fingerprint([GENERATOR_VERSION, self.seed, task, split]))
        seen: set[str] = set()
        produced = 0
        for _ in range(max(1000, count * 100)):
            episode = _scenario(task, split, rng, names, places)
            if episode.scenario_key in seen:
                continue
            seen.add(episode.scenario_key)
            yield episode
            produced += 1
            if produced == count:
                return
        raise ValueError(f"{task}/{split}: exhausted distinct scenarios ({produced}/{count})")

    def _generate_user_assistant(self, task: str, count: int, *, split: str) -> Iterator[Episode]:
        contexts = [context for context in product(COMMON_PLACES, FEELINGS, DRINKS)
                    if _context_split(context) == split]
        rng = random.Random(fingerprint(["user-assistant-v1", self.seed, task, split]))
        seen: set[str] = set()
        produced = 0
        # Visit all contexts before repeating one. Simple tasks have only one
        # event plan per context; fail explicitly rather than pad with duplicates.
        rounds = 1 if task in ("recall", "unknown", "capability") else max(20, count * 20 // len(contexts))
        for _ in range(rounds):
            rng.shuffle(contexts)
            for context in contexts:
                episode = _scenario(task, split, rng, (), (), context=context)
                if episode.scenario_key in seen:
                    continue
                seen.add(episode.scenario_key)
                yield episode
                produced += 1
                if produced == count:
                    return
        raise ValueError(f"{task}/{split}: exhausted distinct user-assistant scenarios "
                         f"({produced}/{count}); reduce the per-task quota")
