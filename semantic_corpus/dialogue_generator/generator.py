"""Seeded dialogue scenarios; English surfaces and memory share one event plan."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import random
import re
from typing import Iterator

from ..ontology import Label
from ..semantic_generator.substitutions import EntityPool, default_pool
from .models import (
    AssistantAction, Delta, Episode, Evidence, Fact, Memory, Query,
    Replacement, Transition, Turn, answer_query, apply_delta,
)

GENERATOR_VERSION = "dialogue-babi-0.1.0"
TASKS = (
    "recall", "correction", "different_people", "negation", "desire_vs_done",
    "option_reference", "ambiguous_reference", "unknown", "suggestion_vs_done",
    "topic_return", "retraction", "capability",
)
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
FEELINGS = ("tired", "happy", "bored", "excited", "nervous", "relaxed")
DRINKS = ("tea", "coffee", "water", "orange juice", "hot chocolate", "lemonade")
ACTIVITIES = ("a short walk", "a music break", "a snack break", "a rest break", "a coffee break")


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


class _Conversation:
    def __init__(self) -> None:
        self.turns: list[Turn] = []
        self.transitions: list[Transition] = []
        self.memory = Memory()
        self.next_fact = 1

    def user(self, text: str, *, facts: tuple[tuple[str, str, str, str, str], ...] = (),
             invalidate: tuple[str, ...] = (), reference: Turn | None = None) -> None:
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
        delta = Delta(tuple(additions), tuple(replacements), invalidate)
        after = apply_delta(self.memory, delta, tuple(self.turns))
        self.transitions.append(Transition(turn, self.memory, delta, after, previous))
        self.memory = after

    def assistant(self, text: str, *, action: str | None = None, items: tuple[str, ...] = ()) -> Turn:
        turn = Turn(f"a{len(self.transitions)}", "assistant", text)
        self.turns.append(turn)
        if action is not None:
            self.memory = Memory(self.memory.facts, self.memory.retired,
                                 self.memory.assistant_actions + (AssistantAction(action, items, turn.id),))
        return turn

    def pair(self, text: str, **kwargs) -> None:
        self.user(text, **kwargs)
        self.assistant("Got it.")


def _fact(subject: str, relation: str, value: str, *, negative: bool = False,
          desired: bool = False) -> tuple[str, str, str, str, str]:
    return subject, relation, value, "negative" if negative else "positive", "desired" if desired else "asserted"


def _scenario(task: str, split: str, rng: random.Random,
              names: tuple[str, ...], places: tuple[str, ...]) -> Episode:
    name, other = rng.sample(names, 2)
    # A name may be both PERSON and LOCATION in the shared pool.
    place, destination = rng.sample([place for place in places if place not in (name, other)], 2)
    day, new_day = rng.sample(DAYS, 2)
    feeling, other_feeling = rng.sample(FEELINGS, 2)
    drink, other_drink = rng.sample(DRINKS, 2)
    first, second = rng.sample(ACTIVITIES, 2)
    variant = rng.randrange(2)
    choose = lambda a, b: (a, b)[variant]
    c = _Conversation()
    # One irrelevant, explicitly grounded fact tests attention and gives every
    # task natural held-out vocabulary, including otherwise identical refusals.
    c.pair(choose(f"My friend {name} is in {place}.", f"{name}, my friend, is currently in {place}."),
           facts=(_fact(name, "location", place),))

    if task == "recall":
        c.pair(choose(f"I'm in {destination} now.", f"Right now I am in {destination}."),
               facts=(_fact("user", "location", destination),))
        c.user(choose("Where am I now?", "Remind me where I am."))
        query = Query("lookup", relation="location")
    elif task == "correction":
        c.pair(f"My meeting with {other} is on {day}.", facts=(_fact("user", "meeting_day", day),))
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
    elif task in ("negation", "desire_vs_done"):
        done = rng.choice((True, False)) if task == "negation" else rng.choice((True, False, None))
        c.pair(f"I want to visit {destination}.", facts=(_fact("user", "visited", destination, desired=True),))
        if done is not None:
            text = (choose(f"I've visited {destination} now.", f"I have now been to {destination}.") if done
                    else choose(f"I haven't visited {destination} yet.", f"I have not been to {destination}."))
            c.pair(text, facts=(_fact("user", "visited", destination, negative=not done),))
        c.user(choose(f"Have I visited {destination}?", f"Have I already been to {destination}?"))
        query = Query("yes_no", relation="visited", value=destination)
    elif task in ("option_reference", "ambiguous_reference"):
        c.user(f"I'm visiting {destination} and would like a break. Give me two options.",
               facts=(_fact("user", "location", destination),))
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
        c.pair(f"Can you help with my meeting with {other}?")
        c.user(choose("On which day is my meeting?", "What day is that meeting scheduled for?"))
        query = Query("lookup", relation="meeting_day")
    elif task == "suggestion_vs_done":
        c.user(f"I feel {feeling}. Suggest a break.", facts=(_fact("user", "feeling", feeling),))
        suggestion = c.assistant(f"You could take {first}.", action="suggestion", items=(first,))
        done = rng.choice((True, False, None))
        if done is None:
            c.pair(choose("Thanks for the suggestion.", "Okay, thanks."))
        else:
            text = f"I've taken {first}." if done else f"I haven't taken {first} yet."
            c.pair(text, facts=(_fact("user", "completed_break", first, negative=not done),), reference=suggestion)
        c.user(f"Have I taken {first}?")
        query = Query("yes_no", relation="completed_break", value=first)
    elif task == "topic_return":
        c.pair(f"I prefer {drink}.", facts=(_fact("user", "drink", drink),))
        c.pair(f"Changing the subject: {other} prefers {other_drink}.", facts=(_fact(other, "drink", other_drink),))
        c.pair(f"Also, I'm in {destination} today.", facts=(_fact("user", "location", destination),))
        c.user(choose("Back to my drink: what do I prefer?", "Returning to drinks, what was my preference?"))
        query = Query("lookup", relation="drink")
    elif task == "retraction":
        c.pair(f"My meeting with {other} is on {day}.", facts=(_fact("user", "meeting_day", day),))
        old = c.memory.find("user", "meeting_day")
        c.pair(choose("Forget that day. The meeting date hasn't been decided yet.",
                      "That meeting day was wrong; the date is now undecided."), invalidate=(old.id,))
        c.user(choose("When is my meeting?", "Do you know my meeting day?"))
        query = Query("lookup", relation="meeting_day")
    elif task == "capability":
        c.pair(f"Can we talk about contacting {other} on {day}?")
        c.user(choose(f"Send {other} a message for me.", f"Set a reminder to contact {other} on {day}."))
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
    """Draw pilot scenarios using named people/places from the existing pool.

    Task structures and surface families are shared between splits. Names and
    event plans are held out; this is not an unseen-task or free-chat benchmark.
    """

    def __init__(self, *, seed: int = 42, pool: EntityPool | None = None) -> None:
        self.seed = seed
        self.pool = default_pool() if pool is None else pool

    def generate(self, task: str, count: int, *, split: str) -> Iterator[Episode]:
        if task not in TASKS or split not in ("train", "validation", "test"):
            raise ValueError("Unknown task or split")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("Count must be a nonnegative integer")
        if not count:
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
