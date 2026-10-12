"""Reviewed fragment meanings and partial events, without completing missing text.

The catalog is a supplemental, explicit review in the central resource bank.
It does not promote every sentence_review fragment to a fact or full sentence.
"""
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
import random
import re

from ..ontology import Label
from .events import Event, EventArgument
from .models import Episode, Query, answer_query

CATALOG = Path(__file__).resolve().parents[2] / 'data/mined-templates/reviewed/dialogue-fragments.json'


def load_fragment_patterns(path=CATALOG):
    content = json.loads(Path(path).read_text(encoding='utf-8'))
    if content.get('format') != 'reviewed-dialogue-fragments-v1':
        raise ValueError('Unsupported reviewed fragment catalog')
    patterns = content['patterns']
    ids = [p['id'] for p in patterns]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Reviewed fragments need distinct IDs')
    for row in patterns:
        if row['kind'] not in {'fact', 'partial', 'event', 'refine', 'ambiguous', 'no_change'}:
            raise ValueError('Unknown reviewed fragment meaning')
        if type(row['context']) is not bool or not row['assistant'].strip() or not row['reply'].strip():
            raise ValueError('Reviewed fragments need visible dialogue context')
    return patterns


def fragment_episode(pattern, context, *, split, user_only, variant=0):
    from .generator import _Conversation, _fact, fingerprint, COMMON_PLACES, DAYS, FEELINGS

    c = _Conversation()
    if user_only:
        place, feeling, drink = context
        c.pair(f"I'm in {place}. I feel {feeling}, and I prefer {drink}.",
               facts=(_fact('user', 'location', place), _fact('user', 'feeling', feeling),
                      _fact('user', 'drink', drink)))
    else:
        name, place = context
        feeling = FEELINGS[variant % len(FEELINGS)]
        c.pair(f'My friend {name} is in {place}.', facts=(_fact(name, 'location', place),))
    # All values are explicit generator choices, not guessed source entities.
    destinations = [p for p in COMMON_PLACES if p != place]
    values = dict(destination=destinations[variant % len(destinations)],
                  feeling=feeling, day=DAYS[variant % len(DAYS)],
                  object=('a book', 'a pen', 'a bag', 'a toy')[variant % 4])
    # Replace the generic acknowledgement with the reviewed prompt, while
    # keeping alternating roles and the same assistant turn identity.
    c.turns.pop()
    options = tuple(pattern.get('options', ()))
    prompt = c.assistant(pattern['assistant'].format(**values),
                         action='options' if options else None, items=options)
    reference = prompt if pattern['context'] else None
    text = pattern['reply'].format(**values)
    kind = pattern['kind']
    if kind == 'fact':
        c.pair(text, facts=(_fact('user', pattern['relation'], pattern['value'].format(**values),
                                  negative=pattern.get('negative', False)),), reference=reference)
        query = Query('lookup', relation=pattern['relation'])
    elif kind in ('partial', 'event', 'refine'):
        arguments = tuple(sorted(EventArgument(role, value.format(**values), label)
                                 for role, value, label in pattern['arguments']))
        event = Event(pattern['predicate'], arguments, time='unspecified',
                      operators=tuple(pattern['operators']), tense=pattern['tense'],
                      missing_roles=tuple(pattern.get('missing_roles', ())))
        c.pair(text, events=(event,), reference=reference)
        if kind == 'refine':
            prior = c.memory.find_event(event)
            c.turns.pop()
            prompt = c.assistant(pattern['clarification'])
            role, value, label = pattern['fill']
            event = replace(event, arguments=tuple(sorted((*event.arguments,
                            EventArgument(role, value.format(**values), label)))), missing_roles=())
            c.pair(pattern['continuation'].format(**values), events=(event,),
                   event_invalidate=(prior.id,), reference=prompt)
        query = Query('event_completion', event=event)
    elif kind == 'ambiguous':
        c.user(text)
        query = Query('clarify_option')
    else:
        c.pair(text)
        query = Query('lookup', relation=pattern['relation'])
    if kind != 'ambiguous':
        c.user(pattern['question'].format(**values))
    answer, action, supports = answer_query(c.memory, query)
    plan = dict(task='fragment_meaning', pattern=pattern['id'], context=context,
                query=asdict(query), updates=[asdict(t.delta) for t in c.transitions])
    key = fingerprint(plan)
    episode = Episode(f'dialogue-{key[:24]}', 'fragment_meaning', split, key,
                      tuple(c.turns), tuple(c.transitions), c.memory, query, answer, action, supports,
                      condition=pattern['id'])
    episode.validate()
    return episode


def fragment_episodes(generator, count, split):
    from .generator import COMMON_PLACES, FEELINGS, DRINKS, _context_split, fingerprint

    patterns = load_fragment_patterns()
    rng = random.Random(fingerprint(['fragment-meaning-v1', generator.seed, split, patterns]))
    if generator.user_assistant_only:
        contexts = [c for c in product(COMMON_PLACES, FEELINGS, DRINKS) if _context_split(c) == split]
    else:
        pool = generator.pool.split('dev' if split == 'validation' else split)
        safe = [e for e in pool if e.is_named and e.number == 'singular' and re.fullmatch('[A-Z][a-z]{2,}', e.text)]
        names = sorted({e.text for e in safe if Label.PERSON in e.labels})
        places = sorted({e.text for e in safe if Label.LOCATION in e.labels})
        contexts = [(n, p) for n in names for p in places if n != p]
        if not contexts:
            raise ValueError(f'{split} needs distinct person names and place names')
    rng.shuffle(contexts)
    rng.shuffle(patterns)
    seen, produced, index = set(), 0, 0
    for cycle in range(max(20, count)):
        for context in contexts:
            pattern = patterns[index % len(patterns)]
            index += 1
            episode = fragment_episode(pattern, context, split=split,
                                       user_only=generator.user_assistant_only,
                                       variant=rng.randrange(420))
            if episode.scenario_key in seen:
                continue
            seen.add(episode.scenario_key)
            yield episode
            produced += 1
            if produced == count:
                return
        rng.shuffle(contexts)
    raise ValueError(f'fragment_meaning/{split}: exhausted distinct scenarios ({produced}/{count})')
