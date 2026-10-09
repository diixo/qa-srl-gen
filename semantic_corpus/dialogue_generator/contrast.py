"""Matched yes/no/unknown episodes with groups kept in the same data split."""

from itertools import product
import random
import re

from .actions import ACTIONS, action_question, action_statement
from .models import Episode, Query, answer_query
from ..ontology import Label


def contrast_episodes(generator, task, count, split):
    from .generator import (
        COMMON_PLACES, FEELINGS, DRINKS, _context_split, _Conversation, _fact, fingerprint,
    )

    rng = random.Random(fingerprint(['contrast-v1', generator.seed, task, split]))
    if generator.user_assistant_only:
        contexts = [c for c in product(COMMON_PLACES, FEELINGS, DRINKS) if _context_split(c) == split]
        places = ()
    else:
        pool = generator.pool.split('dev' if split == 'validation' else split)
        safe = [e for e in pool if e.is_named and e.number == 'singular' and re.fullmatch(r'[A-Z][a-z]{2,}', e.text)]
        names = tuple(sorted({e.text for e in safe if Label.PERSON in e.labels}))
        places = tuple(sorted({e.text for e in safe if Label.LOCATION in e.labels}))
        if len(names) < 2 or len(places) < 4:
            raise ValueError(f'{split} needs at least two person names and four place names')
        contexts = [(name, place) for name in names for place in places if name != place]
    seen = set()
    produced = 0
    actions = list(ACTIONS)
    rng.shuffle(actions)
    group_index = 0
    for _ in range(max(20, count)):
        rng.shuffle(contexts)
        for context in contexts:
            action = actions[group_index % len(actions)]
            group_index += 1
            objects = (tuple(p for p in (COMMON_PLACES if generator.user_assistant_only else places)
                             if p not in context) if action.relation == 'visited' else action.objects)
            value, other_value = rng.sample(objects, 2)
            style = rng.randrange(3)
            background = rng.choice(('plain', 'other_object', 'other_action', 'accepted'))
            modality = rng.choice(('desired', 'planned'))
            other_action = rng.choice([a for a in ACTIONS if a.relation != action.relation and a.relation != 'visited'])
            distractor = rng.choice(other_action.objects)
            group = fingerprint(['contrast-v1', task, context, action.relation, value, other_value,
                                 background, modality, other_action.relation, distractor])
            if group in seen:
                continue
            seen.add(group)
            outcomes = ['unknown', 'yes', 'no']
            rng.shuffle(outcomes)
            for outcome in outcomes:
                c = _Conversation()
                if generator.user_assistant_only:
                    place, feeling, drink = context
                    c.pair(f"I'm in {place}. I feel {feeling}, and I prefer {drink}.",
                           facts=(_fact('user', 'location', place), _fact('user', 'feeling', feeling),
                                  _fact('user', 'drink', drink)))
                else:
                    name, place = context
                    c.pair(f'My friend {name} is in {place}.', facts=(_fact(name, 'location', place),))
                if task == 'desire_vs_done':
                    c.pair(action_statement(action, value, subject='I', modality=modality, style=style),
                           facts=(('user', action.relation, value, 'positive', modality),))
                else:
                    c.user('Could you suggest something for me to do?')
                    phrase = f'{action.gerund} {value}'
                    c.assistant(f'You could consider {phrase}.', action='suggestion', items=(phrase,))
                    c.pair(('Thanks for the suggestion.', 'Okay, thanks.', 'That sounds good.')[style])
                if background == 'other_object':
                    c.pair(action_statement(action, other_value, subject='I', style=style),
                           facts=(_fact('user', action.relation, other_value),))
                elif background == 'other_action':
                    c.pair(action_statement(other_action, distractor, subject='I', style=style),
                           facts=(_fact('user', other_action.relation, distractor),))
                elif background == 'accepted':
                    c.pair(f'I choose to {action.base} {value}.',
                           facts=(_fact('user', 'selection', f'to {action.base} {value}'),))
                if outcome != 'unknown':
                    c.pair(action_statement(action, value, subject='I', negative=outcome == 'no', style=style),
                           facts=(_fact('user', action.relation, value, negative=outcome == 'no'),))
                c.user(action_question(action, value, style))
                query = Query('yes_no', relation=action.relation, value=value)
                answer, expected, supports = answer_query(c.memory, query)
                key = fingerprint([group, outcome])
                episode = Episode(f'dialogue-{key[:24]}-v{style}', task, split, key, tuple(c.turns),
                                  tuple(c.transitions), c.memory, query, answer, expected, supports,
                                  contrast_group=group, memory_style=style, condition=f'{background}:{outcome}')
                episode.validate()
                yield episode
                produced += 1
                if produced == count:
                    return
    raise ValueError(f'{task}/{split}: exhausted contrast groups ({produced}/{count})')
