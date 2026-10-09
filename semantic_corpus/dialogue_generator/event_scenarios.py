"""Compositional action scenarios sampled through shared typed frame slots."""

from dataclasses import asdict, replace
from itertools import product
import random
import re

from ..ontology import Label
from ..semantic_generator.substitutions import EntityPool, load_pool
from .events import EVENT_LEXEMES, Event, EventArgument, event_description, event_question, event_statement
from .models import Episode, Query, answer_query

EVENT_TASKS = ('event_scope', 'event_progress', 'event_updates')

# Outer-to-inner scopes. Negative need means lack of necessity, not prohibition;
# want/not and not/want belong to different claim channels.
SCOPE_PROFILES = (
    ('want',), ('need',), ('plan',), ('intend',), ('decide',),
    ('not', 'want'), ('want', 'not'), ('not', 'need'), ('need', 'not'),
    ('plan', 'try'), ('want', 'try'), ('need', 'start'),
    ('want', 'manage'), ('not', 'plan'), ('plan', 'not'), ('plan', 'start'),
)
PROGRESS_PROFILES = ('ongoing', 'not_ongoing', 'try', 'start', 'stop', 'future', 'try_not', 'start_try')


class EventSampler:
    def __init__(self, pool: EntityPool):
        common = [entity for entity in pool if not entity.is_named and entity.number == 'singular'
                  and entity.text.startswith(('a ', 'an ', 'the ')) and Label.GROUP not in entity.labels.labels
                  and '\n' not in entity.text and len(entity.text.split()) <= 5]
        self.choices = {}
        for lexeme in EVENT_LEXEMES:
            frame = lexeme.frame
            pattern = frame.pattern(lexeme.pattern)
            for role in pattern.mentioned() - {pattern.subject}:
                slot = frame.slot_map[role]
                candidates = [e for e in common if slot.accepts(e.labels.labels, type_name=e.type_name)]
                # Broad PHYSICAL_OBJECT includes buildings/ships: type checking
                # alone is not selectional plausibility. Use portable objects in
                # the current physical-transfer senses, and animate care targets.
                if role == 'theme' and lexeme.lemma in ('buy', 'give', 'move', 'pick up', 'put down'):
                    candidates = [e for e in candidates if e.type_name in ('document', 'tool', 'toy')]
                if lexeme.lemma == 'look after':
                    candidates = [e for e in candidates if e.type_name in ('child', 'dog', 'cat')]
                if not candidates:
                    raise ValueError(f'No suitable shared-pool fillers for {lexeme.lemma}/{role}')
                self.choices[lexeme.lemma, role] = tuple(sorted(candidates, key=lambda e: (e.text, str(e.labels))))

    def sample(self, lexeme, rng, *, agent='user', time='today') -> Event:
        pattern = lexeme.frame.pattern(lexeme.pattern)
        arguments = [EventArgument(pattern.subject, agent, 'PERSON')]
        for role in sorted(pattern.mentioned() - {pattern.subject}):
            entity = rng.choice(self.choices[lexeme.lemma, role])
            arguments.append(EventArgument(role, entity.text, entity.entity_label.value))
        return Event(lexeme.lemma, tuple(sorted(arguments)), time=time)


def event_episodes(generator, task, count, split):
    from .generator import COMMON_PLACES, FEELINGS, DRINKS, _Conversation, _context_split, _fact, fingerprint

    if not hasattr(generator, '_event_sampler'):
        generator._event_sampler = EventSampler(generator.pool if generator.pool is not None else load_pool())
    sampler = generator._event_sampler
    rng = random.Random(fingerprint(['event-scenarios-v1', generator.seed, task, split]))
    if generator.user_assistant_only:
        contexts = [c for c in product(COMMON_PLACES, FEELINGS, DRINKS) if _context_split(c) == split]
    else:
        pool = generator.pool.split('dev' if split == 'validation' else split)
        safe = [e for e in pool if e.is_named and e.number == 'singular' and re.fullmatch(r'[A-Z][a-z]{2,}', e.text)]
        names = sorted({e.text for e in safe if Label.PERSON in e.labels})
        places = sorted({e.text for e in safe if Label.LOCATION in e.labels})
        if len(names) < 2 or len(places) < 4:
            raise ValueError(f'{split} needs at least two person names and four place names')
        contexts = [(name, place) for name in names for place in places if name != place]
    profiles = (SCOPE_PROFILES if task == 'event_scope' else PROGRESS_PROFILES if task == 'event_progress'
                else ('correction', 'other_object', 'other_time'))
    # Shuffle the product, not two same-length lists: lockstep cycling would
    # associate each predicate with only one progress profile and leak labels.
    combinations = list(product(EVENT_LEXEMES, profiles))
    rng.shuffle(combinations)
    seen, produced, index = set(), 0, 0
    for _ in range(max(20, count)):
        rng.shuffle(contexts)
        for context in contexts:
            if index and index % len(combinations) == 0:
                rng.shuffle(combinations)
            lexeme, profile = combinations[index % len(combinations)]
            index += 1
            base = sampler.sample(lexeme, rng)
            style = rng.randrange(3)
            if task == 'event_scope':
                initial = replace(base, operators=profile)
                condition = '>'.join(profile)
            elif task == 'event_progress':
                condition = profile
                if profile in ('ongoing', 'not_ongoing'):
                    initial = replace(base, phase='ongoing', operators=('not',) if profile == 'not_ongoing' else ())
                elif profile == 'future':
                    initial = replace(base, tense='future')
                else:
                    initial = replace(base, operators=tuple(profile.split('_')), tense='past')
            else:
                condition = profile
                initial = replace(base, phase='completed')
            distractor = sampler.sample(lexeme, rng, time='yesterday')
            if task == 'event_updates' and condition == 'other_object' and len(base.arguments) > 1:
                # Retry a few draws to ensure a genuinely different object or
                # participant; get-up uses the time contrast instead.
                for _ in range(30):
                    distractor = sampler.sample(lexeme, rng)
                    if distractor.identity != base.identity:
                        break
                if distractor.identity == base.identity:
                    distractor = replace(base, time='yesterday')
            else:
                distractor = replace(base, time='yesterday')
            group = fingerprint(['event-v1', task, context, asdict(initial), asdict(distractor)])
            if group in seen:
                continue
            seen.add(group)
            outcomes = ['yes', 'no', 'unknown']
            rng.shuffle(outcomes)
            for outcome in outcomes:
                c = _Conversation()
                if generator.user_assistant_only:
                    place, feeling, drink = context
                    c.pair(f"I'm in {place}. I feel {feeling}, and I prefer {drink}.",
                           facts=(_fact('user', 'location', place), _fact('user', 'feeling', feeling), _fact('user', 'drink', drink)))
                else:
                    name, place = context
                    c.pair(f'My friend {name} is in {place}.', facts=(_fact(name, 'location', place),))
                c.pair(event_statement(initial, perspective='user', style=style), events=(initial,))
                if task == 'event_updates':
                    other = replace(distractor, phase='completed',
                                    tense='past' if distractor.time == 'yesterday' else 'present')
                    c.pair(event_statement(other, perspective='user', style=style), events=(other,))
                    if outcome == 'unknown':
                        old = c.memory.find_event(base)
                        c.pair('Forget my earlier report about ' + event_description(base, style=style) + '.',
                               event_invalidate=(old.id,))
                if outcome != 'unknown':
                    final = replace(base, phase='completed', operators=('not',) if outcome == 'no' else ())
                    if task != 'event_updates' and style:
                        if outcome == 'yes':
                            op = 'manage' if style == 1 and lexeme.bounded else 'finish'
                            final = replace(base, operators=(op,), tense='past')
                        elif lexeme.bounded and style == 1:
                            final = replace(base, operators=('fail',), tense='past')
                        elif style == 2:
                            final = replace(base, operators=('not', 'finish'), tense='past')
                    text = event_statement(final, perspective='user', style=style)
                    c.pair(('Correction: ' if task == 'event_updates' else '') + text, events=(final,))
                c.user(event_question(base, style=style))
                query = Query('event_completion', event=base)
                answer, expected, supports = answer_query(c.memory, query)
                key = fingerprint([group, outcome])
                episode = Episode(f'dialogue-{key[:24]}-v{style}', task, split, key, tuple(c.turns), tuple(c.transitions),
                                  c.memory, query, answer, expected, supports, group, style, f'{condition}:{outcome}')
                episode.validate()
                yield episode
                produced += 1
                if produced == count:
                    return
    raise ValueError(f'{task}/{split}: exhausted typed event scenarios ({produced}/{count})')
