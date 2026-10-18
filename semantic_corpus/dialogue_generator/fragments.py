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
from ..template_bank import fingerprint
from .events import Event, EventArgument
from .mined_subjects import MinedSubject
from .models import Episode, Query, answer_query

CATALOG = Path(__file__).resolve().parents[2] / 'data/mined-templates/reviewed/dialogue-fragments.json'


def load_fragment_patterns(path=CATALOG):
    from ..dailydialog_quarantine import check_fragment_catalog

    content = json.loads(Path(path).read_text(encoding='utf-8'))
    if content.get('format') != 'reviewed-dialogue-fragments-v1':
        raise ValueError('Unsupported reviewed fragment catalog')
    patterns = content['patterns']
    validate_fragment_patterns(patterns)
    check_fragment_catalog(patterns, path)
    return patterns


def validate_fragment_patterns(patterns):
    ids = [p['id'] for p in patterns]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Reviewed fragments need distinct IDs')
    for row in patterns:
        if row['kind'] not in {'fact', 'partial', 'event', 'refine', 'mined_event', 'ambiguous', 'no_change', 'interpretation'}:
            raise ValueError('Unknown reviewed fragment meaning')
        if type(row['context']) is not bool or not row['assistant'].strip() or not row['reply'].strip():
            raise ValueError('Reviewed fragments need visible dialogue context')
        if row['kind'] == 'interpretation':
            if (not row['context'] or not isinstance(row.get('value'), str) or not row['value'].strip()
                    or not row.get('question') or row.get('review', {}).get('method') != 'direct_assistant_fragment_adapter'):
                raise ValueError('An utterance interpretation needs an explicit review and visible context')
        if row['kind'] == 'mined_event':
            if not row['context'] or row.get('phase', 'unspecified') != 'unspecified':
                raise ValueError('Reviewed elliptical events need context and cannot imply completion')
            if row.get('review', {}).get('method') != 'direct_assistant_fragment_adapter':
                raise ValueError('An elliptical mined event requires an explicit contextual adapter')
        if 'review' in row:
            review, source = row['review'], row.get('source', {})
            if (review.get('method') != 'direct_assistant_fragment_adapter' or
                    review.get('scope') != 'adapted_visible_pair_only' or not review.get('adaptation')):
                raise ValueError('Fragment review must state its limited adaptation scope')
            context, turn = source.get('context'), source.get('turn')
            span = source.get('span')
            if (not isinstance(context, list) or not context or
                    any(not isinstance(t, str) for t in context) or type(turn) is not int or
                    not 1 <= turn <= len(context) or not isinstance(span, list) or len(span) != 2 or
                    any(type(n) is not int for n in span) or
                    not 0 <= span[0] < span[1] <= len(context[turn-1]) or
                    context[turn-1][span[0]:span[1]] != source.get('text')):
                raise ValueError('Fragment review needs an exact source span and embedded context')
            if (review.get('context_sha256') != fingerprint(context) or
                    review.get('source_sha256') != fingerprint(source['text'])):
                raise ValueError('Stale fragment review source or context')
            evidence = review.get('evidence_turns')
            if (not isinstance(evidence, list) or turn not in evidence or
                    any(type(t) is not int or not 1 <= t <= turn for t in evidence)):
                raise ValueError('Fragment review cannot use future turns as evidence')


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
    if kind == 'interpretation':
        # A claim about the reply's meaning, not a claim that an embedded
        # action happened. Keep the assistant prompt in its evidence.
        c.pair(text, facts=(_fact('user', 'utterance_interpretation', pattern['value'].format(**values)),), reference=reference)
        query = Query('lookup', relation='utterance_interpretation')
    elif kind == 'fact':
        c.pair(text, facts=(_fact('user', pattern['relation'], pattern['value'].format(**values),
                                  negative=pattern.get('negative', False)),), reference=reference)
        query = Query('lookup', relation=pattern['relation'])
    elif kind in ('partial', 'event', 'refine', 'mined_event'):
        arguments = tuple(sorted(EventArgument(role, value.format(**values), label)
                                 for role, value, label in pattern['arguments']))
        subject = MinedSubject(**pattern['subject']) if kind == 'mined_event' else None
        event = Event(pattern['predicate'], arguments, time=pattern.get('time', 'unspecified').format(**values),
                      operators=tuple(pattern['operators']), tense=pattern['tense'],
                      missing_roles=tuple(pattern.get('missing_roles', ())),
                      mined=kind == 'mined_event', subject=subject)
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
        # A reviewed update may use a literal outside the starter vocabulary.
        # Include that value among possible starting states: otherwise no
        # starter context may share its split, starving this pattern forever.
        # The transition checks below still keep EVERY visible state in one split.
        initial_values = {
            'location': list(COMMON_PLACES),
            'feeling': list(FEELINGS),
            'drink': list(DRINKS),
        }
        for pattern in patterns:
            relation = pattern.get('relation')
            if (pattern['kind'] == 'fact' and not pattern.get('negative', False)
                    and relation in initial_values):
                value = pattern['value']
                if '{' not in value and '}' not in value and value not in initial_values[relation]:
                    initial_values[relation].append(value)
        contexts = [c for c in product(*(initial_values[r] for r in ('location', 'feeling', 'drink')))
                    if _context_split(c) == split]
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
            episode = fragment_episode(pattern, context, split=split,
                                       user_only=generator.user_assistant_only,
                                       variant=rng.randrange(420))
            if generator.user_assistant_only:
                # Extraction exports current facts, not their retired values.
                # A location update can therefore erase the distinction between
                # two starting contexts. Keep every resulting visible context
                # in the same partition as the starting one.
                visible_contexts = [
                    tuple(t.after.find('user', relation) for relation in ('location', 'feeling', 'drink'))
                    for t in episode.transitions
                ]
                if any(any(f is None or f.polarity != 'positive' for f in facts)
                       or _context_split(tuple(f.value for f in facts)) != split
                       for facts in visible_contexts):
                    continue
            # A cross-split rejection must not skip first-pass pattern coverage.
            # A repeated scenario can move on to another pattern as before.
            index += 1
            if episode.scenario_key in seen:
                continue
            seen.add(episode.scenario_key)
            yield episode
            produced += 1
            if produced == count:
                return
        rng.shuffle(contexts)
    raise ValueError(f'fragment_meaning/{split}: exhausted distinct scenarios ({produced}/{count})')
