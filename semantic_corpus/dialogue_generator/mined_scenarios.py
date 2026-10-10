"""Dialogue tasks over every indexed mined head, with explicit coverage.

Compositional candidates test occurrence rather than presumed telic completion.
Attested-only candidates test scope of reported wording, never the truth of an
arbitrary embedded clause. Both keep the source analysis in the episode audit.
"""
from dataclasses import asdict, replace
from itertools import product
import random

from .events import Event, EventArgument, event_question, event_statement, occurrence_evidence
from .observed_clauses import ObservedClause
from .models import Episode, Query, answer_query
from .mined_subjects import POSSIBILITY_PROFILES

PROFILES = (('want',), ('need',), ('plan',), ('not', 'want'), ('want', 'not'),
            ('can',), ('not', 'can'), ('could',), ('not', 'could'),
            ('must',), ('must', 'not'), ('may',), ('may', 'not'), ('might',),
            ('have_to',), ('not', 'have_to'), ('be_able_to',), ('be_going_to',),
            ('may', 'have'), ('must', 'have'), ('let',), ('not', 'let'), ('let_us',))
QUOTED_PROFILES = ('desired', 'planned', 'suggested', 'possible', 'required', 'permitted')


def mined_episodes(generator, count, split):
    from .generator import COMMON_PLACES, FEELINGS, DRINKS, _Conversation, _context_split, _fact, fingerprint
    from .events import MODALS
    bank = generator.mined_bank
    if not bank.lemmas:
        raise ValueError('No eligible mined predicates are available for dialogue generation')
    rng = random.Random(fingerprint(['mined-dialogue-v2', generator.seed, split]))
    # Same context partition as the manual user/assistant tasks. Names in source
    # quotations are source vocabulary, never extra conversation participants.
    contexts = [c for c in product(COMMON_PLACES, FEELINGS, DRINKS) if _context_split(c) == split]
    lemmas = list(bank.lemmas)
    rng.shuffle(lemmas)
    index = produced = 0
    while produced < count:
        lemma = lemmas[index % len(lemmas)]
        cycle = index // len(lemmas)
        def selection_rng(head):
            return random.Random(fingerprint(['mined-dialogue-v2', generator.seed, split, head, cycle]))
        if index % len(lemmas) == 0 and count - produced >= 3 * len(lemmas):
            bank.preload([bank.reference(head, selection_rng(head), native=cycle % 4 != 3) for head in lemmas])
        rng = selection_rng(lemma)
        index += 1
        # All references remain reachable; one attested-only cycle also covers
        # complex templates of verbs which have a simple compositional use.
        row, spec, source = bank.sample(lemma, rng, native=cycle % 4 != 3)
        context = rng.choice(contexts)
        style = rng.randrange(3)
        profile = rng.choice(PROFILES)
        if spec:
            subject = spec['subject']
            if subject.label != 'PERSON':
                profile = rng.choice(POSSIBILITY_PROFILES)
            elif subject.text != 'user' and profile == ('let_us',):
                profile = ('may',)
            args = []
            if spec['tail']:
                args.append(EventArgument('complement', spec['tail'], 'UNRESOLVED'))
            base = Event(spec['predicate'], tuple(args), tense='past', phase='reported', mined=True, subject=subject)
            initial = replace(base, phase='unspecified', operators=profile,
                              tense='modal' if any(op in MODALS for op in profile) else 'present',
                              permitter=EventArgument('permitter', 'user' if subject.text == 'assistant' else 'assistant', 'PERSON') if 'let' in profile else None)
            scope = spec['source_scope']
            preserved = bool(scope.operators not in ((), ('not',)) or scope.phase == 'ongoing' or scope.tense == 'future'
                             or scope.operators == ('not',))
            if preserved:
                initial = replace(base, operators=scope.operators, phase=scope.phase, tense=scope.tense)
                profile = scope.operators
            source = {**source, 'realization': 'compositional',
                      'scope_origin': 'source' if preserved else 'generated',
                      'note': 'Subject and reviewed source operator scope are preserved. Time is normalized to today; complements keep attested order and unresolved roles. Completion is not inferred.'}
            condition = ('source:' if preserved else 'generated:') + ('>'.join(profile) or scope.phase)
            query = Query('mined_occurrence', event=base)
        else:
            base = Event('@clause', (), observation=ObservedClause(row['text'].strip()))
            condition = rng.choice(QUOTED_PROFILES)
            initial = replace(base, observation=replace(base.observation, status=condition))
            source = {**source, 'realization': 'attested_quotation',
                      'note': 'This tests reported wording, not completion or truth of its embedded events.'}
            query = Query('observed_report', event=base)
        group = fingerprint(['mined-dialogue-v2', split, context, asdict(initial), source['template_id'], lemma, cycle])
        outcomes = ['yes', 'no', 'unknown']
        rng.shuffle(outcomes)
        for outcome in outcomes:
            c = _Conversation()
            place, feeling, drink = context
            c.pair(f"I'm in {place}. I feel {feeling}, and I prefer {drink}.",
                   facts=(_fact('user', 'location', place), _fact('user', 'feeling', feeling), _fact('user', 'drink', drink)))
            c.pair(event_statement(initial, perspective='user', style=style), events=(initial,))
            known = occurrence_evidence(initial) if spec else 'unknown'
            if known != 'unknown' and known != outcome:
                old = c.memory.find_event(initial)
                c.pair('Withdraw my last report.', event_invalidate=(old.id,))
            if outcome != 'unknown' and outcome != known:
                final = (replace(base, operators=('not',) if outcome == 'no' else ()) if spec else
                         replace(base, observation=replace(base.observation, status='denied' if outcome == 'no' else 'reported')))
                c.pair(event_statement(final, perspective='user', style=style), events=(final,))
            elif outcome == 'unknown' and known == 'unknown' and cycle % 2:
                # A withdrawn explicit report must not survive in active memory.
                c.pair(event_statement(base, perspective='user', style=style), events=(base,))
                old = c.memory.find_event(base)
                c.pair('Withdraw my last report.', event_invalidate=(old.id,))
            c.user(event_question(base, style=style))
            answer, expected, supports = answer_query(c.memory, query)
            key = fingerprint([group, outcome])
            episode = Episode(f'dialogue-{key[:24]}-v{style}', 'mined_predicates', split, key,
                              tuple(c.turns), tuple(c.transitions), c.memory, query, answer, expected,
                              supports, group, style, f'{condition}:{outcome}', source)
            episode.validate()
            yield episode
            produced += 1
            if produced == count:
                return
