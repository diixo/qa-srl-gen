"""Reviewed complement grammar, with defective modals kept out of inflection.

Could/might are explicit modal forms, not automatic past-tense replacements.
The compact contract preserves ambiguous ability/permission/possibility readings
without treating any of them as evidence that the embedded action occurred.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Operator:
    lemma: str
    past: str | None
    third: str | None
    gerund: str | None
    complement: str
    category: str
    participle: str | None = None


OPERATORS = {item.lemma: item for item in (
    Operator('want', 'wanted', 'wants', 'wanting', 'to', 'desire'),
    Operator('need', 'needed', 'needs', 'needing', 'to', 'necessity'),
    Operator('plan', 'planned', 'plans', 'planning', 'to', 'intention'),
    Operator('intend', 'intended', 'intends', 'intending', 'to', 'intention'),
    Operator('decide', 'decided', 'decides', 'deciding', 'to', 'decision'),
    Operator('try', 'tried', 'tries', 'trying', 'to', 'attempt'),
    Operator('manage', 'managed', 'manages', 'managing', 'to', 'success'),
    Operator('fail', 'failed', 'fails', 'failing', 'to', 'failure'),
    Operator('start', 'started', 'starts', 'starting', 'gerund', 'onset'),
    Operator('finish', 'finished', 'finishes', 'finishing', 'gerund', 'completion'),
    Operator('stop', 'stopped', 'stops', 'stopping', 'gerund', 'cessation'),
    *(Operator(word, None, None, None, 'modal', category) for word, category in (
        ('can', 'ability_or_permission'), ('could', 'remote_ability_or_possibility'),
        ('must', 'obligation_or_deduction'), ('may', 'possibility_or_permission'), ('might', 'possibility'))),
    Operator('have_to', 'had', 'has', 'having', 'have_to', 'obligation', 'had'),
    Operator('be_able_to', 'was', 'is', 'being', 'be_able_to', 'ability', 'been'),
    Operator('be_going_to', 'was', 'is', None, 'be_going_to', 'prospective', 'been'),
    Operator('have', 'had', 'has', 'having', 'participle', 'perfect', 'had'),
    Operator('be', 'was', 'is', 'being', 'gerund', 'progressive', 'been'),
    Operator('let', 'let', 'lets', 'letting', 'let', 'permission', 'let'),
    Operator('let_us', None, None, None, 'hortative', 'proposal'),
)}
MODALS = frozenset(op.lemma for op in OPERATORS.values() if op.complement == 'modal')


def be_form(form, subject):
    if form == 'present':
        return 'am' if subject == 'I' else 'are' if subject.lower() in ('you', 'we', 'they') else 'is'
    if form == 'past':
        return 'were' if subject.lower() in ('you', 'we', 'they') else 'was'
    return {'base': 'be', 'third': 'is', 'gerund': 'being', 'participle': 'been'}[form]


def validate_chain(operators, tense, *, is_state=False):
    if len(operators) > 4 or any(op not in OPERATORS and op != 'not' for op in operators):
        raise ValueError('Unsupported event operator chain')
    if any(a == b == 'not' for a, b in zip(operators, operators[1:])):
        raise ValueError('Repeated adjacent negation is not supported')
    modal_positions = [i for i, op in enumerate(operators) if op in MODALS]
    if modal_positions:
        expected = 1 if operators[:1] == ('not',) else 0
        if modal_positions != [expected] or tense != 'modal':
            raise ValueError('A finite modal must head its chain and use modal tense; no infinitive/gerund modals')
    elif tense == 'modal':
        raise ValueError('Modal tense requires a finite modal')
    if operators[:2] in (('not', 'must'), ('not', 'may'), ('not', 'might')):
        raise ValueError('Use modal > not for embedded negation; not > have_to for absence of obligation')
    if operators[:2] in (('can', 'not'), ('could', 'not')):
        raise ValueError('Use not > can/could for inability; ability to refrain needs a separate lexical construction')
    if operators.count('let') > 1 or operators.count('have') > 1 or operators.count('be') > 1:
        raise ValueError('Repeated control/aspect operator is unsupported')
    if 'let_us' in operators and (operators[0] != 'let_us' or tense != 'present'):
        raise ValueError('Hortative let_us must head a present proposal')
    if is_state and any(op in ('be', 'start', 'stop', 'finish', 'manage', 'fail') for op in operators):
        raise ValueError('These state senses do not license progressive or completion operators')
    if operators in (('be',), ('not', 'be')):
        raise ValueError('Use ongoing phase for a bare progressive, not a duplicate operator encoding')
    form = 'present'
    for op in operators:
        if op == 'not':
            continue
        entry = OPERATORS[op]
        if entry.gerund is None and form == 'gerund':
            raise ValueError(f'{op} has no supported gerund')
        if entry.complement == 'hortative' and form != 'present':
            raise ValueError('A proposal cannot be embedded as an infinitive')
        form = 'gerund' if entry.complement == 'gerund' else 'participle' if entry.complement == 'participle' else 'base'
