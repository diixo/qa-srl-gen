"""Deliberate action forms for the dialogue pilot, independent of QA-SRL frames."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    relation: str
    base: str
    participle: str
    gerund: str
    objects: tuple[str, ...]


ACTIONS = (
    Action('visited', 'visit', 'visited', 'visiting',
           ('the park', 'the library', 'the museum', 'the cafe', 'the bookstore')),
    Action('bought', 'buy', 'bought', 'buying',
           ('a book', 'a ticket', 'a notebook', 'a coat', 'a lamp')),
    Action('called', 'call', 'called', 'calling',
           ('the doctor', 'the dentist', 'the hotel', 'the office', 'the repair service')),
    Action('sent', 'send', 'sent', 'sending',
           ('the email', 'the letter', 'the report', 'the invitation', 'the message')),
    Action('read', 'read', 'read', 'reading',
           ('the book', 'the article', 'the report', 'the instructions', 'the letter')),
    Action('completed_break', 'take', 'taken', 'taking',
           ('a short walk', 'a music break', 'a snack break', 'a rest break', 'a coffee break')),
)
ACTION_BY_RELATION = {action.relation: action for action in ACTIONS}
CONTRAST_TASKS = ('desire_vs_done', 'suggestion_vs_done')


def action_statement(action: Action, value: str, *, subject='You', modality='asserted',
                     negative=False, style=0) -> str:
    """Five equivalent surfaces; styles 3/4 are reserved for challenge exports."""
    if style not in range(5):
        raise ValueError('Unknown memory style')
    user = subject in ('You', 'I')
    have = 'have' if user else 'has'
    not_ = 'not ' if negative else ''
    if modality in ('desired', 'planned'):
        if modality == 'desired':
            verbs = ('want' if user else 'wants', 'would like', 'hope' if user else 'hopes',
                     'expressed a wish', 'would like')
        else:
            verbs = ('plan' if user else 'plans', 'intend' if user else 'intends',
                     'are planning' if subject == 'You' else 'am planning' if subject == 'I' else 'is planning',
                     'made a plan', 'intend' if user else 'intends')
        return f'{subject} {verbs[style]} {not_}to {action.base} {value}.'
    if modality != 'asserted':
        raise ValueError(f'Unsupported action modality: {modality}')
    patterns = (
        f'{subject} {have} {not_}{action.participle} {value}.',
        f'{subject} reported {not_}having {action.participle} {value}.',
        f'{subject} said: "I have {not_}{action.participle} {value}."',
        f'{subject} mentioned {not_}having {action.participle} {value}.',
        f'{subject} stated: "I have {not_}{action.participle} {value}."',
    )
    return patterns[style]


def action_question(action: Action, value: str, style: int) -> str:
    return (
        f'Have I {action.participle} {value}?',
        f'Have I already {action.participle} {value}?',
        f'Did I {action.base} {value}?',
        f'Is there confirmation that I have {action.participle} {value}?',
        f'Can you tell whether I have {action.participle} {value}?',
    )[style]
