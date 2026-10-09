"""Render the pilot's active memory as English context for response training."""

from .models import Fact, Memory
from .actions import ACTION_BY_RELATION, action_statement
from .events import event_statement

MEMORY_TEXT_VERSION = "memory-text-v5"


def _fact_text(fact: Fact, style: int = 0) -> str:
    user = fact.subject == "user"
    subject = "You" if user else fact.subject
    possessive = "Your" if user else f"{subject}'s"
    value = fact.value
    if fact.polarity not in ("positive", "negative"):
        raise ValueError(f"Unsupported fact polarity: {fact.polarity}")
    negative = fact.polarity == "negative"
    if fact.relation in ACTION_BY_RELATION:
        return action_statement(ACTION_BY_RELATION[fact.relation], value, subject=subject,
                                modality=fact.modality, negative=negative, style=style)
    if fact.modality != "asserted":
        raise ValueError(f"Unsupported fact modality: {fact.modality}")

    be = "are" if user else "is"
    do_not = "do not" if user else "does not"
    not_ = " not" if negative else ""
    if fact.relation == "location":
        return f"{subject} {be}{not_} in {value}."
    if fact.relation == "meeting_day":
        return f"{possessive} meeting is{not_} on {value}."
    if fact.relation == "feeling":
        verb = f"{do_not} feel" if negative else "feel" if user else "feels"
        return f"{subject} {verb} {value}."
    if fact.relation == "drink":
        verb = f"{do_not} prefer" if negative else "prefer" if user else "prefers"
        return f"{subject} {verb} {value}."
    if fact.relation == "selection":
        return f"{subject} {'did not choose' if negative else 'chose'} {value}."
    raise ValueError(f"Unsupported fact relation: {fact.relation}")


def render_memory(memory: Memory, *, style: int = 0) -> str:
    """Render state only, without IDs, evidence, retired facts or query hints.

    Suggestions and ordered options stay separate from user reports. Missing
    facts remain unknown; this function never invents negative claims for them.
    """
    if type(style) is not int or style not in range(5):
        raise ValueError('Unknown memory style')
    lines = ["Known facts:"]
    lines.extend(f"- {_fact_text(fact, style)}" for fact in memory.facts)
    if not memory.facts:
        lines.append("- No facts recorded.")
    if memory.events:
        lines.extend(('', 'Reported events and intentions:'))
        lines.extend('- ' + event_statement(claim.event, style=style) for claim in memory.events)
    if memory.assistant_actions:
        lines.extend(("", "Assistant suggestions and options (earliest first):"))
    for action in memory.assistant_actions:
        if not action.items:
            raise ValueError("An assistant action needs at least one item")
        if action.kind == "options":
            lines.append("- The assistant offered these options, in this order:")
            lines.extend(f"  {index}. {item}" for index, item in enumerate(action.items, 1))
        elif action.kind == "suggestion":
            patterns = ('The assistant suggested {}.', 'The assistant offered a suggestion: {}.',
                        'A suggestion from the assistant was: {}.', 'The assistant proposed {}.',
                        'The assistant recommended {}.')
            lines.extend('- ' + patterns[style].format(item) for item in action.items)
        else:
            raise ValueError(f"Unsupported assistant action: {action.kind}")
    return "\n".join(lines)
