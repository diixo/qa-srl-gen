"""Subject agreement over the unchanged, singular QA-SRL verb grammar."""

from .qasrl_core.frame import Frame


def verb_words(frame: Frame, *, plural: bool = False, question: bool = False) -> list[str]:
    """Build the chain, then agree its finite head with an explicit subject.

    Wh-subject gaps keep the grammar's singular default ("Who buys ...?").
    Plurality belongs to surface syntax, not to the semantic GROUP label:
    "the team buys" but "the police buy".
    """
    # The upstream QA-SRL grammar assumes lexical verbs use do-support. Copular
    # be instead inverts/negates itself; leave the ported grammar unchanged.
    if (frame.verb_inflected_forms.stem == 'be' and frame.is_finite and not frame.is_modal
            and not frame.is_perfect and not frame.is_progressive and not frame.structure.is_passive):
        word = ('were' if plural else 'was') if frame.tense == 'past' else ('are' if plural else 'is')
        return [word + ("n't" if frame.is_negated else '')]
    stack = frame.get_verb_stack()
    if question:
        stack = frame.split_verb_stack_if_necessary(stack)
    if not plural or not frame.is_finite or frame.is_modal:
        return stack
    head = stack[0]
    replacements = {"is": "are", "isn't": "aren't", "was": "were", "wasn't": "weren't",
                    "has": "have", "hasn't": "haven't", "does": "do", "doesn't": "don't"}
    if head in replacements:
        stack[0] = replacements[head]
    elif frame.tense == "present" and head == frame.verb_inflected_forms.present_singular_3rd:
        stack[0] = frame.verb_inflected_forms.stem
    return stack
