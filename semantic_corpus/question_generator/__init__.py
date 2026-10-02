"""Turning verified semantic records into natural questions and answers.

This stage never invents ground truth. It is handed spans, predicates,
relations and dialogue annotations that something else already established,
and its only job is to word questions about them — which is why it is a
separate module from the semantic generator.
"""

from .answers import NO_ANSWER_REPLIES, QAExample, QAKind, phrase_answer
from .generator import QuestionGenerator, dialogue_act_questions
from .negatives import (
    missing_role_questions,
    no_answer_questions,
    omitted_argument_questions,
    underdetermined_type_questions,
)
from .paraphrases import RULES, paraphrase, paraphrase_all
from .templates import (
    ArgumentView,
    ParadigmResolver,
    atomic_questions,
    build_views,
    compound_questions,
    entity_type_questions,
    ontology_questions,
    property_questions,
    regular_forms,
    yes_no_questions,
)

__all__ = [
    "QAExample",
    "QAKind",
    "NO_ANSWER_REPLIES",
    "phrase_answer",
    "QuestionGenerator",
    "dialogue_act_questions",
    "ArgumentView",
    "ParadigmResolver",
    "regular_forms",
    "build_views",
    "atomic_questions",
    "compound_questions",
    "entity_type_questions",
    "ontology_questions",
    "property_questions",
    "yes_no_questions",
    "no_answer_questions",
    "omitted_argument_questions",
    "missing_role_questions",
    "underdetermined_type_questions",
    "paraphrase",
    "paraphrase_all",
    "RULES",
]
