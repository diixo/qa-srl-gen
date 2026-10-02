"""Rewording a question without changing what it asks.

The handoff asks for paraphrased variants of one relation, so that a model
learns the relation rather than one phrasing of it. The rewrites here are
rule-based and deterministic: a generated corpus whose paraphrases varied
between runs would make a held-out split meaningless, and there is no
paraphrase model available (nor would one be allowed to come from Hugging
Face).

Rules are conservative by design. A rewrite that subtly changes the question
produces an example whose stored answer is now wrong, which is worse than
having no paraphrase at all — so each rule fires only on a shape it fully
recognises, and anything else is left alone.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable, Iterable, Iterator, Sequence

from .answers import QAExample, QAKind

__all__ = ["RULES", "paraphrase", "paraphrase_all"]

#: ``(pattern, replacement)`` pairs applied to the question text. Each must
#: preserve the answer exactly; that is the whole contract.
RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # Adjuncts: ask for the same thing more explicitly.
    (re.compile(r"^Where did (.+?) (.+)\?$"), r"In what place did \1 \2?"),
    (re.compile(r"^When did (.+?) (.+)\?$"), r"At what time did \1 \2?"),
    (re.compile(r"^Where did this happen\?$"), "In what place did this happen?"),
    (re.compile(r"^When did this happen\?$"), "At what time did this happen?"),
    # Event questions.
    (re.compile(r"^What did (.+?) do\?$"), r"What was \1 doing?"),
    # Entity type.
    (
        re.compile(r"^What kind of entity is (.+?)\?$"),
        r"What sort of thing is \1?",
    ),
    # Recipient, with the preposition fronted.
    (re.compile(r"^Who did (.+?) (.+) to\?$"), r"To whom did \1 \2?"),
)

# Deliberately absent: a polar rewrite such as "Did X run?" -> "Is it true
# that X run?". Undoing do-support needs the finite form, which a regex over
# the surface cannot recover, and the result would be ungrammatical.


def paraphrase(example: QAExample) -> list[QAExample]:
    """Variants of *example* that ask the same thing.

    The answer, evidence and answerability are carried over untouched; only
    the question text changes, and ``kind`` becomes
    :attr:`~.answers.QAKind.PARAPHRASE` so the two can be told apart when
    balancing a corpus.
    """
    question = example.question.strip()
    out: list[QAExample] = []
    seen = {question}
    for pattern, replacement in RULES:
        match = pattern.match(question)
        if not match:
            continue
        rewritten = pattern.sub(replacement, question)
        rewritten = _tidy(rewritten)
        if rewritten in seen or not rewritten.endswith("?"):
            continue
        seen.add(rewritten)
        out.append(
            replace(
                example,
                question=rewritten,
                kind=QAKind.PARAPHRASE,
                metadata={**dict(example.metadata), "paraphrase_of": example.question},
            )
        )
    return out


def _tidy(text: str) -> str:
    """Fix the spacing a rewrite can leave behind."""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1].upper() + text[1:]


def paraphrase_all(examples: Iterable[QAExample]) -> Iterator[QAExample]:
    """Yield each example followed by its paraphrases."""
    for example in examples:
        yield example
        yield from paraphrase(example)
