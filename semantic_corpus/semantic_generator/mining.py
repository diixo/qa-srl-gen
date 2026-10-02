"""Proposing new pool entries from a plain-text corpus.

The generator's weakest point is its pool: 84 entities produce a narrow,
repetitive corpus however many examples are drawn from it. A corpus of
ordinary sentences such as ``eng-base.jsonl`` contains thousands of names
and nouns, and some of their types can be read off context without any
model.

The method is evidence, not guessing. A capitalised form is counted once per
context that implies a type — *Mr Tom* and *Tom said* imply a person, *in
Boston* and *flew to Boston* imply a place — and a form is only proposed
when one type has clear support and the others have almost none. Everything
else is reported as contested and left out.

Two rules keep this honest:

* **Proposals are written to a file, not merged.** The output is a
  candidate pool a person reads before accepting; the generator never picks
  it up on its own. A wrongly typed name silently poisons every sentence it
  appears in and every question asked about it.
* **Precision over recall.** The patterns below are narrow on purpose. The
  pool needs a few thousand reliable entries, not every proper noun in the
  corpus, and a contested form is worth less than no form at all.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from ..ontology import Label, LabelSet
from .substitutions import Entity, EntityPool, entity_to_json

__all__ = [
    "Proposal",
    "MiningReport",
    "mine_entities",
    "iter_sentences",
    "write_proposals",
]

_WORD = re.compile(r"[A-Za-z]+(?:[-][A-Za-z]+)*")


def raw_possessive(word: str) -> bool:
    """Whether a token is a possessive form such as ``Tom's``."""
    stripped = word.strip(",.!?;:\"()")
    return stripped.endswith(("'s", "’s", "s'"))

#: Titles that make the following capitalised word a person.
_TITLES = frozenset({"mr", "mrs", "ms", "dr", "miss", "sir", "madam", "uncle", "aunt"})
#: Verbs whose subject is a person.
_SAYING_VERBS = frozenset(
    """said says say told tells tell asked asks ask thinks think thought
    knows know knew wants want wanted likes like liked loves love hates
    hate believes believe replied answered laughed smiled agreed""".split()
)
#: Prepositions that mark a place when they govern a bare proper noun.
_PLACE_PREPOSITIONS = frozenset({"in", "from", "at", "near", "around", "across"})
#: Verbs of motion whose "to" complement is a destination, not a recipient.
_MOTION_VERBS = frozenset(
    """go goes went gone going come comes came travel travels travelled
    traveled move moves moved fly flies flew drive drives drove walk walks
    walked return returns returned arrive arrives arrived""".split()
)
#: Forms that are capitalised but never entities of ours.
_STOPLIST = frozenset(
    """i i'm i've i'll i'd the a an this that these those
    he she it we they you him her them us me everyone everybody someone
    somebody anyone nobody
    english french german spanish italian russian chinese japanese latin
    greek portuguese arabic hindi korean swedish dutch
    american british canadian australian european
    americans britons canadians australians europeans
    javascript python java
    monday tuesday wednesday thursday friday saturday sunday
    january february march april may june july august september october
    november december god christmas ok tv dvd cd usa uk""".split()
)
#: Capitalised words that never belong inside a name, so a run stops before
#: them: *It's Tom I want to see* must yield ``Tom``, not ``Tom I``.
_NEVER_IN_A_NAME = frozenset({"I", "I'm", "I've", "I'll", "I'd", "The", "A", "An"})


@dataclass(frozen=True, slots=True)
class Proposal:
    """A surface form, the type the evidence supports, and that evidence."""

    text: str
    label: Label
    support: int
    contested_by: Mapping[str, int] = field(default_factory=dict)
    examples: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        return not self.contested_by

    def to_entity(self, type_name: str | None = None) -> Entity:
        return Entity(
            text=self.text,
            labels=LabelSet(frozenset({self.label, Label.NAMED_ENTITY})),
            type_name=type_name,
            is_named=True,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "text": self.text,
            "label": str(self.label),
            "support": self.support,
            "contested_by": dict(self.contested_by),
            "examples": list(self.examples),
        }


@dataclass(slots=True)
class MiningReport:
    """What a pass over a corpus found, including what it refused."""

    sentences: int = 0
    capitalised_forms: int = 0
    proposed: list[Proposal] = field(default_factory=list)
    contested: list[Proposal] = field(default_factory=list)
    below_threshold: int = 0

    def by_label(self) -> Counter:
        return Counter(str(p.label) for p in self.proposed)

    def __str__(self) -> str:
        return (
            f"{self.sentences} sentences, {self.capitalised_forms} capitalised "
            f"forms seen; {len(self.proposed)} proposed "
            f"({dict(self.by_label())}), {len(self.contested)} contested, "
            f"{self.below_threshold} below threshold"
        )


def iter_sentences(path: Path | str, field: str = "example") -> Iterator[str]:
    """Stream sentences from a JSONL corpus, one per line."""
    with Path(path).open("rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get(field)
            if isinstance(text, str) and text.strip():
                yield text


def _strip(word: str) -> str:
    return word.strip(",.!?;:\"'()")


def _capitalised_run(words: Sequence[str], index: int) -> tuple[str, int] | None:
    """The whole capitalised name starting at *index*, and where it ends.

    Taking one token turned *New York* into *New*, which then entered the
    pool as a place name that does not exist.
    """
    end = index
    while end < len(words):
        word = _strip(words[end])
        if not word or not word[0].isupper() or not _WORD.fullmatch(word):
            break
        if word in _NEVER_IN_A_NAME:
            break
        end += 1
        # A run stops at punctuation that closed the previous token.
        if words[end - 1] != word:
            break
    if end == index:
        return None
    return " ".join(_strip(w) for w in words[index:end]), end


def _evidence(words: Sequence[str], index: int, after: int) -> Label | None:
    """The type the context around a name implies, if any."""
    previous = _strip(words[index - 1]).lower() if index else ""
    following = _strip(words[after]).lower() if after < len(words) else ""

    if previous in _TITLES:
        return Label.PERSON
    if following in _SAYING_VERBS:
        return Label.PERSON
    if previous in _PLACE_PREPOSITIONS:
        return Label.LOCATION
    if previous == "to" and index >= 2:
        before = _strip(words[index - 2]).lower()
        if before in _MOTION_VERBS:
            return Label.LOCATION
    return None


def mine_entities(
    sentences: Iterable[str],
    *,
    min_support: int = 5,
    max_contested_share: float = 0.1,
    limit: int | None = None,
    examples_per_form: int = 2,
) -> MiningReport:
    """Collect typed proposals from *sentences*.

    A form is proposed when one type has at least *min_support* observations
    and every other type has at most *max_contested_share* of them. The
    default is deliberately strict: a name that reads as a person in nine
    sentences and a place in one is not reliable enough to generate from.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    examples: dict[tuple[str, Label], list[str]] = defaultdict(list)
    report = MiningReport()

    for sentence in sentences:
        if limit is not None and report.sentences >= limit:
            break
        report.sentences += 1
        words = sentence.split()
        index = 0
        while index < len(words):
            run = _capitalised_run(words, index)
            if run is None:
                index += 1
                continue
            word, after = run
            index = after
            if word.lower() in _STOPLIST:
                continue
            # A possessive names someone's thing, not the thing itself:
            # "at Tom's" is evidence about a place, but "Tom's" is not its
            # name, and putting it in the pool invents a location.
            if raw_possessive(words[after - 1]):
                continue
            report.capitalised_forms += 1
            start = after - len(word.split())
            if start == 0:
                continue  # a sentence-initial capital explains itself
            label = _evidence(words, start, after)
            if label is None:
                continue
            counts[word][label] += 1
            if len(examples[(word, label)]) < examples_per_form:
                examples[(word, label)].append(sentence)

    for text, tally in sorted(counts.items()):
        label, support = tally.most_common(1)[0]
        if support < min_support:
            report.below_threshold += 1
            continue
        others = {
            str(other): n for other, n in tally.items() if other is not label and n
        }
        proposal = Proposal(
            text=text,
            label=label,
            support=support,
            contested_by=others,
            examples=tuple(examples[(text, label)]),
        )
        if others and max(others.values()) > support * max_contested_share:
            report.contested.append(proposal)
        else:
            report.proposed.append(proposal)

    report.proposed.sort(key=lambda p: (-p.support, p.text))
    report.contested.sort(key=lambda p: (-p.support, p.text))
    return report


def write_proposals(report: MiningReport, path: Path | str) -> Path:
    """Save proposals for a person to read before anything is merged."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "_comment": [
                    "Mined entity proposals. Nothing here is in the pool yet.",
                    "Each row records the evidence that typed the form, and",
                    "'contested' rows are listed separately because one bad",
                    "type poisons every sentence the name appears in.",
                ],
                "summary": {
                    "sentences": report.sentences,
                    "proposed": len(report.proposed),
                    "contested": len(report.contested),
                    "by_label": dict(report.by_label()),
                },
                "proposed": [p.to_json() for p in report.proposed],
                "contested": [p.to_json() for p in report.contested],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
