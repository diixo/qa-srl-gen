"""Verb paradigms from the Wiktionary scrape shipped with QA-SRL.

The resource is ``data/wiktionary/en_verb_inflections.txt``: a five-column TSV
with **no header row**, one paradigm per line::

    give    gives    giving    gave    given

What the dictionary is good for, and what it is not
---------------------------------------------------
It proposes candidates and reconstructs paradigms. It does not decide word
class or meaning. Known limits, all observable in the shipped file:

* ``be`` is absent entirely, along with other suppletive paradigms;
* a lemma can appear more than once with competing paradigms
  (``awaken``, ``chide``, ``plead``, ``rewet``), so the forward index maps a
  lemma to a *list*;
* one surface form routinely belongs to many paradigms — 21 523 of the 65 991
  distinct surfaces are ambiguous — so the reverse index maps a surface to a
  list of ``(paradigm, form)`` pairs;
* the scrape contains junk rows (a bare ``-``, stray affixes such as ``ed``),
  which :func:`load_inflections` filters out;
* membership in the dictionary is never on its own a reason to label a token
  as a predicate.

``be`` is supplied separately by :data:`SUPPLETIVE_PARADIGMS` because QA-SRL
needs it to render questions and the scrape does not have it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from .models import InflectedForms, VerbForm

__all__ = [
    "SUPPLETIVE_PARADIGMS",
    "BE_FORMS",
    "DO_FORMS",
    "HAVE_FORMS",
    "AUXILIARY_PARADIGMS",
    "InflectionLexicon",
    "load_inflections",
    "load_postags",
    "load_verb_phrases",
]

#: Paradigms the Wiktionary scrape omits or gets wrong. ``be`` is the only one
#: QA-SRL cannot do without; ``past`` holds the third-person singular form,
#: matching how the bank stores it.
SUPPLETIVE_PARADIGMS: dict[str, InflectedForms] = {
    "be": InflectedForms(
        stem="be",
        present_singular_3rd="is",
        present_participle="being",
        past="was",
        past_participle="been",
    ),
}

#: The three auxiliary paradigms the verb chain conjugates on its own, spelled
#: out rather than looked up: building a question must not depend on what a
#: scraped dictionary happens to contain. ``be`` uses the singular forms,
#: matching ``InflectedForms.beSingularForms`` upstream.
BE_FORMS: InflectedForms = SUPPLETIVE_PARADIGMS["be"]
DO_FORMS: InflectedForms = InflectedForms(
    stem="do",
    present_singular_3rd="does",
    present_participle="doing",
    past="did",
    past_participle="done",
)
HAVE_FORMS: InflectedForms = InflectedForms(
    stem="have",
    present_singular_3rd="has",
    present_participle="having",
    past="had",
    past_participle="had",
)
AUXILIARY_PARADIGMS: tuple[InflectedForms, ...] = (BE_FORMS, DO_FORMS, HAVE_FORMS)

_JUNK_SURFACES = frozenset({"", "-", "--", "_"})


@dataclass(frozen=True, slots=True)
class InflectionLexicon:
    """Forward and reverse views over a set of verb paradigms."""

    by_lemma: Mapping[str, tuple[InflectedForms, ...]]
    by_surface: Mapping[str, tuple[tuple[InflectedForms, VerbForm], ...]]

    def __len__(self) -> int:
        return len(self.by_lemma)

    def __contains__(self, lemma: str) -> bool:
        return lemma in self.by_lemma

    def paradigms(self, lemma: str) -> tuple[InflectedForms, ...]:
        """All paradigms recorded for *lemma* (empty tuple if unknown)."""
        return self.by_lemma.get(lemma, ())

    def paradigm(self, lemma: str) -> InflectedForms | None:
        """The first paradigm for *lemma*, or ``None``.

        Use :meth:`paradigms` when the competing analyses matter; a handful of
        lemmas genuinely have two.
        """
        found = self.by_lemma.get(lemma)
        return found[0] if found else None

    def analyses(self, surface: str) -> tuple[tuple[InflectedForms, VerbForm], ...]:
        """Every ``(paradigm, form)`` the surface string could realise.

        Homonymy is the normal case, not the exception: ``read`` is the stem,
        the past and the past participle of one paradigm, and ``left`` belongs
        to both ``leave`` and ``left``.
        """
        return self.by_surface.get(surface, ())

    def lemmas_for(self, surface: str) -> tuple[str, ...]:
        """Distinct lemmas that *surface* could belong to, in first-seen order."""
        seen: dict[str, None] = {}
        for paradigm, _ in self.analyses(surface):
            seen.setdefault(paradigm.stem, None)
        return tuple(seen)

    def forms_of(self, surface: str) -> tuple[VerbForm, ...]:
        """Distinct forms *surface* could be, in canonical form order."""
        found = {form for _, form in self.analyses(surface)}
        return tuple(f for f in VerbForm if f in found)


def _rows(path: Path) -> Iterator[list[str]]:
    with path.open("rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != 5:
                raise ValueError(
                    f"{path}:{line_number}: expected 5 tab-separated fields, "
                    f"got {len(fields)}"
                )
            yield fields


def load_inflections(
    path: Path | str,
    *,
    include_suppletive: bool = True,
    extra: Iterable[InflectedForms] = (),
) -> InflectionLexicon:
    """Load ``en_verb_inflections.txt`` into forward and reverse indexes.

    Rows with a junk surface in any column are skipped. ``include_suppletive``
    adds :data:`SUPPLETIVE_PARADIGMS` (notably ``be``), which the scrape lacks.
    """
    by_lemma: dict[str, list[InflectedForms]] = defaultdict(list)
    by_surface: dict[str, list[tuple[InflectedForms, VerbForm]]] = defaultdict(list)

    def add(paradigm: InflectedForms) -> None:
        if paradigm in by_lemma[paradigm.stem]:
            return
        by_lemma[paradigm.stem].append(paradigm)
        for form in VerbForm:
            surface = paradigm.get(form)
            entry = (paradigm, form)
            if entry not in by_surface[surface]:
                by_surface[surface].append(entry)

    for fields in _rows(Path(path)):
        if any(f.strip() in _JUNK_SURFACES for f in fields):
            continue
        add(
            InflectedForms(
                stem=fields[0],
                present_singular_3rd=fields[1],
                present_participle=fields[2],
                past=fields[3],
                past_participle=fields[4],
            )
        )

    if include_suppletive:
        for paradigm in SUPPLETIVE_PARADIGMS.values():
            add(paradigm)
    for paradigm in extra:
        add(paradigm)

    return InflectionLexicon(
        by_lemma={k: tuple(v) for k, v in by_lemma.items()},
        by_surface={k: tuple(v) for k, v in by_surface.items()},
    )


def load_postags(path: Path | str) -> dict[str, tuple[str, ...]]:
    """Load ``en_postags_withverb.txt`` as ``word -> tags``.

    Tags repeat on purpose: the file lists one tag per attested sense, so
    ``cat -> noun verb noun noun verb adj noun noun`` carries a frequency
    signal. The repetition is preserved; collapse it yourself if you only
    need the tag set.
    """
    tags: dict[str, tuple[str, ...]] = {}
    with Path(path).open("rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected 2 tab-separated fields"
                )
            tags[fields[0]] = tuple(fields[1].split())
    return tags


def load_verb_phrases(path: Path | str) -> tuple[tuple[str, ...], ...]:
    """Load ``verb_phrases.txt`` as tokenised multi-word verbal expressions."""
    phrases: list[tuple[str, ...]] = []
    with Path(path).open("rt", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                phrases.append(tuple(line.split(" ")))
    return tuple(phrases)


def paradigm_from_sequence(forms: Sequence[str]) -> InflectedForms:
    """Build :class:`~.models.InflectedForms` from five ordered surface strings."""
    stem, third, participle, past, past_participle = forms
    return InflectedForms(
        stem=stem,
        present_singular_3rd=third,
        present_participle=participle,
        past=past,
        past_participle=past_participle,
    )
