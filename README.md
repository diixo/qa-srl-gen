# qa-srl-gen — semantic corpus toolkit

Python core of QA-SRL plus, in later stages, a generator of semantic training
corpora. This repository currently implements **stage 1: the QA-SRL core**
(`semantic_corpus/qasrl_core`). The full plan lives in
[HANDOFF_SEMANTIC_CORPUS_RU.md](HANDOFF_SEMANTIC_CORPUS_RU.md).

No installation, no virtualenv, no build system: the package sits at the
repository root and imports directly. Python 3.10+, standard library only.

```bash
python -m pytest                       # 81 tests
python -m semantic_corpus.cli --help
```

## What works today

| Module | Purpose |
|---|---|
| `qasrl_core/models.py` | `Sentence`, `VerbEntry`, `QuestionLabel`, `QuestionSlots`, `AnswerJudgment`, `Span`, `InflectedForms` |
| `qasrl_core/bank_reader.py` | streaming readers for Bank 2.0/2.1 `.jsonl(.gz)` and the legacy `*.qa` text format |
| `qasrl_core/inflections.py` | Wiktionary paradigms: forward index `lemma → paradigm`, reverse index `surface → (paradigm, form)` |
| `qasrl_core/state_machine.py` | the auxiliary-chain grammar, in both directions |
| `qasrl_core/question_slots.py` | the nominal slots and their conventions |
| `qasrl_core/question_renderer.py` | slots → surface question |
| `qasrl_core/question_parser.py` | surface question → slots |
| `qasrl_core/validation.py` | invariant checks over slots, spans and sentences |
| `cli.py` | `inspect`, `roundtrip`, `validate`, `lookup` |

## Short example

```python
from semantic_corpus.qasrl_core.bank_reader import read_bank
from semantic_corpus.qasrl_core.question_parser import parse_question
from semantic_corpus.qasrl_core.question_renderer import render_question

for sentence in read_bank("data/qasrl-v2/orig/dev.jsonl.gz"):
    print(sentence.sentence_id, sentence.text)
    for entry, label in sentence.question_labels():
        forms = entry.verb_inflected_forms
        answers = [
            span.text(sentence.sentence_tokens)
            for span, votes in label.span_votes().items()
            if votes >= 2
        ]
        print(" ", label.question_string, "->", answers)

        # slots round-trip through the surface string and back
        assert render_question(label.question_slots, forms) == label.question_string
        assert render_question(parse_question(label.question_string, forms), forms) \
            == label.question_string
    break
```

```text
TQA:T_0058_1 Both occur suddenly .
  What occurs? -> ['Both']
  How does something occur? -> ['suddenly']
```

Building a question from scratch, without touching the bank:

```python
from semantic_corpus.qasrl_core.inflections import load_inflections
from semantic_corpus.qasrl_core.question_renderer import render_question
from semantic_corpus.qasrl_core.question_slots import make_slots
from semantic_corpus.qasrl_core.state_machine import TenseFeatures, build_verb_chain

lexicon = load_inflections("data/wiktionary/en_verb_inflections.txt")
give = lexicon.paradigm("give")

chain = build_verb_chain(TenseFeatures(tense="might", is_perfect=True, is_passive=True))
slots = make_slots("what", chain.verb, aux=chain.aux, prep="to", obj2="someone")
print(render_question(slots, give))
# What might have been given to someone?
```

## How the question template works

A question is seven slots, concatenated in a fixed order, with `_` meaning
"unfilled":

```text
wh      aux      subj        verb              obj         prep   obj2
what    does     something   stem              _           to     someone
```

Two details are easy to miss:

* the `verb` slot stores a **form name**, not a word — `being pastParticiple`
  becomes `being given` only once a paradigm is supplied;
* `prep` has three states. `_` means there is no prepositional position at
  all; `""` means the position exists but is silent, which is how a bare
  complement is written (`What does something help something do?`); anything
  else is a surface preposition, possibly multi-word (`to do`, `out of`).

The auxiliary chain itself is generated rather than tabulated. English stacks
verbal heads in a fixed order, each governing the next, and the leftmost one
is fronted by subject–auxiliary inversion:

```text
(modal | finite) > have (perfect) > be (progressive) > be (passive) > main
```

When the chain holds no auxiliary at all, either the questioned argument is
the subject — nothing to invert, so the finite verb stays put (`What
happens?`) — or `do`-support supplies something to front (`What does
something do?`).

## Verified against the released bank

Every claim below is re-checked by `python -m semantic_corpus.cli roundtrip`
on each split; the numbers are from the full QA-SRL Bank 2.0 shipped in
`data/qasrl-v2/`.

| Split | Questions | slots → string | string → slots → string | exact slot recovery |
|---|---:|---:|---:|---:|
| `orig/train` | 215 432 | 100% | 100% | 99.9986% |
| `orig/dev` | 38 487 | 100% | 100% | 99.9974% |
| `orig/test` | 45 389 | 100% | 100% | 100% |
| `expanded/train` | 293 629 | 100% | 100% | 99.9990% |
| `expanded/dev` | 52 370 | 100% | 100% | 99.9981% |
| `dense/dev` | 33 967 | 100% | 100% | 99.9941% |
| `dense/test` | 31 100 | 100% | 100% | 99.9968% |
| **total** | **710 374** | **100%** | **100%** | **99.9985%** |

The two round-trip columns are guarantees; the third is not, and cannot be.
`parse_question_all` returns every analysis the grammar licenses;
`parse_question` picks one by the same preference the original uses.

The 11 remaining mismatches all have the lemma **`do`**, where the bank's own
annotation treats the leading `does`/`did` as the predicate and the following
`do` as a preposition. Both readings render back to the same string; see
[SCALA_TO_PYTHON.md](SCALA_TO_PYTHON.md).

The grammar is also checked for coverage: every one of the 88 distinct
`(aux, verb)` pairs occurring in the bank is produced by `state_machine`, and
for all 710 374 questions the stored tense/aspect/voice features are
consistent with the stored slots. Five further invariants, recovered from the
original's completeness guard, hold with zero exceptions over the same
710 374 questions — among them "only *who*/*what* can question the subject"
and "a *who*/*what* question must leave a slot empty for its answer".

## Known defects found in the released data

Reported rather than silently tolerated:

* a few questions carry two judgments from the same annotator
  (3 in the first 2 000 sentences of `orig/dev`), which `validation` flags as
  `duplicate-judgment`;
* `en_verb_inflections.txt` has no entry for `be`, lists competing paradigms
  for some lemmas (`awaken`, `chide`, `plead`, `rewet`), and contains junk
  rows and misspelled lemmas (`found` is attributed to a non-word `foind`).
  `load_inflections` drops the junk and supplies `be` from
  `SUPPLETIVE_PARADIGMS`.

## Project status and what is missing

* [STATUS_RU.md](STATUS_RU.md) — project-wide status: what is left to build
  across all five stages, which data sits unused, and what blocks what.
* [NOT_PORTED.md](NOT_PORTED.md) — the technical list of rules and components
  deliberately not carried over from the Scala source.
* [SCALA_TO_PYTHON.md](SCALA_TO_PYTHON.md) — the stage-0 audit and the
  file-by-file mapping.

## Licence and attribution

MIT. The data formats, the question template and the auxiliary-chain grammar
originate in [julianmichael/qasrl](https://github.com/julianmichael/qasrl);
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
