# Stage 0 — audit of `julianmichael/qasrl`

Source audited: <https://github.com/julianmichael/qasrl> at commit
`16ab4949` (2023-09-22), MIT, Copyright (c) 2017 Julian Michael.

The upstream tree was read, not modified, and is not vendored into this
repository. Nothing was copied verbatim; the Python modules are
reimplementations, and the correspondences below say which upstream rule each
one carries over.

## Scope of the repository

| Module | Files | Lines | Disposition |
|---|---:|---:|---|
| `qasrl/` | 21 | 3 184 | **the core** — partly ported, see table below |
| `qasrl-bank/` | 14 | 729 | document/index metadata — partly deferred |
| `qasrl-bank-service/` | 6 | 394 | HTTP document service — out of scope per handoff |
| `qasrl-crowd/` | 36 | 4 768 | Mechanical Turk crowdsourcing — out of scope |
| `qasrl-crowd-example/` | 6 | 878 | demo of the above — out of scope |
| `apps/` | 22 | 3 867 | browser, reformatting, alignment CLIs — out of scope |

Only `qasrl/` (and a fraction of `qasrl-bank/`) is relevant to this project:
2 879 of the 13 820 Scala lines.

## Core mapping: `qasrl/src/qasrl/ → semantic_corpus/qasrl_core/`

| Scala source | Python module | State |
|---|---|---|
| `data/Sentence.scala` | `models.Sentence` | **done** |
| `data/VerbEntry.scala` | `models.VerbEntry` | **done** (without `combineWithLike`) |
| `data/QuestionLabel.scala` | `models.QuestionLabel` | **done** (without `combineWithLike`) |
| `data/AnswerJudgment.scala`, `data/AnswerLabel.scala` | `models.AnswerJudgment` | **done** — the Scala `InvalidQuestion \| Answer(spans)` sum type is flattened into `is_valid` + `spans`, matching the JSON |
| `labeling/SlotBasedLabel.scala` | `models.QuestionSlots`, `question_slots`, `question_renderer` | **done** — `renderQuestionString` and the JSON codec are reproduced exactly |
| `Tense.scala` | `state_machine.TENSES`, `MODAL_TENSES` | **done** for `Tense.Finite`; `Tense.NonFinite` (bare/to/gerund) not modelled |
| `Frame.scala` → `getVerbStack`, `splitVerbStackIfNecessary`, `getVerbConjugation`, `modalTokens` | `state_machine.build_verb_chain` | **done** — verified to generate every chain the bank uses |
| `Frame.scala` → `questionsForSlot`, `clauses` | `frame.Frame.questions`, `Frame.clause` | **partial** — one rendering per slot, not the full variant enumeration |
| `SlotBasedLabel.getSlotsForQuestionStructure` | `frame.Frame.to_slots` | **done**, in the released bank's convention; the inverse `frame_from_slots` round-trips all 710 374 questions |
| `TemplateStateMachine.scala` → vocabularies, `lotsOfPrepositions`, `mostCommonPrepositions` | `state_machine.PREPOSITIONS`, `MOST_COMMON_PREPOSITIONS`, `WH_WORDS`, `NOUN_WH`, `ADVERBIAL_WH` | **done** |
| `TemplateStateMachine.scala` → the automaton itself | `question_parser` (search, not an automaton) | **partial** — see "What the port does differently" |
| `QuestionProcessor.scala` → `CompleteState` guard | `question_slots.answer_slot_problems` | **done** — the three completeness conditions are enforced on slots |
| `QuestionProcessor.scala` → character-level traversal | — | not ported |
| `labeling/SlotBasedLabel.getPreferredCompleteState` | `question_parser._Candidate.rank_key` | **done** |
| `ArgumentSlot.scala`, `Argument.scala`, `ArgStructure.scala` | `frame.ArgumentSlot`, `Argument`/`Noun`/`Prep`/`Locative`, `ArgStructure` | **done** |
| `Autocomplete.scala` | — | not ported; needs the incremental automaton |
| `labeling/QuestionTemplate.scala` | `qasrl_core.question_template` | **done** — abstraction plus `normalize_to_active` and `normalize_adverbials` |
| `labeling/ClauseResolution.scala` | — | not ported (slots → clausal frames) |
| `labeling/DiscreteLabel.scala` | — | not ported (questions → discrete role labels) |
| `labeling/QuestionLabelMapper.scala` | — | not ported (an arrow/plumbing abstraction with no Python counterpart) |
| `data/Dataset.scala` | `bank_reader` | **partial** — streaming reads and iteration; the filter/merge algebra is not ported |
| `qasrl-bank/DataIndex`, `Document*`, `SentenceId`, `Domain` | — | deferred; needed only to read `index.json.gz` |

## What the audit changed in the existing port

Stage 1 was built by re-derivation from the released data. Reading the source
confirmed most of it and corrected the rest. Exact slot recovery over the full
bank went from **99.65% to 99.9985%** — from 2 514 mismatching questions to 11.

**Confirmed, unchanged:**

* the rendering rule in `SlotBasedLabel.renderQuestionString`, including the
  dropping of empty tokens and capitalising only the first letter;
* the whole verb chain, including `modalTokens`'s special case for `might`
  (no contraction, so negation surfaces as a literal `not` in the verb slot);
* do-support exactly when the subject is not the questioned argument
  (`splitVerbStackIfNecessary` is applied only when `subj` is spelled out);
* the preference for the analysis that gaps the first object, which
  `getPreferredCompleteState` states in a comment and which had already been
  tuned to empirically on the data.

**Added, from `QuestionProcessor`'s completeness guard and the wh dispatch:**

1. `subj == "_"` ⟹ `wh ∈ {who, what}` — only a nominal wh-word can question
   the subject. The other six wh-words enter the template with
   `subjRequired = true`.
2. `wh ∈ {who, what}` ⟹ some slot is gapped. A who/what question that fills
   every slot has nowhere to put its answer.
3. A bare nominal `obj2` implies a first object, so `obj == "_"` is only
   legal there if `obj` is itself the gap — which needs a nominal wh-word and
   a subject that is not already the gap.
4. `obj2 ∈ {do, doing}` marks a *gapped* complement object, not a
   placeholder, so it too requires a nominal wh-word.
5. `prep` is a closed class after all: every token comes from
   `lotsOfPrepositions` (72 entries) or is `do`/`doing`. A question's own
   preposition set is drawn from the words of its sentence, which is why the
   slot looks open-class in the data. Bounding it stops the parser swallowing
   a placeholder into a multi-word preposition.

All five hold with **zero exceptions** across all 710 374 questions of
QA-SRL Bank 2.0.

## What the port does differently, on purpose

**Search instead of an automaton.** `TemplateStateMachine` is a character-level
automaton; `QuestionProcessor` walks it one character at a time, keeping every
live branch. The Python parser instead locates the inflected verb, splits the
question around it, enumerates the handful of readings the grammar licenses,
and keeps those that render back to the input. It gives the same answers on
this data and is simpler, but it cannot say what may legally follow a prefix —
so `Autocomplete` has no basis to be ported onto. That remains the main
structural gap.

**Slots remain the parser's entry point.** Upstream, slots are a *projection* of a typed
`Frame` + `answerSlot`; parsing produces the frame and the slots are derived.
Python now has a typed `qasrl_core.frame.Frame` and `frame_from_slots`;
slots → frame → slots is covered by the bank checks. The parser still returns
slots first. Full upstream variant enumeration remains a porting task.

## Implementation fixes versus further porting

The implementation audit and fixes recorded in [STATUS_RU.md](STATUS_RU.md)
are a separate workstream. The semantic ontology, synthetic generator,
annotation pipeline, question generator and JSONL store are project-specific
Python code, not translations of the Scala repository. Fixing their correctness
does not complete the missing Scala components listed above.

## Divergence between the released data and current `master`

The slots in QA-SRL Bank 2.0 were produced by an earlier version of
`getSlotsForQuestionStructure` than the one in the audited commit. Two
differences are visible:

* **Bare complements.** The release writes `prep=""` (empty string) where
  current `master` would write `prep="_"`. Upstream's own
  `fromRenderedString` already treats `""` and `"_"` alike on input, which
  suggests `""` was legacy even then. This port follows the **released data**,
  since that is what it reads.
* **`to do` placement.** The release writes `prep="to do", obj2="something"`;
  current `master`'s `getPrepAndMiscPrefix` would instead produce
  `prep="_", obj2="to do something"`, which is not an `obj2` value that occurs
  anywhere in the bank.

Further work on `QuestionTemplate` parity or `ClauseResolution` must account
for their newer upstream convention.

## Remaining mismatches

11 questions out of 710 374 (6 distinct strings) still parse differently from
the bank. Every one has the lemma **`do`**, where the bank's own annotation
treats the leading `does`/`did` as the predicate and the following `do` as a
preposition:

```text
"What did do something?"
  bank:   aux=_      verb=past(did)  obj=_          prep=do  obj2=something
  parser: aux=did    verb=stem(do)   obj=something  prep=_   obj2=_
```

Both readings render back to the same string. This is an inherent ambiguity
of a predicate whose forms collide with the do-support auxiliary, and is not
worth special-casing for 11 records.
