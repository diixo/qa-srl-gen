# Scala → Python: scope and component map

Source: [julianmichael/qasrl](https://github.com/julianmichael/qasrl/tree/16ab4949)
at **16ab4949** (2023-09-22), MIT, Copyright (c) 2017 Julian Michael.
Updated 2026-10-03. The upstream source was read without modification; its local
reference copy is under ignored `artifacts/upstream-qasrl/`. Attribution and
the MIT notice are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The current port covers the core grammar, frame/label operations, dataset
operations, Bank metadata and QANom reformatting. It is not a literal port of
the entire Scala repository: HTTP services, browser/MTurk applications and
Scala-specific Cats/Monocle machinery are excluded by the handoff.

## Core: `qasrl/src/qasrl/` → `semantic_corpus/qasrl_core/`

| Scala source / functionality | Python implementation |
|---|---|
| `data/Sentence.scala` | `models.Sentence`, JSON and strict `combine_with_like` convenience method |
| `data/VerbEntry.scala`, `data/QuestionLabel.scala` | `models.VerbEntry`, `QuestionLabel`, including `combine_with_like` |
| `data/AnswerLabel.scala`, `data/AnswerJudgment.scala` | `models.AnswerJudgment`; answer/invalid sum type uses `is_valid` + `spans` |
| `data/Dataset.scala` | `dataset.Dataset`: filters, mapping, culling and merge with recoverable diagnostics |
| `ArgumentSlot.scala`, `Argument.scala`, `ArgStructure.scala` | `frame.ArgumentSlot`, `Noun`, `Prep`, `Locative`, `ArgStructure`; immutable/hashable structures and JSON |
| `Tense.scala` | `tense.Finite`, `NonFinite`, `Tense`, including bare/to/gerund |
| `Frame.scala`: verb stacks, conjugation, splitting, modals | `state_machine.build_verb_chain` and `frame.Frame` |
| `Frame.scala`: question/clause variants, substitutions, argument markers | `Frame.questions_for_slot`, `questions_for_slot_with_args`, `clauses`, `clauses_with_args`, `gen_clauses_with_args`, `clauses_with_arg_markers` |
| `TemplateStateMachine.scala` | `template_state_machine.TemplateStateMachine`: transition graph, sentence-conditioned prepositions/bigrams, `it`, negation, complements |
| `QuestionProcessor.scala` | `question_processor.QuestionProcessor`: character processing, saved-state continuation, complete/in-progress states, longest valid prefix and completeness guards |
| `Autocomplete.scala` | `autocomplete.Autocomplete`: continuations, invalid-suffix location, unanswered questions from prior frames |
| `labeling/SlotBasedLabel.scala` | `models.QuestionSlots`, `question_renderer`, `slot_based_label`: surface/abstract mappers, preferred complete state, verb instantiation |
| `SlotBasedLabel.getSlotsForQuestionStructure` | `Frame.to_slots_upstream`; released Bank 2.0 convention stays in `Frame.to_slots` |
| `labeling/ClausalQuestion.scala` | `clausal_question.ClausalQuestion`, JSON, question and clause template |
| `labeling/ClauseResolution.scala` | `clause_resolution`: all readings, context pseudocounts, fallback preferences and resolved structures |
| `labeling/DiscreteLabel.scala` | `discrete_label`: noun/adverb roles, passive/dative/preposition handling, all/preferred label mappers |
| `labeling/QuestionLabelMapper.scala` | `question_label_mapper.QuestionLabelMapper`: optional/contextual lifting, composition, first/second, split/fanout |
| `labeling/QuestionTemplate.scala` | `question_template`: abstraction, active-voice and adverbial normalization, clausal conversion |

Scala lenses/typeclass instances use ordinary Python functions, dataclasses and
`dataclasses.replace`. Discrete labels are syntactic QA-SRL roles; they do not
infer semantic AGENT/THEME.

## Bank and nominal data

| Scala source / functionality | Python implementation |
|---|---|
| `qasrl-bank/.../ConsolidatedSentence.scala`, `ConsolidatedDataset.scala` | `dataset.ConsolidatedSentence`, `ConsolidatedDataset`, `nonPredicates` and merge |
| `Domain`, `DatasetPartition`, `DocumentId`, `SentenceId` | `bank_index`: typed IDs, wire codecs and ordering |
| `DocumentMetadata`, `Document`, `DataIndex` | `bank_index`, including `read_index` for `index.json(.gz)` |
| `QuestionSource`, `AnswerSource`, `AnnotationRound`, `package.scala` filtering | `bank_sources`, including `filter_expanded_to_orig` |
| `apps/qanom-reformat/.../QANomSentence.scala`, `Reprocess.scala`, `Data.scala` | `qanom`: sentence codec, raw reformatting, paradigm selection, source rewriting, non-predicate nouns, streaming JSONL/gzip and partition loading |
| Streaming bank reads | `bank_reader.read_bank`, `read_consolidated_bank`; legacy `.qa` has explicit surface-to-form conversion using its paradigm |

QANom's upstream schema stores nominal predicates in `verbEntries` and rejected
nouns in `nonPredicates`. This is not a general decoder for arbitrary Bank 2.1
`nomEntries`. Unknown top-level fields remain in `Sentence.extra`.
QANom tests use constructed fixtures; a real release is not present in `data/`,
so full-corpus verification is pending.

## Two parser APIs and the Bank 2.0 convention

`parse_question` / `parse_question_all` retain the search parser and released
Bank slot convention. The new `QuestionProcessor(TemplateStateMachine(...))`
returns typed frame readings, supports prefixes and follows the audited upstream
grammar. Labeling mappers use this processor. A successful prefix result does
not mean a complete question: use `processor.is_valid(text)` for that check.

The pinned upstream revision projects complements differently from Bank 2.0:

| Example | Bank 2.0 `Frame.to_slots` | `Frame.to_slots_upstream` |
|---|---|---|
| `... someone to do?` | `prep="to", obj2="do"` | `prep="_", obj2="to do"` |
| `... to do something?` | `prep="to do", obj2="something"` | `prep="_", obj2="to do something"` |
| bare complement | may use silent `prep=""` | empty preposition is `"_"` |

`get_verb_tense_abstracted_slots_for_question` returns `QuestionSlots` with
VerbForm keys; `get_slots_for_question` returns `SurfaceQuestionSlots` with
surface verbs. `frame_from_slots` and the default validator consume the released
Bank convention; use the new processor/resolver and
`check_sentence(sentence, dialect="upstream")` for upstream complement labels.

## Deliberate corrections and API differences

- A merge compares both passive flags. Scala's `isPassive != isPassive` typo
  silently missed mismatches.
- Conflicting sentence tokens, unknown fields or non-predicate values produce
  merge failures and retain the left record. Offsets from different tokenizations
  are never combined. Compatible records still merge; inspect `result.failures`.
- Ties that upstream resolved through `Set.head` have deterministic ordering.
  The template's fallback common preposition is explicitly `by`.
- The frame-resolution cache is bounded to 4096 labels.
- Malformed discrete labels and malformed optional index arrays raise errors
  instead of being partially parsed or silently dropped.
- Rendered slot codecs use a literal separator and require exactly seven fields.
  QANom positive judgments require nonempty spans, and repeated span sets are
  deduplicated independently of span order. Scala normalized empty positive
  answers into invalid judgments; the Python QANom reader rejects them explicitly.
- Python's existing `QuestionTemplate.normalize_to_active` correctly recognizes
  passive `by`; Scala compared an Option with a string and missed this case.
  Silent `prep=""` is normalized to absence in question templates.
- Collections use Python tuples/dicts. `Frame` and `ArgStructure` are hashable;
  dataset maps are ordinary in-memory mappings. Streaming readers remain
  available for corpora that should not be loaded as a `Dataset`.

## Verification and limits

- Final integrated suite: **599 passed** (`python -m pytest -q`), including
  197 new tests; compilation, public exports and the README example also pass.
- All **13,616** questions from upstream `qasrl/test/resources/question-strings.txt`
  parse and reproduce their original question through `Frame.questions_for_slot`.
  The exact upstream ambiguity histogram is **10,882 × 1**, **2,599 × 2**,
  **135 × 3** readings.
- The corpus test uses the pinned reference file under `artifacts/upstream-qasrl/`
  when available, and skips without a network request when it is absent.
- Finite/nonfinite rendering, all 48 nonfinite feature bundles, substitutions,
  incremental resume, autocomplete, context resolution, dataset conflicts,
  source formats and real Bank index data have targeted tests.
- Earlier full Bank 2.0 checks covered **710,374** questions: rendering and
  slots→frame→slots were lossless. The legacy search parser recovered exact
  stored slots for all but **11** questions (six distinct strings with lemma
  `do`), where predicate and do-support analyses collide. This is a historical
  result for the compatibility API, not a claim that the new automaton recovers
  the released annotation's chosen reading in every ambiguous case.
- This run compares Python against upstream source and its reference corpus;
  it does not execute Scala for differential testing.

Implementation fixes and DailyDialog results are recorded separately in
[STATUS_RU.md](STATUS_RU.md). Semantic annotation, synthetic generation, QA
generation and JSONL storage are project-specific Python components.
Remaining exclusions and data limits are in [NOT_PORTED.md](NOT_PORTED.md).
