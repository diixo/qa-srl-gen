# Not ported yet

Stage 1 covers the QA-SRL core needed to read the bank and move losslessly
between question slots and question strings. Stage 0 audited the upstream
Scala source and closed the gaps that could be closed from it; see
[SCALA_TO_PYTHON.md](SCALA_TO_PYTHON.md) for the full mapping. This file
records what is still deliberately missing.

## 1. State machine rules not covered

`semantic_corpus/qasrl_core/state_machine.py` implements the verb chain and
the slot vocabularies; `question_slots.answer_slot_problems` implements the
completeness guard. Together they generate 90 `(aux, verb)` chains against the
88 the bank uses, with nothing attested left out, and the answer-placement
rules hold with zero exceptions over all 710 374 questions.

What the original `TemplateStateMachine` additionally does, and this does not:

1. **Incremental, token-by-token transitions.** The Scala machine is a
   character-level automaton; `QuestionProcessor` walks it one character at a
   time, keeping every live branch, and can report the legal continuations of
   any prefix. The Python parser instead locates the inflected verb and
   enumerates the readings around it. Same answers on this data, but nothing
   here can answer "what may follow *What might have*?". **This is the one
   remaining structural gap**, and it is why `Autocomplete` cannot be ported.
2. **Sentence-conditioned preposition sets.** The inventory of 72 prepositions
   is ported, but upstream narrows it per question: the candidate set is the
   prepositions occurring in the sentence being annotated, plus adjacent
   preposition bigrams, plus seven common ones. The port accepts any token
   from the full inventory. This only matters for generation and for
   autocomplete, not for reading the bank.
3. **`Tense.NonFinite`** (`bare`, `to`, `gerund`) is not modelled. It does not
   occur in Bank 2.0 question labels; it is used for clausal representations.
   `Frame.questionsForSlot` and `Frame.clauses` are ported in simplified form
   (`Frame.questions`, `Frame.clause`): they render one string per slot rather
   than enumerating every animacy and placeholder variant as upstream does.
4. **Modals outside the QA-SRL inventory** (`may`, `must`, `could`, `shall`)
   are not modelled, matching upstream.
5. **`subj = "it"`.** The template accepts `it` as a subject placeholder on
   input, though `getSlotsForQuestionStructure` never emits it and it occurs
   nowhere in the bank. The parser here rejects it.

## 2. Components from the Scala project not ported

Deliberately out of scope per the handoff, confirmed by the audit:

* `Autocomplete.scala` — blocked on gap 1 above;
* `labeling/QuestionTemplate.scala` — tense- and adverbial-normalised question
  templates;
* `labeling/ClauseResolution.scala` — slots to clausal frames;
* `labeling/DiscreteLabel.scala` — questions to discrete role labels;
* `labeling/QuestionLabelMapper.scala` — an arrow abstraction with no useful
  Python counterpart;
* `data/Dataset.scala`'s filter/merge algebra (streaming reads are ported);
* `qasrl-crowd/`, `qasrl-crowd-example/` — Mechanical Turk pipeline;
* `qasrl-bank-service/` — HTTP document service;
* `apps/` — browser, reformatting and alignment CLIs.

## 3. Bank formats

* **Bank 2.1 / QANom.** `Sentence.from_json` preserves unmodelled top-level
  fields in `Sentence.extra`, so 2.1 files round-trip without data loss, but
  nominal predicate entries are not parsed into typed models. There is no
  2.1 file in `data/` to test against.
* **`index.json.gz`.** Document, domain and title metadata is not read; this
  needs `qasrl-bank`'s `DataIndex`, `Document`, `SentenceId` and `Domain`.
* The legacy `*.qa` reader returns raw slot strings. Its `verb` field holds a
  surface phrase (`be oxygenated`), not a form placeholder, so those questions
  cannot currently be converted into `QuestionSlots` — doing so needs a
  surface-to-form inverter driven by the predicate's paradigm.

## 4. Later stages of the handoff

Stage 5 is untouched: storage and exporters. Stages 0–4 are done. Storage
will be JSONL: the user has ruled out both SQLite and Parquet, overriding the
handoff.

## 5. Repository hygiene

* `test.py` at the repository root calls `datasets.load_dataset` from Hugging
  Face, which the handoff forbids outright. It is unrelated to this package
  and was left untouched; it needs removing or rewriting against a non-HF
  source.
* `qasrl_v2.py` at the root is the earlier flattening prototype. Its logic is
  now covered by `bank_reader` plus `QuestionLabel.span_votes`, but it is
  still the only thing that writes the flat export, so it was left in place.
