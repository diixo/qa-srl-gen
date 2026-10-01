# Not ported yet

Stage 1 covers the QA-SRL core needed to read the bank and move losslessly
between question slots and question strings. This file records what is
deliberately still missing, so the gaps stay visible instead of being
rediscovered later.

## 1. State machine rules not covered

`semantic_corpus/qasrl_core/state_machine.py` implements the **verb chain**:
the mapping between `(tense, perfect, progressive, passive, negated,
subject-is-questioned)` and the `aux`/`verb` slots, in both directions. It is
complete and exhaustively checked for that job — all 88 `(aux, verb)` pairs
attested across 710 374 bank questions are generated, and no stored feature
bundle contradicts its slots.

What the original `TemplateStateMachine` additionally does, and this does not:

1. **Incremental, token-by-token transitions.** The Scala machine is a real
   automaton that advances one word at a time and exposes the set of legal
   next tokens at every point. This port computes whole chains and validates
   complete slot bundles. Nothing here can answer "what may follow *What might
   have*?" without enumerating candidates.
2. **Argument-structure licensing.** The grammar here constrains the verb
   chain and the slot vocabularies, but it does not know which *argument
   frames* a predicate allows. It will happily accept `obj2="somewhere"` on a
   verb that takes no locative, and it cannot rule out
   `prep="" obj2="do"` for a verb that takes no bare complement. This is the
   single largest gap, and the direct cause of the residual 0.35% of questions
   that parse to a different-but-equivalent slot assignment.
3. **The preposition inventory.** `prep` is treated as an open class. The
   original constrains it to an attested list, which would resolve readings
   such as `prep="out of" obj2="doing"` versus `prep="out of doing"`.
4. **Over-generation of marginal chains.** The grammar produces 114 chains
   while the bank uses 88. The extra 26 are the progressive-passive
   combinations under modals and perfects (`can be being pastParticiple`,
   `has been being pastParticiple`) plus a few modal perfect-progressives.
   They are grammatical English but vanishingly rare, and the original
   excludes most of them. They are harmless for parsing — they only add
   candidates that never match — but they should not be used as a generator
   without filtering.
5. **Auxiliary contraction coverage.** Negation contracts onto the fronted
   auxiliary for every auxiliary attested in the bank, and `might` is special
   cased because it has no usable contraction. Other modals outside the QA-SRL
   tense inventory (`may`, `must`, `could`, `shall`) are not modelled at all.
6. **`isPerfect` + `isProgressive` + `isPassive` all true.** Generated, never
   attested, never verified against real data.

## 2. Components from the Scala project not ported

Per the handoff, these are out of scope for now and were not started:

* `autocomplete` — the crowdsourcing-time question completion service, which
  depends on the incremental state machine (gap 1 above);
* the ScalaJS browser and the old crowdsourcing UI;
* the Mechanical Turk pipeline;
* `qasrl-bank-service` and the HTTP server;
* everything under `apps/`.

## 3. Bank formats

* **Bank 2.1 / QANom.** `Sentence.from_json` preserves unmodelled top-level
  fields in `Sentence.extra`, so 2.1 files round-trip without data loss, but
  nominal predicate entries are not parsed into typed models. There is no
  2.1 file in `data/` to test against.
* **`index.json.gz`.** Document, domain and title metadata is not read.
* The legacy `*.qa` reader returns raw slot strings. Its `verb` field holds a
  surface phrase (`be oxygenated`), not a form placeholder, so those questions
  cannot currently be converted into `QuestionSlots` — doing so needs a
  surface-to-form inverter driven by the predicate's paradigm.

## 4. Later stages of the handoff

Stages 2–5 are untouched: ontology, semantic generator, semantic annotator,
question generator, SQLite storage and exporters. The directory layout for
them is in the handoff, not yet in the repository.

## 5. Repository hygiene

* `test.py` at the repository root calls `datasets.load_dataset` from Hugging
  Face, which the handoff forbids outright. It is unrelated to this package
  and was left untouched; it needs removing or rewriting against a non-HF
  source.
* `qasrl_v2.py` at the root is the earlier flattening prototype. Its logic is
  now covered by `bank_reader` plus `QuestionLabel.span_votes`, but it is
  still the only thing that writes the flat export, so it was left in place.
