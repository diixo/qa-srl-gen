# Remaining exclusions and coverage limits

Updated 2026-10-03. The missing core components previously listed here have
been implemented: incremental grammar/processor, autocomplete, nonfinite Frame
variants, clause resolution, discrete labels, label mappers, Dataset filter/merge,
Bank metadata and QANom reformatting. See [SCALA_TO_PYTHON.md](SCALA_TO_PYTHON.md)
for the exact source/API map and compatibility differences.

## Excluded Scala applications

The handoff explicitly does not require a literal port of the whole repository.
These applications and their infrastructure remain outside this Python port:

- `qasrl-crowd/`, `qasrl-crowd-example/`: Mechanical Turk workflows and UI.
- `qasrl-bank-service/`: HTTP document service.
- Browser UI, alignment utilities and application-specific JVM/Scala.js CLIs.
  QANom's models, reformatter and partition reader are Python APIs; its Mill
  entry point and browser integration are not ported.
- Cats/Monocle typeclass/lens infrastructure, build tooling and deployment.

## Data and compatibility limits

- **QANom:** its upstream `verbEntries` / `nonPredicates` format, raw `isVerbal`
  conversion, inflection selection and worker-source rewriting are implemented.
  Tests use constructed fixtures; no full QANom release is available locally.
  This does not assign ontology types such as ACTION/STATE to nominal predicates.
- **Bank 2.1 / QA-SRL GS:** Bank-shaped records can be read, and index IDs are
  decoded. Unknown top-level fields (including arbitrary `nomEntries`) remain
  in `Sentence.extra`, not interpreted. Full release coverage has not been
  established without those corpora.
- **Legacy `.qa`:** typed conversion now exists when the supplied paradigm and
  seven-slot chain are compatible. Multiple analyses are exposed explicitly;
  unsupported grammar raises an error instead of silently normalizing.
  In a 100-sentence `wiki1.dev.qa` sample, 524/578 questions converted; the other
  54 use different auxiliaries/placeholders/slot conventions. Legacy answer text
  is not guessed into token offsets.
- **Modals:** `may`, `must`, `could`, `shall` remain outside the audited QA-SRL
  grammar, as in Scala. `it` and sentence-conditioned prepositions are supported
  by the new processor; the historical search parser retains its Bank behavior.
- **Ambiguity:** distinct valid readings are retained by the new processor.
  Deterministic fallbacks cannot guarantee recovery of a human's stored choice.
  The old search parser has 11 known exact-slot disagreements on Bank 2.0.

## Project-specific work beyond the port

Modules for all five handoff stages exist, but not every requirement is complete.
The limitations of the seven-frame generator, rule-based teacher, QA templates,
discontinuous semantic spans and JSONL storage are in [STATUS_RU.md](STATUS_RU.md).
SQLite, Parquet and Hugging Face remain excluded by user instruction.

`qasrl_v2.py` remains the older flat-export prototype. Wiktionary's old Python 2
extraction scripts are not used or modernized; supplied resources are read
by the Python core, with their morphological ambiguities retained.
