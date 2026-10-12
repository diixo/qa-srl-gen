# Shared mined-template bank

This directory is the single durable bank for all mined resources. It initially
contains the full compilation of `data/eng-base.jsonl`, moved byte-for-byte from
`data/eng-base-templates/`. Every successful UI mining job appends to these same
files; it does not create another bank under a job directory.

The original manifest and validation describe the 736,040 eng-base records.
On the first addition, `manifest.json` becomes `corpus-templates-pool-v1`: it
retains each original source manifest and validation report, plus the committed
byte ranges in the shared gzip files. Its `validation` field is the authoritative
aggregate report; the original `validation.json` remains the eng-base audit.
New UI source texts are preserved in `sources/<job-id>.jsonl`, and other source
files (metadata, licenses and documentation) in `sources/<job-id>.zip`.
The one-time, model-reviewed DailyDialog import omits those duplicate archives.
Its reviewed turns and exclusions are recorded in `sentence_review.jsonl.gz`;
editable review annotations and progress are in `../dailydialog/review-parts/`.

Rendering uses the patterns, bindings and grammatical data stored in the bank;
it does not require the original input or its archived snapshot. The source
files are still required for validation against external evidence. A missing
source therefore prevents that audit, but does not prevent generation.

Use `TemplateBank` / `BuiltInPool` to read the current bank. Gzip members are
appended without rewriting earlier data; one atomic manifest replacement makes
them visible. Readers keep their existing snapshot. Uncommitted tails after a
crash are ignored by these APIs, and retries never overwrite committed ranges.
Do not concatenate or regroup ledger lines manually: their line numbers are
local to the source identified by the manifest. Inventories and template shapes
may overlap across sources; source entries are deliberately retained intact.

For banks whose components retain their source JSONL, validate against the
preserved source records (fresh report):

```bash
python -m semantic_corpus.template_validation data/eng-base.jsonl --bank data/mined-templates --report artifacts/mined-bank-validation.json
```

The positional source selects the initial eng-base input; subsequent inputs are
read from `sources/` automatically. For a bank created entirely by UI mining,
the positional source can be omitted.
This command requires a source for every component; it cannot re-audit the
DailyDialog manual components from the eng-base source. The manual publisher
validates each new component against a temporary canonical sentence input
constructed from the reviewed annotations, and removes that input afterwards.

These files belong to the centralized **Built-in pool**, accessed through
`semantic_corpus.resource_pool.BuiltInPool` alongside typed entities and manual
semantic frames. **Synthetic corpus** and CLI `build` use the mined templates
by default. Keeping separate files for different resource structures does not
require a separate generation operation.

The eng-base component is an automatic analysis, marked **UNREVIEWED**. It preserves parser
decisions and dictionary alternatives separately, including ambiguous verb/
adjective readings and participles. Complete record coverage does not imply
perfect linguistic disambiguation or verified semantic-role annotations.
DailyDialog sentence templates are explicitly authored by the model; their
downstream semantic annotations likewise remain **UNREVIEWED**. The corpus is
being reviewed in parts, and only completed turn reviews are published.

| Path | Contents |
| --- | --- |
| `verbs.jsonl.gz` | All observed verb/AUX lemmas and forms, counts, morphology, examples, available paradigms |
| `adjectives.jsonl.gz` | Observed adjective lemmas and forms, counts, analysis provenance, morphology/comparison, dependency roles, examples |
| `verb_alternatives.jsonl.gz` | Other verb readings from Wiktionary and WordNet |
| `adjective_alternatives.jsonl.gz` | Other adjective readings from the dictionaries and possible participial readings |
| `constructions.jsonl.gz` | All observed particle/preposition/adverb constructions, with analysis categories |
| `templates/*.jsonl.gz` | Executable patterns, all distinct attested bindings, slot fillers, dependencies, adjective targets and predicate-marker locations |
| `occurrences/*.jsonl.gz` | Every original record, all tokens with offsets and lexical alternatives, source identifiers, template bindings and original syntax patterns |
| `sentence_review.jsonl.gz` | Reviewed DailyDialog turns, sentence boundaries, corrections, fragments and pending cases |
| `entities.jsonl.gz` | Explicitly reviewed typed named readings, loaded into the default entity pool |
| `entity_candidates.jsonl.gz` | Unresolved named readings; excluded from typed entity slots |
| `nominals.jsonl.gz` | Reviewed nominal surfaces and source evidence; no automatic entity typing |

Manual occurrence ledgers contain explicitly annotated lexical slots, which
may include noun phrases, rather than a claim to a complete word-token parse.

Inventories retain three representative examples per entry; **all** evidence is
in `occurrences/`. No frequency threshold or top-N cutoff is applied. Dictionary
homographs and possible participles are alternatives, not extra confirmed
adjective occurrences. Templates include nonverbal records and explicit
alternative analyses for bare commands missed by the parser.

Use **New job → Synthetic corpus → Built-in pool** for the combined corpus.
**Situations per frame** controls manual-frame examples; **Examples per mined template**
controls candidates per mined template, default 1. New UI jobs use the
entire bank; 0 explicitly disables mined candidates. The initial bank contains
595,350 templates; subsequent mining adds entries. Eligibility rules, split
filtering and deduplication determine which candidates enter the corpus.
`generation-summary.json` records the resulting counts and pending annotations.
Manual-frame annotations produce QA/SFT/BIO immediately. Supported mined clauses
produce semantic candidates and QA while retaining `UNREVIEWED`; unsupported
clauses can be processed through a separate **Corpus annotation** job.

For direct inspection, render every template or filter by lemma with the
low-level command (run from the repository root):

```bash
python -m semantic_corpus.cli template-generate --out artifacts/all-templates.jsonl.gz
python -m semantic_corpus.cli template-generate --adjective beautiful --out artifacts/beautiful.jsonl
python -m semantic_corpus.cli template-generate --verb "pick up" --out artifacts/pick-up.jsonl
```

`--count` is a quota per template, default 1. Generation uses attested combinations
unless `--recombine` is explicitly requested. Recombined text needs grammatical
and semantic review. Output files must be new. The runtime does not need spaCy
or a WordNet download. The old Corpus templates operation is kept only for
legacy jobs; use Synthetic corpus for new jobs.

Read the adjective inventory in Python:

```python
from semantic_corpus.resource_pool import BuiltInPool

for adjective in BuiltInPool().adjectives():
    print(adjective["lemma"], adjective["forms"], adjective["morphology"])
```

Recompile and independently validate in fresh directories (see `HOWTO.md` for
the model and optional lexical resource setup):

```bash
python -m semantic_corpus.template_compiler data/eng-base.jsonl --output data/eng-base-templates-rebuilt --work artifacts/template-work-new --model en_core_web_sm --wordnet artifacts/models/wordnet.zip
python -m semantic_corpus.template_validation data/eng-base.jsonl --bank data/eng-base-templates-rebuilt --rendered artifacts/all-verified.jsonl.gz --report artifacts/bank-validation.json
```

WordNet 3.0 terms are reproduced in `WORDNET-LICENSE.txt`; provenance and other
resource notices are in the repository's `THIRD_PARTY_NOTICES.md`. These terms
do not replace those of the source texts in `eng-base.jsonl`.

## Reviewed dialogue fragments

`reviewed/dialogue-fragments.json` supplies explicit context/reply/meaning
patterns to the dialogue generator's `fragment_meaning` task. It includes
source references for reviewed DailyDialog examples and authored contrast cases.
The catalog is self-contained and its SHA-256 is recorded in dataset manifests.
These fragment meanings are not counted as executable **whole-sentence** entries
in `templates/`; they generate contextual dialogue updates, including partial
events with explicitly missing roles. Archived `sentence_review` fragments are
not automatically promoted to semantic training targets.
