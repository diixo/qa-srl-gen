# Complete eng-base surface-template bank

This directory is a durable compilation of `data/eng-base.jsonl`, separate from
temporary runs under `artifacts/`. `manifest.json` records the exact source hash,
NLP model, counts and coverage against the earlier verb/template catalogue.
`validation.json` records the independent check against every source line and
every attested template binding. These files are usable after the manifest
reaches `complete`.

The bank is an automatic analysis, marked **UNREVIEWED**. It preserves parser
decisions and dictionary alternatives separately, including ambiguous verb/
adjective readings and participles. Complete record coverage does not imply
perfect linguistic disambiguation or verified semantic-role annotations.

| Path | Contents |
| --- | --- |
| `verbs.jsonl.gz` | All observed verb/AUX lemmas and forms, counts, morphology, examples, available paradigms |
| `adjectives.jsonl.gz` | All parser-recognized adjective lemmas and forms, counts, morphology/comparison, dependency roles, examples |
| `verb_alternatives.jsonl.gz` | Other verb readings from Wiktionary and WordNet |
| `adjective_alternatives.jsonl.gz` | Other adjective readings from the dictionaries and possible participial readings |
| `constructions.jsonl.gz` | All observed particle/preposition/adverb constructions, with analysis categories |
| `templates/*.jsonl.gz` | Executable patterns, all distinct attested bindings, slot fillers, dependencies, adjective targets and predicate-marker locations |
| `occurrences/*.jsonl.gz` | Every original record, all tokens with offsets and lexical alternatives, source identifiers, template bindings and original syntax patterns |

Inventories retain three representative examples per entry; **all** evidence is
in `occurrences/`. No frequency threshold or top-N cutoff is applied. Dictionary
homographs and possible participles are alternatives, not extra confirmed
adjective occurrences. Templates include nonverbal records and explicit
alternative analyses for bare commands missed by the parser.

Render every template, or filter by lemma (run from the repository root):

```bash
python -m semantic_corpus.cli template-generate --out artifacts/all-templates.jsonl.gz
python -m semantic_corpus.cli template-generate --adjective beautiful --out artifacts/beautiful.jsonl
python -m semantic_corpus.cli template-generate --verb "pick up" --out artifacts/pick-up.jsonl
```

`--count` is a quota per template, default 1. Generation uses attested combinations
unless `--recombine` is explicitly requested. Recombined text needs grammatical
and semantic review. Output files must be new. In Django use **New job → Corpus
templates: verbs and adjectives**; the resulting document store is an input to
**Corpus annotation**. The runtime does not need spaCy or a WordNet download.

Read the adjective inventory in Python:

```python
from semantic_corpus.template_bank import TemplateBank

for adjective in TemplateBank().adjectives():
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
