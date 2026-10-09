# qa-srl-gen — semantic corpus toolkit


#### Run django-server:
```bash
py ui/manage.py runserver
```


#### Create repository structure
```bash
django-admin startproject ui
```

```bash
cd ui
```

```bash
py manage.py startapp app_main
```

- DB migrations with migrates:
```bash
py ui/manage.py makemigrations
py ui/manage.py migrate
```


## Local web interface

The English-language Django interface manages the corpus pipeline from a sidebar: synthetic
generation, DailyDialog, text/dialogue and QA-SRL Bank import, annotation with
rules or an HTTP teacher, QA generation, SFT/BIO export, quality reports,
Bank validation, round-trip checks, candidate extraction and entity mining.
It also provides corpus/table browsing, entity-pool uploads, train/dev/test
coverage of all frame slots, and a persistent queue with 1–4 local process slots.

From the repository root:

```bash
python -m pip install -r ui/requirements.txt
python ui/manage.py migrate
python ui/manage.py runserver 127.0.0.1:8000
```

Open `http://127.0.0.1:8000/workspace`. Submitting a job starts the worker automatically.
The Workers page controls concurrency, pauses dispatch after current jobs finish,
and resumes the queue. Job pages show progress, logs, cancellation, retries,
quality findings and downloadable results. The worker stops after 60 idle seconds.
It can also be run explicitly with `python ui/manage.py pipeline_worker --once`.

**Pipeline → Chat dataset** (`/chat-dataset`) generates controlled English
dialogues for chat-model training. Set the training, validation and test quotas
**per task**, plus the seed. The page shows totals across the selected tasks before
submission. UI defaults use 15 tasks: 3,000 training, 300 validation and 300 test dialogues.
The job page shows progress, supports cancellation/retry and provides both
individual files and a complete `chat-dataset.zip` download. Results under
`artifacts/ui/jobs/<id>/dataset/` contain paired `history` and `memory` response
datasets, separate `extraction` examples, episode audit records and a manifest.
Generation runs locally without a language model; training is a separate step.
Uncheck **Include metadata** to export training records containing only `messages`
instead of also including `id`, `scenario_key` and `task`. The choice persists on retry.
See the [chat dataset guide](semantic_corpus/dialogue_generator/README.md) for
the export format and the existing chat-trainer's tokenizer contract.

**Data and resources → Template mining** (`/template-mining`) adds resources to
the shared Built-in pool. Paste English text (up to 500,000 characters), choose
a UTF-8 `.txt`, `.jsonl` or `.jsonl.gz` file from your computer using the file
picker, or use both inputs. The file is uploaded when you queue the job.
Each nonempty pasted/TXT line is one record; JSONL records provide `example` or
`text`. Uploaded files are saved under `CORPUS_UI_ROOT/uploads/` with unique
names and retained for background processing and retries. Both inputs are
combined and processed in full, without sampling.

The page queues a background job that compiles templates, verbs, adjectives and
multiword constructions, then independently validates the bank. The default
**Add validated results to Built-in pool** option publishes an immutable
component in `data/mined-templates/<job-id>/`, including the source snapshot and
occurrence ledger. Existing `data/eng-base-templates/` files stay intact; new
Synthetic corpus jobs use all published components through the same pool. Turn
the option off to keep the result only in the job's output directory. Resource
counts sum entries across source banks and may include overlapping patterns or
lemmas; generated identical texts are deduplicated by the corpus store.

Mining requires `requirements-mining.txt` and a local English spaCy model with a
parser and lemmatizer. Set `CORPUS_MINING_MODEL` to its package name or directory,
or use the detected installed `en_core_web_sm` / cached model under
`artifacts/models/spacy-en-*/en_core_web_sm/en_core_web_sm-*`. The UI does not
download models automatically. Validation checks structural completeness;
mined syntax remains `UNREVIEWED`. Synthetic corpus derives semantic candidates
and extractive QA from supported clauses without another NLP or HTTP call.

Each job writes to a fresh `artifacts/ui/jobs/<id>/` directory. Annotation works
on a copy and replaces QA for reannotated documents, preserving other documents'
QA and historical annotation runs. QA regeneration copies annotations and rebuilds QA so the selected
balance settings apply. Failed/cancelled output can be inspected but cannot be
selected as another job's input. There is no automatic retry after interruption.
Existing CLI stores under `data/` and `artifacts/` are visible in the corpus
browser; external CLI writes should be finished before using them as UI inputs.

Corpora remain JSONL. Django uses `ui/db.sqlite3` for the persistent queue,
worker settings and Django's own tables. Results, uploads and pool versions
are stored under `artifacts/ui/`. `CORPUS_UI_ROOT` overrides that artifact
directory; it does not change the database path. Use the same value for the
server and worker. Removing `artifacts/` while the server and worker are stopped
removes output files and uploads but preserves job history and database tables.
Published template components under `data/mined-templates/` also survive;
unpublished mining results and models cached under `artifacts/` do not.
If `ui/db.sqlite3` itself is removed, run `python ui/manage.py migrate` to create
empty tables again. The UI is a local operator tool, with CSRF checks
and no user-account access control; public/network deployment is not configured.
Uploaded source files and pool versions are limited to 20 MB; place larger
source files under `data/`. Pool coverage counts individual slot candidates,
not all possible joint assignments. HTTP annotation sends the selected text to
the configured teacher; offline rules cover dialogue acts only.

Operation codes, labels and their display order live in
[`ui/app_main/operations.json`](ui/app_main/operations.json).
`OperationRegistry` loads and validates this catalogue independently of Django
model fields. `Job.operation` stores a plain string; catalogue edits no longer
generate migrations. The form and executor both reject unknown operations and
operations without a registered Python handler. Historical jobs still display
their code if the operation has been removed from the catalogue.

To add an operation, add its `code`/`label` entry to the JSON array, implement
and register a handler in `operations.HANDLERS`, and add any required form
validation and field/hint configuration in `static/app_main/js/pipeline.js`.
Codes use lowercase letters, digits and underscores, start with a letter and
contain at most 24 characters. Restart the server and worker after catalogue
edits: each process caches its validated registry.

The one-time `0003_operation_registry` migration removes the old field choices
from Django's migration state; its table-schema SQL is a no-op. Apply it through
the usual `python ui/manage.py migrate`. Future operation-catalogue changes
need no migration.

UI validation is separate from the dependency-free core tests:

```bash
python ui/manage.py check
python ui/manage.py test app_main
```

## Core

Python QA-SRL core plus a generator of semantic training corpora. The repository
has modules for the five stages of the handoff, with the limitations recorded in
[STATUS_RU.md](STATUS_RU.md) and [NOT_PORTED.md](NOT_PORTED.md): the QA-SRL
core (`semantic_corpus/qasrl_core`), the canonical document representation
(`semantic_corpus/documents.py`), the ontology (`semantic_corpus/ontology`),
the semantic generator (`semantic_corpus/semantic_generator`), the semantic
annotator (`semantic_corpus/semantic_annotator`), the question generator
(`semantic_corpus/question_generator`), the store
(`semantic_corpus/storage`) and the exporters
(`semantic_corpus/exporters`). The full plan lives in
[HANDOFF_SEMANTIC_CORPUS_RU.md](HANDOFF_SEMANTIC_CORPUS_RU.md).

The core package sits at the repository root and imports directly, without a
build system. Python 3.10+, standard library only; the optional web UI needs Django.

```bash
python -m pytest -q
python -m semantic_corpus.cli --help
```

The upstream grammar conformance test uses the bundled reference corpus in
[`tests/fixtures/upstream-qasrl/`](tests/fixtures/upstream-qasrl/README.md).
It runs without `artifacts/`, network access or a Scala runtime.

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
| `qasrl_core/frame.py` | the typed argument structure the slots are a projection of |
| `qasrl_core/tense.py`, `clausal_question.py` | finite/nonfinite tenses, frame variants, clausal questions and JSON |
| `qasrl_core/template_state_machine.py`, `question_processor.py`, `autocomplete.py` | incremental grammar, prefix continuation and question suggestions |
| `qasrl_core/slot_based_label.py`, `question_label_mapper.py` | contextual/composable mappings to abstract and surface slots |
| `qasrl_core/clause_resolution.py`, `discrete_label.py` | ambiguity resolution and syntactic labels |
| `qasrl_core/dataset.py`, `bank_index.py`, `bank_sources.py` | dataset filters/merges, consolidated annotations, metadata and provenance |
| `qasrl_core/qanom.py` | typed QANom JSONL/gzip, raw annotation reformatting and partition loading |
| `qasrl_core/validation.py` | invariant checks over slots, spans and sentences |
| `ontology/models.py` | the label inventory: multi-label spans, speech acts, relations |
| `ontology/hierarchy.py` | type inheritance (`city → LOCATION`) and the candidate lexicon |
| `semantic_generator/frames.py` | 17 typed frames, including six multiword predicates and reviewed go/have/be senses |
| `resource_pool.py` | centralized Built-in pool: typed entities, manual frames and filtered mined templates/inventories |
| `syntax_mining.py` | optional full-corpus verb, construction and template inventory with source evidence |
| `template_compiler.py`, `template_bank.py` | full-corpus executable surface patterns, verbs, adjectives and lexical alternatives; model-free bank rendering |
| `template_validation.py` | independent source/ledger/inventory validation and rendering of every compiled template |
| `semantic_generator/substitutions.py` | entity pools, real and invented names, held-out splits |
| `semantic_generator/realization.py` | situation → sentence, with exact spans and relations |
| `semantic_generator/transforms.py` | tense, voice, negation and modality variation |
| `semantic_generator/generator.py` | seeded sampling, ambiguity pairs, omitted arguments |
| `documents.py` | `Document`, `Passage`, `Utterance`, spans, `AnnotationRun` |
| `qasrl_bridge.py` | QA-SRL Bank → canonical documents, runs and QA examples |
| `qasrl_core/question_template.py` | the abstract shape of a question, for folding and counting |
| `semantic_generator/entities.json` | the entity pool, as data rather than code |
| `semantic_generator/mining.py` | typed entity proposals from a plain-text corpus |
| `semantic_annotator/ingestion.py` | TXT/JSONL/dialogue input, dedup, document-level splits, passages |
| `semantic_annotator/candidates.py` | proposals from morphology, POS counts, the ontology lexicon |
| `semantic_annotator/lexicons.py` | closed lists: stative verbs, discourse markers, contractions |
| `semantic_annotator/rule_teacher.py` | offline teacher for the dialogue layer — no model, no tokens |
| `semantic_annotator/teacher.py` | the provider-agnostic adapter protocol |
| `semantic_annotator/alignment.py` | quoted spans → verified offsets, with rejections |
| `semantic_annotator/verifier.py` | structural checks and an independent second pass |
| `semantic_generator/canonical.py` | generated sentence → `Document` + `AnnotationRun` |
| `question_generator/templates.py` | questions per relation, label and type |
| `question_generator/negatives.py` | genuinely unanswerable questions |
| `question_generator/paraphrases.py` | conservative, deterministic rewordings |
| `question_generator/answers.py` | the `QAExample` record and answer wording |
| `storage/schema.py` | record codecs and the store's invariants |
| `storage/repository.py` | the append-only JSONL store |
| `exporters/sft.py` | decoder-only training records, split at the loss boundary |
| `exporters/bio.py` | the lossy BIO/BILOU projection, with its cost counted |
| `exporters/report.py` | class distribution and leakage |
| `cli.py` | `inspect`, `roundtrip`, `validate`, `lookup`, `frames`, `generate`, `ambiguity`, `ingest`, `candidates`, `annotate`, `questions`, `build`, `report` |

Since generator version `0.4.0`, the UI's **Situations per frame** field and
`--count` in `generate`, `questions` and `build` specify a quota for **each frame**.
For seventeen frames, `100` requests 1700 distinct situation texts, before QA generation.
The UI shows the total before submission and in job progress. Small pools may
exhaust the bounded sampler: reduce the quota or expand the selected split's pool.
Failed UI jobs may contain partial data and do not publish a completed version.

Generator version `0.5.0` adds the `GROUP` entity label with `human_group`,
`animal_group` and `physical_object_group` subtypes. Slot constraints distinguish
groups of people from groups of objects. Pool entries may specify
`"number": "plural"` (for example, `the police`); omitted number means singular.
Agreement is preserved in generated sentences, stored mentions and questions.
Older stores remain readable; rebuild generated datasets to use the new pool.

Version `0.6.0` adds `pick up`, `put down`, `give up`, `get up`, `look after`
and `put up with`. Frames distinguish particles from governed prepositions;
only the head verb inflects. Separable particles support both noun-object orders
and obligatory pronoun separation (`pick it up`). Canonical predicates retain
each lexical part's exact span, so `picked the book up` never labels the book
as part of the action. JSONL preserves these links; BIO/BILOU reports their loss.
The six frames cover explicit senses and argument structures, not every sense
of these expressions. The Slots page shows their actual word-order templates.

**Built-in pool** is the shared resource catalogue for **Synthetic corpus** and
CLI `build`. `BuiltInPool` in `semantic_corpus/resource_pool.py` provides access
to typed entities, the seventeen manual frames, verb forms, adjectives,
constructions, lexical alternatives and eligible mined templates. The existing files
stay in `semantic_corpus/semantic_generator/entities.json`,
`data/eng-base-templates/` and published components under `data/mined-templates/`;
their different structures do not require separate generation operations or
flattening everything into entity entries.

The source-derived bank covers every record of `data/eng-base.jsonl`, including
adjective slots and lexical alternatives. All observed bindings and source
occurrences are retained without a top-N cutoff. New Synthetic corpus jobs and
CLI `build` use the bank by default alongside the manual frames. `--count` remains
the quota per manual frame; `--template-count` is a separate candidate quota per
selected mined template, default 1, with 0 explicitly disabling mined templates.
CLI `--template-verb` and `--template-adjective` can narrow the mined selection.
New UI jobs use the eligible working bank. A custom Entity pool replaces typed entities only.

The archive retains **595,350 templates** from eng-base. The working policy excludes
`can`, `must`, `may`, `be`, `let`, `have`, `go` as mined lexical/copular predicates,
including recognized forms and multiword heads. Auxiliary uses within other
predicates, such as `has bought` and `is moving`, remain eligible. This leaves
**371,665 eligible templates** and **4,031 verb inventory entries** in the current
eng-base component (4,047 archived entries). A template with any excluded lexical
head is skipped as a whole. No source archive or manifest is rewritten.
Raw `TemplateBank` / `template-generate` remain available for audit/reprocessing;
normal Synthetic corpus uses `BuiltInPool`. UI inventory totals are archival.

The default mined quota traverses eligible templates from eng-base plus
eligible templates from every published component in new UI jobs before split filtering
and deduplication. It does not promise that every
candidate belongs to the selected split or has a semantic annotation.
`generation-summary.json` reports candidates, split skips, duplicates, stored
documents and documents awaiting annotation. Mined texts receive a deterministic
80/10/10 split from their hash; a whole, case-sensitive held-out test-entity phrase
forces test membership. With the same entity pool and split policy, a mined text
keeps its split across seeds and jobs. This does not guarantee disjoint templates
or vocabulary.

Manual-frame documents have verified annotations and produce QA/SFT/BIO immediately.
Mined documents enter the **same store**, retaining their syntax and provenance.
`template_semantics.py` converts supported saved dependencies into an
`AnnotationRun` and extractive QA during Synthetic corpus / CLI `build`.
It supports finite active clauses, passive clauses with or without an agent,
direct and supported prepositional recipients, particles (including separated
ones), the curated governed-preposition patterns, auxiliaries and simple
attributive properties. It works with the existing eng-base bank without
re-mining it or running spaCy. Support is structural, not restricted to the
seventeen manual frame lemmas.

These automatic annotations and their QA retain `UNREVIEWED` status and
`annotation_source=template_semantics`; they are candidates for review, not gold.
Unknown entity types and predicate senses stay unspecified. Clauses with missing
dependency information, unresolved inflections, complex complements, conditionals,
unsupported modality or negative quantifier scope remain pending. Partial
documents retain supported QA and report the omitted clauses. This does not
claim full semantic coverage of every source sentence or validate the meaning
of recombined fillers. `generation-summary.json` and the job page show annotated
documents, QA counts, pending/partial documents and skip reasons. Documents
without annotations are not exported as all-`O` BIO examples.

Mined candidate QA are included in SFT. Automatic mined runs are excluded from
BIO/BILOU export and counted in `skipped_mined_candidates`: unknown types and
unanalysed spans must not become false `O` targets. For BIO, run Corpus annotation
with **Only documents awaiting annotation** disabled to also annotate the
documents that already have complete candidate QA. Manual-frame BIO is unchanged.

New **Corpus annotation** jobs default to **Only documents awaiting annotation**.
This preserves the existing curated runs and applies the document limit after
selecting documents with no annotation run or an incomplete automatic mined run.
Disable it for intentional
reannotation; saved historical jobs keep their previous behavior.

Build identity includes the inflection dictionary checksum used by mined
semantics, so changing that dictionary requires a new dataset version.

For direct template inspection, the low-level `template-generate` command remains
available without spaCy:

```bash
python -m semantic_corpus.cli template-generate --out artifacts/templates-all.jsonl.gz
python -m semantic_corpus.cli template-generate --adjective beautiful --out artifacts/beautiful.jsonl
python -m semantic_corpus.cli template-generate --verb "pick up" --out artifacts/pick-up.jsonl
```

For this low-level command, `--count` is a distinct-example quota **per selected
template**, default 1. Default generation uses attested combinations;
`--recombine` explicitly enables new combinations of observed fillers that
require grammatical and semantic review. In the UI, use **Synthetic corpus**;
the separate Corpus templates operation is retained only for legacy jobs.
The bank survives removal of `artifacts/`; `manifest.json` and `validation.json`
record coverage and independent verification. All adjective forms, comparison
morphology, source evidence and possible participial readings are retained.

To inventory a source corpus, install the optional `requirements-mining.txt`
and an English spaCy parser model, then run:

```bash
python -m semantic_corpus.syntax_mining data/eng-base.jsonl --output artifacts/eng-base-syntax --model en_core_web_sm
```

This processes every record, retaining verb forms, syntactic templates, multiword
construction candidates, counts and source examples. A compressed occurrence
ledger and input SHA-256 make observations traceable. Outputs are unreviewed
parser observations, not automatically approved semantic frames. Details and
file descriptions are in [HOWTO.md](HOWTO.md#извлечение-шаблонов-из-eng-basejsonl).

Each lemma has an independent seeded stream and document numbering. Adding or
reordering frames preserves existing frames' texts, IDs and QA when definitions,
pool and generation/QA settings stay unchanged. Increasing the quota extends
each sequence. Use `Generator.generate_per_frame(count)` for this behavior;
`Generator.generate(count)` retains the legacy total-count random-frame stream.
Saved UI jobs without `count_scope` and their retries retain the old total limit.
Use a fresh store for builds with the new generator version.

Version `0.3.0` added literal purchases of physical goods (`buy`):
a person or organization buys an object, optionally at a location and time.
Active, passive and agentless passive patterns are supported. Sellers and prices
are not modeled by this frame. The added frame changed legacy seeded sampling.

Version `0.2.0` fixed annotation integrity, contextual answerability,
verb-chain questions and duplicate builds. Existing `jsonl-v1` records remain
readable; old exports must be regenerated into a fresh store to obtain corrected
training data. New records preserve predicate extra labels, omitted slots and
whether a synthetic entity type is stated in the text. Unknown bank predicate
types, polarity and semantic roles are stored as `null`; native questions remain.

`build` can be repeated with the same configuration and version. Changed input
under an existing document ID is rejected; use a new prefix and version for a
different build. Storage supports one writer and does not provide crash-atomic
transactions across its JSONL files. Text and annotation reads stream; in-memory
identity and offset indexes grow with record count.

The writer keeps bounded caches for the last 128 stored document fingerprints
and run memberships, avoiding immediate disk rereads during annotation/QA
storage. Reopened stores fall back to indexed file reads. Document fingerprints
include metadata, passages and utterances; ownership and immutability checks
remain in place. Profiling results and limits are in [STATUS_RU.md](STATUS_RU.md).

## Short example

The upstream core is adapted from Scala commit `16ab4949`; the source/API map,
deliberate differences and verification limits are in
[SCALA_TO_PYTHON.md](SCALA_TO_PYTHON.md). Application services and browser/MTurk
infrastructure remain excluded. The search parser below retains the Bank 2.0
slot convention. Incremental parsing and upstream labeling use a separate API:

```python
from semantic_corpus.qasrl_core import (
    InflectedForms, TemplateStateMachine, QuestionProcessor, Autocomplete,
    get_discrete_labels,
)

forms = InflectedForms("give", "gives", "giving", "gave", "given")
processor = QuestionProcessor(TemplateStateMachine(["Pat", "gave", "a", "book"], forms))
prefix = processor.process_string("What did someone")
completed = processor.advance(prefix, " give?")
assert completed.valid_states[0].is_complete
suggestions = Autocomplete(processor)("What did someone")
assert str(get_discrete_labels([], forms, ["What did someone give?"])[0]) == "obj/-"
```

For QANom, `read_qanom(path)` reads reformatted records and
`reprocess_qanom(path, lexicon)` streams raw records through the upstream
conversion. Validate their complement slots with
`check_sentence(sentence, dialect="upstream")` from `qasrl_core.validation`.
`Dataset.merge(other)` returns both `dataset` and `failures`; inspect failures
before using the partially merged dataset. No new runtime dependencies are needed.

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

## Generating situations

For a separate controlled dialogue-training experiment inspired by bAbI:

```bash
python -m semantic_corpus.cli dialogue-build --out artifacts/dialogue-pilot --train 200 --validation 20 --test 20 --seed 42
```

This creates 16 task families with paired history/memory response datasets and
a separate memory-update dataset, all in `messages` format. Quotas are per task;
the CLI defaults give 3,200 training, 320 validation and 320 test episodes. Names and
places come from the existing entity pool. No model or network is needed.
For conversations about the user alone, add `--user-assistant-only`: no personal
names or named places, 15 tasks, and 3,000 / 300 / 300 episodes by default.
In **Pipeline → Chat dataset**, **User and assistant only** is checked by default;
uncheck it to include named people. **Include metadata** separately controls the
top-level IDs and task labels in training records.
Start response training with `memory/`: its message content now uses short
English facts and ordered assistant options, with natural-language targets.
`history/` also uses text. New `extraction/` exports use text context and
`SET` / `SET_CONTEXT` / `REMOVE` / `EVENT` / `EVENT_CONTEXT` / `REMOVE_EVENT` /
`NO_CHANGE` commands inside messages;
IDs and exact source quotes are handled by code. The outer files remain JSON
arrays for training. Generate a new dataset to obtain this extraction format.
Typed events reuse the shared frames and entity pool. They distinguish participants,
time, operator scope and completion: wanting or attempting to pick something up
does not establish that it happened. Four task families cover scope,
progress, event corrections and possession/location states. Reviewed grammar
distinguishes modals, auxiliary be/have, permissive let and prospective going to;
the original flat-fact tasks remain as controls.
See [dialogue generation](semantic_corpus/dialogue_generator/README.md) for the
scenario schema, split limits, files and the existing chat-trainer adapter.

For a local model experiment, the optional [GPT-2 SFT trainer](semantic_corpus/dialogue_generator/trainer/README.md)
uses pretrained `gpt2`, ChatML and `transformers.Trainer` on plain-text Memory data:

```powershell
python -m pip install -r requirements-training.txt
python -m semantic_corpus.dialogue_generator.trainer train --mode sft --model gpt2 --data artifacts/dialogue-gpt2-memory/memory --out artifacts/training/gpt2-memory
```

Generate the dataset first as described in the trainer guide, or supply a new
UI job's `dataset/memory/` path. Training is CLI-only; no training operation is
added to the UI.

To train the updater, select the new dataset's `extraction/` directory and add
`--view extraction`. Use `--view history` for the history response experiment;
`--view memory` remains the default. Keep each task in its own training run.

New Chat dataset exports include matched yes/no/unknown examples for wishes,
plans and suggestions, covering visits, purchases, calls, messages, reading and
breaks, with varied Memory wording. CLI `dialogue-build --contrast-multiplier 3`
increases the two completion-task quotas. The [dialogue guide](semantic_corpus/dialogue_generator/README.md#contrasts-and-varied-memory)
also describes a frozen test with unseen wording and the trainer's comparison
reports. Existing files are not rewritten; generate a new dataset to use these changes.

```python
from semantic_corpus.semantic_generator import Generator

generator = Generator(seed=42)
for realized in generator.generate(3):
    print(realized.text)
    for mention in realized.mentions:
        print("   ", mention.slot, mention.text, mention.labels)
```

```text
Sorel saw Kyiv.
    experiencer Sorel PERSON + NAMED_ENTITY
    stimulus Kyiv LOCATION + NAMED_ENTITY
```

The output is a semantic record, not a string: every argument carries the
exact character span it occupies and the relation it bears to the predicate,
so later stages never have to recover offsets by searching for substrings.

Frames are typed, so a binding that does not fit is refused rather than
realised — `give` takes an abstract theme (*gave an idea*) but not an
emotional one, and an animal cannot be its agent. English morphology is not
reimplemented: the verb chain comes from the same builder that reproduces
every verb form in the bank, so `might not give`, `is being given` and
`has given` are correct by construction.

Two generators exist for the harder test sets:

```bash
python -m semantic_corpus.cli ambiguity --seed 2
```

```text
Rex: ANIMAL vs PERSON
  Leila gives Rex, a German shepherd, a book in Vustal.
  Rex, the new mechanic, gives Luna a red ball.
  What kind of entity is Rex?
```

`Generator.omitted_argument_examples` produces sentences that leave a known
argument unexpressed, which is the honest basis for a *the text does not say*
example: the record knows there was an agent, the sentence does not name it.

## Annotating real text

```python
from semantic_corpus.semantic_annotator import (
    CandidateResources, HttpTeacher, annotate_document, ingest,
)

teacher = HttpTeacher("http://localhost:8000/annotate", name="local-7b")
resources = CandidateResources.load()

for document in ingest(["corpus.jsonl"]):
    outcome = annotate_document(document, teacher, run_id="run-1", resources=resources)
    print(outcome.summary())
    for rejection in outcome.rejected:
        print("  dropped:", rejection)
```

The teacher returns *text*, never offsets — asking a language model to count
characters invites silent corruption. `alignment` locates each quote in the
source and **drops anything it cannot find**, with a reason:

```text
d1: 8 annotations from 8 candidates, 3 rejected
  dropped: 'Berlin': not found in the passage (occurrence 0)
  dropped: 'Anna': a span has at most one entity type, got PERSON, ANIMAL
  dropped: 'Monday': labels not in the ontology: ['WEEKDAY']
```

### Annotating without a model

To run the local DailyDialog train corpus through annotation, storage, SFT,
BIO and validation, choose a new output directory:

```bash
python scripts/run_dailydialog.py --out artifacts/dailydialog-train
```

Use `--limit 50` for a smoke run. The script keeps the source train split,
deduplicates before annotation, and records stage timings, input and code
hashes in `summary.json`. Outputs stay under the ignored `artifacts/`
directory. See [STATUS_RU.md](STATUS_RU.md) for the latest full-run results.

`RuleBasedTeacher` decides the part of the task that surface form really
settles — mood from punctuation, polarity from negation, speech acts from
interrogatives, imperatives and a closed cue list, discourse markers from a
closed inventory in the positions where a word cannot be a modifier. It
refuses the rest: entity types and predicate senses need context a rule
cannot read, and rubber-stamping the candidate layer would record guesses as
fact. `RuleBasedTeacher.covers` says so, so an empty mention list reads as
"not attempted" rather than "none found".

The full `dailydialog-train.jsonl` run on 2026-10-03 reads 11 118 dialogues,
keeps 10 402 after deduplication, annotates 81 869 turns and stores
**266 379 speech-act QA examples**, using no external model:

```text
<context>
I finally got the job.
Wow, that's great!
</context>
<question>
What does "Wow, that's great" express?
</question>
<answer>
Surprise, reaction and approval.
</answer>
```

Candidates are proposals with evidence, never decisions. The dictionary says
`ball` can be a verb, so `ball` is proposed as both `ACTION` and
`PHYSICAL_OBJECT`, with the POS counts that make the noun reading likelier
attached:

```bash
python -m semantic_corpus.cli candidates --text "The jaguar ran up to the gate."
```

```text
  jaguar   ANIMAL                               ontology
  ran      ACTION                               inflections
  gate     ACTION + LOCATION                    inflections,postags,ontology
```

Annotation is append-only. Verifying does not edit a run: it produces a
second one, formed without seeing the first, and `promote_agreed` marks what
both passes found as `VERIFIED`. What only one pass found keeps its status —
a single disagreement is evidence, not a verdict.

## Generating questions

Both inputs converge on one shape first — a `Document` plus an
`AnnotationRun` — so the question generator never needs to know whether the
facts were invented or annotated:

```python
from semantic_corpus.question_generator import QuestionGenerator
from semantic_corpus.semantic_generator import Generator
from semantic_corpus.semantic_generator.canonical import to_canonical
from semantic_corpus.semantic_generator.generator import DEFAULT_PARADIGMS

generator = Generator(seed=11)
questions = QuestionGenerator(paradigms=DEFAULT_PARADIGMS, seed=1)

for index, realized in enumerate(generator.generate(5)):
    document, run = to_canonical(realized, document_id=f"gen-{index}")
    for example in questions.for_document(document, run):
        print(example.question, "->", example.answer)
```

```text
Who gave a red ball to Rex?        -> Anna.
What did Anna give to Rex?         -> A red ball.
Where did Anna give a red ball?    -> In Kyiv.
What did Anna do?                  -> Anna gave a red ball to Rex.
What kind of entity is Rex?        -> Animal.
Is Kyiv a location?                -> Yes. Kyiv is a city, and every city is a location.
```

The generator invents no ground truth: every answer is a span something else
already recorded. Verb forms come from the same chain builder as the bank, so
*Who gave...?* keeps its finite verb while *What did Anna give?* gets
do-support, and an agentless passive is asked about in the active voice —
the record says `given`, the question says *gave*.

**Negatives are earned, not assumed.** Absence from a record is not absence
from the text: *On Monday, Anna gave Rex a ball* has a time whether or not
anything annotated it. So inferences that read meaning into silence apply
only to synthetic runs, where the record is complete by construction. For
annotated runs they return nothing.

**Polarity is respected.** A negated clause is asked about in the
affirmative and answered *No.*, never *Yes.*, and the justifying clause is
rebuilt from the verb chain — the predicate's own token is only the bare
stem once do-support has split it.

Paraphrases come in two kinds. Regular expressions handle shapes they fully
recognise; voice alternation is done through the frame, because turning
*What did Anna give to Rex?* into *What was given to Rex by Anna?* means
rebuilding the verb chain rather than rewriting words.

`balance()` imposes a target mix — a cap per kind and a ceiling on the share
of refusals — deterministically, so two runs of the same corpus can be
compared.

`python -m semantic_corpus.cli questions --count 3 --prompts` prints the
decoder-only layout, with the loss computed over `<answer>` alone.

## QA-SRL Bank as annotated real text

The bank is real text that is already annotated — 64 018 sentences with
predicates, argument spans and questions people wrote and voted on.
`qasrl_bridge` converts it into the same `Document` + `AnnotationRun` shape
everything else uses:

```python
from semantic_corpus.qasrl_bridge import iter_bank_canonical, bank_qa_examples

for document, run in iter_bank_canonical("data/qasrl-v2/orig/dev.jsonl.gz"):
    assert run.validate_against(document) == []
```

Over the whole bank: **0 invalid runs and 257 549 QA examples**, with the
roles normalised through the typed `Frame` and the original question kept on
each relation — the handoff makes the question primary and the role derived.
Entity types are *not* invented: the bank never recorded them, so mentions
carry an empty label set, which reads as "not annotated".

## Storing and exporting

One command runs the whole pipeline — generate, store, export, report:

```bash
python -m semantic_corpus.cli build --count 100 --seed 42 --no-answer-share 0.15
```

This creates 100 situations per manual frame (1700 with the current seventeen
frames) and, by default, one candidate per mined template. Only mined candidates
assigned to the selected split are stored, and duplicate texts are deduplicated.
This is a large build: use `--template-verb` or `--template-adjective` to narrow
the mined selection, or `--template-count 0` for a manual-frame-only smoke run.

The annotated portion exports QA to `sft.jsonl` and token annotations to
`bio.jsonl`; unreviewed mined documents remain in the same store for subsequent
semantic annotation. `generation-summary.json` records generation and pending
annotation counts; `report.json` contains quality checks. QA counts depend on
available annotations and balancing options. The version manifest records
`count_scope: "per_frame"` so the manual quota's unit is explicit.

The store is a directory of append-only JSONL files — SQLite and Parquet are
both ruled out, and JSONL is the only option that needs no dependency. Four
guards are built in, each covering a failure that is otherwise silent:
ontology versions cannot mix, documents are deduplicated by content hash, a
document cannot change split, and a run whose offsets do not match its text
is refused outright. A published version names exactly what the store held,
with a content hash over every table.

**The SFT export splits prompt from completion** rather than emitting one
blob. The handoff puts the loss on the answer alone; a trainer handed a
single `text` field has to find the boundary by string matching, and a
question containing the tag text would break that silently — training the
model to reproduce the context, which looks like learning and is not.

**BIO is a projection and says what it cost.** A per-token tag sequence
cannot represent a multi-label or nested annotation, so the exporter counts
every span it dropped for overlapping and every label it discarded, instead
of quietly keeping one.

**The report looks for the failure that hides.** Leakage is checked by
content hash rather than identifier — the same text under two ids is exactly
the case an id-based check misses — and by held-out entity name, which is
what turns `unseen_names_test` from a claim into a verified property. An
identical example on both sides of the split line is fatal; the same
question and answer under *different* contexts is reported separately, since
the model still has to read the context.

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

## Deviations from the handoff's file layout

* The ontology ships `ontology/defaults.json`, not `defaults.yaml`, so that
  the core keeps no third-party dependency for what is a config file.
* `semantic_generator/generator.py` is not in the handoff's list. Sampling
  needs frames, pools and realisation at once, and putting it in any of the
  three would make them import one another.
* `documents.py` holds the canonical representation, which the handoff
  specifies but assigns to no module. It sits at the top level because the
  annotator, the storage layer and the exporters all need it.
* `semantic_annotator/ingestion.py` and `pipeline.py` are likewise not in the
  list: reading sources is step 1 of the stage, and the pipeline is the
  wiring that keeps the other four modules independently testable.
* `question_generator/generator.py` and `semantic_generator/canonical.py` are
  the same pattern: a driver that needs every template family at once, and
  the bridge that lets generated and annotated records share one shape.
* Storage is JSONL. The handoff specifies SQLite; the user has ruled out both
  SQLite and Parquet, and JSONL is the only remaining option that keeps the
  package dependency-free.
* `exporters/report.py` is not in the handoff's file list, but the
  distribution-and-leakage report it produces is step 6 of stage 5.

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

## Knowledges

1. https://arxiv.org/pdf/2004.15006

2. https://arxiv.org/pdf/2205.11413

3. https://arxiv.org/pdf/1909.05855

4. https://arxiv.org/pdf/1909.05858

5. https://aclanthology.org/D15-1076.pdf

6. https://aclanthology.org/D15-1044.pdf

7. https://aclanthology.org/D16-1059.pdf

8. https://aclanthology.org/P19-1630.pdf

9. https://aclanthology.org/P18-1191.pdf
