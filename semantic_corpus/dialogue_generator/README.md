# Controlled dialogue generation

An English pilot for training and testing dialogue memory on a small model.
It follows the bAbI idea of generating a situation with a known answer, then
rendering it as language. It does not import TensorFlow or copy bAbI data/code.
The linked Tensor2Tensor [babi_qa.py](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/babi_qa.py)
loads existing story/question/answer data; our simulator creates new dialogues.

From the repository root:

```powershell
python -m semantic_corpus.cli dialogue-build --out artifacts/dialogue-pilot --train 200 --validation 20 --test 20 --seed 42
```

Quotas are **per task**: 16 tasks produce 3,200 training, 320 validation and
320 test episodes. The destination must be new. Generation uses only Python's
standard library, shared semantic frames and the existing
`semantic_generator/entities.json`, respecting its named-entity split rules. It does
not call a model or mine `eng-base.jsonl`.

In the Django UI, open **Pipeline → Chat dataset** at `/chat-dataset`. The page
has the same per-task quotas and seed, displays the total sizes, and queues a
background job. Its job page provides progress, cancellation/retry, split counts,
individual downloads and `chat-dataset.zip`. The archive contains this directory
layout at its root; files are also retained in `artifacts/ui/jobs/<id>/dataset/`.
The UI exports model-independent data. The optional trainer-report adapter below
is available through the CLI.

**User and assistant only** is checked by default in the UI. It generates
conversations about the user's own location, feelings, preferences and plans,
without personal names, named cities or facts about other people. Ordinary places
such as `the kitchen` and `the park` replace city names. The `different_people`
task is excluded: 15 tasks give **3,000 / 300 / 300** episodes at the default
quotas. Uncheck it to use the 16-task named-pool scenarios.

The CLI and Python API retain the original behavior by default. To select the
simplified conversations, add `--user-assistant-only` or pass
`user_assistant_only=True` to `build_dialogue_dataset` / `DialogueGenerator`.
The setting is saved with a UI job and retained on retry; older jobs without it
keep their named-pool behavior. This changes the scenario content, not the
History / Memory / Extraction formats or the system instructions.

In simplified conversations, the first user turn provides a location, feeling
and drink preference. These facts stay available while the conversation tests
recall, meeting corrections, options and the other tasks. Their combinations
provide variation without invented names. The vocabulary is finite: an excessive
quota fails explicitly instead of filling the dataset with duplicate scenarios.

**Include metadata** is checked by default. Uncheck it to export only `messages`
in every training JSON record, omitting the top-level `id`, `scenario_key` and
`task` fields. The equivalent CLI flag is `--no-metadata`; the Python API accepts
`include_metadata=False`. The messages themselves, split/deduplication checks,
episode audit JSONL and manifest remain intact. The selected value is saved with
the job and preserved on retry; older jobs default to including metadata.

## Tasks

| Task | What the target tests |
| --- | --- |
| `recall` | Recall the user's location from the conversation |
| `correction` | Use the corrected meeting day, retaining the old claim only for audit |
| `different_people` | Distinguish the user's feeling from another person's feeling (named-pool only) |
| `negation` | Answer yes or no according to an explicit positive/negative report |
| `desire_vs_done` | A wish or plan does not establish completion: visit, buy, call, send, read, take a break |
| `option_reference` | Resolve “the first/second one” using the assistant's options |
| `ambiguous_reference` | Ask which of two options “that one” refers to |
| `unknown` | Admit a meeting day is unknown when it was never stated |
| `suggestion_vs_done` | A suggestion/thanks is not evidence of completion; handle yes/no/unknown |
| `topic_return` | Recover the user's preference after discussing another topic |
| `retraction` | Withdraw a day without inventing a replacement |
| `capability` | Explain that this chat cannot send messages or set reminders |
| `event_scope` | Separate desire, necessity, intention, decision and nested operator/negation scopes from completion |
| `event_progress` | Distinguish an attempt, start, stop, ongoing action and future report from completion |
| `event_updates` | Correct or withdraw one event while retaining other objects, participants and time contexts |
| `event_states` | Distinguish current possession/location from wishes, modal claims and past states |

The simulator creates the user's assertions and state updates together. Answers
are derived from the resulting state. Every update carries exact evidence;
references to the assistant's options include both turns. `Episode.validate()`
replays updates and checks the answer, IDs, sources, replacements and retirements.
An evidence substring check alone is **not** semantic verification of an external
extractor's output.

Memory v4 retains the original eleven flat relations (`location`,
`meeting_day`, `feeling`, `visited`, `selection`, `completed_break`, `drink`,
`bought`, `called`, `sent`, `read`). Each
subject/relation/modality has one active value. `asserted`, `desired` and `planned` are
separate; positive and negative claims are explicit. Facts describe reports in
this conversation. Assistant suggestions/options live separately from user facts.
These flat facts remain in the original control tasks. The four `event_*`
families use a separate typed event store, described below; they can retain
multiple objects, participants and time contexts without overwriting one another.
Old dataset files are not migrated or rewritten.

## Typed actions, scope and completion

`ACTION` remains an ontology label for a predicate. It does not imply execution.
An `Event` in `events.py` carries the predicate, typed role fillers, explicit time
context, an outer-to-inner operator chain, phase and tense. `EventClaim` adds
internal identity and source evidence. Shared `SemanticFrame` slots validate the
roles/types; `event_scenarios.py` samples eligible common entities from the shared
pool. Broad type compatibility is not enough for plausibility: physical transfer
uses portable document/tool/toy subtypes, and care uses child/dog/cat subtypes.

Reviewed senses currently cover **buy, visit, give, move, pick up, put down,
get up, look after, go**, plus the states **have** (possession) and **be** (location).
The renderer inflects the head, preserves particles and
governed prepositions, and supports both `pick up a book` and `pick a book up`.
Pronoun objects use `pick it up`. Intransitive `get up` has no invented object;
`give` retains its recipient and `move` its destination.

| Operator family | Supported operators | Does the construction establish completion? |
| --- | --- | --- |
| Desire / necessity | `want`, `need` | No |
| Intention / decision | `plan`, `intend`, `decide` | No |
| Attempt | `try` | No |
| Onset / cessation | `start`, `stop` | No |
| Success / failure | `manage`, `fail` | For the reviewed bounded action in the past: yes / no |
| Completion | `finish` | In the past: yes; `not > finish`: no |

The result means **reported completion**, not independently verified world truth.
Starting/stopping an activity does not establish its completion. For the reviewed
`visit` and `look after` senses, an explicit finish is required: past activity
alone is insufficient. Future actions remain unknown. Uncertainty is not denial.

Scope is preserved: `not > want` means absence of desire; `want > not` means a
desire not to act. Similarly, `not > need` is lack of necessity, while
`need > not` is rendered as `need to avoid ...`. Chains such as `plan > try`,
`need > start`, `want > manage` and `start > try` are generated. A success verb
inside a wish does not establish actual success. The contract accepts up to four
operators; generation samples 43 scope profiles and nine progress profiles,
crossing each with all nine action predicates as quotas allow. Eight state
profiles cover possession and location. A small quota does not cover every pair.

Event identity includes **predicate + every typed participant + time**.
Different operator paths have separate claim channels. A replacement within a
channel can change outer polarity, phase or tense; inner negation changes the
channel. Corrections retire the previous claim for audit, and retractions remove
only the addressed claim. Conflicting completion reports in separate channels
give an unknown answer rather than an arbitrary yes/no choice.

Limits: eleven reviewed predicate senses, no quoted beliefs, conditionals or unrestricted passive
grammar. Time phrases are exact context keys, not calendar-aware event IDs; two
identical actions with the same participants and time phrase share a channel.
GROUP fillers require subtype information not yet present in this compact
contract and are rejected. Typed events and legacy flat action facts are not
automatically reconciled with each other.

To expand vocabulary, add eligible entities to the existing pool. To add a new
predicate sense, define/reuse its shared frame, register its forms and active
pattern in `EVENT_LEXEMES`, review filler constraints and completion behavior,
and add contrast tests. New operators also need scope, complement grammar and
truth rules; adding a word to a list alone is insufficient. Real domain examples
with roles, context and contrasting intention/attempt/completion readings help
review these choices. Mined constructions are useful candidates, not certified
event semantics. No external dataset is required for the current generator.

### Controlled modals, auxiliaries and states

The shared Synthetic corpus generator now has 17 manual frames. New `go`, `have`
and `be` frames mean motion to a destination, possession and location respectively.
Their paradigms include `goes/went/gone`, `has/had` and `am/is/are/was/were/been`;
copular questions use inversion, and the possession/location senses reject
progressive aspect. Dialogue generation uses these shared slots plus reviewed
complement grammar in `verb_operators.py` (23 operators, excluding `not`).

| Construction | Representation | Interpretation for the target action/state |
| --- | --- | --- |
| `can/could buy`, `cannot buy` | `can`, `could`, `not > can` | Ability/permission/possibility or inability; no completion evidence |
| `must buy`, `must not buy` | `must`, `must > not` | Obligation/deduction or prohibition; no completion evidence |
| `may/might buy`, `may not buy` | `may`, `might`, `may > not` | Possibility/permission; no completion evidence |
| `have to buy`, `do not have to buy` | `have_to`, `not > have_to` | Obligation versus absence of obligation |
| `was able to buy` | `be_able_to`, past | Conservatively does not confirm execution in this contract |
| `am/was going to buy` | `be_going_to` | Prospective intention, distinct from motion `go to the park` |
| `may have bought`, `might have been buying` | `may > have`, `might > have > be` | Perfect/progressive nested under a modal, still no confirmed completion |
| `Anna let me buy` | `let`, actor `user`, permitter `Anna` | Permission with distinct participants |
| `Let's buy`, `Let's not buy` | `let_us`, `let_us > not` | A user proposal to the interlocutors; no completion evidence |
| `I have a book`, `I am in the park` | Bare `have`/`be` state | A reported state, not a completed action |

Finite modals use `tense=modal`: `could` and `might` are explicit forms, not
automatic replacements for past `can`/`may`. Invalid chains such as `want to can`
or stacked finite modals are rejected. `must not` differs from `do not have to`.
Unambiguous negative contractions are sampled; ambiguous `'d`/`'s` are avoided.
The grammar follows [Cambridge's modal forms](https://dictionary.cambridge.org/us/grammar/british-grammar/modality-forms),
[let](https://dictionary.cambridge.org/grammar/british-grammar/let) and
[going to](https://dictionary.cambridge.org/grammar/british-grammar/going-to).

Bare completed bounded actions use phase `completed`, and bare progressives use
phase `ongoing`; duplicate operator encodings are refused. Nested auxiliary
`have` and `be` remain separate from lexical possession/location. A past state
does not establish the current state. Permission claims include the permitter
in their claim channel, while the underlying target event retains its actor.
`let_us` currently assumes a user proposal involving the interlocutors; it does
not encode an arbitrary group. Other operator chains retain the same actor.

These are reviewed senses, not complete polysemy coverage: adjectival/identity
`be`, experiential/causative `have`, rental `let`, and additional `go` idioms
need separate semantic rules. Modal ambiguity is retained, not automatically
resolved into a single permission/ability reading.

### Working mined bank and immutable archives

`controlled_verbs.py` excludes `can`, `must`, `may`, `be`, `let`, `have`, `go`
and their recognized forms from the **working Built-in mined predicate view**.
Verb inventories, alternatives and constructions are filtered; a template with
any controlled lexical/copular head is skipped before generation, including
embedded heads and multiword predicates headed by these verbs. Auxiliary uses
inside other predicates (`has bought`, `is moving`, `can buy`) remain eligible.

The policy applies to all published banks and cannot be bypassed by a Built-in
lemma filter. It does not delete or rewrite `data/eng-base-templates/`, occurrences,
manifests, or other source components. `TemplateBank` and raw `template-generate`
remain archive/reprocessing APIs; normal Synthetic corpus uses the filtered
`BuiltInPool`. UI inventory totals describe archived entries, not the number of
eligible templates. Policy version is recorded in generated document metadata.

## Files and training targets

```text
dialogue-pilot/
  history/{train,validation,test}.json
  memory/{train,validation,test}.json
  extraction/{train,validation,test}.json
  {train,validation,test}.episodes.jsonl
  manifest.json
```

By default, each `.json` is an array of records containing `messages`, `id`, `scenario_key`
and `task`, formatted with two-space indentation. Every record has
**system → user → assistant** roles. For both
response tasks only the final answer is an assistant target; earlier exchanges
are context inside the user message, so one view is not additionally trained on
all the intermediate “Got it” replies.

- **history:** the preceding dialogue and current request → natural answer.
- **memory:** simulated current facts rendered as short English sentences,
  assistant suggestions/ordered options, last assistant message and current
  request → the **same** natural answer. Message content contains no embedded
  JSON, fact/turn IDs or evidence objects. This is oracle memory, not a prediction
  from a trained extractor.
- **extraction:** active facts, assistant options/suggestions, previous assistant
  message and current user message as text → text memory-update commands.
  No embedded JSON, internal IDs or evidence objects are shown to the model.
  Repeated transition prompts are deduplicated. This is a separate task;
  keep updater training separate from natural-answer training.

For the first training stage, use **memory**. For example, the model's input is:

```text
Known facts:
- Your meeting is on Friday.
- You want to visit the park.
- You have not visited the park.

Last assistant message: Got it.
Latest user message: When is my meeting?
```

The target is `Your meeting is on Friday.` The outer dataset remains a JSON
array of `messages`; the text seen by the response model is plain English.
This applies with either value of **Include metadata** and with or without
named people. Metadata controls only the record's outer fields.

`memory_text.py` renders active facts only. Corrected/withdrawn facts stay in
the audit, and missing information does not become a negative assertion.
Wishes, negative reports and completed actions use distinct sentences; assistant
suggestions are separate, and offered options retain their order. Unsupported
relations/actions fail explicitly instead of silently omitting information.
The internal memory and audit keep their structured schema. **Extraction** uses
the text command contract described below, with lossless conversion to internal
deltas for the supported pilot scenarios.

The manifest now records `schema_version: dialogue-memory-v4`,
`prompt_version: dialogue-prompts-v6`, `content_formats.memory: memory-text-v4` and
`content_formats.extraction: extraction-text-v3`. Generate a new job/directory to get
the text format; existing output files are not rewritten. Text-based prompt
deduplication can select different episodes when older prompts differed only
by internal identifiers. Paired History/Memory targets and split checks remain
enforced.

## Text extraction contract

`extraction_text.py` provides the renderer, parser and updater adapter. This is
a compact structured text language, not unrestricted prose. An input can be:

```text
Current memory:
user | meeting_day | asserted | positive | Monday

Assistant actions (earliest first):
(none)

Last assistant message: text: Got it.
Latest user message: Actually, my meeting is on Friday.
```

Its assistant target is:

```text
SET user | meeting_day | asserted | positive | Friday
```

The field order is `subject | relation | modality | polarity | value`.

| Command | Meaning |
| --- | --- |
| `SET subject \| relation \| modality \| polarity \| value` | Add a fact or replace the current fact with that subject/relation/modality |
| `SET_CONTEXT subject \| relation \| modality \| polarity \| value` | Same update, citing both the user and previous assistant turn for a resolved reference |
| `REMOVE subject \| relation \| modality` | Withdraw the active fact; absence is not a negative claim |
| `EVENT predicate \| time \| scope \| phase \| tense \| role=TYPE:value ...` | Add/replace a typed event claim; all selected frame roles are required |
| `EVENT_CONTEXT` followed by the event fields | Same update, citing the user and previous assistant turn |
| `REMOVE_EVENT` followed by the exact active event fields | Withdraw that event claim |
| `NO_CHANGE` | No memory update; this command must be alone |

Multiple updates use one command per line. Wishes/plans remain separate from
completion: `SET user | bought | desired | positive | a book` does not assert
that the book was bought. Those legacy relations remain supported. The new event
families use typed commands instead, for example:

```text
EVENT pick up | today | want>try | unspecified | present | agent=PERSON:user | theme=PHYSICAL_OBJECT:a book
```

This represents `I want to try to pick up a book today.` It does not report
completion. Scope `none` means no operator; phases are `unspecified`, `ongoing`,
`completed`, and tenses are `present`, `past`, `future`, `modal`. Nontrivial operators use
`unspecified` phase so phase scope is not silently lost. Future supports bare actions
and nonmodal operator chains. Permissive `let` appends `permitter=PERSON:...`;
the actor stays in the base frame role. Active events appear in a separate
`Event memory:` input block.

`decode_update(text, memory, user_turn, previous_assistant)` allocates fresh
fact/event IDs and attaches the exact current utterance as evidence. `SET_CONTEXT` / `EVENT_CONTEXT`
also attaches the previous assistant utterance. Unknown relations/modalities,
invalid event roles/types/scopes, malformed fields, conflicting commands and removal of a missing claim are
rejected atomically. Source recording is not a semantic truth check: an updater
can still interpret the utterance incorrectly, which evaluation must measure.
Every exported target is decoded and compared to the original delta, including
its IDs and evidence, before publication. Unsupported evidence shapes fail
instead of silently losing provenance.

Fields escape backslash, pipe and line breaks (`\\`, `\|`, `\n`, `\r`, `\t`).
Internal audit IDs, retired facts and full evidence never enter the model input.
`parse_prompt` reconstructs visible state with local IDs for single-turn
evaluation; live updates must use the actual stored Memory instead.

Old datasets are retained unchanged. The previous event-v3 experiment used:

```powershell
python -m semantic_corpus.cli dialogue-build --out artifacts/dialogue-events-v3-final --user-assistant-only --train 200 --validation 20 --test 20 --contrast-multiplier 3 --seed 20261009
```

This configuration produces 3,600/360/360 paired response episodes and
9,400/966/967 deduplicated Extraction records in the frozen local dataset.
GPT-2 ChatML encoding uses at most 504 tokens per Extraction record (47 for the
target), with no truncation. Memory and History maxima are 190 and 215 tokens.
The validation report is `artifacts/reviews/dialogue-events-v3-validation.json`;
it checks all three views, split separation and the 450-record wording challenge
in `artifacts/dialogue-events-v3-challenge/`. No full GPT-2 training/quality
measurement has been performed on these new event records. The
earlier text-v1 dataset (before event tasks) remains in
`artifacts/dialogue-extraction-text-v1/` with 8,057/830/832 extraction examples;
the original expanded-v2 files with embedded JSON remain unchanged too.

The current controlled-verbs dataset can be generated into a new directory with:

```powershell
python -m semantic_corpus.cli dialogue-build --out artifacts/dialogue-controlled-verbs-v4-final --user-assistant-only --train 200 --validation 20 --test 20 --contrast-multiplier 3 --seed 20261009
python -m semantic_corpus.dialogue_generator.challenge --out artifacts/dialogue-controlled-verbs-v4-final-challenge --groups-per-task 30 --seed 20261009
```

The frozen local run contains **3,800/380/380** paired Memory/History records,
**9,802/1,009/1,008** Extraction records and **540** challenge cases. All command
targets are parsed/applied, all views are checked for split overlap, and paired
answers and GPT-2 ChatML lengths are validated in
`artifacts/reviews/dialogue-controlled-verbs-v4-validation.json`.
The maxima are 190 tokens for Memory, 214 for History and 559 for Extraction;
the longest Extraction target is 54 tokens. Use `--max-length 768` for Extraction.
No model training or quality comparison has been performed on this new dataset.
The earlier directory without `-final` is an intermediate validation run.

The audit JSONL retains full turns, state before/after each user turn, deltas,
retired facts, reference query, expected dialogue action and supporting IDs.
Answers, reference queries and action labels are never included in response
prompts. The manifest records versions, quotas, per-task/action/update counts,
resource/code hashes and output checksums. It is written only after a successful
build. A failed build leaves the requested output directory unpublished.

Sampling uses independent seeded streams per task and split. Increasing a quota
extends that stream. One rendering of each distinct event plan is retained;
all response views and transition examples from an episode keep its split.
Normalized duplicate prompts and conflicting targets are checked. Named-pool
scenarios partition names through the shared entity pool before rendering.
Simplified scenarios hash the complete starting `(location, feeling, drink)`
context into train/validation/test groups (80/10/10 buckets). All tasks and
surface variants sharing that context stay in one split, independent of seed
and quotas. This also keeps identical initial memory-update prompts out of
different splits. It holds out combinations, not individual words.
**Task families, sentence patterns, days, feelings and other common vocabulary
are shared**. Test performance measures this controlled distribution with
held-out names or contexts and plans, not unseen reasoning tasks or unrestricted
chat. Keep a separate human-written test set for broader assessment.

## Local GPT-2 SFT experiment

### Contrasts and varied memory

New exports (including UI jobs) generate `desire_vs_done` and `suggestion_vs_done`
in matched groups: the context, question and wording style stay fixed while
confirmation, explicit denial or missing evidence changes the correct answer.
Groups cover visits, purchases, calls, messages, reading and breaks. They also
include a different completed action/object and accepting a suggestion without
reporting completion. The model still receives natural English, without outcome
labels or group IDs. Three surface families vary the actual Memory input as
well as the dialogue and question; changing only History would not diversify Memory.

Outer metadata additionally records `contrast_group`, `condition`, `answer_kind`;
`--no-metadata` removes these too. All members of a group stay in one split,
across seeds and quotas. Nonmultiples of three can leave the final group partial;
normal prompt deduplication also takes precedence over retaining a whole group.
These two original tasks retain their single-active-value facts. The new
`event_*` families use the typed event store and also produce matched
yes/no/unknown groups, including simultaneous events and nested scope.

For experiments, CLI/API can multiply the quotas for these two tasks:

```powershell
python -m semantic_corpus.cli dialogue-build --out artifacts/dialogue-memory-expanded --user-assistant-only --train 200 --validation 20 --test 20 --contrast-multiplier 3 --seed 20261009
python -m semantic_corpus.dialogue_generator.challenge --out artifacts/dialogue-memory-challenge --groups-per-task 30
```

The first command now creates **3800/380/380** response records. The API option is
`contrast_multiplier=3`; its default is 1, including in the UI, so existing
per-task quota fields keep their meaning. The manifest records actual task weights.
The second command now freezes **540 test-only** records across six task families
in `test.json`, with two
additional memory/question surface families absent from normal training exports.
It saves data/code hashes and never overwrites a previous challenge. This measures
unseen wording with shared action vocabulary, not unrestricted chat. Freeze it
before training and do not tune on its results. It is an evaluation dataset,
not a new UI generation mode.
Previously frozen 180-record challenges are retained unchanged.
With the same seed as the ordinary dataset, some held-out event plans are
deliberately shared: this is a paired wording stress test, not an independent
second sample. Neither its contexts nor its groups enter train/validation.

The optional [GPT-2 trainer](trainer/README.md) fine-tunes the standard pretrained
model with `transformers.Trainer`, ChatML and loss on assistant answers only.
It supports text `memory/`, `history/` and `extraction/` views in separate runs,
with `memory/` as the default, and has separate `train` and `evaluate`
CLI commands. Training is not connected to Django and is never started by
generating a dataset. Install `requirements-training.txt` only for this component.

## Using the existing `test-llm` trainer

The `messages` structure matches `train_chat.py`'s `load_training_rows` contract.
That trainer also requires `report.json` with a model, token limit, seed and chat
template hash. You can supply its existing preparation report:

```powershell
python -m semantic_corpus.cli dialogue-build --out artifacts/dialogue-pilot-training --train 200 --validation 20 --test 20 --seed 42 --training-report C:/git/test-llm/data/chitchat/report.json
```

This creates a small `report.json` in each view directory, carrying the existing
tokenizer contract and the new seed. It does **not** copy the old dataset's
statistics or pretend to have tokenized the new data. The trainer checks the
actual tokenizer hash and sequence lengths before loading weights; it must
refuse overlong records instead of truncating context. Token counts, padding
and assistant-only loss are responsibilities of the training project.

For the first experiment, in `C:/git/test-llm/train_chat.py` set `DATA_DIR` to the
absolute `.../dialogue-pilot-training/memory` directory and use a **new**
`OUTPUT_DIR`. Train the response task first. Then repeat with `history`, starting
from the **same original checkpoint** and using identical training settings and
the paired held-out cases. Do not continue the second experiment from the first
experiment's trained weights. `test.json` is reserved for evaluation; do not
merge it into training or use it to choose training settings.

At evaluation, pass `row["messages"][:-1]` to the model; the final message is the
reference answer. Report results by `task`, including yes/no/unknown and
clarification behavior. Exact string match is a useful strict baseline, but
correct paraphrases need semantic review. Preserve the original 100 handwritten
memory cases as an additional regression set. This generator does not train a
model or demonstrate any quality improvement by itself.

Only after the model can answer using oracle memory should the `extraction`
dataset be used to train/check an updater and the complete rolling dialogue
pipeline. The first pilot has short, controlled conversations with fixed prior
assistant replies; it is not a substitute for testing free-running conversations.
The existing Synthetic corpus and mined QA exports are unchanged.
