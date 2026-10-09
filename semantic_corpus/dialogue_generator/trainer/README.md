# GPT-2 ChatML SFT trainer

An optional **CLI-only** experiment for the generator's text **Memory**,
**History** and **Extraction** datasets, selected explicitly with `--view`.
It fine-tunes the standard pretrained `gpt2` using `transformers.Trainer`.
Memory is the default. Old Extraction JSON is refused. Training is not
registered as a Django operation.

Install the optional dependencies from the repository root:

```powershell
python -m pip install -r requirements-training.txt
```

PyTorch with CUDA support is needed for GPU training. The default `auto` device
uses CUDA when available and otherwise CPU. The tested package versions are
PyTorch 2.8.0 and Transformers 4.56.1. Only this optional trainer uses Hugging Face;
the corpus generator and Django UI do not import these dependencies.
Optional generation-time token limits use the smaller `requirements-tokenization.txt`
dependency set and a cached tokenizer; they do not load this training engine.

## Prepare and train

Generate a new dataset with textual memory, or use the `dataset/memory/`
directory from a new Chat dataset job:

```powershell
python -m semantic_corpus.cli dialogue-build --out artifacts/dialogue-gpt2-memory --user-assistant-only --train 200 --validation 20 --test 20
python -m semantic_corpus.dialogue_generator.trainer train --mode sft --model gpt2 --data artifacts/dialogue-gpt2-memory/memory --out artifacts/training/gpt2-memory --epochs 3 --batch-size 2 --accumulation 8
```

Both output directories must be new. The first command produces 3,200 training,
320 validation and 320 test examples. Add `--mined-train 0` to the generator
command for complete mined-head coverage in training; UI jobs use it by default.
The trainer accepts both records with
metadata and records containing only `messages`. It never feeds outer IDs or
task labels to the model. Retain metadata if you want evaluation broken down
by task; otherwise all examples are reported as `unlabeled`.

To focus on completion reasoning, use the generator's `--contrast-multiplier 3`
when preparing data; at the same base quotas this produces 4,000 training records
unless the mined training quota is separately overridden.
See [contrast generation and the frozen challenge](../README.md#contrasts-and-varied-memory).

On its first online run, model/tokenizer files are cached under
`artifacts/models/huggingface/`. To use an existing local pretrained GPT-2:

```powershell
python -m semantic_corpus.dialogue_generator.trainer train --model C:/models/gpt2 --local-files-only --data artifacts/dialogue-gpt2-memory/memory --out artifacts/training/gpt2-memory-local
```

`--max-steps 4` limits a smoke test to four optimizer updates. It overrides the
epoch budget according to Trainer's semantics. `--save-only-model` omits
optimizer states from intermediate checkpoints to reduce disk use. Use a fresh
output directory for each experiment; automatic resume is not exposed by this CLI.

## What the model sees and learns

```text
<|im_start|>system
You are a friendly AI personal assistant. ...<|im_end|>
<|im_start|>user
Known facts:
- Your meeting is on Friday.

Last assistant message: Got it.
Latest user message: When is my meeting?<|im_end|>
<|im_start|>assistant
Your meeting is on Friday.<|im_end|>
```

`<|im_start|>`, `<|im_end|>` and `<|pad|>` are added as **individual special
tokens** to the original GPT-2 tokenizer. The pretrained embeddings are resized;
new token embeddings are initialized and trained along with the model. The saved
tokenizer includes the ChatML chat template, and `<|im_end|>` is the generation
stop token. The terminal newline after the assistant's end marker is not part
of the supervised target.

This is **SFT with loss only on the final assistant answer and its end marker**.
System instructions, context, role headers and padding are masked with `-100`.
The custom collator preserves those labels; the loss function normalizes by
supervised tokens across accumulated microbatches. Earlier exchanges embedded
in the user content are context, not additional answer targets.

The trainer requires the generator's three-message `system/user/assistant`
contract. `--view memory` requires `Known facts:`, `--view history` requires
`Conversation so far:`, and `--view extraction` requires `Current memory:`
with the text extraction contract. Extraction reference commands must parse
and apply to the supplied state. Old exports with embedded JSON and datasets
that do not match the selected view are refused.
Records containing literal control tokens are refused. Examples exceeding
`--max-length` (default 512) fail explicitly; there is no truncation or packing.
The selected length must also fit the model's positional embeddings.
Chat dataset's **Maximum example tokens (GPT-2)** field can enforce this limit
before export. It defaults to 768 in the UI; pass `--max-length 768` here too.
The UI setting does not change the trainer's default or the model context size.
See the generated `token-lengths.json` for the exact tokenizer contract and maxima.

Training reads only `train.json` and `validation.json`, checks normalized prompt
and available scenario/contrast-group overlap, and chooses the best trained checkpoint by
validation loss. `test.json` is not used to train or select a checkpoint. The
default effective batch size is 2 × 8 = 16 examples on one device. Automatic
precision selects BF16 on supported CUDA devices, otherwise FP16 on CUDA, and
FP32 on CPU; `--precision float32` overrides it. This CLI supports one process.

## Train the text memory updater

New exports use `extraction-text-v4` (older files are not converted
in place). Select `extraction/` explicitly. The following commands refer to the
older frozen v3 updater dataset, whose token lengths were measured:

```powershell
python -m semantic_corpus.dialogue_generator.trainer train --view extraction --model gpt2 --data artifacts/dialogue-controlled-verbs-v4-final/extraction --out artifacts/training/gpt2-extraction-controlled --epochs 3 --max-length 768 --save-only-model
python -m semantic_corpus.dialogue_generator.trainer evaluate --run artifacts/training/gpt2-extraction-controlled --data artifacts/dialogue-controlled-verbs-v4-final/extraction --out artifacts/training/gpt2-extraction-controlled-test --max-new-tokens 256
```

These are training/evaluation commands, not a claim of measured updater quality.
The existing recorded GPT-2 experiment below still concerns Memory only.
The frozen controlled-verbs-v4 Extraction files contain 9,802/1,009/1,008 records.
Full ChatML sequences reach 559 GPT-2 tokens (54 target tokens); use 768 for this dataset.
The report is `artifacts/reviews/dialogue-controlled-verbs-v4-validation.json`.
The previous event-v3 files remain unchanged (9,400/966/967, maximum 504 tokens).
New quotas, vocabulary or constructions can
change the maximum; the trainer validates every record before training.
An offline tiny-model integration test covers SFT and reload/evaluation for all
three views. Use a fresh pretrained checkpoint for each comparison, not the
already trained Memory checkpoint.

For the new mined-bank sample, use `artifacts/dialogue-mined-v5-verified/extraction`.
Its 32,064/857/853 records reach 725 GPT-2 ChatML tokens (69 target tokens), so
`--max-length 768` is sufficient for this frozen sample. The validation report is
`artifacts/reviews/dialogue-mined-v5-validation.json`. The data has not been used
for a training run. All 4,031 inventory entries occur in train, but only seven
are sampled in each evaluation split. Distinguish compositional occurrences
from quotation tasks when interpreting future metrics; neither coverage nor
successful command roundtrips establish model quality or reviewed semantics.

The updater predicts `SET`, `SET_CONTEXT`, `REMOVE`, `EVENT`, `EVENT_CONTEXT`,
`REMOVE_EVENT` or `NO_CHANGE`; code creates
IDs and exact source quotes. The dataset container remains a JSON array of
`messages`, while message content is text. See the
[text contract](../README.md#text-extraction-contract) for fields and provenance.
The view is saved in `run.json`; evaluation uses that view and rejects mismatches.
History is enabled similarly with `--view history` and its own input/output paths.

Extraction evaluation adds parse/apply validity, order-independent update
matching (including source mode), active-fact and event-state matching, false changes on
`NO_CHANGE` examples, and breakdowns by fact/event add/replace/remove/no-change. Invalid
predictions count as failures. A correct final state can still have wrong source
mode, so both metrics are retained. These checks use the supplied prior state;
they do not measure accumulated errors in a rolling conversation. Increase
`--max-new-tokens` for multi-command answers; outputs are never silently repaired.

## Evaluate the held-out examples

```powershell
python -m semantic_corpus.dialogue_generator.trainer evaluate --run artifacts/training/gpt2-memory --data artifacts/dialogue-gpt2-memory/memory --out artifacts/training/gpt2-memory-test
```

This loads the saved **best** model and its tokenizer, checks split overlap,
computes answer-token loss and greedily generates answers without supplying
reference answers to `generate()`. `--split validation` selects validation;
`--limit 20` evaluates a labeled subset of the chosen split. The default is the
entire test split. Generation stops at `<|im_end|>` or `--max-new-tokens` (64 by
default); hitting the limit is recorded as `ended: false`.

Training output:

```text
gpt2-memory/
  model/                 # Best model, tokenizer, ChatML template, generation config
  checkpoints/           # Trainer checkpoints and state
  run.json               # Settings, versions, data checksums, losses, best checkpoint
  split_keys.json        # Prompt/scenario keys for leakage checks
```

Evaluation output contains `evaluation.json` and `predictions.jsonl` with the
input, reference answer, generated answer, exact-match result and stop status.
For Memory/History, diagnostics include counts/confusion by task, answer kind and condition,
`unknown_false_yes`, and the fraction of complete three-example contrast groups
whose answers are all correct. Partial groups are reported separately. Answer-kind
classification uses explicit text prefixes and is not a semantic judge; metadata
is needed for task/group breakdowns but never enters the model input.

Evaluate the frozen wording challenge by pointing `--data` to its directory:

```powershell
python -m semantic_corpus.dialogue_generator.trainer evaluate --run artifacts/training/gpt2-memory --data artifacts/dialogue-memory-challenge --out artifacts/training/gpt2-memory-challenge
```

Exact match trims surrounding whitespace but otherwise compares literally;
it is a strict baseline, not a semantic correctness score. Examine mistakes and
per-task counts before drawing conclusions. The synthetic Memory input is still
the generator's oracle state, so this tests answering from supplied facts, not
the quality of an automatic memory extractor or an unrestricted chat system.

To compare saved runs on the same frozen files, with separate predictions for
every model/dataset pair and an aggregate JSON/Markdown table:

```powershell
python -m semantic_corpus.dialogue_generator.trainer.compare --run baseline=artifacts/training/gpt2-memory-baseline-3ep --run expanded=artifacts/training/gpt2-memory-expanded-v2-6ep --dataset expanded=artifacts/dialogue-gpt2-memory-expanded-v2/memory --dataset challenge=artifacts/dialogue-gpt2-memory-challenge-v2 --out artifacts/training/memory-comparison
```

Use `--split validation` with ordinary dataset directories for development.
The default is test; never use the frozen challenge to select a checkpoint or
adjust generation rules. Increasing training length and expanding data are
different interventions; keep a same-data, longer-training control.

Architectural/API references: [GPT-2](https://huggingface.co/docs/transformers/v4.56.1/en/model_doc/gpt2)
and [Trainer](https://huggingface.co/docs/transformers/v4.56.1/en/main_classes/trainer).

## Recorded comparison (2026-10-09)

All runs started independently from the same standard GPT-2 weights with seed 42,
SFT, learning rate 5e-5, batch 2, accumulation 8, BF16. The baseline used the
2,200-record Memory dataset from generator revision `cedad85`. Expanded training
used 3,000 records, with 600 examples in each completion task. Checkpoint selection
used validation loss. The six-epoch follow-up was decided from validation errors;
its data matched the three-epoch run byte for byte.

| Training | Expanded test (300) | Unseen wording (180) | Unknown answered yes (60 challenge cases) |
| --- | --- | --- | --- |
| Original data, 1 epoch | 247/300 (82.3%) | 67/180 (37.2%) | 59 |
| Original data, 3 epochs | 278/300 (92.7%) | 134/180 (74.4%) | 44 |
| Expanded data, 3 epochs | 292/300 (97.3%) | 167/180 (92.8%) | 13 |
| Expanded data, 6 epochs | 300/300 (100%) | 169/180 (93.9%) | 11 |

The final model is `artifacts/training/gpt2-memory-expanded-v2-6ep/model/`.
It also answers all 220 original test records correctly. The final checkpoint
is epoch 6, step 1128; training took 571.8 seconds on the local RTX 3070 Laptop GPU.
The following commands describe that historical run. Its generator revision had
11 user-only tasks; the current version adds four typed-event tasks and one mined
task, producing 4,000 records with the same uniform quotas. Use the retained files for a like-for-like
comparison, or a fresh destination for new data; do not regenerate over old files.

```powershell
python -m semantic_corpus.cli dialogue-build --out artifacts/dialogue-gpt2-memory-expanded-v2 --user-assistant-only --train 200 --validation 20 --test 20 --contrast-multiplier 3 --seed 20261009
python -m semantic_corpus.dialogue_generator.trainer train --data artifacts/dialogue-gpt2-memory-expanded-v2/memory --out artifacts/training/gpt2-memory-expanded-v2-6ep --model gpt2 --epochs 6 --save-only-model
```

The combined local report is
`artifacts/training/gpt2-memory-comparison-20261009/summary.json` (and `summary.md`);
it links to every evaluation and its predictions. Correct complete challenge
groups increased from 14/60 for baseline3 to 49/60 for expanded6.

All 11 final mistakes are `accepted:unknown`: choosing an action is mistaken
for completing it under unseen wording. For example, choosing to send an email
does not establish that it was sent. Expanded3 instead failed 12 other-object
and one other-action cases, so aggregate improvement does not mean every
condition improved. The challenge was not changed to remove these failures.
This remains a single-seed oracle-memory experiment. Forty challenge groups
share held-out event plans with the ordinary expanded test; the tests are not
independent. Action vocabulary/task families are shared with training. Data
size, variety and epoch-dependent learning-rate schedules also differ, so this
does not isolate the causal contribution of each change or establish chat quality.
