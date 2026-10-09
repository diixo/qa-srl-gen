# GPT-2 ChatML SFT trainer

An optional **CLI-only** experiment for the generator's plain-text **Memory**
datasets. It fine-tunes the standard pretrained `gpt2` using
`transformers.Trainer`. It does not initialize a model from scratch, train on
Extraction JSON, or register any Django operation.

Install the optional dependencies from the repository root:

```powershell
python -m pip install -r requirements-training.txt
```

PyTorch with CUDA support is needed for GPU training. The default `auto` device
uses CUDA when available and otherwise CPU. The tested package versions are
PyTorch 2.8.0 and Transformers 4.56.1. Only this optional trainer uses Hugging Face;
the corpus generator and Django UI do not import these dependencies.

## Prepare and train

Generate a new dataset with textual memory, or use the `dataset/memory/`
directory from a new Chat dataset job:

```powershell
python -m semantic_corpus.cli dialogue-build --out artifacts/dialogue-gpt2-memory --user-assistant-only --train 200 --validation 20 --test 20
python -m semantic_corpus.dialogue_generator.trainer train --mode sft --model gpt2 --data artifacts/dialogue-gpt2-memory/memory --out artifacts/training/gpt2-memory --epochs 3 --batch-size 2 --accumulation 8
```

Both output directories must be new. The first command produces 2,200 training,
220 validation and 220 test examples. The trainer accepts both records with
metadata and records containing only `messages`. It never feeds outer IDs or
task labels to the model. Retain metadata if you want evaluation broken down
by task; otherwise all examples are reported as `unlabeled`.

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
contract and a prompt beginning with `Known facts:`. Old Memory exports with
embedded JSON, History exports and Extraction exports are refused in this stage.
Records containing literal control tokens are refused. Examples exceeding
`--max-length` (default 512) fail explicitly; there is no truncation or packing.
The selected length must also fit the model's positional embeddings.

Training reads only `train.json` and `validation.json`, checks normalized prompt
and available scenario-key overlap, and chooses the best trained checkpoint by
validation loss. `test.json` is not used to train or select a checkpoint. The
default effective batch size is 2 × 8 = 16 examples on one device. Automatic
precision selects BF16 on supported CUDA devices, otherwise FP16 on CUDA, and
FP32 on CPU; `--precision float32` overrides it. This CLI supports one process.

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
Exact match trims surrounding whitespace but otherwise compares literally;
it is a strict baseline, not a semantic correctness score. Examine mistakes and
per-task counts before drawing conclusions. The synthetic Memory input is still
the generator's oracle state, so this tests answering from supplied facts, not
the quality of an automatic memory extractor or an unrestricted chat system.

Architectural/API references: [GPT-2](https://huggingface.co/docs/transformers/v4.56.1/en/model_doc/gpt2)
and [Trainer](https://huggingface.co/docs/transformers/v4.56.1/en/main_classes/trainer).
