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

Quotas are **per task**: 12 tasks produce 2,400 training, 240 validation and
240 test episodes. The destination must be new. Generation uses only Python's
standard library and the named PERSON/LOCATION entries of the existing
`semantic_generator/entities.json`, respecting its entity split rules. It does
not call a model or mine `eng-base.jsonl`.

In the Django UI, open **Pipeline → Chat dataset** at `/chat-dataset`. The page
has the same per-task quotas and seed, displays the total sizes, and queues a
background job. Its job page provides progress, cancellation/retry, split counts,
individual downloads and `chat-dataset.zip`. The archive contains this directory
layout at its root; files are also retained in `artifacts/ui/jobs/<id>/dataset/`.
The UI exports model-independent data. The optional trainer-report adapter below
is available through the CLI.

**Include metadata** is checked by default. Uncheck it to export only `messages`
in every training JSON record, omitting the top-level `id`, `scenario_key` and
`task` fields. The equivalent CLI flag is `--no-metadata`; the Python API accepts
`include_metadata=False`. The messages themselves, split/deduplication checks,
episode audit JSONL and manifest remain intact. The selected value is saved with
the job and preserved on retry; older jobs default to including metadata.

## Tasks

| Task | What the target tests |
| --- | --- |
| `recall` | Recall the user's location despite a fact about someone else |
| `correction` | Use the corrected meeting day, retaining the old claim only for audit |
| `different_people` | Distinguish the user's feeling from another person's feeling |
| `negation` | Answer yes or no according to an explicit positive/negative report |
| `desire_vs_done` | A wish to visit somewhere does not establish a completed visit |
| `option_reference` | Resolve “the first/second one” using the assistant's options |
| `ambiguous_reference` | Ask which of two options “that one” refers to |
| `unknown` | Admit a meeting day is unknown when it was never stated |
| `suggestion_vs_done` | A suggestion/thanks is not evidence of completion; handle yes/no/unknown |
| `topic_return` | Recover the user's preference after other topics and people's preferences |
| `retraction` | Withdraw a day without inventing a replacement |
| `capability` | Explain that this chat cannot send messages or set reminders |

The simulator creates the user's assertions and state updates together. Answers
are derived from the resulting state. Every update carries exact evidence;
references to the assistant's options include both turns. `Episode.validate()`
replays updates and checks the answer, IDs, sources, replacements and retirements.
An evidence substring check alone is **not** semantic verification of an external
extractor's output.

Memory v1 is a deliberately small projection: seven relations (`location`,
`meeting_day`, `feeling`, `visited`, `selection`, `completed_break`, `drink`). Each
subject/relation/modality has one active value. `asserted` and `desired` are
separate; positive and negative claims are explicit. Facts describe reports in
this conversation. Assistant suggestions/options live separately from user facts.
This is not a general event log: arbitrary tense, multiple visits, quoted beliefs,
world truth verification and unrestricted conversation are not modeled.

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
- **memory:** simulated current facts, assistant actions, last assistant message
  and current request → the **same** natural answer. This is oracle memory,
  not a prediction from a trained extractor.
- **extraction:** state before a user turn, previous assistant message, current
  user turn and next fact ID → JSON delta with `add`, `replace`, `invalidate`.
  Exact repeated transition prompts are deduplicated. This is a separate task;
  do not blindly merge its JSON targets into response training.

The audit JSONL retains full turns, state before/after each user turn, deltas,
retired facts, reference query, expected dialogue action and supporting IDs.
Answers, reference queries and action labels are never included in response
prompts. The manifest records versions, quotas, per-task/action/update counts,
resource/code hashes and output checksums. It is written only after a successful
build. A failed build leaves the requested output directory unpublished.

Sampling uses independent seeded streams per task and split. Increasing a quota
extends that stream. One rendering of each distinct event plan is retained;
all response views and transition examples from an episode keep its split.
Normalized duplicate prompts and conflicting targets are checked. Names are
partitioned through the shared entity pool before rendering; **task families,
sentence patterns, days, feelings and other common vocabulary are shared**.
Test performance therefore measures this controlled distribution with held-out
names/plans, not unseen reasoning tasks or unrestricted chat. Keep a separate
human-written test set for broader assessment.

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
