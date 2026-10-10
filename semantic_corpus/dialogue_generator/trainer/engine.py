"""Optional pretrained GPT-2 SFT for one explicitly selected dialogue view."""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time

# This command uses PyTorch exclusively, even if an unrelated TensorFlow
# installation is present in the user's environment.
os.environ['USE_TORCH'] = '1'
os.environ['USE_TF'] = '0'

import torch
from torch.nn import functional as F
import transformers
from transformers import AutoConfig, GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments, set_seed

from .chatml import CHAT_TEMPLATE, FORMAT_VERSION, IM_START, IM_END, PAD, encode_example
from .data import check_disjoint, file_sha256, load_rows, validate_view
from .metrics import AnswerMetrics, ExtractionMetrics


@dataclass(frozen=True)
class TrainSettings:
    data: Path
    out: Path
    model: str = 'gpt2'
    cache_dir: Path = Path('artifacts/models/huggingface')
    local_files_only: bool = False
    epochs: int = 3
    batch_size: int = 2
    accumulation: int = 8
    learning_rate: float = 5e-5
    max_length: int = 512
    seed: int = 42
    max_steps: int = -1
    device: str = 'auto'
    precision: str = 'auto'
    threads: int = 4
    log_every: int = 10
    save_only_model: bool = False
    view: str = 'memory'

    def validate(self):
        if self.view not in ('memory', 'history', 'extraction'):
            raise ValueError('view must be memory, history or extraction')
        for field in ('epochs', 'batch_size', 'accumulation', 'max_length', 'threads', 'log_every'):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError(f'{field} must be a positive integer')
        if type(self.max_steps) is not int or self.max_steps == 0 or self.max_steps < -1:
            raise ValueError('max_steps must be -1 (all epochs) or positive')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite')
        if int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise ValueError('This pilot trainer supports a single process/device')


def _write_json(path, value):
    pending = path.with_suffix(path.suffix + '.partial')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + '\n',
                       encoding='utf-8')
    pending.replace(path)


def _device(name):
    if name not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is not available; choose --device cpu')
    return torch.device('cuda' if name == 'auto' and torch.cuda.is_available() else 'cpu' if name == 'auto' else name)


def _precision(device, choice):
    if choice not in ('auto', 'float32', 'float16', 'bfloat16'):
        raise ValueError('Unsupported precision')
    if choice == 'auto':
        choice = ('bfloat16' if torch.cuda.is_bf16_supported() else 'float16') if device.type == 'cuda' else 'float32'
    if device.type == 'cpu' and choice != 'float32':
        raise ValueError('This trainer uses float32 on CPU')
    if choice == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This device does not support bfloat16')
    return choice


def configure_chatml(tokenizer):
    tokenizer.add_special_tokens({'additional_special_tokens': [IM_START], 'eos_token': IM_END, 'pad_token': PAD})
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.padding_side = 'right'
    for marker in (IM_START, IM_END, PAD):
        if tokenizer.encode(marker, add_special_tokens=False) != [tokenizer.convert_tokens_to_ids(marker)]:
            raise ValueError(f'ChatML marker is not a single token: {marker}')


class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        row = self.examples[index]
        return {'input_ids': row.input_ids, 'labels': row.labels}


class AnswerCollator:
    """Right-pad inputs; keep prompt and padding labels at -100."""
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        length = max(len(row['input_ids']) for row in rows)
        inputs = torch.full((len(rows), length), self.pad_id, dtype=torch.long)
        labels = torch.full_like(inputs, -100)
        mask = torch.zeros_like(inputs)
        for index, row in enumerate(rows):
            size = len(row['input_ids'])
            inputs[index, :size] = torch.tensor(row['input_ids'])
            labels[index, :size] = torch.tensor(row['labels'])
            mask[index, :size] = 1
        return {'input_ids': inputs, 'labels': labels, 'attention_mask': mask}


def assistant_loss(outputs, labels, num_items_in_batch=None):
    # Trainer supplies the total supervised tokens in the accumulated batch.
    # Position zero is always masked, so its token count matches the causal shift.
    targets = labels[:, 1:]
    loss = F.cross_entropy(outputs.logits[:, :-1].float().reshape(-1, outputs.logits.shape[-1]),
                           targets.reshape(-1), ignore_index=-100, reduction='sum')
    count = (targets != -100).sum() if num_items_in_batch is None else num_items_in_batch
    return loss / count


def _encode(rows, tokenizer, max_length, split, view='memory'):
    validate_view(rows, view, split)
    encoded = []
    for index, row in enumerate(rows, 1):
        try:
            encoded.append(encode_example(row.messages, tokenizer, max_length))
        except ValueError as error:
            raise ValueError(f'{split}, record {index}: {error}') from error
    return encoded


def train(settings: TrainSettings) -> dict:
    settings.validate()
    if settings.out.exists():
        raise FileExistsError(f'Choose a new training directory: {settings.out}')

    device = _device(settings.device)
    precision = _precision(device, settings.precision)
    torch.set_num_threads(settings.threads)
    set_seed(settings.seed)

    train_path, val_path = settings.data / 'train.json', settings.data / 'validation.json'
    rows, val_rows = load_rows(train_path), load_rows(val_path)

    check_disjoint(rows, val_rows)
    validate_view(rows, settings.view, 'train')
    validate_view(val_rows, settings.view, 'validation')

    loading = {'cache_dir': str(settings.cache_dir), 'local_files_only': settings.local_files_only}
    
    config = AutoConfig.from_pretrained(settings.model, **loading)
    
    if config.model_type != 'gpt2':
        raise ValueError('The trainer supports GPT-2 checkpoints only')
    if settings.max_length > config.n_positions:
        raise ValueError(f'max_length exceeds the GPT-2 context limit ({config.n_positions})')
    
    tokenizer = GPT2TokenizerFast.from_pretrained(settings.model, **loading)
    
    configure_chatml(tokenizer)
    
    examples = _encode(rows, tokenizer, settings.max_length, 'train', settings.view)
    validation = _encode(val_rows, tokenizer, settings.max_length, 'validation', settings.view)
    print(f'Validated {len(rows)} train / {len(val_rows)} validation records. Loading pretrained {settings.model}.', flush=True)
    
    model = GPT2LMHeadModel.from_pretrained(settings.model, **loading)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = False
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.suppress_tokens = list({
        tokenizer.pad_token_id,
        tokenizer.convert_tokens_to_ids(IM_START),
        tokenizer.convert_tokens_to_ids('<|endoftext|>')
        })
    settings.out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'mode': 'sft', 'initialization': 'pretrained', 'view': settings.view,
              'trainer': 'transformers.Trainer', 'format': FORMAT_VERSION,
              'loss_scope': 'final assistant answer and im_end only',
              'settings': asdict(settings), 'device': str(device), 'precision': precision,
              'versions': {'torch': torch.__version__, 'transformers': transformers.__version__},
              'parameters': model.num_parameters(), 'vocab_size': len(tokenizer),
              'source_files': {'train.json': file_sha256(train_path), 'validation.json': file_sha256(val_path)},
              'records': {'train': len(rows), 'validation': len(val_rows)},
              'max_record_tokens': max(len(e.input_ids) for e in examples + validation),
              'test_used_during_training': False}
    _write_json(settings.out / 'run.json', report)
    _write_json(settings.out / 'split_keys.json', {
        split: {'prompts': [row.prompt_key for row in data],
                'contrast_groups': sorted({row.contrast_group for row in data if row.contrast_group}),
                'scenarios': sorted({row.scenario_key for row in data if row.scenario_key})}
        for split, data in (('train', rows), ('validation', val_rows))})
    trainer = None
    started = time.monotonic()
    try:
        arguments = TrainingArguments(
            output_dir=str(settings.out / 'checkpoints'),
            num_train_epochs=settings.epochs,
            max_steps=settings.max_steps,
            per_device_train_batch_size=settings.batch_size,
            per_device_eval_batch_size=settings.batch_size,
            gradient_accumulation_steps=settings.accumulation,
            learning_rate=settings.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.05,
            seed=settings.seed,
            data_seed=settings.seed,
            optim='adamw_torch',
            use_cpu=device.type == 'cpu',
            bf16=precision == 'bfloat16',
            fp16=precision == 'float16',
            eval_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2,
            save_only_model=settings.save_only_model,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            prediction_loss_only=True,
            report_to=[],
            push_to_hub=False,
            logging_steps=settings.log_every,
            logging_first_step=True,
            disable_tqdm=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=device.type == 'cuda',
        )
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=ChatDataset(examples),
            eval_dataset=ChatDataset(validation),
            data_collator=AnswerCollator(tokenizer.pad_token_id),
            processing_class=tokenizer,
            compute_loss_func=assistant_loss
        )
        report['initial_validation'] = trainer.evaluate()
        _write_json(settings.out / 'run.json', report)
        result = trainer.train()
        # Trainer reloads its best validation checkpoint before this export.
        model.config.use_cache = True
        trainer.save_model(str(settings.out / 'model'))
        tokenizer.save_pretrained(settings.out / 'model')
        trainer.save_state()
        report.update(
            status='complete',
            steps=trainer.state.global_step,
            best_checkpoint=trainer.state.best_model_checkpoint,
            best_validation_loss=trainer.state.best_metric,
            train_metrics=result.metrics,
            final_validation=trainer.evaluate()
        )
        report['tokenizer_files'] = {name: file_sha256(settings.out / 'model' / name)
                                     for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')}
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        report.update(history=trainer.state.log_history if trainer is not None else [],
                      elapsed_seconds=round(time.monotonic() - started, 3))
        _write_json(settings.out / 'run.json', report)
    return report


def load_run(run: Path, device):
    report = json.loads((run / 'run.json').read_text(encoding='utf-8'))
    if report.get('status') != 'complete' or report.get('format') != FORMAT_VERSION:
        raise ValueError('Expected a completed GPT-2 ChatML training run')
    for name, digest in report['tokenizer_files'].items():
        if file_sha256(run / 'model' / name) != digest:
            raise ValueError('Saved tokenizer contract has changed')
    tokenizer = GPT2TokenizerFast.from_pretrained(run / 'model', local_files_only=True)
    if tokenizer.chat_template != CHAT_TEMPLATE or tokenizer.eos_token != IM_END or tokenizer.pad_token != PAD:
        raise ValueError('Saved ChatML tokenizer contract is invalid')
    model = GPT2LMHeadModel.from_pretrained(run / 'model', local_files_only=True).to(device).eval()
    eos_ids = model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    if model.config.vocab_size != len(tokenizer) or eos_ids != [tokenizer.eos_token_id]:
        raise ValueError('Saved model and tokenizer do not match')
    return model, tokenizer


@torch.inference_mode()
def evaluate(run: Path, data: Path, out: Path, *, split='test', device='auto', max_new_tokens=64,
             batch_size=2, limit=0, threads=4) -> dict:
    if split not in ('validation', 'test') or min(max_new_tokens, batch_size, threads) <= 0 or limit < 0:
        raise ValueError('Invalid evaluation settings')
    if out.exists():
        raise FileExistsError(f'Choose a new evaluation directory: {out}')
    torch.set_num_threads(threads)
    device = _device(device)
    run_report = json.loads((run / 'run.json').read_text(encoding='utf-8'))
    view = run_report.get('view', 'memory')
    model, tokenizer = load_run(run, device)
    path = data / f'{split}.json'
    rows = load_rows(path)
    keys = json.loads((run / 'split_keys.json').read_text(encoding='utf-8'))
    for name in (('train', 'validation') if split == 'test' else ('train',)):
        if set(keys[name].get('contrast_groups', [])) & {row.contrast_group for row in rows if row.contrast_group}:
            raise ValueError(f'Evaluation contrast group leakage from {name}')
        if set(keys[name]['prompts']) & {row.prompt_key for row in rows} or \
                set(keys[name]['scenarios']) & {row.scenario_key for row in rows if row.scenario_key}:
            raise ValueError(f'Evaluation leakage from {name}')
    all_count = len(rows)
    if limit:
        rows = rows[:limit]
    examples = _encode(rows, tokenizer, model.config.n_positions, split, view)
    if any(len(row.prompt_ids) + max_new_tokens > model.config.n_positions for row in examples):
        raise ValueError('Prompt plus max_new_tokens exceeds model context; reduce --max-new-tokens')
    out.mkdir(parents=True, exist_ok=False)
    collator = AnswerCollator(tokenizer.pad_token_id)
    loss_sum, target_count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        batch = {key: value.to(device) for key, value in collator(
            [ChatDataset(examples)[i] for i in range(start, min(start + batch_size, len(examples)))]).items()}
        labels = batch.pop('labels')
        count = (labels[:, 1:] != -100).sum().item()
        loss_sum += assistant_loss(model(**batch), labels).item() * count
        target_count += count
    totals = {'count': 0, 'exact_matches': 0, 'ended': 0}
    by_task = {}
    diagnostics = ExtractionMetrics() if view == 'extraction' else AnswerMetrics()
    with (out / 'predictions.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for index, (row, encoded) in enumerate(zip(rows, examples), 1):
            # Only prompt tokens enter generate(); the reference is never supplied.
            inputs = torch.tensor([encoded.prompt_ids], device=device)
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                    max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
            answer_ids = output[0, inputs.shape[1]:].tolist()
            ended = bool(answer_ids and answer_ids[-1] == tokenizer.eos_token_id)
            prediction = tokenizer.decode(answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            target = row.messages[-1]['content']
            if view == 'extraction':
                diagnostics.add(target, prediction, prompt=row.messages[1]['content'])
            else:
                diagnostics.add(target, prediction, task=row.task, condition=row.condition, group=row.contrast_group)
            exact = prediction.strip() == target.strip()
            task = by_task.setdefault(row.task, {'count': 0, 'exact_matches': 0, 'ended': 0})
            for counts in (totals, task):
                counts['count'] += 1
                counts['exact_matches'] += int(exact)
                counts['ended'] += int(ended)
            stream.write(json.dumps({'record': index, 'task': row.task, 'prompt': row.messages[1]['content'],
                                     'reference': target, 'prediction': prediction, 'exact_match': exact,
                                     'ended': ended, 'contrast_group': row.contrast_group,
                                     'condition': row.condition}, ensure_ascii=False) + '\n')
            if index == 1 or index % 20 == 0 or index == len(rows):
                print(f'Evaluated {index}/{len(rows)}', flush=True)
    loss = loss_sum / target_count
    report = {'view': view, 'split': split, 'source_sha256': file_sha256(path), 'available_records': all_count,
              'model': str(run / 'model'), 'decoding': 'greedy', 'max_new_tokens': max_new_tokens,
              'answer_loss': loss, 'perplexity': math.exp(loss) if loss < 700 else None,
              **totals, 'exact_match_rate': totals['exact_matches'] / totals['count'], 'by_task': by_task,
              **diagnostics.report(),
              'note': 'Exact match is a strict reference comparison; review predictions for semantic quality.'}
    _write_json(out / 'evaluation.json', report)
    return report
