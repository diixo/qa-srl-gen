"""CLI only: python -m semantic_corpus.dialogue_generator.trainer --help."""

import argparse
from pathlib import Path
import sys


def _positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('Must be a positive integer')
    return number


def build_parser():
    parser = argparse.ArgumentParser(description='Pretrained GPT-2 + ChatML SFT on text dialogue datasets.')
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train', help='supervised fine-tuning with transformers.Trainer')
    training.add_argument('--data', type=Path, required=True, help='directory with train.json and validation.json for one view')
    training.add_argument('--view', choices=['memory', 'history', 'extraction'], default='memory', help='training task; default: memory')
    training.add_argument('--out', type=Path, required=True, help='new directory for checkpoints and the final model')
    training.add_argument('--mode', choices=['sft'], default='sft')
    training.add_argument('--model', default='gpt2', help='pretrained GPT-2 identifier or local model directory')
    training.add_argument('--cache-dir', type=Path, default=Path('artifacts/models/huggingface'))
    training.add_argument('--local-files-only', action='store_true', help='do not download model/tokenizer files')
    training.add_argument('--epochs', type=_positive, default=3)
    training.add_argument('--batch-size', type=_positive, default=2)
    training.add_argument('--accumulation', type=_positive, default=8, help='gradient accumulation microbatches')
    training.add_argument('--learning-rate', type=float, default=5e-5)
    training.add_argument('--max-length', type=_positive, default=512, help='reject longer examples; never truncate')
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--max-steps', type=_positive, default=-1, help='limit optimizer steps for a smoke test')
    training.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    training.add_argument('--precision', choices=['auto', 'float32', 'float16', 'bfloat16'], default='auto')
    training.add_argument('--threads', type=_positive, default=4)
    training.add_argument('--log-every', type=_positive, default=10)
    training.add_argument('--save-only-model', action='store_true', help='omit optimizer state in intermediate checkpoints')
    evaluation = commands.add_parser('evaluate', help='measure held-out answer loss and generate reference comparisons')
    evaluation.add_argument('--run', type=Path, required=True, help='completed training output directory')
    evaluation.add_argument('--data', type=Path, required=True, help='dataset directory matching the training run view')
    evaluation.add_argument('--out', type=Path, required=True, help='new directory for predictions and metrics')
    evaluation.add_argument('--split', choices=['validation', 'test'], default='test')
    evaluation.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    evaluation.add_argument('--max-new-tokens', type=_positive, default=64)
    evaluation.add_argument('--batch-size', type=_positive, default=2)
    evaluation.add_argument('--limit', type=_positive, default=0, help='evaluate only the first N records (default: all)')
    evaluation.add_argument('--threads', type=_positive, default=4)
    return parser


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    command = args.pop('command')
    try:
        from .engine import TrainSettings, evaluate, train
        if command == 'train':
            args.pop('mode')
            report = train(TrainSettings(**args))
            print(f"Training complete. Best validation loss: {report['best_validation_loss']:.4f}")
            print(f"Model: {args['out'] / 'model'}")
        else:
            report = evaluate(**args)
            print(f"{report['count']} examples; exact match {report['exact_match_rate']:.1%}; "
                  f"answer loss {report['answer_loss']:.4f}")
            print(f"Predictions: {args['out'] / 'predictions.jsonl'}")
    except ModuleNotFoundError as error:
        print(f'Optional training dependency is missing: {error.name}. '
              'Install with: python -m pip install -r requirements-training.txt', file=sys.stderr)
        return 2
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('Training interrupted; see the run directory for saved checkpoints.', file=sys.stderr)
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
