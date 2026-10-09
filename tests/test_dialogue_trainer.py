"""Optional offline tests of the real Transformers Trainer integration."""

import json
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
os.environ['USE_TORCH'] = '1'
os.environ['USE_TF'] = '0'
pytest.importorskip('transformers')
pytest.importorskip('accelerate')
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from semantic_corpus.dialogue_generator.trainer.chatml import IM_END, IM_START, encode_example, prompt_text
from semantic_corpus.dialogue_generator.trainer.engine import (
    AnswerCollator, TrainSettings, assistant_loss, configure_chatml, evaluate, load_run, train,
)


def _row(place, view='memory'):
    if view == 'extraction':
        from semantic_corpus.dialogue_generator.extraction_text import render_prompt, render_update
        from semantic_corpus.dialogue_generator.models import Memory, Turn, Delta, EventClaim, Evidence, Transition, apply_delta
        from semantic_corpus.dialogue_generator.events import Event, EventArgument
        user = Turn('u1', 'user', f'I need to visit the {place} today.')
        event = Event('visit', (EventArgument('destination', f'the {place}', 'LOCATION'),
                                EventArgument('visitor', 'user', 'PERSON')), operators=('need',))
        delta = Delta(event_add=(EventClaim('e1', event, (Evidence(user.id, user.text),)),))
        transition = Transition(user, Memory(), delta, apply_delta(Memory(), delta, (user,)), None)
        return {'messages': [{'role': 'system', 'content': 'Update memory.'},
                             {'role': 'user', 'content': render_prompt(Memory(), user)},
                             {'role': 'assistant', 'content': render_update(transition)}], 'task': 'recall'}
    if view == 'history':
        return {'messages': [{'role': 'system', 'content': 'Reply briefly.'},
                             {'role': 'user', 'content': f'Conversation so far:\nUser: I am in the {place}.\nLatest user message: Where am I?'},
                             {'role': 'assistant', 'content': f'You are in the {place}.'}], 'task': 'recall'}
    return {'messages': [{'role': 'system', 'content': 'Reply briefly.'},
                         {'role': 'user', 'content': f'Known facts:\n- You are in the {place}.\nWhere am I?'},
                         {'role': 'assistant', 'content': f'You are in the {place}.'}], 'task': 'recall'}


@pytest.fixture
def fixture_model(tmp_path):
    source = tmp_path / 'initial'
    source.mkdir()
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(['Reply briefly. Known facts: You are in the park. Where am I?'],
                            vocab_size=300, special_tokens=['<|endoftext|>'])
    bpe.save_model(str(source))
    tokenizer = GPT2TokenizerFast(vocab_file=str(source / 'vocab.json'), merges_file=str(source / 'merges.txt'))
    tokenizer.save_pretrained(source)
    torch.manual_seed(3)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_positions=512, n_embd=32, n_layer=1, n_head=2,
                                      bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id))
    model.save_pretrained(source)
    data = tmp_path / 'memory'
    data.mkdir()
    for split, places in {'train': ['park', 'office', 'kitchen'], 'validation': ['garden'], 'test': ['cafe']}.items():
        (data / f'{split}.json').write_text(json.dumps([_row(p) for p in places]), encoding='utf-8')
    return source, data


def test_special_tokens_template_padding_and_assistant_loss(fixture_model):
    tokenizer = GPT2TokenizerFast.from_pretrained(fixture_model[0], local_files_only=True)
    configure_chatml(tokenizer)
    messages = _row('park')['messages']
    assert tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True) == prompt_text(messages)
    first = encode_example(messages, tokenizer, 256)
    second = encode_example(_row('very large supermarket')['messages'], tokenizer, 256)
    batch = AnswerCollator(tokenizer.pad_token_id)([
        {'input_ids': first.input_ids, 'labels': first.labels},
        {'input_ids': second.input_ids, 'labels': second.labels},
    ])
    assert torch.all(batch['labels'][0, :len(first.prompt_ids)] == -100)
    assert torch.all(batch['labels'][0, len(first.input_ids):] == -100)
    assert torch.all(batch['attention_mask'][0, len(first.input_ids):] == 0)
    assert first.labels[-1] == tokenizer.eos_token_id
    logits = torch.randn(2, batch['labels'].shape[1], len(tokenizer), requires_grad=True)
    combined = assistant_loss(SimpleNamespace(logits=logits), batch['labels'])
    count = (batch['labels'][:, 1:] != -100).sum()
    accumulated = sum(assistant_loss(SimpleNamespace(logits=logits[i:i + 1]), batch['labels'][i:i + 1], count)
                      for i in range(2))
    assert torch.allclose(combined, accumulated)
    combined.backward()
    assert torch.count_nonzero(logits.grad[0, :len(first.prompt_ids) - 1]) == 0
    assert torch.count_nonzero(logits.grad[0, len(first.prompt_ids) - 1]) > 0
    assert torch.count_nonzero(logits.grad[0, len(first.input_ids) - 1:]) == 0


@pytest.mark.parametrize('view', ['memory', 'history', 'extraction'])
def test_real_trainer_roundtrip_and_held_out_generation(tmp_path, fixture_model, monkeypatch, view):
    source, data = fixture_model
    for split, places in {'train': ['park', 'office', 'kitchen'], 'validation': ['garden'], 'test': ['cafe']}.items():
        (data / f'{split}.json').write_text(json.dumps([_row(p, view) for p in places]), encoding='utf-8')
    # Training must not read or evaluate test.json.
    test_path = data / 'test.json'
    held_out = test_path.read_text(encoding='utf-8')
    test_path.write_text('not JSON yet', encoding='utf-8')
    settings = TrainSettings(data=data, out=tmp_path / 'run', view=view, model=str(source), local_files_only=True,
                             max_length=512, max_steps=2, batch_size=1, accumulation=2, threads=1,
                             device='cpu', learning_rate=1e-3, save_only_model=True)
    report = train(settings)
    assert report['status'] == 'complete' and report['steps'] == 2
    assert report['mode'] == 'sft' and report['trainer'] == 'transformers.Trainer'
    assert report['view'] == view
    assert report['test_used_during_training'] is False
    assert report['best_validation_loss'] > 0
    model, tokenizer = load_run(settings.out, torch.device('cpu'))
    original = GPT2LMHeadModel.from_pretrained(source, local_files_only=True)
    assert not torch.equal(model.transformer.wte.weight[:len(original.transformer.wte.weight)], original.transformer.wte.weight)
    assert model.config.vocab_size == len(tokenizer)
    assert model.generation_config.eos_token_id == [tokenizer.convert_tokens_to_ids(IM_END)]
    assert model.config.use_cache is True
    test_path.write_text(held_out, encoding='utf-8')
    calls = []
    generate = GPT2LMHeadModel.generate

    def checked_generate(self, *args, **kwargs):
        expected = tokenizer.encode(prompt_text(_row('cafe', view)['messages']), add_special_tokens=False)
        assert kwargs['input_ids'].tolist() == [expected]
        calls.append(1)
        return generate(self, *args, **kwargs)

    monkeypatch.setattr(GPT2LMHeadModel, 'generate', checked_generate)
    result = evaluate(settings.out, data, tmp_path / 'evaluation', device='cpu', max_new_tokens=8, threads=1)
    assert result['count'] == 1 and calls == [1]
    assert result['by_task']['recall']['count'] == 1
    assert result['answer_loss'] > 0
    prediction = json.loads((tmp_path / 'evaluation/predictions.jsonl').read_text(encoding='utf-8'))
    assert prediction['reference'] == _row('cafe', view)['messages'][-1]['content']
    assert ('extraction' in result) == (view == 'extraction')
    # Cross-split checks still work for datasets without outer metadata.
    test_path.write_bytes((data / 'train.json').read_bytes())
    with pytest.raises(ValueError, match='Evaluation leakage'):
        evaluate(settings.out, data, tmp_path / 'leak', device='cpu')
    assert not (tmp_path / 'leak').exists()
    with pytest.raises(FileExistsError):
        train(settings)
    test_path.write_text(json.dumps([_row('new place', 'memory' if view != 'memory' else 'history')]), encoding='utf-8')
    with pytest.raises(ValueError, match='plain-text'):
        evaluate(settings.out, data, tmp_path / 'wrong-view', device='cpu')
    assert not (tmp_path / 'wrong-view').exists()


def test_bad_or_overlong_memory_refuses_training_before_loading_weights(tmp_path, fixture_model):
    source, data = fixture_model
    settings = TrainSettings(data=data, out=tmp_path / 'bad', model=str(source), local_files_only=True,
                             device='cpu', max_length=8)
    with pytest.raises(ValueError, match='no truncation'):
        train(settings)
    assert not settings.out.exists()
    first = _row('park')
    first['messages'][1]['content'] = '{"facts": []}'
    (data / 'train.json').write_text(json.dumps([first]), encoding='utf-8')
    with pytest.raises(ValueError, match='plain-text Memory'):
        train(settings)
    assert not settings.out.exists()


def test_trainer_initialization_failure_marks_run_failed(tmp_path, fixture_model, monkeypatch):
    source, data = fixture_model
    settings = TrainSettings(data=data, out=tmp_path / 'failed', model=str(source), local_files_only=True,
                             device='cpu', max_length=256)

    def fail_initialization(**kwargs):
        raise RuntimeError('Trainer initialization failed')

    monkeypatch.setattr('semantic_corpus.dialogue_generator.trainer.engine.Trainer', fail_initialization)
    with pytest.raises(RuntimeError, match='initialization failed'):
        train(settings)
    report = json.loads((settings.out / 'run.json').read_text(encoding='utf-8'))
    assert report['status'] == 'failed'
    assert report['error'] == 'Trainer initialization failed'
    assert report['history'] == []


@pytest.mark.parametrize('view', ['memory', 'extraction'])
def test_comparison_preserves_view_specific_metrics(tmp_path, monkeypatch, view):
    import sys
    from semantic_corpus.dialogue_generator.trainer import compare
    from semantic_corpus.dialogue_generator.trainer.data import file_sha256

    run, data, output = (tmp_path / name for name in ('run', 'data', 'comparison'))
    run.mkdir()
    data.mkdir()
    (run / 'run.json').write_text(json.dumps({'view': view}), encoding='utf-8')
    (data / 'test.json').write_text('[]', encoding='utf-8')
    def evaluate_stub(run_path, data_path, out_path, **kwargs):
        assert kwargs['max_new_tokens'] == 128
        report = {'count': 1, 'exact_matches': 1, 'exact_match_rate': 1.0, 'answer_loss': 0.2,
                  'source_sha256': file_sha256(data_path / 'test.json')}
        if view == 'extraction':
            report['extraction'] = {'valid_predictions': 1, 'state_matches': 1}
        else:
            report.update(unknown_count=0, unknown_false_yes=0, contrast_groups={'all_correct': 0, 'complete': 0})
        return report
    monkeypatch.setattr('semantic_corpus.dialogue_generator.trainer.engine.evaluate', evaluate_stub)
    monkeypatch.setattr(sys, 'argv', ['compare', '--run', f'model={run}', '--dataset', f'data={data}',
                                    '--out', str(output), '--max-new-tokens', '128'])
    compare.main()
    rows = json.loads((output / 'comparison.json').read_text(encoding='utf-8'))
    assert ('extraction' in rows[0]) == (view == 'extraction')
    assert ('Matching states' in (output / 'comparison.md').read_text(encoding='utf-8')) == (view == 'extraction')
