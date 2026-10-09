"""Trainer input checks and ChatML loss scope without optional ML packages."""

import json
import re

import pytest

from semantic_corpus.dialogue_generator.trainer.chatml import (
    IM_START, IM_END, PAD, encode_example, prompt_text,
)
from semantic_corpus.dialogue_generator.trainer.data import check_disjoint, load_rows


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        special = {IM_START: 1000, IM_END: 1001, PAD: 1002}
        return [special.get(token, ord(token) if len(token) == 1 else -1)
                for token in re.findall(r'<\|[^>]+\|>|[\s\S]', text)]


def row(question='Where am I?', answer='You are in the park.', **metadata):
    return {'messages': [{'role': 'system', 'content': 'Reply briefly.'},
                         {'role': 'user', 'content': question},
                         {'role': 'assistant', 'content': answer}], **metadata}


def save(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text(json.dumps(rows), encoding='utf-8')
    return path


def test_chatml_masks_the_context_and_supervises_answer_plus_end_token():
    messages = row()['messages']
    tokenizer = CharacterTokenizer()
    encoded = encode_example(messages, tokenizer, 512)
    assert prompt_text(messages) == (
        '<|im_start|>system\nReply briefly.<|im_end|>\n'
        '<|im_start|>user\nWhere am I?<|im_end|>\n<|im_start|>assistant\n'
    )
    assert 'You are in the park.' not in prompt_text(messages)
    boundary = len(encoded.prompt_ids)
    assert encoded.input_ids[:boundary] == encoded.prompt_ids
    assert encoded.labels[:boundary] == [-100] * boundary
    assert encoded.labels[boundary:] == tokenizer.encode('You are in the park.' + IM_END)
    assert encoded.labels[-1] == 1001
    with pytest.raises(ValueError, match='no truncation'):
        encode_example(messages, tokenizer, len(encoded.input_ids) - 1)


@pytest.mark.parametrize('bad', [[], {}, [{'role': 'user', 'content': 'Hi'}],
                                row(answer='')['messages'], row(question=IM_START)['messages']])
def test_malformed_messages_and_embedded_chatml_tokens_are_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        load_rows(save(tmp_path, 'bad.json', [{'messages': bad}]))


def test_optional_metadata_does_not_change_training_content(tmp_path):
    plain = load_rows(save(tmp_path, 'plain.json', [row()]))
    full = load_rows(save(tmp_path, 'full.json', [row(id='ignored', task='recall', scenario_key='plan')]))
    assert plain[0].messages == full[0].messages
    assert plain[0].prompt_key == full[0].prompt_key
    assert plain[0].task == 'unlabeled' and full[0].task == 'recall'
    with pytest.raises(ValueError, match='Prompt leakage'):
        check_disjoint(plain, full)


def test_duplicates_conflicts_and_scenario_leakage_are_refused(tmp_path):
    for answer in ('You are in the park.', 'You are in the office.'):
        with pytest.raises(ValueError, match='Repeated normalized'):
            load_rows(save(tmp_path, 'duplicate.json', [row(), row(question='WHERE am I!', answer=answer)]))
    first = load_rows(save(tmp_path, 'train.json', [row(scenario_key='same-plan')]))
    second = load_rows(save(tmp_path, 'validation.json', [row(question='Recall my location.', scenario_key='same-plan')]))
    with pytest.raises(ValueError, match='Scenario leakage'):
        check_disjoint(first, second)
