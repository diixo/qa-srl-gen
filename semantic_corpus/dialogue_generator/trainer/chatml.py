"""ChatML formatting and final-answer loss boundaries, without ML dependencies."""

from dataclasses import dataclass
from typing import Protocol

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
PAD = "<|pad|>"
FORMAT_VERSION = "gpt2-chatml-v1"
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    prompt_ids: list[int]


def validate_messages(messages: object) -> list[dict[str, str]]:
    """Accept the generator's system/user/assistant contract, not arbitrary JSON."""
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("Expected exactly system, user, assistant messages")
    for message, role in zip(messages, ("system", "user", "assistant")):
        if not isinstance(message, dict) or message.get("role") != role:
            raise ValueError("Expected exactly system, user, assistant messages")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Empty or invalid {role} content")
        if any(marker in content for marker in (IM_START, IM_END, PAD, "<|endoftext|>")):
            raise ValueError("Message content contains a reserved control token")
    return messages


def prompt_text(messages: list[dict[str, str]]) -> str:
    validate_messages(messages)
    return "".join(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages[:-1]) + \
        f"{IM_START}assistant\n"


def encode_example(messages: list[dict[str, str]], tokenizer: Tokenizer, max_length: int) -> EncodedExample:
    """Supervise only the final answer and its end marker; never truncate."""
    prompt = prompt_text(messages)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(prompt + messages[-1]["content"] + IM_END, add_special_tokens=False)
    if ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer merges across the assistant boundary; check leading answer whitespace")
    if len(ids) > max_length:
        raise ValueError(f"Example has {len(ids)} tokens, exceeds max_length={max_length}; no truncation is allowed")
    if len(ids) <= len(prompt_ids):
        raise ValueError("Example has no assistant target tokens")
    return EncodedExample(ids, [-100] * len(prompt_ids) + ids[len(prompt_ids):], prompt_ids)
