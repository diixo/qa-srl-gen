"""Optional offline GPT-2 length checks; no model, PyTorch or training engine."""

from hashlib import sha256
from pathlib import Path
import sys

from .trainer.chatml import CHAT_TEMPLATE, FORMAT_VERSION, IM_START, IM_END, PAD, encode_example


class GPT2TokenCounter:
    def __init__(self):
        try:
            from huggingface_hub import try_to_load_from_cache
            from tokenizers import AddedToken, Tokenizer, __version__
        except ImportError as error:
            raise ValueError('Token limits require optional tokenizer dependencies. Run: '
                             'python -m pip install -r requirements-tokenization.txt; '
                             'or leave Maximum example tokens blank.') from error
        project_cache = Path(__file__).resolve().parents[2] / 'artifacts/models/huggingface'
        source = None
        for cache in (project_cache, None):
            candidate = try_to_load_from_cache('gpt2', 'tokenizer.json', cache_dir=cache)
            if isinstance(candidate, str) and Path(candidate).is_file():
                source = Path(candidate)
                break
        if source is None:
            raise ValueError('GPT-2 tokenizer.json is not cached. Prepare it once with: '
                             'python -c "from huggingface_hub import hf_hub_download; '
                             "hf_hub_download('gpt2', 'tokenizer.json')\"; "
                             'or leave Maximum example tokens blank. Generation never downloads files.')
        self.backend = Tokenizer.from_file(str(source))
        self.backend.no_truncation()
        self.backend.no_padding()
        self.backend.add_special_tokens([AddedToken(marker, special=True, normalized=False)
                                         for marker in (IM_START, IM_END, PAD)])
        for marker in (IM_START, IM_END, PAD):
            if self.encode(marker) != [self.backend.token_to_id(marker)]:
                raise ValueError(f'ChatML marker is not a single token: {marker}')
        self.provenance = {
            'model': 'gpt2', 'chatml_format': FORMAT_VERSION, 'tokenizers_version': __version__,
            'source_sha256': sha256(source.read_bytes()).hexdigest(),
            'configured_tokenizer_sha256': sha256(self.backend.to_str().encode()).hexdigest(),
            'chat_template_sha256': sha256(CHAT_TEMPLATE.encode()).hexdigest(),
        }

    def encode(self, text, *, add_special_tokens=False):
        return self.backend.encode(text, add_special_tokens=add_special_tokens).ids

    def length(self, messages):
        # Reuse the trainer's exact input/target boundary and control-token checks.
        return len(encode_example(messages, self, sys.maxsize).input_ids)
