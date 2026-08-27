"""Shared test fixtures.

Everything here is constructed locally from a config with random weights. No test
touches the Hugging Face Hub, so the suite runs offline, in CI, and in seconds.
That is deliberate: a test suite that needs a 240MB download is a test suite people
stop running.
"""

from __future__ import annotations

import pytest
import torch
from transformers import (
    BatchEncoding,
    GPT2Config,
    GPT2LMHeadModel,
    T5Config,
    T5ForConditionalGeneration,
)

VOCAB = 64
PAD, EOS = 0, 1


class ToyTokenizer:
    """Character-code tokenizer implementing the slice of the HF API we use.

    Maps each character to `ord(c) % (VOCAB - 4) + 4`, keeping ids clear of the
    special range. Not reversible in a linguistically meaningful way, which does
    not matter: these tests exercise control flow and tensor bookkeeping, not
    language quality.
    """

    all_special_ids = [PAD, EOS]

    def __call__(self, text: str, return_tensors: str | None = None) -> BatchEncoding:
        ids = [ord(c) % (VOCAB - 4) + 4 for c in text[:16]] or [4]
        ids.append(EOS)
        data = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }
        return BatchEncoding(data, tensor_type=return_tensors)

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        out = [int(i) for i in ids]
        if skip_special_tokens:
            out = [i for i in out if i not in self.all_special_ids]
        return "".join(chr(65 + (i % 26)) for i in out)


@pytest.fixture
def tokenizer() -> ToyTokenizer:
    return ToyTokenizer()


@pytest.fixture
def seq2seq_model() -> T5ForConditionalGeneration:
    torch.manual_seed(0)
    cfg = T5Config(
        vocab_size=VOCAB,
        d_model=32,
        d_ff=64,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        d_kv=16,
        decoder_start_token_id=PAD,
        eos_token_id=EOS,
        pad_token_id=PAD,
    )
    return T5ForConditionalGeneration(cfg).eval()


@pytest.fixture
def causal_model() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    cfg = GPT2Config(
        vocab_size=VOCAB,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_positions=128,
        eos_token_id=EOS,
        pad_token_id=PAD,
        bos_token_id=PAD,
    )
    return GPT2LMHeadModel(cfg).eval()
