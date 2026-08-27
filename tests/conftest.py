"""Shared test fixtures.

The default fixtures are constructed locally from a config with random weights.
No test in the fast suite touches the Hugging Face Hub, so it runs offline, in
CI, and in seconds. That is deliberate: a test suite that needs a 240MB download
is a test suite people stop running.

Tests that genuinely need a trained model and a real tokenizer are marked
`@pytest.mark.integration` and skipped unless `--run-integration` is passed. They
exist because some of this package's invariants -- decode/re-encode exactness in
particular -- cannot be tested against a toy tokenizer without the toy tokenizer
becoming the thing under test.
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

# Embedding rows in the toy models. The tokenizer deliberately covers fewer ids
# than this, mirroring real checkpoints: t5-small has 32128 embedding rows for a
# 32100-token vocabulary, and the attack must never substitute in the difference.
VOCAB = 64
PAD, EOS = 0, 1
SPECIAL_IDS = [0, 1, 2, 3]
FIRST_TOKEN_ID = 4
# 26 lowercase + space + 26 uppercase + 5 punctuation = 58 characters.
ALPHABET = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?-"
TOKENIZER_VOCAB = FIRST_TOKEN_ID + len(ALPHABET)  # 62; ids 62 and 63 have no token


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests that download real Hugging Face models",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: needs a real Hugging Face model download"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="needs --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


class ToyTokenizer:
    """Character tokenizer implementing the slice of the HF API this package uses.

    Unlike a throwaway stub, this one is a genuine bijection on its alphabet:
    `decode` is the exact inverse of `__call__` on ids in `[FIRST_TOKEN_ID,
    TOKENIZER_VOCAB)`. That matters because the attacker enforces a
    decode/re-encode round-trip invariant, and a tokenizer whose decode is not an
    inverse would make every candidate look invalid -- the tests would pass or
    fail for reasons that have nothing to do with the code under test.

    Two properties are modelled on purpose because the attack has to cope with
    them in the real world:

    * ids 0-3 are special and are stripped by `skip_special_tokens=True`;
    * ids 62-63 exist in the embedding table but map to no character, exactly as
      t5-small's 28 surplus embedding rows do.

    Both must be excluded from the candidate set, and `test_attacker.py` checks
    that they are.
    """

    all_special_ids = SPECIAL_IDS

    def __len__(self) -> int:
        return TOKENIZER_VOCAB

    def __call__(self, text: str, return_tensors: str | None = None) -> BatchEncoding:
        ids = [FIRST_TOKEN_ID + ALPHABET.index(c) for c in text if c in ALPHABET]
        ids = ids or [FIRST_TOKEN_ID]
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
        return "".join(
            ALPHABET[i - FIRST_TOKEN_ID]
            for i in out
            if FIRST_TOKEN_ID <= i < TOKENIZER_VOCAB
        )


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
