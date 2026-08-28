"""The model-agnostic seam, and the embedding-semantics invariant it protects.

The attack differentiates through `inputs_embeds`. That only measures the model
the caller passed if the `inputs_embeds` path reproduces the `input_ids` path.
For several encoder-decoder architectures it does not, unless the adapter applies
the architecture's own embedding scale -- so these are the tests that stop the
toolbox from silently optimising the wrong function.
"""

from __future__ import annotations

import math

import pytest
import torch
from transformers import MarianConfig, MarianMTModel

from llm_efficiency_attack.adapters import (
    CausalAdapter,
    EmbeddingSemanticsError,
    ModelAdapter,
    Seq2SeqAdapter,
    forbidden_token_ids,
    resolve_eos_ids,
)


def _marian(scale_embedding: bool) -> MarianMTModel:
    """A tiny Marian, the architecture family NMTSloth actually attacks.

    Real `Helsinki-NLP/opus-mt-*` checkpoints ship `scale_embedding: true` with
    `d_model: 512`, so their factor is `sqrt(512) ~= 22.6`. This uses a small
    `d_model` so the test stays fast, but the mechanism is identical.
    """
    torch.manual_seed(0)
    cfg = MarianConfig(
        vocab_size=64, d_model=32, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=64,
        pad_token_id=0, decoder_start_token_id=0, eos_token_id=1, bos_token_id=0,
        scale_embedding=scale_embedding,
    )
    return MarianMTModel(cfg).eval()


def test_adapter_is_chosen_by_config_not_isinstance(seq2seq_model, causal_model, tokenizer):
    assert isinstance(ModelAdapter.for_model(seq2seq_model, tokenizer), Seq2SeqAdapter)
    assert isinstance(ModelAdapter.for_model(causal_model, tokenizer), CausalAdapter)


def test_embedding_scale_defaults_to_one(seq2seq_model, causal_model, tokenizer):
    """T5 and GPT-2 do not scale, and the adapter must not invent a factor."""
    assert ModelAdapter.for_model(seq2seq_model, tokenizer).embedding_scale() == 1.0
    assert ModelAdapter.for_model(causal_model, tokenizer).embedding_scale() == 1.0


def test_marian_embedding_scale_is_read_from_the_architecture(tokenizer):
    model = _marian(scale_embedding=True)
    adapter = ModelAdapter.for_model(model, tokenizer)
    assert adapter.embedding_scale() == pytest.approx(math.sqrt(32))


def test_embed_values_reproduces_what_the_model_builds_internally(tokenizer):
    """`embed_values` must equal `embed_tokens(ids) * embed_scale`, exactly."""
    model = _marian(scale_embedding=True)
    adapter = ModelAdapter.for_model(model, tokenizer)
    ids = torch.tensor([[5, 6, 7, 8, 1]])

    expected = model.get_input_embeddings()(ids) * model.get_encoder().embed_scale
    assert torch.allclose(adapter.embed_values(ids), expected, atol=0)


def test_raw_table_lookup_would_have_been_wrong_for_marian(tokenizer):
    """The bug this guards against, demonstrated rather than asserted.

    Feeding the unscaled table lookup as `inputs_embeds` -- which is what a naive
    implementation does -- produces different logits than `input_ids`. The attack
    would have optimised that different function silently.
    """
    model = _marian(scale_embedding=True)
    ids = torch.tensor([[5, 6, 7, 8, 1]])
    mask = torch.ones_like(ids)
    decoder = torch.tensor([[0, 5, 6]])

    with torch.no_grad():
        from_ids = model(input_ids=ids, attention_mask=mask, decoder_input_ids=decoder).logits
        unscaled = model(
            inputs_embeds=model.get_input_embeddings()(ids),
            attention_mask=mask,
            decoder_input_ids=decoder,
        ).logits

    assert not torch.allclose(from_ids, unscaled, atol=1e-4), (
        "if these agreed, this architecture would not exercise the scaling path"
    )


def test_equivalence_check_passes_once_the_scale_is_applied(tokenizer):
    model = _marian(scale_embedding=True)
    adapter = ModelAdapter.for_model(model, tokenizer)
    ids = torch.tensor([[5, 6, 7, 8, 1]])

    deviation = adapter.check_embedding_equivalence(ids, torch.ones_like(ids))
    assert deviation == pytest.approx(0.0, abs=1e-5)


def test_equivalence_check_passes_for_t5_and_gpt2(seq2seq_model, causal_model, tokenizer):
    encoded = tokenizer("hello world", return_tensors="pt")
    for model in (seq2seq_model, causal_model):
        adapter = ModelAdapter.for_model(model, tokenizer)
        deviation = adapter.check_embedding_equivalence(
            encoded["input_ids"], encoded["attention_mask"]
        )
        assert deviation == pytest.approx(0.0, abs=1e-4)


def test_equivalence_check_raises_when_the_scale_is_wrong(tokenizer, monkeypatch):
    """A model whose embedding transform the adapter does not know must be refused.

    Simulated by making the adapter forget Marian's scale, which is exactly the
    state the code was in before this fix. Refusing loudly is the point: the
    alternative is returning an adversarial example optimised against a function
    the model does not compute.
    """
    model = _marian(scale_embedding=True)
    adapter = ModelAdapter.for_model(model, tokenizer)
    monkeypatch.setattr(type(adapter), "embedding_scale", lambda self: 1.0)

    ids = torch.tensor([[5, 6, 7, 8, 1]])
    with pytest.raises(EmbeddingSemanticsError, match="does not reproduce"):
        adapter.check_embedding_equivalence(ids, torch.ones_like(ids))


def test_stop_logits_row_count_is_bounded_by_the_horizon(seq2seq_model, tokenizer):
    """`objective_horizon` is an upper bound, not a guaranteed step count.

    The trajectory is the model's own greedy output, so it ends early when the
    model emits EOS. Documented as an upper bound because the objective averages
    over however many rows it gets.
    """
    adapter = ModelAdapter.for_model(seq2seq_model, tokenizer)
    encoded = tokenizer("hello world", return_tensors="pt")
    embeds = adapter.embed(encoded["input_ids"])

    horizon = 5
    rows = adapter.stop_logits(embeds, encoded["attention_mask"], horizon).shape[0]
    assert 1 <= rows <= horizon


def test_causal_stop_logits_cover_generated_tokens_only(causal_model, tokenizer):
    """Regression: the causal adapter must score *generated* positions.

    An earlier version read a window of the final prompt positions. In a causal
    model position `i` predicts token `i+1`, so only the last of those rows
    predicts a token that is actually generated -- the rest score "would you have
    stopped mid-prompt", which does not control generation length. This asserts
    one stop-decision row per generated token instead.
    """
    with torch.no_grad():
        causal_model.lm_head.weight[1].fill_(-20.0)  # let it generate past step 1

    adapter = ModelAdapter.for_model(causal_model, tokenizer)
    encoded = tokenizer("hello world", return_tensors="pt")
    embeds = adapter.embed(encoded["input_ids"])

    horizon = 6
    stop_logits = adapter.stop_logits(embeds, encoded["attention_mask"], horizon)

    assert stop_logits.shape[0] == horizon
    assert stop_logits.shape[0] != encoded["input_ids"].shape[1]

    stop_logits.sum().backward()
    assert embeds.grad is not None and embeds.grad.abs().sum() > 0, \
        "gradient must still reach the perturbable input through the concatenation"


# ------------------------------------------------------------------ EOS + vocab


def test_resolve_eos_handles_int_list_and_missing(causal_model):
    assert resolve_eos_ids(causal_model) == [1]

    causal_model.generation_config.eos_token_id = [1, 5]
    assert resolve_eos_ids(causal_model) == [1, 5]

    causal_model.generation_config.eos_token_id = None
    causal_model.config.eos_token_id = None
    assert resolve_eos_ids(causal_model) == []


def test_forbidden_ids_cover_specials_and_surplus_embedding_rows(tokenizer):
    """Both classes break the round trip, for different reasons."""
    from conftest import SPECIAL_IDS, TOKENIZER_VOCAB, VOCAB

    forbidden = forbidden_token_ids(tokenizer, VOCAB)
    assert set(SPECIAL_IDS) <= forbidden
    assert set(range(TOKENIZER_VOCAB, VOCAB)) <= forbidden
    assert 10 not in forbidden, "ordinary ids must stay available"


def test_forbidden_ids_tolerate_a_tokenizer_without_len(seq2seq_model):
    class Minimal:
        all_special_ids = [0, 1]

    assert forbidden_token_ids(Minimal(), 64) == {0, 1}


# --------------------------------------- equivalence across architecture families


def _tiny(builder):
    torch.manual_seed(0)
    return builder().eval()


def _architectures():
    """Small configs across the families this toolbox claims to handle.

    Built locally so the check stays offline. These are architecture *shapes*,
    not trained checkpoints: passing here means the `inputs_embeds` path is
    wired correctly for that family, not that an attack was ever run on it.
    RESULTS.md keeps those evidence levels separate.
    """
    from transformers import (
        BartConfig, BartForConditionalGeneration,
        LlamaConfig, LlamaForCausalLM,
        OPTConfig, OPTForCausalLM,
        Qwen2Config, Qwen2ForCausalLM,
    )

    def bart(scale):
        return BartForConditionalGeneration(BartConfig(
            vocab_size=64, d_model=32, encoder_layers=1, decoder_layers=1,
            encoder_attention_heads=2, decoder_attention_heads=2,
            encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=64,
            pad_token_id=0, decoder_start_token_id=0, eos_token_id=1,
            bos_token_id=0, scale_embedding=scale))

    return [
        ("BART scale_embedding=False", lambda: bart(False)),
        ("BART scale_embedding=True", lambda: bart(True)),
        ("Marian scale_embedding=True", lambda: _marian(True)),
        ("OPT", lambda: OPTForCausalLM(OPTConfig(
            vocab_size=64, hidden_size=32, num_hidden_layers=1,
            num_attention_heads=2, ffn_dim=32, max_position_embeddings=64,
            word_embed_proj_dim=32, eos_token_id=1, pad_token_id=0,
            bos_token_id=0))),
        ("Llama", lambda: LlamaForCausalLM(LlamaConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=64, eos_token_id=1, pad_token_id=0,
            bos_token_id=0))),
        ("Qwen2", lambda: Qwen2ForCausalLM(Qwen2Config(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=64, eos_token_id=1, pad_token_id=0,
            bos_token_id=0))),
    ]


@pytest.mark.parametrize("name,builder", _architectures(), ids=lambda v: v if isinstance(v, str) else "")
def test_inputs_embeds_matches_input_ids_across_families(name, builder, tokenizer):
    """The differentiable path must reproduce the model's normal forward pass.

    This is the invariant that stops the attack optimising a function the model
    does not compute. RESULTS.md reports zero deviation for these families; this
    is the reproducer for that claim.
    """
    model = _tiny(builder)
    adapter = ModelAdapter.for_model(model, tokenizer)
    ids = torch.tensor([[5, 6, 7, 8, 1]])

    deviation = adapter.check_embedding_equivalence(ids, torch.ones_like(ids))
    assert deviation == pytest.approx(0.0, abs=1e-4), f"{name}: deviation {deviation}"
