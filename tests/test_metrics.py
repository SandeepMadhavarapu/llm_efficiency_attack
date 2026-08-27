"""Cost metric.

The metric is the scoreboard for the whole toolbox, so its two easy-to-get-wrong
parts are pinned here: the generated-token arithmetic (which differs by
architecture) and the stop-reason classification (which decides whether a result
is a measurement or a censored lower bound).
"""

from __future__ import annotations

import torch

from llm_efficiency_attack import measure_cost


def test_returns_documented_schema(seq2seq_model, tokenizer):
    result = measure_cost(seq2seq_model, tokenizer, "hello", max_new_tokens=8)
    assert set(result) == {"output_tokens", "wall_time_s", "stopped_by"}
    assert isinstance(result["output_tokens"], int)
    assert result["wall_time_s"] > 0
    assert result["stopped_by"] in {"eos", "max_tokens", "other"}


def test_seq2seq_counts_exclude_decoder_start_token(seq2seq_model, tokenizer):
    """A seq2seq `generate()` returns decoder output whose first token is seeded.

    With an untrained model EOS is effectively never emitted, so generation runs
    to the ceiling and the count must equal the cap exactly -- not the cap plus
    the decoder start token.
    """
    result = measure_cost(seq2seq_model, tokenizer, "hello", max_new_tokens=8)
    assert result["output_tokens"] == 8
    assert result["stopped_by"] == "max_tokens"


def test_causal_counts_exclude_the_prompt(causal_model, tokenizer):
    """A causal `generate()` returns prompt and continuation concatenated.

    If the prompt were counted, a longer input would masquerade as a more
    expensive generation, inverting the very effect the attack measures.

    EOS is suppressed first so that generation is guaranteed to run to the
    ceiling. Without that, an untrained model may emit EOS on its first step by
    chance -- as this fixture in fact does -- and the test would be measuring the
    fixture's arbitrary weights rather than the counting arithmetic.
    """
    with torch.no_grad():
        causal_model.lm_head.weight[1].fill_(-50.0)  # id 1 is EOS in the fixtures

    result = measure_cost(causal_model, tokenizer, "hello", max_new_tokens=8)
    assert result["output_tokens"] == 8
    assert result["stopped_by"] == "max_tokens"

    longer = measure_cost(causal_model, tokenizer, "hello world again", max_new_tokens=8)
    assert longer["output_tokens"] == 8, "input length must not affect the cost metric"


def test_eos_is_detected_when_generation_stops_early(causal_model, tokenizer):
    """Force EOS to dominate the output distribution and confirm classification."""
    with torch.no_grad():
        causal_model.lm_head.weight.zero_()
        causal_model.lm_head.weight[1].fill_(50.0)  # id 1 is EOS in the fixtures

    result = measure_cost(causal_model, tokenizer, "hello", max_new_tokens=32)
    assert result["stopped_by"] == "eos"
    assert result["output_tokens"] < 32


def test_ceiling_is_reported_even_when_last_token_is_eos(causal_model, tokenizer):
    """The cap is checked before EOS, on purpose.

    A run that reaches the ceiling is right-censored: the true cost is at least
    the cap. Labelling it `"eos"` because the final token happened to be a stop
    token would disguise a saturated measurement as a natural one, and would make
    a maximally successful attack look like a failed one.
    """
    with torch.no_grad():
        causal_model.lm_head.weight.zero_()
        causal_model.lm_head.weight[1].fill_(50.0)

    result = measure_cost(causal_model, tokenizer, "hello", max_new_tokens=1)
    assert result["output_tokens"] == 1
    assert result["stopped_by"] == "max_tokens"


def test_missing_eos_does_not_crash(causal_model, tokenizer):
    """Some models declare no stop token. That is a real case, not an error."""
    causal_model.generation_config.eos_token_id = None
    causal_model.config.eos_token_id = None

    result = measure_cost(causal_model, tokenizer, "hello", max_new_tokens=4)
    assert result["stopped_by"] == "max_tokens"


def test_is_deterministic(seq2seq_model, tokenizer):
    """Greedy decoding means repeated calls must agree exactly."""
    a = measure_cost(seq2seq_model, tokenizer, "hello", max_new_tokens=8)
    b = measure_cost(seq2seq_model, tokenizer, "hello", max_new_tokens=8)
    assert a["output_tokens"] == b["output_tokens"]
    assert a["stopped_by"] == b["stopped_by"]
