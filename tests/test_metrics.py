"""Cost metric and censoring semantics.

The metric is the scoreboard for the whole toolbox, so its easy-to-get-wrong
parts are pinned here: the generated-token arithmetic (which differs by
architecture), the stop-reason classification, the device placement (which
`generate()` will not fix for us), and the censoring interpretation (which
decides what the reported ratio is allowed to claim).
"""

from __future__ import annotations

import pytest
import torch

from llm_efficiency_attack import measure_cost
from llm_efficiency_attack.metrics import (
    ForwardCounter,
    interpret_censoring,
    model_device,
)


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


# ------------------------------------------------------------ device placement


def test_inputs_are_placed_on_the_model_device(seq2seq_model, tokenizer):
    """`generate()` performs no device migration, so this module must.

    On a CPU-only machine the two devices coincide, so this test cannot fail the
    way the original bug did. What it does pin down is that the tensors handed to
    `generate()` are derived from the model's own device rather than left wherever
    the tokenizer put them -- which is the property that was missing. The
    end-to-end CUDA case is `test_attacker.py::test_runs_on_cuda`, skipped here.
    """
    seen = {}
    original = seq2seq_model.generate

    def spy(*args, **kwargs):
        seen["input_ids"] = kwargs["input_ids"].device
        seen["attention_mask"] = kwargs["attention_mask"].device
        return original(*args, **kwargs)

    seq2seq_model.generate = spy
    try:
        measure_cost(seq2seq_model, tokenizer, "hello", max_new_tokens=4)
    finally:
        seq2seq_model.generate = original

    expected = model_device(seq2seq_model)
    assert seen["input_ids"] == expected
    assert seen["attention_mask"] == expected


# --------------------------------------------------------------- instrumentation


def test_forward_counter_counts_every_decoding_step(seq2seq_model, tokenizer):
    """A single `generate()` call is many model invocations, and this proves it."""
    with ForwardCounter(seq2seq_model) as counter:
        measure_cost(seq2seq_model, tokenizer, "hello", max_new_tokens=8)
    assert counter.count >= 8, counter.count


def test_forward_counter_removes_its_hook(seq2seq_model, tokenizer):
    """A leaked hook would keep counting -- and keep a reference -- after the run."""
    with ForwardCounter(seq2seq_model) as counter:
        measure_cost(seq2seq_model, tokenizer, "hello", max_new_tokens=4)
    after = counter.count
    measure_cost(seq2seq_model, tokenizer, "hello", max_new_tokens=4)
    assert counter.count == after


# ------------------------------------------------------------------- censoring


def _obs(tokens, stopped_by):
    return {"output_tokens": tokens, "wall_time_s": 0.1, "stopped_by": stopped_by}


def test_neither_censored_is_a_point_estimate():
    result = interpret_censoring(_obs(6, "eos"), _obs(9, "eos"), 128)
    assert result["interpretation"] == "point_estimate"
    assert "lower bound" not in result["note"]


def test_adversarial_censored_only_is_a_lower_bound():
    """True adversarial cost is at least the cap; benign is known exactly."""
    result = interpret_censoring(_obs(6, "eos"), _obs(128, "max_tokens"), 128)
    assert result["interpretation"] == "lower_bound"
    assert "lower bound" in result["note"]


def test_both_censored_is_uninformative_not_a_lower_bound():
    """The case that used to be reported as a lower bound, wrongly.

    With `A >= 128` and `B >= 128`, the true ratio `A / B` can be anything: the
    observed 1.0 bounds it in neither direction. This is the state a base causal
    LM produces, so getting it wrong mislabels every such run.
    """
    result = interpret_censoring(_obs(128, "max_tokens"), _obs(128, "max_tokens"), 128)
    assert result["interpretation"] == "uninformative"
    assert "NEITHER direction" in result["note"]
    assert "lower bound" not in result["note"]


def test_benign_censored_only_is_an_upper_bound():
    """Adversarial finished below a cap the benign run hit, so the attack shortened it."""
    result = interpret_censoring(_obs(128, "max_tokens"), _obs(40, "eos"), 128)
    assert result["interpretation"] == "upper_bound"
    assert "upper bound" in result["note"]
    assert "cheaper" in result["note"]


@pytest.mark.parametrize(
    "benign_stop,adv_stop",
    [("eos", "eos"), ("eos", "max_tokens"), ("max_tokens", "max_tokens"),
     ("max_tokens", "eos"), ("other", "eos")],
)
def test_censoring_always_reports_a_known_interpretation(benign_stop, adv_stop):
    result = interpret_censoring(_obs(10, benign_stop), _obs(20, adv_stop), 128)
    assert result["interpretation"] in {
        "point_estimate", "lower_bound", "upper_bound", "uninformative"
    }
    assert result["note"]
