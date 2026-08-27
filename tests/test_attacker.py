"""End-to-end attack behaviour.

These tests assert *mechanics*, not attack strength. The fixtures are randomly
initialised models, which have no learned notion of when to stop, so demanding a
large efficiency ratio here would be testing the fixture rather than the code.
Effectiveness is demonstrated on a real trained model in `examples/quickstart.py`.
"""

from __future__ import annotations

import json

import pytest

from llm_efficiency_attack import Attacker

FAST = {
    "max_iterations": 3,
    "perturbation_budget": 2,
    "top_k": 6,
    "max_new_tokens": 12,
    "objective_horizon": 3,
}


def test_public_api_shape(seq2seq_model, tokenizer):
    """The exact snippet from the task description must work as written."""
    attack = Attacker(seq2seq_model, tokenizer)
    adv_x, logs = attack.run("hello world", FAST)

    assert isinstance(adv_x, str)
    assert isinstance(logs, dict)
    for key in ("config", "benign", "adversarial", "cost", "censored",
                "perturbation", "attack_cost", "iterations"):
        assert key in logs, f"logs missing {key!r}"


def test_logs_are_json_serialisable(seq2seq_model, tokenizer):
    """A reviewer must be able to save and diff a run."""
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    json.loads(json.dumps(logs))


def test_works_on_causal_models_unchanged(causal_model, tokenizer):
    """Model-agnosticism: only the `model` argument differs from the seq2seq test."""
    adv_x, logs = Attacker(causal_model, tokenizer).run("hello world", FAST)
    assert isinstance(adv_x, str)
    assert logs["cost"]["benign_output_tokens"] > 0


def test_objective_never_increases(seq2seq_model, tokenizer):
    """The loop commits a substitution only when it strictly improves.

    Monotonicity is the invariant that makes the search meaningful; if it were
    violated the attack would be wandering rather than optimising.
    """
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    values = [it["objective"] for it in logs["iterations"]]
    assert values == sorted(values, reverse=True), values


def test_perturbation_budget_is_respected(seq2seq_model, tokenizer):
    """The imperceptibility constraint is a hard limit, not a target."""
    cfg = dict(FAST, perturbation_budget=1, max_iterations=5)
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", cfg)

    assert logs["perturbation"]["tokens_changed"] <= 1

    original = logs["perturbation"]["original_token_ids"]
    adversarial = logs["perturbation"]["adversarial_token_ids"]
    assert len(original) == len(adversarial), "an attack must not change sequence length"
    differing = sum(a != b for a, b in zip(original, adversarial))
    assert differing <= 1


def test_protected_prefix_is_never_touched(seq2seq_model, tokenizer):
    """Instruction prefixes must survive, or the attack measures task damage."""
    cfg = dict(FAST, protected_prefix_tokens=4)
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", cfg)

    original = logs["perturbation"]["original_token_ids"]
    adversarial = logs["perturbation"]["adversarial_token_ids"]
    assert original[:4] == adversarial[:4]
    assert all(p >= 4 for p in logs["perturbation"]["positions_changed"])


def test_same_seed_gives_same_result(seq2seq_model, tokenizer):
    """Requirement 5: same config plus same input yields the same output."""
    cfg = dict(FAST, strategy="random", seed=123)
    a_x, a_logs = Attacker(seq2seq_model, tokenizer).run("hello world", cfg)
    b_x, b_logs = Attacker(seq2seq_model, tokenizer).run("hello world", cfg)

    assert a_x == b_x
    assert a_logs["perturbation"]["adversarial_token_ids"] == \
        b_logs["perturbation"]["adversarial_token_ids"]


def test_different_seeds_diverge_for_the_random_control(seq2seq_model, tokenizer):
    """If the seed had no effect, the reproducibility test above would be vacuous."""
    cfg_a = dict(FAST, strategy="random", seed=1, top_k=3)
    cfg_b = dict(FAST, strategy="random", seed=999, top_k=3)
    _, a = Attacker(seq2seq_model, tokenizer).run("hello world", cfg_a)
    _, b = Attacker(seq2seq_model, tokenizer).run("hello world", cfg_b)
    assert a["perturbation"]["positions_changed"] != b["perturbation"]["positions_changed"] \
        or a["perturbation"]["adversarial_token_ids"] != b["perturbation"]["adversarial_token_ids"]


def test_random_control_runs_and_is_labelled(seq2seq_model, tokenizer):
    """The control must be a first-class strategy, so the comparison is honest."""
    _, logs = Attacker(seq2seq_model, tokenizer).run(
        "hello world", dict(FAST, strategy="random")
    )
    assert logs["config"]["strategy"] == "random"
    assert logs["attack_cost"]["backward_passes"] == 0, \
        "the control must not use gradients"


def test_gradient_strategy_uses_gradients(seq2seq_model, tokenizer):
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    assert logs["attack_cost"]["backward_passes"] > 0


def test_censoring_is_reported(seq2seq_model, tokenizer):
    """Untrained fixtures never emit EOS, so both runs hit the ceiling.

    That makes this a good test of the honesty machinery: the ratio must be
    labelled a lower bound rather than presented as a measurement.
    """
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    assert logs["censored"]["adversarial_hit_ceiling"] is True
    assert logs["censored"]["ratio_is_lower_bound"] is True
    assert "lower bound" in logs["censored"]["note"]


def test_attack_cost_is_accounted(seq2seq_model, tokenizer):
    """Reviewers should be able to judge whether the attack is economical."""
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    assert logs["attack_cost"]["forward_passes"] > 0


def test_rejects_a_fully_protected_input(seq2seq_model, tokenizer):
    """Nothing left to perturb is a configuration error, not a silent no-op."""
    with pytest.raises(ValueError, match="No perturbable token"):
        Attacker(seq2seq_model, tokenizer).run(
            "hi", dict(FAST, protected_prefix_tokens=999)
        )


def test_unknown_objective_is_rejected(seq2seq_model, tokenizer):
    with pytest.raises(ValueError, match="Unknown objective"):
        Attacker(seq2seq_model, tokenizer).run(
            "hello world", dict(FAST, objective="does_not_exist")
        )


def test_causal_stop_logits_cover_generated_tokens_only(causal_model, tokenizer):
    """Regression: the causal adapter must score *generated* positions.

    An earlier version read a window of the final prompt positions. In a causal
    model position `i` predicts token `i+1`, so only the last of those rows
    predicts a token that is actually generated -- the rest score "would you have
    stopped mid-prompt", which does not control generation length. This asserts
    one stop-decision row per generated token instead.
    """
    import torch

    from llm_efficiency_attack.adapters import ModelAdapter

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
