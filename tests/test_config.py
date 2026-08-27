"""Config schema and validation."""

from __future__ import annotations

import pytest

from llm_efficiency_attack import AttackConfig


def test_defaults_are_valid():
    cfg = AttackConfig.from_dict(None)
    assert cfg.strategy == "gradient"
    assert cfg.objective == "eos_suppression"
    assert cfg.perturbation_budget <= cfg.max_iterations


def test_round_trips_through_plain_dict():
    """The config must survive JSON, since the task requires a dict interface."""
    import json

    cfg = AttackConfig.from_dict({"seed": 7, "top_k": 5})
    restored = AttackConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert restored == cfg


def test_unknown_field_is_rejected_not_ignored():
    """A typo must fail loudly.

    Silently dropping `topk` would leave the attack running on the default k and
    look like a weak attack rather than a misconfiguration.
    """
    with pytest.raises(ValueError, match="Unknown config field"):
        AttackConfig.from_dict({"topk": 5})


@pytest.mark.parametrize(
    "bad",
    [
        {"strategy": "nope"},
        {"max_iterations": 0},
        {"top_k": -1},
        {"max_new_tokens": 0},
        {"objective_horizon": 0},
        {"protected_prefix_tokens": -1},
        {"seed": "zero"},
        {"objective": ""},
        {"verbose": "yes"},
    ],
)
def test_incoherent_values_are_rejected(bad):
    with pytest.raises(ValueError):
        AttackConfig.from_dict(bad)


def test_booleans_are_not_accepted_as_ints():
    """`True` is an int in Python. It is not a valid iteration count."""
    with pytest.raises(ValueError):
        AttackConfig.from_dict({"max_iterations": True})


def test_budget_cannot_exceed_iterations():
    """One substitution is committed per iteration, so surplus budget is unusable."""
    with pytest.raises(ValueError, match="cannot exceed max_iterations"):
        AttackConfig.from_dict({"perturbation_budget": 10, "max_iterations": 3})
