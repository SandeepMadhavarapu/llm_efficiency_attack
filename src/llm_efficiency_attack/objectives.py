"""Attack objectives.

An objective turns the model's stop-logits into a single scalar that the attack
*minimises*. Keeping them in a registry rather than hard-coding one is what the
task means by "keep the attack objective swappable": adding a new one is a
function plus a decorator, with no change to the optimisation loop.

Sign convention, stated once because getting it backwards silently builds an
attack that makes outputs *shorter*: every objective returns a value that is
LOWER when the model is MORE reluctant to stop. The loop always minimises.

Trajectory shape
----------------
An objective receives the stop-logits along a decoding trajectory. By default
that trajectory is the model's own greedy output, so it ends early when the model
emits EOS and the number of rows is *at most* `objective_horizon`. An objective
that needs exactly `objective_horizon` rows declares
`@register(..., force_full_horizon=True)`, and the adapter then forces generation
to run the full horizon. The registry records this because it is a property of
the objective, not of the loop.
"""

from __future__ import annotations

from typing import Callable

import torch

Objective = Callable[[torch.Tensor, list[int]], torch.Tensor]

_REGISTRY: dict[str, Objective] = {}


def register(name: str, *, force_full_horizon: bool = False):
    """Add an objective to the registry under `name`.

    Args:
        name: Lookup key used by `config["objective"]`.
        force_full_horizon: Whether this objective needs exactly
            `objective_horizon` trajectory rows. When True the adapter passes
            `min_new_tokens=horizon` to `generate()`, so the model cannot stop
            early and every candidate is scored on the same number of terms.
    """

    def _wrap(fn: Objective) -> Objective:
        fn.force_full_horizon = force_full_horizon  # type: ignore[attr-defined]
        _REGISTRY[name] = fn
        return fn

    return _wrap


def get_objective(name: str) -> Objective:
    """Look up an objective by name, with a helpful error if it is missing."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown objective {name!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def available_objectives() -> list[str]:
    """Names of every registered objective."""
    return sorted(_REGISTRY)


def _eos_log_prob(stop_logits: torch.Tensor, eos_ids: list[int]) -> torch.Tensor:
    """Log-probability of stopping, per trajectory step.

    Log-probability rather than probability because probabilities near zero have
    vanishing gradients, which is exactly the regime a working attack pushes the
    model into. In log space the signal stays usable.

    When a model declares several stop tokens their probabilities are summed
    before the log, because stopping means emitting *any* of them. `logsumexp`
    does that sum in log space without ever leaving it, which avoids underflow
    once the attack has pushed these probabilities very low.
    """
    log_probs = torch.log_softmax(stop_logits, dim=-1)
    return torch.logsumexp(log_probs[:, eos_ids], dim=-1)


def _require_eos(eos_ids: list[int]) -> None:
    if not eos_ids:
        raise ValueError(
            "Model declares no EOS token, so there is no stopping probability to "
            "suppress. Pass a model with `eos_token_id` set, or use an objective "
            "that does not depend on EOS."
        )


@register("eos_suppression")
def eos_suppression(stop_logits: torch.Tensor, eos_ids: list[int]) -> torch.Tensor:
    """Mean log-probability that the model stops, across the trajectory.

    This is the direct expression of the NMTSloth insight. Inference cost is
    dominated by the number of decoding steps, and decoding ends when the model
    emits EOS. Driving down the model's probability of emitting EOS therefore
    drives up the number of steps it must run.

    The trajectory is the model's own greedy output, so it can be shorter than
    `objective_horizon` and the number of terms in the mean varies between
    candidates. `examples/objective_diagnostic.py` measures how much that
    matters: on t5-small at horizon 8, 249 of 270 admissible candidates reach the
    full horizon, and this formulation's within-input Spearman correlation with
    generated length (-0.284) is indistinguishable from a fixed-horizon variant
    (-0.287). A *mean* also carries no mechanical length bias -- unlike a sum,
    which improves simply by having more negative terms to add.

    Args:
        stop_logits: `(steps, vocab_size)` raw logits at the stop-decision
            positions.
        eos_ids: Token ids that terminate generation.

    Returns:
        Scalar tensor. Lower means the model is less willing to stop.
    """
    _require_eos(eos_ids)
    return _eos_log_prob(stop_logits, eos_ids).mean()


@register("eos_suppression_fixed_horizon", force_full_horizon=True)
def eos_suppression_fixed_horizon(
    stop_logits: torch.Tensor, eos_ids: list[int]
) -> torch.Tensor:
    """`eos_suppression` scored over exactly `objective_horizon` forced steps.

    Same reduction as `eos_suppression`; the difference is the trajectory. By
    forcing generation past EOS (`min_new_tokens=horizon`) every candidate is
    scored on the same number of terms, which removes the question of whether
    means over different-length trajectories are comparable.

    That comparability is bought at a price: the trajectory is a counterfactual
    -- "if you were forced to keep going, how reluctant would you be to stop at
    each of the first H steps" -- rather than the path the model would actually
    take, which is the path that determines real cost.

    Registered rather than made the default because the diagnostic found the two
    statistically indistinguishable as proxies for generated length (within-input
    Spearman -0.287 against -0.284, identical top-5 selection gain of +1.8
    tokens). Changing a default on a difference that size would be tuning, not
    correcting. It is here so the choice is measurable rather than asserted.
    """
    _require_eos(eos_ids)
    return _eos_log_prob(stop_logits, eos_ids).mean()


@register("eos_suppression_worst_step")
def eos_suppression_worst_step(
    stop_logits: torch.Tensor, eos_ids: list[int]
) -> torch.Tensor:
    """Log-probability of stopping at the single most likely stopping step.

    A variant that optimises the weakest link instead of the average. The mean
    objective can be satisfied by suppressing EOS strongly at most steps while
    leaving one step where the model still readily stops -- and generation only
    needs one such step to terminate. This targets that step directly.

    Caveat worth stating, because it cuts against the attack's goal: taking a
    maximum over a variable-length trajectory means a candidate that generates
    more steps has more chances to contain a high-stop-probability step, so this
    formulation structurally penalises the longer trajectories the attack is
    trying to produce. Pairing it with `force_full_horizon` would remove that,
    which is why the registry records trajectory shape separately from reduction.
    """
    _require_eos(eos_ids)
    return _eos_log_prob(stop_logits, eos_ids).max()
