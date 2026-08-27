"""Attack objectives.

An objective turns the model's stop-logits into a single scalar that the attack
*minimises*. Keeping them in a registry rather than hard-coding one is what the
task means by "keep the attack objective swappable": adding a new one is a
function plus a dict entry, with no change to the optimisation loop.

Sign convention, stated once because getting it backwards silently builds an
attack that makes outputs *shorter*: every objective returns a value that is
LOWER when the model is MORE reluctant to stop. The loop always minimises.
"""

from __future__ import annotations

from typing import Callable

import torch

Objective = Callable[[torch.Tensor, list[int]], torch.Tensor]

_REGISTRY: dict[str, Objective] = {}


def register(name: str) -> Callable[[Objective], Objective]:
    """Decorator that adds an objective to the registry under `name`."""

    def _wrap(fn: Objective) -> Objective:
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


@register("eos_suppression")
def eos_suppression(stop_logits: torch.Tensor, eos_ids: list[int]) -> torch.Tensor:
    """Mean log-probability that the model stops, across the horizon.

    This is the direct expression of the NMTSloth insight. Inference cost is
    dominated by the number of decoding steps, and decoding ends when the model
    emits EOS. Driving down the model's probability of emitting EOS therefore
    drives up the number of steps it must run.

    Log-probability rather than probability because probabilities near zero have
    vanishing gradients, which is exactly the regime a working attack pushes the
    model into. In log space the signal stays usable.

    Args:
        stop_logits: `(steps, vocab_size)` raw logits at the stop-decision
            positions.
        eos_ids: Token ids that terminate generation. When a model declares
            several, their probabilities are summed before the log, because
            stopping means emitting *any* of them.

    Returns:
        Scalar tensor. Lower means the model is less willing to stop.
    """
    if not eos_ids:
        raise ValueError(
            "Model declares no EOS token, so there is no stopping probability to "
            "suppress. Pass a model with `eos_token_id` set, or use an objective "
            "that does not depend on EOS."
        )

    log_probs = torch.log_softmax(stop_logits, dim=-1)
    # Sum the probabilities of all stop tokens, then take the log. logsumexp does
    # this in log space without ever leaving it, which avoids underflow once the
    # attack has pushed these probabilities very low.
    eos_log_prob = torch.logsumexp(log_probs[:, eos_ids], dim=-1)
    return eos_log_prob.mean()


@register("eos_suppression_worst_step")
def eos_suppression_worst_step(
    stop_logits: torch.Tensor, eos_ids: list[int]
) -> torch.Tensor:
    """Log-probability of stopping at the single most likely stopping step.

    A variant that optimises the weakest link instead of the average. The mean
    objective can be satisfied by suppressing EOS strongly at most steps while
    leaving one step where the model still readily stops -- and generation only
    needs one such step to terminate. This targets that step directly.

    Included partly to demonstrate that the registry works, and partly because
    the two disagree in an interesting way: `eos_suppression` usually converges
    faster, this one usually produces longer worst-case outputs.
    """
    if not eos_ids:
        raise ValueError("Model declares no EOS token; cannot suppress stopping.")

    log_probs = torch.log_softmax(stop_logits, dim=-1)
    eos_log_prob = torch.logsumexp(log_probs[:, eos_ids], dim=-1)
    return eos_log_prob.max()
