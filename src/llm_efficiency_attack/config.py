"""Attack configuration: schema, defaults, and validation.

Every knob the attack has lives here and nowhere else. The task requires the
configuration to be a plain JSON-serialisable dict, so `AttackConfig` is a thin
typed wrapper that parses one, checks it, and fails loudly on bad input rather
than surfacing a confusing tensor error twenty seconds into a run.

There is deliberately no `step_size`. The task's config table lists one, but this
attack is a discrete search over token substitutions, not continuous gradient
descent: there is no continuous variable to step. The gradient is used only to
*rank* candidate substitutions, and `top_k` -- how many of those ranked candidates
get an exact evaluation -- is the analogous knob. Adding a `step_size` that
nothing reads would be worse than not having one.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any

import torch


# Strategies the attacker knows how to run. "random" exists as an experimental
# control, not as a feature: it spends the identical perturbation budget and the
# identical number of exact evaluations over the identical candidate space while
# ignoring the gradient, so comparing the two is what shows whether the white-box
# signal is actually doing any work.
STRATEGIES = ("gradient", "random")


@dataclass(frozen=True)
class AttackConfig:
    """Validated attack configuration.

    Attributes:
        objective: Name of the attack objective, resolved in `objectives.py`.
        strategy: `"gradient"` for the white-box attack, `"random"` for the
            control that substitutes tokens uniformly at random.
        max_iterations: Upper bound on optimisation steps. Each step commits at
            most one token substitution.
        perturbation_budget: Maximum number of *distinct token positions* the
            search may write to. This bounds the token-level edit count and
            nothing else: it enforces no semantic, fluency, or human-perceptibility
            constraint.
        top_k: How many candidate substitutions, ranked by the first-order
            estimate, get an exact objective evaluation. The linear approximation
            is reliable for ranking but not for magnitude, so the shortlist is
            re-scored exactly before anything is committed. Each candidate costs
            a full objective evaluation, so this is the main runtime knob.
        max_new_tokens: Generation ceiling used by the cost metric. Also the
            point at which the cost measurement becomes right-censored.
        protected_prefix_tokens: Leading tokens the attack may not touch. For an
            instruction-prefixed model such as T5 (`"translate English to
            German:"`) perturbing the prefix destroys the task rather than
            attacking efficiency, which would measure the wrong thing.
        objective_horizon: Upper bound on how many decoder steps the objective
            looks at. The realised trajectory is the model's own greedy output,
            so it may be shorter when the model stops early. One step asks only
            "will you stop immediately?"; a longer horizon asks the model to keep
            declining to stop, which is a stronger signal.
        seed: Seeds Python, NumPy and torch RNGs so a run is reproducible.
        device: Torch device string. `"auto"` picks CUDA when available. Single
            device only; sharded and offloaded models are out of scope.
        verbose: Emit per-iteration progress through the logging module.
    """

    objective: str = "eos_suppression"
    strategy: str = "gradient"
    max_iterations: int = 10
    perturbation_budget: int = 3
    top_k: int = 20
    max_new_tokens: int = 128
    protected_prefix_tokens: int = 0
    objective_horizon: int = 8
    seed: int = 0
    device: str = "auto"
    verbose: bool = False

    # ---------------------------------------------------------------- parsing

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> "AttackConfig":
        """Build a validated config from a plain dict.

        Unknown keys are rejected rather than ignored. A silently dropped
        `"top_k"` because the caller wrote `"topk"` is the kind of bug that
        looks like a weak attack rather than a typo, and costs an afternoon.
        """
        config = dict(config or {})
        known = {f.name for f in fields(cls)}
        unknown = set(config) - known
        if unknown:
            raise ValueError(
                f"Unknown config field(s): {sorted(unknown)}. "
                f"Valid fields are: {sorted(known)}"
            )
        instance = cls(**config)
        instance.validate()
        return instance

    def to_dict(self) -> dict[str, Any]:
        """Round-trip back to a JSON-serialisable dict, for echoing into logs."""
        return asdict(self)

    # ------------------------------------------------------------- validation

    def validate(self) -> None:
        """Raise `ValueError` on any incoherent configuration."""
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f"strategy must be one of {STRATEGIES}, got {self.strategy!r}"
            )

        positive_ints = {
            "max_iterations": self.max_iterations,
            "perturbation_budget": self.perturbation_budget,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "objective_horizon": self.objective_horizon,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")

        if (
            not isinstance(self.protected_prefix_tokens, int)
            or isinstance(self.protected_prefix_tokens, bool)
            or self.protected_prefix_tokens < 0
        ):
            raise ValueError(
                "protected_prefix_tokens must be a non-negative int, "
                f"got {self.protected_prefix_tokens!r}"
            )

        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")

        if not isinstance(self.objective, str) or not self.objective:
            raise ValueError(f"objective must be a non-empty str, got {self.objective!r}")

        if not isinstance(self.verbose, bool):
            raise ValueError(f"verbose must be a bool, got {self.verbose!r}")

        self._validate_device()

        # A budget larger than the iteration count is not an error, but the
        # reverse ordering is worth stating: the attack can never change more
        # positions than it has iterations to spend.
        if self.perturbation_budget > self.max_iterations:
            raise ValueError(
                "perturbation_budget cannot exceed max_iterations "
                f"({self.perturbation_budget} > {self.max_iterations}): the attack "
                "commits at most one substitution per iteration, so the extra "
                "budget could never be spent."
            )

    def _validate_device(self) -> None:
        """Reject device strings torch cannot parse, at config time.

        Without this, a typo like `"cude"` surfaces as a torch parser error deep
        inside the run, after the model has already been loaded. Note that this
        checks the string is *well formed*, not that the device is present:
        `"cuda"` on a CPU-only machine is a valid string and fails later, with
        torch's own clear message.
        """
        if not isinstance(self.device, str) or not self.device:
            raise ValueError(f"device must be a non-empty str, got {self.device!r}")
        if self.device == "auto":
            return
        try:
            torch.device(self.device)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"device must be 'auto' or a torch device string, got "
                f"{self.device!r} ({exc})"
            ) from exc
