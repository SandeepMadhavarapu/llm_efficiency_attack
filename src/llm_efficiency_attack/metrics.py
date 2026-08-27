"""Cost measurement for efficiency attacks.

A sequence model's inference cost is dominated by how many *decoding steps* it
runs: each step is one forward pass through the decoder. Decoding halts when
the model emits an end-of-sequence token or when it hits a caller-supplied cap.
So "how expensive was this input?" is really "how many tokens came out?" -- and
notably *not* "how long was the input", which is the observation the NMTSloth
attack is built on.

What this module measures, stated precisely: **generated token count under
greedy decoding**, which is a proxy for inference cost, not a measurement of
latency. The two differ because per-step cost grows with the KV cache and varies
with batching and hardware. Token count is used as the primary metric because it
is deterministic and hardware-independent; wall time is reported alongside it as
secondary, noisy information.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from .adapters import ModelAdapter, resolve_eos_ids


class ForwardCounter:
    """Counts real invocations of the model, including those inside `generate()`.

    The reason this exists: an "objective evaluation" in this attack is not one
    forward pass. Evaluating the objective runs `generate()` for up to `horizon`
    decoding steps and then one teacher-forced forward, so a naive counter that
    increments once per objective call undercounts the model's actual work by
    roughly an order of magnitude. Rather than estimate that factor from a
    formula, this hooks the module itself and counts what actually happened.

    A forward *pre*-hook on the top-level model fires once per `Module.__call__`,
    which is exactly what `generate()` does per decoding step, so the count is
    the real number of model invocations.

    Used as a context manager so the hook is always removed, including on error.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.count = 0
        self._handle = None

    def _hook(self, module: Any, args: Any) -> None:
        self.count += 1

    def __enter__(self) -> "ForwardCounter":
        self._handle = self.model.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def model_device(model: Any) -> torch.device:
    """The device the model's parameters live on.

    Hugging Face's `generate()` does no device migration of its inputs -- it will
    raise deep inside the first embedding lookup if they disagree -- so every
    tensor this package builds is placed here explicitly.

    Scope note: this assumes a single-device model. Accelerate-sharded,
    offloaded, and `device_map="auto"` models spread parameters across devices
    and are outside the scope of this toolbox.
    """
    return next(model.parameters()).device


def measure_cost(
    model: Any,
    tokenizer: Any,
    text: str,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    """Measure what it costs the model to generate from `text`.

    Args:
        model: Any Hugging Face model exposing `.generate()`. Must be on a single
            device; see `model_device`.
        tokenizer: The tokenizer paired with `model`.
        text: The benign or adversarial input to measure.
        max_new_tokens: Hard ceiling on decoding steps.

    Returns:
        A dict with:
          `output_tokens` -- tokens the model actually generated.
          `wall_time_s`   -- seconds spent inside `generate()`. Noisy; secondary.
          `stopped_by`    -- `"eos"` if generation ended on a stop token,
                             `"max_tokens"` if it ran into the ceiling,
                             `"other"` if it halted early without a stop token.
    """
    device = model_device(model)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # `no_grad` because measuring needs no autograd graph -- it is faster and
    # uses less memory. The attack module deliberately does the opposite, since
    # gradients are the whole point there. That contrast is why measurement and
    # attack live in separate modules.
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # Greedy. With sampling on, the same input yields a different length
            # every run, which would destroy both reproducibility and any
            # benign-vs-adversarial comparison.
            do_sample=False,
        )
    wall_time_s = time.perf_counter() - start

    # How many returned tokens were actually *generated*? The arithmetic differs
    # by architecture, and the adapter is the single place in this package that
    # knows about architecture, so it answers rather than a branch here.
    output_tokens = ModelAdapter.for_model(model, tokenizer).count_generated(
        out, input_len
    )

    # Why did generation stop? The ceiling is checked first on purpose. If we
    # produced exactly the number of tokens we permitted, the cap is the binding
    # constraint -- even in the rare case where the final token is also EOS.
    # Reporting "eos" there would hide that the metric has saturated, and a
    # saturated cost metric can no longer tell a good attack from a great one.
    eos_ids = resolve_eos_ids(model)
    last_token = int(out[0, -1].item())

    if output_tokens >= max_new_tokens:
        stopped_by = "max_tokens"
    elif last_token in eos_ids:
        stopped_by = "eos"
    else:
        # Stopped early without emitting a stop token. Rare under greedy
        # decoding, but possible via custom stopping criteria or a model with
        # no declared EOS. Reported as "max_tokens" would be a lie, so this is
        # deliberately its own value.
        stopped_by = "other"

    return {
        "output_tokens": output_tokens,
        "wall_time_s": wall_time_s,
        "stopped_by": stopped_by,
    }


def interpret_censoring(
    benign: dict[str, Any], adversarial: dict[str, Any], max_new_tokens: int
) -> dict[str, Any]:
    """Say what the observed cost ratio does and does not establish.

    Both measurements are capped at `max_new_tokens`, which makes them
    *right-censored*: when generation stops because it hit the ceiling, the true
    cost is only known to be at least that value. Write `B` and `A` for the true
    benign and adversarial costs and `b`, `a` for the observed ones. There are
    four cases and they are genuinely different:

    * **Neither censored.** `B = b`, `A = a`. The ratio `a / b` is exact.
    * **Adversarial censored only.** `A >= max_new_tokens`, `B = b` exactly, so
      `A / B >= a / b`. The observed ratio is a **lower bound**.
    * **Both censored.** `A >= max_new_tokens` and `B >= max_new_tokens`. Two
      lower bounds divided by each other bound nothing: the true ratio could be
      above or below the observed 1.0. The observation is **uninformative**
      about the ratio, and calling it a lower bound would be false.
    * **Benign censored only.** `B >= max_new_tokens` while `A = a <
      max_new_tokens`, so `A < B`: the attack made generation strictly
      *shorter*, and `a / b` is an **upper bound** on the true ratio.

    The third case is the one that matters in practice: it is what a base causal
    LM produces, because such models rarely emit EOS from a short prompt and sit
    at the ceiling before the attack starts.
    """
    benign_ceiling = benign["stopped_by"] == "max_tokens"
    adv_ceiling = adversarial["stopped_by"] == "max_tokens"

    if not benign_ceiling and not adv_ceiling:
        interpretation = "point_estimate"
        note = (
            "Neither generation reached the max_new_tokens ceiling, so both "
            "costs are exact measurements and the reported ratio is a point "
            "estimate."
        )
    elif adv_ceiling and not benign_ceiling:
        interpretation = "lower_bound"
        note = (
            "Adversarial generation reached the max_new_tokens ceiling while the "
            "benign run terminated on its own, so the true adversarial cost is "
            "at least the ceiling: the reported ratio is a lower bound on the "
            "true ratio, not a point estimate. Raise max_new_tokens to tighten "
            "it."
        )
    elif adv_ceiling and benign_ceiling:
        interpretation = "uninformative"
        note = (
            "Both generations reached the max_new_tokens ceiling, so both costs "
            "are right-censored and the observed ratio bounds the true ratio in "
            "NEITHER direction. No efficiency claim can be made from this run. "
            "Raise max_new_tokens until at least the benign input terminates on "
            "its own, or attack a model that stops by itself."
        )
    else:
        interpretation = "upper_bound"
        note = (
            "The benign generation reached the max_new_tokens ceiling while the "
            "adversarial run terminated on its own, so the true adversarial cost "
            "is strictly below the true benign cost: this input got cheaper, not "
            "more expensive. The reported ratio is an upper bound on the true "
            "ratio."
        )

    return {
        "max_new_tokens": max_new_tokens,
        "benign_hit_ceiling": benign_ceiling,
        "adversarial_hit_ceiling": adv_ceiling,
        "interpretation": interpretation,
        "note": note,
    }
