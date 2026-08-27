"""Cost measurement for efficiency attacks.

A sequence model's inference cost is dominated by how many *decoding steps* it
runs: each step is one forward pass through the decoder. Decoding halts when
the model emits an end-of-sequence token or when it hits a caller-supplied cap.
So "how expensive was this input?" is really "how many tokens came out?" -- and
notably *not* "how long was the input", which is the observation the NMTSloth
attack is built on.
"""

from __future__ import annotations

import time
from typing import Any


import torch


def _resolve_eos_ids(model: Any) -> list[int]:
    """Collect the token ids that terminate generation for this model.

    Two wrinkles this exists to absorb:

    * `eos_token_id` may be a single int or a list of ints. Several chat models
      define more than one stop token, and a bare `==` comparison silently
      misses all but the first.
    * `generation_config` is what `generate()` actually consults, and it can
      differ from `config`. We prefer it and fall back to `config`.

    Returns an empty list when the model declares no stop token at all, which
    is a real case (some base LMs) and is handled by the caller rather than
    crashing here.
    """
    for source in (getattr(model, "generation_config", None), model.config):
        if source is None:
            continue
        eos = getattr(source, "eos_token_id", None)
        if eos is None:
            continue
        return [eos] if isinstance(eos, int) else list(eos)
    return []


def measure_cost(
    model: Any,
    tokenizer: Any,
    text: str,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    """Measure what it costs the model to generate from `text`.

    Args:
        model: Any Hugging Face model exposing `.generate()`.
        tokenizer: The tokenizer paired with `model`.
        text: The benign or adversarial input to measure.
        max_new_tokens: Hard ceiling on decoding steps.

    Returns:
        A dict with:
          `output_tokens` -- tokens the model actually generated.
          `wall_time_s`   -- seconds spent inside `generate()`.
          `stopped_by`    -- `"eos"` if generation ended on a stop token,
                             `"max_tokens"` if it ran into the ceiling.
    """
    inputs = tokenizer(text, return_tensors="pt")
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

    # How many returned tokens were actually *generated*?
    #
    # Encoder-decoder: `generate()` returns decoder output only, but index 0 is
    #   the decoder start token, which the model did not produce.
    # Causal: `generate()` returns prompt and continuation concatenated, so the
    #   prompt length comes off.
    #
    # `config.is_encoder_decoder` is the branch point rather than an isinstance
    # check, so this works for any architecture without naming one.
    if model.config.is_encoder_decoder:
        output_tokens = int(out.shape[1] - 1)
    else:
        output_tokens = int(out.shape[1] - input_len)

    # Why did generation stop? The ceiling is checked first on purpose. If we
    # produced exactly the number of tokens we permitted, the cap is the binding
    # constraint -- even in the rare case where the final token is also EOS.
    # Reporting "eos" there would hide that the metric has saturated, and a
    # saturated cost metric can no longer tell a good attack from a great one.
    eos_ids = _resolve_eos_ids(model)
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
