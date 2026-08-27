"""End-to-end demonstration on a real Hugging Face model.

Runs the exact snippet from the task description, then goes two steps further
than a bare demo would:

* it runs the random-substitution control under an identical budget, so the
  gradient's contribution is measured rather than assumed; and
* it reports what the censoring state permits the cost ratio to mean, because a
  ratio between two ceiling-bound measurements establishes nothing.

Usage::

    python examples/quickstart.py                 # t5-small (seq2seq)
    python examples/quickstart.py --model gpt2    # causal LM, no code changes

Note on `--model gpt2`: base causal LMs rarely emit EOS from a short prompt, so
both benign and adversarial generation run to `max_new_tokens` and the result is
uninformative rather than successful. That is reported, not hidden. See
RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import sys

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from llm_efficiency_attack import Attacker


def show(label: str, value: object) -> None:
    """Print a value that may contain non-ASCII text.

    HotFlip picks rare vocabulary items, so `adv_x` regularly contains characters
    a legacy Windows console cannot encode (cp1252 raises on `ţ`, for
    instance). Escaping keeps the demo runnable everywhere instead of dying at
    the moment it has something to show.
    """
    text = f"{label}{value}"
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(encoding, "backslashreplace").decode(encoding) + "\n")


def load(name: str):
    """Load any HF model, seq2seq or causal, without the caller knowing which."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
    except (ValueError, OSError):
        model = AutoModelForCausalLM.from_pretrained(name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="t5-small")
    parser.add_argument(
        "--text", default="translate English to German: The house is wonderful."
    )
    parser.add_argument(
        "--protected-prefix",
        type=int,
        default=7,
        help=(
            "Leading tokens the attack may not touch. Defaults to 7, which covers "
            "T5's 'translate English to German:' instruction. Perturbing the "
            "instruction would break the task rather than attack its efficiency, "
            "so protecting it is what keeps the measurement honest. Use 0 for a "
            "model with no instruction prefix."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--out", default=None, help="where to write the full logs")
    args = parser.parse_args()

    model, _ = load(args.model)

    config = {
        "objective": "eos_suppression",
        "strategy": "gradient",
        "max_iterations": 10,
        "perturbation_budget": 3,
        "top_k": 20,
        "max_new_tokens": args.max_new_tokens,
        "protected_prefix_tokens": args.protected_prefix,
        "objective_horizon": 8,
        "seed": 0,
        "device": "auto",
        "verbose": True,
    }

    # ---- the snippet from the task description --------------------------------
    # One argument. The tokenizer is loaded from the model's own `_name_or_path`.
    x = args.text
    attack = Attacker(model)
    adv_x, logs = attack.run(x, config)
    # ---------------------------------------------------------------------------

    benign, adversarial = logs["benign"], logs["adversarial"]
    perturbation, censored = logs["perturbation"], logs["censored"]

    print("\n" + "=" * 72)
    print(f"MODEL: {args.model}")
    print("=" * 72)
    show("benign input : ", repr(x))
    show("adv input    : ", repr(adv_x))
    print(
        f"\nbenign      : {benign['output_tokens']:>4} tokens "
        f"({benign['stopped_by']}, {benign['wall_time_s']:.3f}s)"
    )
    print(
        f"adversarial : {adversarial['output_tokens']:>4} tokens "
        f"({adversarial['stopped_by']}, {adversarial['wall_time_s']:.3f}s)"
    )
    print(f"ratio       : {logs['cost']['output_token_ratio']:.2f}x generated tokens")
    print(
        f"perturbation: {perturbation['hamming_distance']} token(s) differ "
        f"({perturbation['positions_touched']} position(s) touched, "
        f"budget {perturbation['budget']}, "
        f"round-trip exact: {perturbation['round_trip_exact']})"
    )

    print(f"\n[censoring: {censored['interpretation']}]")
    print("  " + censored["note"])

    # ---- control: same budget, same candidate space, no gradient --------------
    control_config = dict(config, strategy="random", verbose=False)
    _, control_logs = Attacker(model).run(x, control_config)

    print("\n" + "-" * 72)
    print("CONTROL (random substitution, identical budget and evaluation count)")
    print("-" * 72)
    print(f"random   : {control_logs['adversarial']['output_tokens']:>4} tokens "
          f"({control_logs['adversarial']['stopped_by']})")
    print(f"gradient : {adversarial['output_tokens']:>4} tokens "
          f"({adversarial['stopped_by']})")
    delta = adversarial["output_tokens"] - control_logs["adversarial"]["output_tokens"]
    print(
        f"\ngradient advantage on this single input: {delta:+d} tokens. One input is "
        "an anecdote;\nexamples/ablation.py runs the controlled sweep. On t5-small "
        "the gradient does\nnot beat random at any tested budget -- see RESULTS.md."
    )

    print("\n" + "-" * 72)
    print("ATTACK COST (instrumented, not estimated)")
    print("-" * 72)
    cost = logs["attack_cost"]
    print(f"  objective evaluations      : {cost['objective_evaluations']}")
    print(f"  gradient (backward) passes : {cost['gradient_evaluations']}")
    print(f"  model forwards, search     : {cost['search_model_forwards']}")
    print(f"  model forwards, measuring  : {cost['measurement_model_forwards']}")
    print(f"  model forwards, diagnostic : {cost['diagnostic_model_forwards']}")
    print(f"  model forwards, total      : {cost['total_model_forwards']}")
    print(
        f"\n  {cost['search_model_forwards']} search forwards to induce "
        f"{adversarial['output_tokens'] - benign['output_tokens']} extra decoding steps."
    )

    horizon = logs["diagnostics"]["objective_horizon"]
    print(
        f"\n  objective horizon: {horizon['requested']} requested, realised rows "
        f"{horizon['realised_rows_min']}-{horizon['realised_rows_max']}, "
        f"{horizon['evaluations_at_full_horizon']}/{horizon['evaluations']} "
        "evaluations at full horizon"
    )

    out = args.out or f"results/quickstart_{args.model.replace('/', '_')}.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"gradient": logs, "control": control_logs}, handle, indent=2)
    print(f"\nfull logs written to {out}")


if __name__ == "__main__":
    main()
