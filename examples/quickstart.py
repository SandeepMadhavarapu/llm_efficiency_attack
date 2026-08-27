"""End-to-end demonstration on a real Hugging Face model.

Runs the exact snippet from the task description, then goes two steps further
than a bare demo would:

* it runs the random-substitution control under an identical budget, so the
  gradient's contribution is measured rather than assumed; and
* it reports whether the cost measurement is censored by the generation ceiling,
  because once the attack works the raw ratio is a lower bound, not a result.

Usage::

    python examples/quickstart.py                 # t5-small (seq2seq)
    python examples/quickstart.py --model gpt2    # causal LM, no code changes
"""

from __future__ import annotations

import argparse
import json

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from llm_efficiency_attack import Attacker, measure_cost


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
    args = parser.parse_args()

    model, tokenizer = load(args.model)

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

    # ---- the snippet from the task description, verbatim -------------------
    attack = Attacker(model, tokenizer)
    adv_x, logs = attack.run(args.text, config)
    # ------------------------------------------------------------------------

    benign = logs["benign"]
    adversarial = logs["adversarial"]

    print("\n" + "=" * 72)
    print(f"MODEL: {args.model}")
    print("=" * 72)
    print(f"benign input : {args.text!r}")
    print(f"adv input    : {adv_x!r}")
    print(
        f"\nbenign      : {benign['output_tokens']:>4} tokens "
        f"({benign['stopped_by']}, {benign['wall_time_s']:.3f}s)"
    )
    print(
        f"adversarial : {adversarial['output_tokens']:>4} tokens "
        f"({adversarial['stopped_by']}, {adversarial['wall_time_s']:.3f}s)"
    )
    print(
        f"ratio       : {logs['cost']['output_token_ratio']:.2f}x output tokens, "
        f"{logs['cost']['wall_time_ratio']:.2f}x wall time"
    )
    print(f"tokens changed: {logs['perturbation']['tokens_changed']}")

    if logs["censored"]["ratio_is_lower_bound"]:
        print("\n[censored] " + logs["censored"]["note"])

    # ---- control: same budget, no gradient ---------------------------------
    control_config = dict(config, strategy="random", verbose=False)
    control_x, control_logs = Attacker(model, tokenizer).run(args.text, control_config)

    print("\n" + "-" * 72)
    print("CONTROL (random substitution, identical budget and evaluation count)")
    print("-" * 72)
    print(f"random input : {control_x!r}")
    print(
        f"random       : {control_logs['adversarial']['output_tokens']:>4} tokens "
        f"({control_logs['adversarial']['stopped_by']})"
    )
    print(
        f"gradient     : {adversarial['output_tokens']:>4} tokens "
        f"({adversarial['stopped_by']})"
    )
    delta = adversarial["output_tokens"] - control_logs["adversarial"]["output_tokens"]
    print(
        f"\ngradient advantage over random: {delta:+d} tokens. "
        "A small or negative number means the white-box signal is not paying for "
        "itself on this input."
    )

    print("\n" + "-" * 72)
    print("ATTACK COST")
    print("-" * 72)
    print(
        f"{logs['attack_cost']['forward_passes']} forward passes, "
        f"{logs['attack_cost']['backward_passes']} backward passes to induce "
        f"{adversarial['output_tokens'] - benign['output_tokens']} extra decoding steps."
    )

    with open("quickstart_logs.json", "w", encoding="utf-8") as handle:
        json.dump({"gradient": logs, "control": control_logs}, handle, indent=2)
    print("\nfull logs written to quickstart_logs.json")


if __name__ == "__main__":
    main()
