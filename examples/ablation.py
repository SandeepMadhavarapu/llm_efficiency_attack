"""Does the white-box signal actually earn its cost?

The quickstart runs a single input, which is an anecdote rather than a result.
Three tokens on one sentence is inside the noise, and reporting it as evidence
either way would be exactly the kind of claim this toolbox is built to be
sceptical of.

The comparison
--------------
Gradient-guided search is compared against the random control at three candidate
budgets. Both strategies get identical inputs, iterations, perturbation budget,
seed, candidate space (same positions, same legal token ids, never the token
already in place) and the same number of exact objective evaluations per
iteration. The only difference is whether the gradient chose the shortlist.

`top_k` is swept because it is where the gradient should pay off if it pays off
anywhere: the first-order estimate exists to find good substitutions *without*
evaluating many of them, so its advantage should be largest when few candidates
can be afforded and should shrink as random search gets more draws.

This reports whatever it finds. A negative result is a result.

Usage::

    python examples/ablation.py                 # t5-small
    python examples/ablation.py --model gpt2    # causal LM (see the ceiling warning)
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from llm_efficiency_attack import Attacker, measure_cost

# Several inputs of similar shape. A single sentence cannot distinguish a real
# effect from a lucky substitution.
INPUTS_SEQ2SEQ = [
    "translate English to German: The house is wonderful.",
    "translate English to German: She walked to the market this morning.",
    "translate English to German: The weather today is unusually cold.",
    "translate English to German: He forgot his keys on the kitchen table.",
    "translate English to German: They are building a new library downtown.",
    "summarize: The committee met on Tuesday to review the annual budget report.",
    "translate English to French: Good morning, how are you today?",
]

INPUTS_CAUSAL = [
    "The house is wonderful and",
    "She walked to the market and",
    "The weather today is unusually",
    "He forgot his keys on the",
    "They are building a new",
]

BASE = {
    "objective": "eos_suppression",
    "max_iterations": 6,
    "perturbation_budget": 3,
    "max_new_tokens": 128,
    "objective_horizon": 8,
    "seed": 0,
    "verbose": False,
}

TOP_K_SWEEP = (5, 20, 100)


def load(name: str):
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
        "--protected-prefix",
        type=int,
        default=None,
        help="Leading tokens the attack may not touch. Defaults to 7 for T5's "
        "instruction prefix, 0 otherwise.",
    )
    parser.add_argument("--out", default="results/ablation_t5_small.json")
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    is_seq2seq = model.config.is_encoder_decoder
    inputs = INPUTS_SEQ2SEQ if is_seq2seq else INPUTS_CAUSAL
    protected = args.protected_prefix
    if protected is None:
        protected = 7 if is_seq2seq else 0

    environment = {
        "model": args.model,
        "tokenizer": type(tokenizer).__name__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(next(model.parameters()).device),
        "protected_prefix_tokens": protected,
        "base_config": BASE,
        "top_k_sweep": list(TOP_K_SWEEP),
    }
    print(f"model: {args.model}  |  {len(inputs)} inputs  |  protected prefix: {protected}")
    print(f"torch {torch.__version__}  transformers {transformers.__version__}\n")

    benign = [
        measure_cost(model, tokenizer, text, max_new_tokens=BASE["max_new_tokens"])
        for text in inputs
    ]
    benign_lengths = [b["output_tokens"] for b in benign]
    benign_mean = statistics.mean(benign_lengths)
    ceilings = sum(b["stopped_by"] == "max_tokens" for b in benign)
    print(f"benign mean output: {benign_mean:.1f} tokens "
          f"({ceilings}/{len(inputs)} already at the ceiling)\n")

    if ceilings == len(inputs):
        print("WARNING: every benign input already saturates max_new_tokens.")
        print("Both benign and adversarial costs are right-censored, so the ratio")
        print("bounds the true ratio in NEITHER direction and no efficiency claim")
        print("can be made from this run. This is the normal state for a base")
        print("causal LM, which rarely emits EOS from a short prompt.\n")

    header = (
        f"{'strategy':<10}{'top_k':>7}{'mean out':>10}{'vs benign':>11}"
        f"{'wins':>7}{'censor':>8}{'fwd':>9}{'sec':>7}"
    )
    print(header)
    print("-" * len(header))

    results: dict[str, dict] = {}
    for top_k in TOP_K_SWEEP:
        for strategy in ("gradient", "random"):
            config = dict(
                BASE,
                strategy=strategy,
                top_k=top_k,
                protected_prefix_tokens=protected,
            )
            lengths, deltas, forwards, uninformative, per_input = [], [], 0, 0, []
            started = time.perf_counter()

            for text, base_cost in zip(inputs, benign_lengths):
                adv_x, logs = Attacker(model, tokenizer).run(text, config)
                adv_len = logs["adversarial"]["output_tokens"]
                lengths.append(adv_len)
                deltas.append(adv_len - base_cost)
                forwards += logs["attack_cost"]["search_model_forwards"]
                uninformative += int(
                    logs["censored"]["interpretation"] == "uninformative"
                )
                per_input.append(
                    {
                        "input": text,
                        "adv_x": adv_x,
                        "benign_output_tokens": base_cost,
                        "adversarial_output_tokens": adv_len,
                        "delta": adv_len - base_cost,
                        "hamming_distance": logs["perturbation"]["hamming_distance"],
                        "positions_touched": logs["perturbation"]["positions_touched"],
                        "round_trip_exact": logs["perturbation"]["round_trip_exact"],
                        "censoring": logs["censored"]["interpretation"],
                    }
                )

            elapsed = time.perf_counter() - started
            label = f"{strategy}_k{top_k}"
            results[label] = {
                "strategy": strategy,
                "top_k": top_k,
                "mean_output_tokens": statistics.mean(lengths),
                "mean_delta_tokens": statistics.mean(deltas),
                "inputs_improved": sum(d > 0 for d in deltas),
                "inputs_unchanged": sum(d == 0 for d in deltas),
                "inputs_worsened": sum(d < 0 for d in deltas),
                "uninformative_censoring": uninformative,
                "search_model_forwards": forwards,
                "seconds": elapsed,
                "per_input": per_input,
            }
            r = results[label]
            print(
                f"{strategy:<10}{top_k:>7}{r['mean_output_tokens']:>10.1f}"
                f"{r['mean_delta_tokens']:>+11.2f}"
                f"{r['inputs_improved']:>4}/{len(inputs)}"
                f"{uninformative:>6}/{len(inputs)}{forwards:>9}{elapsed:>7.0f}"
            )

    # The comparison the whole study exists to make, per candidate budget.
    print("\n" + "-" * len(header))
    print(f"{'top_k':>7}{'gradient':>12}{'random':>10}{'advantage':>12}")
    comparison = {}
    for top_k in TOP_K_SWEEP:
        g = results[f"gradient_k{top_k}"]["mean_delta_tokens"]
        r = results[f"random_k{top_k}"]["mean_delta_tokens"]
        comparison[top_k] = {"gradient": g, "random": r, "advantage": g - r}
        print(f"{top_k:>7}{g:>+12.2f}{r:>+10.2f}{g - r:>+12.2f}")

    wins = sum(1 for c in comparison.values() if c["advantage"] > 0)
    print(
        f"\ngradient beats random at {wins}/{len(TOP_K_SWEEP)} candidate budgets "
        "(mean extra output tokens vs the benign input)."
    )
    if wins == 0:
        print(
            "\nThe gradient does not beat random search on this model at any of the\n"
            "budgets tested. That is a finding, not a bug. With a shortlist of k\n"
            "candidates re-scored exactly, random search over a 32k vocabulary is\n"
            "already a strong baseline, and the first-order ranking is only a weak\n"
            "signal here -- see RESULTS.md and examples/hotflip_diagnostic.py."
        )

    payload = {
        "environment": environment,
        "inputs": inputs,
        "benign_output_tokens": benign_lengths,
        "benign_mean_output_tokens": benign_mean,
        "benign_at_ceiling": ceilings,
        "results": results,
        "gradient_vs_random": comparison,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
