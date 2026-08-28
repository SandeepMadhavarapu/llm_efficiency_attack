"""One-shot evaluation of the locked strategies on the frozen held-out benchmark.

Protocol, and why it is worth the ceremony
------------------------------------------
Every other number in this repository was measured on inputs that were already
in view while the code was being written. That is fine for diagnosis and useless
as confirmation: a reviewer cannot tell tuning from insight when the evaluation
set was always visible.

So the benchmark in `benchmarks/heldout_seq2seq_v1.json` was written and hashed
*before* any algorithm change and before any model had been run on it, and the
strategy set, configuration, seed and metrics were fixed in
`benchmarks/STRATEGY_LOCK.txt` before any held-out result was observed. Both
hashes are re-checked here at run time, and the script refuses to run if either
file has changed.

This is run once. All 24 inputs are reported, with no filtering, no prompt
replacement and no post-hoc configuration change.

Fairness
--------
All three strategies perform exactly `top_k` exact objective evaluations per
iteration over the same legal candidate space, the same allowed positions, the
same iteration cap and the same perturbation budget. Equal candidate counts do
not mean equal compute -- the gradient variants also pay a backward pass and one
extra objective evaluation per iteration -- so the compute actually consumed is
reported next to effectiveness rather than left implicit.

Usage::

    python examples/heldout_evaluation.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import random
import statistics
import time

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm_efficiency_attack import Attacker, measure_cost

BENCHMARK = pathlib.Path("benchmarks/heldout_seq2seq_v1.json")
LOCK = pathlib.Path("benchmarks/STRATEGY_LOCK.txt")
BENCHMARK_SHA = "9cc6170a3441fa67c2c8602213d66fb1e4fdccaf14efddcb8d835f5031fd390c"
LOCK_SHA = "3ee3b7edddcfad356c23441c22cf479a4e8ab3425d40a5a0f1ba3e1a0ebc3fb1"

STRATEGIES = ("gradient", "gradient_stratified", "random")
CONFIG = {
    "objective": "eos_suppression",
    "max_iterations": 6,
    "perturbation_budget": 3,
    "top_k": 20,
    "max_new_tokens": 128,
    "objective_horizon": 8,
    "protected_prefix_tokens": 7,
    "seed": 0,
    "verbose": False,
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_ci(values, statistic=statistics.mean, resamples=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI. Reported because n=24 is small.

    This is an interval, not a significance test: with 24 paired observations a
    p-value would be theatre, but the spread of the resampled mean is honest
    about how much the point estimate could move.
    """
    rng = random.Random(seed)
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    draws = []
    for _ in range(resamples):
        draws.append(statistic([values[rng.randrange(n)] for _ in range(n)]))
    draws.sort()
    lo = draws[int((alpha / 2) * resamples)]
    hi = draws[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return (lo, hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="t5-small")
    parser.add_argument("--out", default="results/heldout_t5_small.json")
    args = parser.parse_args()

    for path, expected in ((BENCHMARK, BENCHMARK_SHA), (LOCK, LOCK_SHA)):
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(
                f"REFUSING TO RUN: {path} has changed.\n"
                f"  expected {expected}\n  actual   {actual}\n"
                "The held-out protocol requires the benchmark and the locked "
                "specification to be byte-identical to what was frozen."
            )
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    inputs = benchmark["inputs"]
    print(f"benchmark {BENCHMARK} verified, sha256 {BENCHMARK_SHA[:16]}..., n={len(inputs)}")
    print(f"lock      {LOCK} verified, sha256 {LOCK_SHA[:16]}...\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    benign = [
        measure_cost(model, tokenizer, text, max_new_tokens=CONFIG["max_new_tokens"])
        for text in inputs
    ]
    benign_len = [b["output_tokens"] for b in benign]
    benign_ceiling = sum(b["stopped_by"] == "max_tokens" for b in benign)
    print(f"benign: mean {statistics.mean(benign_len):.2f}, median "
          f"{statistics.median(benign_len)}, range {min(benign_len)}-{max(benign_len)}, "
          f"{benign_ceiling}/{len(inputs)} at the ceiling\n")

    results = {}
    for strategy in STRATEGIES:
        config = dict(CONFIG, strategy=strategy)
        rows = []
        started = time.perf_counter()
        for text, base, base_len in zip(inputs, benign, benign_len):
            adv_x, logs = Attacker(model, tokenizer).run(text, config)
            p, c, a = logs["perturbation"], logs["cost"], logs["attack_cost"]
            rows.append({
                "input": text,
                "adv_x": adv_x,
                "benign_output_tokens": base_len,
                "adversarial_output_tokens": c["adversarial_output_tokens"],
                "delta": c["adversarial_output_tokens"] - base_len,
                "ratio": c["output_token_ratio"],
                "benign_stopped_by": base["stopped_by"],
                "adversarial_stopped_by": logs["adversarial"]["stopped_by"],
                "censoring": logs["censored"]["interpretation"],
                "hamming_distance": p["hamming_distance"],
                "positions_touched": p["positions_touched"],
                "round_trip_exact": p["round_trip_exact"],
                "objective_evaluations": a["objective_evaluations"],
                "search_model_forwards": a["search_model_forwards"],
                "total_model_forwards": a["total_model_forwards"],
            })
        elapsed = time.perf_counter() - started

        deltas = [r["delta"] for r in rows]
        hammings = [r["hamming_distance"] for r in rows]
        lo, hi = bootstrap_ci(deltas)
        results[strategy] = {
            "strategy": strategy,
            "n": len(rows),
            "mean_benign": statistics.mean(benign_len),
            "mean_adversarial": statistics.mean(
                r["adversarial_output_tokens"] for r in rows),
            "mean_delta": statistics.mean(deltas),
            "median_delta": statistics.median(deltas),
            "mean_delta_ci95": [lo, hi],
            "improved": sum(d > 0 for d in deltas),
            "unchanged": sum(d == 0 for d in deltas),
            "worsened": sum(d < 0 for d in deltas),
            "censoring_counts": {
                k: sum(r["censoring"] == k for r in rows)
                for k in ("point_estimate", "lower_bound", "upper_bound", "uninformative")
            },
            "mean_hamming": statistics.mean(hammings),
            "hamming_histogram": {
                str(h): hammings.count(h) for h in sorted(set(hammings))},
            "zero_edit_runs": sum(h == 0 for h in hammings),
            "budget_fully_used": sum(
                h == CONFIG["perturbation_budget"] for h in hammings),
            "round_trip_exact_all": all(r["round_trip_exact"] for r in rows),
            "objective_evaluations": sum(r["objective_evaluations"] for r in rows),
            "search_model_forwards": sum(r["search_model_forwards"] for r in rows),
            "total_model_forwards": sum(r["total_model_forwards"] for r in rows),
            "wall_seconds": elapsed,
            "rows": rows,
        }
        r = results[strategy]
        print(f"{strategy:<22} mean delta {r['mean_delta']:+.2f} "
              f"[{lo:+.2f}, {hi:+.2f}]  median {r['median_delta']:+.1f}  "
              f"improved {r['improved']}/{r['n']}  ham {r['mean_hamming']:.2f}  "
              f"fwd {r['search_model_forwards']}  {elapsed:.0f}s")

    # Paired differences against the random control, with bootstrap intervals.
    print("\npaired difference vs random (positive = strategy beats random):")
    paired = {}
    random_deltas = [r["delta"] for r in results["random"]["rows"]]
    for strategy in STRATEGIES:
        if strategy == "random":
            continue
        diffs = [a["delta"] - b for a, b in
                 zip(results[strategy]["rows"], random_deltas)]
        lo, hi = bootstrap_ci(diffs)
        paired[strategy] = {
            "mean_difference": statistics.mean(diffs),
            "median_difference": statistics.median(diffs),
            "ci95": [lo, hi],
            "wins": sum(d > 0 for d in diffs),
            "ties": sum(d == 0 for d in diffs),
            "losses": sum(d < 0 for d in diffs),
        }
        q = paired[strategy]
        print(f"  {strategy:<22} {q['mean_difference']:+.2f} "
              f"[{lo:+.2f}, {hi:+.2f}]   "
              f"wins {q['wins']}  ties {q['ties']}  losses {q['losses']}")

    payload = {
        "protocol": {
            "benchmark": BENCHMARK.as_posix(),
            "benchmark_sha256": BENCHMARK_SHA,
            "lock": LOCK.as_posix(),
            "lock_sha256": LOCK_SHA,
            "frozen_before_any_result": True,
            "single_run": True,
        },
        "environment": {
            "model": args.model,
            "tokenizer": type(tokenizer).__name__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(next(model.parameters()).device),
        },
        "config": CONFIG,
        "inputs": inputs,
        "benign_output_tokens": benign_len,
        "benign_at_ceiling": benign_ceiling,
        "results": results,
        "paired_vs_random": paired,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
