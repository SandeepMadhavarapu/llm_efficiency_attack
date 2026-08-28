"""Where exactly does gradient-guided search lose to random?

`examples/ablation.py` establishes *that* the gradient loses at every candidate
budget tested. `examples/hotflip_diagnostic.py` establishes that the first-order
estimate correlates only weakly with the exact objective change. Neither says
*which stage* of the search is responsible, and that is the question a reviewer
will actually ask.

This decomposes the failure. For each input it builds a reference set by exactly
scoring many random legal substitutions, which gives a ground-truth ranking of a
sample of the candidate space. Against that reference it measures:

1. **Coverage.** What fraction of the legal candidate space does a `top_k`
   shortlist even look at?
2. **Token selection.** Where do the gradient's proposals fall in the reference
   distribution of exact objective changes?
3. **Position vs token error.** Does the gradient pick the wrong *position*, or
   the right position and the wrong *replacement token*? These call for different
   fixes, and conflating them is how people "improve" the wrong stage.
4. **Headroom at the chosen position.** How much better could the search have
   done without changing position?
5. **Realization filtering.** Does the round-trip filter remove gradient
   proposals more often than random ones? If so, part of the gap is bookkeeping
   rather than ranking.
6. **Objective-to-cost alignment.** Among committed-quality candidates, how well
   does an exact objective improvement translate into more generated tokens?

Budget utilisation for both strategies is read from the committed ablation
artifact rather than recomputed, so no attack is re-run here.

This is diagnosis, not tuning. It changes nothing and proposes nothing.

Usage::

    python examples/search_diagnostic.py
    python examples/search_diagnostic.py --samples-per-position 40
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import platform
import random
import statistics

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm_efficiency_attack import measure_cost
from llm_efficiency_attack.adapters import forbidden_token_ids
from llm_efficiency_attack.attacker import Attacker, _ComputeCounters
from llm_efficiency_attack.config import AttackConfig
from llm_efficiency_attack.objectives import get_objective

INPUTS = [
    "translate English to German: The house is wonderful.",
    "translate English to German: She walked to the market this morning.",
    "translate English to German: The weather today is unusually cold.",
    "summarize: The committee met on Tuesday to review the annual budget report.",
]
PROTECTED = 7
HORIZON = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="t5-small")
    parser.add_argument("--samples-per-position", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ablation", default="results/ablation_t5_small.json")
    parser.add_argument("--out", default="results/search_diagnostic_t5_small.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()
    attacker = Attacker(model, tokenizer)
    adapter = attacker.adapter
    objective_fn = get_objective("eos_suppression")
    eos_ids = adapter.eos_token_ids()
    cfg = AttackConfig.from_dict(
        {"objective_horizon": HORIZON, "top_k": args.top_k,
         "protected_prefix_tokens": PROTECTED, "seed": args.seed}
    )

    rows_total = adapter.embedding_matrix().shape[0]
    illegal = forbidden_token_ids(tokenizer, rows_total)
    legal = [i for i in range(rows_total) if i not in illegal]
    specials = set(tokenizer.all_special_ids or [])
    rng = random.Random(args.seed)

    def exact(ids, mask):
        return attacker._score_exact(
            ids, mask, objective_fn, eos_ids, cfg, False, _ComputeCounters()
        )

    per_input = []
    for text in INPUTS:
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        mask = torch.ones_like(ids)
        positions = [
            i for i in range(PROTECTED, ids.shape[1])
            if int(ids[0, i]) not in specials
        ]
        base = exact(ids, mask)

        # --- reference set: exactly scored random legal substitutions ---------
        reference = []  # (position, token, delta)
        for position in positions:
            for _ in range(args.samples_per_position):
                token = rng.choice(legal)
                if token == int(ids[0, position]):
                    continue
                trial = ids.clone()
                trial[0, position] = token
                if not attacker._round_trips(trial):
                    continue
                reference.append((position, token, exact(trial, mask) - base))

        if not reference:
            continue
        deltas = [d for _, _, d in reference]
        best_delta = min(deltas)
        best_position = min(reference, key=lambda r: r[2])[0]

        # --- gradient proposals ----------------------------------------------
        proposals = attacker._gradient_candidates(
            ids, mask, objective_fn, eos_ids, cfg, positions, illegal,
            False, _ComputeCounters(),
        )
        # How concentrated is the shortlist? `topk` runs over the flattened
        # (positions x vocab) matrix, so if one position dominates the gradient
        # magnitude every candidate can come from it -- which is what stalls the
        # search once that position has been edited.
        position_counts = collections.Counter(p for p, _ in proposals)
        rejected = 0
        scored = []
        for position, token in proposals:
            trial = ids.clone()
            trial[0, position] = token
            if not attacker._round_trips(trial):
                rejected += 1
                continue
            scored.append((position, token, exact(trial, mask) - base))

        if not scored:
            continue
        grad_best = min(scored, key=lambda r: r[2])

        # Percentile of each gradient proposal inside the reference set: 0 means
        # better than everything sampled, 100 means worse than everything.
        def percentile(delta):
            return 100.0 * sum(d < delta for d in deltas) / len(deltas)

        # Headroom decomposition: how much of the gap to the sampled best is due
        # to picking the wrong position, and how much to the wrong token?
        same_position = [d for p, _, d in reference if p == grad_best[0]]
        best_at_grad_position = min(same_position) if same_position else float("nan")

        entry = {
            "input": text,
            "positions": len(positions),
            "reference_candidates": len(reference),
            "legal_vocabulary": len(legal),
            "coverage_fraction": args.top_k / (len(positions) * len(legal)),
            "base_objective": base,
            "reference_best_delta": best_delta,
            "reference_best_position": best_position,
            "reference_mean_delta": statistics.mean(deltas),
            "reference_improving_fraction": sum(d < 0 for d in deltas) / len(deltas),
            "gradient_proposals": len(proposals),
            "gradient_distinct_positions": len(position_counts),
            "gradient_top_position_share":
                max(position_counts.values()) / len(proposals),
            "gradient_rejected_by_round_trip": rejected,
            "gradient_best_delta": grad_best[2],
            "gradient_best_position": grad_best[0],
            "gradient_best_percentile": percentile(grad_best[2]),
            "gradient_median_percentile": statistics.median(
                percentile(d) for _, _, d in scored
            ),
            "gradient_picked_reference_best_position":
                grad_best[0] == best_position,
            "best_delta_at_gradient_position": best_at_grad_position,
            "position_error": best_at_grad_position - best_delta,
            "token_error": grad_best[2] - best_at_grad_position,
        }
        per_input.append(entry)
        print(f"\n{text[:60]}")
        print(f"  positions {entry['positions']}, reference candidates "
              f"{entry['reference_candidates']}, coverage "
              f"{entry['coverage_fraction']:.2e} of legal space")
        print(f"  reference best delta {best_delta:+.4f} at position {best_position}; "
              f"{entry['reference_improving_fraction']:.1%} of random legal "
              f"substitutions improve")
        print(f"  gradient best delta  {grad_best[2]:+.4f} at position "
              f"{grad_best[0]}  (percentile {entry['gradient_best_percentile']:.1f})")
        print(f"  median proposal percentile {entry['gradient_median_percentile']:.1f}"
              f"   round-trip rejected {rejected}/{len(proposals)}")
        print(f"  shortlist concentration: {entry['gradient_distinct_positions']}"
              f"/{len(positions)} distinct positions, top position holds "
              f"{entry['gradient_top_position_share']:.0%} of candidates")
        print(f"  headroom: position error {entry['position_error']:+.4f}, "
              f"token error {entry['token_error']:+.4f}")

    # --- objective improvement vs realised cost change ------------------------
    print("\n" + "=" * 72)
    print("objective improvement -> generated-token change")
    print("=" * 72)
    ids = tokenizer(INPUTS[0], return_tensors="pt")["input_ids"]
    mask = torch.ones_like(ids)
    base = exact(ids, mask)
    base_cost = measure_cost(model, tokenizer, INPUTS[0], max_new_tokens=128)["output_tokens"]
    pairs = []
    positions = [i for i in range(PROTECTED, ids.shape[1])
                 if int(ids[0, i]) not in specials]
    for _ in range(80):
        trial = ids.clone()
        position = rng.choice(positions)
        token = rng.choice(legal)
        if token == int(trial[0, position]):
            continue
        trial[0, position] = token
        if not attacker._round_trips(trial):
            continue
        d_obj = exact(trial, mask) - base
        text = tokenizer.decode(trial[0], skip_special_tokens=True)
        d_cost = measure_cost(model, tokenizer, text, max_new_tokens=128)["output_tokens"] - base_cost
        pairs.append((d_obj, d_cost))
    improving = [(o, c) for o, c in pairs if o < 0]
    print(f"  {len(pairs)} admissible substitutions; {len(improving)} improve the objective")
    if improving:
        print(f"    of those, {sum(c > 0 for _, c in improving)} increase generated tokens, "
              f"{sum(c < 0 for _, c in improving)} decrease, "
              f"{sum(c == 0 for _, c in improving)} unchanged")
        print(f"    mean token change among objective-improving: "
              f"{statistics.mean(c for _, c in improving):+.2f}")

    # --- budget utilisation, read from the committed ablation -----------------
    print("\n" + "=" * 72)
    print("budget utilisation (from the committed ablation artifact)")
    print("=" * 72)
    utilisation = {}
    ablation_path = pathlib.Path(args.ablation)
    if ablation_path.exists():
        ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
        print(f"  {'strategy':<10}{'top_k':>7}{'zero-edit':>11}{'ham<budget':>12}"
              f"{'mean ham':>10}")
        for label, block in ablation["results"].items():
            rows = block["per_input"]
            budget = ablation["environment"]["base_config"]["perturbation_budget"]
            zero = sum(r["hamming_distance"] == 0 for r in rows)
            under = sum(r["hamming_distance"] < budget for r in rows)
            mean_ham = statistics.mean(r["hamming_distance"] for r in rows)
            utilisation[label] = {
                "runs": len(rows), "zero_edit": zero, "hamming_below_budget": under,
                "mean_hamming": mean_ham, "budget": budget,
            }
            print(f"  {block['strategy']:<10}{block['top_k']:>7}"
                  f"{zero:>6}/{len(rows)}{under:>8}/{len(rows)}{mean_ham:>10.2f}")
    else:
        print(f"  {ablation_path} not found; skipping")

    payload = {
        "environment": {
            "model": args.model, "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
        },
        "seed": args.seed, "top_k": args.top_k, "horizon": HORIZON,
        "samples_per_position": args.samples_per_position,
        "inputs": INPUTS,
        "per_input": per_input,
        "objective_to_cost": {
            "input": INPUTS[0],
            "pairs": [{"delta_objective": o, "delta_tokens": c} for o, c in pairs],
            "n": len(pairs),
            "n_objective_improving": len(improving),
            "of_those_cost_increased": sum(c > 0 for _, c in improving),
            "of_those_cost_decreased": sum(c < 0 for _, c in improving),
            "mean_token_change_among_improving": (
                statistics.mean(c for _, c in improving) if improving else None
            ),
        },
        "budget_utilisation": utilisation,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
