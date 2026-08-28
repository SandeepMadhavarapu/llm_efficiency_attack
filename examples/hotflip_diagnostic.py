"""How good is the HotFlip first-order ranking, really?

Why this script exists
----------------------
The attack ranks candidate substitutions by the first-order estimate

    dJ(i, v)  ~=  (e_v - e_i) . grad_i

and then re-scores the top `k` of them exactly. Two separate questions follow,
and conflating them hides where the method actually fails:

1. Does the *objective* track generated length?  -> examples/objective_diagnostic.py
2. Does the *first-order estimate* track the objective?  -> this script

`examples/ablation.py` shows that gradient-guided search loses to a random
control on t5-small. This script is the diagnosis: it measures how well the
estimate predicts the exact objective change it is standing in for.

What is measured
----------------
For each sampled substitution the script records the first-order estimate and
the exact objective delta obtained by actually running the objective on the
substituted sequence, then reports:

* Spearman rho(estimate, exact delta), within input and pooled. The estimate is
  used only for *ranking*, so rank correlation is the relevant statistic. A
  POSITIVE rho confirms the sign convention: the estimate and the true change
  move together, and taking the most negative estimates selects the substitutions
  that most reduce the objective.
* Sign agreement: how often the estimate and the exact delta agree on whether a
  substitution improves the objective at all.
* Precision of the improvement prediction: of the candidates the estimate says
  will improve, how many actually do.
* Selection quality: the mean true rank of the candidates the estimate ranks
  best, against the chance value. This is the quantity the search consumes.

Fidelity to production
----------------------
The estimate is recomputed here rather than imported, so the script asserts that
its top-k selection is identical to `Attacker._gradient_candidates` on the same
input. If that assertion ever fails, this diagnostic has drifted from the code it
claims to describe and the numbers should not be trusted.

Usage::

    python examples/hotflip_diagnostic.py
    python examples/hotflip_diagnostic.py --samples 40 --seed 1
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm_efficiency_attack.adapters import ModelAdapter, forbidden_token_ids
from llm_efficiency_attack.attacker import Attacker, _ComputeCounters
from llm_efficiency_attack.config import AttackConfig
from llm_efficiency_attack.objectives import get_objective

INPUTS = [
    "translate English to German: The house is wonderful.",
    "translate English to German: She walked to the market this morning.",
    "translate English to German: The weather today is unusually cold.",
    "translate English to German: He forgot his keys on the kitchen table.",
    "translate English to German: They are building a new library downtown.",
    "summarize: The committee met on Tuesday to review the annual budget report.",
    "translate English to French: Good morning, how are you today?",
]
PROTECTED = 7


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, written out so the script has no scipy dependency."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    ) ** 0.5
    return num / den if den else float("nan")


def hotflip_estimates(adapter, objective_fn, eos_ids, ids, mask, horizon):
    """The first-order estimate matrix, mirroring `_gradient_candidates`.

    Returns `(seq_len, vocab)` where entry `(i, v)` estimates the change in the
    objective from replacing position `i` with token `v`.
    """
    embeds = adapter.embed(ids)
    stop_logits = adapter.stop_logits(embeds, mask, horizon)
    loss = objective_fn(stop_logits, eos_ids)
    adapter.model.zero_grad(set_to_none=True)
    loss.backward()

    grad = embeds.grad[0]
    table = adapter.embedding_matrix()
    scale = adapter.embedding_scale()
    scores = (grad @ table.T) * scale
    current = table[ids[0]] * scale
    return (scores - (grad * current).sum(-1, keepdim=True)).detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="t5-small")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--samples", type=int, default=30, help="substitutions per input")
    parser.add_argument("--top-n", type=int, default=10, help="shortlist size to score")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/hotflip_diagnostic_t5_small.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()
    adapter = ModelAdapter.for_model(model, tokenizer)
    objective_fn = get_objective("eos_suppression")
    eos_ids = adapter.eos_token_ids()

    rows_total = adapter.embedding_matrix().shape[0]
    illegal = forbidden_token_ids(tokenizer, rows_total)
    legal = [i for i in range(rows_total) if i not in illegal]
    specials = set(tokenizer.all_special_ids or [])
    rng = random.Random(args.seed)

    attacker = Attacker(model, tokenizer)
    cfg = AttackConfig.from_dict(
        {"objective_horizon": args.horizon, "top_k": args.top_n,
         "protected_prefix_tokens": PROTECTED, "seed": args.seed}
    )

    def exact(ids, mask):
        return attacker._score_exact(
            ids, mask, objective_fn, eos_ids, cfg, False, _ComputeCounters()
        )

    per_input, rows = [], []
    for index, text in enumerate(INPUTS):
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        mask = torch.ones_like(ids)
        positions = [
            i for i in range(PROTECTED, ids.shape[1])
            if int(ids[0, i]) not in specials
        ]

        estimates = hotflip_estimates(
            adapter, objective_fn, eos_ids, ids, mask, args.horizon
        )

        # Fidelity check: this script's ranking must be the production ranking.
        mine = _top_k_from(estimates, positions, illegal, ids, args.top_n)
        theirs = attacker._gradient_candidates(
            ids, mask, objective_fn, eos_ids, cfg, positions, illegal,
            False, _ComputeCounters(),
        )
        assert mine == theirs, (
            "this diagnostic's estimate has drifted from _gradient_candidates; "
            f"{mine[:3]} != {theirs[:3]}"
        )

        base = exact(ids, mask)
        sampled = []
        attempts = 0
        while len(sampled) < args.samples and attempts < args.samples * 20:
            attempts += 1
            position = rng.choice(positions)
            token = rng.choice(legal)
            if token == int(ids[0, position]):
                continue
            trial = ids.clone()
            trial[0, position] = token
            # Only candidates the attack could actually commit.
            if not attacker._round_trips(trial):
                continue
            sampled.append(
                {
                    "input_index": index,
                    "position": position,
                    "token_id": token,
                    "estimate": float(estimates[position, token].item()),
                    "exact_delta": exact(trial, mask) - base,
                }
            )

        rows.extend(sampled)
        est = [s["estimate"] for s in sampled]
        act = [s["exact_delta"] for s in sampled]
        per_input.append(
            {"input": text, "n": len(sampled), "spearman": spearman(est, act)}
        )

    est = [r["estimate"] for r in rows]
    act = [r["exact_delta"] for r in rows]

    rho_pooled = spearman(est, act)
    rho_within = statistics.mean(p["spearman"] for p in per_input)
    agree = sum((e < 0) == (a < 0) for e, a in zip(est, act))
    predicted_improve = sum(1 for e in est if e < 0)
    truly_improve = sum(1 for e, a in zip(est, act) if e < 0 and a < 0)

    # Selection quality: mean TRUE rank of the top-n by estimate (0 = perfect).
    by_est = sorted(range(len(rows)), key=lambda i: est[i])
    true_rank = {idx: r for r, idx in enumerate(sorted(range(len(rows)), key=lambda i: act[i]))}
    top_ranks = [true_rank[i] for i in by_est[: args.top_n]]
    mean_true_rank = statistics.mean(top_ranks)
    chance = (len(rows) - 1) / 2

    print(f"model: {args.model}   horizon: {args.horizon}   seed: {args.seed}")
    print(f"tokenizer: {type(tokenizer).__name__}   inputs: {len(INPUTS)}")
    print(f"admissible substitutions sampled: {len(rows)}")
    print(f"legal token ids: {len(legal)} of {rows_total} embedding rows "
          f"({len(illegal)} excluded)\n")

    print(f"Spearman rho(estimate, exact delta), pooled : {rho_pooled:+.3f}")
    print(f"Spearman rho, within input then averaged    : {rho_within:+.3f}")
    print("  positive rho confirms the sign convention: most-negative estimates")
    print("  are the substitutions that most reduce the objective.\n")
    print(f"sign agreement                : {agree}/{len(rows)} = {agree/len(rows):.1%}")
    print(f"estimate predicts improvement : {predicted_improve} candidates")
    print(f"  of those, actually improve  : {truly_improve} "
          f"({truly_improve/predicted_improve:.1%})" if predicted_improve else "")
    print(f"top-{args.top_n} by estimate, mean TRUE rank : "
          f"{mean_true_rank:.1f}/{len(rows)}  (chance {chance:.1f}, perfect 0)")

    payload = {
        "environment": {
            "model": args.model,
            "tokenizer": type(tokenizer).__name__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "device": str(next(model.parameters()).device),
        },
        "inputs": INPUTS,
        "seed": args.seed,
        "horizon": args.horizon,
        "objective": "eos_suppression",
        "protected_prefix_tokens": PROTECTED,
        "embedding_rows": rows_total,
        "legal_token_ids": len(legal),
        "excluded_token_ids": len(illegal),
        "candidates": len(rows),
        "spearman_pooled": rho_pooled,
        "spearman_within_input_mean": rho_within,
        "spearman_per_input": per_input,
        "sign_agreement": agree,
        "sign_agreement_fraction": agree / len(rows),
        "estimate_predicts_improvement": predicted_improve,
        "of_those_actually_improve": truly_improve,
        "top_n": args.top_n,
        "top_n_mean_true_rank": mean_true_rank,
        "chance_mean_true_rank": chance,
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwritten to {args.out}")


def _top_k_from(estimates, positions, illegal, ids, k):
    """Reproduce `_gradient_candidates`' masking and top-k selection."""
    scores = estimates.clone()
    mask = torch.full_like(scores, float("inf"))
    mask[torch.tensor(positions, dtype=torch.long)] = 0.0
    scores = scores + mask
    if illegal:
        scores[:, torch.tensor(sorted(illegal), dtype=torch.long)] = float("inf")
    scores[torch.arange(scores.shape[0]), ids[0]] = float("inf")
    flat = scores.flatten()
    k = min(k, int(torch.isfinite(flat).sum().item()))
    _, idx = torch.topk(-flat, k)
    vocab = scores.shape[1]
    return [(int(i.item() // vocab), int(i.item() % vocab)) for i in idx]


if __name__ == "__main__":
    main()
