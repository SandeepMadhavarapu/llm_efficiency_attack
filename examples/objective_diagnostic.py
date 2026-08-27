"""Which stop-suppression objective best tracks real generation cost?

Why this script exists
----------------------
`objective_horizon` is an upper bound, not a step count. The stop-logits the
objective reads come from the model's *own* greedy trajectory, which ends when
the model emits EOS -- so a horizon of 8 can yield 4 rows on one candidate and 8
on another. The current `eos_suppression` objective takes the mean over however
many rows it gets, which means candidates are being compared by means over
different numbers of terms.

That is a real comparability problem, but the fix is not obvious, so this
measures three formulations before anything is changed:

  mean   mean_t log P(stop at t)  over the realised greedy trajectory  [current]
  sum    sum_t  log P(stop at t)  over the realised greedy trajectory
  fixed  mean_t log P(stop at t)  over exactly `horizon` forced steps

`fixed` uses `min_new_tokens=horizon` so generation cannot stop early: every
candidate is scored on exactly the same number of terms. It answers a
counterfactual -- "if you were forced to keep going, how reluctant would you be
to stop at each of the first H steps" -- rather than scoring the path the model
would really take.

What is reported, and why
-------------------------
* Spearman rho(objective, output_tokens). All three objectives are *minimised*,
  so a NEGATIVE rho is good: lower objective should mean longer output.
* Spearman rho(objective, trajectory rows). This is the structural-bias check.
  If an objective improves simply because the trajectory got longer, it is partly
  measuring length rather than stop-reluctance, and the search can be rewarded
  for the wrong reason.
* Top-10 selection quality: the mean output length of the ten candidates each
  objective likes best, against the pool mean. This is what the search actually
  consumes.

Usage::

    python examples/objective_diagnostic.py
    python examples/objective_diagnostic.py --samples 60 --horizon 8
"""

from __future__ import annotations

import argparse
import json
import random
import statistics

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm_efficiency_attack import measure_cost
from llm_efficiency_attack.adapters import ModelAdapter, forbidden_token_ids

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


def stop_logits_free(adapter, embeds, mask, horizon):
    """The production path: score the model's own greedy trajectory."""
    return adapter.stop_logits(embeds, mask, horizon)


def stop_logits_fixed(adapter, embeds, mask, horizon):
    """Force exactly `horizon` decoder steps, so every candidate gets H rows."""
    model = adapter.model
    with torch.no_grad():
        decoded = model.generate(
            inputs_embeds=embeds.detach(),
            attention_mask=mask,
            min_new_tokens=horizon,
            max_new_tokens=horizon,
            do_sample=False,
        )
    out = model(
        inputs_embeds=embeds,
        attention_mask=mask,
        decoder_input_ids=decoded[:, :-1],
    )
    return out.logits[0]


def eos_log_prob(stop_logits: torch.Tensor, eos_ids: list[int]) -> torch.Tensor:
    log_probs = torch.log_softmax(stop_logits, dim=-1)
    return torch.logsumexp(log_probs[:, eos_ids], dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="t5-small")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--samples", type=int, default=40, help="substitutions per input")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/objective_diagnostic_t5_small.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()
    adapter = ModelAdapter.for_model(model, tokenizer)
    eos_ids = adapter.eos_token_ids()
    illegal = forbidden_token_ids(tokenizer, adapter.embedding_matrix().shape[0])
    legal = [i for i in range(adapter.embedding_matrix().shape[0]) if i not in illegal]
    rng = random.Random(args.seed)

    rows: list[dict] = []
    for input_index, text in enumerate(INPUTS):
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        mask = torch.ones_like(ids)
        positions = [
            i
            for i in range(PROTECTED, ids.shape[1])
            if int(ids[0, i]) not in set(tokenizer.all_special_ids or [])
        ]
        for _ in range(args.samples):
            trial = ids.clone()
            position = rng.choice(positions)
            token = rng.choice(legal)
            while token == int(trial[0, position]):
                token = rng.choice(legal)
            trial[0, position] = token

            # Only score candidates the attack would actually be allowed to
            # commit, so the diagnostic describes the real search space.
            decoded = tokenizer.decode(trial[0], skip_special_tokens=True)
            if tokenizer(decoded)["input_ids"] != trial[0].tolist():
                continue

            cost = measure_cost(
                model, tokenizer, decoded, max_new_tokens=args.max_new_tokens
            )
            with torch.no_grad():
                embeds = adapter.embed_values(trial)
                free = eos_log_prob(
                    stop_logits_free(adapter, embeds, mask, args.horizon), eos_ids
                )
                fixed = eos_log_prob(
                    stop_logits_fixed(adapter, embeds, mask, args.horizon), eos_ids
                )
            rows.append(
                {
                    "input_index": input_index,
                    "output_tokens": cost["output_tokens"],
                    "stopped_by": cost["stopped_by"],
                    "trajectory_rows": int(free.shape[0]),
                    "mean": float(free.mean().item()),
                    "sum": float(free.sum().item()),
                    "fixed": float(fixed.mean().item()),
                }
            )

    lengths = [r["output_tokens"] for r in rows]
    traj = [r["trajectory_rows"] for r in rows]
    pool_mean = statistics.mean(lengths)

    print(f"model: {args.model}   horizon: {args.horizon}   seed: {args.seed}")
    print(f"{len(rows)} admissible substitutions over {len(INPUTS)} inputs")
    print(f"output tokens: mean {pool_mean:.1f}, range {min(lengths)}..{max(lengths)}")
    print(f"trajectory rows actually scored: range {min(traj)}..{max(traj)} "
          f"(horizon requested: {args.horizon})")
    censored = sum(r["stopped_by"] == "max_tokens" for r in rows)
    print(f"right-censored candidates: {censored}/{len(rows)}")
    print()

    by_input: dict[int, list[dict]] = {}
    for row in rows:
        by_input.setdefault(row["input_index"], []).append(row)

    header = (
        f"{'objective':<10}{'rho within':>12}{'rho pooled':>12}"
        f"{'top5 gain':>11}{'rho vs rows':>13}"
    )
    print(header)
    print("-" * len(header))

    summary = {}
    for name in ("mean", "sum", "fixed"):
        # Within-input correlation is the one that matters: the search only ever
        # compares candidates derived from the SAME input. Pooling across inputs
        # mixes in "this sentence is naturally longer", which the attack does not
        # get to choose and which swamps the substitution effect.
        per_input_rho, per_input_gain = [], []
        for group in by_input.values():
            if len(group) < 5:
                continue
            values = [g[name] for g in group]
            lens = [g["output_tokens"] for g in group]
            per_input_rho.append(spearman(values, lens))
            best = sorted(range(len(group)), key=lambda i: values[i])[:5]
            per_input_gain.append(
                statistics.mean(lens[i] for i in best) - statistics.mean(lens)
            )

        rho_within = statistics.mean(per_input_rho)
        rho_pooled = spearman([r[name] for r in rows], lengths)
        gain = statistics.mean(per_input_gain)

        # Structural bias: does the score improve merely because the trajectory
        # is longer? `fixed` scores a constant number of rows by construction, so
        # the question does not apply to it.
        if name == "fixed":
            rho_rows = float("nan")
        else:
            rho_rows = spearman([r[name] for r in rows], traj)

        summary[name] = {
            "spearman_within_input_mean": rho_within,
            "spearman_within_input_per_input": per_input_rho,
            "spearman_pooled_across_inputs": rho_pooled,
            "top5_within_input_mean_gain_tokens": gain,
            "spearman_vs_trajectory_rows": None if name == "fixed" else rho_rows,
            "scored_rows": "constant (= horizon)" if name == "fixed" else "variable",
        }
        rows_cell = "n/a (const)" if name == "fixed" else f"{rho_rows:.3f}"
        print(f"{name:<10}{rho_within:>12.3f}{rho_pooled:>12.3f}"
              f"{gain:>+11.1f}{rows_cell:>13}")

    print()
    print("reading:")
    print("  rho within  -- Spearman(objective, output tokens) computed separately")
    print("                 per input and averaged. Objectives are MINIMISED, so")
    print("                 more NEGATIVE is a better proxy. This is the number")
    print("                 that matters: the search only compares candidates")
    print("                 built from the same input.")
    print("  rho pooled  -- the same across all inputs at once. Weaker for all")
    print("                 three because it also contains between-sentence")
    print("                 length variation the attack cannot influence.")
    print("  top5 gain   -- extra output tokens, vs that input's own mean, for the")
    print("                 five candidates the objective ranks best.")
    print("  rho vs rows -- structural bias: correlation between the score and how")
    print("                 many trajectory rows it summed over.")

    print()
    print("mean objective value by trajectory length (structural-bias detail):")
    print(f"  {'rows':<6}{'n':>5}{'mean':>10}{'sum':>10}{'fixed':>10}{'out tok':>10}")
    length_profile = {}
    for r_count in sorted(set(traj)):
        group = [r for r in rows if r["trajectory_rows"] == r_count]
        entry = {
            "n": len(group),
            "mean": statistics.mean(g["mean"] for g in group),
            "sum": statistics.mean(g["sum"] for g in group),
            "fixed": statistics.mean(g["fixed"] for g in group),
            "output_tokens": statistics.mean(g["output_tokens"] for g in group),
        }
        length_profile[r_count] = entry
        print(f"  {r_count:<6}{entry['n']:>5}{entry['mean']:>10.2f}"
              f"{entry['sum']:>10.2f}{entry['fixed']:>10.2f}"
              f"{entry['output_tokens']:>10.1f}")
    summary_extra = {"length_profile": length_profile}

    payload = {
        "model": args.model,
        "horizon": args.horizon,
        "seed": args.seed,
        "samples_per_input": args.samples,
        "inputs": INPUTS,
        "admissible_candidates": len(rows),
        "pool_mean_output_tokens": pool_mean,
        "trajectory_rows_range": [min(traj), max(traj)],
        "right_censored_candidates": censored,
        "summary": summary,
        "length_profile": summary_extra["length_profile"],
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
