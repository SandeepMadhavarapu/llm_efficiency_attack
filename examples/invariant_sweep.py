"""Check the attack's safety invariants across many real runs.

The unit suite asserts these invariants on toy fixtures and the integration suite
asserts them on real checkpoints, but both are single runs. This sweeps a real
model across strategies, seeds and inputs and checks every run, so the claim in
RESULTS.md ("zero violations across N runs") has a reproducer rather than being
an assertion.

The invariants, and why each matters:

* `round_trip_exact` -- the returned text re-tokenises to exactly the optimised
  ids. Without it the perturbation budget does not apply to what the caller
  actually feeds the model.
* `hamming <= touched <= budget` -- the budget bounds positions the search wrote
  to; Hamming counts positions whose final token differs. Conflating them would
  overstate the guarantee.
* no committed token is a special id or an embedding row past `len(tokenizer)` --
  both decode away, changing the input's length.
* no committed substitution is a no-op, the protected prefix is untouched, and
  the sequence length is unchanged.

It also reports how often the round-trip filter actually rejected a candidate,
which is the evidence that the filter is doing work rather than sitting unused.

Usage::

    python examples/invariant_sweep.py
    python examples/invariant_sweep.py --seeds 5 --top-k 60
"""

from __future__ import annotations

import argparse
import json
import platform

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm_efficiency_attack import Attacker
from llm_efficiency_attack.adapters import forbidden_token_ids

INPUTS = [
    "translate English to German: The house is wonderful.",
    "translate English to German: She walked to the market this morning.",
    "translate English to German: He forgot his keys on the kitchen table.",
    "summarize: The committee met on Tuesday to review the annual budget report.",
]
PROTECTED = 7


def check_run(logs, illegal, protected):
    """Return {invariant: bool} for one completed run."""
    p = logs["perturbation"]
    original, adversarial = p["original_token_ids"], p["adversarial_token_ids"]
    changed = p["positions_changed"]
    return {
        "round_trip_exact": p["round_trip_exact"] is True,
        "hamming_le_touched": p["hamming_distance"] <= p["positions_touched"],
        "touched_le_budget": p["positions_touched"] <= p["budget"],
        "hamming_is_true_count": p["hamming_distance"]
        == sum(a != b for a, b in zip(original, adversarial)),
        "touched_matches_positions": p["positions_touched"] == len(changed),
        "no_illegal_token_committed": all(adversarial[i] not in illegal for i in changed),
        "no_noop_committed": all(adversarial[i] != original[i] for i in changed),
        "protected_prefix_intact": original[:protected] == adversarial[:protected],
        "length_unchanged": len(original) == len(adversarial),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="t5-small")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--out", default="results/invariant_sweep.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    rows_total = model.get_input_embeddings().weight.shape[0]
    illegal = forbidden_token_ids(tokenizer, rows_total)

    print(f"model {args.model}: {rows_total} embedding rows, "
          f"{len(tokenizer)} tokenizer entries, {len(illegal)} excluded ids "
          f"({len(tokenizer.all_special_ids)} special + "
          f"{rows_total - len(tokenizer)} surplus rows)")

    base = {
        "max_iterations": 6,
        "perturbation_budget": 3,
        "top_k": args.top_k,
        "max_new_tokens": 128,
        "protected_prefix_tokens": PROTECTED,
        "objective_horizon": 8,
    }

    runs, violations, rejected = [], [], 0
    for strategy in ("gradient", "random"):
        for seed in range(args.seeds):
            for text in INPUTS:
                config = dict(base, strategy=strategy, seed=seed)
                adv_x, logs = Attacker(model, tokenizer).run(text, config)
                checks = check_run(logs, illegal, PROTECTED)
                rejected += logs["attack_cost"]["candidates_rejected_non_round_trip"]

                # Independent of the logged flag: re-tokenise the returned text.
                realised = tokenizer(adv_x)["input_ids"]
                checks["realised_equals_optimised"] = (
                    realised == logs["perturbation"]["adversarial_token_ids"]
                )

                failed = [k for k, ok in checks.items() if not ok]
                if failed:
                    violations.append(
                        {"strategy": strategy, "seed": seed, "input": text,
                         "failed": failed, "perturbation": logs["perturbation"]}
                    )
                runs.append({
                    "strategy": strategy, "seed": seed, "input": text,
                    "adv_x": adv_x,
                    "hamming_distance": logs["perturbation"]["hamming_distance"],
                    "positions_touched": logs["perturbation"]["positions_touched"],
                    "checks_passed": not failed,
                })

    n = len(runs)
    print(f"\n{n} runs ({2} strategies x {args.seeds} seeds x {len(INPUTS)} inputs)"
          f" at top_k={args.top_k}")
    print(f"  invariant violations                  : {len(violations)}")
    print(f"  candidates rejected by round-trip filter: {rejected}")
    for v in violations[:5]:
        print(f"    VIOLATION {v['strategy']} seed={v['seed']} -> {v['failed']}")

    payload = {
        "environment": {
            "model": args.model,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "config": base,
        "seeds": args.seeds,
        "inputs": INPUTS,
        "embedding_rows": rows_total,
        "tokenizer_entries": len(tokenizer),
        "excluded_token_ids": len(illegal),
        "special_token_ids": len(tokenizer.all_special_ids),
        "runs": n,
        "invariant_violations": len(violations),
        "violations": violations,
        "candidates_rejected_non_round_trip": rejected,
        "per_run": runs,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
