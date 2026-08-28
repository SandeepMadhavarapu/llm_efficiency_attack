"""Validate the whole attack path on real NVIDIA CUDA hardware.

Why this exists
---------------
Most of this project was developed on a CPU-only build, so the CUDA path was
fixed and unit-pinned long before it could be executed. The bug it guards against
is specific and silent in the worst way: `transformers.generate()` performs no
device migration of its inputs, so a tensor left on the CPU while the model sits
on the GPU raises deep inside the first embedding lookup. `run()` measures the
benign cost before doing anything else, so on a GPU box the failure lands
immediately.

It has since been run on real hardware -- an RTX 5060, torch 2.11.0+cu128, 14/14
checks passing, with the headline result reproducing exactly on device (see
RESULTS.md section 10). It is kept as a gate rather than deleted, because CI has
no GPU and nothing else re-checks that path.

This script is the smallest thing that would have caught that, plus the
invariants a device bug could corrupt without raising. It drives the ordinary
public API rather than reimplementing any attack logic, so what it validates is
what a user runs.

It exits non-zero on any failure, so it is usable as a gate.

Running it
----------
On Google Colab, choose Runtime -> Change runtime type -> T4 GPU, then::

    !git clone https://github.com/SandeepMadhavarapu/llm_efficiency_attack.git
    %cd llm_efficiency_attack
    !pip install -q -e ".[dev,examples]"
    !python examples/cuda_smoke.py
    !python -m pytest -q -k cuda

On any ordinary NVIDIA Linux box::

    git clone https://github.com/SandeepMadhavarapu/llm_efficiency_attack.git
    cd llm_efficiency_attack
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev,examples]"
    python examples/cuda_smoke.py
    python -m pytest -q -k cuda

Both the script and `pytest -k cuda` matter: the script proves the end-to-end
public API on device, and the tests assert the same invariants from inside.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm_efficiency_attack import Attacker

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="t5-small",
                        help="the headline model, so results are comparable to RESULTS.md")
    parser.add_argument(
        "--text", default="translate English to German: The house is wonderful."
    )
    parser.add_argument("--protected-prefix", type=int, default=7)
    parser.add_argument("--out", default="results/cuda_smoke.json")
    args = parser.parse_args()

    print("=" * 72)
    print("CUDA smoke test")
    print("=" * 72)

    if not torch.cuda.is_available():
        print("\nFAILED: CUDA is not available in this environment.")
        print(f"  torch                : {torch.__version__}")
        print(f"  torch.version.cuda   : {torch.version.cuda}")
        print("\nThis script must run on real NVIDIA hardware. A CPU-only run "
              "proves nothing\nabout the CUDA path and is deliberately treated "
              "as a failure rather than a skip.")
        return 1

    device_name = torch.cuda.get_device_name(0)
    environment = {
        "gpu": device_name,
        "gpu_count": torch.cuda.device_count(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model": args.model,
    }
    print(f"\nGPU          : {device_name} (count {torch.cuda.device_count()})")
    print(f"torch        : {torch.__version__}  (CUDA {torch.version.cuda})")
    print(f"transformers : {transformers.__version__}")
    print(f"model        : {args.model}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    check("model loads on CPU before the attack moves it",
          next(model.parameters()).device.type == "cpu")

    # Catch any tensor that reaches generate() on the wrong device. This is the
    # exact failure mode the device fix addresses, and it is invisible to an
    # assertion that only inspects the returned logs.
    seen_devices: set[str] = set()
    original_generate = model.generate

    def spy(*a, **k):
        for key in ("input_ids", "inputs_embeds", "attention_mask"):
            if key in k and torch.is_tensor(k[key]):
                seen_devices.add(k[key].device.type)
        return original_generate(*a, **k)

    model.generate = spy

    config = {
        "objective": "eos_suppression",
        "strategy": "gradient",
        "max_iterations": 10,
        "perturbation_budget": 3,
        "top_k": 20,
        "max_new_tokens": 128,
        "protected_prefix_tokens": args.protected_prefix,
        "objective_horizon": 8,
        "seed": 0,
        "device": "cuda",
    }

    print("running the public API on cuda ...\n")
    try:
        attack = Attacker(model, tokenizer)
        adv_x, logs = attack.run(args.text, config)
    except Exception as exc:  # noqa: BLE001 - the point is to report, then fail
        print(f"\nFAILED during Attacker.run: {type(exc).__name__}: {exc}")
        return 1
    finally:
        model.generate = original_generate

    cost, perturbation = logs["cost"], logs["perturbation"]
    counters, censored = logs["attack_cost"], logs["censored"]

    check("model resides on cuda after run()",
          next(model.parameters()).device.type == "cuda")
    check("every tensor reaching generate() was on cuda",
          seen_devices == {"cuda"}, f"observed devices: {sorted(seen_devices)}")
    check("benign cost measured", cost["benign_output_tokens"] > 0,
          f"{cost['benign_output_tokens']} tokens")
    check("adversarial cost measured", cost["adversarial_output_tokens"] > 0,
          f"{cost['adversarial_output_tokens']} tokens")
    check("gradients were computed on device", counters["gradient_evaluations"] > 0,
          f"{counters['gradient_evaluations']} backward passes")
    check("candidate scoring ran", counters["objective_evaluations"] > 0,
          f"{counters['objective_evaluations']} objective evaluations")
    check("forward accounting is internally consistent",
          counters["total_model_forwards"] == (
              counters["search_model_forwards"]
              + counters["measurement_model_forwards"]
              + counters["diagnostic_model_forwards"]),
          f"{counters['total_model_forwards']} total")
    check("inputs_embeds path matches input_ids path on device",
          abs(logs["diagnostics"]["embedding_equivalence_max_logit_deviation"]) < 1e-3,
          f"max logit deviation "
          f"{logs['diagnostics']['embedding_equivalence_max_logit_deviation']:.3g}")
    check("returned text re-tokenises to the optimised ids",
          perturbation["round_trip_exact"] is True)
    check("returned text really does round-trip (checked independently)",
          tokenizer(adv_x)["input_ids"] == perturbation["adversarial_token_ids"])
    check("perturbation budget respected",
          perturbation["hamming_distance"]
          <= perturbation["positions_touched"]
          <= perturbation["budget"],
          f"hamming {perturbation['hamming_distance']}, touched "
          f"{perturbation['positions_touched']}, budget {perturbation['budget']}")
    check("censoring interpretation is well formed",
          censored["interpretation"] in
          {"point_estimate", "lower_bound", "upper_bound", "uninformative"},
          censored["interpretation"])
    check("logs are JSON-serialisable", _serialisable(logs))

    print(f"\nbenign      : {cost['benign_output_tokens']} tokens")
    print(f"adversarial : {cost['adversarial_output_tokens']} tokens")
    print(f"ratio       : {cost['output_token_ratio']:.2f}x  "
          f"({censored['interpretation']})")
    print(f"adv_x       : {adv_x!r}")
    print(f"\ncompute: {counters['objective_evaluations']} objective evaluations, "
          f"{counters['gradient_evaluations']} backward, "
          f"{counters['search_model_forwards']} search forwards, "
          f"{counters['total_model_forwards']} total forwards")

    failed = [name for name, ok, _ in CHECKS if not ok]
    payload = {
        "environment": environment,
        "config": config,
        "input": args.text,
        "adv_x": adv_x,
        "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in CHECKS],
        "all_passed": not failed,
        "cost": cost,
        "censored": censored,
        "perturbation": {k: v for k, v in perturbation.items() if k != "note"},
        "attack_cost": {k: v for k, v in counters.items() if k != "note"},
    }
    try:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nevidence written to {args.out}")
    except OSError as exc:
        print(f"\ncould not write {args.out}: {exc}")

    print("\n" + "=" * 72)
    if failed:
        print(f"RESULT: FAILED ({len(failed)} of {len(CHECKS)} checks)")
        for name in failed:
            print(f"  - {name}")
        return 1
    print(f"RESULT: PASSED ({len(CHECKS)}/{len(CHECKS)} checks) on {device_name}")
    print("=" * 72)
    return 0


def _serialisable(obj) -> bool:
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    sys.exit(main())
