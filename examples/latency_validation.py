"""Does the token-count proxy actually translate into wall-clock latency?

Why this exists
---------------
The primary cost metric in this toolbox is *generated-token count*, and RESULTS
is careful to call it a proxy for autoregressive decoding work rather than a
latency measurement: per-step cost grows with the KV cache and varies with
hardware and batching. That caution is correct, but it leaves a question open.
The threat model is about inflating serving cost, and the NMTSloth paper reports
latency. If a 1.50x increase in generated tokens does not move wall-clock time,
the proxy is not measuring the thing anyone cares about.

This measures that one transfer, on one GPU, for the one pair already committed
in `results/quickstart_t5-small.json`. It does not change the primary metric and
is reported as secondary evidence.

Protocol, fixed before any measurement was taken
------------------------------------------------
* **No attack is run.** The benign and adversarial strings are read from the
  committed quickstart artifact, so this cannot become a search for a pair with
  a better latency ratio.
* Same model instance, same device, same `model.eval()`, same greedy generation
  config, `max_new_tokens=128`.
* **30 warm-up generations per condition** before any timing, so CUDA context
  creation, kernel autotuning and allocator growth are excluded.
* **100 timed trials per condition**, predeclared. The count is not adjusted
  after seeing results.
* Trials **alternate** benign/adversarial so that any thermal or clock drift
  affects both conditions equally rather than whichever ran first.
* `torch.cuda.synchronize()` immediately before starting and immediately after
  stopping each timer -- CUDA kernel launches are asynchronous, so timing without
  synchronisation measures launch overhead rather than execution.
* `torch.inference_mode()` for the timed region.
* Reports **median and IQR**, not mean and standard deviation: generation latency
  is right-skewed by occasional scheduler interference, and the median is the
  robust summary.
* All raw per-trial timings are retained in the artifact.

Interpretation is honest in all directions. Latency rising, latency not rising,
and a difference buried in noise are all reportable outcomes; the script draws
its conclusion from an explicit comparison of the median gap against the spread.

Usage (requires a CUDA device)::

    python examples/latency_validation.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import statistics
import time

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

QUICKSTART = pathlib.Path("results/quickstart_t5-small.json")
WARMUP = 30
TRIALS = 100
MAX_NEW_TOKENS = 128


def quartiles(values):
    s = sorted(values)
    n = len(s)
    q1 = s[n // 4]
    q3 = s[(3 * n) // 4]
    return q1, statistics.median(s), q3


def generate_once(model, inputs):
    """One greedy generation, timed the way CUDA requires."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="t5-small")
    parser.add_argument("--out", default="results/latency_t5_small.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "This validation requires a CUDA device. A CPU run would measure a "
            "different machine's behaviour and is deliberately refused rather "
            "than reported as GPU evidence."
        )

    logs = json.loads(QUICKSTART.read_text(encoding="utf-8"))["gradient"]
    # The committed artifact stores the token ids; decode them with the same
    # tokenizer so the strings are exactly the pair that was already measured.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    benign_text = tokenizer.decode(
        logs["perturbation"]["original_token_ids"], skip_special_tokens=True
    )
    adv_text = tokenizer.decode(
        logs["perturbation"]["adversarial_token_ids"], skip_special_tokens=True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).cuda().eval()
    device = next(model.parameters()).device

    conditions = {
        "benign": tokenizer(benign_text, return_tensors="pt").to(device),
        "adversarial": tokenizer(adv_text, return_tensors="pt").to(device),
    }

    print("=" * 74)
    print("Latency validation (secondary evidence; token count remains primary)")
    print("=" * 74)
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"torch        : {torch.__version__}  (CUDA {torch.version.cuda})")
    print(f"transformers : {transformers.__version__}")
    print(f"benign       : {benign_text!r}")
    print(f"adversarial  : {adv_text!r}")
    print(f"protocol     : {WARMUP} warm-ups + {TRIALS} alternating timed trials "
          f"per condition\n")

    tokens = {}
    for name, inputs in conditions.items():
        for _ in range(WARMUP):
            _, out = generate_once(model, inputs)
        tokens[name] = int(out.shape[1] - 1)  # seq2seq: drop decoder start token

    timings = {name: [] for name in conditions}
    for _ in range(TRIALS):
        for name, inputs in conditions.items():   # alternating order
            elapsed, _ = generate_once(model, inputs)
            timings[name].append(elapsed)

    stats = {}
    for name in conditions:
        q1, med, q3 = quartiles(timings[name])
        stats[name] = {
            "generated_tokens": tokens[name],
            "median_s": med, "q1_s": q1, "q3_s": q3, "iqr_s": q3 - q1,
            "min_s": min(timings[name]), "max_s": max(timings[name]),
            "trials": len(timings[name]),
        }
        print(f"  {name:<12} {tokens[name]:>3} tokens | median {med*1000:7.2f} ms "
              f"| IQR [{q1*1000:.2f}, {q3*1000:.2f}] ms")

    b, a = stats["benign"], stats["adversarial"]
    gap = a["median_s"] - b["median_s"]
    ratio = a["median_s"] / b["median_s"]
    token_ratio = a["generated_tokens"] / b["generated_tokens"]
    spread = max(b["iqr_s"], a["iqr_s"])
    separated = gap > spread

    print(f"\n  median latency ratio : {ratio:.2f}x")
    print(f"  generated-token ratio: {token_ratio:.2f}x")
    print(f"  median gap           : {gap*1000:.2f} ms")
    print(f"  larger IQR           : {spread*1000:.2f} ms")
    print(f"  gap exceeds spread   : {separated}")

    if separated and ratio > 1.0:
        verdict = ("latency_increased",
                   "The median latency gap exceeds the larger of the two "
                   "interquartile ranges, so the token-count increase does "
                   "translate into wall-clock time on this device for this pair.")
    elif ratio > 1.0:
        verdict = ("latency_increase_within_noise",
                   "Median latency is higher but the gap does not exceed the "
                   "interquartile spread, so this run does not separate the two "
                   "conditions.")
    else:
        verdict = ("no_latency_increase",
                   "Median latency did not rise despite more generated tokens.")
    print(f"\n  VERDICT: {verdict[0]}\n  {verdict[1]}")
    print("\n  Scope: one GPU, one model, one input pair, batch size 1, greedy "
          "decoding.\n  Generated-token count remains the primary metric because "
          "it is deterministic\n  and hardware-independent; this is secondary "
          "evidence that the proxy transfers.")

    out = {
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "model": args.model, "device": str(device),
        },
        "protocol": {
            "source_pair": QUICKSTART.as_posix(),
            "attack_rerun": False,
            "warmup_per_condition": WARMUP,
            "timed_trials_per_condition": TRIALS,
            "trial_order": "alternating benign/adversarial",
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "synchronised": True,
            "summary_statistic": "median with IQR",
            "predeclared": True,
        },
        "inputs": {"benign": benign_text, "adversarial": adv_text},
        "results": stats,
        "median_latency_ratio": ratio,
        "generated_token_ratio": token_ratio,
        "median_gap_s": gap,
        "gap_exceeds_iqr": separated,
        "verdict": verdict[0],
        "verdict_note": verdict[1],
        "raw_timings_s": timings,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
