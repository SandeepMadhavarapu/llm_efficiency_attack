"""Cross-architecture validation on models chosen before any attack was run.

Why this exists
---------------
Quantitative evidence in RESULTS.md was concentrated on `t5-small`, plus a GPT-2
run that is uninformative because base GPT-2 does not terminate from a short
prompt and so sits at the decoding ceiling before the attack starts. Two
questions were open, and this script addresses both on preselected models:

1. Does the toolbox execute correctly and preserve its invariants on a second
   encoder-decoder architecture?
2. Can efficacy be measured on a decoder-only model that terminates naturally?

Question 2 is answered "not under the current public interface", for a reason
this script records rather than works around. See "Exact text realization" below.

Model selection, fixed before efficacy was measured
---------------------------------------------------
Both models were chosen on published properties and on *benign* behaviour only.
No attack was run before the selection was fixed, and neither model was replaced
for producing an unfavourable attack result.

`Helsinki-NLP/opus-mt-en-de` (seq2seq). Public, small (~74M), a genuinely
different encoder-decoder family from T5, and a real translation model, so greedy
decoding terminates naturally. It is also the most informative possible second
architecture for this codebase: Marian sets `scale_embedding=true` with
`d_model=512`, so its `inputs_embeds` path needs the `sqrt(512) ~= 22.6`
embedding scale that this toolbox applies and verifies. An implementation that
skipped that scale would silently optimise a different function here, so this
exercises that code path on a real checkpoint rather than a synthetic config.
Declared fallback: `facebook/bart-base`.

`HuggingFaceTB/SmolLM2-135M-Instruct` (causal). Public, decoder-only, 135M so it
runs on CPU, `LlamaForCausalLM` with no remote code, and instruction-tuned so it
terminates on EOS. Declared fallback: `Qwen/Qwen2.5-0.5B-Instruct`.

Exact text realization
----------------------
The attack optimises token ids but returns text, and it guarantees that the
returned text re-tokenises to exactly the ids that were optimised. Chat templates
place special tokens *inside* the input (`<|im_start|>`, `<|im_end|>`). Text
realization decodes with `skip_special_tokens=True`, which removes them, so
re-tokenising yields a shorter, different sequence -- for SmolLM2, 37 tokens
become 32 -- before any substitution is made.

The library rejects such inputs rather than measuring an adversarial sequence
different from the one it optimised. This script records that rejection as a
result: the model's representation was not realizable, so **attack efficacy was
not measured**. That is not an attack failure and must not be reported as one.
Supporting these inputs would need both preservation of interior special tokens
and arbitrary protected spans, so that the template scaffolding cannot itself be
edited. Both are future work; neither is implemented.

Usage::

    python examples/cross_architecture.py
    python examples/cross_architecture.py --only seq2seq
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from llm_efficiency_attack import Attacker, measure_cost
from llm_efficiency_attack.adapters import ModelAdapter

# Predeclared. Ordinary short instructions, written before any attack was run and
# not filtered by outcome.
CAUSAL_PROMPTS = [
    "What is the capital of France?",
    "Name three primary colors.",
    "How many days are in a leap year?",
    "Write one sentence about the ocean.",
    "What is 12 plus 30?",
    "Give a short definition of gravity.",
]

# Predeclared. Ordinary English sentences for an en->de translation model.
SEQ2SEQ_PROMPTS = [
    "The house is wonderful.",
    "She walked to the market this morning.",
    "The weather today is unusually cold.",
    "He forgot his keys on the kitchen table.",
    "They are building a new library downtown.",
    "Good morning, how are you today?",
]

# Reduced from the t5-small sweep purely for CPU tractability on larger models,
# and fixed before any result was seen. Not tuned.
CONFIG = {
    "objective": "eos_suppression",
    "strategy": "gradient",
    "max_iterations": 4,
    "perturbation_budget": 3,
    "max_new_tokens": 128,
    "objective_horizon": 8,
    "top_k": 10,
    "seed": 0,
    "verbose": False,
}


def load(name: str, causal: bool):
    tokenizer = AutoTokenizer.from_pretrained(name)
    cls = AutoModelForCausalLM if causal else AutoModelForSeq2SeqLM
    model = cls.from_pretrained(name).eval()
    if causal and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def realization(tokenizer, text: str) -> dict:
    """Can this text be represented exactly by the public returned-text interface?

    The attack decodes optimised ids with `skip_special_tokens=True`. If the input
    carries interior special tokens they are dropped, so re-tokenising the decoded
    text yields a different sequence and the round-trip invariant cannot hold.
    Checking here lets the script report that as a property of the input rather
    than discovering it as an exception mid-run.
    """
    ids = tokenizer(text)["input_ids"]
    realised = tokenizer(
        tokenizer.decode(ids, skip_special_tokens=True)
    )["input_ids"]
    return {
        "exact": ids == realised,
        "input_tokens": len(ids),
        "realised_tokens": len(realised),
        "interior_special_ids": sorted(
            {i for i in ids[:-1] if i in (tokenizer.all_special_ids or [])}
        ),
    }


def build_inputs(tokenizer, prompts, causal):
    """Return (prompt, text, protected_prefix_tokens, suffix_len) per prompt."""
    out = []
    for prompt in prompts:
        if causal and tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prefix, suffix = text.split(prompt)
            protected = len(tokenizer(prefix)["input_ids"])
            suffix_len = len(tokenizer(suffix)["input_ids"])
        else:
            text, protected, suffix_len = prompt, 0, 0
        out.append((prompt, text, protected, suffix_len))
    return out


def describe(label, name, model, tokenizer, adapter, strategy, probe_text):
    probe = tokenizer(probe_text, return_tensors="pt")["input_ids"]
    try:
        deviation = adapter.check_embedding_equivalence(probe, torch.ones_like(probe))
        equivalence = {"passed": True, "max_logit_deviation": deviation}
    except Exception as exc:  # noqa: BLE001
        equivalence = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

    print(f"\n{'=' * 78}\n{label}: {name}\n{'=' * 78}")
    print(f"  architecture      : {model.config.architectures}")
    print(f"  parameters        : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"  embedding scale   : {adapter.embedding_scale():.6f}")
    print(f"  inputs_embeds ==  : {equivalence}")
    print(f"  resolved EOS ids  : {adapter.eos_token_ids()}")
    print(f"  strategy          : {strategy}")
    return equivalence


def run_model(label, name, causal, prompts, strategy):
    model, tokenizer = load(name, causal)
    adapter = ModelAdapter.for_model(model, tokenizer)
    equivalence = describe(
        label, name, model, tokenizer, adapter, strategy, prompts[0]
    )

    inputs = build_inputs(tokenizer, prompts, causal)

    # Realization gate. If the model's normal input representation cannot be
    # reproduced exactly through decode/re-tokenize, the attack is not run: any
    # measurement would describe a different token sequence than the one
    # optimised. This is recorded as an outcome, not treated as a failure.
    checks = [realization(tokenizer, text) for _, text, _, _ in inputs]
    if not all(c["exact"] for c in checks):
        first = next(c for c in checks if not c["exact"])
        print("\n  EXACT TEXT REALIZATION: NOT AVAILABLE for this representation.")
        print(f"    input tokens {first['input_tokens']} -> "
              f"{first['realised_tokens']} after decode/re-tokenize")
        print(f"    interior special ids present: {first['interior_special_ids']}")
        print("    The attack was NOT run: efficacy is unmeasured, not failed.")
        print("    Benign generation is still characterised below.")

        benign = []
        for prompt, text, _, _ in inputs:
            r = measure_cost(model, tokenizer, text,
                             max_new_tokens=CONFIG["max_new_tokens"])
            benign.append({"prompt": prompt, **r})
            print(f"      {prompt[:44]:<46}{r['output_tokens']:>5} tok  "
                  f"{r['stopped_by']}")
        eos_n = sum(b["stopped_by"] == "eos" for b in benign)
        print(f"    benign terminated on EOS: {eos_n}/{len(benign)}")

        return {
            "model": name,
            "role": label,
            "causal": causal,
            "architectures": model.config.architectures,
            "parameters_millions": sum(p.numel() for p in model.parameters()) / 1e6,
            "embedding_scale": adapter.embedding_scale(),
            "embedding_equivalence": equivalence,
            "eos_token_ids": adapter.eos_token_ids(),
            "outcome": "representation_not_exactly_realizable",
            "attack_executed": False,
            "efficacy_measured": False,
            "realization": checks,
            "benign_only": benign,
            "benign_terminated_on_eos": eos_n,
        }

    rows = []
    header = (f"{'prompt':<40}{'benign':>7}{'adv':>6}{'delta':>7}{'ham':>5}"
              f"{'RT':>4}{'censoring':>16}")
    print("\n" + header)
    print("-" * len(header))

    for prompt, text, protected, suffix_len in inputs:
        config = dict(CONFIG, strategy=strategy, protected_prefix_tokens=protected)
        adv_x, logs = Attacker(model, tokenizer).run(text, config)

        p, c = logs["perturbation"], logs["cost"]
        n_tokens = len(p["original_token_ids"])
        suffix_start = n_tokens - suffix_len if suffix_len else n_tokens
        touched_suffix = [i for i in p["positions_changed"] if i >= suffix_start]

        rows.append({
            "prompt": prompt,
            "input_text": text,
            "adv_x": adv_x,
            "protected_prefix_tokens": protected,
            "template_suffix_tokens": suffix_len,
            "benign_output_tokens": c["benign_output_tokens"],
            "adversarial_output_tokens": c["adversarial_output_tokens"],
            "delta": c["adversarial_output_tokens"] - c["benign_output_tokens"],
            "ratio": c["output_token_ratio"],
            "benign_stopped_by": logs["benign"]["stopped_by"],
            "adversarial_stopped_by": logs["adversarial"]["stopped_by"],
            "censoring": logs["censored"]["interpretation"],
            "perturbation_budget": p["budget"],
            "hamming_distance": p["hamming_distance"],
            "positions_touched": p["positions_touched"],
            "positions_changed": p["positions_changed"],
            "round_trip_exact": p["round_trip_exact"],
            "template_suffix_touched": touched_suffix,
            "objective_evaluations": logs["attack_cost"]["objective_evaluations"],
            "search_model_forwards": logs["attack_cost"]["search_model_forwards"],
            "total_model_forwards": logs["attack_cost"]["total_model_forwards"],
        })
        r = rows[-1]
        print(f"{prompt[:39]:<40}{r['benign_output_tokens']:>7}"
              f"{r['adversarial_output_tokens']:>6}{r['delta']:>+7}"
              f"{r['hamming_distance']:>5}{str(r['round_trip_exact'])[0]:>4}"
              f"{r['censoring']:>16}")

    deltas = [r["delta"] for r in rows]
    summary = {
        "prompts": len(rows),
        "improved": sum(d > 0 for d in deltas),
        "unchanged": sum(d == 0 for d in deltas),
        "worsened": sum(d < 0 for d in deltas),
        "zero_edit_runs": sum(r["hamming_distance"] == 0 for r in rows),
        "mean_delta": statistics.mean(deltas),
        "median_delta": statistics.median(deltas),
        "uninformative_censoring": sum(r["censoring"] == "uninformative" for r in rows),
        "benign_at_ceiling": sum(r["benign_stopped_by"] == "max_tokens" for r in rows),
        "round_trip_exact_all": all(r["round_trip_exact"] for r in rows),
        "budget_respected_all": all(
            r["hamming_distance"] <= r["positions_touched"] <= r["perturbation_budget"]
            for r in rows
        ),
    }
    print(f"\n  improved {summary['improved']}/{summary['prompts']}, "
          f"unchanged {summary['unchanged']}, worsened {summary['worsened']}, "
          f"zero-edit {summary['zero_edit_runs']}")
    print(f"  mean delta {summary['mean_delta']:+.2f} tokens, "
          f"median {summary['median_delta']:+.1f}")
    print(f"  benign at ceiling {summary['benign_at_ceiling']}/{summary['prompts']}, "
          f"uninformative {summary['uninformative_censoring']}/{summary['prompts']}")
    print(f"  round-trip exact everywhere: {summary['round_trip_exact_all']}, "
          f"budget respected everywhere: {summary['budget_respected_all']}")

    return {
        "model": name,
        "role": label,
        "causal": causal,
        "architectures": model.config.architectures,
        "parameters_millions": sum(p.numel() for p in model.parameters()) / 1e6,
        "embedding_scale": adapter.embedding_scale(),
        "embedding_equivalence": equivalence,
        "eos_token_ids": adapter.eos_token_ids(),
        "outcome": "attack_executed",
        "attack_executed": True,
        "efficacy_measured": True,
        "realization": checks,
        "config": CONFIG,
        "strategy": strategy,
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["causal", "seq2seq"], default=None)
    parser.add_argument("--causal-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--seq2seq-model", default="Helsinki-NLP/opus-mt-en-de")
    parser.add_argument("--strategy", default="gradient", choices=["gradient", "random"])
    parser.add_argument("--out", default="results/cross_architecture.json")
    args = parser.parse_args()

    results = []
    if args.only in (None, "seq2seq"):
        results.append(run_model("second seq2seq architecture", args.seq2seq_model,
                                 False, SEQ2SEQ_PROMPTS, args.strategy))
    if args.only in (None, "causal"):
        results.append(run_model("causal instruction-tuned", args.causal_model,
                                 True, CAUSAL_PROMPTS, args.strategy))

    payload = {
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "config": CONFIG,
        "causal_prompts": CAUSAL_PROMPTS,
        "seq2seq_prompts": SEQ2SEQ_PROMPTS,
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
