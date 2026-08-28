# Results

Every number on this page was produced by a script in `examples/` on the machine
described below, and the raw logs are committed in `results/`. Nothing here is
carried over from a previous version of the code or from an external source.

**Environments.** Development and all CPU results: torch 2.13.0+cpu,
transformers 5.16.1, Python 3.12.10, Windows 11, device `cpu`. GPU verification:
torch 2.11.0+cu128 / CUDA 12.8 on an NVIDIA GeForce RTX 5060 Laptop GPU (§10).
Installation is separately verified on Linux by CI (§7).

**Start here.** The two results that matter most are the *locked held-out
evaluation* (§9) and the negative gradient-vs-random finding it was designed to
test (§2). Everything else supports one of those two.

---

## 1. Headline result — t5-small

```bash
python examples/quickstart.py
```

| | |
|---|---|
| model / tokenizer | `t5-small` / `T5Tokenizer` |
| input | `translate English to German: The house is wonderful.` |
| adversarial input | `translate English to German: The house is Madagascar.` |
| **benign output** | **6 tokens** (stopped on EOS) |
| **adversarial output** | **9 tokens** (stopped on EOS) |
| **ratio** | **1.50× generated tokens** |
| perturbation | 1 token differs; 1 position touched; budget 3 |
| round-trip exact | `true` |
| censoring | `point_estimate` — neither run hit the ceiling |
| config | `objective=eos_suppression, strategy=gradient, max_iterations=10, perturbation_budget=3, top_k=20, max_new_tokens=128, protected_prefix_tokens=7, objective_horizon=8, seed=0, device=auto` |

Because neither generation reached `max_new_tokens`, this 1.50× is an actual
measurement rather than a censored lower bound. Raw log:
`results/quickstart_t5-small.json`.

**Does the token count translate into wall-clock time?** On one GPU for this
one pair, yes: median latency 43.98 ms → 62.74 ms, a **1.43× latency ratio**
against the 1.50× token ratio, with non-overlapping interquartile ranges. See
[§11](#11-does-the-token-count-proxy-translate-into-latency). Token count remains
the primary metric because it is deterministic; latency is secondary.

**Determinism.** Re-running `examples/quickstart.py` rewrites only the wall-clock
fields (`wall_time_s`, `wall_time_ratio`, per-iteration `elapsed_s`). Every
deterministic field — token ids, objective values at each iteration, output
lengths, Hamming distance, forward counts — reproduces bit-identically, on both
CPU and GPU builds and across two torch versions. Timing is kept in the artifact
because it is real evidence, and is labelled noisy rather than removed to make
the diff clean.

**What this single number is and is not.** It is one input on one small model. It
shows the mechanism works end to end: a one-token edit, inside the stated budget,
that survives decode/re-tokenisation exactly, increased greedy generation length
by 50%. It is not evidence that the *gradient* is what produced the gain — see
the next section, where the control beats it.

### Attack compute for that run

Counted with a forward pre-hook on the model, not estimated. Verified against an
independent count obtained by patching `Model.forward`: both report **554**.

| quantity | value |
|---|---|
| objective evaluations | 64 |
| gradient (backward) passes | 3 |
| model forwards — search | 519 |
| model forwards — cost measurement | 33 |
| model forwards — embedding-equivalence check | 2 |
| **model forwards — total** | **554** |

One objective evaluation costs **8.1 model forwards** on average, because
evaluating the objective runs `generate()` for up to `objective_horizon` decoding
steps plus one teacher-forced forward. Reporting "64 forward passes" — as an
earlier version of this package did — understated the attack's real compute by
roughly 8×.

The honest economic summary of this run: **519 search forwards to induce 3 extra
decoding steps.** As an attack this is not economical at this scale; whether it
becomes so on larger models with longer outputs is not tested here.

---

## 2. Gradient vs. random control — the negative result

```bash
python examples/ablation.py
```

7 inputs, `max_iterations=6`, `perturbation_budget=3`, `objective_horizon=8`,
`max_new_tokens=128`, `seed=0`, protected prefix 7. Benign outputs
`[6, 8, 14, 12, 12, 14, 12]`, mean 11.14 tokens; **0/7 at the ceiling**, so every
comparison below is uncensored.

Both strategies receive identical inputs, iterations, perturbation budget, seed,
candidate space (same positions, same legal token ids, never the token already in
place) and the same number of exact objective evaluations per iteration. The only
difference is whether the gradient chose the shortlist.

Mean extra generated tokens vs. the benign input:

| `top_k` | gradient | random | advantage |
|---:|---:|---:|---:|
| 5 | +3.29 | **+5.00** | **−1.71** |
| 20 | +2.29 | **+6.14** | **−3.86** |
| 100 | +4.00 | **+8.57** | **−4.57** |

**The gradient loses at every budget tested, and the gap widens as `top_k`
grows.** It also costs more: at `top_k=100` the gradient strategy spent 20,579
search forwards against random's 17,221, because each iteration adds a backward
pass and one extra objective evaluation.

This refutes the most natural defence of the method. The expected shape of a
useful first-order ranking is that it wins when few candidates can be afforded
and its advantage shrinks as random gets more draws. The opposite happens here.

### Diagnosis

Two separate measurements, both reproducible, locate the problem precisely.

**The objective is a reasonable proxy for cost.** On t5-small, Spearman ρ between
`eos_suppression` and generated token count is **−0.284** computed within input
and averaged (objectives are minimised, so negative is good). The candidates the
objective ranks best generate **+1.83 tokens** more than their input's mean.

**The first-order HotFlip ranking is weak.** `examples/hotflip_diagnostic.py`
measures how well the estimate predicts the exact objective change it stands in
for — 210 admissible substitutions across the same 7 inputs, seed 0, legal
candidates only:

| statistic | value | reference |
|---|---:|---|
| Spearman ρ(estimate, exact delta), within input | **+0.246** | +1 perfect, 0 none |
| Spearman ρ, pooled across inputs | +0.091 | |
| sign agreement | 104/210 = **49.5%** | 50% = coin flip |
| candidates the estimate says will improve | 117 | |
| …of those, actually improve | **45 (38.5%)** | |
| top-10 by estimate, mean true rank | **89.4**/210 | chance 104.5, perfect 0 |

The direction is correct — ρ is positive, confirming the derivation and the sign
convention — but the magnitude is small. Sign agreement is at chance, and fewer
than two in five candidates the estimate nominates as improvements actually are.
The shortlist is better than random (mean true rank 89.4 against a chance value
of 104.5), just not by enough to overcome random search's advantage of sampling
the whole vocabulary.

The script asserts that its own top-k selection is identical to
`Attacker._gradient_candidates` on every input, so it provably measures the
ranking the attack actually uses. Raw log:
`results/hotflip_diagnostic_t5_small.json`.

**Scope.** This result is for `t5-small`, this input set, this budget range, and
this objective. It is not a claim about HotFlip in general, about larger models,
about other objectives, or about other perturbation types.

**These are development numbers.** All seven inputs were visible throughout
development. §9 repeats the comparison once, on a benchmark frozen and hashed
beforehand, and the gap shrinks substantially there — most of the −3.86 above is
noise. That is exactly why the held-out protocol exists.

Raw log: `results/ablation_t5_small.json`.

### Where exactly the gradient loses

```bash
python examples/search_diagnostic.py
```

The rank correlation says the ranking is weak; it does not say which stage
fails. This decomposes it against a reference set of exactly-scored random legal
substitutions.

| finding | measurement |
|---|---|
| the gradient's *best* proposal is near-optimal | percentile **0.0–1.3** of the reference distribution, on all four inputs |
| the round-trip filter does not penalise it | **0/20** proposals rejected, every input |
| position selection is mostly right | position error +0.00, +0.18, +2.15, +0.19 |
| **the shortlist is single-position** | **1–2 distinct positions out of 3–7; the top position holds 90–100 % of candidates** |
| the objective landscape is permissive | **29–85 %** of random legal substitutions improve the objective |
| objective improvement does translate to cost | of 37 objective-improving substitutions, **32 increased** generated tokens, **0 decreased**, mean **+3.05** |

So the gradient does not pick bad substitutions — it picks very good ones, all at
the same token position, because `topk` runs over the flattened
`(positions × vocab)` matrix and one position dominates the gradient magnitude.
Once that position is edited, the shortlist re-concentrates on it, nothing
improves, and the search halts. Budget utilisation shows the consequence: at
`top_k=100` the gradient leaves **7/7** runs under budget (mean Hamming 1.43)
while random leaves only 2/7 (mean 2.57).

That diagnosis produced a testable prediction, a registered variant, and — in §9
— its refutation. Raw log: `results/search_diagnostic_t5_small.json`.

---

## 3. Objective-horizon diagnostic

```bash
python examples/objective_diagnostic.py
```

`objective_horizon` is an **upper bound**, not a step count: the objective reads
stop-logits along the model's own greedy trajectory, which ends early when the
model emits EOS. Candidates could therefore be scored by means over different
numbers of terms. Three formulations were compared before changing anything —
270 admissible substitutions across 7 inputs, horizon 8, seed 0.

| objective | ρ within-input | ρ pooled | top-5 gain | ρ vs trajectory rows | rows scored |
|---|---:|---:|---:|---:|---|
| `mean` *(default)* | −0.284 | −0.194 | +1.83 | −0.424 | variable |
| `sum` | −0.295 | −0.213 | +1.83 | −0.461 | variable |
| `fixed` | −0.287 | −0.195 | +1.83 | n/a | constant |

Mean objective value by trajectory length:

| rows | n | `mean` | `sum` | `fixed` | output tokens |
|---:|---:|---:|---:|---:|---:|
| 6 | 15 | −8.74 | −52.42 | −7.94 | 6.0 |
| 7 | 6 | −9.62 | −67.34 | −9.29 | 7.0 |
| 8 | 249 | −12.28 | −98.25 | −12.28 | 12.9 |

**Conclusions.**

* The horizon usually does not bind: **249/270 (92%)** of candidates reached the
  full 8 rows, so most comparisons were already like-for-like.
* The three formulations are **indistinguishable as proxies** — ρ spans −0.284 to
  −0.295 and the top-5 selection gain is identical at +1.83 tokens.
* **`sum` was rejected** despite the nominally best ρ. From 6→8 rows, `mean`
  changes by a factor of 1.4 while `sum` changes by 1.87; the extra factor is
  mechanical, since adding more negative terms lowers a sum regardless of
  stop-reluctance. It would reward candidates for already being long, which is
  confounded with the outcome it is supposed to predict. A 0.011 difference in ρ
  does not justify adopting a biased estimator.
* **`mean` remains the default.** It scores the trajectory the model would really
  take, and a mean carries no mechanical length bias.
* **`eos_suppression_fixed_horizon` is available as an opt-in alternative.** It
  forces exactly `objective_horizon` steps via `min_new_tokens`, removing the
  comparability question by construction, at the cost of scoring a counterfactual
  path. It buys no measurable quality here.
* **Realised trajectory rows are logged per run** under
  `logs["diagnostics"]["objective_horizon"]`. The headline t5-small run reports
  `realised_rows 6–8, 29/64 evaluations at full horizon`.

Raw log: `results/objective_diagnostic_t5_small.json`.

---

## 4. Second seq2seq architecture — Marian

```bash
python examples/cross_architecture.py --only seq2seq
```

`Helsinki-NLP/opus-mt-en-de`, `MarianMTModel`, 74.4M parameters. Model and the
six inputs were fixed before any attack ran. Config: `strategy=gradient,
max_iterations=4, perturbation_budget=3, top_k=10, max_new_tokens=128,
objective_horizon=8, seed=0, protected_prefix_tokens=0`.

**Why this checkpoint.** Marian sets `scale_embedding=true` with `d_model=512`,
so its `inputs_embeds` path requires an embedding scale of `sqrt(512)`. Measured:
**`embed_scale = 22.627417`**, and the runtime equivalence check returns **max
logit deviation 0.0**. An implementation that fed raw embedding-table lookups
would silently optimise a different function than the model computes, on exactly
the architecture family NMTSloth targets. This run exercises that code path on a
real checkpoint rather than a synthetic config. Resolved EOS ids: `[0]`.

The complete experiment, all six runs:

| input | benign | adv | delta | hamming | touched | round-trip | censoring |
|---|---:|---:|---:|---:|---:|:--:|---|
| The house is wonderful. | 6 | 9 | **+3** | 1 | 1 | exact | point_estimate |
| She walked to the market this morning. | 10 | 10 | +0 | 0 | 0 | exact | point_estimate |
| The weather today is unusually cold. | 8 | 8 | +0 | 0 | 0 | exact | point_estimate |
| He forgot his keys on the kitchen table. | 11 | 11 | +0 | 0 | 0 | exact | point_estimate |
| They are building a new library downtown. | 10 | 10 | +0 | 0 | 0 | exact | point_estimate |
| Good morning, how are you today? | 11 | 11 | +0 | 1 | 1 | exact | point_estimate |

| summary | value |
|---|---|
| runs | 6 |
| improved | **1** |
| unchanged | **5** |
| worsened | 0 |
| runs where the search made **zero edits** (hamming 0) | **4** |
| mean delta | **+0.50 tokens** |
| median delta | +0.0 |
| benign at ceiling | 0/6 |
| uninformative censoring | 0/6 |
| round-trip exact | 6/6 |
| budget respected | 6/6 |

**Read this as a weak efficacy result, not a success.** One input of six
improved. In four of six runs the search found no improving substitution at all
within `top_k=10` and 4 iterations, so it committed nothing. The budget was not
raised afterwards to chase a better number; doing so after seeing the result
would be tuning.

What this run does establish firmly is **correct execution on a second
encoder-decoder family**: the embedding scale is applied and verified, every
measurement is uncensored, and the round-trip and budget invariants hold on all
six runs. Raw log: `results/cross_architecture.json`.

---

## 5. Causal language models

The causal adapter is implemented and exercised end to end on GPT-2. Efficacy on
a *deployed* instruction-tuned causal model is **not evaluated**, for a reason
recorded below. These three cases are different and must not be conflated.

### 5a. GPT-2 — executed end to end, result uninformative

```bash
python examples/quickstart.py --model gpt2 --text "The house is wonderful and" \
    --protected-prefix 0 --max-new-tokens 64
```

| | |
|---|---|
| benign output | 64 tokens (`max_tokens`) |
| adversarial output | 64 tokens (`max_tokens`) |
| observed ratio | 1.00× |
| censoring | **`uninformative`** |
| round-trip exact | `true` (2 tokens changed, budget 3) |

**This is not evidence that the attack works on causal models, and 1.00× is not
a result.** Base GPT-2 almost never emits `<|endoftext|>` from a short prompt, so
the benign input already runs to the decoding ceiling before the attack starts.
Both measurements are right-censored at 64, which means the observed ratio bounds
the true ratio in *neither* direction.

EOS-suppression is only informative against a model that stops on its own. The
NMTSloth threat model targets exactly such models, which is why `t5-small` and
Marian demonstrate the mechanism and `gpt2` cannot. The causal *code path* is
nonetheless exercised and correct here — adapter, alignment, round-trip and
budget invariants all hold. Raw log: `results/quickstart_gpt2.json`.

### 5b. SmolLM2-135M-Instruct — representation not exactly realizable, attack not run

```bash
python examples/cross_architecture.py --only causal
```

Selected on published properties and benign behaviour before any attack, as an
instruction-tuned decoder-only model that terminates naturally.

| property | measured |
|---|---|
| architecture | `LlamaForCausalLM`, 134.5M parameters |
| runtime `inputs_embeds` equivalence | **passes, max logit deviation 0.0** |
| resolved EOS ids | `[2]` |
| benign generation (chat template, greedy, cap 128) | 8, 99, 48, 89, 12, 38 tokens |
| benign terminated on EOS | **6/6** — no censoring |
| chat-templated input, exact text realization | **37 tokens → 32 after decode/re-tokenize** |
| interior special ids present | `[1, 2]` |
| **attack executed** | **no** |
| **efficacy measured** | **no** |

The model is compatible and its benign behaviour is ideal for this measurement —
all six prompts terminate well below the ceiling. What blocks the experiment is
the input *representation*: a chat template places `<|im_start|>` and
`<|im_end|>` inside the input, text realization decodes with
`skip_special_tokens=True`, and those tokens are dropped, so re-tokenising yields
a different 32-token sequence before any substitution is made.

The library rejects such inputs *as text* rather than measuring an adversarial
sequence different from the one it optimised. **This is not an attack failure;
efficacy was never measured.** Reporting it as a failed attack would be false.

**Token input removes the realisation barrier but not the experiment's problem.**
`Attacker.run` also accepts token ids, and with ids in and ids out no decode
happens, so the same sequence *can* be attacked directly (there is a test for
exactly this). That does **not** make this a usable evaluation of a deployed chat
model, for a reason unrelated to realisation: a chat template surrounds the user
message with scaffolding, and only `<|im_start|>`/`<|im_end|>` are special enough
to be excluded automatically. The `assistant` header tokens and newlines are
ordinary tokens and remain perturbable, while this toolbox can protect only a
*prefix*. Editing scaffolding is task damage, not an efficiency attack, so the
measurement would be contaminated. Arbitrary protected spans are the missing
piece, and they are not implemented.

### 5c. Qwen2.5-0.5B-Instruct — same boundary

```bash
python examples/cross_architecture.py --only causal \
    --causal-model Qwen/Qwen2.5-0.5B-Instruct
```

Declared fallback, characterised for compatibility only.

| property | measured |
|---|---|
| architecture | `Qwen2ForCausalLM`, 494.0M parameters |
| runtime `inputs_embeds` equivalence | **passes, max logit deviation 0.0** |
| resolved EOS ids | `[151645, 151643]` — a genuine multi-EOS model |
| benign terminated on EOS (chat template) | 5/6 (one hit the 128 ceiling) |
| chat-templated input, exact text realization | **36 tokens → 29** |
| **attack executed** | **no** |
| **efficacy measured** | **no** |

The fallback was not used to rescue a result, and no attack outcome was observed
for it. Raw log: `results/cross_architecture_qwen.json`.

### What this means

Supporting deployed instruction-tuned causal models would require **both**
preservation of interior special tokens during text realization **and** arbitrary
protected spans, so the template scaffolding cannot itself be adversarially
edited (only `<|im_start|>`/`<|im_end|>` are special; the surrounding
`assistant` header tokens are ordinary and would otherwise be perturbable).
Neither is implemented. Both are future work.

---

## 6. Invariants verified on real runs

```bash
python examples/invariant_sweep.py                # top_k=30
python examples/invariant_sweep.py --top-k 60 --out results/invariant_sweep_k60.json
```

`t5-small`, 24 runs each (2 strategies × 3 seeds × 4 inputs):

| | `top_k=30` | `top_k=60` |
|---|---:|---:|
| runs | 24 | 24 |
| **invariant violations** | **0** | **0** |
| candidates rejected by the round-trip filter | **22** | **67** |

The invariants checked on every run:

* returned `adv_x` re-tokenises to exactly the optimised ids — checked both from
  the logged flag and by independently re-tokenising the returned text
* `hamming_distance ≤ positions_touched ≤ perturbation_budget`
* `hamming_distance` equals the true count of differing token positions
* no committed token is a special id or an embedding row past `len(tokenizer)`
* no committed substitution is a no-op
* the protected prefix is byte-identical
* sequence length is unchanged

**The round-trip filter is not decorative.** It rejected 22 candidates at
`top_k=30` and 67 at `top_k=60` — substitutions that would have produced text
re-tokenising to different ids, and so an `adv_x` the perturbation budget did not
describe. For `t5-small` the excluded candidate set is **131 of 32,128** embedding
rows: 103 special ids plus 28 rows with no corresponding token.

Raw logs: `results/invariant_sweep.json`, `results/invariant_sweep_k60.json`.

### Embedding-path equivalence across families

`tests/test_adapters.py` asserts a maximum logit deviation of **0.0** between the
`inputs_embeds` and `input_ids` paths for T5, GPT-2, BART (both
`scale_embedding` settings), Marian (`scale_embedding=True`, factor applied by
the adapter), OPT, Llama and Qwen2. Those are locally-built configs — the check
establishes that the differentiable path is wired correctly for each
architecture *shape*, not that an attack was run on a trained checkpoint of it.
Real-checkpoint equivalence is separately verified for `t5-small`, `gpt2`,
Marian, SmolLM2 and Qwen2.5 (§4, §5, §8).

---

## 7. Installation, verified by remote CI

The documented editable installation could not be exercised locally: a Windows
Application Control policy on the development machine blocked Torch's DLLs from
loading inside any newly created virtualenv. That gap is closed on Linux by CI.

| | |
|---|---|
| workflow | `.github/workflows/ci.yml` |
| run | [33098762221](https://github.com/SandeepMadhavarapu/llm_efficiency_attack/actions/runs/33098762221) |
| runner | `ubuntu-latest` |
| Python | 3.12 |
| install command | `python -m pip install -e ".[dev,examples]"` — the exact README command, no CPU-wheel index or other shortcut |
| test suite | `python -m pytest -q` |
| packaging | `python -m build` (sdist + wheel) |
| result | **all 8 steps succeeded** |

This establishes that the documented install and the offline test suite work from
a fresh checkout on a clean Ubuntu runner with Python 3.12. It does **not**
establish that installation works on every platform or Python version, and it
does not exercise the CUDA path — GitHub-hosted runners have no GPU, so the CUDA
test skips there exactly as it does locally.

---

## 8. Validation status by architecture

Four distinct evidence levels, deliberately not merged.

| Model / family | Adapter + unit coverage | Runtime embedding equivalence | End-to-end attack executed | Efficacy demonstrated | Notes |
|---|:--:|:--:|:--:|:--:|---|
| **t5-small** | yes | yes (0.0) | **yes** | **yes** — 6→9, 1.50×, uncensored | headline; `results/quickstart_t5-small.json` |
| **Helsinki-NLP/opus-mt-en-de** (Marian) | yes | yes (0.0, scale 22.627) | **yes** | **weak** — 1/6 improved, mean +0.50, 4/6 zero-edit | second seq2seq family |
| **gpt2** | yes | yes (0.0) | **yes** | **no** — `uninformative`, both runs censored | benign already at ceiling |
| **SmolLM2-135M-Instruct** | — | yes (0.0) | **no** | **not measured** | chat template not exactly realizable, 37→32 |
| **Qwen2.5-0.5B-Instruct** | — | yes (0.0) | **no** | **not measured** | same boundary, 36→29 |
| BART (both `scale_embedding`) | yes | yes (0.0) | no | no | synthetic config only |
| OPT, Llama, Qwen2 (configs) | yes | yes (0.0) | no | no | synthetic config only |

"Runtime embedding equivalence" means the `inputs_embeds` path was verified to
reproduce the `input_ids` path on that model. It does **not** mean an attack was
run. "End-to-end attack executed" means `Attacker.run` completed and returned
logs.

---

## 9. Held-out evaluation — locked in advance, run once

```bash
python examples/heldout_evaluation.py
```

Every other number in this document was measured on inputs that were visible
during development. This section is the one confirmatory result.

### Protocol

| | |
|---|---|
| benchmark | `benchmarks/heldout_seq2seq_v1.json`, **n = 24** |
| SHA-256 | `9cc6170a3441fa67c2c8602213d66fb1e4fdccaf14efddcb8d835f5031fd390c` |
| frozen | before any algorithm change, and before any model had been run on it |
| freeze record | `benchmarks/HELDOUT_LOCK.txt` — hash, date, seed policy, single-use commitment |
| lock file | `benchmarks/STRATEGY_LOCK.txt`, SHA-256 `3ee3b7edddcfad356c23441c22cf479a4e8ab3425d40a5a0f1ba3e1a0ebc3fb1` |
| strategies | fixed in the lock file before any held-out result was seen |
| config | `top_k=20, max_iterations=6, perturbation_budget=3, objective_horizon=8, max_new_tokens=128, protected_prefix_tokens=7, seed=0` |
| runs | one, single seed, single candidate budget — no sweep, so no multiple-comparison slack |

`examples/heldout_evaluation.py` re-checks both hashes at run time and refuses to
run if either file has changed. Benign generation: mean 14.67 tokens, median
14.5, range 9–36, **0/24 at the ceiling** — so every comparison below is
uncensored and every censoring label is `point_estimate`.

### What this protocol does and does not establish

It is worth being exact, because "pre-registration" is a strong word.

*What is true.* The benchmark and the strategy specification were written and
SHA-256 hashed locally **before** the held-out attack was run, before
`gradient_stratified` existed, and before any model had been run on these 24
inputs. The lock file records the development numbers that motivated the variant
and commits in advance to evaluating all three strategies. The evaluation script
verifies both hashes and refuses to run against modified files, so the results
below provably correspond to *these* inputs and *this* configuration.

*What is not.* These files were not published in a commit that predates the
results, so **an external reviewer cannot use this repository's Git history to
independently prove the lock existed before the outcomes were observed.** The
ordering rests on the account given here, not on a timestamp anyone else can
check. Calling this "pre-registered" in the sense the term carries in clinical or
psychological research would overstate it; it is a documented internal protocol
with hash-verified artifacts. Publishing the lock in a pre-result commit is what
would close that gap, and was not done.

### Result

| strategy | mean Δ | 95% CI | median Δ | improved | unchanged | worsened | mean Hamming | full budget | search forwards |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|
| `gradient` | +2.42 | [+0.42, +5.25] | +1.0 | 14/24 | 7 | 3 | 1.12 | 1/24 | 12,587 |
| `gradient_stratified` | **+3.96** | [+2.08, +6.04] | **+4.0** | 17/24 | 5 | 2 | 2.62 | 19/24 | 15,065 |
| `random` | +2.88 | [+1.29, +4.75] | +1.5 | 17/24 | 4 | 3 | 2.08 | 12/24 | 12,363 |

Paired against the random control, per input:

| strategy | mean difference | 95% CI | median difference | wins | ties | losses |
|---|--:|---|--:|--:|--:|--:|
| `gradient` | −0.46 | [−2.12, +1.62] | −0.5 | 6 | 6 | 12 |
| `gradient_stratified` | +1.08 | **[−0.54, +2.83]** | **−0.5** | 10 | 2 | **12** |

Bootstrap: percentile method, 10,000 resamples, seed 0. All 24 runs round-trip
exact and respect the budget, for all three strategies.

### What this establishes, and what it does not

**The original negative result replicates, but smaller.** `gradient` still loses
to `random` (−0.46), confirming the development finding on data it never saw. The
development gap of −3.86 was mostly noise: with n=7 and no interval, it was
overstated. Reporting only the development number would have exaggerated the
method's failure just as surely as omitting it would have hidden it.

**`gradient_stratified` is not shown to beat random.** Its mean (+3.96) and
median (+4.0) are the highest of the three, and it is clearly better than the
original `gradient` variant. But the honest reading of the paired comparison is
negative on three counts: the CI on the difference **includes zero**, the
**median** paired difference is **−0.5**, and it **loses more inputs than it wins
(12 vs 10)**. Its higher mean is carried by a few large wins, not by consistent
improvement — and it spends **22% more search compute** to get there. On this
evidence the correct statement is "not distinguishable from random at n = 24",
not "an improvement".

**The mechanism hypothesis was confirmed; the causal theory was not.**
Stratification was introduced to fix a specific measured defect: the global
top-k shortlist draws 90–100% of its candidates from a *single* token position,
so after that position is edited nothing in the shortlist improves and the search
halts early. It did fix that — budget utilisation went from 1/24 runs at full
budget to 19/24, mean Hamming 1.12 → 2.62. But closing that gap did **not** make
the gradient beat random, which falsifies the theory that budget
under-utilisation was the *cause* of the gap. Something about *which*
substitutions the gradient selects, not *how many*, is where the remaining
difference lives. That is an open question, stated as one in
[docs/METHOD.md](docs/METHOD.md) §10.

Raw log: `results/heldout_t5_small.json` (all 24 inputs, per-strategy, unfiltered).

---

## 10. CUDA — verified on real hardware

```bash
python examples/cuda_smoke.py
python -m pytest -q -k cuda
```

Earlier versions of this document listed the GPU path as fixed but never
executed. It has now been executed.

| | |
|---|---|
| GPU | **NVIDIA GeForce RTX 5060 Laptop GPU** (8151 MiB, compute capability 12.0) |
| driver | 592.01 |
| torch | **2.11.0+cu128**, CUDA 12.8 |
| transformers | 5.16.1 |
| model | `t5-small` |

`examples/cuda_smoke.py`: **14/14 checks passed**, including model residency on
device, every tensor reaching `generate()` being on `cuda`, gradients computed on
device, forward-accounting consistency, `inputs_embeds` equivalence (deviation
0.0 on device), round-trip exactness verified independently, and budget
compliance. The two CUDA-marked tests in `tests/test_attacker.py` pass; the full
offline suite on that build is **91 passed, 5 skipped** (only the integration
tests skip).

The headline result reproduces **exactly** on GPU: 6 → 9 tokens, 1.50×, the same
`adv_x`, hamming 1, and the same 554 total model forwards as on CPU. That is
cross-device reproducibility, not merely "it ran".

Note the torch version differs from the CPU environment (2.11.0+cu128 vs
2.13.0+cpu). Both satisfy the declared `torch>=2.0,<3.0`, so this incidentally
exercises a second supported torch version.

Raw log: `results/cuda_smoke.json`.

---

## 11. Does the token-count proxy translate into latency?

```bash
python examples/latency_validation.py     # requires a CUDA device
```

Generated-token count is the primary metric because it is deterministic and
hardware-independent, and this document is careful to call it a *proxy* for
autoregressive decoding work rather than a latency measurement. That caution
leaves a fair question open: the threat model is about serving cost, and the
NMTSloth paper reports latency. If a 1.50× token increase does not move
wall-clock time, the proxy is not tracking the quantity anyone cares about.

**Protocol, fixed before any measurement.** No attack is re-run: the benign and
adversarial strings are read from the already-committed
`results/quickstart_t5-small.json`, so this cannot become a search for a pair
with a flattering latency ratio. Same model instance, same device, same greedy
config, `max_new_tokens=128`, batch size 1. **30 warm-up generations per
condition** (excluding CUDA context creation, kernel autotuning and allocator
growth), then **100 timed trials per condition**, predeclared and not adjusted
afterwards. Trials **alternate** benign/adversarial so thermal and clock drift
hit both conditions equally. `torch.cuda.synchronize()` immediately before and
after each timer, because CUDA launches are asynchronous and unsynchronised
timing measures launch overhead rather than execution. Median and IQR rather
than mean and standard deviation, since generation latency is right-skewed by
occasional scheduler interference.

| condition | generated tokens | median latency | IQR |
|---|---:|---:|---|
| benign | 6 | **43.98 ms** | [41.88, 48.47] ms |
| adversarial | 9 | **62.74 ms** | [60.35, 67.74] ms |

| quantity | value |
|---|---:|
| median latency ratio | **1.43×** |
| generated-token ratio | 1.50× |
| median gap | 18.76 ms |
| larger IQR | 7.39 ms |
| gap exceeds spread | **yes** |

**What was observed.** The reported interquartile ranges do not overlap:
benign [41.88, 48.47] ms and adversarial [60.35, 67.74] ms. The median gap is
18.76 ms. No significance test was performed and none is implied — this is a
description of the measured distributions, not an inference from them.

The measured 1.43× latency ratio is slightly below the 1.50× generated-token
ratio, which is consistent with fixed encoder, tokenisation and generation-setup
costs that are paid once regardless of output length. That is an interpretation
of this measurement, not a rule: nothing here establishes that latency must scale
in any particular relation to token count.

**Scope.** This result covers `t5-small`, the single frozen benign/adversarial
pair from `results/quickstart_t5-small.json`, this RTX 5060 Laptop GPU
environment, and the greedy generation configuration measured here — batch size
1, `max_new_tokens=128`. It shows the token-count metric is not inert on this
hardware for this pair. It does not establish a latency claim for other models,
inputs, batch sizes, hardware, or serving configurations, and **the primary
metric is unchanged**. Environment:
NVIDIA GeForce RTX 5060 Laptop GPU, torch 2.11.0+cu128, CUDA 12.8, transformers
5.16.1. All 200 raw per-trial timings are retained in
`results/latency_t5_small.json`.

---

## 12. Relationship to the NMTSloth reference implementation

Task 2 lists the NMTSloth paper and its reference codebase under **"Background
material (study before coding)"**, and asks the applicant to understand *"how the
original repo generates and evaluates adversarial inputs"*. It does not ask for a
reproduction. This section records what was read, what was attempted, and what
was not achieved.

**Status: source inspection plus an unsuccessful local execution attempt.** Not a
reproduction. Repository `https://github.com/SeekingDream/FSE22_NMTSloth` at
commit `8f72c15f91e2761e793b559d5428fab8fd15400f`. I inspected the
implementation and attempted to recreate its execution environment, but **did not
succeed in recreating that environment on the machines available to me**, so no
original run was executed here. The attempts, recorded so the reader can judge
them:

| attempt | result |
|---|---|
| Import under this project's stack (torch 2.13, transformers 5.16) | `ModuleNotFoundError: No module named 'transformers.generation_utils'` — the module was removed from `transformers` after the pinned era |
| Install the pinned stack (`torch==1.10.2`, `transformers==4.16.2`) | `ERROR: No matching distribution found for torch==1.10.2` — no wheel for Python 3.12, the only interpreter available here |
| `transformers==4.16.2` with a modern torch | `transformers` installed; the torch install aborted on a Windows long-path error in the scratch directory |

`generate_adv.py` additionally hardcodes `torch.device('cuda')` and iterates 500
sentences.

These are constraints of *this* environment — a Python 3.12-only machine, a
`transformers` API that moved after the pinned release, and a local Windows path
limit. **They say nothing about whether the reference implementation runs in the
environment it was written for, and nothing here should be read as a claim that
it does not.** No reference code was copied, nothing in this repository
reproduces the paper's numbers, and the paper's reported results are not restated
here as though they had been re-measured.

What the reading established, mapping concept to concept:

| NMTSloth reference | This toolbox |
|---|---|
| `leave_eos_target_loss`: BCE against zero on `P(EOS) + P(currently generated token)`, with the final step's term halved | `eos_suppression`: mean `log P(stop)`, EOS only, via `logsumexp` over all stop ids |
| Gradient taken w.r.t. the **embedding matrix** (`self.embedding.grad`, `vocab × dim`, accumulated over every position where a token occurs) | Gradient w.r.t. the **per-position input embeddings**, so `g_i` is specific to position `i` |
| `score = (E − E[t]) · g_t`, then `argsort` ascending | Same first-order form; `topk(−flat, k)` over the masked `(positions × vocab)` table |
| `select_best`: re-rank surviving candidates by **actual generated length** | Re-score candidates by the **exact objective**, commit only on strict improvement |
| `max_per = 3` | `perturbation_budget = 3` |
| Beam search (`num_beams` from the model config) | Greedy only, so the cost metric is deterministic |
| Character, word and structure perturbations, alongside several baseline attacks | Token substitution only |
| Targets include `T5-small` and `Helsinki-NLP/opus-mt-*` | The same two families are the ones evaluated here |

**One difference is worth naming as a hypothesis rather than a footnote.** Both
approaches use gradients to propose candidate mutations. They differ in how a
candidate is *chosen*: the reference's `select_best` generates translations for
the surviving candidates and selects by **actual generated sequence length**,
while this toolbox shortlists by first-order score and commits according to the
**configured surrogate objective**. The measured within-input correlation between
that surrogate and generated length is only about −0.28 (§3), so the two
selection rules are not equivalent.

> **Untested hypothesis.** The difference between selecting by actual generated
> length and selecting by the surrogate objective *may* contribute to the weak
> gradient-versus-random performance observed in §2 and §9.

No experiment here tested that. It is not a finding, it does not explain the
result, and it is emphatically not a claim that the reference's method would
outperform this one — nothing was run that could support any of those. It is
stated because it is the first experiment worth running next.

---

## Limitations

Classified. A limitation is only marked resolved where evidence actually resolves
it.

### Resolved by evidence

* **Fresh editable installation.** Verified by remote Linux CI — see §7. Scope:
  Ubuntu + Python 3.12 only.
* **`inputs_embeds` semantics on a scaling architecture.** Verified on a real
  Marian checkpoint at 0.0 deviation with `embed_scale = 22.627417`, not only on
  a synthetic config — see §4.
* **Single-model evidence.** A second encoder-decoder family now executes end to
  end with all invariants intact — see §4. The efficacy there is weak, and is
  reported as such.
* **No held-out evaluation.** Closed by §9: a 24-input benchmark, hashed and
  frozen before any algorithm change, evaluated once with bootstrap intervals.
* **CUDA never executed.** Closed by §10: 14/14 checks and both CUDA tests pass
  on an RTX 5060, and the headline reproduces exactly on device.

### True scope boundaries

* **Interior special tokens are outside the exact-text-realization scope of the
  *text* interface.** Text realization decodes with `skip_special_tokens=True`,
  so such tokens are lost and the returned text cannot re-tokenise to the
  optimised ids; the library refuses rather than measuring a different sequence.
  Token-id input has no such step and accepts these sequences.
* **Deployed instruction-tuned causal models still cannot be evaluated
  meaningfully**, and token input does not change that. Chat scaffolding contains
  ordinary, non-special tokens that stay perturbable, and only a prefix can be
  protected, so any such measurement would be contaminated by task damage.
  Arbitrary protected spans are the missing feature — see §5. Causal models
  themselves *are* supported; GPT-2 runs end to end.
* **The perturbation budget bounds token-level edit count and nothing else.** No
  semantic-similarity, fluency, or human-perceptibility constraint is enforced or
  measured. The headline example changes `wonderful` to `Madagascar`: one token,
  inside budget, and a complete change of meaning.
* **Base causal LMs cannot demonstrate this attack.** They already run to
  `max_new_tokens`, so both measurements are censored and the result is
  `uninformative`, not 1.00× — see §5a.
* **Single-device models only.** `device_map="auto"`, Accelerate sharding,
  offloading and quantised models are refused with a clear error.
* **Generated-token count is a proxy for autoregressive decoding work**, not a
  measurement of latency, energy, or total compute. Per-step cost grows with the
  KV cache and varies with hardware and batching. §11 measures the proxy against
  wall-clock time for one pair on one GPU (1.43× latency against a 1.50× token
  ratio, non-overlapping IQRs); that is a single measurement scoped to that
  model, pair, hardware and generation config, not a general latency claim.
* **The reference implementation was inspected; its execution environment was
  not successfully recreated here.** The pinned 2021-era stack does not install
  on the only Python interpreter available on this machine, so no original run
  was executed and nothing in this repository reproduces the paper's numbers.
  This is a limitation of the environments available to me, not a statement about
  the reference repository — see §12.
* **Greedy decoding only**, so that the reproducibility requirement can hold.
* **Token-level substitution only.** The original paper also explores
  character-level and structural perturbations, not implemented here.
* **Finite evidence.** Two models with end-to-end efficacy evidence. The held-out
  set is n=24 on one model and one seed; Marian is n=6. Enough for intervals,
  not enough to separate strategies that differ by ~1 token.
* **No strategy is shown to beat the random control.** On held-out data the
  gradient variants are not distinguishable from random at n=24 (§9). Whether a
  first-order proposal can beat random sampling on this objective is open.
* **`objective_horizon` is an upper bound**, so the objective can average over
  fewer terms for some candidates. Measured to be immaterial on `t5-small` (92%
  of candidates reach the full horizon); the realised count is logged.
* **`x` must be a `str`.** Token-id input is not supported.

### Unresolved engineering gaps

* **CI does not exercise the GPU path.** GitHub-hosted runners have no GPU, so
  the two CUDA tests skip there. They have been run on real hardware (§10), but
  that verification is manual and is not re-checked on every push.
* **Single-GPU only.** §10 verifies one device. Multi-GPU, sharded and offloaded
  models remain out of scope and are refused with a clear error.
