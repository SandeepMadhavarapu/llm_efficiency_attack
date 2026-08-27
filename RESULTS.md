# Results

Every number on this page was produced by a script in `examples/` on the machine
described below, and the raw logs are committed in `results/`. Nothing here is
carried over from a previous version of the code or from an external source.

**Environment.** torch 2.13.0+cpu, transformers 5.16.1, Python 3.12.10, Windows
11, device `cpu`. No CUDA was available, so the GPU path is untested — see
[Limitations](#limitations).

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
decoding steps.** As an attack this is not economical at this scale; the
interesting question is whether it becomes so on larger models with longer
outputs, which is not tested here.

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

So: the target is right and the search is right; the *ranking heuristic* is what
fails to add value here. Random sampling over a 32k-token vocabulary, followed by
exact rescoring of `top_k` candidates, is already a strong baseline — and it gets
strictly better with more draws while the gradient's shortlist stays concentrated
in a region the linear approximation likes.

**This is why exact rescoring matters.** Neither strategy commits anything on the
estimate alone; every shortlisted candidate is re-scored with a real objective
evaluation. That is what keeps the gradient strategy from being much worse than
it is.

**Scope.** This result is for `t5-small`, this input set, this budget range, and
this objective. It is not a claim about HotFlip in general, about larger models,
about other objectives, or about other perturbation types. Testing whether the
gradient pays off on a model whose embedding geometry is better conditioned is
the obvious next experiment and has not been run.

Raw log: `results/ablation_t5_small.json`.

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
* **Realised trajectory rows are now logged per run** under
  `logs["diagnostics"]["objective_horizon"]`, so the caveat is visible in the
  data rather than only in prose. The headline t5-small run reports
  `realised_rows 6–8, 29/64 evaluations at full horizon`.

Raw log: `results/objective_diagnostic_t5_small.json`.

---

## 4. Causal models — a negative result

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

**This is not evidence that the attack works on causal models, and 1.00× is not a
result.** Base GPT-2 almost never emits `<|endoftext|>` from a short prompt, so
the benign input already runs to the decoding ceiling before the attack starts.
Both measurements are right-censored at 64, which means the observed ratio bounds
the true ratio in *neither* direction — the true benign cost could be far above
the true adversarial cost or far below it.

EOS-suppression is only informative against a model that stops on its own. The
NMTSloth threat model targets exactly such models (neural machine translation),
which is why `t5-small` demonstrates the effect and `gpt2` cannot.

The causal *code path* is exercised and correct — the adapter, alignment,
round-trip and budget invariants all hold, and the run reports
`round_trip_exact: true` with 2 tokens changed inside a budget of 3. What is
missing is a causal model that terminates naturally. Testing an
instruction-tuned causal model with a real EOS is the obvious next step and has
not been done.

Raw log: `results/quickstart_gpt2.json`.

---

## 5. Invariants verified on real runs

Across **24 real t5-small runs** (2 strategies × 3 seeds × 4 inputs), zero
violations of:

* returned `adv_x` re-tokenises to exactly the optimised ids (`round_trip_exact`)
* `hamming_distance ≤ positions_touched ≤ perturbation_budget`
* `hamming_distance` equals the true count of differing token positions
* no committed token is a special id or an embedding row past `len(tokenizer)`
* no committed substitution is a no-op
* the protected prefix is byte-identical
* sequence length is unchanged

The round-trip filter is not decorative: across 12 further runs at `top_k=60` it
rejected **2** candidates that would have produced text re-tokenising to
different ids. For `t5-small` the excluded set is 131 of 32,128 embedding rows —
103 special ids plus 28 rows with no corresponding token.

The embedding-equivalence check reported a maximum logit deviation of **0.0** on
T5, BART (both `scale_embedding` settings), Marian (`scale_embedding=True`,
factor applied by the adapter), GPT-2, OPT, Llama and Qwen2.

---

## Limitations

* **No GPU testing.** This machine is CPU-only. The device-placement fix is
  covered by a CPU test that pins the invariant and by a CUDA test that is
  skipped here; the end-to-end GPU path has not been executed.
* **Single device only.** `device_map="auto"`, Accelerate sharding, offloading
  and quantised models are refused with a clear error, not supported.
* **One small seq2seq model.** All quantitative results are `t5-small`. Nothing
  here establishes behaviour on larger models.
* **Token-level substitution only.** The original paper also explores
  character-level and structural perturbations, which are not implemented.
* **The budget bounds token edits, nothing else.** No semantic-similarity,
  fluency, or human-perceptibility constraint is enforced or measured. The
  headline example changes `wonderful` to `Madagascar` — one token inside budget,
  and a complete change of meaning.
* **Greedy decoding only.** The cost metric fixes `do_sample=False` so
  reproducibility holds. Attacks against sampled decoding would need a
  distributional cost metric.
* **Generated-token count is a proxy for cost, not latency.** Per-step cost grows
  with the KV cache and varies with batching and hardware. Wall-clock times are
  reported but are noisy enough at these output lengths to be uninformative.
