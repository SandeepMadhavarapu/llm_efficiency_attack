# llm_efficiency_attack

A white-box efficiency-attack toolbox for Hugging Face sequence models. It
searches for a small, budgeted edit to an input that makes the model generate a
longer output, inflating decoding cost.

Generalises the NMTSloth attack ([Chen et al., FSE'22](https://dl.acm.org/doi/10.1145/3540250.3549102))
from neural machine translation into a toolbox that *runs* on both seq2seq and
causal language models. Increased cost is demonstrated on two encoder-decoder
models; on causal models the attack executes but efficacy is not demonstrated —
see [What has actually been run](#what-has-actually-been-run).

**Measured on `t5-small`: 6 → 9 generated tokens (1.50×) from a one-token edit**,
uncensored, reproducing identically on CPU and GPU. Evaluated once more on a
**hash-locked 24-input held-out benchmark**, where no strategy — including the
gradient — is distinguishable from a random control. Both the positive and the
negative result are in [RESULTS.md](RESULTS.md); the derivation, with each
equation tied to code and tests, is in [docs/METHOD.md](docs/METHOD.md).

---

## Install

```bash
git clone https://github.com/SandeepMadhavarapu/llm_efficiency_attack.git
cd llm_efficiency_attack
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev,examples]"
```

## Usage

```python
from llm_efficiency_attack import Attacker

attack = Attacker(model)
adv_x, logs = attack.run(x, config)
```

`Attacker(model)` loads the tokenizer from the model's own `_name_or_path`.
Passing it explicitly, `Attacker(model, tokenizer)`, avoids the lookup.

`x` is either **text** (`str`) or **token ids** (`list[int]`, `tuple[int]`, or a
1-D / `(1, n)` integer tensor). `adv_x` comes back in the same representation, so
it can be fed to the model exactly the way `x` was. One example at a time —
batched input is refused with a clear error.

Run the demo and the experiments:

```bash
python examples/quickstart.py                    # t5-small, the snippet above
python examples/quickstart.py --model gpt2       # causal LM (see the caveat below)
python examples/ablation.py                      # gradient vs random, top_k sweep
python examples/objective_diagnostic.py          # objective formulation comparison
python examples/hotflip_diagnostic.py            # quality of the first-order ranking
python examples/cross_architecture.py            # Marian + instruction-tuned causal
python examples/invariant_sweep.py               # safety invariants across 24 real runs
python examples/search_diagnostic.py             # where gradient search loses to random
python examples/heldout_evaluation.py            # one-shot, hash-verified held-out run
```

On an NVIDIA GPU:

```bash
python examples/cuda_smoke.py                    # full public API on device, 14 checks
pytest -k cuda
```

Real-checkpoint tests are marked `integration` and skipped by default (they
download `t5-small` and `gpt2`). They pin the headline result and check
decode/re-tokenise exactness against the real SentencePiece and byte-level BPE
vocabularies:

```bash
pytest --run-integration
```

Run the tests (offline, no downloads):

```bash
pytest
```

**Dependencies.** The library needs only `torch` and `transformers`.
`sentencepiece` sits in the `examples` extra for older `transformers` releases in
the supported range. Running `examples/cross_architecture.py` against Marian
prints a `Recommended: pip install sacremoses` notice from `transformers`; it is
a recommendation, not a requirement, and the example runs correctly without it.

---

## How the attack works

For a fixed model and prompt setting, **generated-token count is a reproducible
proxy for autoregressive decoding work**: each generated token costs one decoder
forward pass, and decoding stops when the model emits an end-of-sequence token.
So the attack objective is: **make the model unwilling to stop.**

It is a proxy, not a measurement of latency or total compute — per-step cost
grows with the KV cache and varies with hardware and batching. Token count is the
primary metric because it is deterministic and hardware-independent; wall-clock
time is reported alongside it but is noisy enough at these output lengths to be
uninformative.

The obstacle is that the input is a sequence of discrete token ids while gradients
live in continuous embedding space, so the input cannot simply be stepped. The
loop is HotFlip:

1. Embed the current input and take `∂objective/∂embedding` at every position.
2. First-order-estimate the objective change for substituting any position with
   any vocabulary token: `(e_v − e_i) · ∇_i`. This is a single matmul against the
   embedding table, so the whole vocabulary is scored at once.
3. Take the top-`k` candidates by that estimate and re-score them **exactly** with
   real objective evaluations. The linear approximation is used only to shortlist
   candidates; every shortlisted substitution is re-scored exactly because the
   approximation is weak on the tested model. Nothing is committed on the
   estimate's word alone.
4. Commit the single best substitution. Repeat until the perturbation budget is
   spent or no candidate improves the objective.

On `t5-small` the first-order ranking turns out to be a weak signal (rank
correlation +0.246 with the exact objective change; sign agreement at chance),
and gradient-guided search does **not** beat a random control given the same
budget. That is measured, not assumed — on development data in
[RESULTS.md §2](RESULTS.md), and again on a frozen held-out benchmark in
[RESULTS.md §9](RESULTS.md), where the gap is smaller than development suggested
but still not in the gradient's favour.

Why a weak ranking is still a *sound* mechanism: the estimate only shortlists.
Every shortlisted candidate is re-scored with the exact objective, and a
substitution is committed only on strict improvement, so the search is valid
greedy coordinate descent on the exact objective for *any* proposal
distribution. Soundness and effectiveness are separate claims; see
[docs/METHOD.md §5](docs/METHOD.md).

## What the attack optimises is what you get back

The attack works on token ids; you receive text. Those can come apart, because
`encode(decode(ids))` is not an identity for arbitrary id sequences: BPE and
SentencePiece merges are context-dependent, special tokens are stripped on
decode, and embedding tables are usually padded past the end of the vocabulary
(`t5-small` has 32128 rows for 32100 tokens). A substitution can re-segment its
neighbours, changing positions the attack never touched — which would break the
budget in the text you actually feed the model.

Two mechanisms prevent that:

* special ids and out-of-vocabulary embedding rows are excluded from the
  candidate set before scoring (131 of 32128 rows for `t5-small`);
* any candidate whose committed sequence would not survive decode → re-encode is
  rejected during the search.

`logs["perturbation"]["round_trip_exact"]` records the outcome, and `run()`
raises rather than returning text the budget does not apply to.

### The boundary this creates

The same guarantee excludes a class of inputs. Text realization decodes with
`skip_special_tokens=True`, so an input carrying special tokens *inside* it
cannot be represented exactly: the tokens are dropped and re-tokenising yields a
different sequence. Modern chat templates do exactly this — measured,
`SmolLM2-135M-Instruct`'s templated prompt goes from **37 tokens to 32**, and
`Qwen2.5-0.5B-Instruct`'s from **36 to 29**, before any substitution is made.

Such inputs are rejected *as text* before the attack runs, rather than silently
measuring a different token sequence than the one optimised. Passing the same
sequence as **token ids** skips realisation entirely and is accepted. So:

* **causal models are supported** — the adapter is exercised end to end on GPT-2;
* **deployed instruction-tuned causal models are still not meaningfully
  evaluable**, and token input does not fix that: chat scaffolding contains
  ordinary tokens that remain perturbable and only a prefix can be protected, so
  the measurement would be contaminated by task damage. Efficacy there is
  **unevaluated**, not failed.

Arbitrary protected spans are the missing feature. That is future work.

## Where model-specific logic lives

All of it is in `adapters.py`. `ModelAdapter.for_model` is the only place in the
package that branches on architecture; `measure_cost` asks an adapter rather than
branching itself. Four things genuinely differ:

| | seq2seq (T5, BART, Marian) | causal (GPT-2, Llama, Qwen) |
|---|---|---|
| forward pass | needs `decoder_input_ids` | does not |
| stop-decision logits | teacher-forced along the model's own greedy decoder trajectory | teacher-forced along the model's own greedy continuation, one row per generated token |
| counting generated tokens | `len − 1` (drop seeded decoder start token) | `len − input_len` (drop the prompt) |
| input-embedding scale | `encoder.embed_scale` if present | decoder stack's `embed_scale` if present |

Supporting a new architecture means adding a `ModelAdapter` subclass.

### What has actually been run

Four evidence levels, kept separate on purpose. Full detail in
[RESULTS.md §8](RESULTS.md).

| Model | Embedding equivalence | End-to-end attack | Efficacy |
|---|:--:|:--:|:--:|
| `t5-small` | yes | yes | **yes** — 6→9, 1.50×, uncensored |
| `Helsinki-NLP/opus-mt-en-de` | yes (scale 22.627) | yes | weak — 1/6 improved, mean +0.50 |
| `gpt2` | yes | yes | no — `uninformative`, both runs censored |
| `SmolLM2-135M-Instruct` | yes | **no** | not measured — see the boundary note above |
| `Qwen2.5-0.5B-Instruct` | yes | **no** | not measured — see the boundary note above |

"Embedding equivalence" means the differentiable `inputs_embeds` path was
verified to reproduce the model's `input_ids` path. It does not mean an attack
was run.

### Why the embedding scale matters

The attack needs a differentiable input, so it feeds `inputs_embeds` instead of
`input_ids`. Those two paths are not always equivalent: Marian, M2M100 and BART
with `scale_embedding=True` compute `embed_tokens(ids) * embed_scale` on the
`input_ids` path and skip the multiplication when the caller supplies
`inputs_embeds`. For `Helsinki-NLP/opus-mt-*` that factor is `sqrt(512) ≈ 22.6`,
so an attack passing raw table lookups would optimise a function the model does
not compute — silently, looking like a weak attack rather than a bug.

The adapter applies the architecture's own factor, and **every `run()` verifies
it** by comparing one `input_ids` forward against one `inputs_embeds` forward. A
mismatch raises `EmbeddingSemanticsError` instead of returning a result. Verified
at zero deviation on T5, BART (both settings), Marian, GPT-2, OPT, Llama and
Qwen2; an unrecognised architecture is refused loudly rather than mis-optimised.

---

## Config reference

Every field is optional; defaults are shown. The config is a plain dict and must
round-trip through JSON.

| field | type | default | meaning |
|---|---|---|---|
| `objective` | str | `"eos_suppression"` | Which objective to minimise. See below. |
| `strategy` | str | `"gradient"` | `"gradient"` (global top-k HotFlip shortlist), `"gradient_stratified"` (same scores spread across token positions), or `"random"` (control). See [RESULTS.md §9](RESULTS.md) for the measured differences. |
| `max_iterations` | int | `10` | Optimisation steps. At most one substitution is committed per step. |
| `perturbation_budget` | int | `3` | Max number of **distinct token positions** the search may write to. Must not exceed `max_iterations`. |
| `top_k` | int | `20` | How many first-order-ranked candidates get an exact objective evaluation each step. The main runtime knob; each candidate costs several model forwards. |
| `max_new_tokens` | int | `128` | Generation ceiling for the cost metric, and the point at which measurements become censored. |
| `protected_prefix_tokens` | int | `0` | Leading tokens the attack may not touch. Set this to cover instruction prefixes. |
| `objective_horizon` | int | `8` | **Upper bound** on stop-decision positions the objective reads. The realised count can be lower; it is logged. |
| `seed` | int | `0` | Seeds Python, torch and CUDA RNGs. |
| `device` | str | `"auto"` | `"auto"`, `"cpu"`, `"cuda"`, … Validated at config time. Single device only. |
| `verbose` | bool | `false` | Per-iteration progress logging. |

Unknown fields are **rejected, not ignored** — a silently dropped `topk` would
look like a weak attack rather than a typo.

**There is no `step_size`.** The task's config table lists one, but this attack is
a discrete search over token substitutions, not continuous gradient descent:
there is no continuous variable to step. The gradient only *ranks* candidates,
and `top_k` — how many ranked candidates get an exact evaluation — is the
analogous knob. Adding a `step_size` that nothing reads would be worse than not
having one.

### Why `protected_prefix_tokens` exists

T5's input is `"translate English to German: The house is wonderful."` If the
attack is free to rewrite `"translate English to German:"`, it will, because
destroying the instruction is the easiest way to confuse the model into rambling.
That measures *task damage*, not efficiency degradation. Protecting the prefix
keeps the experiment measuring what it claims to.

### Objectives

| name | what it minimises |
|---|---|
| `eos_suppression` | mean log P(stop) across the model's own greedy trajectory |
| `eos_suppression_fixed_horizon` | the same, over exactly `objective_horizon` forced steps |
| `eos_suppression_worst_step` | log P(stop) at the single most likely stopping step |

Register your own with `@register("name")` in `objectives.py`; the optimisation
loop needs no changes. An objective that requires a fixed-length trajectory
declares `@register(..., force_full_horizon=True)`.

**Sign convention:** every objective returns a value that is *lower* when the
model is *more* reluctant to stop. The loop always minimises. Getting this
backwards silently builds an attack that makes outputs shorter.

The three formulations were compared before choosing a default; they are
statistically indistinguishable as proxies for generated length, and a summed
variant was rejected for a mechanical length bias. See
[RESULTS.md §3](RESULTS.md).

---

## Reading the logs

`logs` is a JSON-serialisable dict. Three parts matter more than the headline
ratio.

### `censored` — read this before quoting a number

Both cost measurements are capped at `max_new_tokens`, which makes them
right-censored. `logs["censored"]["interpretation"]` says what the observed ratio
establishes:

| state | interpretation | what it means |
|---|---|---|
| neither hit the ceiling | `point_estimate` | the ratio is an exact measurement |
| adversarial only | `lower_bound` | true ratio is at least the observed one |
| **both** | **`uninformative`** | the ratio bounds the true ratio in **neither** direction |
| benign only | `upper_bound` | the attack made generation *shorter* |

The `uninformative` case is not a technicality: it is what every base causal LM
produces, because such models rarely emit EOS from a short prompt and sit at the
ceiling before the attack starts.

### `attack_cost` — instrumented, not estimated

```json
"attack_cost": {
  "objective_evaluations": 64,
  "gradient_evaluations": 3,
  "search_model_forwards": 519,
  "measurement_model_forwards": 33,
  "diagnostic_model_forwards": 2,
  "total_model_forwards": 554
}
```

Model forwards are counted with a forward pre-hook, so they include every
decoding step inside the `generate()` calls each objective evaluation performs.
That is why `search_model_forwards` is about 8× `objective_evaluations`. Counting
one objective call as one "forward pass" — which an earlier version of this
package did — understates real attack compute by roughly that factor.

### `perturbation`

`positions_touched` counts positions the search wrote to; `hamming_distance`
counts positions whose final token actually differs. They are not the same
number: a position written twice counts once as touched, and a position restored
to its original token still counts as touched. The budget bounds
`positions_touched`.

### The `random` control

`strategy: "random"` runs the identical loop over the identical candidate space —
same positions, same legal token ids, never the token already in place — with the
same number of exact evaluations. The only difference is whether the gradient
chose the shortlist. `examples/ablation.py` sweeps `top_k` and reports both.

---

## Limitations

Stated plainly, because a toolbox that hides its failure modes is harder to trust
than one that names them.

- **The gradient does not beat the random control on `t5-small`** at any budget
  tested, and the gap widens with `top_k`. The objective is a reasonable proxy for
  cost; the first-order ranking is the weak link. See [RESULTS.md §2](RESULTS.md).
- **The perturbation budget bounds token-level edit count and nothing else.** No
  semantic-similarity, fluency, or human-perceptibility constraint is enforced or
  measured. The headline example changes `wonderful` to `Madagascar`: one token,
  inside budget, and a complete change of meaning. Do not read the budget as
  imperceptibility.
- **Base causal LMs cannot demonstrate this attack.** They already run to
  `max_new_tokens`, so both measurements are censored and the result is
  `uninformative`, not 1.00×. See [RESULTS.md §5a](RESULTS.md).
- **Chat-templated inputs are outside the exact-realization scope.** Interior
  special tokens do not survive text realization, so those inputs are rejected
  rather than mismeasured. Causal efficacy on deployed instruction-tuned models
  is therefore unevaluated. See [RESULTS.md §5](RESULTS.md).
- **Efficacy evidence is two models.** `t5-small` (1.50× on the headline input;
  n=24 held-out) and Marian (1 of 6 inputs improved, mean +0.50 tokens, 4 of 6
  runs made no edit at all). The Marian result is weak and is reported in full
  rather than headlined.
- **No strategy beats the random control.** On the held-out benchmark the
  gradient variants are not distinguishable from random at n=24: the paired
  confidence interval includes zero and the median paired difference is negative.
  Whether a first-order proposal can beat random sampling on this objective is
  an open question, not a solved one.
- **Single-device models only.** `device_map="auto"`, Accelerate sharding,
  offloading and quantised models are refused with a clear error.
- **GPU verified once, manually.** `examples/cuda_smoke.py` passes 14/14 checks
  and both CUDA tests pass on an RTX 5060 (torch 2.11.0+cu128), reproducing the
  headline exactly on device. CI has no GPU, so that path is not re-checked on
  every push, and only a single device has been tested.
- **Generated-token count is a proxy for inference cost, not latency.** Per-step
  cost grows with the KV cache and varies with batching and hardware. Wall-clock
  time is reported but is noisy enough at these output lengths to be
  uninformative.
- **`objective_horizon` is an upper bound**, so the objective can average over
  fewer terms for some candidates. Measured to be immaterial on `t5-small`
  (92% of candidates reach the full horizon); the realised count is logged.
- **Token-level substitution only.** The original paper also explores
  character-level and structural perturbations, not implemented here.
- **Greedy decoding only**, so that the reproducibility requirement can hold.
- **Untrained models are not a demo.** The fast test suite uses randomly
  initialised fixtures and therefore asserts *mechanics* — budget, round-trip,
  monotone objective, reproducibility, compute accounting — not attack strength.
  Effectiveness is measured on a trained model in `examples/` and recorded in
  `results/`.

---

## Layout

```
src/llm_efficiency_attack/
├── __init__.py     public API: Attacker
├── config.py       schema, defaults, validation          (requirement 3)
├── adapters.py     the only model-specific code          (requirement 1)
├── objectives.py   swappable attack objectives           (requirement 2)
├── metrics.py      cost measurement + censoring          (requirement 4)
└── attacker.py     the HotFlip optimisation loop
tests/              117 tests. 110 pass offline (2 CUDA + 5 integration skip);
                    115 with --run-integration; 117 with both, on a CUDA build
examples/
├── quickstart.py            the task snippet on a real model
├── ablation.py              gradient vs random, top_k sweep
├── objective_diagnostic.py  objective formulation comparison
├── hotflip_diagnostic.py    first-order ranking quality
├── cross_architecture.py    Marian + instruction-tuned causal models
├── invariant_sweep.py       safety invariants across many real runs
├── search_diagnostic.py     where gradient search loses to random
├── heldout_evaluation.py    one-shot evaluation on the frozen benchmark
└── cuda_smoke.py            full public API on real CUDA hardware
benchmarks/         frozen held-out inputs + SHA-256 freeze and strategy locks
results/            committed logs backing every number in RESULTS.md
docs/METHOD.md      derivation, with each equation tied to code and tests
.github/workflows/  CI: clean install + test suite on Ubuntu / Python 3.12
RESULTS.md          measured results, including negative ones
```

Installation is verified on a clean Ubuntu runner with Python 3.12 by CI, using
the exact command above. That covers that platform and Python version, not all
of them.

Every quantitative claim in this README and in `RESULTS.md` is reproduced by a
committed script writing to a committed file in `results/`.
