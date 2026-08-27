# llm_efficiency_attack

A white-box efficiency-attack toolbox for Hugging Face sequence models. It
searches for a small, budgeted edit to an input that makes the model generate a
longer output, inflating decoding cost.

Generalises the NMTSloth attack ([Chen et al., FSE'22](https://dl.acm.org/doi/10.1145/3540250.3549102))
from neural machine translation to seq2seq and causal language models.

**Measured result on `t5-small`: 6 → 9 generated tokens (1.50×) from a one-token
edit.** Full numbers, including a negative result where the gradient loses to a
random control, are in [RESULTS.md](RESULTS.md).

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

`x` is a `str`. Token-id input is not supported.

Run the demo and the experiments:

```bash
python examples/quickstart.py                    # t5-small, the snippet above
python examples/quickstart.py --model gpt2       # causal LM (see the caveat below)
python examples/ablation.py                      # gradient vs random, top_k sweep
python examples/objective_diagnostic.py          # objective formulation comparison
python examples/hotflip_diagnostic.py            # quality of the first-order ranking
```

Run the tests (offline, no downloads):

```bash
pytest
```

---

## How the attack works

Inference cost is dominated by **how many decoding steps** the model runs, not by
how long the input is. Decoding stops when the model emits an end-of-sequence
token. So the attack objective is: **make the model unwilling to stop.**

The obstacle is that the input is a sequence of discrete token ids while gradients
live in continuous embedding space, so the input cannot simply be stepped. The
loop is HotFlip:

1. Embed the current input and take `∂objective/∂embedding` at every position.
2. First-order-estimate the objective change for substituting any position with
   any vocabulary token: `(e_v − e_i) · ∇_i`. This is a single matmul against the
   embedding table, so the whole vocabulary is scored at once.
3. Take the top-`k` candidates by that estimate and re-score them **exactly** with
   real objective evaluations. The linear approximation ranks well enough to be
   worth using but its magnitudes are unreliable, so nothing is committed on its
   word alone.
4. Commit the single best substitution. Repeat until the perturbation budget is
   spent or no candidate improves the objective.

On `t5-small` the first-order ranking turns out to be a weak signal (rank
correlation +0.246 with the exact objective change; sign agreement at chance),
and this gradient-guided search does **not** beat a random control given the same
budget. That is measured, not assumed — see [RESULTS.md §2](RESULTS.md).

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
| `strategy` | str | `"gradient"` | `"gradient"` for the white-box attack; `"random"` for the control. |
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
  `uninformative`, not 1.00×. See [RESULTS.md §4](RESULTS.md).
- **Single-device models only.** `device_map="auto"`, Accelerate sharding,
  offloading and quantised models are refused with a clear error.
- **No GPU testing.** Developed on a CPU-only machine. The device-placement
  invariant has a CPU test; the end-to-end CUDA test exists but is skipped there.
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
tests/              75 tests (74 pass, 1 CUDA test skipped without a GPU)
examples/
├── quickstart.py            the task snippet on a real model
├── ablation.py              gradient vs random, top_k sweep
├── objective_diagnostic.py  objective formulation comparison
└── hotflip_diagnostic.py    first-order ranking quality
results/            committed logs backing every number in RESULTS.md
RESULTS.md          measured results, including negative ones
```

Every quantitative claim in this README and in `RESULTS.md` is reproduced by a
committed script writing to a committed file in `results/`.
