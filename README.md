# llm_efficiency_attack

A white-box efficiency-attack toolbox for Hugging Face sequence models. It finds
a near-imperceptible edit to an input that forces the model to run many more
decoding steps, inflating latency and serving cost.

Generalises the NMTSloth attack ([Chen et al., FSE'22](https://dl.acm.org/doi/10.1145/3540250.3549102))
from neural machine translation to arbitrary seq2seq and causal language models.

---

## Install

```bash
git clone <this-repo> && cd llm_efficiency_attack
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
Passing it explicitly, `Attacker(model, tokenizer)`, avoids the lookup and is
preferred.

Run the demo:

```bash
python examples/quickstart.py                # t5-small, seq2seq
python examples/quickstart.py --model gpt2   # causal LM, no library changes
```

Run the tests (offline, ~4 seconds):

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
   real forward passes. The linear approximation ranks well but its magnitudes are
   unreliable, so nothing is committed on its word alone.
4. Commit the single best substitution. Repeat until the perturbation budget is
   spent or no candidate improves the objective.

## Where model-specific logic lives

All of it is in `adapters.py`, and there is less of it than expected because
Hugging Face already provides uniform accessors: `get_input_embeddings()` returns
the embedding table for any architecture, and `config.is_encoder_decoder` reports
the family. Only three things genuinely differ:

| | seq2seq (T5, BART, Marian) | causal (GPT-2, Llama, Qwen) |
|---|---|---|
| forward pass | needs `decoder_input_ids` | does not |
| stop-decision logits | across teacher-forced decoder steps | at the end of the prompt |
| counting generated tokens | `len − 1` (drop seeded decoder start token) | `len − input_len` (drop the prompt) |

Nothing outside `adapters.py` branches on model type. **Supporting a new
architecture means adding a `ModelAdapter` subclass and changing nothing else.**

---

## Config reference

Every field is optional; defaults are shown. The config is a plain dict and must
round-trip through JSON.

| field | type | default | meaning |
|---|---|---|---|
| `objective` | str | `"eos_suppression"` | Which objective to minimise. See below. |
| `strategy` | str | `"gradient"` | `"gradient"` for the white-box attack; `"random"` for the control. |
| `max_iterations` | int | `10` | Optimisation steps. At most one substitution is committed per step. |
| `perturbation_budget` | int | `3` | Max number of **distinct token positions** that may differ between `x` and `adv_x`. This is the imperceptibility constraint. Must not exceed `max_iterations`. |
| `top_k` | int | `20` | How many first-order-ranked candidates get an exact forward pass each step. |
| `max_new_tokens` | int | `128` | Generation ceiling for the cost metric, and the point at which measurements become censored. |
| `protected_prefix_tokens` | int | `0` | Leading tokens the attack may not touch. Set this to cover instruction prefixes. |
| `objective_horizon` | int | `8` | How many stop-decision positions the objective looks at. |
| `seed` | int | `0` | Seeds Python, torch and CUDA RNGs. |
| `device` | str | `"auto"` | `"auto"`, `"cpu"`, `"cuda"`, … |
| `verbose` | bool | `false` | Per-iteration progress logging. |

Unknown fields are **rejected, not ignored** — a silently dropped `topk` would
look like a weak attack rather than a typo.

### Why `protected_prefix_tokens` exists

T5's input is `"translate English to German: The house is wonderful."` If the
attack is free to rewrite `"translate English to German:"`, it will, because
destroying the instruction is the easiest way to confuse the model into rambling.
But that measures *task damage*, not efficiency degradation, and the resulting
input is no longer imperceptibly different in the way that matters. Protecting the
prefix keeps the experiment measuring what it claims to.

### Objectives

| name | what it minimises |
|---|---|
| `eos_suppression` | mean log P(stop) across the horizon |
| `eos_suppression_worst_step` | log P(stop) at the single most likely stopping step |

Register your own with the `@register("name")` decorator in `objectives.py`; the
optimisation loop needs no changes. **Sign convention:** every objective returns a
value that is *lower* when the model is *more* reluctant to stop. The loop always
minimises. Getting this backwards silently builds an attack that makes outputs
shorter.

---

## Reading the logs

`logs` is a JSON-serialisable dict. Three parts are worth more than the headline
ratio.

### `censored` — read this before quoting a number

Once the attack works, every generation runs into `max_new_tokens`. Output length
pins to the ceiling and the metric **saturates**: it can no longer distinguish a
good attack from a great one. Those observations are *right-censored* — the true
cost is at least the cap and possibly far more.

```json
"censored": {
  "max_new_tokens": 128,
  "adversarial_hit_ceiling": true,
  "ratio_is_lower_bound": true
}
```

When `ratio_is_lower_bound` is `true`, **the reported ratio is a lower bound, not
a measurement.** Raise `max_new_tokens` to tighten it. This is also why
`measure_cost` checks the ceiling *before* checking EOS: a run that reaches the cap
and happens to end on a stop token is still ceiling-bound, and labelling it `"eos"`
would disguise a saturated measurement as a natural one.

### `attack_cost` — is the attack economical?

```json
"attack_cost": { "forward_passes": 61, "backward_passes": 3 }
```

The attack spends compute to make the victim spend more. Comparing the two says
whether it is worth running. Cost-metric measurements are instrumentation and are
excluded from these counts.

### The `random` control

Setting `strategy: "random"` runs the identical loop with the identical
perturbation budget and the identical number of exact evaluations, choosing
substitutions uniformly at random instead of by gradient. The only difference
between the two runs is whether the gradient was consulted.

If gradient-guided search does not clearly beat random search, the white-box
signal is not paying for itself — which is a result worth knowing rather than an
outcome to avoid measuring. `examples/quickstart.py` reports both side by side.

---

## Limitations

Stated plainly, because a toolbox that hides its failure modes is harder to trust
than one that names them.

- **The reported ratio is usually a lower bound.** See `censored` above. Any
  successful attack saturates the metric.
- **Wall-clock time is noisy.** It is reported because latency is the real-world
  quantity of interest, but token count is the primary metric: it is deterministic
  and hardware-independent, and wall time is neither.
- **The first-order ranking is only moderately reliable.** Measured on a toy
  model over 120 sampled substitutions, Spearman ρ between the HotFlip estimate
  and the exact objective change was **0.554**, and the top-10 by estimate had a
  mean exact-rank of 38/120 (chance would be 60). Better than random, but far from
  a perfect oracle — which is precisely why nothing is committed on the estimate
  alone and every shortlisted candidate is re-scored exactly. Raising `top_k`
  trades compute for a better chance of catching the true best substitution.
  `objectives.py` and the diagnostic in the audit notes make this measurable
  rather than assumed.
- **Decode/re-encode is not guaranteed to round-trip.** `adv_x` is returned as
  text; re-tokenising it can in principle yield a slightly different id sequence
  than the attack optimised. The exact ids are in
  `logs["perturbation"]["adversarial_token_ids"]` and should be preferred when
  exactness matters.
- **The perturbation budget is token-level only.** The original paper also
  explores character-level and structure-level perturbations, which are not
  implemented here.
- **Greedy decoding only.** The cost metric fixes `do_sample=False` so that the
  reproducibility requirement can hold. Attacks against sampled decoding would
  need a distributional cost metric rather than a point measurement.
- **Untrained models are not a good demo.** Randomly initialised models have no
  learned stopping behaviour, so both benign and adversarial runs hit the ceiling
  and the ratio is meaningless. The test suite therefore asserts *mechanics*
  (budget respected, objective monotonically improving, reproducibility) rather
  than attack strength; effectiveness is demonstrated in the quickstart on a real
  trained model.

---

## Layout

```
src/llm_efficiency_attack/
├── __init__.py     public API: Attacker
├── config.py       schema, defaults, validation          (requirement 3)
├── adapters.py     the only model-specific code          (requirement 1)
├── objectives.py   swappable attack objectives           (requirement 2)
├── metrics.py      cost measurement                      (requirement 4)
└── attacker.py     the HotFlip optimisation loop
tests/              35 tests, offline, ~4s
examples/quickstart.py
```

Each module owns exactly one requirement from the task specification, so every
requirement has one place it lives and one place it can break.
