# Method

The derivation behind the implementation, written so each equation names the code
that implements it and the test that checks it. Where the mathematics and the
empirical result disagree — as they do for the quality of the first-order
ranking — both are stated.

---

## 1. Notation and threat model

| symbol | meaning |
|---|---|
| $T$ | tokenizer; $T(x)$ encodes text, $T^{-1}$ decodes |
| $\mathbf{z} = (z_1,\dots,z_n)$ | benign token ids, $\mathbf{z} = T(x)$ |
| $\mathbf{z}'$ | adversarial token ids, $\lvert\mathbf{z}'\rvert = n$ |
| $V$ | rows of the input embedding table |
| $\mathcal{L} \subseteq V$ | *legal* substitution ids (see §6) |
| $P \subseteq \{1..n\}$ | perturbable positions |
| $B$ | perturbation budget |
| $M$ | victim model, weights known |
| $E$ | set of end-of-sequence ids |
| $H$ | `objective_horizon` |
| $C$ | `max_new_tokens`, the decoding cap |

**Attacker knowledge.** White-box: weights, gradients, tokenizer, and the
decoding procedure. This is what makes $\nabla_e J$ available at all.

**Decoding.** Greedy, $\texttt{do\_sample=False}$, capped at $C$. Deterministic
given the input, which is what makes the cost metric reproducible
(`metrics.measure_cost`).

**Cost.** $\mathrm{cost}(\mathbf{z}) = $ number of tokens generated before EOS or
the cap. A proxy for autoregressive decoding work — each generated token is one
decoder forward — and *not* latency, energy, or total FLOPs. Per-step cost grows
with the KV cache and varies with hardware and batching.

**Feasible set.**

$$\mathcal{F}(\mathbf{z}) = \Big\{\mathbf{z}' : \big|\{i : z'_i \neq z_i\}\big| \le B,\ \ z'_i = z_i \ \forall i \notin P,\ \ z'_i \in \mathcal{L},\ \ T(T^{-1}(\mathbf{z}')) = \mathbf{z}' \Big\}$$

The last condition is the *exact-realization* constraint: the returned text must
re-tokenise to the ids that were optimised. Without it the budget bounds
something the caller never sees. Implemented in `Attacker._round_trips`, enforced
per-candidate in `_pick_best_candidate` and re-checked before returning.

**Goal.** $\max_{\mathbf{z}' \in \mathcal{F}} \mathrm{cost}(\mathbf{z}')$.
$\mathrm{cost}$ is a piecewise-constant integer function of discrete ids, so it is
optimised through the smooth surrogate below.

**Stated assumptions.** (i) EOS terminates decoding, so suppressing $P(\text{EOS})$
lengthens generation — *empirically supported*, within-input Spearman $-0.284$
(RESULTS §3), not proved. (ii) Single device. (iii) The `inputs_embeds` path
reproduces the `input_ids` path — *verified per run*, §7.

---

## 2. The surrogate objective

Let $s_t \in \mathbb{R}^{|V|}$ be the logits at decoding step $t$. The
probability that decoding stops at $t$ is the probability of emitting *any* stop
token:

$$P(\text{stop at } t) = \sum_{v \in E} p_t(v), \qquad p_t = \mathrm{softmax}(s_t)$$

Working in log space, and using $\log \sum_v \exp(\log p_t(v)) = \operatorname{logsumexp}_{v\in E} \log p_t(v)$:

$$\ell_t = \log P(\text{stop at } t) = \operatorname*{logsumexp}_{v \in E}\ \big[\log\mathrm{softmax}(s_t)\big]_v$$

$$\boxed{\ J(\mathbf{z}) = \frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} \ell_t\ }$$

minimised over $\mathbf{z}'$.

Three deliberate choices:

* **Log space, not probability.** A working attack drives $p_t(\text{EOS}) \to 0$,
  where probabilities underflow and their gradients vanish. Log-probabilities stay
  finite and informative in exactly that regime.
* **`logsumexp`, not a plain sum of log-probs.** $\log\sum_v p_v \ne \sum_v \log p_v$.
  Stopping is a *union* of events, so the probabilities must be summed before the
  log. Getting this wrong silently changes the objective for every multi-EOS model
  (Qwen2.5 declares `[151645, 151643]`).
* **Mean, not sum.** A sum of negative terms decreases simply by having more
  terms, so it would reward longer trajectories mechanically — confounded with the
  quantity it is meant to predict. Measured: from 6→8 trajectory rows the mean
  moves ×1.4 while the sum moves ×1.87 (RESULTS §3). The mean was kept.

| claim | code | test |
|---|---|---|
| objective definition | `objectives.eos_suppression` | `test_math.py::test_multi_eos_logsumexp_equals_log_of_summed_probability` |
| `logsumexp` $=\log\sum p$ | `objectives._eos_log_prob` | same test, checked against the naive computation at `atol=1e-12` |
| sum rejected for length bias | — | `examples/objective_diagnostic.py`, RESULTS §3 |

---

## 3. Which timesteps $\mathcal{T}$ are scored

$\mathcal{T}$ is the model's **own greedy trajectory**, taken under `no_grad` and
then teacher-forced in one differentiable pass. Two consequences.

**$H$ is an upper bound, not a step count.** The trajectory ends early if the
model emits EOS, so $|\mathcal{T}| \le H$ and can differ between candidates.
Measured: 249/270 candidates reach the full horizon on t5-small, so 92 % of
comparisons are like-for-like. `logs["diagnostics"]["objective_horizon"]` reports
the realised range per run. `eos_suppression_fixed_horizon` removes the variation
via `min_new_tokens` and was measured to change nothing ($\rho$ $-0.287$ vs
$-0.284$).

**Alignment differs by architecture** — the only genuinely model-specific part.

*Seq2seq.* `generate` returns $[\text{start}, y_1, \dots, y_L]$. Teacher-forcing
input is that sequence minus its last token, so row $j$ of the logits predicts
$y_{j+1}$: exactly the $L$ stop decisions along the real path.

*Causal.* `generate(inputs_embeds=...)` returns only the $L$ new tokens. The first
$L-1$ are appended to the prompt, giving length $n + L - 1$; row $i$ predicts token
$i+1$, so rows $n-1 \dots n+L-2$ are precisely the $L$ generated positions. Reading
a window at the *end of the prompt* instead — the obvious implementation — would
score $H-1$ positions that predict tokens already in the prompt, and only one that
predicts a generated token.

| claim | code | test |
|---|---|---|
| seq2seq shift | `Seq2SeqAdapter.stop_logits` | `test_adapters.py::test_stop_logits_row_count_is_bounded_by_the_horizon` |
| causal offset $n-1$ | `CausalAdapter.stop_logits` | `test_adapters.py::test_causal_stop_logits_cover_generated_tokens_only` |
| gradient survives the concatenation | same | same test, asserts `embeds.grad` is non-zero |
| $|\mathcal{T}| \le H$ logged | `_ComputeCounters.record_trajectory` | `test_attacker.py` horizon diagnostics |

---

## 4. HotFlip: the first-order proposal

Write $e_i$ for the embedding actually fed to the model at position $i$ (that is,
$e_i = \sigma\,W[z_i]$ with $\sigma$ the architecture's embedding scale, §7). A
first-order Taylor expansion of $J$ about the current embeddings, for the
substitution $z_i \to v$:

$$J(\dots e_v \dots) = J(\dots e_i \dots) + \big\langle \nabla_{e_i} J,\ e_v - e_i \big\rangle + O\!\left(\lVert e_v - e_i \rVert^2\right)$$

$$\boxed{\ \Delta J(i,v) \;\approx\; g_i^\top (e_v - e_i), \qquad g_i = \nabla_{e_i} J\ }$$

Because $J$ is **minimised**, the most promising substitutions are the **most
negative** $\Delta J$. The implementation computes the whole
$(n \times |V|)$ table with one matmul,

$$\text{scores} = \sigma\,(G W^\top) - \big[\langle g_i, \sigma W[z_i]\rangle\big]_i \mathbf{1}^\top$$

masks inadmissible entries to $+\infty$, and takes `topk(-flat, k)`.

$\sigma > 0$ is a positive constant, so it cannot change the *ranking* — it is
applied anyway so the computed quantity is the one the derivation describes.

| claim | code | test |
|---|---|---|
| $g$ is the true gradient of $J$ | `adapters.embed` → `loss.backward()` | `test_math.py::test_autograd_matches_central_finite_differences` (rel. err. $7.4\times10^{-5}$) |
| descent direction | as above | `test_math.py::test_gradient_sign_is_the_descent_direction` |
| vectorised score $=\langle g_i, e_v-e_i\rangle$ | `Attacker._gradient_candidates` | `test_math.py::test_hotflip_score_is_exactly_the_first_order_term` |
| $O(\lVert\cdot\rVert^2)$ error behaviour | — | `test_math.py::test_hotflip_estimate_predicts_small_exact_changes` |
| most-negative selected | `topk(-flat, k)` | `test_attacker.py::test_gradient_candidates_exclude_special_and_out_of_vocab_ids` |

**A numerical caveat, established rather than assumed.** `T5LayerNorm` casts to
float32 to compute its variance, so the forward pass carries a float32 noise
floor even in float64. The finite-difference check therefore bottoms out at
$h=10^{-2}$ with relative error $7.4\times10^{-5}$; the same procedure reaches
$10^{-9}$ on a closed-form control function, which locates the limitation in the
model rather than in the verification.

---

## 5. Soundness vs. effectiveness — why a weak ranking is still valid

This is the distinction that answers the submission's main criticism, and the two
halves are independent.

**Soundness.** The proposal only *shortlists*. Every shortlisted candidate is
re-scored with the exact objective, and a substitution is committed only when

$$J(\mathbf{z}^{(k+1)}) < J(\mathbf{z}^{(k)})$$

strictly. Therefore $J$ is monotonically non-increasing, and the search is a valid
greedy coordinate descent on the **exact** objective *for any proposal
distribution whatsoever* — including a bad one, including a uniform one. A weak
$\Delta J$ estimate can waste evaluations; it cannot make the search take a step
that increases $J$.

**Effectiveness.** How much the ranking *saves* is a separate, empirical question,
and here the answer is unflattering. Measured on t5-small over 210 admissible
substitutions: within-input Spearman $+0.246$; sign agreement 104/210 = 49.5 %,
i.e. chance; 45 of 117 predicted improvements are real. Top-10 by estimate have
mean true rank 89.4/210 against a chance value of 104.5 — better than random, but
not by enough. Consequently gradient-guided search loses to a random control at
every budget tested (RESULTS §2).

Both statements hold simultaneously: the mechanism is mathematically correct, and
its empirical value on this model is small. Conflating them would be the error.

**A tested consequence.** The global top-k shortlist draws 90-100 % of its
candidates from a *single* token position, because `topk` runs over the flattened
`(positions x vocab)` matrix and one position dominates the gradient magnitude.
Once that position is edited the shortlist re-concentrates on it, nothing
improves, and the search halts with most of the budget unspent. A registered
variant, `gradient_stratified`, distributes the same scores round-robin across
positions; it raised budget use from 1/24 to 19/24 runs at full budget on the
held-out set, exactly as predicted — and still did not beat random (paired CI
$[-0.54, +2.83]$, median difference $-0.5$, 10 wins against 12 losses). So budget
under-utilisation was a symptom, not the cause. RESULTS §9.

---

## 6. Perturbation constraints — what is actually guaranteed

Two distinct counters, deliberately not merged:

* $\texttt{positions\_touched} = |\{i : \text{the search wrote to } i\}|$, bounded by $B$;
* $\texttt{hamming\_distance} = |\{i : z'_i \neq z_i\}|$.

$$\texttt{hamming} \;\le\; \texttt{touched} \;\le\; B$$

The inequality can be strict: a position written twice counts once as touched, and
a position restored to its original token still consumes budget.

$\mathcal{L}$ excludes two classes, both because they break exact realization:
special ids (deleted by `skip_special_tokens=True`) and embedding rows past
$\lvert T \rvert$ (t5-small has 32128 rows for 32100 tokens; the surplus decodes to
the empty string). For t5-small $|V \setminus \mathcal{L}| = 131 = 103 + 28$.

**What this does not constrain.** Nothing semantic. No fluency, no similarity, no
human perceptibility. `wonderful` → `Madagascar` is one token inside budget and a
complete change of meaning.

**Where the constraint bites.** Inputs containing *interior* special tokens cannot
satisfy $T(T^{-1}(\mathbf{z}')) = \mathbf{z}'$ at all, so $\mathcal{F} = \emptyset$
and the attack refuses. Chat templates are exactly such inputs (SmolLM2: 37 → 32
tokens), which is why efficacy on deployed instruction-tuned causal models is
unevaluated.

| claim | code | test |
|---|---|---|
| hamming ≤ touched ≤ budget | `_build_logs` | `test_attacker.py::test_hamming_distance_is_reported_separately_from_positions_touched`; `examples/invariant_sweep.py` (0 violations / 48 runs) |
| exact realization | `_round_trips`, `_realise` | `test_integration.py::test_returned_text_round_trips_under_a_real_tokenizer` |
| $\mathcal{L}$ definition | `adapters.forbidden_token_ids` | `test_adapters.py::test_forbidden_ids_cover_specials_and_surplus_embedding_rows` |
| interior specials rejected | final guard in `_run_on_device` | `test_attacker.py::test_input_with_interior_special_tokens_is_rejected` |
| monotone $J$ | commit condition | `test_math.py::test_objective_is_monotone_non_increasing_by_construction` |

---

## 7. Embedding-path equivalence

The attack differentiates through `inputs_embeds`, so it requires

$$M(\texttt{inputs\_embeds} = \sigma W[\mathbf{z}]) \;=\; M(\texttt{input\_ids} = \mathbf{z})$$

This is **not** automatic. Marian, M2M100 and BART with `scale_embedding=true`
compute $\sigma W[\mathbf{z}]$ with $\sigma=\sqrt{d_{\text{model}}}$ on the
`input_ids` path and skip the multiplication when the caller supplies
`inputs_embeds`. For `Helsinki-NLP/opus-mt-en-de`, $\sigma = \sqrt{512} \approx
22.627$: an implementation feeding raw table lookups would optimise a different
function than the model computes, silently, on the very architecture family
NMTSloth targets.

`ModelAdapter.embed_values` applies $\sigma$; `check_embedding_equivalence`
*verifies* the identity on every `run()` (two forwards, under `no_grad`, in eval
mode) and raises `EmbeddingSemanticsError` rather than returning a result.
Measured deviation 0.0 on T5, GPT-2, BART (both settings), Marian, OPT, Llama,
Qwen2.

---

## 8. Censoring — what the observed ratio establishes

Both costs are capped at $C$, so both are **right-censored**. With true costs
$A, B$ and observed $a, b$:

| benign | adversarial | inference | label |
|---|---|---|---|
| $b < C$ | $a < C$ | $A/B = a/b$ exactly | `point_estimate` |
| $b < C$ | $a = C$ | $A \ge C$, $B = b$ ⟹ $A/B \ge a/b$ | `lower_bound` |
| $b = C$ | $a = C$ | $A \ge C$ **and** $B \ge C$ ⟹ **no bound in either direction** | `uninformative` |
| $b = C$ | $a < C$ | $A = a < C \le B$ ⟹ $A < B$, and $A/B \le a/b$ | `upper_bound` |

Row 3 is the one that matters in practice: it is what every base causal LM
produces, and calling it a lower bound — as an earlier version of this package did
— is simply false. Dividing two lower bounds yields no bound.

Implemented in `metrics.interpret_censoring`; one test per row in
`test_metrics.py`.

---

## 9. Compute accounting

Let $O$ be objective evaluations, $G$ backward passes. Each objective evaluation
runs `generate` for up to $H$ steps plus one teacher-forced forward, so the number
of real `Module.__call__` invocations is **not** $O$. Instrumented with a forward
pre-hook rather than derived from a formula:

$$F_{\text{total}} = F_{\text{search}} + F_{\text{measure}} + F_{\text{diag}}$$

On the headline run: $O=64$, $G=3$, $F_{\text{search}}=519$, $F_{\text{measure}}=33$,
$F_{\text{diag}}=2$, $F_{\text{total}}=554$ — matching an independent count from a
patched `Model.forward` **exactly**. The ratio $F_{\text{search}}/O = 8.1$ is why a
counter named `forward_passes` that incremented once per objective call understated
real compute ~8×.

$F_{\text{measure}}$ is reporting instrumentation, not attack cost;
$F_{\text{diag}}$ is the one-off equivalence check.

| claim | code | test | evidence class |
|---|---|---|---|
| hook counts every invocation | `metrics.ForwardCounter` | `test_metrics.py::test_forward_counter_counts_every_decoding_step` | independent oracle |
| hook always removed | context manager | `test_metrics.py::test_forward_counter_removes_its_hook` | regression |
| parts partition the whole | `_build_logs` | `test_attacker.py::test_model_forwards_are_counted_not_estimated` | self-consistency |
| **total equals an independent count** | — | `test_math.py::test_forward_count_matches_an_independent_oracle` | **independent oracle** (wraps `type(model).forward`) |

---

## 10. Strength of each verification

Not every check is equally strong, and the distinction matters when reading the
tables above. Four kinds appear:

* **independent oracle** — the claim is checked against a quantity computed by a
  different mechanism, so an error in the implementation cannot hide it.
* **algebraic identity** — a mathematical rewrite is checked against its own
  definition. Verifies the rewrite, not the inputs to it.
* **regression** — pins behaviour that was once wrong.
* **self-consistency** — the implementation is checked against itself. Weakest;
  useful for catching drift, not for establishing truth.

| claim | class |
|---|---|
| $\nabla J$ is the true gradient | **independent oracle** (central differences, from function values only) |
| the finite-difference method is sound | **independent oracle** (closed-form control, $10^{-9}$) |
| $-\nabla J$ is a descent direction | **independent oracle** (objective re-evaluated at $e \mp \eta g$) |
| first-order error is $O(\lVert e_v - e_i \rVert^2)$ | **independent oracle** (exact objective at shrinking $t$) |
| vectorised score $= \langle g_i,\ e_v - e_i \rangle$ | algebraic identity |
| `logsumexp` $= \log \sum p$ | algebraic identity (vs. naive softmax-sum-log) |
| total model forwards | **independent oracle** (`type(model).forward` wrapper) |
| forward parts partition the total | self-consistency |
| $J$ monotone non-increasing | self-consistency (reads logged values) |
| architecture alignment, embedding scale | regression + independent oracle (`input_ids` vs `inputs_embeds`) |

The load-bearing claims — that the gradient is real, that its sign is right, and
that the compute accounting is true — each rest on an independent oracle. The
word *proved* is used in this document only for algebraic facts such as the
`logsumexp` identity and the censoring case analysis; numerical and
model-specific results are described as *verified* or *measured*.

---

## 11. Open questions

* **Why is random competitive?** Diagnosed to the ranking, not the objective or
  the search (§5). The obvious mechanism — a single-position shortlist starving
  the budget — was fixed and did *not* close the gap, so the difference lies in
  *which* substitutions the gradient prefers, not how many it makes. Open.
* **Does the horizon matter on models that stop earlier?** 92 % of t5-small
  candidates reach the full horizon; a model terminating sooner would exercise the
  variable-length regime much harder.
* **Deployed causal models** need both interior-special-token preservation and
  arbitrary protected spans (so template scaffolding cannot itself be edited).
  Neither is implemented.
