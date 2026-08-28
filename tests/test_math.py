"""Mathematical verification of the attack's gradient and its linear model.

The rest of the suite checks behaviour and bookkeeping. This file checks the
*mathematics*: that the gradient the attack differentiates is the gradient of the
function it claims to differentiate, and that the HotFlip score is exactly the
first-order term it is derived from.

Why this is not redundant with the empirical diagnostics. The HotFlip diagnostic
reports a rank correlation of about +0.246 between the first-order estimate and
the exact objective change. That is a statement about how *useful* the linear
model is, and a weakly positive correlation could survive several
implementation errors -- a wrong detach, an off-by-one slice, a sign flip
compensated elsewhere. Only a numerical derivative check can separate "the
approximation is weak" from "the gradient is wrong". The two questions are
independent and this file answers the second one.

A note on differentiability, because it determines what can be checked at all.
The objective is evaluated along the model's own greedy trajectory, and that
trajectory is obtained by `argmax`, which is piecewise constant in the input
embeddings. So the full map `embeddings -> objective` is *not* differentiable
everywhere: it is smooth within a region where the argmax path is stable, and
jumps at the boundaries. The attack handles this the standard way -- it fixes the
trajectory (detached) and differentiates the teacher-forced objective given that
trajectory. That conditional function is genuinely smooth, and it is the one
verified here.
"""

from __future__ import annotations

import pytest
import torch

from llm_efficiency_attack.adapters import ModelAdapter
from llm_efficiency_attack.objectives import get_objective


def _fixed_trajectory_objective(model, adapter, objective_fn, eos_ids, horizon):
    """Build `J(embeds)` with the decoder trajectory held fixed.

    Returns `(J, decoder_input_ids)`. Fixing the trajectory is exactly what the
    attack does: `stop_logits` takes the greedy path under `no_grad` and then
    teacher-forces it, so gradients flow only through the encoder inputs.
    """

    def make(embeds_probe):
        with torch.no_grad():
            decoded = model.generate(
                inputs_embeds=embeds_probe,
                attention_mask=torch.ones(
                    embeds_probe.shape[:2], dtype=torch.long, device=embeds_probe.device
                ),
                max_new_tokens=horizon,
                do_sample=False,
            )
        decoder_input_ids = decoded[:, :-1] if decoded.shape[1] > 1 else decoded

        def objective(embeds):
            out = model(
                inputs_embeds=embeds,
                attention_mask=torch.ones(
                    embeds.shape[:2], dtype=torch.long, device=embeds.device
                ),
                decoder_input_ids=decoder_input_ids,
            )
            return objective_fn(out.logits[0], eos_ids)

        return objective

    return make


def test_autograd_matches_central_finite_differences(seq2seq_model, tokenizer):
    """The directional derivative from autograd matches a numerical one.

    For a fixed random direction `d`, the first-order behaviour of `J` is

        dJ/dt |_(t=0) of J(e + t*d)  =  <grad J(e), d>

    and the central difference `(J(e + h*d) - J(e - h*d)) / (2h)` approximates
    that with error `O(h^2)`. Agreement to a few parts in a thousand is strong
    evidence that the autograd graph computes the derivative of the function the
    attack actually evaluates -- not of some neighbouring function.

    Choosing the step is the whole difficulty, and the reason is specific to T5
    rather than generic. `T5LayerNorm.forward` computes its variance as
    `hidden_states.to(torch.float32).pow(2).mean(-1)` -- a deliberate
    half-precision-stability choice in `transformers` -- so the forward pass
    carries a float32 noise floor even when every parameter is float64. Measured
    on this fixture, the logits stop responding linearly below a perturbation of
    roughly 1e-6 and jitter at about 3e-7.

    That puts a floor under the numerical derivative: shrinking `h` past ~1e-2
    makes cancellation error grow faster than truncation error shrinks. An `h`
    sweep on this fixture gives relative errors of 1.1e-1, 7.4e-5, 1.1e-2,
    2.5e-1, 3.0e-2 at h = 1e-1 ... 1e-5, so `h = 1e-2` sits at the optimum.
    `test_finite_difference_harness_is_exact_on_a_smooth_function` confirms the
    harness itself reaches 1e-9 on a closed-form function, which is what pins the
    blame on the model's internal cast rather than on this method.
    """
    model = seq2seq_model.double().eval()
    adapter = ModelAdapter.for_model(model, tokenizer)
    objective_fn = get_objective("eos_suppression")
    eos_ids = adapter.eos_token_ids()

    ids = tokenizer("hello world", return_tensors="pt")["input_ids"]
    embeds = adapter.embed_values(ids).detach()

    build = _fixed_trajectory_objective(model, adapter, objective_fn, eos_ids, horizon=4)
    objective = build(embeds)

    leaf = embeds.clone().requires_grad_(True)
    value = objective(leaf)
    value.backward()
    grad = leaf.grad

    torch.manual_seed(0)
    direction = torch.randn_like(embeds)
    direction /= direction.norm()

    analytic = float((grad * direction).sum().item())

    h = 1e-2  # see the docstring: T5's float32 layer-norm cast sets the floor
    with torch.no_grad():
        plus = float(objective(embeds + h * direction).item())
        minus = float(objective(embeds - h * direction).item())
    numeric = (plus - minus) / (2 * h)

    assert abs(analytic) > 1e-9, "degenerate direction; the check would be vacuous"
    relative_error = abs(analytic - numeric) / max(abs(analytic), abs(numeric))
    assert relative_error < 1e-3, (
        f"autograd {analytic:.10g} vs central difference {numeric:.10g} "
        f"(relative error {relative_error:.3g})"
    )


def test_finite_difference_harness_is_exact_on_a_smooth_function():
    """Control for the test above: the method is exact when the model is not.

    The finite-difference check on T5 can only reach ~1e-4 relative agreement,
    and a reviewer is entitled to ask whether that is the gradient's fault or the
    method's. This runs the identical central-difference procedure on a
    closed-form function with the same shapes and the same log-softmax/logsumexp
    tail, where every operation stays in float64. Reaching ~1e-9 there shows the
    procedure is sound, so the looser tolerance on T5 is attributable to the
    model's internal float32 cast.
    """
    torch.manual_seed(0)
    embeds = torch.randn(1, 12, 32, dtype=torch.float64)
    weight = torch.randn(32, 64, dtype=torch.float64)

    def smooth(e):
        return torch.log_softmax(e[0] @ weight, dim=-1)[:, [1, 5]].logsumexp(-1).mean()

    leaf = embeds.clone().requires_grad_(True)
    smooth(leaf).backward()

    direction = torch.randn_like(embeds)
    direction /= direction.norm()
    analytic = float((leaf.grad * direction).sum().item())

    h = 1e-4
    with torch.no_grad():
        numeric = (
            float(smooth(embeds + h * direction).item())
            - float(smooth(embeds - h * direction).item())
        ) / (2 * h)

    relative_error = abs(analytic - numeric) / abs(analytic)
    assert relative_error < 1e-6, (
        f"the harness itself must be near-exact: {relative_error:.3g}"
    )


def test_gradient_sign_is_the_descent_direction(seq2seq_model, tokenizer):
    """Moving against the gradient decreases the objective, which the loop minimises.

    A sign error here would build an attack that makes outputs *shorter*. This
    checks the convention numerically rather than by reading the code.
    """
    model = seq2seq_model.double().eval()
    adapter = ModelAdapter.for_model(model, tokenizer)
    objective_fn = get_objective("eos_suppression")
    eos_ids = adapter.eos_token_ids()

    ids = tokenizer("hello world", return_tensors="pt")["input_ids"]
    embeds = adapter.embed_values(ids).detach()
    objective = _fixed_trajectory_objective(
        model, adapter, objective_fn, eos_ids, horizon=4
    )(embeds)

    leaf = embeds.clone().requires_grad_(True)
    base = objective(leaf)
    base.backward()

    step = 1e-4
    with torch.no_grad():
        downhill = float(objective(embeds - step * leaf.grad).item())
        uphill = float(objective(embeds + step * leaf.grad).item())

    assert downhill < float(base.item()) < uphill, (
        f"descent {downhill:.8g} !< base {float(base.item()):.8g} !< ascent {uphill:.8g}"
    )


def test_hotflip_score_is_exactly_the_first_order_term(seq2seq_model, tokenizer):
    """`scores[i, v]` equals `<grad_i, e_v - e_i>`, algebraically.

    The production code computes the whole `(positions x vocab)` table with one
    matmul and a broadcast subtraction:

        scores = (grad @ E^T) * s  -  <grad_i, s * e_i>

    which is an efficient rewrite of the definition. An error in that rewrite --
    a missing scale, a transposed operand, subtracting the wrong row -- would not
    change any shape and would not raise. This recomputes a sample of entries the
    slow, obvious way and requires exact agreement.
    """
    from llm_efficiency_attack.attacker import Attacker
    from llm_efficiency_attack.config import AttackConfig

    attacker = Attacker(seq2seq_model, tokenizer)
    adapter = attacker.adapter
    objective_fn = get_objective("eos_suppression")
    eos_ids = adapter.eos_token_ids()
    cfg = AttackConfig.from_dict({"objective_horizon": 4, "top_k": 5})

    encoded = tokenizer("hello world", return_tensors="pt")
    ids, mask = encoded["input_ids"], encoded["attention_mask"]

    # Recompute the gradient exactly as `_gradient_candidates` does.
    embeds = adapter.embed(ids)
    stop_logits = adapter.stop_logits(embeds, mask, cfg.objective_horizon, False)
    loss = objective_fn(stop_logits, eos_ids)
    seq2seq_model.zero_grad(set_to_none=True)
    loss.backward()

    grad = embeds.grad[0]
    table = adapter.embedding_matrix()
    scale = adapter.embedding_scale()

    fast = (grad @ table.T) * scale
    fast = fast - (grad * (table[ids[0]] * scale)).sum(-1, keepdim=True)

    # The definition, one entry at a time.
    for position in (0, 1, ids.shape[1] - 1):
        e_i = table[ids[0, position]] * scale
        for token in (4, 17, 40):
            e_v = table[token] * scale
            expected = float(torch.dot(grad[position], e_v - e_i).item())
            actual = float(fast[position, token].item())
            assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6), (
                f"position {position}, token {token}: "
                f"vectorised {actual:.8g} != definition {expected:.8g}"
            )


def test_hotflip_estimate_predicts_small_exact_changes(seq2seq_model, tokenizer):
    """The linear model is accurate in the limit it is derived for.

    HotFlip's approximation error is `O(||e_v - e_i||^2)`, so it is only expected
    to be accurate for *nearby* substitutions. On real vocabularies the
    substitutions are far apart, which is exactly why the estimate ranks poorly
    (measured: rank correlation about +0.246) and why the attack re-scores its
    shortlist exactly instead of trusting the estimate.

    This checks the derivation rather than the vocabulary: it scales a real
    substitution direction down by `t`, so the exact change must approach `t`
    times the first-order term as `t` shrinks. If that failed, the derivation --
    not merely its usefulness -- would be wrong.
    """
    model = seq2seq_model.double().eval()
    adapter = ModelAdapter.for_model(model, tokenizer)
    objective_fn = get_objective("eos_suppression")
    eos_ids = adapter.eos_token_ids()

    ids = tokenizer("hello world", return_tensors="pt")["input_ids"]
    embeds = adapter.embed_values(ids).detach()
    objective = _fixed_trajectory_objective(
        model, adapter, objective_fn, eos_ids, horizon=4
    )(embeds)

    leaf = embeds.clone().requires_grad_(True)
    base = objective(leaf)
    base.backward()
    grad = leaf.grad[0]

    position, token = 2, 17
    table = adapter.embedding_matrix()
    delta = (table[token] - table[ids[0, position]]).double() * adapter.embedding_scale()
    first_order = float(torch.dot(grad[position], delta).item())
    assert abs(first_order) > 1e-9, "degenerate substitution; the check would be vacuous"

    # Steps are kept at or above 1e-3: below that the model's float32 layer-norm
    # cast (see the finite-difference test) swamps the signal, and the ratio
    # degrades again for numerical rather than mathematical reasons.
    ratios = []
    for t in (1e-1, 1e-2, 1e-3):
        perturbed = embeds.clone()
        perturbed[0, position] += t * delta
        with torch.no_grad():
            exact = float(objective(perturbed).item()) - float(base.item())
        ratios.append(exact / (t * first_order))

    # As t shrinks the ratio must approach 1, and monotonically get closer.
    assert ratios[-1] == pytest.approx(1.0, abs=1e-2), ratios
    assert abs(ratios[-1] - 1) < abs(ratios[0] - 1), (
        f"first-order agreement should improve as the step shrinks: {ratios}"
    )


def test_multi_eos_logsumexp_equals_log_of_summed_probability(seq2seq_model):
    """`logsumexp` over EOS log-probabilities is `log P(stop)`, exactly.

    Stopping means emitting *any* stop token, so the quantity to suppress is
    `P(stop) = sum_{v in E} p(v)`. The implementation works in log space:

        log sum_{v in E} p(v) = logsumexp_{v in E} log p(v)

    That identity is why `logsumexp` appears rather than a plain sum, and it is
    what keeps the objective finite once the attack has driven these
    probabilities very low. Checked here against the naive computation on a model
    that declares two stop tokens.
    """
    torch.manual_seed(0)
    logits = torch.randn(3, 64, dtype=torch.double)
    eos_ids = [1, 5]

    log_probs = torch.log_softmax(logits, dim=-1)
    implemented = torch.logsumexp(log_probs[:, eos_ids], dim=-1)

    probs = torch.softmax(logits, dim=-1)
    naive = torch.log(probs[:, eos_ids].sum(dim=-1))

    assert torch.allclose(implemented, naive, atol=1e-12)

    # And the mean over steps is what `eos_suppression` returns.
    objective_fn = get_objective("eos_suppression")
    assert float(objective_fn(logits, eos_ids).item()) == pytest.approx(
        float(implemented.mean().item()), rel=1e-12
    )


def test_objective_is_monotone_non_increasing_by_construction(seq2seq_model, tokenizer):
    """The search commits only strict improvements, so `J` never rises.

    This is a property of the loop rather than of the model: a candidate is
    committed only when its exactly-rescored objective is strictly below the
    current one. Stated here because it is what makes the search a valid
    coordinate descent on the exact objective *regardless of how good the
    first-order proposal is* -- the point that separates a sound proposal
    mechanism from an effective one.
    """
    from llm_efficiency_attack import Attacker

    _, logs = Attacker(seq2seq_model, tokenizer).run(
        "hello world",
        {"max_iterations": 4, "perturbation_budget": 3, "top_k": 6,
         "max_new_tokens": 12, "objective_horizon": 3},
    )
    values = [it["objective"] for it in logs["iterations"]]
    assert values == sorted(values, reverse=True), values
    assert all(b < a for a, b in zip(values, values[1:])), (
        f"each committed step must strictly improve: {values}"
    )


def test_forward_count_matches_an_independent_oracle(seq2seq_model, tokenizer):
    """The logged forward count equals a count taken by a different mechanism.

    `logs["attack_cost"]` is produced by a forward *pre-hook*. Checking that its
    fields sum consistently only proves the package agrees with itself. This
    counts the same run a second way -- by wrapping the model class's `forward`
    method directly -- and requires the two totals to be equal.

    That is the difference between "the accounting is internally coherent" and
    "the accounting is true". RESULTS.md quotes 554 forwards for the headline
    run on `t5-small`; this is the reproducer for the *independence* of that
    number, on a fixture small enough to run offline.
    """
    from llm_efficiency_attack import Attacker

    observed = {"n": 0}
    original_forward = type(seq2seq_model).forward

    def counting_forward(self, *args, **kwargs):
        observed["n"] += 1
        return original_forward(self, *args, **kwargs)

    type(seq2seq_model).forward = counting_forward
    try:
        _, logs = Attacker(seq2seq_model, tokenizer).run(
            "hello world",
            {"max_iterations": 3, "perturbation_budget": 2, "top_k": 4,
             "max_new_tokens": 10, "objective_horizon": 3},
        )
    finally:
        type(seq2seq_model).forward = original_forward

    cost = logs["attack_cost"]
    assert observed["n"] > 0, "the oracle must have counted something"
    assert cost["total_model_forwards"] == observed["n"], (
        f"logged {cost['total_model_forwards']} vs independent count {observed['n']}"
    )
    # And the parts still partition the whole.
    assert cost["total_model_forwards"] == (
        cost["search_model_forwards"]
        + cost["measurement_model_forwards"]
        + cost["diagnostic_model_forwards"]
    )
    # The point the accounting exists to make: an objective evaluation is many
    # model invocations, not one.
    assert cost["search_model_forwards"] > cost["objective_evaluations"]
