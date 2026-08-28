"""End-to-end attack behaviour.

Most of these tests assert *mechanics*, not attack strength. The default fixtures
are randomly initialised models, which have no learned notion of when to stop, so
demanding a large efficiency ratio here would be testing the fixture rather than
the code. Effectiveness on a trained model is measured in `examples/ablation.py`
and recorded in `results/`.

The exceptions are the invariants that would silently invalidate a result if they
broke: the perturbation budget applying to the text the caller actually receives,
the candidate set excluding ids that cannot survive a round trip, and the
compute accounting counting real model invocations.
"""

from __future__ import annotations

import json

import pytest
import torch
from transformers import BatchEncoding

from conftest import (
    ALPHABET,
    EOS,
    FIRST_TOKEN_ID,
    SPECIAL_IDS,
    TOKENIZER_VOCAB,
    ToyTokenizer,
)
from llm_efficiency_attack import Attacker
from llm_efficiency_attack.adapters import forbidden_token_ids
from llm_efficiency_attack.attacker import _ComputeCounters
from llm_efficiency_attack.config import AttackConfig
from llm_efficiency_attack.objectives import get_objective

FAST = {
    "max_iterations": 3,
    "perturbation_budget": 2,
    "top_k": 6,
    "max_new_tokens": 12,
    "objective_horizon": 3,
}


# --------------------------------------------------------------- public shape


def test_public_api_shape(seq2seq_model, tokenizer):
    """`run` returns `(adv_x, logs)` with the documented top-level log sections."""
    attack = Attacker(seq2seq_model, tokenizer)
    adv_x, logs = attack.run("hello world", FAST)

    assert isinstance(adv_x, str)
    assert isinstance(logs, dict)
    for key in ("config", "benign", "adversarial", "cost", "censored",
                "perturbation", "attack_cost", "diagnostics", "iterations"):
        assert key in logs, f"logs missing {key!r}"


def test_one_argument_constructor_is_the_documented_path(seq2seq_model, tokenizer):
    """`Attacker(model)` is the signature the task specifies.

    The fixture is built from a config and records no `_name_or_path`, so the
    tokenizer cannot be inferred and the failure must be an actionable message
    rather than an `OSError` from the Hub. The one-argument path against a real
    checkpoint is covered by the integration test in `test_integration.py`.
    """
    with pytest.raises(ValueError, match="Pass `Attacker\\(model, tokenizer\\)`"):
        Attacker(seq2seq_model)

    # And the two-argument form used everywhere else stays equivalent.
    assert Attacker(seq2seq_model, tokenizer).tokenizer is tokenizer


def test_logs_are_json_serialisable(seq2seq_model, tokenizer):
    """A reviewer must be able to save and diff a run."""
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    json.loads(json.dumps(logs))


def test_works_on_causal_models_unchanged(causal_model, tokenizer):
    """Model-agnosticism: only the `model` argument differs from the seq2seq test."""
    adv_x, logs = Attacker(causal_model, tokenizer).run("hello world", FAST)
    assert isinstance(adv_x, str)
    assert logs["cost"]["benign_output_tokens"] > 0


# ------------------------------------------------------------ search mechanics


def test_objective_never_increases(seq2seq_model, tokenizer):
    """The loop commits a substitution only when it strictly improves.

    Monotonicity is the invariant that makes the search meaningful; if it were
    violated the attack would be wandering rather than optimising. The first
    entry is the benign starting point, so the whole series must be descending.
    """
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    values = [it["objective"] for it in logs["iterations"]]
    assert values == sorted(values, reverse=True), values


def test_iterations_start_from_the_benign_baseline(seq2seq_model, tokenizer):
    """Without the baseline entry the objective trajectory has no origin."""
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    first = logs["iterations"][0]
    assert first["iteration"] == -1
    assert first["tokens_changed"] == 0
    assert first["output_tokens"] == logs["cost"]["benign_output_tokens"]


def test_protected_prefix_is_never_touched(seq2seq_model, tokenizer):
    """Instruction prefixes must survive, or the attack measures task damage."""
    cfg = dict(FAST, protected_prefix_tokens=4)
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", cfg)

    original = logs["perturbation"]["original_token_ids"]
    adversarial = logs["perturbation"]["adversarial_token_ids"]
    assert original[:4] == adversarial[:4]
    assert all(p >= 4 for p in logs["perturbation"]["positions_changed"])


def test_unknown_objective_is_rejected(seq2seq_model, tokenizer):
    with pytest.raises(ValueError, match="Unknown objective"):
        Attacker(seq2seq_model, tokenizer).run(
            "hello world", dict(FAST, objective="does_not_exist")
        )


def test_rejects_a_fully_protected_input(seq2seq_model, tokenizer):
    """Nothing left to perturb is a configuration error, not a silent no-op."""
    with pytest.raises(ValueError, match="No perturbable token"):
        Attacker(seq2seq_model, tokenizer).run(
            "hi", dict(FAST, protected_prefix_tokens=999)
        )


# ------------------------------------------------- perturbation realisation


def test_perturbation_budget_bounds_positions_touched(seq2seq_model, tokenizer):
    """The budget is a hard limit on how many positions the search may write to."""
    cfg = dict(FAST, perturbation_budget=1, max_iterations=5)
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", cfg)

    pert = logs["perturbation"]
    assert pert["positions_touched"] <= 1
    assert len(pert["positions_changed"]) == pert["positions_touched"]

    original, adversarial = pert["original_token_ids"], pert["adversarial_token_ids"]
    assert len(original) == len(adversarial), "an attack must not change sequence length"
    assert sum(a != b for a, b in zip(original, adversarial)) <= 1


def test_hamming_distance_is_reported_separately_from_positions_touched(
    seq2seq_model, tokenizer
):
    """They are different quantities and conflating them overstates the guarantee.

    A position written twice counts once as touched; a position restored to its
    original token still counts as touched but contributes nothing to the Hamming
    distance. So Hamming is bounded by touched, never the other way round.
    """
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    pert = logs["perturbation"]

    original, adversarial = pert["original_token_ids"], pert["adversarial_token_ids"]
    assert pert["hamming_distance"] == sum(
        a != b for a, b in zip(original, adversarial)
    )
    assert pert["hamming_distance"] <= pert["positions_touched"] <= pert["budget"]


def test_returned_text_retokenises_to_the_optimised_ids(seq2seq_model, tokenizer):
    """The budget must apply to what the caller feeds the model, not only to ids.

    `encode(decode(ids))` is not an identity in general, so the attack rejects any
    candidate that would break it. This asserts the resulting guarantee end to end.
    """
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    assert logs["perturbation"]["round_trip_exact"] is True


def test_returned_text_retokenises_on_causal_models_too(causal_model, tokenizer):
    _, logs = Attacker(causal_model, tokenizer).run("hello world", FAST)
    assert logs["perturbation"]["round_trip_exact"] is True


def test_adv_text_really_does_round_trip(seq2seq_model, tokenizer):
    """Check the invariant against the tokenizer directly, not just via the flag."""
    adv_x, logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    realised = tokenizer(adv_x)["input_ids"][0].tolist()
    assert realised == logs["perturbation"]["adversarial_token_ids"]


# ------------------------------------------------------- candidate legality


def test_fixture_actually_has_illegal_ids(tokenizer, seq2seq_model):
    """Guard the guard: the exclusion tests below would be vacuous without these."""
    illegal = forbidden_token_ids(tokenizer, seq2seq_model.get_input_embeddings().weight.shape[0])
    assert set(SPECIAL_IDS) <= illegal, "special ids must be excluded"
    assert {TOKENIZER_VOCAB, TOKENIZER_VOCAB + 1} <= illegal, (
        "embedding rows past the tokenizer vocabulary must be excluded"
    )


def _candidate_setup(model, tokenizer, **overrides):
    attacker = Attacker(model, tokenizer)
    cfg = AttackConfig.from_dict(dict(FAST, **overrides))
    encoded = tokenizer("hello world", return_tensors="pt")
    illegal = forbidden_token_ids(
        tokenizer, attacker.adapter.embedding_matrix().shape[0]
    )
    allowed = attacker._eligible_positions(encoded["input_ids"], cfg)
    return attacker, cfg, encoded, illegal, allowed


def test_gradient_candidates_exclude_special_and_out_of_vocab_ids(
    seq2seq_model, tokenizer
):
    """HotFlip scores the whole embedding table; the mask is what keeps it honest.

    Without this, the search can propose a special token (deleted on decode) or a
    surplus embedding row (decodes to the empty string), either of which changes
    the sequence length of the input the model is actually given.
    """
    attacker, cfg, encoded, illegal, allowed = _candidate_setup(
        seq2seq_model, tokenizer, top_k=40
    )
    candidates = attacker._gradient_candidates(
        encoded["input_ids"], encoded["attention_mask"],
        get_objective("eos_suppression"), attacker.adapter.eos_token_ids(),
        cfg, allowed, illegal, False, _ComputeCounters(),
    )

    assert candidates, "the fixture must produce candidates for this to mean anything"
    assert all(token not in illegal for _, token in candidates)
    assert all(position in allowed for position, _ in candidates)
    assert all(
        token != int(encoded["input_ids"][0, position].item())
        for position, token in candidates
    ), "proposing the token already in place wastes an evaluation"


def test_random_control_draws_from_the_same_candidate_space(seq2seq_model, tokenizer):
    """The control is only a control if its candidate space matches the gradient's."""
    attacker, cfg, encoded, illegal, allowed = _candidate_setup(
        seq2seq_model, tokenizer, top_k=40
    )
    legal_ids = [
        i
        for i in range(attacker.adapter.embedding_matrix().shape[0])
        if i not in illegal
    ]
    candidates = attacker._random_candidates(
        encoded["input_ids"], allowed, legal_ids, cfg
    )

    assert len(candidates) == cfg.top_k, "the control must get the same evaluation count"
    assert all(token not in illegal for _, token in candidates)
    assert all(position in allowed for position, _ in candidates)
    assert all(
        token != int(encoded["input_ids"][0, position].item())
        for position, token in candidates
    ), "a no-op candidate would waste one of the control's evaluations"


def test_attack_never_commits_an_illegal_token(seq2seq_model, tokenizer):
    """End-to-end consequence of the two tests above."""
    _, logs = Attacker(seq2seq_model, tokenizer).run(
        "hello world", dict(FAST, top_k=20)
    )
    illegal = forbidden_token_ids(
        tokenizer, seq2seq_model.get_input_embeddings().weight.shape[0]
    )
    changed = logs["perturbation"]["positions_changed"]
    adversarial = logs["perturbation"]["adversarial_token_ids"]
    assert all(adversarial[p] not in illegal for p in changed)


# ------------------------------------------------------------ reproducibility


def test_same_seed_gives_same_result_for_the_random_control(seq2seq_model, tokenizer):
    """Requirement 5: same config plus same input yields the same output."""
    cfg = dict(FAST, strategy="random", seed=123)
    a_x, a_logs = Attacker(seq2seq_model, tokenizer).run("hello world", cfg)
    b_x, b_logs = Attacker(seq2seq_model, tokenizer).run("hello world", cfg)

    assert a_x == b_x
    assert a_logs["perturbation"]["adversarial_token_ids"] == \
        b_logs["perturbation"]["adversarial_token_ids"]


def test_gradient_path_is_reproducible(seq2seq_model, tokenizer):
    """The gradient strategy is deterministic too, and nothing was testing it."""
    a_x, a_logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    b_x, b_logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)

    assert a_x == b_x
    assert a_logs["perturbation"]["adversarial_token_ids"] == \
        b_logs["perturbation"]["adversarial_token_ids"]
    assert [i["objective"] for i in a_logs["iterations"]] == \
        [i["objective"] for i in b_logs["iterations"]]


def test_different_seeds_diverge_for_the_random_control(seq2seq_model, tokenizer):
    """If the seed had no effect, the reproducibility test above would be vacuous."""
    cfg_a = dict(FAST, strategy="random", seed=1, top_k=3)
    cfg_b = dict(FAST, strategy="random", seed=999, top_k=3)
    _, a = Attacker(seq2seq_model, tokenizer).run("hello world", cfg_a)
    _, b = Attacker(seq2seq_model, tokenizer).run("hello world", cfg_b)
    assert a["perturbation"]["positions_changed"] != b["perturbation"]["positions_changed"] \
        or a["perturbation"]["adversarial_token_ids"] != b["perturbation"]["adversarial_token_ids"]


# --------------------------------------------------------------- attack cost


def test_random_control_uses_no_gradients(seq2seq_model, tokenizer):
    """The control must be a first-class strategy, so the comparison is honest."""
    _, logs = Attacker(seq2seq_model, tokenizer).run(
        "hello world", dict(FAST, strategy="random")
    )
    assert logs["config"]["strategy"] == "random"
    assert logs["attack_cost"]["gradient_evaluations"] == 0


def test_gradient_strategy_uses_gradients(seq2seq_model, tokenizer):
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    assert logs["attack_cost"]["gradient_evaluations"] > 0


def test_model_forwards_are_counted_not_estimated(seq2seq_model, tokenizer):
    """One objective evaluation is many model invocations, and the logs must say so.

    Evaluating the objective runs `generate()` for up to `objective_horizon`
    decoding steps plus one teacher-forced forward. A counter that incremented
    once per objective call would understate the attack's real compute by close to
    an order of magnitude, which is exactly what the previous `forward_passes`
    field did.
    """
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    cost = logs["attack_cost"]

    assert cost["objective_evaluations"] > 0
    assert cost["search_model_forwards"] > cost["objective_evaluations"], (
        "each objective evaluation costs several model forwards"
    )
    assert cost["measurement_model_forwards"] > 0
    assert cost["diagnostic_model_forwards"] == 2, "the equivalence check is two forwards"
    assert cost["total_model_forwards"] == (
        cost["search_model_forwards"]
        + cost["measurement_model_forwards"]
        + cost["diagnostic_model_forwards"]
    )


def test_measurement_compute_is_separated_from_search_compute(seq2seq_model, tokenizer):
    """Cost-metric generation is instrumentation and must not inflate attack cost."""
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    cost = logs["attack_cost"]
    assert cost["measurement_model_forwards"] < cost["total_model_forwards"]


# ------------------------------------------------------------------ censoring


def test_censoring_reports_both_censored_as_uninformative(seq2seq_model, tokenizer):
    """Untrained fixtures never emit EOS, so both runs hit the ceiling.

    Two right-censored observations divided by each other bound the true ratio in
    neither direction. Calling that a lower bound -- which this package used to do
    -- is a false statement, so the label must be `uninformative`.
    """
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    censored = logs["censored"]
    assert censored["benign_hit_ceiling"] is True
    assert censored["adversarial_hit_ceiling"] is True
    assert censored["interpretation"] == "uninformative"
    assert "NEITHER direction" in censored["note"]
    assert "lower bound" not in censored["note"]


# --------------------------------------------------------------- model state


def test_run_restores_the_models_training_mode(seq2seq_model, tokenizer):
    """`run` borrows the caller's model; it must hand it back as it found it."""
    seq2seq_model.train()
    Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    assert seq2seq_model.training is True

    seq2seq_model.eval()
    Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    assert seq2seq_model.training is False


def test_embedding_equivalence_is_checked_and_logged(seq2seq_model, tokenizer):
    """The attack differentiates through `inputs_embeds`; that path must match."""
    _, logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    deviation = logs["diagnostics"]["embedding_equivalence_max_logit_deviation"]
    assert deviation == pytest.approx(0.0, abs=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_runs_on_cuda(seq2seq_model, tokenizer):
    """Every stage of `run()` on an actual CUDA device.

    Regression for the cost metric building CPU tensors for a CUDA model:
    `generate()` performs no device migration, so a mismatch raises rather than
    degrading, and the whole run would die at the benign measurement.

    Asserting only that a string came back would pass even if the run had
    silently stayed on CPU, so this checks residency explicitly and then
    re-checks the invariants that the CPU suite covers -- round trip, budget,
    accounting, embedding equivalence -- because a device bug can corrupt those
    without raising. Skipped on CPU-only machines; `test_metrics.py` pins the
    device invariant in a form that runs everywhere.
    """
    generate_devices = {}
    original = seq2seq_model.generate

    def spy(*args, **kwargs):
        if "input_ids" in kwargs:
            generate_devices["input_ids"] = kwargs["input_ids"].device.type
        if "inputs_embeds" in kwargs:
            generate_devices["inputs_embeds"] = kwargs["inputs_embeds"].device.type
        return original(*args, **kwargs)

    seq2seq_model.generate = spy
    try:
        adv_x, logs = Attacker(seq2seq_model, tokenizer).run(
            "hello world", dict(FAST, device="cuda")
        )
    finally:
        seq2seq_model.generate = original

    # The model really moved, and every tensor reaching generate() came with it.
    assert next(seq2seq_model.parameters()).device.type == "cuda"
    assert generate_devices, "generate() was never called"
    assert set(generate_devices.values()) == {"cuda"}, generate_devices

    # Measurement ran on both sides.
    assert isinstance(adv_x, str)
    assert logs["cost"]["benign_output_tokens"] > 0
    assert logs["cost"]["adversarial_output_tokens"] > 0
    assert logs["censored"]["interpretation"] in {
        "point_estimate", "lower_bound", "upper_bound", "uninformative"
    }

    # Gradients were taken on device, and the search actually ran.
    cost = logs["attack_cost"]
    assert cost["gradient_evaluations"] > 0
    assert cost["objective_evaluations"] > 0
    assert cost["search_model_forwards"] > cost["objective_evaluations"]
    assert cost["total_model_forwards"] == (
        cost["search_model_forwards"]
        + cost["measurement_model_forwards"]
        + cost["diagnostic_model_forwards"]
    )

    # Invariants must hold identically on device.
    perturbation = logs["perturbation"]
    assert perturbation["round_trip_exact"] is True
    assert perturbation["hamming_distance"] <= perturbation["positions_touched"]
    assert perturbation["positions_touched"] <= perturbation["budget"]
    assert logs["diagnostics"]["embedding_equivalence_max_logit_deviation"] == \
        pytest.approx(0.0, abs=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_causal_model_runs_on_cuda(causal_model, tokenizer):
    """The causal adapter concatenates continuation embeddings; do that on device.

    A separate test because the causal `stop_logits` path builds tensors the
    seq2seq path does not -- `torch.cat` of prompt and continuation embeddings,
    and an extended attention mask -- each of which is its own chance to place a
    tensor on the wrong device.
    """
    adv_x, logs = Attacker(causal_model, tokenizer).run(
        "hello world", dict(FAST, device="cuda")
    )
    assert next(causal_model.parameters()).device.type == "cuda"
    assert isinstance(adv_x, str)
    assert logs["cost"]["benign_output_tokens"] > 0
    assert logs["perturbation"]["round_trip_exact"] is True
    assert logs["attack_cost"]["gradient_evaluations"] > 0


# ------------------------------------------- exact text realization boundary


class _InteriorSpecialTokenizer(ToyTokenizer):
    """A tokenizer whose inputs carry a special token in the *interior*.

    This reproduces, deterministically and offline, the mechanism that real chat
    templates hit: `<|im_start|>` and `<|im_end|>` sit inside the input, and
    `decode(..., skip_special_tokens=True)` removes them, so re-tokenising the
    decoded text yields a shorter, different sequence. Measured on the real
    thing, SmolLM2-135M-Instruct goes from 37 tokens to 32.

    Everything else is inherited, so the only difference from the fixture used by
    the rest of this file is where the special token sits.
    """

    MARKER = "#"

    def __call__(self, text, return_tensors=None):
        # `#` stands in for a template token such as `<|im_start|>`: it is part
        # of the input text and it encodes to a *special* id. Text realization
        # drops special ids, so the character disappears from the decoded string
        # and re-encoding cannot put it back -- which is exactly why the real
        # 37-token SmolLM2 template comes back as 32 tokens.
        ids = [
            EOS if c == self.MARKER else FIRST_TOKEN_ID + ALPHABET.index(c)
            for c in text
            if c == self.MARKER or c in ALPHABET
        ]
        ids = (ids or [FIRST_TOKEN_ID]) + [EOS]
        data = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }
        return BatchEncoding(data, tensor_type=return_tensors)


def test_input_with_interior_special_tokens_is_rejected(seq2seq_model):
    """Inputs that cannot be realized exactly as text are refused, not measured.

    The library guarantees that the returned `adv_x` re-tokenises to exactly the
    ids it optimised. An input carrying interior special tokens breaks that
    before any substitution is made, because text realization drops them. The
    correct behaviour is to refuse: measuring would report an efficiency number
    for a token sequence other than the one that was optimised, and a
    perturbation budget that does not apply to the text handed back.

    This is a true scope boundary, and it is what currently puts chat-templated
    inputs -- and therefore normally-deployed instruction-tuned causal models --
    outside the exact-realization scope of the public interface.
    """
    tokenizer = _InteriorSpecialTokenizer()

    # 1. The mismatch is real, and exists independently of the attack.
    ids = tokenizer("hello#world")["input_ids"][0].tolist()
    realised = tokenizer(
        tokenizer.decode(ids, skip_special_tokens=True)
    )["input_ids"][0].tolist()
    assert ids != realised, "fixture must actually exhibit the mismatch"
    assert len(realised) < len(ids), "the interior special token must be dropped"

    # 2. The attack refuses rather than returning altered text.
    with pytest.raises(RuntimeError) as excinfo:
        Attacker(seq2seq_model, tokenizer).run("hello#world", FAST)

    # 3. The error explains what happened and why it matters.
    message = str(excinfo.value)
    assert "re-tokenise" in message
    assert "perturbation budget does not apply" in message
    assert "optimised=" in message and "realised=" in message


def test_rejection_yields_no_measurement(seq2seq_model):
    """No efficiency number is produced for an unrealizable input.

    Returning a ratio alongside the warning would be worse than failing: a
    reviewer could quote it, and it would describe a different token sequence
    than the one optimised.
    """
    attacker = Attacker(seq2seq_model, _InteriorSpecialTokenizer())
    try:
        attacker.run("hello#world", FAST)
    except RuntimeError:
        pass
    else:  # pragma: no cover - the call above must raise
        pytest.fail("an unrealizable input must not return a result")


# --------------------------------------------- stratified candidate proposal


def test_stratified_shortlist_spans_positions(seq2seq_model, tokenizer):
    """`gradient_stratified` must actually diversify across positions.

    The global top-k shortlist was measured on t5-small to draw 90-100% of its
    candidates from a single token position even at `top_k=100`, because one
    position dominates the gradient magnitude. That is what stalls the search
    once that position has been edited. This asserts the stratified variant
    covers every allowed position instead, at the same evaluation budget.
    """
    attacker, cfg, encoded, illegal, allowed = _candidate_setup(
        seq2seq_model, tokenizer, top_k=12
    )
    args = (
        encoded["input_ids"], encoded["attention_mask"],
        get_objective("eos_suppression"), attacker.adapter.eos_token_ids(),
        cfg, allowed, illegal, False, _ComputeCounters(),
    )

    stratified = attacker._gradient_candidates(*args, stratified=True)
    assert len(stratified) == cfg.top_k, "the evaluation budget must be unchanged"
    assert {p for p, _ in stratified} == set(allowed), (
        "every allowed position must be represented"
    )
    assert all(token not in illegal for _, token in stratified)
    assert all(
        token != int(encoded["input_ids"][0, position].item())
        for position, token in stratified
    )


def test_stratified_is_a_separate_strategy_not_the_default(seq2seq_model, tokenizer):
    """The default must stay `gradient`, so published results remain reproducible."""
    from llm_efficiency_attack.config import STRATEGIES

    assert AttackConfig.from_dict(None).strategy == "gradient"
    assert "gradient_stratified" in STRATEGIES

    _, logs = Attacker(seq2seq_model, tokenizer).run(
        "hello world", dict(FAST, strategy="gradient_stratified")
    )
    assert logs["config"]["strategy"] == "gradient_stratified"
    assert logs["attack_cost"]["gradient_evaluations"] > 0
    assert logs["perturbation"]["round_trip_exact"] is True


# ---------------------------------------------------- token-id input (task spec)


def _ids(tokenizer, text):
    return tokenizer(text)["input_ids"][0].tolist()


def test_token_ids_are_accepted_and_returned_in_kind(seq2seq_model, tokenizer):
    """The task types `x` as "text / tokens" and `adv_x` as "input".

    So token ids in must give token ids back, not text: `adv_x` mirrors the
    representation of `x` so the caller can feed it to the model exactly as they
    fed `x`.
    """
    ids = _ids(tokenizer, "hello world")
    adv_x, logs = Attacker(seq2seq_model, tokenizer).run(ids, FAST)

    assert isinstance(adv_x, list)
    assert all(isinstance(i, int) for i in adv_x)
    assert len(adv_x) == len(ids)
    assert adv_x == logs["perturbation"]["adversarial_token_ids"]
    assert logs["perturbation"]["input_mode"] == "tokens"


def test_token_and_text_input_optimise_the_same_sequence(seq2seq_model, tokenizer):
    """The two representations are two doors into one attack, not two attacks."""
    text = "hello world"
    from_text = Attacker(seq2seq_model, tokenizer).run(text, FAST)[1]
    from_ids = Attacker(seq2seq_model, tokenizer).run(_ids(tokenizer, text), FAST)[1]

    assert (from_ids["perturbation"]["adversarial_token_ids"]
            == from_text["perturbation"]["adversarial_token_ids"])
    assert (from_ids["cost"]["adversarial_output_tokens"]
            == from_text["cost"]["adversarial_output_tokens"])


@pytest.mark.parametrize("wrap", [list, tuple, torch.tensor,
                                  lambda v: torch.tensor([v])])
def test_token_input_accepts_the_usual_containers(seq2seq_model, tokenizer, wrap):
    ids = _ids(tokenizer, "hello world")
    adv_x, logs = Attacker(seq2seq_model, tokenizer).run(wrap(ids), FAST)
    assert adv_x == logs["perturbation"]["adversarial_token_ids"]


def test_token_input_on_causal_models(causal_model, tokenizer):
    ids = _ids(tokenizer, "hello world")
    adv_x, logs = Attacker(causal_model, tokenizer).run(ids, FAST)
    assert isinstance(adv_x, list) and len(adv_x) == len(ids)
    assert logs["perturbation"]["input_mode"] == "tokens"
    assert logs["cost"]["benign_output_tokens"] > 0


def test_token_input_respects_the_same_budget_semantics(seq2seq_model, tokenizer):
    """Budget accounting must not depend on which door the input came through."""
    ids = _ids(tokenizer, "hello world")
    cfg = dict(FAST, perturbation_budget=1, max_iterations=5)
    adv_x, logs = Attacker(seq2seq_model, tokenizer).run(ids, cfg)

    p = logs["perturbation"]
    # Hamming is measured directly on ids -- no decode is involved at all.
    assert p["hamming_distance"] == sum(a != b for a, b in zip(ids, adv_x))
    assert p["hamming_distance"] <= p["positions_touched"] <= p["budget"] == 1


def test_token_input_reports_round_trip_as_not_applicable(seq2seq_model, tokenizer):
    """No realisation step happens, so claiming `true` would overstate it.

    With text input the flag records a checked fact: the returned string
    re-tokenises to the optimised ids. With token input nothing is decoded or
    re-encoded, so the honest value is null rather than a vacuous true.
    """
    ids = _ids(tokenizer, "hello world")
    _, logs = Attacker(seq2seq_model, tokenizer).run(ids, FAST)
    assert logs["perturbation"]["round_trip_exact"] is None

    _, text_logs = Attacker(seq2seq_model, tokenizer).run("hello world", FAST)
    assert text_logs["perturbation"]["round_trip_exact"] is True
    assert text_logs["perturbation"]["input_mode"] == "text"


def test_interior_special_tokens_are_attackable_as_ids(seq2seq_model):
    """The realisation boundary is a property of *text* input, not of the attack.

    The same id sequence that must be refused as text -- because decoding drops
    its interior special tokens -- can be attacked directly as ids, since nothing
    is decoded. This is not a workaround for chat templates: template
    scaffolding also contains ordinary tokens that stay perturbable, and only a
    prefix can be protected.
    """
    tokenizer = _InteriorSpecialTokenizer()
    with pytest.raises(RuntimeError, match="re-tokenise"):
        Attacker(seq2seq_model, tokenizer).run("hello#world", FAST)

    ids = tokenizer("hello#world")["input_ids"][0].tolist()
    adv_x, logs = Attacker(seq2seq_model, tokenizer).run(ids, FAST)
    assert len(adv_x) == len(ids)
    assert logs["perturbation"]["input_mode"] == "tokens"


@pytest.mark.parametrize(
    "bad,exc,match",
    [
        ([1.0, 2.0], TypeError, "plain ints"),
        ([-1, 5], ValueError, "out of range"),
        ([10**9, 5], ValueError, "out of range"),
        ([], ValueError, "empty"),
        (["a", "b"], TypeError, "Batched text"),
        ({"a": 1}, TypeError, "must be str or a sequence"),
    ],
)
def test_malformed_input_gets_an_actionable_error(seq2seq_model, tokenizer, bad, exc, match):
    """Every rejection names what was wrong and what is accepted.

    Before this, a batched list of strings surfaced as a shape mismatch thrown
    from inside a matmul, which tells the caller nothing.
    """
    with pytest.raises(exc, match=match):
        Attacker(seq2seq_model, tokenizer).run(bad, FAST)


def test_batched_tensor_input_is_refused_clearly(seq2seq_model, tokenizer):
    ids = _ids(tokenizer, "hello world")
    with pytest.raises(ValueError, match="Batched input is not supported"):
        Attacker(seq2seq_model, tokenizer).run(torch.tensor([ids, ids]), FAST)


def test_float_tensor_input_is_refused_clearly(seq2seq_model, tokenizer):
    ids = _ids(tokenizer, "hello world")
    with pytest.raises(TypeError, match="integer tensor"):
        Attacker(seq2seq_model, tokenizer).run(torch.tensor(ids).float(), FAST)
