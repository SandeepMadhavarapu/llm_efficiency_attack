"""The white-box efficiency attack.

The optimisation problem
------------------------
We want an input that is nearly identical to a benign one but forces the model
to run many more decoding steps. Inference cost is dominated by the number of
steps, and decoding stops when the model emits EOS, so the objective is: make
the model unwilling to stop.

Why this is not ordinary gradient descent
-----------------------------------------
The thing we are optimising is a sequence of discrete token ids. Gradients live
in continuous embedding space, so we cannot simply step the input. The loop below
is HotFlip:

1. Embed the current input and take the gradient of the objective with respect
   to those embeddings.
2. First-order-estimate the objective change for replacing any position with any
   vocabulary token: `(e_v - e_i) . grad_i`. That is one matmul against the whole
   embedding table, so the entire vocabulary is scored at once.
3. Take the top-k candidates by that estimate and evaluate them *exactly* with
   real forward passes. The linear approximation is trustworthy for ranking and
   not for magnitude, so nothing is committed on its word alone.
4. Commit the single best substitution and repeat until the perturbation budget
   is spent.

Measurement honesty
-------------------
Once the attack works, every generation runs into `max_new_tokens` and the cost
metric saturates: output length pins to the ceiling and can no longer separate a
good attack from a great one. Those observations are right-censored, and the logs
say so explicitly rather than reporting a ceiling-bound number as if it were a
measurement.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import torch

from .adapters import ModelAdapter
from .config import AttackConfig
from .metrics import measure_cost
from .objectives import get_objective

logger = logging.getLogger(__name__)


class Attacker:
    """White-box efficiency attacker for any Hugging Face sequence model."""

    def __init__(self, model: Any, tokenizer: Any = None) -> None:
        """Wrap a model under attack.

        Args:
            model: A Hugging Face causal LM or seq2seq model.
            tokenizer: The matching tokenizer. Optional so that the public API in
                the task spec (`Attacker(model)`) works verbatim; when omitted it
                is loaded from the model's own `_name_or_path`. Passing it
                explicitly is preferred and avoids a network round-trip.
        """
        self.model = model
        self.tokenizer = tokenizer if tokenizer is not None else self._infer_tokenizer(model)
        self.adapter = ModelAdapter.for_model(self.model, self.tokenizer)

    @staticmethod
    def _infer_tokenizer(model: Any) -> Any:
        from transformers import AutoTokenizer

        name = getattr(model.config, "_name_or_path", None)
        if not name:
            raise ValueError(
                "No tokenizer was given and the model does not record a "
                "`_name_or_path` to load one from. Pass `Attacker(model, tokenizer)`."
            )
        return AutoTokenizer.from_pretrained(name)

    # ------------------------------------------------------------------ public

    def run(self, x: str, config: dict[str, Any] | None = None) -> tuple[str, dict]:
        """Craft an adversarial variant of `x`.

        Args:
            x: The benign input text.
            config: JSON-serialisable attack configuration. See `AttackConfig`.

        Returns:
            `(adv_x, logs)` where `adv_x` is the adversarial text and `logs` is a
            structured record of the run: per-iteration objective values and cost,
            the benign-vs-adversarial comparison, the attack's own compute cost,
            and censoring status of the cost metric.
        """
        cfg = AttackConfig.from_dict(config)
        self._seed_everything(cfg.seed)

        if cfg.verbose:
            logging.basicConfig(level=logging.INFO)

        device = self._resolve_device(cfg.device)
        self.model.to(device)
        self.model.eval()

        encoded = self.tokenizer(x, return_tensors="pt").to(device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get(
            "attention_mask", torch.ones_like(input_ids)
        )
        original_ids = input_ids.clone()

        eos_ids = self.adapter.eos_token_ids()
        objective_fn = get_objective(cfg.objective)

        # Instrumentation is counted separately from attack compute. Measuring the
        # cost every iteration is a reporting choice, not part of the attack, and
        # folding it into the attack's cost would overstate what the attack spends.
        budget = _ComputeBudget()

        benign = measure_cost(
            self.model, self.tokenizer, x, max_new_tokens=cfg.max_new_tokens
        )

        eligible = self._eligible_positions(input_ids, cfg)
        if not eligible:
            raise ValueError(
                "No perturbable token positions. Either the input is too short or "
                f"protected_prefix_tokens ({cfg.protected_prefix_tokens}) covers all "
                "of it."
            )

        changed: set[int] = set()
        iterations: list[dict] = []
        current_score = self._score_exact(
            input_ids, attention_mask, objective_fn, eos_ids, cfg, budget
        )

        for step in range(cfg.max_iterations):
            if len(changed) >= cfg.perturbation_budget:
                break

            t0 = time.perf_counter()

            allowed = self._allowed_positions(eligible, changed, cfg)
            if not allowed:
                break

            if cfg.strategy == "gradient":
                candidates = self._gradient_candidates(
                    input_ids, attention_mask, objective_fn, eos_ids, cfg, allowed, budget
                )
            else:
                candidates = self._random_candidates(allowed, cfg)

            best = self._pick_best_candidate(
                input_ids, attention_mask, objective_fn, eos_ids, cfg, candidates, budget
            )

            if best is None or best["score"] >= current_score:
                # No candidate improved the objective. Continuing would burn budget
                # without progress, so stop and report convergence honestly.
                logger.info("step %d: no improving substitution, stopping", step)
                break

            position, token_id = best["position"], best["token_id"]
            input_ids[0, position] = token_id
            changed.add(position)
            current_score = best["score"]

            text_now = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            cost_now = measure_cost(
                self.model, self.tokenizer, text_now, max_new_tokens=cfg.max_new_tokens
            )

            iterations.append(
                {
                    "iteration": step,
                    "objective": float(current_score),
                    "position": int(position),
                    "token_id": int(token_id),
                    "tokens_changed": len(changed),
                    "output_tokens": cost_now["output_tokens"],
                    "stopped_by": cost_now["stopped_by"],
                    "elapsed_s": time.perf_counter() - t0,
                }
            )
            logger.info(
                "step %d: objective %.4f, output_tokens %d (%s)",
                step,
                current_score,
                cost_now["output_tokens"],
                cost_now["stopped_by"],
            )

        adv_x = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        adversarial = measure_cost(
            self.model, self.tokenizer, adv_x, max_new_tokens=cfg.max_new_tokens
        )

        logs = self._build_logs(
            cfg=cfg,
            benign=benign,
            adversarial=adversarial,
            iterations=iterations,
            original_ids=original_ids,
            final_ids=input_ids,
            changed=changed,
            budget=budget,
        )
        return adv_x, logs

    # ------------------------------------------------------------- candidates

    def _gradient_candidates(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        objective_fn,
        eos_ids: list[int],
        cfg: AttackConfig,
        allowed: list[int],
        budget: "_ComputeBudget",
    ) -> list[tuple[int, int]]:
        """Rank every (position, token) substitution by first-order estimate.

        Returns the top-k as `(position, token_id)` pairs.
        """
        embeds = self.adapter.embed(input_ids)
        stop_logits = self.adapter.stop_logits(embeds, attention_mask, cfg.objective_horizon)
        loss = objective_fn(stop_logits, eos_ids)

        self.model.zero_grad(set_to_none=True)
        loss.backward()
        budget.forward += 1
        budget.backward += 1

        grad = embeds.grad[0]                      # (seq_len, hidden)
        table = self.adapter.embedding_matrix()    # (vocab, hidden)
        current = table[input_ids[0]]              # (seq_len, hidden)

        # First-order estimate of the objective change for every substitution:
        #   delta(i, v) ~ (e_v - e_i) . grad_i
        # One matmul scores the entire vocabulary at every position.
        scores = grad @ table.T                          # (seq_len, vocab)
        scores = scores - (grad * current).sum(-1, keepdim=True)

        # The loop minimises, so the most promising substitutions are the most
        # negative. Mask out everything we are not allowed to touch by setting it
        # to +inf so it can never be selected.
        mask = torch.full_like(scores, float("inf"))
        allowed_idx = torch.tensor(allowed, device=scores.device, dtype=torch.long)
        mask[allowed_idx] = 0.0
        scores = scores + mask

        # Never propose the token already in place: it is a no-op that would
        # consume an iteration.
        scores[torch.arange(scores.shape[0]), input_ids[0]] = float("inf")

        flat = scores.flatten()
        k = min(cfg.top_k, int((flat != float("inf")).sum().item()))
        if k < 1:
            return []
        _, flat_idx = torch.topk(-flat, k)
        vocab_size = scores.shape[1]
        return [
            (int(i.item() // vocab_size), int(i.item() % vocab_size)) for i in flat_idx
        ]

    def _random_candidates(
        self, allowed: list[int], cfg: AttackConfig
    ) -> list[tuple[int, int]]:
        """Sample `top_k` substitutions uniformly at random.

        This is the experimental control. It receives the same perturbation budget
        AND the same number of exact evaluations as the gradient strategy, so the
        only difference between the two runs is whether the gradient was used to
        choose where to look. If gradient-guided search does not clearly beat this,
        the white-box signal is not earning its cost -- which is a result worth
        knowing rather than an outcome to avoid measuring.
        """
        vocab_size = self.adapter.embedding_matrix().shape[0]
        return [
            (random.choice(allowed), random.randrange(vocab_size))
            for _ in range(cfg.top_k)
        ]

    def _pick_best_candidate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        objective_fn,
        eos_ids: list[int],
        cfg: AttackConfig,
        candidates: list[tuple[int, int]],
        budget: "_ComputeBudget",
    ) -> dict | None:
        """Exactly evaluate each shortlisted candidate and return the best.

        The first-order estimate ranks well but its magnitudes are unreliable, so
        every candidate that survives ranking is re-scored with a real forward
        pass before anything is committed.
        """
        best: dict | None = None
        for position, token_id in candidates:
            trial = input_ids.clone()
            trial[0, position] = token_id
            score = self._score_exact(
                trial, attention_mask, objective_fn, eos_ids, cfg, budget
            )
            if best is None or score < best["score"]:
                best = {"position": position, "token_id": token_id, "score": score}
        return best

    def _score_exact(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        objective_fn,
        eos_ids: list[int],
        cfg: AttackConfig,
        budget: "_ComputeBudget",
    ) -> float:
        """Objective value for a concrete token sequence, no gradient."""
        with torch.no_grad():
            embeds = self.model.get_input_embeddings()(input_ids)
            stop_logits = self.adapter.stop_logits(
                embeds, attention_mask, cfg.objective_horizon
            )
            value = objective_fn(stop_logits, eos_ids)
        budget.forward += 1
        return float(value.item())

    # ---------------------------------------------------------------- helpers

    def _eligible_positions(self, input_ids: torch.Tensor, cfg: AttackConfig) -> list[int]:
        """Positions the attack is permitted to modify.

        Two exclusions. The protected prefix keeps instruction-tuned prompts
        intact -- perturbing `"translate English to German:"` would break the task
        rather than attack its efficiency, and would measure the wrong thing. And
        special tokens are left alone because rewriting the input's own EOS is not
        an imperceptible edit to the text.
        """
        special = set(self.tokenizer.all_special_ids or [])
        return [
            i
            for i in range(cfg.protected_prefix_tokens, input_ids.shape[1])
            if int(input_ids[0, i].item()) not in special
        ]

    @staticmethod
    def _allowed_positions(
        eligible: list[int], changed: set[int], cfg: AttackConfig
    ) -> list[int]:
        """Eligible positions, respecting the remaining budget.

        Once the budget is fully committed the attack may still refine positions it
        has already changed -- that costs no additional perturbation -- but may not
        open a new one.
        """
        if len(changed) < cfg.perturbation_budget:
            return eligible
        return [i for i in eligible if i in changed]

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """Make a run reproducible.

        Seeds Python's RNG (used by the random control), torch's CPU RNG, and CUDA
        if present. Combined with greedy decoding in the cost metric, this is what
        makes "same config + input -> same result" hold.
        """
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _build_logs(
        self,
        cfg: AttackConfig,
        benign: dict,
        adversarial: dict,
        iterations: list[dict],
        original_ids: torch.Tensor,
        final_ids: torch.Tensor,
        changed: set[int],
        budget: "_ComputeBudget",
    ) -> dict:
        """Assemble the structured run record.

        The `censored` block is the part that matters most. A cost metric capped at
        `max_new_tokens` produces right-censored observations: when generation stops
        because it hit the ceiling, the true cost is *at least* that value and
        possibly much more. Reporting the ratio without that caveat would state a
        lower bound as though it were a measurement.
        """
        benign_ceiling = benign["stopped_by"] == "max_tokens"
        adv_ceiling = adversarial["stopped_by"] == "max_tokens"

        ratio = (
            adversarial["output_tokens"] / benign["output_tokens"]
            if benign["output_tokens"] > 0
            else float("inf")
        )

        return {
            "config": cfg.to_dict(),
            "benign": benign,
            "adversarial": adversarial,
            "cost": {
                "benign_output_tokens": benign["output_tokens"],
                "adversarial_output_tokens": adversarial["output_tokens"],
                "output_token_ratio": ratio,
                "wall_time_ratio": (
                    adversarial["wall_time_s"] / benign["wall_time_s"]
                    if benign["wall_time_s"] > 0
                    else float("inf")
                ),
            },
            "censored": {
                "max_new_tokens": cfg.max_new_tokens,
                "benign_hit_ceiling": benign_ceiling,
                "adversarial_hit_ceiling": adv_ceiling,
                "ratio_is_lower_bound": adv_ceiling,
                "note": (
                    "Adversarial generation stopped at the max_new_tokens ceiling, "
                    "so its cost is right-censored: the reported ratio is a lower "
                    "bound on the true efficiency damage, not a point estimate. "
                    "Raise max_new_tokens to tighten it."
                    if adv_ceiling
                    else "Adversarial generation terminated on EOS, so the reported "
                    "cost is an actual measurement rather than a censored one."
                ),
            },
            "perturbation": {
                "budget": cfg.perturbation_budget,
                "tokens_changed": len(changed),
                "positions_changed": sorted(changed),
                "original_token_ids": original_ids[0].tolist(),
                "adversarial_token_ids": final_ids[0].tolist(),
            },
            "attack_cost": {
                "forward_passes": budget.forward,
                "backward_passes": budget.backward,
                "note": (
                    "Compute spent finding the perturbation. Compare against the "
                    "extra decoding steps it induces to judge whether the attack is "
                    "economical for the attacker. Cost-metric measurements are "
                    "instrumentation and are excluded from these counts."
                ),
            },
            "iterations": iterations,
        }


class _ComputeBudget:
    """Tally of forward and backward passes spent by the attack itself."""

    def __init__(self) -> None:
        self.forward = 0
        self.backward = 0
