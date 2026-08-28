"""The white-box efficiency attack.

The optimisation problem
------------------------
We want an input that differs from a benign one in at most a fixed number of
token positions but forces the model to run many more decoding steps. Inference
cost is dominated by the number of steps, and decoding stops when the model emits
EOS, so the objective is: make the model unwilling to stop.

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
   real objective evaluations. The linear approximation is trustworthy for
   ranking and not for magnitude, so nothing is committed on its word alone.
4. Commit the single best substitution and repeat until the perturbation budget
   is spent.

What the attack optimises is what the caller receives
-----------------------------------------------------
The attack works on token ids; the caller receives text. Those can come apart:
`tokenizer.encode(tokenizer.decode(ids))` is not an identity for arbitrary id
sequences, because BPE and SentencePiece merges are context-dependent, special
tokens are stripped on decode, and embedding tables are often padded past the end
of the vocabulary. A substitution can therefore re-segment its neighbours,
changing token positions the attack never touched and quietly breaking the
advertised perturbation budget in the text the caller actually feeds the model.

Two mechanisms keep that from happening, and they are belt and braces on purpose:
illegal ids are excluded from the candidate set before scoring, and any candidate
whose committed sequence would not survive a decode/re-encode round trip is
rejected during the search. `logs["perturbation"]["round_trip_exact"]` records
the outcome rather than asserting it silently.

Measurement honesty
-------------------
Once the attack works, generation can run into `max_new_tokens` and the cost
metric saturates. Those observations are right-censored, and
`metrics.interpret_censoring` states exactly what the observed ratio does and
does not establish rather than reporting a ceiling-bound number as a measurement.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Sequence

import torch

from .adapters import ModelAdapter, forbidden_token_ids
from .config import AttackConfig
from .metrics import (
    ForwardCounter,
    interpret_censoring,
    measure_cost_from_ids,
    model_device,
)
from .objectives import get_objective

logger = logging.getLogger(__name__)

# Above this many candidates per iteration the exact-rescoring stage dominates
# wall time badly enough to be worth a warning: each candidate costs a full
# objective evaluation, which is itself several model forwards.
_TOP_K_WARN_THRESHOLD = 200


class Attacker:
    """White-box efficiency attacker for a Hugging Face sequence model."""

    def __init__(self, model: Any, tokenizer: Any = None) -> None:
        """Wrap a model under attack.

        Args:
            model: A Hugging Face causal LM or seq2seq model, on a single device.
            tokenizer: The matching tokenizer. Optional so that the public API in
                the task spec (`Attacker(model)`) works verbatim; when omitted it
                is loaded from the model's own `_name_or_path`. Passing it
                explicitly is preferred and avoids a network round-trip.
        """
        self.model = model
        self.tokenizer = tokenizer if tokenizer is not None else self._infer_tokenizer(model)
        self.adapter = ModelAdapter.for_model(self.model, self.tokenizer)

    def _coerce_input(
        self, x: Any
    ) -> "tuple[str | None, list[int] | None]":
        """Resolve `x` into exactly one of the two supported representations.

        Returns `(text, None)` or `(None, ids)`. Rejecting bad input here, with a
        message naming the accepted forms, is the difference between an
        actionable error and a shape mismatch thrown from inside a matmul --
        which is what a batched list of strings used to produce.
        """
        if isinstance(x, str):
            return x, None

        if torch.is_tensor(x):
            if x.dtype.is_floating_point or x.dtype.is_complex:
                raise TypeError(
                    "Token ids must be an integer tensor, got dtype "
                    f"{x.dtype}. Token ids index an embedding table; a float "
                    "tensor is not a valid index."
                )
            if x.dim() == 2:
                if x.shape[0] != 1:
                    raise ValueError(
                        f"Batched input is not supported: got shape {tuple(x.shape)}. "
                        "Attack one example at a time."
                    )
                x = x[0]
            elif x.dim() != 1:
                raise ValueError(
                    f"Token ids must be 1-D or (1, n), got shape {tuple(x.shape)}."
                )
            ids = [int(i) for i in x.tolist()]
        elif isinstance(x, (list, tuple)):
            if x and all(isinstance(i, str) for i in x):
                raise TypeError(
                    "Batched text is not supported: pass a single string, or "
                    "token ids for one example. Attack one example at a time."
                )
            if not all(isinstance(i, int) and not isinstance(i, bool) for i in x):
                raise TypeError(
                    "Token ids must all be plain ints. Got element types "
                    f"{sorted({type(i).__name__ for i in x})}."
                )
            ids = list(x)
        else:
            raise TypeError(
                f"x must be str or a sequence of token ids, got {type(x).__name__}."
            )

        if not ids:
            raise ValueError("Token id sequence is empty; nothing to attack.")

        rows = self.adapter.embedding_matrix().shape[0]
        out_of_range = [i for i in ids if i < 0 or i >= rows]
        if out_of_range:
            raise ValueError(
                f"Token ids out of range for this model's embedding table "
                f"(0..{rows - 1}): {sorted(set(out_of_range))[:8]}."
            )
        return None, ids

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

    def run(
        self, x: "str | Sequence[int] | torch.Tensor", config: dict[str, Any] | None = None
    ) -> tuple[Any, dict]:
        """Craft an adversarial variant of `x`.

        Args:
            x: The benign input, in either representation:

                * **text** (`str`) -- tokenised here, and `adv_x` is returned as
                  text whose re-tokenisation is guaranteed to equal the ids that
                  were optimised.
                * **token ids** (`list[int]`, `tuple[int]`, or a 1-D or
                  `(1, n)` integer tensor) -- used directly, and `adv_x` is
                  returned in the same representation, as a `list[int]`.

                `adv_x` always mirrors the representation of `x`, so the returned
                value can be fed straight back to the model the same way `x` was.

            config: JSON-serialisable attack configuration. See `AttackConfig`.

        Returns:
            `(adv_x, logs)` -- `logs` records per-iteration objective values and
            cost, the benign-vs-adversarial comparison, instrumented attack
            compute, and what the censoring state permits the cost ratio to mean.
            `logs["perturbation"]["input_mode"]` says which representation was
            used.

        Note on the two representations. Text input has to survive a
        decode/re-encode step, which is not an identity for arbitrary id
        sequences, so the attack rejects any candidate that would break it and
        refuses inputs it cannot realise exactly (interior special tokens, for
        instance). Token input performs no such step -- the object optimised *is*
        the object returned -- so that constraint does not arise and
        `round_trip_exact` is reported as `null` rather than `true`, because
        nothing was round-tripped.

        This does **not** make chat-templated inputs attackable in a meaningful
        sense. Template scaffolding contains ordinary, non-special tokens (an
        `assistant` header, newlines) that remain perturbable, and this toolbox
        protects only a prefix. Editing scaffolding is task damage, not an
        efficiency attack. See RESULTS.md.
        """
        cfg = AttackConfig.from_dict(config)
        text_input, token_ids = self._coerce_input(x)
        self._seed_everything(cfg.seed)

        if cfg.verbose:
            logging.basicConfig(level=logging.INFO)
        if cfg.top_k > _TOP_K_WARN_THRESHOLD:
            logger.warning(
                "top_k=%d: every candidate costs one exact objective evaluation, "
                "which is several model forward passes. Expect roughly %dx the "
                "runtime of the default top_k=20.",
                cfg.top_k,
                cfg.top_k // 20,
            )

        device = self._resolve_device(cfg.device)
        was_training = self.model.training
        self._move_model(device)
        self.model.eval()
        try:
            return self._run_on_device(text_input, token_ids, cfg, device)
        finally:
            # `run()` borrows the caller's model; it should not silently leave it
            # in a different mode than it found it.
            self.model.train(was_training)

    def _run_on_device(
        self,
        text: "str | None",
        token_ids: "list[int] | None",
        cfg: AttackConfig,
        device: torch.device,
    ):
        realize = text is not None
        if realize:
            encoded = self.tokenizer(text, return_tensors="pt").to(device)
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
        else:
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
        original_ids = input_ids.clone()

        eos_ids = self.adapter.eos_token_ids()
        objective_fn = get_objective(cfg.objective)
        # Trajectory shape is a property of the objective, not of the loop. See
        # `objectives.register`.
        force_full = bool(getattr(objective_fn, "force_full_horizon", False))
        illegal = forbidden_token_ids(
            self.tokenizer, self.adapter.embedding_matrix().shape[0]
        )
        legal_ids = [
            i
            for i in range(self.adapter.embedding_matrix().shape[0])
            if i not in illegal
        ]
        if not legal_ids:
            raise ValueError(
                "No legal substitution tokens: every embedding row is either a "
                "special token or outside the tokenizer's vocabulary."
            )

        counters = _ComputeCounters()

        # One hook for the whole run counts every real model invocation,
        # including each decoding step inside `generate()`. Measurement calls are
        # bracketed so instrumentation can be subtracted from search compute.
        with ForwardCounter(self.model) as forwards:
            # Before anything is optimised, confirm that the differentiable input
            # path the attack uses actually reproduces the model's normal forward
            # pass. If it does not, every gradient below would describe a function
            # the model does not compute. Two forwards, once per run, attributed
            # to diagnostics so they inflate neither search nor measurement cost.
            embedding_deviation = self.adapter.check_embedding_equivalence(
                input_ids, attention_mask
            )
            counters.diagnostic_forwards = forwards.count

            benign = self._measure(input_ids, cfg, forwards, counters)

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
                input_ids, attention_mask, objective_fn, eos_ids, cfg,
                force_full, counters
            )
            iterations.append(
                {
                    "iteration": -1,
                    "objective": current_score,
                    "position": None,
                    "token_id": None,
                    "tokens_changed": 0,
                    "output_tokens": benign["output_tokens"],
                    "stopped_by": benign["stopped_by"],
                    "elapsed_s": 0.0,
                    "note": "benign starting point, before any substitution",
                }
            )

            for step in range(cfg.max_iterations):
                if len(changed) >= cfg.perturbation_budget:
                    break

                t0 = time.perf_counter()
                allowed = self._allowed_positions(eligible, changed, cfg)

                if cfg.strategy in ("gradient", "gradient_stratified"):
                    candidates = self._gradient_candidates(
                        input_ids, attention_mask, objective_fn, eos_ids, cfg,
                        allowed, illegal, force_full, counters,
                        stratified=cfg.strategy == "gradient_stratified",
                    )
                else:
                    candidates = self._random_candidates(
                        input_ids, allowed, legal_ids, cfg
                    )

                best = self._pick_best_candidate(
                    input_ids, attention_mask, objective_fn, eos_ids, cfg,
                    candidates, force_full, realize, counters,
                )

                if best is None or best["score"] >= current_score:
                    # No admissible candidate improved the objective. Continuing
                    # would burn budget without progress, so stop and report
                    # convergence honestly.
                    logger.info("step %d: no improving substitution, stopping", step)
                    break

                position, token_id = best["position"], best["token_id"]
                input_ids[0, position] = token_id
                changed.add(position)
                current_score = best["score"]

                cost_now = self._measure(input_ids, cfg, forwards, counters)

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

            adversarial = self._measure(input_ids, cfg, forwards, counters)
            counters.total_forwards = forwards.count

        if not realize:
            # Token in, token out. There is no text realisation step, so nothing
            # can be lost in one; `round_trip_exact` is reported as null rather
            # than true, because no round trip happened.
            adv_x: Any = input_ids[0].tolist()
            round_trip_exact = None
            return adv_x, self._build_logs(
                cfg=cfg, benign=benign, adversarial=adversarial,
                iterations=iterations, original_ids=original_ids,
                final_ids=input_ids, changed=changed, counters=counters,
                round_trip_exact=round_trip_exact,
                embedding_deviation=embedding_deviation,
                force_full_horizon=force_full, input_mode="tokens",
            )

        adv_x = self._decode(input_ids)
        realised_ids = self._realise(adv_x, input_ids.device)
        round_trip_exact = self._ids_equal(realised_ids, input_ids)
        if not round_trip_exact:
            # Unreachable via the search, which rejects any candidate that would
            # not round-trip. Kept as a hard guard because the alternative is
            # returning text whose perturbation is larger than the budget the
            # caller was promised, with logs that say otherwise.
            raise RuntimeError(
                "Internal invariant violated: the returned adversarial text does "
                "not re-tokenise to the optimised token ids, so the perturbation "
                "budget does not apply to it. optimised="
                f"{input_ids[0].tolist()} realised={realised_ids[0].tolist()}"
            )

        logs = self._build_logs(
            cfg=cfg,
            benign=benign,
            adversarial=adversarial,
            iterations=iterations,
            original_ids=original_ids,
            final_ids=input_ids,
            changed=changed,
            counters=counters,
            round_trip_exact=round_trip_exact,
            embedding_deviation=embedding_deviation,
            force_full_horizon=force_full,
            input_mode="text",
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
        illegal: set[int],
        force_full: bool,
        counters: "_ComputeCounters",
        stratified: bool = False,
    ) -> list[tuple[int, int]]:
        """Rank every legal (position, token) substitution by first-order estimate.

        Returns `top_k` `(position, token_id)` pairs, so both gradient variants
        and the random control spend the same number of exact evaluations.

        Two ways of turning the score matrix into a shortlist:

        * **global** (`strategy="gradient"`): `topk` over the flattened
          `(positions x vocab)` matrix. This is the textbook HotFlip shortlist and
          the original implementation.
        * **stratified** (`strategy="gradient_stratified"`): round-robin over
          positions, taking each position's best remaining candidate in turn.

        The distinction is not cosmetic. Measured on t5-small, the global
        shortlist draws 90-100% of its candidates from a *single* position even at
        `top_k=100`, because one position dominates the gradient magnitude. Once
        that position has been edited the shortlist re-concentrates on it, nothing
        improves, and the search halts with most of the perturbation budget
        unspent -- mean Hamming 1.43 against a budget of 3, while the random
        control reaches 2.57. A falsifier check confirmed the objective has *not*
        saturated at that point: 39 improving substitutions were found at
        non-edited positions across the development inputs.
        """
        embeds = self.adapter.embed(input_ids)
        stop_logits = self.adapter.stop_logits(
            embeds, attention_mask, cfg.objective_horizon, force_full
        )
        counters.record_trajectory(stop_logits.shape[0])
        loss = objective_fn(stop_logits, eos_ids)

        self.model.zero_grad(set_to_none=True)
        loss.backward()
        counters.objective_evaluations += 1
        counters.gradient_evaluations += 1

        grad = embeds.grad[0]                      # (seq_len, hidden)
        table = self.adapter.embedding_matrix()    # (vocab, hidden)
        scale = self.adapter.embedding_scale()

        # First-order estimate of the objective change for every substitution:
        #   delta(i, v) ~ (e_v - e_i) . grad_i
        # where e_v is the vector actually fed to the model, i.e. the scaled
        # embedding. One matmul scores the entire vocabulary at every position.
        # (A positive scale cannot change the ranking, only the magnitudes; it is
        # applied anyway so the computed quantity is the one the docs describe.)
        scores = (grad @ table.T) * scale
        current = table[input_ids[0]] * scale
        scores = scores - (grad * current).sum(-1, keepdim=True)

        # The loop minimises, so the most promising substitutions are the most
        # negative. Everything inadmissible is set to +inf so it can never be
        # selected: positions outside the allowed set, ids that would not survive
        # the decode/re-encode round trip, and the token already in place.
        mask = torch.full_like(scores, float("inf"))
        allowed_idx = torch.tensor(allowed, device=scores.device, dtype=torch.long)
        mask[allowed_idx] = 0.0
        scores = scores + mask

        if illegal:
            illegal_idx = torch.tensor(
                sorted(illegal), device=scores.device, dtype=torch.long
            )
            scores[:, illegal_idx] = float("inf")

        scores[torch.arange(scores.shape[0]), input_ids[0]] = float("inf")

        vocab_size = scores.shape[1]
        budget = min(cfg.top_k, int(torch.isfinite(scores).sum().item()))
        if budget < 1:
            return []

        if not stratified:
            flat = scores.flatten()
            _, flat_idx = torch.topk(-flat, budget)
            return [
                (int(i.item() // vocab_size), int(i.item() % vocab_size))
                for i in flat_idx
            ]

        # Round-robin: each allowed position offers its best remaining candidate
        # before any position offers its second. Guarantees every position is
        # represented while still preferring high-scoring tokens within a
        # position.
        per_position = min(budget, vocab_size)
        ranked: dict[int, list[int]] = {}
        for position in allowed:
            row = scores[position]
            finite = int(torch.isfinite(row).sum().item())
            if finite < 1:
                continue
            take = min(per_position, finite)
            _, idx = torch.topk(-row, take)
            ranked[position] = [int(i.item()) for i in idx]

        picked: list[tuple[int, int]] = []
        depth = 0
        while len(picked) < budget and ranked:
            progressed = False
            for position in sorted(ranked):
                if depth < len(ranked[position]):
                    picked.append((position, ranked[position][depth]))
                    progressed = True
                    if len(picked) == budget:
                        break
            if not progressed:
                break
            depth += 1
        return picked

    def _random_candidates(
        self,
        input_ids: torch.Tensor,
        allowed: list[int],
        legal_ids: list[int],
        cfg: AttackConfig,
    ) -> list[tuple[int, int]]:
        """Sample `top_k` substitutions uniformly at random from the same space.

        This is the experimental control. To make the comparison a test of the
        gradient rather than of bookkeeping, it draws from exactly the candidate
        space the gradient strategy is allowed to draw from: the same positions,
        the same legal token ids, and -- like the gradient path -- never the token
        already in place, which would be a no-op that wastes an evaluation. Both
        strategies then spend the same number of exact objective evaluations.

        If gradient-guided search does not clearly beat this, the white-box signal
        is not earning its cost -- which is a result worth knowing rather than an
        outcome to avoid measuring.
        """
        candidates: list[tuple[int, int]] = []
        for _ in range(cfg.top_k):
            position = random.choice(allowed)
            token_id = random.choice(legal_ids)
            # Resample a no-op rather than dropping the candidate, so the control
            # really does get `top_k` usable candidates per iteration.
            while token_id == int(input_ids[0, position].item()):
                token_id = random.choice(legal_ids)
            candidates.append((position, token_id))
        return candidates

    def _pick_best_candidate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        objective_fn,
        eos_ids: list[int],
        cfg: AttackConfig,
        candidates: list[tuple[int, int]],
        force_full: bool,
        realize: bool,
        counters: "_ComputeCounters",
    ) -> dict | None:
        """Exactly evaluate each admissible candidate and return the best.

        Two filters, in cost order. First the round-trip check, which is a decode
        plus an encode and is far cheaper than a forward pass: a candidate whose
        committed sequence would not re-tokenise to itself is discarded, because
        committing it would mean the returned text is not the thing that was
        optimised. Then exact rescoring, because the first-order estimate ranks
        well but its magnitudes are unreliable.
        """
        best: dict | None = None
        for position, token_id in candidates:
            trial = input_ids.clone()
            trial[0, position] = token_id

            if realize and not self._round_trips(trial):
                counters.rejected_non_round_trip += 1
                continue

            score = self._score_exact(
                trial, attention_mask, objective_fn, eos_ids, cfg, force_full, counters
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
        force_full: bool,
        counters: "_ComputeCounters",
    ) -> float:
        """Objective value for a concrete token sequence, no gradient."""
        with torch.no_grad():
            embeds = self.adapter.embed_values(input_ids)
            stop_logits = self.adapter.stop_logits(
                embeds, attention_mask, cfg.objective_horizon, force_full
            )
            counters.record_trajectory(stop_logits.shape[0])
            value = objective_fn(stop_logits, eos_ids)
        counters.objective_evaluations += 1
        return float(value.item())

    # ------------------------------------------------------- token realisation

    def _decode(self, input_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def _realise(self, text: str, device: torch.device) -> torch.Tensor:
        """Token ids the model will actually see when fed `text`."""
        return self.tokenizer(text, return_tensors="pt")["input_ids"].to(device)

    @staticmethod
    def _ids_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
        return a.shape == b.shape and bool(torch.equal(a, b))

    def _round_trips(self, input_ids: torch.Tensor) -> bool:
        """Does this id sequence survive decode followed by re-encode?"""
        return self._ids_equal(
            self._realise(self._decode(input_ids), input_ids.device), input_ids
        )

    def _measure(
        self,
        input_ids: torch.Tensor,
        cfg: AttackConfig,
        forwards: ForwardCounter,
        counters: "_ComputeCounters",
    ) -> dict:
        """Measure cost, attributing the model forwards it spends to instrumentation.

        Measures from ids rather than from decoded text. In text mode the two are
        equivalent by construction -- the search only commits candidates whose
        text re-tokenises to exactly these ids -- and measuring from ids keeps a
        single measurement path for both input representations.
        """
        before = forwards.count
        result = measure_cost_from_ids(
            self.model, self.tokenizer, input_ids, max_new_tokens=cfg.max_new_tokens
        )
        counters.measurement_forwards += forwards.count - before
        return result

    # ---------------------------------------------------------------- helpers

    def _eligible_positions(self, input_ids: torch.Tensor, cfg: AttackConfig) -> list[int]:
        """Positions the attack is permitted to modify.

        Two exclusions. The protected prefix keeps instruction-tuned prompts
        intact -- perturbing `"translate English to German:"` would break the task
        rather than attack its efficiency, and would measure the wrong thing. And
        positions already holding a special token are left alone because
        rewriting the input's own EOS is not a token substitution, it is a change
        of structure.
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
        """Eligible positions, given the budget still has room.

        The caller checks `len(changed) < perturbation_budget` before every call,
        so there is no separate budget-exhausted branch here: when the budget is
        full the loop stops rather than continuing to refine.
        """
        return eligible

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

    def _move_model(self, device: torch.device) -> None:
        """Put the model on `device`, with a comprehensible error if it cannot move.

        Multi-device models -- `device_map="auto"`, Accelerate-sharded, offloaded
        -- refuse `.to()`. They are out of scope for this toolbox, and saying so
        here is better than letting an Accelerate internal error surface.
        """
        if model_device(self.model) == device:
            return
        try:
            self.model.to(device)
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            raise RuntimeError(
                f"Could not move the model to {device}: {exc}. This toolbox "
                "supports single-device models only; models loaded with "
                '`device_map="auto"`, Accelerate sharding, offloading, or '
                "quantisation are out of scope."
            ) from exc

    def _build_logs(
        self,
        cfg: AttackConfig,
        benign: dict,
        adversarial: dict,
        iterations: list[dict],
        original_ids: torch.Tensor,
        final_ids: torch.Tensor,
        changed: set[int],
        counters: "_ComputeCounters",
        round_trip_exact: "bool | None",
        embedding_deviation: float,
        force_full_horizon: bool,
        input_mode: str,
    ) -> dict:
        """Assemble the structured run record."""
        rows_seen = counters.trajectory_rows
        original = original_ids[0].tolist()
        adversarial_ids = final_ids[0].tolist()
        hamming = sum(a != b for a, b in zip(original, adversarial_ids))

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
                "note": (
                    "output_token_ratio is the primary metric: generated token "
                    "count under greedy decoding, which is deterministic and "
                    "hardware-independent. It is a proxy for inference cost, not "
                    "a latency measurement. wall_time_ratio is secondary and "
                    "noisy -- on short generations it is dominated by scheduling "
                    "jitter and can come out below 1.0 even when the attack "
                    "succeeded. Read `censored.interpretation` before quoting "
                    "either number."
                ),
            },
            "censored": interpret_censoring(benign, adversarial, cfg.max_new_tokens),
            "perturbation": {
                "budget": cfg.perturbation_budget,
                "positions_touched": len(changed),
                "positions_changed": sorted(changed),
                "hamming_distance": hamming,
                "input_mode": input_mode,
                "round_trip_exact": round_trip_exact,
                "original_token_ids": original,
                "adversarial_token_ids": adversarial_ids,
                "note": (
                    "positions_touched counts positions the search wrote to; "
                    "hamming_distance counts positions whose final token actually "
                    "differs from the original. They are not the same number: a "
                    "position written twice counts once as touched, and a position "
                    "restored to its original token still counts as touched. The "
                    "budget bounds positions_touched. With text input, "
                    "round_trip_exact records that the returned text re-tokenises "
                    "to exactly these ids, so the bound applies to what the "
                    "caller feeds the model; with token input it is null, because "
                    "the ids are returned directly and no realisation step "
                    "occurs. This is a bound on token-level edit count only: no "
                    "semantic, fluency, or human-perceptibility constraint is "
                    "enforced."
                ),
            },
            "attack_cost": {
                "objective_evaluations": counters.objective_evaluations,
                "gradient_evaluations": counters.gradient_evaluations,
                "candidates_rejected_non_round_trip": counters.rejected_non_round_trip,
                "search_model_forwards": counters.total_forwards
                - counters.measurement_forwards
                - counters.diagnostic_forwards,
                "measurement_model_forwards": counters.measurement_forwards,
                "diagnostic_model_forwards": counters.diagnostic_forwards,
                "total_model_forwards": counters.total_forwards,
                "note": (
                    "Model forward counts are instrumented with a forward "
                    "pre-hook, not estimated: they include every decoding step "
                    "inside the `generate()` calls that each objective evaluation "
                    "performs, which is why search_model_forwards is roughly an "
                    "order of magnitude larger than objective_evaluations. "
                    "search_model_forwards is what the attack spends; "
                    "measurement_model_forwards is the benign/adversarial cost "
                    "metric, which is reporting instrumentation rather than part "
                    "of the attack; diagnostic_model_forwards is the one-off "
                    "embedding-equivalence check. The three sum to "
                    "total_model_forwards. Compare search_model_forwards against "
                    "the extra decoding steps induced to judge whether the attack "
                    "is economical for the attacker."
                ),
            },
            "diagnostics": {
                "objective_horizon": {
                    "requested": cfg.objective_horizon,
                    "force_full_horizon": force_full_horizon,
                    "realised_rows_min": min(rows_seen) if rows_seen else None,
                    "realised_rows_max": max(rows_seen) if rows_seen else None,
                    "evaluations": len(rows_seen),
                    "evaluations_at_full_horizon": sum(
                        r >= cfg.objective_horizon for r in rows_seen
                    ),
                    "note": (
                        "objective_horizon is an UPPER BOUND on the number of "
                        "stop-decision rows the objective sees, not a step "
                        "count: the trajectory is the model's own greedy output "
                        "and ends when it emits EOS. When realised_rows_min is "
                        "below requested, some candidates were scored by a mean "
                        "over fewer terms than others. The "
                        "'eos_suppression_fixed_horizon' objective forces every "
                        "evaluation to exactly `requested` rows; "
                        "examples/objective_diagnostic.py measures whether that "
                        "changes anything (on t5-small it does not)."
                    ),
                },
                "embedding_equivalence_max_logit_deviation": embedding_deviation,
                "note": (
                    "The attack differentiates through `inputs_embeds`. This is "
                    "the maximum absolute logit difference between that path and "
                    "the model's normal `input_ids` path on this input, checked "
                    "once per run. It should be at or near zero; a run that got "
                    "this far did not exceed the adapter's tolerance."
                ),
            },
            "iterations": iterations,
        }


class _ComputeCounters:
    """Tally of what the attack spent. See `logs["attack_cost"]` for semantics."""

    def __init__(self) -> None:
        self.objective_evaluations = 0
        self.gradient_evaluations = 0
        self.measurement_forwards = 0
        self.diagnostic_forwards = 0
        self.total_forwards = 0
        self.rejected_non_round_trip = 0
        self.trajectory_rows: list[int] = []

    def record_trajectory(self, rows: int) -> None:
        """Note how many stop-decision rows an objective evaluation actually got.

        `objective_horizon` is an upper bound: the trajectory is the model's own
        greedy output and ends early when it emits EOS. Recording the realised
        count makes that visible in the logs instead of leaving it as a caveat in
        the documentation.
        """
        self.trajectory_rows.append(int(rows))
