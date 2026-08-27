"""The model-agnostic seam.

Hugging Face already hides most architectural differences: `get_input_embeddings()`
returns the embedding table for any model, and `config.is_encoder_decoder` says
which family we are in. Only four things genuinely differ between a seq2seq and
a causal LM for this attack, and this module is the only place that knows them:

1. A seq2seq forward pass needs decoder inputs; a causal one does not.
2. Where the "will you stop now?" logits live.
3. How to count generated tokens out of `generate()`.
4. Which submodule declares the input-embedding scaling factor, if any.

Everything else in the package is written against `ModelAdapter` and therefore
never branches on model type. Supporting a new architecture means adding a
subclass here and nothing else.

Why point 4 exists
------------------
The attack needs a *differentiable* input, so it feeds the model `inputs_embeds`
rather than `input_ids`. Those two paths are not always equivalent. Several
encoder-decoder architectures (Marian, M2M100, BART with `scale_embedding=True`)
compute `embed_tokens(input_ids) * embed_scale` on the `input_ids` path and skip
the multiplication when the caller supplies `inputs_embeds`. For
`Helsinki-NLP/opus-mt-*` that factor is `sqrt(512) ~= 22.6` -- large enough that
an attack feeding raw table lookups would be optimising a function the model does
not compute, silently, and would look like a weak attack rather than a bug.
`embed_values()` applies the architecture's own factor, and
`check_embedding_equivalence()` verifies the result rather than trusting it.
"""

from __future__ import annotations

from typing import Any

import torch


def resolve_eos_ids(model: Any) -> list[int]:
    """Token ids that terminate generation for `model`.

    Two wrinkles this exists to absorb:

    * `eos_token_id` may be a single int or a list of ints. Several chat models
      define more than one stop token, and a bare `==` comparison silently
      misses all but the first.
    * `generation_config` is what `generate()` actually consults, and it can
      differ from `config`. We prefer it and fall back to `config`.

    Returns an empty list when the model declares no stop token at all, which is
    a real case (some base LMs) and is handled by callers rather than crashing
    here.
    """
    for source in (getattr(model, "generation_config", None), model.config):
        if source is None:
            continue
        eos = getattr(source, "eos_token_id", None)
        if eos is None:
            continue
        return [int(eos)] if isinstance(eos, int) else [int(e) for e in eos]
    return []


def forbidden_token_ids(tokenizer: Any, embedding_rows: int) -> set[int]:
    """Token ids the attack must never substitute *into* the input.

    Two classes, both of which break the guarantee that the returned text is the
    thing that was optimised:

    * **Special ids.** `decode(..., skip_special_tokens=True)` deletes them, so
      the realised input is a token shorter than the optimised one.
    * **Rows past the end of the tokenizer.** Embedding tables are often padded
      to a friendly multiple: t5-small has 32128 rows for a 32100-token
      vocabulary. Those 28 rows are real parameters that HotFlip will happily
      score, and they decode to the empty string.
    """
    forbidden = {int(i) for i in (getattr(tokenizer, "all_special_ids", None) or [])}
    try:
        vocab = len(tokenizer)
    except TypeError:  # a tokenizer that does not implement __len__
        vocab = embedding_rows
    forbidden.update(range(min(vocab, embedding_rows), embedding_rows))
    return forbidden


class EmbeddingSemanticsError(RuntimeError):
    """The `inputs_embeds` path does not reproduce the `input_ids` path.

    Raised instead of returning a result, because a mismatch here means the
    attack would be optimising a different function than the model computes.
    """


class ModelAdapter:
    """Uniform view of a Hugging Face model for the attack loop."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    # ------------------------------------------------------------- factory

    @staticmethod
    def for_model(model: Any, tokenizer: Any) -> "ModelAdapter":
        """Pick the right adapter by asking the config, not by isinstance.

        `is_encoder_decoder` is set on every Hugging Face config, so this works
        for architectures that did not exist when this code was written. This is
        the only architecture branch in the package.
        """
        if model.config.is_encoder_decoder:
            return Seq2SeqAdapter(model, tokenizer)
        return CausalAdapter(model, tokenizer)

    # -------------------------------------------------------------- shared

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def embedding_matrix(self) -> torch.Tensor:
        """The `(vocab_size, hidden_dim)` input embedding table.

        This is the table the HotFlip substitution search scores against. Note
        that it is the *unscaled* table; `embedding_scale()` is applied
        separately so the scale factor stays visible where it matters.
        """
        return self.model.get_input_embeddings().weight

    def embedding_scale(self) -> float:
        """The architecture's input-embedding multiplier, or 1.0.

        Looked up on the submodule that actually consumes the embeddings, which
        is the encoder for a seq2seq model and the decoder stack for a causal
        one. Subclasses supply that submodule via `_embedding_consumer()`.
        """
        module = self._embedding_consumer()
        scale = getattr(module, "embed_scale", None)
        if scale is None:
            return 1.0
        return float(scale)

    def _embedding_consumer(self) -> Any:
        raise NotImplementedError

    def embed_values(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The exact tensor the model would build internally from `input_ids`.

        This is what must be passed as `inputs_embeds` for the two paths to
        agree. No gradient bookkeeping -- see `embed()` for that.
        """
        return self.model.get_input_embeddings()(input_ids) * self.embedding_scale()

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """`embed_values` as a differentiable leaf tensor.

        Detached and cloned so that `.grad` on it is the gradient of the
        objective with respect to the input embeddings, and nothing else.
        """
        return self.embed_values(input_ids).detach().clone().requires_grad_(True)

    def eos_token_ids(self) -> list[int]:
        """Token ids that terminate generation. See `resolve_eos_ids`."""
        return resolve_eos_ids(self.model)

    def check_embedding_equivalence(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> float:
        """Verify that `inputs_embeds=embed_values(ids)` reproduces `input_ids`.

        Design notes, because the obvious version of this check is fragile:

        * **Dropout.** Run only after the caller has put the model in eval mode;
          `Attacker.run` does this before calling. Otherwise the two forward
          passes differ for reasons unrelated to embeddings.
        * **No gradient.** Under `no_grad`, so this builds no autograd graph and
          cannot perturb the attack's gradient state.
        * **Decoder inputs.** A seq2seq forward needs them; the seq2seq subclass
          supplies a single decoder-start token, which is enough to get logits.
        * **Tolerance.** When the scale is applied correctly, both paths execute
          the same ops in the same order, so the expected deviation is exactly
          zero. The tolerance is therefore tight, and relative to the logit
          magnitude so it stays meaningful in fp16 as well as fp32.
        * **Cost.** Two forward passes, once per `run()`. Against the hundreds a
          real attack spends, under half a percent.

        Returns the observed maximum absolute logit deviation.

        Raises:
            EmbeddingSemanticsError: if the two paths disagree, meaning this
                architecture's `inputs_embeds` path is not equivalent and is
                therefore not supported by this toolbox.
        """
        kwargs: dict[str, Any] = {"attention_mask": attention_mask}
        kwargs.update(self._equivalence_extra_inputs(input_ids))

        with torch.no_grad():
            from_ids = self.model(input_ids=input_ids, **kwargs).logits
            from_embeds = self.model(
                inputs_embeds=self.embed_values(input_ids), **kwargs
            ).logits

        deviation = float((from_ids - from_embeds).abs().max().item())
        magnitude = max(1.0, float(from_ids.abs().max().item()))
        tolerance = 1e-4 * magnitude

        if not deviation <= tolerance:
            raise EmbeddingSemanticsError(
                f"{type(self.model).__name__}: feeding `inputs_embeds` does not "
                f"reproduce the `input_ids` forward pass (max logit deviation "
                f"{deviation:.4g} > tolerance {tolerance:.4g}). The attack "
                "differentiates through `inputs_embeds`, so on this architecture "
                "it would optimise a different function than the model computes. "
                "This usually means the model applies a transformation to the "
                "embedding table that `ModelAdapter.embedding_scale()` does not "
                "know about. Add it to a ModelAdapter subclass before attacking "
                "this architecture."
            )
        return deviation

    def _equivalence_extra_inputs(self, input_ids: torch.Tensor) -> dict[str, Any]:
        return {}

    # ------------------------------------------------------ per-architecture

    def stop_logits(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        horizon: int,
        force_full_horizon: bool = False,
    ) -> torch.Tensor:
        """Logits at the positions where the model decides whether to stop.

        Returns shape `(steps, vocab_size)`. By default `steps` is the number of
        tokens the model actually generated, which is *at most* `horizon`: the
        trajectory is the model's own greedy output and ends when it emits EOS.
        With `force_full_horizon`, generation is given `min_new_tokens=horizon`
        so it cannot stop early and `steps == horizon` exactly, at the cost of
        scoring a path the model would not have taken. Objectives declare which
        they need; see `objectives.register`.

        The attack reads the EOS column of this and pushes it down.
        """
        raise NotImplementedError

    def count_generated(self, generated: torch.Tensor, input_len: int) -> int:
        """Number of tokens in `generate()`'s output the model actually produced."""
        raise NotImplementedError


class Seq2SeqAdapter(ModelAdapter):
    """Encoder-decoder models (T5, BART, Marian, ...)."""

    def _embedding_consumer(self) -> Any:
        return self.model.get_encoder()

    def _decoder_start_id(self) -> int:
        for source in (getattr(self.model, "generation_config", None), self.model.config):
            if source is None:
                continue
            for name in ("decoder_start_token_id", "bos_token_id", "pad_token_id"):
                value = getattr(source, name, None)
                if isinstance(value, int):
                    return value
        return 0

    def _equivalence_extra_inputs(self, input_ids: torch.Tensor) -> dict[str, Any]:
        start = torch.tensor([[self._decoder_start_id()]], device=input_ids.device)
        return {"decoder_input_ids": start}

    def stop_logits(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        horizon: int,
        force_full_horizon: bool = False,
    ) -> torch.Tensor:
        """Teacher-force the model's own greedy output, then read every step.

        Asking only "will you stop at step 1?" is a weak objective: a model can
        decline to stop once and still terminate at step 2. So we first take the
        model's current greedy continuation (no gradient needed -- it only tells
        us *which* decoder states to look at), then run one differentiable
        forward pass teacher-forced on it. That gives the stop-logits at every
        step along the path the model would actually take.
        """
        with torch.no_grad():
            decoded = self.model.generate(
                inputs_embeds=inputs_embeds.detach(),
                attention_mask=attention_mask,
                max_new_tokens=horizon,
                min_new_tokens=horizon if force_full_horizon else None,
                do_sample=False,
            )

        # `generate()` already prefixes the decoder start token, so `decoded` is
        # a valid decoder input; dropping the final token gives the standard
        # shifted-right teacher-forcing input.
        decoder_input_ids = decoded[:, :-1] if decoded.shape[1] > 1 else decoded

        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        return out.logits[0]

    def count_generated(self, generated: torch.Tensor, input_len: int) -> int:
        # Decoder-only output, but index 0 is the seeded decoder start token,
        # which the model did not generate.
        return int(generated.shape[1] - 1)


class CausalAdapter(ModelAdapter):
    """Decoder-only models (GPT-2, Llama, Qwen, ...)."""

    def _embedding_consumer(self) -> Any:
        get_decoder = getattr(self.model, "get_decoder", None)
        if callable(get_decoder):
            try:
                return get_decoder()
            except (AttributeError, NotImplementedError):
                pass
        return getattr(self.model, "base_model", self.model)

    def stop_logits(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        horizon: int,
        force_full_horizon: bool = False,
    ) -> torch.Tensor:
        """Stop decisions along the model's own greedy continuation.

        A causal LM has no separate decoder, so the naive approach is to read the
        logits at the end of the prompt. That is subtly wrong, and measurably so:
        in a causal model position `i` predicts token `i+1`, so a window of the
        final `horizon` prompt positions contains exactly ONE row that predicts a
        token which will actually be generated. The other `horizon - 1` rows
        predict tokens that are already in the prompt, and suppressing EOS there
        optimises "would you have stopped mid-prompt" -- not the quantity that
        controls generation length.

        So this mirrors the seq2seq path instead: take the model's greedy
        continuation, append it, and read the stop decision at each position that
        predicts a generated token.

        Note that `generate(inputs_embeds=...)` on a causal model returns only the
        new tokens; unlike the `input_ids` path it has no prompt ids to prepend.
        """
        with torch.no_grad():
            continuation = self.model.generate(
                inputs_embeds=inputs_embeds.detach(),
                attention_mask=attention_mask,
                max_new_tokens=horizon,
                min_new_tokens=horizon if force_full_horizon else None,
                do_sample=False,
            )

        prompt_len = inputs_embeds.shape[1]

        if continuation.shape[1] > 1:
            # Teacher-force every generated token but the last. Appending them
            # gives the forward pass a stop-decision position for each one.
            # Gradient still reaches `inputs_embeds` through the concatenation.
            # `embed_values` rather than a raw table lookup, so the appended part
            # carries the same embedding scaling as the part being attacked.
            cont_embeds = self.embed_values(continuation[:, :-1])
            full_embeds = torch.cat([inputs_embeds, cont_embeds], dim=1)
            full_mask = torch.cat(
                [attention_mask, torch.ones_like(continuation[:, :-1])], dim=1
            )
        else:
            full_embeds, full_mask = inputs_embeds, attention_mask

        out = self.model(inputs_embeds=full_embeds, attention_mask=full_mask)
        # Every position from `prompt_len - 1` onward predicts a generated token.
        return out.logits[0][prompt_len - 1 :]

    def count_generated(self, generated: torch.Tensor, input_len: int) -> int:
        # Prompt and continuation come back concatenated, so the prompt comes off.
        return int(generated.shape[1] - input_len)
