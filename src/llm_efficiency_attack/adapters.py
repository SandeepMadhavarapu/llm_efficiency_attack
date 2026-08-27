"""The model-agnostic seam.

Hugging Face already hides most architectural differences: `get_input_embeddings()`
returns the embedding table for any model, and `config.is_encoder_decoder` says
which family we are in. Only three things genuinely differ between a seq2seq and
a causal LM for this attack, and this module is the only place that knows them:

1. A seq2seq forward pass needs decoder inputs; a causal one does not.
2. Where the "will you stop now?" logits live.
3. How to count generated tokens out of `generate()`.

Everything else in the package is written against `ModelAdapter` and therefore
never branches on model type. Supporting a new architecture means adding a
subclass here and nothing else.
"""

from __future__ import annotations

from typing import Any

import torch


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
        for architectures that did not exist when this code was written.
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

        This is the table the HotFlip substitution search scores against.
        """
        return self.model.get_input_embeddings().weight

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for `input_ids` as a differentiable leaf tensor."""
        table = self.model.get_input_embeddings()
        return table(input_ids).detach().clone().requires_grad_(True)

    def eos_token_ids(self) -> list[int]:
        """Token ids that terminate generation.

        `generation_config` is what `generate()` actually consults and can differ
        from `config`, so it wins. The value may be a single int or a list; both
        are normalised to a list here so callers never special-case it.
        """
        for source in (getattr(self.model, "generation_config", None), self.model.config):
            if source is None:
                continue
            eos = getattr(source, "eos_token_id", None)
            if eos is None:
                continue
            return [eos] if isinstance(eos, int) else list(eos)
        return []

    # ------------------------------------------------------ per-architecture

    def stop_logits(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        """Logits at the positions where the model decides whether to stop.

        Returns shape `(steps, vocab_size)`. The attack reads the EOS column of
        this and pushes it down.
        """
        raise NotImplementedError

    def count_generated(self, generated: torch.Tensor, input_len: int) -> int:
        """Number of tokens in `generate()`'s output the model actually produced."""
        raise NotImplementedError


class Seq2SeqAdapter(ModelAdapter):
    """Encoder-decoder models (T5, BART, Marian, ...)."""

    def stop_logits(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, horizon: int
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

    def stop_logits(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, horizon: int
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
                do_sample=False,
            )

        prompt_len = inputs_embeds.shape[1]

        if continuation.shape[1] > 1:
            # Teacher-force every generated token but the last. Appending them
            # gives the forward pass a stop-decision position for each one.
            # Gradient still reaches `inputs_embeds` through the concatenation.
            cont_embeds = self.model.get_input_embeddings()(continuation[:, :-1])
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
