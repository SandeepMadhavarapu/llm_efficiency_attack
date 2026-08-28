"""Tests that need a real Hugging Face checkpoint.

Skipped unless `--run-integration` is passed, because they download models and
the default suite is deliberately offline and fast::

    pytest --run-integration

These exist because three of this package's guarantees cannot honestly be tested
against `ToyTokenizer`. The toy tokenizer is a clean bijection by construction,
so it can never exhibit the context-dependent re-segmentation that makes
`encode(decode(ids))` fail on a real BPE or SentencePiece vocabulary -- which is
the exact failure the round-trip machinery exists to prevent. Likewise, tokenizer
inference from `_name_or_path` needs a model that actually has one, and the
headline result in RESULTS.md deserves a test rather than only a script.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

T5 = "t5-small"
GPT2 = "gpt2"
FAST = {
    "max_iterations": 3,
    "perturbation_budget": 2,
    "top_k": 6,
    "max_new_tokens": 32,
    "objective_horizon": 4,
}


def _seq2seq(name):
    from transformers import AutoModelForSeq2SeqLM

    return AutoModelForSeq2SeqLM.from_pretrained(name)


def _causal(name):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(name)


def test_one_argument_constructor_infers_the_tokenizer():
    """`Attacker(model)` -- the exact signature the task specifies.

    The offline suite can only assert the error path here, because fixtures built
    from a config record no `_name_or_path`. This is the case that matters: a
    real checkpoint, no tokenizer passed, everything resolved from the model.
    """
    from llm_efficiency_attack import Attacker

    model = _seq2seq(T5)
    attack = Attacker(model)

    assert attack.tokenizer is not None
    assert attack.tokenizer.__class__.__name__.startswith("T5Tokenizer")

    adv_x, logs = attack.run(
        "translate English to German: The house is wonderful.",
        dict(FAST, protected_prefix_tokens=7),
    )
    assert isinstance(adv_x, str)
    assert isinstance(logs, dict)
    assert logs["perturbation"]["round_trip_exact"] is True


def test_headline_result_is_reproducible():
    """The number RESULTS.md leads with, pinned as a test.

    If this ever drifts, the documentation is wrong and should be regenerated
    rather than quietly left in place.
    """
    from llm_efficiency_attack import Attacker

    _, logs = Attacker(_seq2seq(T5)).run(
        "translate English to German: The house is wonderful.",
        {
            "max_iterations": 10,
            "perturbation_budget": 3,
            "top_k": 20,
            "max_new_tokens": 128,
            "protected_prefix_tokens": 7,
            "objective_horizon": 8,
            "seed": 0,
        },
    )

    assert logs["cost"]["benign_output_tokens"] == 6
    assert logs["cost"]["adversarial_output_tokens"] == 9
    assert logs["cost"]["output_token_ratio"] == pytest.approx(1.5)
    assert logs["censored"]["interpretation"] == "point_estimate"
    assert logs["perturbation"]["hamming_distance"] == 1
    assert logs["perturbation"]["positions_touched"] == 1
    assert logs["perturbation"]["round_trip_exact"] is True


@pytest.mark.parametrize(
    "name,loader,text,protected",
    [
        (T5, _seq2seq, "translate English to German: The house is wonderful.", 7),
        (GPT2, _causal, "The house is wonderful and", 0),
    ],
)
def test_returned_text_round_trips_under_a_real_tokenizer(name, loader, text, protected):
    """SentencePiece and byte-level BPE, the two families that actually break.

    A substitution can re-segment its neighbours and change token positions the
    attack never touched, which would silently violate the perturbation budget in
    the text the caller feeds the model. The search rejects such candidates; this
    asserts the outcome against the real vocabularies rather than a toy one.
    """
    from llm_efficiency_attack import Attacker

    model = loader(name)
    attack = Attacker(model)
    adv_x, logs = attack.run(text, dict(FAST, protected_prefix_tokens=protected))

    optimised = logs["perturbation"]["adversarial_token_ids"]
    realised = attack.tokenizer(adv_x)["input_ids"]

    assert logs["perturbation"]["round_trip_exact"] is True
    assert realised == optimised, "returned text must re-tokenise to the optimised ids"

    original = logs["perturbation"]["original_token_ids"]
    assert len(original) == len(optimised)
    assert logs["perturbation"]["hamming_distance"] == sum(
        a != b for a, b in zip(original, optimised)
    )
    assert (
        logs["perturbation"]["hamming_distance"]
        <= logs["perturbation"]["positions_touched"]
        <= logs["perturbation"]["budget"]
    )


def test_committed_candidates_are_never_special_or_out_of_vocabulary():
    """t5-small has 32128 embedding rows for 32100 tokens, plus 103 special ids.

    Both classes decode away or decode to nothing, so committing one would change
    the length of the input the model actually receives.
    """
    from llm_efficiency_attack import Attacker
    from llm_efficiency_attack.adapters import forbidden_token_ids

    attack = Attacker(_seq2seq(T5))
    _, logs = attack.run(
        "translate English to German: The house is wonderful.",
        dict(FAST, top_k=20, protected_prefix_tokens=7),
    )

    illegal = forbidden_token_ids(
        attack.tokenizer, attack.adapter.embedding_matrix().shape[0]
    )
    assert len(illegal) == 131, "expected 103 special ids plus 28 surplus rows"

    adversarial = logs["perturbation"]["adversarial_token_ids"]
    for position in logs["perturbation"]["positions_changed"]:
        assert adversarial[position] not in illegal
