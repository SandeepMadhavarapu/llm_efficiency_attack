"""White-box efficiency attacks against Hugging Face sequence models.

An efficiency attack edits an input within a fixed token budget so that the model
runs more decoding steps than it otherwise would. This package generalises the
NMTSloth attack (FSE'22) behind a model-type-agnostic adapter, so the same
optimisation loop drives seq2seq and causal models.

Two things this does *not* claim, because neither is measured:

* The budget bounds the number of token positions that may differ. It enforces
  no semantic-similarity, fluency, or human-perceptibility constraint.
* The cost metric is generated-token count, a reproducible proxy for
  autoregressive decoding work -- not latency, energy, or total compute.

See README.md for scope, and RESULTS.md for what has actually been measured,
including the cases where the attack does not help.

Typical use::

    from llm_efficiency_attack import Attacker

    attack = Attacker(model)
    adv_x, logs = attack.run(x, config)
"""

from .attacker import Attacker
from .config import AttackConfig
from .metrics import measure_cost, measure_cost_from_ids
from .objectives import available_objectives, get_objective, register

__all__ = [
    "Attacker",
    "AttackConfig",
    "measure_cost",
    "measure_cost_from_ids",
    "available_objectives",
    "get_objective",
    "register",
]

__version__ = "0.1.0"
