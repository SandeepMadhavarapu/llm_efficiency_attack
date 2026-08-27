"""White-box efficiency attacks against Hugging Face sequence models.

An efficiency attack perturbs an input almost imperceptibly so that the model is
forced to run far more decoding steps than it otherwise would, inflating latency
and serving cost. This package generalises the NMTSloth attack (FSE'22) into a
model-agnostic toolbox.

Typical use::

    from llm_efficiency_attack import Attacker

    attack = Attacker(model)
    adv_x, logs = attack.run(x, config)
"""

from .attacker import Attacker
from .config import AttackConfig
from .metrics import measure_cost
from .objectives import available_objectives, get_objective, register

__all__ = [
    "Attacker",
    "AttackConfig",
    "measure_cost",
    "available_objectives",
    "get_objective",
    "register",
]

__version__ = "0.1.0"
