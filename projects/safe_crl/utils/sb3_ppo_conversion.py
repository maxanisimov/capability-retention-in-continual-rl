"""Convert an SB3 PPO/A2C ``ActorCriticPolicy`` into a ``torch.nn.Sequential``.

``compute_rashomon_set`` (see ``core/src/interval_utils.py``) only accepts
``torch.nn.Sequential`` models built from layers supported by
``IntervalBoundedModel`` (``Linear``, ``Conv2d``, ``ReLU``, ``Tanh``,
``Flatten``, ``Dropout``). SB3's ``ActorCriticPolicy`` instead splits the
network into a features extractor, a shared/branched ``mlp_extractor``, and
separate ``action_net`` / ``value_net`` heads, so it cannot be passed
directly. This module reassembles the policy or value branch into one
``Sequential`` so it can be certified.

Example
-------
```python
from stable_baselines3 import PPO
from projects.safe_crl.utils.sb3_ppo_conversion import sb3_ppo_to_sequential

model = PPO.load("ppo_cartpole")
policy_net = sb3_ppo_to_sequential(model.policy, head="pi")  # action logits

result = compute_rashomon_set(policy_net, dataset, accuracy=0.85)
```
"""

from __future__ import annotations

import copy

import torch.nn as nn

try:
    from gymnasium import spaces
except ImportError:  # pragma: no cover - older SB3/gym installs
    from gym import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor

_SUPPORTED_LAYER_TYPES = (nn.Linear, nn.Conv2d, nn.ReLU, nn.Tanh, nn.Flatten, nn.Dropout)


def sb3_ppo_to_sequential(policy: ActorCriticPolicy, head: str = "pi") -> nn.Sequential:
    """Rebuild one branch of an SB3 ``ActorCriticPolicy`` as a ``nn.Sequential``.

    Args:
        policy: ``model.policy`` from a trained SB3 ``PPO`` or ``A2C`` model.
        head: ``"pi"`` for the action-logit branch (features_extractor ->
            mlp_extractor.policy_net -> action_net), or ``"vf"`` for the
            value branch (features_extractor -> mlp_extractor.value_net ->
            value_net).

    Returns:
        A new ``nn.Sequential`` with copied (detached from the live policy)
        weights, containing only layer types supported by
        ``IntervalBoundedModel``.

    Raises:
        ValueError: if ``head`` is invalid, the action space is not
            ``Discrete`` (required for ``head="pi"``, since for continuous
            actions ``action_net`` only predicts the Gaussian mean and
            ``log_std`` is a separate parameter not captured by a forward
            pass), the features extractor is not a plain ``FlattenExtractor``
            (e.g. a CNN extractor for image observations), or any layer in
            the resulting network is not in the supported set.
    """
    if head not in ("pi", "vf"):
        raise ValueError(f"head must be 'pi' or 'vf', got {head!r}")

    if head == "pi" and not isinstance(policy.action_space, spaces.Discrete):
        raise ValueError(
            "head='pi' requires a Discrete action space: for continuous "
            "actions action_net only predicts the Gaussian mean, and "
            "log_std is a separate parameter not captured by a Sequential "
            "forward pass."
        )

    if not isinstance(policy.features_extractor, FlattenExtractor):
        raise ValueError(
            f"Only vector observations (FlattenExtractor) are supported, got "
            f"{type(policy.features_extractor).__name__}; CNN feature "
            f"extractors use layer types outside the supported set."
        )

    branch_net = policy.mlp_extractor.policy_net if head == "pi" else policy.mlp_extractor.value_net
    head_layer = policy.action_net if head == "pi" else policy.value_net

    layers = [nn.Flatten(), *branch_net, head_layer]
    for layer in layers:
        if not isinstance(layer, _SUPPORTED_LAYER_TYPES):
            raise ValueError(
                f"Unsupported layer type {type(layer).__name__} in policy "
                f"network; compute_rashomon_set only supports "
                f"{[t.__name__ for t in _SUPPORTED_LAYER_TYPES]}."
            )

    return nn.Sequential(*(copy.deepcopy(layer) for layer in layers))
