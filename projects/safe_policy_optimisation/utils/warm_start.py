"""Load a PSPO safe-initialisation policy into a baseline method's actor.

PSPO's base policy is a plain ``nn.Sequential``
(``Linear -> Tanh -> ... -> Linear``) saved by
``stages/compute_shield_rashomon_set.py`` as ``base_policy.pt`` with an
``architecture`` block beside the ``state_dict``.

Two actor layouts consume it:

* the **safe-RL baselines** (``CPO``, ``PPOLagrangian``, ``PPOPIDLagrangian``)
  build ``model.actor`` with ``safe_rl_baselines._mlp``, which produces exactly
  that ``nn.Sequential`` layout, so the state dict transfers unchanged;
* the **Stable-Baselines3 policies** (``PPO``, ``ProvablySafePPO``) split the
  same computation across ``mlp_extractor.policy_net`` and ``action_net``, so
  the parameters have to be renamed --
  ``train_pspo.base_state_dict_to_ppo_actor`` already owns that mapping and is
  reused here rather than duplicated.

Only the *actor* is warm-started. Critics, Lagrange multipliers, and optimizer
state all start fresh, so the ablation isolates the initialisation of the
policy itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_base_policy(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Read a ``base_policy.pt`` payload and return ``(architecture, state_dict)``."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(
            f"{path} is not a base-policy payload with a 'state_dict' entry."
        )
    architecture = dict(payload.get("architecture") or {})
    if not architecture:
        raise ValueError(f"{path} does not record an 'architecture' block.")
    state_dict = {k: v.detach().clone() for k, v in payload["state_dict"].items()}
    return architecture, state_dict


def check_architecture_compatibility(
    architecture: dict[str, Any],
    *,
    obs_dim: int,
    n_actions: int,
    hidden_dim: int,
    n_hidden: int,
) -> None:
    """Fail loudly when the warm start does not fit the method being trained.

    A silent shape mismatch would either raise deep inside torch or -- worse --
    load partially under ``strict=False`` and leave a half-initialised actor
    that still trains, producing a result that looks like the ablation but is
    not.
    """

    expected = {
        "input_dim": int(obs_dim),
        "n_actions": int(n_actions),
        "hidden_dim": int(hidden_dim),
        "n_hidden": int(n_hidden),
        "activation": "Tanh",
    }
    mismatches = {
        key: {"base_policy": architecture.get(key), "method": value}
        for key, value in expected.items()
        if architecture.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"PSPO base policy is incompatible with this method: {mismatches}."
        )


def warm_start_actor(model: Any, path: Path, *, hidden_dim: int, n_hidden: int) -> dict[str, Any]:
    """Copy a PSPO base policy into ``model``'s actor, in place.

    Returns a provenance record for the run's ``config.json`` so a completed
    result always states which initialisation it started from.
    """

    architecture, state_dict = load_base_policy(path)

    if hasattr(model, "actor") and isinstance(model.actor, torch.nn.Sequential):
        # safe_rl_baselines layout: identical Sequential, load as-is.
        obs_dim = int(model.actor[0].in_features)
        n_actions = int(model.actor[-1].out_features)
        check_architecture_compatibility(
            architecture,
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
        )
        model.actor.load_state_dict(state_dict, strict=True)
        target = "safe_rl_baselines.actor"
    elif hasattr(model, "policy"):
        # Stable-Baselines3 layout: rename into the split actor.
        from projects.safe_policy_optimisation.stages.train_pspo import (
            base_state_dict_to_ppo_actor,
        )

        # Read the dimensions off the layers, not off the observation space:
        # these environments use Discrete spaces (shape ``()``), which SB3
        # one-hots inside its feature extractor, so the actor's true input
        # width lives on the first linear layer.
        obs_dim = int(model.policy.mlp_extractor.policy_net[0].in_features)
        n_actions = int(model.policy.action_net.out_features)
        check_architecture_compatibility(
            architecture,
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
        )
        mapped = base_state_dict_to_ppo_actor(architecture, state_dict)
        current = model.policy.state_dict()
        unknown = sorted(set(mapped) - set(current))
        if unknown:
            raise KeyError(f"Mapped parameters absent from the SB3 policy: {unknown}.")
        current.update(mapped)
        # strict=True against the merged dict: every policy parameter is still
        # present, so this cannot silently skip a tensor.
        model.policy.load_state_dict(current, strict=True)
        target = "stable_baselines3.policy"
    else:
        raise TypeError(
            f"{type(model).__name__} exposes neither a Sequential 'actor' nor a "
            "'policy'; cannot warm-start it."
        )

    return {
        "init_policy_path": str(Path(path).resolve()),
        "init_policy_architecture": architecture,
        "init_policy_target": target,
        "init_policy_parameters": int(sum(v.numel() for v in state_dict.values())),
    }
