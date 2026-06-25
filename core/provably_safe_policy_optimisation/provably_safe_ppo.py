"""Provably-safe PPO: projection (retention) + shielded exploration (safety).

:class:`ProvablySafePPO` extends
:class:`provably_safe_policy_optimisation.projected_ppo.ProjectedPPO` with a
user-provided safety :class:`~provably_safe_policy_optimisation.shield.Shield`.

During rollout collection, the policy's ``forward`` is wrapped so that, after the
policy samples an action, any unsafe action is replaced by one drawn uniformly
from the current state's safe actions. The **executed (shielded)** action is what
gets stored in the rollout buffer, and its log-prob is recomputed under the
current policy, so PPO learns to *propose* safe actions and the importance ratios
stay consistent.

Scope: only ``collect_rollouts`` calls ``policy.forward``; PPO's ``train()`` uses
``evaluate_actions`` and ``predict()`` uses ``_predict``, so neither is shielded
(shielding applies to training exploration only).

Usage
-----
>>> from provably_safe_policy_optimisation import ProvablySafePPO
>>> model = ProvablySafePPO("MlpPolicy", env, shield=mask)  # mask: (n_states, n_actions)
>>> model.learn(total_timesteps=100_000)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces

from provably_safe_policy_optimisation._projection_logging import record_shield_window
from provably_safe_policy_optimisation.projected_ppo import ProjectedPPO
from provably_safe_policy_optimisation.shield import Shield, as_shield


class ProvablySafePPO(ProjectedPPO):
    """PPO with projected updates and shielded exploration."""

    def __init__(
        self,
        *args: Any,
        shield: Any = None,
        obs_to_state: Any = None,
        shield_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._shield: Shield | None = None
        # _setup_model (called inside super().__init__) wraps policy.forward; the
        # wrapper reads self._shield live, so attaching the shield afterwards is fine.
        super().__init__(*args, **kwargs)
        if shield is not None:
            self.set_shield(shield, obs_to_state, seed=shield_seed)
        else:
            warnings.warn(
                "ProvablySafePPO constructed without a shield; no shielding will be applied "
                "until set_shield(...) is called (e.g. after load).",
                stacklevel=2,
            )

    def set_shield(self, shield: Any, obs_to_state: Any = None, *, seed: int | None = None) -> None:
        """Attach (or re-attach) the safety shield, validating it against the action space."""
        shield = as_shield(shield, obs_to_state, seed=seed)
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("ProvablySafePPO requires a Discrete action space.")
        if shield.n_actions != int(self.action_space.n):
            raise ValueError(
                f"Shield n_actions ({shield.n_actions}) does not match the action "
                f"space ({int(self.action_space.n)}).",
            )
        self._shield = shield

    def _setup_model(self) -> None:
        super()._setup_model()
        self._wrap_policy_forward_for_shielding()

    def _wrap_policy_forward_for_shielding(self) -> None:
        """Wrap ``policy.forward`` so rollout actions are shielded (read ``self._shield`` live)."""
        if getattr(self, "_shield_forward_wrapped", False):
            return
        policy = self.policy
        original_forward = policy.forward
        model = self

        def shielded_forward(obs: th.Tensor, deterministic: bool = False):  # type: ignore[no-untyped-def]
            actions, values, log_prob = original_forward(obs, deterministic=deterministic)
            shield = model._shield
            if shield is None:
                return actions, values, log_prob
            states = shield.obs_to_state(obs)
            actions_np = actions.detach().cpu().numpy().reshape(-1)
            shielded_np = shield.override(states, actions_np)
            if not np.array_equal(shielded_np, actions_np):
                shielded = th.as_tensor(
                    shielded_np, device=actions.device, dtype=actions.dtype
                ).reshape(actions.shape)
                # Recompute the log-prob of the executed (shielded) action so the
                # rollout buffer is consistent for PPO's importance ratio.
                log_prob = policy.get_distribution(obs).log_prob(shielded)
                actions = shielded
            return actions, values, log_prob

        policy.forward = shielded_forward  # type: ignore[method-assign]
        self._shield_forward_wrapped = True

    def shield_diagnostics(self) -> dict[str, float]:
        """Cumulative shield intervention diagnostics (see ``Shield.diagnostics``)."""
        return self._shield.diagnostics() if self._shield is not None else {}

    def train(self, *args: Any, **kwargs: Any) -> Any:
        result = super().train(*args, **kwargs)
        record_shield_window(self)
        return result
