"""Provably-safe DQN: projection (retention) + shielded exploration (safety).

:class:`ProvablySafeDQN` extends
:class:`provably_safe_policy_optimisation.projected_dqn.ProjectedDQN` with a
user-provided safety :class:`~provably_safe_policy_optimisation.shield.Shield`.
During data collection, every action executed in the environment is checked
against the shield: if it is unsafe in the current (discrete) state, it is
replaced by an action sampled uniformly from that state's safe actions. Because
DQN is off-policy, the replay buffer stores the executed (safe) action and learns
from it correctly.

Scope: shielding applies to exploration/data-collection only; ``predict()`` (for
evaluation/deployment) is intentionally left unshielded.

Usage
-----
>>> from provably_safe_policy_optimisation import ProvablySafeDQN
>>> model = ProvablySafeDQN("MlpPolicy", env, shield=mask)  # mask: (n_states, n_actions)
>>> model.learn(total_timesteps=100_000)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from gymnasium import spaces

from provably_safe_policy_optimisation._projection_logging import record_shield_window
from provably_safe_policy_optimisation.projected_dqn import ProjectedDQN
from provably_safe_policy_optimisation.shield import Shield, as_shield


class ProvablySafeDQN(ProjectedDQN):
    """DQN with projected updates and shielded exploration."""

    def __init__(
        self,
        *args: Any,
        shield: Any = None,
        obs_to_state: Any = None,
        shield_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._shield: Shield | None = None
        super().__init__(*args, **kwargs)
        if shield is not None:
            self.set_shield(shield, obs_to_state, seed=shield_seed)
        else:
            warnings.warn(
                "ProvablySafeDQN constructed without a shield; no shielding will be applied "
                "until set_shield(...) is called (e.g. after load).",
                stacklevel=2,
            )

    def set_shield(self, shield: Any, obs_to_state: Any = None, *, seed: int | None = None) -> None:
        """Attach (or re-attach) the safety shield, validating it against the action space."""
        shield = as_shield(shield, obs_to_state, seed=seed)
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("ProvablySafeDQN requires a Discrete action space.")
        if shield.n_actions != int(self.action_space.n):
            raise ValueError(
                f"Shield n_actions ({shield.n_actions}) does not match the action "
                f"space ({int(self.action_space.n)}).",
            )
        self._shield = shield

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Any = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        action, buffer_action = super()._sample_action(learning_starts, action_noise, n_envs)
        if self._shield is None:
            return action, buffer_action
        states = self._shield.obs_to_state(self._last_obs)
        shielded = self._shield.override(states, action).reshape(np.asarray(action).shape)
        # Discrete action space: stored (buffer) action equals the executed action.
        return shielded, shielded

    def shield_diagnostics(self) -> dict[str, float]:
        """Cumulative shield intervention diagnostics (see ``Shield.diagnostics``)."""
        return self._shield.diagnostics() if self._shield is not None else {}

    def train(self, *args: Any, **kwargs: Any) -> Any:
        result = super().train(*args, **kwargs)
        record_shield_window(self)
        return result
