"""Feature-observation variant of the structured slippery FrozenLake adapter.

The index-mode adapter in :mod:`frozen_lake_experiment` feeds the actor a
one-hot state id, so the actor's input dimension *is* ``|S|`` and the exhaustive
certificate dataset is a dense ``|W| x |S|`` one-hot matrix. That matrix is the
quadratic memory term that stops layouts above 256 from running at all.

This module keeps the identical environment -- same layout, dynamics, rewards,
shield and witness proofs -- and changes only what the agent observes: the
normalised ``(row, col)`` pair. The certificate dataset then becomes ``|W| x 2``.

The feature encoding is exactly invertible, so the shield (indexed by state id)
keeps working through :meth:`make_obs_to_state`, mirroring the contract that
``projects/safe_crl/utils/masa_tabular_envs/base.TabularEnv`` defines.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from projects.safe_policy_optimisation.utils.frozen_lake_experiment import (
    SparseFrozenLake,
)

ENV_NAME = "PSPOFrozenLakeFeatures-v0"
ENV_ID = f"projects.safe_policy_optimisation.archive.utils.frozen_lake_features_experiment:{ENV_NAME}"


class FeatureFrozenLake(SparseFrozenLake):
    """Structured slippery FrozenLake observed as normalised ``(row, col)``.

    ``observation_mode="features"`` replaces the ``Discrete(|S|)`` observation
    space with ``Box(0, 1, (2,))``; ``"index"`` reproduces the parent exactly.
    """

    def __init__(self, *, observation_mode: str = "features", **kwargs: Any) -> None:
        if observation_mode not in ("index", "features"):
            raise ValueError(
                f"observation_mode must be 'index' or 'features'; got {observation_mode!r}."
            )
        # The parent only accepts "index"; it defines the dynamics, and this
        # subclass re-declares the observation space afterwards.
        super().__init__(observation_mode="index", **kwargs)
        self._observation_mode = observation_mode
        self._feature_maxima = np.asarray(self._component_maxima(), dtype=np.float64)
        self._feature_dim = int(self._feature_maxima.shape[0])
        if observation_mode == "features":
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self._feature_dim,), dtype=np.float32
            )

    # ---------------------------------------------------- feature observations

    def _state_components(self, state: int) -> tuple[int, ...]:
        return (int(state) // self.ncol, int(state) % self.ncol)

    def _components_to_state(self, components: Any) -> int:
        row, col = (int(c) for c in components)
        return row * self.ncol + col

    def _component_maxima(self) -> tuple[int, ...]:
        return (self.nrow - 1, self.ncol - 1)

    def state_to_features(self, state: int) -> np.ndarray:
        comps = np.asarray(self._state_components(int(state)), dtype=np.float64)
        denom = np.where(self._feature_maxima > 0, self._feature_maxima, 1.0)
        return (comps / denom).astype(np.float32)

    def features_to_state(self, features: Any) -> int:
        arr = (
            features.detach().cpu().numpy()
            if hasattr(features, "detach")
            else np.asarray(features)
        )
        comps = np.rint(
            arr.reshape(-1).astype(np.float64) * self._feature_maxima
        ).astype(int)
        return int(self._components_to_state(comps))

    def _observe(self, state: int) -> Any:
        if self._observation_mode == "features":
            return self.state_to_features(int(state))
        return int(state)

    def make_obs_to_state(self):
        """Batched observation -> state-id map for the shield.

        Vectorised on purpose: the shield calls this on every environment step
        and on whole rollout batches, so a per-row Python decode would dominate.
        """
        mode = self._observation_mode
        maxima = self._feature_maxima
        ncol = self.ncol

        def decode(obs: Any) -> np.ndarray:
            arr = obs.detach().cpu().numpy() if hasattr(obs, "detach") else np.asarray(obs)
            if mode != "features":
                return arr.reshape(-1).astype(np.int64)
            rows = arr.reshape(-1, int(maxima.shape[0])).astype(np.float64)
            comps = np.rint(rows * maxima).astype(np.int64)
            return comps[:, 0] * ncol + comps[:, 1]

        return decode

    # ------------------------------------------------------------- gym plumbing

    def reset(self, **kwargs: Any):
        obs, info = super().reset(**kwargs)
        return self._observe(int(obs)), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._observe(int(obs)), reward, terminated, truncated, info


def feature_matrix(env: FeatureFrozenLake) -> np.ndarray:
    """Dense ``|S| x 2`` feature table, used by the initial-actor fit."""
    states = np.arange(env._n_states, dtype=np.int64)
    rows = states // env.ncol
    cols = states % env.ncol
    maxima = np.asarray(env._component_maxima(), dtype=np.float64)
    return (np.stack([rows, cols], axis=1) / maxima).astype(np.float32)


if ENV_NAME not in gym.registry:
    gym.register(ENV_NAME, entry_point=FeatureFrozenLake)
