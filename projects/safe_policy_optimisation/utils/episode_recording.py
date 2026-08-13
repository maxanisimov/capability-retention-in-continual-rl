"""Shared per-episode recording wrapper for training and evaluation envs.

Previously duplicated across ``stages/train_ppo_shield.py`` and
``stages/train_masa_shielded_policy.py``. This is the richer of the two: it also
tracks unsafe-state visits and derives the per-step cost from the environment's
``label_fn``/``cost_fn`` when ``info`` does not carry one, so it works with the
tabular envs that report cost only through their labelling functions.

The rows appended to :attr:`EpisodeRecorderWrapper.episodes` are the canonical
episode-record schema consumed by ``utils/io.record_rows`` and
``utils/safe_rl.aggregate_violations``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class EpisodeRecorderWrapper(gym.Wrapper):
    """Record completed episode reward/cost/length for a Gymnasium env."""

    def __init__(self, env: gym.Env, *, cost_limit: float) -> None:
        super().__init__(env)
        self.cost_limit = float(cost_limit)
        self.episodes: list[dict[str, float | int | bool]] = []
        self._episode_index = 0
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self._episode_reward = 0.0
        self._episode_cost = 0.0
        self._episode_unsafe_state_visits = 0
        self._episode_length = 0

    def reset(self, **kwargs: Any):
        self._reset_accumulators()
        obs, info = self.env.reset(**kwargs)
        initial_cost = self._state_cost(obs, dict(info))
        self._episode_cost += initial_cost
        self._episode_unsafe_state_visits += int(initial_cost > 0.0)
        return obs, info

    def _state_cost(self, obs: Any, info: dict[str, Any]) -> float:
        if "cost" in info:
            return float(info["cost"])
        unwrapped = self.unwrapped
        if hasattr(unwrapped, "label_fn") and hasattr(unwrapped, "cost_fn"):
            # label_fn/cost_fn are keyed by integer state id. The observation may
            # be a Box feature vector (features mode), so decode it back. Key off
            # the mode, not the shape: single-component features are also 1-D.
            if getattr(unwrapped, "_observation_mode", "index") == "features":
                state = int(unwrapped.features_to_state(obs))
            else:
                state = int(np.asarray(obs).item())
            return float(unwrapped.cost_fn(unwrapped.label_fn(state)))
        return 0.0

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        cost = self._state_cost(obs, info)
        info["cost"] = cost
        self._episode_reward += float(reward)
        self._episode_cost += cost
        self._episode_unsafe_state_visits += int(cost > 0.0)
        self._episode_length += 1
        if terminated or truncated:
            self.episodes.append(
                {
                    "episode": self._episode_index,
                    "reward": self._episode_reward,
                    "cost": self._episode_cost,
                    "length": self._episode_length,
                    "violated": self._episode_cost > self.cost_limit,
                    "unsafe_state_visit_count": int(self._episode_unsafe_state_visits),
                    "safe_trajectory": bool(self._episode_unsafe_state_visits == 0),
                }
            )
            self._episode_index += 1
            self._reset_accumulators()
        return obs, reward, terminated, truncated, info
