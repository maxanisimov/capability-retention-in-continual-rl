"""MASA-style environment construction with a probabilistic shield.

Extracted from ``stages/train_masa_shielded_policy.py`` when that trainer was
archived: the trainer was unreachable from any pipeline, but this env builder is
still a live dependency of ``stages/rollout_policy_gif.py``, which renders MASA
rollouts.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from masa.common.wrappers import ConstraintPersistentWrapper
from masa.prob_shield.prob_shield_wrapper_v2 import ProbShieldWrapperDisc

from projects.safe_policy_optimisation.utils.episode_recording import EpisodeRecorderWrapper
from projects.safe_policy_optimisation.utils.safe_crl_bridge import make_custom_masa_env

ALGORITHM_NAME = "masa_shielded_ppo"


class SafetyBoundArrayWrapper(gym.ObservationWrapper):
    """Convert MASA's scalar safety bound to the declared one-element Box."""

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        obs = dict(observation)
        obs["safety_bound"] = np.asarray([obs["safety_bound"]], dtype=np.float32)
        return obs


class CostInfoWrapper(gym.Wrapper):
    """Expose the label-derived cost in ``info`` after shield projection."""

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = int(np.asarray(obs["orig_obs"]).item())
        cost = float(self.unwrapped.cost_fn(self.unwrapped.label_fn(state)))
        info = dict(info)
        info["cost"] = cost
        info["violated_step"] = cost > 0.0
        info["orig_obs"] = state
        info["safety_bound"] = float(np.asarray(obs["safety_bound"]).reshape(-1)[0])
        return obs, reward, terminated, truncated, info


def make_masa_shielded_env(
    env_id: str,
    *,
    max_episode_steps: int,
    env_kwargs: dict[str, Any],
    safety_tolerance: float,
    theta: float,
    max_vi_steps: int,
    granularity: int,
    cost_limit: float,
    record_episodes: bool,
    render_mode: str | None = None,
) -> gym.Env:
    """Build a MASA-style env with ``ProbShieldWrapperDisc`` at the requested tolerance."""

    base_env = make_custom_masa_env(
        env_id,
        max_episode_steps=max_episode_steps,
        env_kwargs=env_kwargs,
        render_mode=render_mode,
    )
    shielded = ProbShieldWrapperDisc(
        ConstraintPersistentWrapper(base_env),
        label_fn=base_env.unwrapped.label_fn,
        cost_fn=base_env.unwrapped.cost_fn,
        theta=theta,
        max_vi_steps=max_vi_steps,
        init_safety_bound=safety_tolerance,
        granularity=granularity,
    )
    env: gym.Env = SafetyBoundArrayWrapper(shielded)
    env = CostInfoWrapper(env)
    if record_episodes:
        env = EpisodeRecorderWrapper(env, cost_limit=cost_limit)
    return env
