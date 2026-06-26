"""FrozenLake environment construction helpers."""

from __future__ import annotations

import gymnasium as gym

from projects.safe_crl.pipelines.trajectory_retention.frozenlake.core.env.wrappers import (
    CoordObsWrapper,
    DenseShapingWrapper,
    SafetyFlagWrapper,
)


def make_env_from_layout(
    env_map: list[str],
    max_episode_steps: int,
    *,
    task_num: float = 0.0,
    shaped: bool,
    render_mode: str | None = None,
) -> gym.Env:
    env = gym.make(
        "FrozenLake-v1",
        desc=env_map,
        is_slippery=False,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )
    env = CoordObsWrapper(env, task_num=task_num)
    env = SafetyFlagWrapper(env)
    if shaped:
        env = DenseShapingWrapper(env)
    return env

