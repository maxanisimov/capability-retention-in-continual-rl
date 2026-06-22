"""Breakout environment construction helpers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from experiments.pipelines.envs.breakout.core.env.tunable_breakout import (
    TUNABLE_BREAKOUT_V5_ID,
    ensure_tunable_breakout_registered,
)


def make_breakout_env(
    env_id: str = TUNABLE_BREAKOUT_V5_ID,
    *,
    mode: int | None = None,
    difficulty: int | None = None,
    obs_type: str = "rgb",
    frameskip: tuple[int, int] | list[int] | int = 4,
    repeat_action_probability: float = 0.25,
    full_action_space: bool = False,
    continuous: bool = False,
    continuous_action_threshold: float = 0.5,
    max_num_frames_per_episode: int | None = 108000,
    sound_obs: bool = False,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a Breakout env with optional ALE stochasticity and mode shifts."""
    ensure_tunable_breakout_registered()

    make_kwargs: dict[str, Any] = {
        "render_mode": render_mode,
        "mode": mode,
        "difficulty": difficulty,
        "obs_type": obs_type,
        "frameskip": frameskip,
        "repeat_action_probability": repeat_action_probability,
        "full_action_space": full_action_space,
        "continuous": continuous,
        "continuous_action_threshold": continuous_action_threshold,
        "max_num_frames_per_episode": max_num_frames_per_episode,
        "sound_obs": sound_obs,
    }
    return gym.make(env_id, **make_kwargs)


_make_breakout_env = make_breakout_env
