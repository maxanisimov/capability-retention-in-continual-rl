"""Mountain Car environment construction helpers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from experiments.pipelines.mountaincar.core.env.tunable_mountain_car import (
    TUNABLE_MOUNTAIN_CAR_V0_ID,
    ensure_tunable_mountain_car_registered,
)
from experiments.pipelines.mountaincar.core.env.wrappers import AppendTaskIDObservationWrapper


def make_mountaincar_env(
    env_id: str = TUNABLE_MOUNTAIN_CAR_V0_ID,
    *,
    force: float | None = None,
    gravity: float | None = None,
    goal_position: float | None = None,
    goal_velocity: float | None = None,
    min_position: float | None = None,
    max_position: float | None = None,
    max_speed: float | None = None,
    reset_low: float | None = None,
    reset_high: float | None = None,
    task_id: float = 0.0,
    append_task_id: bool = True,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a Mountain Car env with optional dynamics/task-id shifts."""
    ensure_tunable_mountain_car_registered()

    make_kwargs: dict[str, Any] = {"render_mode": render_mode}
    if force is not None:
        make_kwargs["force"] = float(force)
    if gravity is not None:
        make_kwargs["gravity"] = float(gravity)
    if goal_position is not None:
        make_kwargs["goal_position"] = float(goal_position)
    if goal_velocity is not None:
        make_kwargs["goal_velocity"] = float(goal_velocity)
    if min_position is not None:
        make_kwargs["min_position"] = float(min_position)
    if max_position is not None:
        make_kwargs["max_position"] = float(max_position)
    if max_speed is not None:
        make_kwargs["max_speed"] = float(max_speed)
    if reset_low is not None:
        make_kwargs["reset_low"] = float(reset_low)
    if reset_high is not None:
        make_kwargs["reset_high"] = float(reset_high)

    env = gym.make(env_id, **make_kwargs)
    if append_task_id:
        env = AppendTaskIDObservationWrapper(env, task_id=task_id)
    return env


_make_mountaincar_env = make_mountaincar_env

