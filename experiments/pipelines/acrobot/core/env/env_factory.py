"""Acrobot environment construction helpers."""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from experiments.pipelines.acrobot.core.env.tunable_acrobot import (
    TUNABLE_ACROBOT_V1_ID,
    ensure_tunable_acrobot_registered,
)
from experiments.pipelines.acrobot.core.env.wrappers import AppendTaskIDObservationWrapper


def make_acrobot_env(
    env_id: str = TUNABLE_ACROBOT_V1_ID,
    *,
    gravity: float | None = None,
    link_length_1: float | None = None,
    link_length_2: float | None = None,
    link_mass_1: float | None = None,
    link_mass_2: float | None = None,
    link_com_pos_1: float | None = None,
    link_com_pos_2: float | None = None,
    link_moi: float | None = None,
    max_vel_1: float | None = None,
    max_vel_2: float | None = None,
    available_torque: Sequence[float] | np.ndarray | None = None,
    torque_noise_max: float | None = None,
    dt: float | None = None,
    book_or_nips: str | None = None,
    terminal_height: float | None = None,
    reset_low: float | None = None,
    reset_high: float | None = None,
    initial_state: Sequence[float] | np.ndarray | None = None,
    task_id: float = 0.0,
    append_task_id: bool = True,
    render_mode: str | None = None,
) -> gym.Env:
    """Create an Acrobot env with optional dynamics/task-id shifts."""
    ensure_tunable_acrobot_registered()

    make_kwargs: dict[str, Any] = {"render_mode": render_mode}
    for name, value in {
        "gravity": gravity,
        "link_length_1": link_length_1,
        "link_length_2": link_length_2,
        "link_mass_1": link_mass_1,
        "link_mass_2": link_mass_2,
        "link_com_pos_1": link_com_pos_1,
        "link_com_pos_2": link_com_pos_2,
        "link_moi": link_moi,
        "max_vel_1": max_vel_1,
        "max_vel_2": max_vel_2,
        "torque_noise_max": torque_noise_max,
        "dt": dt,
        "terminal_height": terminal_height,
        "reset_low": reset_low,
        "reset_high": reset_high,
    }.items():
        if value is not None:
            make_kwargs[name] = float(value)
    if available_torque is not None:
        make_kwargs["available_torque"] = tuple(float(x) for x in available_torque)
    if book_or_nips is not None:
        make_kwargs["book_or_nips"] = str(book_or_nips)
    if initial_state is not None:
        make_kwargs["initial_state"] = tuple(float(x) for x in initial_state)

    env = gym.make(env_id, **make_kwargs)
    if append_task_id:
        env = AppendTaskIDObservationWrapper(env, task_id=task_id)
    return env


_make_acrobot_env = make_acrobot_env

