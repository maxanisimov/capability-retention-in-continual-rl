"""CartPole environment construction helpers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from projects.safe_crl.pipelines.envs.cartpole.core.env.tunable_cartpole import (
    TUNABLE_CARTPOLE_V1_ID,
    ensure_tunable_cartpole_registered,
)
from projects.safe_crl.pipelines.envs.cartpole.core.env.wrappers import AppendTaskIDObservationWrapper


def make_cartpole_env(
    env_id: str = TUNABLE_CARTPOLE_V1_ID,
    *,
    gravity: float | None = None,
    masscart: float | None = None,
    masspole: float | None = None,
    length: float | None = None,
    force_mag: float | None = None,
    tau: float | None = None,
    theta_threshold_radians: float | None = None,
    x_threshold: float | None = None,
    kinematics_integrator: str | None = None,
    reset_low: float | None = None,
    reset_high: float | None = None,
    sutton_barto_reward: bool = False,
    task_id: float = 0.0,
    append_task_id: bool = True,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a CartPole env with optional dynamics/task-id shifts."""
    ensure_tunable_cartpole_registered()

    make_kwargs: dict[str, Any] = {
        "render_mode": render_mode,
        "sutton_barto_reward": bool(sutton_barto_reward),
    }
    for name, value in {
        "gravity": gravity,
        "masscart": masscart,
        "masspole": masspole,
        "length": length,
        "force_mag": force_mag,
        "tau": tau,
        "theta_threshold_radians": theta_threshold_radians,
        "x_threshold": x_threshold,
        "reset_low": reset_low,
        "reset_high": reset_high,
    }.items():
        if value is not None:
            make_kwargs[name] = float(value)
    if kinematics_integrator is not None:
        make_kwargs["kinematics_integrator"] = str(kinematics_integrator)

    env = gym.make(env_id, **make_kwargs)
    if append_task_id:
        env = AppendTaskIDObservationWrapper(env, task_id=task_id)
    return env


_make_cartpole_env = make_cartpole_env

