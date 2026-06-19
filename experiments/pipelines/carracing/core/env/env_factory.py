"""CarRacing environment construction helpers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from experiments.pipelines.carracing.core.env.tunable_car_racing import (
    TUNABLE_CAR_RACING_V3_ID,
    ensure_tunable_car_racing_registered,
)


def make_carracing_env(
    env_id: str = TUNABLE_CAR_RACING_V3_ID,
    *,
    verbose: bool = False,
    lap_complete_percent: float | None = None,
    domain_randomize: bool = False,
    continuous: bool = True,
    discrete_steer: float | None = None,
    discrete_gas: float | None = None,
    discrete_brake: float | None = None,
    frame_dt: float | None = None,
    world_velocity_iterations: int | None = None,
    world_position_iterations: int | None = None,
    per_step_penalty: float | None = None,
    off_track_penalty: float | None = None,
    playfield: float | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a CarRacing env with optional task and reward shifts."""
    ensure_tunable_car_racing_registered()

    make_kwargs: dict[str, Any] = {
        "render_mode": render_mode,
        "verbose": bool(verbose),
        "domain_randomize": bool(domain_randomize),
        "continuous": bool(continuous),
    }
    for name, value in {
        "lap_complete_percent": lap_complete_percent,
        "discrete_steer": discrete_steer,
        "discrete_gas": discrete_gas,
        "discrete_brake": discrete_brake,
        "frame_dt": frame_dt,
        "per_step_penalty": per_step_penalty,
        "off_track_penalty": off_track_penalty,
        "playfield": playfield,
    }.items():
        if value is not None:
            make_kwargs[name] = float(value)
    if world_velocity_iterations is not None:
        make_kwargs["world_velocity_iterations"] = int(world_velocity_iterations)
    if world_position_iterations is not None:
        make_kwargs["world_position_iterations"] = int(world_position_iterations)

    return gym.make(env_id, **make_kwargs)


_make_carracing_env = make_carracing_env

