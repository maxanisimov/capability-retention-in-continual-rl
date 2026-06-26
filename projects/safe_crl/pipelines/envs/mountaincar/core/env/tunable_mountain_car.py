"""Tunable Mountain Car environment for continual-learning task shifts."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control import utils as classic_control_utils
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
from gymnasium.envs.registration import register, registry


TUNABLE_MOUNTAIN_CAR_V0_ID = "TunableMountainCar-v0"
TUNABLE_MOUNTAIN_CAR_V0_ENTRY_POINT = (
    "projects.safe_crl.pipelines.envs.mountaincar.core.env.tunable_mountain_car:TunableMountainCarEnv"
)


def _finite_float(name: str, value: Any) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return out


class TunableMountainCarEnv(MountainCarEnv):
    """MountainCar with explicit dynamics, bounds, and reset-range knobs."""

    def __init__(
        self,
        render_mode: str | None = None,
        goal_velocity: float = 0.0,
        *,
        force: float = 0.001,
        gravity: float = 0.0025,
        goal_position: float = 0.5,
        min_position: float = -1.2,
        max_position: float = 0.6,
        max_speed: float = 0.07,
        reset_low: float = -0.6,
        reset_high: float = -0.4,
    ):
        super().__init__(render_mode=render_mode, goal_velocity=goal_velocity)

        self.force = _finite_float("force", force)
        self.gravity = _finite_float("gravity", gravity)
        self.goal_position = _finite_float("goal_position", goal_position)
        self.goal_velocity = _finite_float("goal_velocity", goal_velocity)
        self.min_position = _finite_float("min_position", min_position)
        self.max_position = _finite_float("max_position", max_position)
        self.max_speed = _finite_float("max_speed", max_speed)
        self.reset_low = _finite_float("reset_low", reset_low)
        self.reset_high = _finite_float("reset_high", reset_high)

        self._validate_tunable_parameters()
        self._rebuild_observation_space()

    def _validate_tunable_parameters(self) -> None:
        if self.force < 0.0:
            raise ValueError(f"force must be >= 0, got {self.force}.")
        if self.gravity < 0.0:
            raise ValueError(f"gravity must be >= 0, got {self.gravity}.")
        if self.min_position >= self.max_position:
            raise ValueError(
                "min_position must be lower than max_position, "
                f"got {self.min_position} >= {self.max_position}.",
            )
        if self.max_speed <= 0.0:
            raise ValueError(f"max_speed must be > 0, got {self.max_speed}.")
        if not (self.min_position <= self.goal_position <= self.max_position):
            raise ValueError(
                "goal_position must be within [min_position, max_position], "
                f"got {self.goal_position} outside [{self.min_position}, {self.max_position}].",
            )
        if self.reset_low > self.reset_high:
            raise ValueError(
                f"reset_low must be <= reset_high, got {self.reset_low} > {self.reset_high}.",
            )

    def _rebuild_observation_space(self) -> None:
        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)
        self.observation_space = gym.spaces.Box(self.low, self.high, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        gym.Env.reset(self, seed=seed)
        low, high = classic_control_utils.maybe_parse_reset_bounds(
            options,
            self.reset_low,
            self.reset_high,
        )
        self.state = np.array([self.np_random.uniform(low=low, high=high), 0])

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        step_info = dict(info)
        step_info["is_success"] = bool(terminated)
        step_info["safe"] = True
        return obs, reward, terminated, truncated, step_info


def ensure_tunable_mountain_car_registered() -> None:
    """Register TunableMountainCar-v0 to point at TunableMountainCarEnv."""
    existing = registry.get(TUNABLE_MOUNTAIN_CAR_V0_ID)
    if (
        existing is not None
        and str(getattr(existing, "entry_point", "")) == TUNABLE_MOUNTAIN_CAR_V0_ENTRY_POINT
    ):
        return

    if existing is not None:
        del registry[TUNABLE_MOUNTAIN_CAR_V0_ID]

    register(
        id=TUNABLE_MOUNTAIN_CAR_V0_ID,
        entry_point=TUNABLE_MOUNTAIN_CAR_V0_ENTRY_POINT,
        max_episode_steps=200,
    )


ensure_tunable_mountain_car_registered()

