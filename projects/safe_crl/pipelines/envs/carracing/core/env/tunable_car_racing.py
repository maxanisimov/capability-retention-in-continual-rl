"""Tunable CarRacing environment for continual-learning task shifts."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.car_racing import FPS, PLAYFIELD, CarRacing
from gymnasium.envs.registration import register, registry
from gymnasium.error import InvalidAction


TUNABLE_CAR_RACING_V3_ID = "TunableCarRacing-v3"
TUNABLE_CAR_RACING_V3_ENTRY_POINT = (
    "projects.safe_crl.pipelines.envs.carracing.core.env.tunable_car_racing:TunableCarRacingEnv"
)


def _finite_float(name: str, value: Any) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return out


def _positive_int(name: str, value: Any) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}.")
    return out


class TunableCarRacingEnv(CarRacing):
    """CarRacing with constructor knobs for common task and reward shifts."""

    def __init__(
        self,
        render_mode: str | None = None,
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        continuous: bool = True,
        *,
        discrete_steer: float = 0.6,
        discrete_gas: float = 0.2,
        discrete_brake: float = 0.8,
        frame_dt: float = 1.0 / FPS,
        world_velocity_iterations: int = 6 * 30,
        world_position_iterations: int = 2 * 30,
        per_step_penalty: float = 0.1,
        off_track_penalty: float = -100.0,
        playfield: float = PLAYFIELD,
    ):
        self.discrete_steer = _finite_float("discrete_steer", discrete_steer)
        self.discrete_gas = _finite_float("discrete_gas", discrete_gas)
        self.discrete_brake = _finite_float("discrete_brake", discrete_brake)
        self.frame_dt = _finite_float("frame_dt", frame_dt)
        self.world_velocity_iterations = _positive_int(
            "world_velocity_iterations",
            world_velocity_iterations,
        )
        self.world_position_iterations = _positive_int(
            "world_position_iterations",
            world_position_iterations,
        )
        self.per_step_penalty = _finite_float("per_step_penalty", per_step_penalty)
        self.off_track_penalty = _finite_float("off_track_penalty", off_track_penalty)
        self.playfield = _finite_float("playfield", playfield)

        self._validate_tunable_parameters()
        super().__init__(
            render_mode=render_mode,
            verbose=verbose,
            lap_complete_percent=lap_complete_percent,
            domain_randomize=domain_randomize,
            continuous=continuous,
        )

    def _validate_tunable_parameters(self) -> None:
        for name in ("discrete_steer", "discrete_gas", "discrete_brake", "per_step_penalty"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}.")
        if self.frame_dt <= 0.0:
            raise ValueError(f"frame_dt must be > 0, got {self.frame_dt}.")
        if self.playfield <= 0.0:
            raise ValueError(f"playfield must be > 0, got {self.playfield}.")

    def step(self, action: np.ndarray | int):
        assert self.car is not None
        if action is not None:
            if self.continuous:
                action = action.astype(np.float64)
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`",
                    )
                self.car.steer(
                    -self.discrete_steer * (action == 1)
                    + self.discrete_steer * (action == 2),
                )
                self.car.gas(self.discrete_gas * (action == 3))
                self.car.brake(self.discrete_brake * (action == 4))

        self.car.step(self.frame_dt)
        self.world.Step(
            self.frame_dt,
            self.world_velocity_iterations,
            self.world_position_iterations,
        )
        self.t += self.frame_dt

        self.state = self._render("state_pixels")

        step_reward = 0
        terminated = False
        truncated = False
        info: dict[str, bool] = {}
        if action is not None:
            self.reward -= self.per_step_penalty
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self.new_lap:
                terminated = True
                info["lap_finished"] = True
            x, y = self.car.hull.position
            if abs(x) > self.playfield or abs(y) > self.playfield:
                terminated = True
                info["lap_finished"] = False
                step_reward = self.off_track_penalty

        info["is_success"] = bool(terminated and info.get("lap_finished", False))
        info["safe"] = not bool(terminated and info.get("lap_finished") is False)

        if self.render_mode == "human":
            self.render()
        return self.state, step_reward, terminated, truncated, info


def ensure_tunable_car_racing_registered() -> None:
    """Register TunableCarRacing-v3 to point at TunableCarRacingEnv."""
    existing = registry.get(TUNABLE_CAR_RACING_V3_ID)
    if (
        existing is not None
        and str(getattr(existing, "entry_point", "")) == TUNABLE_CAR_RACING_V3_ENTRY_POINT
    ):
        return

    if existing is not None:
        del registry[TUNABLE_CAR_RACING_V3_ID]

    register(
        id=TUNABLE_CAR_RACING_V3_ID,
        entry_point=TUNABLE_CAR_RACING_V3_ENTRY_POINT,
        max_episode_steps=1000,
        reward_threshold=900,
    )


ensure_tunable_car_racing_registered()

