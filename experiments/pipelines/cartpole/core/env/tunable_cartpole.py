"""Tunable CartPole environment for continual-learning task shifts."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control import utils as classic_control_utils
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.registration import register, registry


TUNABLE_CARTPOLE_V1_ID = "TunableCartPole-v1"
TUNABLE_CARTPOLE_V1_ENTRY_POINT = (
    "experiments.pipelines.cartpole.core.env.tunable_cartpole:TunableCartPoleEnv"
)


def _finite_float(name: str, value: Any) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return out


class TunableCartPoleEnv(CartPoleEnv):
    """CartPole with explicit constructor knobs for physics, bounds, and resets."""

    def __init__(
        self,
        sutton_barto_reward: bool = False,
        render_mode: str | None = None,
        *,
        gravity: float = 9.8,
        masscart: float = 1.0,
        masspole: float = 0.1,
        length: float = 0.5,
        force_mag: float = 10.0,
        tau: float = 0.02,
        theta_threshold_radians: float = 12 * 2 * math.pi / 360,
        x_threshold: float = 2.4,
        kinematics_integrator: str = "euler",
        reset_low: float = -0.05,
        reset_high: float = 0.05,
    ):
        super().__init__(
            sutton_barto_reward=sutton_barto_reward,
            render_mode=render_mode,
        )
        self.gravity = _finite_float("gravity", gravity)
        self.masscart = _finite_float("masscart", masscart)
        self.masspole = _finite_float("masspole", masspole)
        self.length = _finite_float("length", length)
        self.force_mag = _finite_float("force_mag", force_mag)
        self.tau = _finite_float("tau", tau)
        self.theta_threshold_radians = _finite_float(
            "theta_threshold_radians",
            theta_threshold_radians,
        )
        self.x_threshold = _finite_float("x_threshold", x_threshold)
        self.kinematics_integrator = str(kinematics_integrator)
        self.reset_low = _finite_float("reset_low", reset_low)
        self.reset_high = _finite_float("reset_high", reset_high)

        self._validate_tunable_parameters()
        self._rebuild_dependent_parameters()
        self._rebuild_observation_space()

    def _validate_tunable_parameters(self) -> None:
        for name in (
            "gravity",
            "masscart",
            "masspole",
            "length",
            "force_mag",
            "tau",
            "theta_threshold_radians",
            "x_threshold",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}.")
        if self.kinematics_integrator not in {"euler", "semi-implicit euler"}:
            raise ValueError(
                "kinematics_integrator must be 'euler' or 'semi-implicit euler', "
                f"got {self.kinematics_integrator!r}.",
            )
        if self.reset_low > self.reset_high:
            raise ValueError(
                f"reset_low must be <= reset_high, got {self.reset_low} > {self.reset_high}.",
            )

    def _rebuild_dependent_parameters(self) -> None:
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length

    def _rebuild_observation_space(self) -> None:
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

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
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        step_info = dict(info)
        step_info["failure"] = bool(terminated)
        step_info["safe"] = not bool(terminated)
        return obs, reward, terminated, truncated, step_info


def ensure_tunable_cartpole_registered() -> None:
    """Register TunableCartPole-v1 to point at TunableCartPoleEnv."""
    existing = registry.get(TUNABLE_CARTPOLE_V1_ID)
    if (
        existing is not None
        and str(getattr(existing, "entry_point", "")) == TUNABLE_CARTPOLE_V1_ENTRY_POINT
    ):
        return

    if existing is not None:
        del registry[TUNABLE_CARTPOLE_V1_ID]

    register(
        id=TUNABLE_CARTPOLE_V1_ID,
        entry_point=TUNABLE_CARTPOLE_V1_ENTRY_POINT,
        max_episode_steps=500,
        reward_threshold=475.0,
    )


ensure_tunable_cartpole_registered()

