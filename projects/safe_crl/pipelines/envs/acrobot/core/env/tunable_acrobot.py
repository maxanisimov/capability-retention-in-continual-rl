"""Tunable Acrobot environment for continual-learning task shifts."""

from __future__ import annotations

from math import cos, pi, sin
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control import utils as classic_control_utils
from gymnasium.envs.classic_control.acrobot import AcrobotEnv
from gymnasium.envs.registration import register, registry


TUNABLE_ACROBOT_V1_ID = "TunableAcrobot-v1"
TUNABLE_ACROBOT_V1_ENTRY_POINT = (
    "projects.safe_crl.pipelines.envs.acrobot.core.env.tunable_acrobot:TunableAcrobotEnv"
)


def _finite_float(name: str, value: Any) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return out


def _finite_float_tuple(name: str, value: Sequence[float] | np.ndarray) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a sequence of finite numbers.")
    out = tuple(_finite_float(f"{name}[{idx}]", item) for idx, item in enumerate(value))
    if not out:
        raise ValueError(f"{name} must contain at least one value.")
    return out


def _normalize_initial_state(
    initial_state: Sequence[float] | np.ndarray | None,
) -> tuple[float, float, float, float] | None:
    if initial_state is None:
        return None
    values = _finite_float_tuple("initial_state", initial_state)
    if len(values) != 4:
        raise ValueError(f"initial_state must contain exactly 4 values, got {len(values)}.")
    return (values[0], values[1], values[2], values[3])


class TunableAcrobotEnv(AcrobotEnv):
    """Acrobot with constructor-level control over transition dynamics and resets."""

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        gravity: float = 9.8,
        link_length_1: float = 1.0,
        link_length_2: float = 1.0,
        link_mass_1: float = 1.0,
        link_mass_2: float = 1.0,
        link_com_pos_1: float = 0.5,
        link_com_pos_2: float = 0.5,
        link_moi: float = 1.0,
        max_vel_1: float = 4.0 * pi,
        max_vel_2: float = 9.0 * pi,
        available_torque: Sequence[float] | np.ndarray = (-1.0, 0.0, 1.0),
        torque_noise_max: float = 0.0,
        dt: float = 0.2,
        book_or_nips: str = "book",
        terminal_height: float = 1.0,
        reset_low: float = -0.1,
        reset_high: float = 0.1,
        initial_state: Sequence[float] | np.ndarray | None = None,
    ):
        super().__init__(render_mode=render_mode)

        self.gravity = _finite_float("gravity", gravity)
        self.LINK_LENGTH_1 = _finite_float("link_length_1", link_length_1)
        self.LINK_LENGTH_2 = _finite_float("link_length_2", link_length_2)
        self.LINK_MASS_1 = _finite_float("link_mass_1", link_mass_1)
        self.LINK_MASS_2 = _finite_float("link_mass_2", link_mass_2)
        self.LINK_COM_POS_1 = _finite_float("link_com_pos_1", link_com_pos_1)
        self.LINK_COM_POS_2 = _finite_float("link_com_pos_2", link_com_pos_2)
        self.LINK_MOI = _finite_float("link_moi", link_moi)
        self.MAX_VEL_1 = _finite_float("max_vel_1", max_vel_1)
        self.MAX_VEL_2 = _finite_float("max_vel_2", max_vel_2)
        self.AVAIL_TORQUE = list(_finite_float_tuple("available_torque", available_torque))
        self.torque_noise_max = _finite_float("torque_noise_max", torque_noise_max)
        self.dt = _finite_float("dt", dt)
        self.book_or_nips = str(book_or_nips)
        self.terminal_height = _finite_float("terminal_height", terminal_height)
        self.reset_low = _finite_float("reset_low", reset_low)
        self.reset_high = _finite_float("reset_high", reset_high)
        self.initial_state = _normalize_initial_state(initial_state)

        self._validate_tunable_parameters()
        self._rebuild_spaces()

    def _validate_tunable_parameters(self) -> None:
        for name in (
            "gravity",
            "LINK_LENGTH_1",
            "LINK_LENGTH_2",
            "LINK_MASS_1",
            "LINK_MASS_2",
            "LINK_MOI",
            "MAX_VEL_1",
            "MAX_VEL_2",
            "dt",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}.")
        for name in ("LINK_COM_POS_1", "LINK_COM_POS_2", "torque_noise_max"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}.")
        if self.book_or_nips not in {"book", "nips"}:
            raise ValueError("book_or_nips must be either 'book' or 'nips'.")
        if self.reset_low > self.reset_high:
            raise ValueError(
                f"reset_low must be <= reset_high, got {self.reset_low} > {self.reset_high}.",
            )

    def _rebuild_spaces(self) -> None:
        high = np.array(
            [1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self.AVAIL_TORQUE))

    def _dsdt(self, s_augmented):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = self.gravity
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        d2 = m2 * (lc2**2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
            + phi2
        )
        if self.book_or_nips == "nips":
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2**2 + I2 - d2**2 / d1)
        else:
            ddtheta2 = (
                a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * sin(theta2) - phi2
            ) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0

    def _terminal(self):
        assert self.state is not None, "Call reset before using AcrobotEnv object."
        return bool(-cos(self.state[0]) - cos(self.state[1] + self.state[0]) > self.terminal_height)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        gym.Env.reset(self, seed=seed)
        reset_options = dict(options or {})
        option_initial_state = reset_options.pop("initial_state", self.initial_state)

        initial_state = _normalize_initial_state(option_initial_state)
        if initial_state is None:
            low, high = classic_control_utils.maybe_parse_reset_bounds(
                reset_options if reset_options else None,
                self.reset_low,
                self.reset_high,
            )
            self.state = self.np_random.uniform(low=low, high=high, size=(4,)).astype(
                np.float32,
            )
        else:
            self.state = np.asarray(initial_state, dtype=np.float32)

        if self.render_mode == "human":
            self.render()

        info = {"initial_state": tuple(float(x) for x in self.state)}
        return self._get_ob(), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        step_info = dict(info)
        step_info["is_success"] = bool(terminated)
        step_info["safe"] = True
        return obs, reward, terminated, truncated, step_info


def ensure_tunable_acrobot_registered() -> None:
    """Register TunableAcrobot-v1 to point at TunableAcrobotEnv."""
    existing = registry.get(TUNABLE_ACROBOT_V1_ID)
    if (
        existing is not None
        and str(getattr(existing, "entry_point", "")) == TUNABLE_ACROBOT_V1_ENTRY_POINT
    ):
        return

    if existing is not None:
        del registry[TUNABLE_ACROBOT_V1_ID]

    register(
        id=TUNABLE_ACROBOT_V1_ID,
        entry_point=TUNABLE_ACROBOT_V1_ENTRY_POINT,
        max_episode_steps=500,
        reward_threshold=-100.0,
    )


ensure_tunable_acrobot_registered()

