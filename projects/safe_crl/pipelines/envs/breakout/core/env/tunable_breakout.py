"""Tunable ALE Breakout environment for continual-learning task shifts."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from ale_py.env import AtariEnv
from gymnasium.envs.registration import register, registry


TUNABLE_BREAKOUT_V5_ID = "TunableBreakout-v5"
TUNABLE_ALE_BREAKOUT_V5_ID = "TunableALE/Breakout-v5"
TUNABLE_BREAKOUT_V5_IDS = (TUNABLE_BREAKOUT_V5_ID, TUNABLE_ALE_BREAKOUT_V5_ID)
TUNABLE_BREAKOUT_V5_ENTRY_POINT = (
    "projects.safe_crl.pipelines.envs.breakout.core.env.tunable_breakout:TunableBreakoutEnv"
)


def _finite_float(name: str, value: Any) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return out


def _normalize_frameskip(frameskip: tuple[int, int] | list[int] | int) -> tuple[int, int] | int:
    if isinstance(frameskip, bool):
        raise ValueError("frameskip must be a positive int or a length-2 int tuple.")
    if isinstance(frameskip, int):
        if frameskip <= 0:
            raise ValueError(f"frameskip must be > 0, got {frameskip}.")
        return frameskip
    if isinstance(frameskip, (list, tuple)):
        if len(frameskip) != 2:
            raise ValueError(f"frameskip must contain 2 values, got {len(frameskip)}.")
        low, high = int(frameskip[0]), int(frameskip[1])
        if low <= 0:
            raise ValueError(f"frameskip lower bound must be > 0, got {low}.")
        if low > high:
            raise ValueError(
                f"frameskip lower bound must be <= upper bound, got {low} > {high}.",
            )
        return (low, high)
    raise ValueError("frameskip must be a positive int or a length-2 int tuple.")


def _normalize_optional_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be >= 0 when provided, got {value!r}.")
    return out


class TunableBreakoutEnv(AtariEnv):
    """ALE Breakout-v5 with explicit constructor knobs for stochasticity and modes."""

    def __init__(
        self,
        render_mode: str | None = None,
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
    ):
        self.game = "breakout"
        self.mode = _normalize_optional_int("mode", mode)
        self.difficulty = _normalize_optional_int("difficulty", difficulty)
        self.obs_type = str(obs_type)
        self.frameskip = _normalize_frameskip(frameskip)
        self.repeat_action_probability = _finite_float(
            "repeat_action_probability",
            repeat_action_probability,
        )
        self.full_action_space = bool(full_action_space)
        self.continuous_action_threshold = _finite_float(
            "continuous_action_threshold",
            continuous_action_threshold,
        )
        self.max_num_frames_per_episode = _normalize_optional_int(
            "max_num_frames_per_episode",
            max_num_frames_per_episode,
        )
        self.sound_obs = bool(sound_obs)

        self._validate_tunable_parameters()
        super().__init__(
            game=self.game,
            mode=self.mode,
            difficulty=self.difficulty,
            obs_type=self.obs_type,
            frameskip=self.frameskip,
            repeat_action_probability=self.repeat_action_probability,
            full_action_space=self.full_action_space,
            continuous=bool(continuous),
            continuous_action_threshold=self.continuous_action_threshold,
            max_num_frames_per_episode=self.max_num_frames_per_episode,
            render_mode=render_mode,
            sound_obs=self.sound_obs,
        )

    def _validate_tunable_parameters(self) -> None:
        if self.obs_type not in {"rgb", "grayscale", "ram"}:
            raise ValueError("obs_type must be one of 'rgb', 'grayscale', or 'ram'.")
        if not (0.0 <= self.repeat_action_probability <= 1.0):
            raise ValueError(
                "repeat_action_probability must be in [0, 1], "
                f"got {self.repeat_action_probability}.",
            )
        if self.continuous_action_threshold < 0.0:
            raise ValueError(
                "continuous_action_threshold must be >= 0, "
                f"got {self.continuous_action_threshold}.",
            )
        if (
            self.max_num_frames_per_episode is not None
            and self.max_num_frames_per_episode <= 0
        ):
            raise ValueError(
                "max_num_frames_per_episode must be > 0 when provided, "
                f"got {self.max_num_frames_per_episode}.",
            )

    def step(self, action: int | np.ndarray):
        obs, reward, terminated, truncated, info = super().step(action)
        step_info = dict(info)
        step_info["safe"] = True
        step_info["is_success"] = False
        return obs, reward, terminated, truncated, step_info


def ensure_tunable_breakout_registered() -> None:
    """Register TunableBreakout-v5 to point at TunableBreakoutEnv."""
    for env_id in TUNABLE_BREAKOUT_V5_IDS:
        existing = registry.get(env_id)
        if (
            existing is not None
            and str(getattr(existing, "entry_point", "")) == TUNABLE_BREAKOUT_V5_ENTRY_POINT
        ):
            continue

        if existing is not None:
            del registry[env_id]

        register(
            id=env_id,
            entry_point=TUNABLE_BREAKOUT_V5_ENTRY_POINT,
        )


ensure_tunable_breakout_registered()
