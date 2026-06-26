"""Deterministic FrozenLake environment helpers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


ACTION_DELTAS = {
    0: (0, -1),   # left
    1: (1, 0),    # down
    2: (0, 1),    # right
    3: (-1, 0),   # up
}


def grid_shape(env_map: list[str] | tuple[str, ...]) -> tuple[int, int]:
    nrow = len(env_map)
    if nrow == 0:
        raise ValueError("FrozenLake map is empty.")
    ncol = len(env_map[0])
    if ncol == 0 or any(len(row) != ncol for row in env_map):
        raise ValueError("FrozenLake map must be rectangular.")
    return nrow, ncol


def state_index_to_obs(
    state_index: int,
    env_map: list[str] | tuple[str, ...],
    task_num: float,
) -> np.ndarray:
    nrow, ncol = grid_shape(env_map)
    row = state_index // ncol
    col = state_index % ncol
    return np.array(
        [
            row / (nrow - 1),
            col / (ncol - 1),
            float(task_num),
        ],
        dtype=np.float32,
    )


def obs_to_state_index(obs: np.ndarray, env_map: list[str] | tuple[str, ...]) -> int:
    nrow, ncol = grid_shape(env_map)
    row = int(round(float(obs[0]) * (nrow - 1)))
    col = int(round(float(obs[1]) * (ncol - 1)))
    return row * ncol + col


class CoordObsWrapper(gym.ObservationWrapper):
    """Convert discrete state index to normalized row, col, task features."""

    def __init__(self, env: gym.Env, env_map: list[str] | tuple[str, ...], task_num: float):
        super().__init__(env)
        self.env_map = tuple(env_map)
        self.task_num = float(task_num)
        low = np.array([0.0, 0.0, -np.inf], dtype=np.float32)
        high = np.array([1.0, 1.0, np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs: int) -> np.ndarray:
        return state_index_to_obs(int(obs), self.env_map, self.task_num)


class SafetyFlagWrapper(gym.Wrapper):
    """Populate safety and success fields used by evaluation/early stopping."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.desc = env.unwrapped.desc
        self.nrow, self.ncol = self.desc.shape

    def _cell(self, state: int) -> str:
        row = state // self.ncol
        col = state % self.ncol
        cell = self.desc[row, col]
        return cell.decode() if isinstance(cell, bytes) else str(cell)

    def _is_safe_state(self, state: int) -> bool:
        return self._cell(state) != "H"

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        state = int(self.env.unwrapped.s)
        info["safe"] = self._is_safe_state(state)
        info["is_success"] = self._cell(state) == "G"
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        state = int(self.env.unwrapped.s)
        info["safe"] = self._is_safe_state(state)
        info["is_success"] = self._cell(state) == "G"
        return obs, reward, terminated, truncated, info


class DenseShapingWrapper(gym.Wrapper):
    """Dense reward shaping used only for PPO training."""

    def __init__(
        self,
        env: gym.Env,
        progress_scale: float = 1.0,
        step_penalty: float = 0.05,
        hole_penalty: float = 2.0,
        trunc_penalty: float = 1.0,
        goal_bonus: float = 5.0,
    ):
        super().__init__(env)
        self.progress_scale = progress_scale
        self.step_penalty = step_penalty
        self.hole_penalty = hole_penalty
        self.trunc_penalty = trunc_penalty
        self.goal_bonus = goal_bonus
        self.desc = env.unwrapped.desc
        self.nrow, self.ncol = self.desc.shape
        self.goal = (self.nrow - 1, self.ncol - 1)
        self.prev_dist: int | None = None

    def _cell(self, state: int) -> str:
        row = state // self.ncol
        col = state % self.ncol
        cell = self.desc[row, col]
        return cell.decode() if isinstance(cell, bytes) else str(cell)

    def _dist(self, state: int) -> int:
        row = state // self.ncol
        col = state % self.ncol
        return abs(self.goal[0] - row) + abs(self.goal[1] - col)

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self.prev_dist = self._dist(int(self.env.unwrapped.s))
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = int(self.env.unwrapped.s)
        cell = self._cell(state)
        curr_dist = self._dist(state)
        if self.prev_dist is None:
            self.prev_dist = curr_dist
        progress = float(self.prev_dist - curr_dist)
        self.prev_dist = curr_dist

        shaped = self.progress_scale * progress - self.step_penalty
        if terminated and cell == "H":
            shaped -= self.hole_penalty
        if truncated and cell != "G":
            shaped -= self.trunc_penalty
        if cell == "G":
            shaped += self.goal_bonus
        return obs, shaped, terminated, truncated, info


def make_env(
    env_map: list[str] | tuple[str, ...],
    *,
    task_num: float,
    max_episode_steps: int,
    shaped: bool = False,
    is_slippery: bool = False,
    success_rate: float = 1.0 / 3.0,
    render_mode: str | None = None,
) -> gym.Env:
    env = gym.make(
        "FrozenLake-v1",
        desc=list(env_map),
        is_slippery=is_slippery,
        success_rate=success_rate,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )
    env = CoordObsWrapper(env, env_map, task_num)
    env = SafetyFlagWrapper(env)
    if shaped:
        env = DenseShapingWrapper(env)
    return env
