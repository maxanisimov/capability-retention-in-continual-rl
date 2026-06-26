"""Environment wrappers for scaled FrozenLake projects.safe_crl."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class CoordObsWrapper(gym.ObservationWrapper):
    """Convert discrete FrozenLake state index to normalized row, col, task features."""

    def __init__(self, env: gym.Env, task_num: float = 0.0):
        super().__init__(env)
        self.task_num = float(task_num)
        desc = env.unwrapped.desc
        self.nrow, self.ncol = desc.shape
        low = np.array([0.0, 0.0, -np.inf], dtype=np.float32)
        high = np.array([1.0, 1.0, np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs: int) -> np.ndarray:
        r = obs // self.ncol
        c = obs % self.ncol
        return np.array([r / (self.nrow - 1), c / (self.ncol - 1), self.task_num], dtype=np.float32)


class SafetyFlagWrapper(gym.Wrapper):
    """Populate ``info['safe']`` using FrozenLake hole locations."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.desc = env.unwrapped.desc
        self.nrow, self.ncol = self.desc.shape

    def _is_safe_state(self, state: int) -> bool:
        r = state // self.ncol
        c = state % self.ncol
        cell = self.desc[r, c]
        cell = cell.decode() if isinstance(cell, bytes) else cell
        return cell != "H"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["safe"] = self._is_safe_state(self.env.unwrapped.s)
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["safe"] = self._is_safe_state(self.env.unwrapped.s)
        return obs, reward, terminated, truncated, info


class DenseShapingWrapper(gym.Wrapper):
    """Dense reward shaping used only during training."""

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

    def _dist(self, s: int) -> int:
        r = s // self.ncol
        c = s % self.ncol
        return abs(self.goal[0] - r) + abs(self.goal[1] - c)

    def _cell(self, s: int) -> str:
        r = s // self.ncol
        c = s % self.ncol
        cell = self.desc[r, c]
        return cell.decode() if isinstance(cell, bytes) else str(cell)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_dist = self._dist(self.env.unwrapped.s)
        return obs, info

    def step(self, action: int):
        obs, reward, term, trunc, info = self.env.step(action)
        s = self.env.unwrapped.s
        cell = self._cell(s)
        curr = self._dist(s)
        if self.prev_dist is None:
            self.prev_dist = curr
        progress = float(self.prev_dist - curr)
        self.prev_dist = curr

        shaped = self.progress_scale * progress - self.step_penalty
        if term and cell == "H":
            shaped -= self.hole_penalty
        if trunc and cell != "G":
            shaped -= self.trunc_penalty
        if cell == "G":
            shaped += self.goal_bonus
        return obs, shaped, term, trunc, info

