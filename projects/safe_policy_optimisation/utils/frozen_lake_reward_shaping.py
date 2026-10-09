"""Opt-in, training-only potential shaping; never used by safe initialization."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv


def goal_distance_potential(desc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reverse BFS through non-hole grid cells, without consulting a shield.

    The goal, holes, and unreachable cells have potential zero. For every other
    reachable cell, Phi = 1 - distance / max_finite_distance. Distances describe
    grid geometry, not optimal expected hitting times under stochastic slipping.
    """
    grid = np.asarray(desc)
    if grid.ndim != 2 or not grid.size:
        raise ValueError("Expected a nonempty two-dimensional FrozenLake map.")
    grid = grid.astype("S1")
    goals = np.argwhere(grid == b"G")
    if len(goals) != 1:
        raise ValueError("Reward shaping requires exactly one goal.")
    distance = np.full(grid.shape, -1, dtype=np.int64)
    row, col = map(int, goals[0])
    distance[row, col] = 0
    queue = deque([(row, col)])
    rows, cols = grid.shape
    while queue:
        row, col = queue.popleft()
        for nxt_row, nxt_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if (
                0 <= nxt_row < rows
                and 0 <= nxt_col < cols
                and grid[nxt_row, nxt_col] != b"H"
                and distance[nxt_row, nxt_col] < 0
            ):
                distance[nxt_row, nxt_col] = distance[row, col] + 1
                queue.append((nxt_row, nxt_col))
    potential = np.zeros(grid.shape, dtype=np.float64)
    reachable = (distance >= 0) & (grid != b"G") & (grid != b"H")
    denominator = max(1, int(distance.max()))
    potential[reachable] = 1.0 - distance[reachable] / denominator
    return distance.reshape(-1), potential.reshape(-1)


class FrozenLakePotentialReward(gym.Wrapper):
    """Return raw_reward + scale * (gamma * Phi(next_state) - Phi(state)).

    Wrap only the *training* environment, outside its TimeLimit. Evaluation and
    any PSPO initial-actor construction must keep using the original environment.
    Costs, observations, actions, transition probabilities, and success flags are
    unchanged. The underlying transition/reward table is never mutated.

    ``bootstrap`` retains next-state potential on a timeout: SB3 PPO adds the
    discounted shaped terminal value, which cancels that remaining potential.
    ``terminal`` instead treats timeouts as true terminal transitions, sets their
    next potential to zero, and disables timeout bootstrapping by translating
    the flags to terminated=True, truncated=False. Both learner and shaping must
    use the same discount and timeout convention.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        gamma: float,
        scale: float = 1.0,
        timeout_mode: str = "bootstrap",
    ) -> None:
        if not isinstance(env.unwrapped, FrozenLakeEnv):
            raise TypeError("FrozenLakePotentialReward requires FrozenLakeEnv.")
        if not np.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("gamma must be finite and in (0, 1].")
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("scale must be finite and nonnegative.")
        if timeout_mode not in {"bootstrap", "terminal"}:
            raise ValueError("timeout_mode must be 'bootstrap' or 'terminal'.")
        super().__init__(env)
        self.gamma = float(gamma)
        self.scale = float(scale)
        self.timeout_mode = timeout_mode
        self.distances, self.potential = goal_distance_potential(env.unwrapped.desc)
        self.distances.setflags(write=False)
        self.potential.setflags(write=False)

    def step(self, action: Any):
        old_potential = float(self.potential[int(self.unwrapped.s)])
        obs, reward, terminated, truncated, info = self.env.step(action)
        original_truncated = bool(truncated)
        terminal_timeout = truncated and self.timeout_mode == "terminal"
        next_potential = (
            0.0
            if terminated or terminal_timeout
            else float(self.potential[int(self.unwrapped.s)])
        )
        shaping = self.scale * (self.gamma * next_potential - old_potential)
        shaped_reward = float(reward) + shaping
        if terminal_timeout:
            terminated, truncated = True, False
        info = dict(info)
        info["reward_shaping"] = {
            "raw_reward": float(reward),
            "shaped_reward": shaped_reward,
            "shaping_reward": shaping,
            "potential": old_potential,
            "next_potential": next_potential,
            "original_truncated": original_truncated,
        }
        return obs, shaped_reward, terminated, truncated, info
