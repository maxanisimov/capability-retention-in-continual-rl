"""Sweep PPO on diagonal FrozenLake source environments and store successful settings."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

from experiments.utils.ppo_utils import PPOConfig, evaluate, ppo_train


class CoordObsWrapper(gym.ObservationWrapper):
    """Convert discrete FrozenLake state index -> normalized (row, col, task)."""

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
    """Populate info['safe'] using FrozenLake hole locations."""

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
    """Dense reward shaping used only during training.

    Goal-reaching is still validated on the raw sparse-reward environment.
    """

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


def make_diagonal_source_map(size: int, corridor_half_width: int = 1) -> list[str]:
    rows: list[str] = []
    for r in range(size):
        row: list[str] = []
        for c in range(size):
            if r == 0 and c == 0:
                row.append("S")
            elif r == size - 1 and c == size - 1:
                row.append("G")
            elif abs(r - c) <= corridor_half_width:
                row.append("F")
            else:
                row.append("H")
        rows.append("".join(row))
    return rows


def make_env(
    size: int,
    corridor_half_width: int,
    shaped: bool,
    task_num: float = 0.0,
) -> gym.Env:
    env = gym.make(
        "FrozenLake-v1",
        desc=make_diagonal_source_map(size=size, corridor_half_width=corridor_half_width),
        is_slippery=False,
        max_episode_steps=4 * size,
    )
    env = CoordObsWrapper(env, task_num=task_num)
    env = SafetyFlagWrapper(env)
    if shaped:
        env = DenseShapingWrapper(env)
    return env


def schedule_for_size(size: int) -> tuple[int, int, int, int]:
    """(total_timesteps, hidden, rollout_steps, minibatch_size)."""
    total = 50_000 + (size - 10) * 5_000
    if size >= 80:
        total = max(total, 400_000)
    if size == 100:
        total = 500_000

    hidden = 128 if size <= 60 else 256
    rollout_steps = 1024 if size <= 50 else 2048
    minibatch_size = 256 if size <= 50 else 512
    return total, hidden, rollout_steps, minibatch_size


def train_and_eval(size: int, seed: int, corridor_half_width: int = 1) -> dict:
    total, hidden, rollout_steps, minibatch_size = schedule_for_size(size)

    env = make_env(size=size, corridor_half_width=corridor_half_width, shaped=True)
    obs_dim = env.observation_space.shape[0]

    actor = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 4),
    )
    critic = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1),
    )

    cfg = PPOConfig(
        seed=seed,
        total_timesteps=total,
        eval_episodes=5,
        rollout_steps=rollout_steps,
        update_epochs=10,
        minibatch_size=minibatch_size,
        gamma=0.995,
        gae_lambda=0.97,
        clip_coef=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        lr=1e-4,
        max_grad_norm=0.5,
        device="cpu",
        early_stop=False,
    )

    actor, critic, _ = ppo_train(env, cfg, actor, critic, return_training_data=True)  # type: ignore[assignment]

    raw_env = make_env(size=size, corridor_half_width=corridor_half_width, shaped=False)
    mean, std, failure_rate = evaluate(raw_env, actor, episodes=20, deterministic=True, device="cpu")

    success = float(mean) >= 1.0
    return {
        "grid_size": size,
        "corridor_half_width": corridor_half_width,
        "max_episode_steps": 4 * size,
        "success": bool(success),
        "raw_eval": {
            "episodes": 20,
            "mean_reward": float(mean),
            "std_reward": float(std),
            "failure_rate": float(failure_rate),
        },
        "ppo": {
            "seed": seed,
            "total_timesteps": total,
            "hidden": hidden,
            "rollout_steps": rollout_steps,
            "update_epochs": 10,
            "minibatch_size": minibatch_size,
            "gamma": 0.995,
            "gae_lambda": 0.97,
            "clip_coef": 0.2,
            "ent_coef": 0.005,
            "vf_coef": 0.5,
            "lr": 1e-4,
            "max_grad_norm": 0.5,
            "device": "cpu",
        },
        "training_wrappers": {
            "coordinate_observation": True,
            "dense_reward_shaping": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scaled diagonal FrozenLake PPO sweep.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size-min", type=int, default=10)
    parser.add_argument("--size-max", type=int, default=100)
    parser.add_argument("--size-step", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "settings" / "train_source_policy_settings.yaml",
    )
    args = parser.parse_args()

    sizes = list(range(args.size_min, args.size_max + 1, args.size_step))
    all_results: dict[str, dict] = {}
    for size in sizes:
        key = f"diagonal_{size}x{size}"
        print(f"\n=== Running {key} ===")
        entry = train_and_eval(size=size, seed=args.seed, corridor_half_width=1)
        all_results[key] = entry
        print(
            f"{key}: success={entry['success']} | "
            f"mean_reward={entry['raw_eval']['mean_reward']:.3f} | "
            f"failure_rate={entry['raw_eval']['failure_rate']:.3f}",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(all_results, sort_keys=False), encoding="utf-8")
    print(f"\nWrote successful settings to {args.output}")


if __name__ == "__main__":
    main()
