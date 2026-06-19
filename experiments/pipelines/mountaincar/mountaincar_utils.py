from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import numpy as np
from PIL import Image


ENV_ID = "MountainCar-v0"
GOAL_BONUS = 100.0
STEP_REWARD_SCALE = 1.0


class MountainCarShapedReward(gym.Wrapper):
    """Reward shaping for training only; evaluation uses the original env."""

    def __init__(self, env: gym.Env, gamma: float = 0.99) -> None:
        super().__init__(env)
        self.gamma = gamma
        self.previous_potential = 0.0

    @staticmethod
    def potential(obs: np.ndarray) -> float:
        position = float(obs[0])
        velocity = float(obs[1])
        return 10.0 * position + 100.0 * abs(velocity)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.previous_potential = self.potential(obs)
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_potential = self.potential(obs)
        potential_delta = self.gamma * current_potential - self.previous_potential
        self.previous_potential = current_potential

        goal_position = float(self.unwrapped.goal_position)
        goal_bonus = GOAL_BONUS if float(obs[0]) >= goal_position else 0.0
        shaped_reward = float(STEP_REWARD_SCALE * reward + potential_delta + goal_bonus)
        return obs, shaped_reward, terminated, truncated, info


@dataclass
class EvalStats:
    episodes: int
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    mean_length: float
    min_length: int
    max_length: int
    successes: int
    success_rate: float
    returns: list[float]
    lengths: list[int]


def make_env(
    seed: int | None = None,
    *,
    shaped: bool = False,
    render_mode: str | None = None,
) -> gym.Env:
    env = gym.make(ENV_ID, render_mode=render_mode)
    if shaped:
        env = MountainCarShapedReward(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def as_scalar_action(action: Any) -> int:
    if isinstance(action, np.ndarray):
        return int(action.item())
    return int(action)


def evaluate_policy(
    model: Any,
    *,
    episodes: int,
    seed: int,
    deterministic: bool = True,
) -> EvalStats:
    env = make_env(seed=seed, shaped=False)
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    goal_position = float(env.unwrapped.goal_position)

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            episode_length = 0
            reached_goal = False

            while True:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(
                    as_scalar_action(action)
                )
                episode_return += float(reward)
                episode_length += 1
                reached_goal = reached_goal or float(obs[0]) >= goal_position

                if terminated or truncated:
                    break

            returns.append(episode_return)
            lengths.append(episode_length)
            successes += int(reached_goal)
    finally:
        env.close()

    returns_array = np.asarray(returns, dtype=np.float64)
    lengths_array = np.asarray(lengths, dtype=np.int64)
    return EvalStats(
        episodes=episodes,
        mean_return=float(returns_array.mean()),
        std_return=float(returns_array.std()),
        min_return=float(returns_array.min()),
        max_return=float(returns_array.max()),
        mean_length=float(lengths_array.mean()),
        min_length=int(lengths_array.min()),
        max_length=int(lengths_array.max()),
        successes=successes,
        success_rate=float(successes / episodes),
        returns=[float(value) for value in returns],
        lengths=[int(value) for value in lengths],
    )


def rollout_once(
    model: Any,
    *,
    seed: int,
    deterministic: bool = True,
) -> dict[str, Any]:
    env = make_env(seed=seed, shaped=False)
    goal_position = float(env.unwrapped.goal_position)
    episode_return = 0.0
    episode_length = 0
    reached_goal = False

    try:
        obs, _ = env.reset(seed=seed)
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(as_scalar_action(action))
            episode_return += float(reward)
            episode_length += 1
            reached_goal = reached_goal or float(obs[0]) >= goal_position
            if terminated or truncated:
                break
    finally:
        env.close()

    return {
        "seed": seed,
        "episode_return": episode_return,
        "episode_length": episode_length,
        "reached_goal": reached_goal,
    }


def select_render_seed(
    model: Any,
    *,
    seed: int,
    search_episodes: int,
    deterministic: bool = True,
) -> tuple[int, dict[str, Any]]:
    best_summary = rollout_once(model, seed=seed, deterministic=deterministic)
    if best_summary["reached_goal"]:
        return seed, best_summary

    best_seed = seed
    for offset in range(1, search_episodes + 1):
        candidate_seed = seed + offset
        summary = rollout_once(
            model, seed=candidate_seed, deterministic=deterministic
        )
        if summary["reached_goal"]:
            return candidate_seed, summary
        if summary["episode_return"] > best_summary["episode_return"]:
            best_seed = candidate_seed
            best_summary = summary

    return best_seed, best_summary


def save_policy_frames(
    model: Any,
    *,
    frames_dir: Path,
    gif_path: Path,
    seed: int,
    render_seed_search: int = 50,
    frame_limit: int = 200,
    gif_duration_ms: int = 40,
    deterministic: bool = True,
) -> dict[str, Any]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame_*.png"):
        old_frame.unlink()

    selected_seed, pre_render_summary = select_render_seed(
        model,
        seed=seed,
        search_episodes=render_seed_search,
        deterministic=deterministic,
    )

    env = make_env(seed=selected_seed, shaped=False, render_mode="rgb_array")
    frames: list[Image.Image] = []
    episode_return = 0.0
    episode_length = 0
    reached_goal = False
    goal_position = float(env.unwrapped.goal_position)

    try:
        obs, _ = env.reset(seed=selected_seed)
        for step_index in range(frame_limit):
            frame = env.render()
            image = Image.fromarray(frame)
            frame_path = frames_dir / f"frame_{step_index:04d}.png"
            image.save(frame_path)
            frames.append(image.copy())

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(as_scalar_action(action))
            episode_return += float(reward)
            episode_length += 1
            reached_goal = reached_goal or float(obs[0]) >= goal_position

            if terminated or truncated:
                if len(frames) < frame_limit:
                    final_frame = Image.fromarray(env.render())
                    final_path = frames_dir / f"frame_{step_index + 1:04d}.png"
                    final_frame.save(final_path)
                    frames.append(final_frame.copy())
                break
    finally:
        env.close()

    if not frames:
        raise RuntimeError("No frames were rendered.")

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=gif_duration_ms,
        loop=0,
    )

    return {
        "frames_dir": str(frames_dir),
        "gif_path": str(gif_path),
        "requested_seed": seed,
        "selected_seed": selected_seed,
        "pre_render_summary": pre_render_summary,
        "frame_count": len(frames),
        "episode_return": episode_return,
        "episode_length": episode_length,
        "reached_goal": reached_goal,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def eval_stats_to_dict(stats: EvalStats) -> dict[str, Any]:
    return asdict(stats)
