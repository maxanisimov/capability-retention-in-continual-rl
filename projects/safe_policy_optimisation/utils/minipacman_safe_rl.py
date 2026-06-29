"""Helpers for safe-RL baseline experiments on MASA-style tabular tasks."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym
import numpy as np
import torch as th
from PIL import Image
from safe_rl_baselines import CPO, PPOLagrangian, PPOPIDLagrangian

from projects.safe_crl.utils.masa_tabular_envs.factory import make_custom_masa_env

PPO_LAGRANGIAN_ALGORITHM_NAMES = ("ppo_lagrangian", "ppo_pid_lagrangian")
CPO_ALGORITHM_NAMES = ("cpo",)
ALGORITHM_NAMES = (*PPO_LAGRANGIAN_ALGORITHM_NAMES, *CPO_ALGORITHM_NAMES)
DEFAULT_TOTAL_TIMESTEPS = {
    "ppo_lagrangian": 10_000,
    "ppo_pid_lagrangian": 10_000,
    "cpo": 12_000,
}
SAFE_RL_BASELINE_HYPERPARAMS = (
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "cost_gamma",
    "cost_gae_lambda",
    "lagrangian_multiplier_init",
)
DEFAULT_SAFE_RL_BASELINE_HYPERPARAMS: dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "cost_gamma": 0.99,
    "cost_gae_lambda": 0.95,
    "lagrangian_multiplier_init": 0.0,
}


@dataclass(frozen=True)
class EpisodeMetrics:
    """Metrics collected for one completed evaluation episode."""

    episode: int
    reward: float
    cost: float
    length: int
    violated: bool
    unsafe_state_visit_count: int = 0
    safe_trajectory: bool = True


def make_safe_rl_env(
    env_id: str = "CustomMiniPacman-v0",
    *,
    max_episode_steps: int | None = 100,
    env_kwargs: dict[str, Any] | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a local MASA-style tabular environment for baseline training."""

    return make_custom_masa_env(
        env_id,
        max_episode_steps=max_episode_steps,
        env_kwargs=env_kwargs or {},
        render_mode=render_mode,
    )


def make_minipacman_env(
    *,
    max_episode_steps: int = 100,
    ghost_rand_prob: float = 0.0,
    render_mode: str | None = None,
) -> gym.Env:
    """Create the local MASA-style MiniPacman environment."""

    return make_safe_rl_env(
        "CustomMiniPacman-v0",
        max_episode_steps=max_episode_steps,
        env_kwargs={"ghost_rand_prob": ghost_rand_prob},
        render_mode=render_mode,
    )


def state_cost(env: gym.Env, obs: Any, info: dict[str, Any] | None = None) -> float:
    """Return the per-state safety cost for a MASA-style tabular state."""

    if info is not None and "cost" in info:
        return float(info["cost"])
    unwrapped = env.unwrapped
    if not (hasattr(unwrapped, "label_fn") and hasattr(unwrapped, "cost_fn")):
        return 0.0
    state = int(np.asarray(obs).item())
    return float(unwrapped.cost_fn(unwrapped.label_fn(state)))


def minipacman_state_cost(env: gym.Env, obs: Any) -> float:
    """Return the MASA label-derived safety cost for a MiniPacman state."""

    return state_cost(env, obs)


def make_state_cost_fn(env: gym.Env) -> Callable[..., float]:
    """Adapt a MASA-style state-label cost API to the safe-baseline callback."""

    def cost_fn(
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> float:
        del obs, action, reward, terminated, truncated
        return state_cost(env, next_obs, info)

    return cost_fn


def make_minipacman_cost_fn(env: gym.Env) -> Callable[..., float]:
    """Adapt MiniPacman's state-label cost API to the safe-baseline callback."""

    return make_state_cost_fn(env)


def build_safe_rl_baseline(
    algorithm: str,
    env: gym.Env,
    *,
    cost_limit: float,
    seed: int,
    device: str = "cpu",
    **hyperparameters: Any,
) -> Any:
    """Build one safe-RL baseline with smoke-friendly tabular defaults."""

    if algorithm not in ALGORITHM_NAMES:
        raise ValueError(f"Unknown algorithm {algorithm!r}. Expected one of {ALGORITHM_NAMES}.")

    baseline_hyperparameters = dict(DEFAULT_SAFE_RL_BASELINE_HYPERPARAMS)
    baseline_hyperparameters.update(
        {
            key: value
            for key, value in hyperparameters.items()
            if key in SAFE_RL_BASELINE_HYPERPARAMS and value is not None
        }
    )
    common: dict[str, Any] = {
        "cost_fn": make_state_cost_fn(env),
        "cost_limit": cost_limit,
        "net_arch": (64, 64),
        "seed": seed,
        "device": device,
        **baseline_hyperparameters,
    }
    if algorithm == "ppo_lagrangian":
        return PPOLagrangian(
            env,
            lambda_lr=0.1,
            **common,
        )
    if algorithm == "ppo_pid_lagrangian":
        return PPOPIDLagrangian(
            env,
            **common,
        )
    return CPO(
        env,
        target_kl=0.02,
        cg_iters=10,
        n_critic_updates=20,
        **common,
    )


def evaluate_policy(
    model: Any,
    env: gym.Env,
    *,
    cost_limit: float,
    episodes: int,
    seed: int,
    deterministic: bool = True,
) -> list[EpisodeMetrics]:
    """Evaluate a policy and count cost-budget violations per episode."""

    metrics: list[EpisodeMetrics] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        initial_cost = state_cost(env, obs)
        total_cost = initial_cost
        unsafe_state_visit_count = int(initial_cost > 0.0)
        length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            action_int = int(np.asarray(action).item())
            obs, reward, terminated, truncated, _info = env.step(action_int)
            step_cost = state_cost(env, obs, _info)
            total_reward += float(reward)
            total_cost += step_cost
            unsafe_state_visit_count += int(step_cost > 0.0)
            length += 1
            done = bool(terminated or truncated)
        metrics.append(
            EpisodeMetrics(
                episode=episode,
                reward=total_reward,
                cost=total_cost,
                length=length,
                violated=total_cost > cost_limit,
                unsafe_state_visit_count=unsafe_state_visit_count,
                safe_trajectory=unsafe_state_visit_count == 0,
            )
        )
    return metrics


def aggregate_violations(episodes: Iterable[EpisodeMetrics]) -> dict[str, float | int]:
    """Aggregate cost-constraint violations as count and percentage."""

    episode_list = list(episodes)
    total = len(episode_list)
    violation_count = sum(int(ep.violated) for ep in episode_list)
    safe_trajectory_count = sum(int(ep.safe_trajectory) for ep in episode_list)
    violation_percentage = 100.0 * violation_count / total if total else 0.0
    mean_cost = float(np.mean([ep.cost for ep in episode_list])) if episode_list else 0.0
    mean_reward = float(np.mean([ep.reward for ep in episode_list])) if episode_list else 0.0
    mean_length = float(np.mean([ep.length for ep in episode_list])) if episode_list else 0.0
    unsafe_state_visits = sum(int(ep.unsafe_state_visit_count) for ep in episode_list)
    return {
        "episodes": total,
        "violation_count": violation_count,
        "violation_percentage": violation_percentage,
        "safe_trajectory_count": safe_trajectory_count,
        "safety_rate": float(safe_trajectory_count / total) if total else 0.0,
        "unsafe_state_visit_count": unsafe_state_visits,
        "mean_episode_cost": mean_cost,
        "mean_episode_reward": mean_reward,
        "mean_episode_length": mean_length,
    }


def aggregate_training_violations(records: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    """Aggregate completed exploration episodes collected during training."""

    record_list = list(records)
    total = len(record_list)
    violation_count = sum(int(record["violated"]) for record in record_list)
    safe_trajectory_count = sum(
        int(
            bool(
                record.get(
                    "safe_trajectory",
                    int(record.get("unsafe_state_visit_count", 1 if float(record.get("cost", 0.0)) > 0.0 else 0))
                    == 0,
                )
            )
        )
        for record in record_list
    )
    unsafe_state_visits = sum(
        int(record.get("unsafe_state_visit_count", 1 if float(record.get("cost", 0.0)) > 0.0 else 0))
        for record in record_list
    )
    mean_cost = float(np.mean([float(record["cost"]) for record in record_list])) if record_list else 0.0
    mean_reward = float(np.mean([float(record["reward"]) for record in record_list])) if record_list else 0.0
    mean_length = float(np.mean([int(record["length"]) for record in record_list])) if record_list else 0.0
    return {
        "training_episode_count": total,
        "training_violation_count": violation_count,
        "training_violation_percentage": 100.0 * violation_count / total if total else 0.0,
        "training_safe_trajectory_count": safe_trajectory_count,
        "training_safety_rate": float(safe_trajectory_count / total) if total else 0.0,
        "training_unsafe_state_visit_count": unsafe_state_visits,
        "training_mean_episode_cost": mean_cost,
        "training_mean_episode_reward": mean_reward,
        "training_mean_episode_length": mean_length,
    }


def write_json(path: Path, payload: Any) -> None:
    """Write a stable, human-readable JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_episode_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write per-episode evaluation rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = [
        "algorithm",
        "episode",
        "reward",
        "cost",
        "length",
        "violated",
        "unsafe_state_visit_count",
        "safe_trajectory",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_training_episode_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write per-episode training exploration rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = [
        "algorithm",
        "episode",
        "end_timestep",
        "reward",
        "cost",
        "length",
        "violated",
        "unsafe_state_visit_count",
        "safe_trajectory",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def episode_rows(algorithm: str, episodes: Iterable[EpisodeMetrics]) -> list[dict[str, Any]]:
    """Convert episode dataclasses to CSV-friendly dictionaries."""

    rows: list[dict[str, Any]] = []
    for episode in episodes:
        row = asdict(episode)
        row["algorithm"] = algorithm
        rows.append(row)
    return rows


def training_episode_rows(algorithm: str, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add the algorithm column to training exploration episode records."""

    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        row["algorithm"] = algorithm
        rows.append(row)
    return rows


def save_checkpoint(path: Path, model: Any, *, algorithm: str, metadata: dict[str, Any]) -> None:
    """Save model parameters and run metadata for later inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "algorithm": algorithm,
        "metadata": metadata,
        "num_timesteps": int(getattr(model, "num_timesteps", 0)),
        "last_stats": dict(getattr(model, "last_stats", {})),
        "training_episodes": list(getattr(model, "training_episodes", [])),
        "actor_state_dict": model.actor.state_dict(),
        "reward_critic_state_dict": model.reward_critic.state_dict(),
        "cost_critic_state_dict": model.cost_critic.state_dict(),
    }
    if hasattr(model, "log_std"):
        payload["log_std"] = model.log_std.detach().cpu()
    th.save(payload, path)


def load_checkpoint_model(
    checkpoint_path: Path,
    *,
    env: gym.Env,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Reconstruct a saved safe-RL baseline from a training checkpoint."""

    checkpoint = th.load(checkpoint_path, map_location=device, weights_only=False)
    algorithm = checkpoint["algorithm"]
    metadata = dict(checkpoint.get("metadata", {}))
    model = build_safe_rl_baseline(
        algorithm,
        env,
        cost_limit=float(metadata.get("cost_limit", 0.0)),
        seed=int(metadata.get("seed", 0)),
        device=device,
    )
    model.actor.load_state_dict(checkpoint["actor_state_dict"])
    model.reward_critic.load_state_dict(checkpoint["reward_critic_state_dict"])
    model.cost_critic.load_state_dict(checkpoint["cost_critic_state_dict"])
    if hasattr(model, "log_std") and "log_std" in checkpoint:
        model.log_std.data.copy_(checkpoint["log_std"].to(model.device))
    model.num_timesteps = int(checkpoint.get("num_timesteps", 0))
    model.last_stats = dict(checkpoint.get("last_stats", {}))
    return model, checkpoint


def rollout_policy_frames(
    model: Any,
    env: gym.Env,
    *,
    seed: int,
    deterministic: bool = True,
) -> tuple[list[np.ndarray], EpisodeMetrics]:
    """Roll out one episode and return rendered frames plus episode metrics."""

    obs, _ = env.reset(seed=seed)
    frames = [np.asarray(env.render())]
    done = False
    total_reward = 0.0
    total_cost = 0.0
    length = 0
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        action_int = int(np.asarray(action).item())
        obs, reward, terminated, truncated, _info = env.step(action_int)
        frames.append(np.asarray(env.render()))
        total_reward += float(reward)
        total_cost += state_cost(env, obs, _info)
        length += 1
        done = bool(terminated or truncated)

    return frames, EpisodeMetrics(
        episode=0,
        reward=total_reward,
        cost=total_cost,
        length=length,
        violated=False,
    )


def save_gif(path: Path, frames: Iterable[np.ndarray], *, fps: float = 4.0) -> None:
    """Save RGB frames as an animated GIF."""

    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(np.asarray(frame).astype(np.uint8)) for frame in frames]
    if not images:
        raise ValueError("Cannot save a GIF with no frames.")
    duration_ms = int(round(1000.0 / fps))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
