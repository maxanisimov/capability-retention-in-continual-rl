"""Train PPO on a MASA-style environment wrapped with a probabilistic shield."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym
import numpy as np
from masa.common.wrappers import ConstraintPersistentWrapper
from masa.prob_shield.prob_shield_wrapper_v2 import ProbShieldWrapperDisc
from stable_baselines3 import PPO


REPO_ROOT = Path(__file__).resolve().parents[3]
for import_path in (REPO_ROOT, REPO_ROOT / "core"):
    path_str = str(import_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from projects.safe_crl.utils.masa_tabular_envs.factory import make_custom_masa_env  # noqa: E402
from projects.safe_policy_optimisation.utils.minipacman_safe_rl import (  # noqa: E402
    EpisodeMetrics,
    aggregate_training_violations,
    aggregate_violations,
    write_json,
)


ALGORITHM_NAME = "masa_shielded_ppo"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "projects"
    / "safe_policy_optimisation"
    / "artifacts"
    / "masa_shielded_policy"
)


class SafetyBoundArrayWrapper(gym.ObservationWrapper):
    """Convert MASA's scalar safety bound to the declared one-element Box."""

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        obs = dict(observation)
        obs["safety_bound"] = np.asarray([obs["safety_bound"]], dtype=np.float32)
        return obs


class CostInfoWrapper(gym.Wrapper):
    """Expose the label-derived cost in ``info`` after shield projection."""

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = int(np.asarray(obs["orig_obs"]).item())
        cost = float(self.unwrapped.cost_fn(self.unwrapped.label_fn(state)))
        info = dict(info)
        info["cost"] = cost
        info["violated_step"] = cost > 0.0
        info["orig_obs"] = state
        info["safety_bound"] = float(np.asarray(obs["safety_bound"]).reshape(-1)[0])
        return obs, reward, terminated, truncated, info


class EpisodeRecorderWrapper(gym.Wrapper):
    """Record completed episode reward/cost/length during training or evaluation."""

    def __init__(self, env: gym.Env, *, cost_limit: float) -> None:
        super().__init__(env)
        self.cost_limit = float(cost_limit)
        self.episodes: list[dict[str, float | int | bool]] = []
        self._episode_index = 0
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self._episode_reward = 0.0
        self._episode_cost = 0.0
        self._episode_length = 0

    def reset(self, **kwargs: Any):
        self._reset_accumulators()
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_reward += float(reward)
        self._episode_cost += float(info.get("cost", 0.0))
        self._episode_length += 1
        if terminated or truncated:
            self.episodes.append(
                {
                    "episode": self._episode_index,
                    "reward": self._episode_reward,
                    "cost": self._episode_cost,
                    "length": self._episode_length,
                    "violated": self._episode_cost > self.cost_limit,
                }
            )
            self._episode_index += 1
            self._reset_accumulators()
        return obs, reward, terminated, truncated, info


def make_masa_shielded_env(
    env_id: str,
    *,
    max_episode_steps: int,
    env_kwargs: dict[str, Any],
    safety_tolerance: float,
    theta: float,
    max_vi_steps: int,
    granularity: int,
    cost_limit: float,
    record_episodes: bool,
    render_mode: str | None = None,
) -> gym.Env:
    """Build a MASA-style env with ``ProbShieldWrapperDisc`` at the requested tolerance."""

    base_env = make_custom_masa_env(
        env_id,
        max_episode_steps=max_episode_steps,
        env_kwargs=env_kwargs,
        render_mode=render_mode,
    )
    shielded = ProbShieldWrapperDisc(
        ConstraintPersistentWrapper(base_env),
        label_fn=base_env.unwrapped.label_fn,
        cost_fn=base_env.unwrapped.cost_fn,
        theta=theta,
        max_vi_steps=max_vi_steps,
        init_safety_bound=safety_tolerance,
        granularity=granularity,
    )
    env: gym.Env = SafetyBoundArrayWrapper(shielded)
    env = CostInfoWrapper(env)
    if record_episodes:
        env = EpisodeRecorderWrapper(env, cost_limit=cost_limit)
    return env


def _episode_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        row = dict(record)
        row["algorithm"] = ALGORITHM_NAME
        rows.append(row)
    return rows


def _write_episode_csv(path: Path, rows: list[dict[str, Any]], *, include_end_timestep: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["algorithm", "episode", "reward", "cost", "length", "violated"]
    if include_end_timestep:
        fieldnames.insert(2, "end_timestep")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _training_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    end_timestep = 0
    for record in records:
        row = dict(record)
        end_timestep += int(row["length"])
        row["end_timestep"] = end_timestep
        row["algorithm"] = ALGORITHM_NAME
        rows.append(row)
    return rows


def _evaluate(model: PPO, args: argparse.Namespace, env_kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    env = make_masa_shielded_env(
        args.env_id,
        max_episode_steps=args.max_episode_steps,
        env_kwargs=env_kwargs,
        safety_tolerance=args.safety_tolerance,
        theta=args.theta,
        max_vi_steps=args.max_vi_steps,
        granularity=args.granularity,
        cost_limit=args.cost_limit,
        record_episodes=True,
    )
    try:
        for episode in range(args.eval_episodes):
            obs, _ = env.reset(seed=int(args.seed) + 10_000 + episode)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
        return list(env.episodes)
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SB3 PPO on a MASA-style Gymnasium env shielded by ProbShieldWrapperDisc.",
    )
    parser.add_argument("--env-id", default=None, help="MASA-style Gymnasium env id to train on.")
    parser.add_argument("--env-kwargs", default=None, help="JSON object passed to the MASA env factory.")
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument(
        "--ghost-rand-prob",
        type=float,
        default=0.0,
        help="Backward-compatible MiniPacman shortcut used only when --env-kwargs is omitted.",
    )
    parser.add_argument(
        "--safety-tolerance",
        type=float,
        default=0.0,
        help="MASA initial safety bound/tolerance. Default 0.0 allows only zero-risk behaviour.",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=0.0,
        help="Episode cost budget used for reporting violation counts.",
    )
    parser.add_argument("--theta", type=float, default=1e-10)
    parser.add_argument("--max-vi-steps", type=int, default=1000)
    parser.add_argument("--granularity", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    return parser


def _parse_env_kwargs(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--env-kwargs must decode to a JSON object.")
    return payload


def _env_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.env_kwargs is not None:
        return _parse_env_kwargs(args.env_kwargs)
    if args.env_id == "CustomMiniPacman-v0":
        return {"ghost_rand_prob": float(args.ghost_rand_prob)}
    return {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.env_id is None:
        raise ValueError("--env-id is required for MASA-shielded policy training.")
    env_kwargs = _env_kwargs_from_args(args)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "algorithm": ALGORITHM_NAME,
        "env_id": args.env_id,
        "env_kwargs": env_kwargs,
        "wrapper": "masa.prob_shield.prob_shield_wrapper_v2.ProbShieldWrapperDisc",
        "safety_tolerance": float(args.safety_tolerance),
        "cost_limit": float(args.cost_limit),
        "total_timesteps": int(args.total_timesteps),
        "eval_episodes": int(args.eval_episodes),
        "seed": int(args.seed),
        "max_episode_steps": int(args.max_episode_steps),
        "theta": float(args.theta),
        "max_vi_steps": int(args.max_vi_steps),
        "granularity": int(args.granularity),
    }
    write_json(run_dir / "config.json", config)

    train_env = make_masa_shielded_env(
        args.env_id,
        max_episode_steps=args.max_episode_steps,
        env_kwargs=env_kwargs,
        safety_tolerance=args.safety_tolerance,
        theta=args.theta,
        max_vi_steps=args.max_vi_steps,
        granularity=args.granularity,
        cost_limit=args.cost_limit,
        record_episodes=True,
    )
    try:
        model = PPO(
            "MultiInputPolicy",
            train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
        print(f"[{ALGORITHM_NAME}] training for {args.total_timesteps} timesteps")
        model.learn(total_timesteps=args.total_timesteps)
        model.save(run_dir / "model.zip")
        training_records = list(train_env.episodes)
    finally:
        train_env.close()

    eval_records = _evaluate(model, args, env_kwargs)
    training_rows = _training_rows(training_records)
    eval_rows = _episode_rows(eval_records)
    _write_episode_csv(run_dir / "training_episodes.csv", training_rows, include_end_timestep=True)
    _write_episode_csv(run_dir / "episodes.csv", eval_rows)

    summary = {
        "algorithm": ALGORITHM_NAME,
        "model_path": str(run_dir / "model.zip"),
        "training": aggregate_training_violations(training_records),
        "evaluation": aggregate_violations(
            [
                EpisodeMetrics(
                    episode=int(record["episode"]),
                    reward=float(record["reward"]),
                    cost=float(record["cost"]),
                    length=int(record["length"]),
                    violated=bool(record["violated"]),
                )
                for record in eval_records
            ]
        ),
    }
    write_json(run_dir / "summary.json", summary)
    print(
        "[{algorithm}] training exploration violations: {count}/{episodes} ({pct:.2f}%)".format(
            algorithm=ALGORITHM_NAME,
            count=summary["training"]["training_violation_count"],
            episodes=summary["training"]["training_episode_count"],
            pct=summary["training"]["training_violation_percentage"],
        )
    )
    print(
        "[{algorithm}] eval violations: {count}/{episodes} ({pct:.2f}%)".format(
            algorithm=ALGORITHM_NAME,
            count=summary["evaluation"]["violation_count"],
            episodes=summary["evaluation"]["episodes"],
            pct=summary["evaluation"]["violation_percentage"],
        )
    )
    print(f"Artifacts written to {run_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
