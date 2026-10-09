#!/usr/bin/env python3
"""Test plain PPO with opt-in distance shaping and entirely raw evaluation.

This separate runner intentionally leaves source-locked live sweeps untouched.
Use --shaping-scale 0 for a matched unshaped control. No safe actor, witness,
shield, demonstrations, or action masking are used by this PPO experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ccl-frozenlake-shaping-matplotlib")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "core"))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import stable_baselines3 as sb3  # noqa: E402
import torch  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from projects.safe_policy_optimisation.utils.cli import (  # noqa: E402
    add_architecture_args,
    add_ppo_hyperparameter_args,
    net_arch_from_args,
)
from projects.safe_policy_optimisation.utils.frozen_lake_experiment import (  # noqa: E402
    ENV_ID,
)
from projects.safe_policy_optimisation.utils.frozen_lake_reward_shaping import (  # noqa: E402
    FrozenLakePotentialReward,
)
from projects.safe_policy_optimisation.utils.io import write_json  # noqa: E402
from projects.safe_policy_optimisation.utils.learning_curves import (  # noqa: E402
    LearningCurveLogger,
    UnshieldedRewardCurveCallback,
    evaluate_unshielded_total_rewards,
)

TRAINING_FIELDS = (
    "episode",
    "end_timestep",
    "length",
    "raw_reward",
    "shaped_reward",
    "raw_discounted_reward",
    "shaped_discounted_reward",
    "potential_start",
    "potential_end",
    "telescoping_error",
    "goal_reached",
    "safe_trajectory",
    "cost",
    "terminated",
    "truncated",
    "original_truncated",
)


class TrainingRewardLogger(BaseCallback):
    """Stream paired raw/shaped returns and an exact telescoping sanity check."""

    def __init__(self, path: Path, *, gamma: float, scale: float) -> None:
        super().__init__()
        self.gamma, self.scale = gamma, scale
        self.handle = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=TRAINING_FIELDS)
        self.writer.writeheader()
        self.episode = 0
        self.max_telescoping_error = 0.0
        self._reset()

    def _reset(self) -> None:
        self.length = 0
        self.raw = self.shaped = self.raw_discounted = self.shaped_discounted = (
            self.cost
        ) = 0.0
        self.start_potential = 0.0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        audit = info["reward_shaping"]
        if self.length == 0:
            self.start_potential = audit["potential"]
        discount = self.gamma**self.length
        self.raw += audit["raw_reward"]
        self.shaped += audit["shaped_reward"]
        self.raw_discounted += discount * audit["raw_reward"]
        self.shaped_discounted += discount * audit["shaped_reward"]
        self.cost += float(info.get("cost", 0))
        self.length += 1
        if self.locals["dones"][0]:
            correction = self.scale * (
                self.gamma**self.length * audit["next_potential"] - self.start_potential
            )
            error = self.shaped_discounted - self.raw_discounted - correction
            self.max_telescoping_error = max(self.max_telescoping_error, abs(error))
            truncated = bool(info.get("TimeLimit.truncated", False))
            self.writer.writerow(
                {
                    "episode": self.episode,
                    "end_timestep": self.num_timesteps,
                    "length": self.length,
                    "raw_reward": self.raw,
                    "shaped_reward": self.shaped,
                    "raw_discounted_reward": self.raw_discounted,
                    "shaped_discounted_reward": self.shaped_discounted,
                    "potential_start": self.start_potential,
                    "potential_end": audit["next_potential"],
                    "telescoping_error": error,
                    "goal_reached": bool(info.get("is_success", False)),
                    "safe_trajectory": self.cost == 0,
                    "cost": self.cost,
                    "terminated": not truncated,
                    "truncated": truncated,
                    "original_truncated": audit["original_truncated"],
                }
            )
            self.handle.flush()
            self.episode += 1
            self._reset()
        return True

    def close(self) -> None:
        self.handle.close()


class TimedRewardCurve(UnshieldedRewardCurveCallback):
    evaluation_seconds = 0.0

    def _evaluate_and_log(self, *, timestep: int) -> dict:
        started = time.perf_counter()
        try:
            return super()._evaluate_and_log(timestep=timestep)
        finally:
            self.evaluation_seconds += time.perf_counter() - started


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shaping-scale", type=float, default=1.0)
    parser.add_argument(
        "--timeout-mode", choices=("bootstrap", "terminal"), default="bootstrap"
    )
    parser.add_argument("--step-penalty", type=float, default=0.001)
    parser.add_argument("--success-rate", type=float, default=0.8)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--curve-eval-freq", type=int, default=20_000)
    parser.add_argument("--curve-eval-episodes", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    add_ppo_hyperparameter_args(parser)
    add_architecture_args(parser)
    parser.set_defaults(gamma=0.999)
    return parser


def evaluate(
    model: Any, env_factory: Any, *, episodes: int, seed: int
) -> tuple[list[dict], dict]:
    rows = evaluate_unshielded_total_rewards(
        model,
        env_factory,
        episodes=episodes,
        seed=seed,
        reward_threshold=0,
    )
    metrics = {
        "algorithm": "plain_ppo",
        "eval_episodes": len(rows),
        "reward": {
            "mean_total_reward": float(np.mean([r["total_reward"] for r in rows]))
        },
        "safety": {"safety_rate": float(np.mean([r["safe_trajectory"] for r in rows]))},
        "success": {
            "success_mode": "goal_reached",
            "success_count": sum(r["success"] for r in rows),
            "success_rate": float(np.mean([r["success"] for r in rows])),
        },
        "mean_episode_length": float(np.mean([r["length"] for r in rows])),
        "evaluation_policy": "greedy_unshielded",
        "reward_shaping_enabled": False,
    }
    return rows, metrics


def run(args: argparse.Namespace) -> dict:
    if args.size < 16 or args.total_timesteps <= 0 or args.eval_episodes <= 0:
        raise ValueError("Require size >= 16 and positive training/evaluation budgets.")
    if args.curve_eval_freq < 0 or args.curve_eval_episodes <= 0:
        raise ValueError("Invalid curve-evaluation settings.")
    torch.set_num_threads(1)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    cap = (
        args.max_episode_steps
        if args.max_episode_steps is not None
        else 5000 * args.size // 128
    )
    if cap <= 0:
        raise ValueError("max_episode_steps must be positive.")
    kwargs = {
        "size": args.size,
        "is_slippery": True,
        "success_rate": args.success_rate,
        "step_penalty": args.step_penalty,
    }

    def raw_env_factory():
        return gym.make(ENV_ID, max_episode_steps=cap, **kwargs)

    train_env = FrozenLakePotentialReward(
        raw_env_factory(),
        gamma=args.gamma,
        scale=args.shaping_scale,
        timeout_mode=args.timeout_mode,
    )
    source_files = [
        Path(__file__),
        REPO / "projects/safe_policy_optimisation/utils/frozen_lake_reward_shaping.py",
        REPO / "projects/safe_policy_optimisation/utils/frozen_lake_experiment.py",
        REPO / "projects/safe_policy_optimisation/utils/learning_curves.py",
        REPO / "projects/safe_policy_optimisation/utils/cli.py",
        REPO / "projects/safe_policy_optimisation/utils/safe_rl.py",
        REPO / "projects/safe_policy_optimisation/utils/metrics.py",
    ]
    config = {
        "algorithm": "plain_ppo",
        "seed": args.seed,
        "env_id": ENV_ID,
        "env_kwargs": kwargs,
        "max_episode_steps": cap,
        "requested_timesteps": args.total_timesteps,
        "eval_episodes": args.eval_episodes,
        "curve_eval_freq": args.curve_eval_freq,
        "curve_eval_episodes": args.curve_eval_episodes,
        "device": args.device,
        "architecture": {"hidden_dim": args.hidden_dim, "n_hidden": args.n_hidden},
        "shaping": {
            "enabled": args.shaping_scale != 0,
            "scale": args.shaping_scale,
            "gamma": args.gamma,
            "timeout_mode": args.timeout_mode,
            "formula": "r + scale * (gamma * Phi(next_state) - Phi(state))",
            "potential": "1 - non-hole grid distance / maximum finite distance; terminals/unreachable=0",
            "training_only": True,
            "distance_graph_uses_shield": False,
        },
        "initialization": {
            "random_ppo": True,
            "warm_start": False,
            "shield": False,
            "goal_information_used": False,
            "reward_information_used": False,
        },
        "training_hyperparameters": {
            key: getattr(args, key)
            for key in (
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
            )
        },
        "source_sha256": {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files
        },
        "versions": {
            "python": platform.python_version(),
            "gymnasium": gym.__version__,
            "stable_baselines3": sb3.__version__,
            "torch": torch.__version__,
        },
    }
    write_json(run_dir / "config.json", config)
    with zipfile.ZipFile(
        run_dir / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED
    ) as snapshot:
        for path in source_files:
            snapshot.write(path, str(path.relative_to(REPO)))
    np.savez_compressed(
        run_dir / "potential.npz",
        distances=train_env.distances,
        potential=train_env.potential,
    )
    curve_logger = LearningCurveLogger(
        curve_dir=run_dir / "learning_curves",
        tensorboard_log_dir=run_dir / "tensorboard",
    )
    training_logger = TrainingRewardLogger(
        run_dir / "training_episodes.csv", gamma=args.gamma, scale=args.shaping_scale
    )
    curve = TimedRewardCurve(
        env_factory=raw_env_factory,
        curve_logger=curve_logger,
        eval_freq=args.curve_eval_freq,
        eval_episodes=args.curve_eval_episodes,
        seed=args.seed + 30_000,
        reward_threshold=0,
    )
    try:
        model = sb3.PPO(
            "MlpPolicy",
            train_env,
            **config["training_hyperparameters"],
            policy_kwargs={"net_arch": net_arch_from_args(args)},
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
        )
        digest = hashlib.sha256()
        for name, value in sorted(model.policy.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        config["initialization"]["policy_sha256"] = digest.hexdigest()
        write_json(run_dir / "config.json", config)
        _, initial_metrics = evaluate(
            model,
            raw_env_factory,
            episodes=args.curve_eval_episodes,
            seed=args.seed + 10_000,
        )
        write_json(run_dir / "initial_metrics.json", initial_metrics)
        curve_logger.start_timing()
        started = time.perf_counter()
        model.learn(
            total_timesteps=args.total_timesteps, callback=[training_logger, curve]
        )
        learn_seconds = time.perf_counter() - started
        model.save(run_dir / "model.zip")
        started = time.perf_counter()
        rows, metrics = evaluate(
            model, raw_env_factory, episodes=args.eval_episodes, seed=args.seed + 10_000
        )
        evaluation_seconds = time.perf_counter() - started
        write_json(run_dir / "metrics.json", metrics)
        with (run_dir / "episodes.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "final_timesteps": model.num_timesteps,
            "completed_training_episodes": training_logger.episode,
            "learn_wall_seconds_including_curve_evaluation": learn_seconds,
            "curve_evaluation_seconds": curve.evaluation_seconds,
            "training_seconds_excluding_curve_evaluation": learn_seconds
            - curve.evaluation_seconds,
            "final_evaluation_seconds": evaluation_seconds,
            "max_discounted_telescoping_error": training_logger.max_telescoping_error,
            "initial_policy_sha256": digest.hexdigest(),
            "metrics": metrics,
        }
        write_json(run_dir / "summary.json", summary)
        print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2), flush=True)
        return summary
    finally:
        train_env.close()
        training_logger.close()
        curve_logger.close()


if __name__ == "__main__":
    run(build_parser().parse_args())
