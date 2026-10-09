#!/usr/bin/env python3
"""Train a cost-constrained FrozenLake baseline with training-only shaping.

Reuse the established baseline factory and hyperparameters, without changing
the baseline algorithms or the source-locked older FrozenLake experiments.
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

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ccl-frozenlake-shaping-matplotlib")
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "core")]

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from projects.safe_policy_optimisation.scripts import (  # noqa: E402
    run_frozenlake_ppo_shaping as ppo,
)
from projects.safe_policy_optimisation.scripts.run_stochastic_frozenlake_pspo import (  # noqa: E402
    source_paths as pspo_source_paths,
)
from projects.safe_policy_optimisation.utils.cli import net_arch_from_args  # noqa: E402
from projects.safe_policy_optimisation.utils.frozen_lake_experiment import (  # noqa: E402
    ENV_ID,
)
from projects.safe_policy_optimisation.utils.frozen_lake_reward_shaping import (  # noqa: E402
    FrozenLakePotentialReward,
)
from projects.safe_policy_optimisation.utils.io import write_json  # noqa: E402
from projects.safe_policy_optimisation.utils.learning_curves import (  # noqa: E402
    CallableUnshieldedRewardCurveCallback,
    LearningCurveLogger,
)
from projects.safe_policy_optimisation.utils.safe_rl import (  # noqa: E402
    ALGORITHM_NAMES,
    SAFE_RL_BASELINE_HYPERPARAMS,
    build_safe_rl_baseline,
    save_checkpoint,
)


def source_paths() -> list[Path]:
    paths = set(pspo_source_paths())
    paths.update((REPO / "core/safe_rl_baselines").rglob("*.py"))
    paths.update(
        {
            Path(__file__).resolve(),
            Path(ppo.__file__).resolve(),
            REPO
            / "projects/safe_policy_optimisation/utils/frozen_lake_reward_shaping.py",
            REPO / "projects/safe_policy_optimisation/utils/safe_rl.py",
        }
    )
    return sorted(paths)


def model_hash(model) -> str:
    digest = hashlib.sha256()
    for prefix in ("actor", "reward_critic", "cost_critic"):
        for name, value in sorted(getattr(model, prefix).state_dict().items()):
            digest.update(f"{prefix}.{name}".encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class TrainingAuditEnv(gym.Wrapper):
    """Adapt the existing per-step audit to the baselines' non-SB3 interface."""

    def __init__(self, env, logger):
        super().__init__(env)
        self.logger = logger
        self.steps = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        audit_info = {**info, "TimeLimit.truncated": bool(truncated and not terminated)}
        self.logger.num_timesteps = self.steps
        self.logger.locals = {
            "infos": [audit_info],
            "dones": [bool(terminated or truncated)],
        }
        self.logger._on_step()
        return obs, reward, terminated, truncated, info


class TimedCallableRewardCurve(CallableUnshieldedRewardCurveCallback):
    evaluation_seconds = 0.0

    def _evaluate_and_log(self, model, *, timestep):
        started = time.perf_counter()
        try:
            return super()._evaluate_and_log(model, timestep=timestep)
        finally:
            self.evaluation_seconds += time.perf_counter() - started


def build_parser() -> argparse.ArgumentParser:
    parser = ppo.build_parser()
    parser.description = __doc__
    parser.add_argument("--algorithm", choices=ALGORITHM_NAMES, required=True)
    parser.add_argument("--cost-limit", type=float, default=0.0)
    parser.add_argument("--cost-gamma", type=float, default=0.99)
    parser.add_argument("--cost-gae-lambda", type=float, default=0.95)
    parser.add_argument("--lagrangian-multiplier-init", type=float, default=0.0)
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.size < 16 or args.total_timesteps <= 0 or args.eval_episodes <= 0:
        raise ValueError("Require size >= 16 and positive budgets")
    if args.curve_eval_freq < 0 or args.curve_eval_episodes <= 0:
        raise ValueError("Invalid curve-evaluation settings")
    cap = (
        args.max_episode_steps
        if args.max_episode_steps is not None
        else 5000 * args.size // 128
    )
    if cap <= 0:
        raise ValueError("Episode cap must be positive")
    torch.set_num_threads(1)
    root = args.output_dir / (
        args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    root.mkdir(parents=True, exist_ok=False)
    kwargs = {
        "size": args.size,
        "is_slippery": True,
        "success_rate": args.success_rate,
        "step_penalty": args.step_penalty,
    }

    def raw_factory():
        return gym.make(ENV_ID, max_episode_steps=cap, **kwargs)

    raw = raw_factory()
    training = ppo.TrainingRewardLogger(
        root / "training_episodes.csv", gamma=args.gamma, scale=args.shaping_scale
    )
    logger = LearningCurveLogger(
        curve_dir=root / "learning_curves", tensorboard_log_dir=root / "tensorboard"
    )
    curve = TimedCallableRewardCurve(
        env_factory=raw_factory,
        curve_logger=logger,
        eval_freq=args.curve_eval_freq,
        eval_episodes=args.curve_eval_episodes,
        seed=args.seed + 30000,
        reward_threshold=0,
    )
    hyper = {key: getattr(args, key) for key in SAFE_RL_BASELINE_HYPERPARAMS}
    env = None
    try:
        started = time.perf_counter()
        model = build_safe_rl_baseline(
            args.algorithm,
            raw,
            cost_limit=args.cost_limit,
            seed=args.seed,
            device=args.device,
            net_arch=tuple(net_arch_from_args(args)),
            **hyper,
        )
        model_seconds = time.perf_counter() - started
        initial_hash = model_hash(model)
        shaped = FrozenLakePotentialReward(
            raw,
            gamma=args.gamma,
            scale=args.shaping_scale,
            timeout_mode=args.timeout_mode,
        )
        env = TrainingAuditEnv(shaped, training)
        model.env = env
        env.action_space.seed(args.seed)
        env.reset(seed=args.seed)
        if model_hash(model) != initial_hash:
            raise AssertionError("Attaching shaping changed initial model parameters")
        paths = source_paths()
        algorithm_settings = (
            {"lambda_lr": model.lambda_lr}
            if args.algorithm == "ppo_lagrangian"
            else {
                "target_kl": model.target_kl,
                "cg_iters": model.cg_iters,
                "n_critic_updates": model.n_critic_updates,
            }
            if args.algorithm == "cpo"
            else {key: getattr(model, key) for key in ("pid_kp", "pid_ki", "pid_kd")}
        )
        config = {
            "algorithm": args.algorithm,
            "seed": args.seed,
            "env_id": ENV_ID,
            "env_kwargs": kwargs,
            "max_episode_steps": cap,
            "requested_timesteps": args.total_timesteps,
            "eval_episodes": args.eval_episodes,
            "curve_eval_freq": args.curve_eval_freq,
            "curve_eval_episodes": args.curve_eval_episodes,
            "curve_evaluation_alignment": "first complete rollout at or beyond frequency",
            "device": args.device,
            "architecture": {"hidden_dim": args.hidden_dim, "n_hidden": args.n_hidden},
            "state_representation": "one_hot",
            "training_hyperparameters": hyper,
            "cost_limit": args.cost_limit,
            "algorithm_settings": algorithm_settings,
            "shaping": {
                "enabled": args.shaping_scale != 0,
                "scale": args.shaping_scale,
                "gamma": args.gamma,
                "timeout_mode": args.timeout_mode,
                "training_only": True,
                "distance_graph_uses_shield": False,
                "formula": "r + scale * (gamma * Phi(next_state) - Phi(state))",
            },
            "initialization": {
                "random_actor_and_critics": True,
                "warm_start": False,
                "shield": False,
                "goal_information_used": False,
                "reward_information_used": False,
                "policy_sha256": initial_hash,
                "actor_and_critics_unchanged_by_shaping": True,
            },
            "layout_sha256": hashlib.sha256(raw.unwrapped.desc.tobytes()).hexdigest(),
            "source_sha256": {
                str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths
            },
            "versions": {
                "python": platform.python_version(),
                "gymnasium": gym.__version__,
                "torch": torch.__version__,
            },
        }
        write_json(root / "config.json", config)
        with zipfile.ZipFile(
            root / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED
        ) as z:
            for p in paths:
                z.write(p, str(p.relative_to(REPO)))
        np.savez_compressed(
            root / "potential.npz",
            distances=shaped.distances,
            potential=shaped.potential,
        )
        _, initial = ppo.evaluate(
            model,
            raw_factory,
            episodes=args.curve_eval_episodes,
            seed=args.seed + 10000,
        )
        initial["algorithm"] = args.algorithm
        write_json(root / "initial_metrics.json", initial)
        logger.start_timing()
        started = time.perf_counter()

        def on_rollout(current):
            if args.verbose:
                print(
                    json.dumps(
                        {
                            "total_timesteps": current.num_timesteps,
                            "time_elapsed": time.perf_counter() - started,
                            "training_stats": current.last_stats,
                        }
                    ),
                    flush=True,
                )
            return curve(current)

        model.learn(total_timesteps=args.total_timesteps, callback=on_rollout)
        curve.record_final_evaluation(model)
        learn_seconds = time.perf_counter() - started
        save_checkpoint(
            root / "model.pt",
            model,
            algorithm=args.algorithm,
            metadata={**config, "final_timesteps": model.num_timesteps},
        )
        started = time.perf_counter()
        rows, metrics = ppo.evaluate(
            model, raw_factory, episodes=args.eval_episodes, seed=args.seed + 10000
        )
        eval_seconds = time.perf_counter() - started
        metrics["algorithm"] = args.algorithm
        write_json(root / "metrics.json", metrics)
        with (root / "episodes.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "algorithm": args.algorithm,
            "final_timesteps": model.num_timesteps,
            "completed_training_episodes": training.episode,
            "learn_wall_seconds_including_curve_evaluation": learn_seconds,
            "curve_evaluation_seconds": curve.evaluation_seconds,
            "training_seconds_excluding_curve_evaluation": learn_seconds
            - curve.evaluation_seconds,
            "final_evaluation_seconds": eval_seconds,
            "model_initialisation_seconds": model_seconds,
            "initial_policy_sha256": initial_hash,
            "max_discounted_telescoping_error": training.max_telescoping_error,
            "training_stats": model.last_stats,
            "metrics": metrics,
        }
        write_json(root / "summary.json", summary)
        print(json.dumps({"run_dir": str(root), **summary}, indent=2), flush=True)
        return summary
    finally:
        (raw if env is None else env).close()
        training.close()
        logger.close()


if __name__ == "__main__":
    run(build_parser().parse_args())
