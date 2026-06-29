"""Train plain PPO on an unshielded Gymnasium environment."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

REPO_ROOT = Path(__file__).resolve().parents[3]
for import_path in (REPO_ROOT, REPO_ROOT / "core"):
    path_str = str(import_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from projects.safe_policy_optimisation.stages.train_discrete_shielded_policy import (  # noqa: E402
    EarlyStopOnUnshieldedSuccessCallback,
    _parse_env_kwargs,
    _records_to_metrics,
    _resolve_curve_eval_freq,
    load_shield_mask,
    make_unshielded_env,
)
from projects.safe_policy_optimisation.utils.learning_curves import (  # noqa: E402
    LearningCurveLogger,
    UnshieldedRewardCurveCallback,
)
from projects.safe_policy_optimisation.utils.minipacman_safe_rl import (  # noqa: E402
    aggregate_training_violations,
    aggregate_violations,
    write_json,
)


ALGORITHM_NAME = "plain_ppo"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "projects" / "safe_policy_optimisation" / "artifacts" / "plain_ppo_policy"
)


def _write_episode_csv(path: Path, rows: list[dict[str, Any]], *, include_end_timestep: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if include_end_timestep:
        fieldnames.insert(2, "end_timestep")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _episode_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        row = dict(record)
        row["algorithm"] = ALGORITHM_NAME
        rows.append(row)
    return rows


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


def _write_early_stop_evaluations(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timesteps",
                "episodes",
                "success_count",
                "success_rate",
                "mean_reward",
                "mean_length",
                "eval_policy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


class ShieldUnsafeActionProposalCallback(BaseCallback):
    """Audit PPO's sampled exploration actions against a shield without masking them."""

    def __init__(self, shield_mask: Any, curve_logger: LearningCurveLogger) -> None:
        super().__init__()
        self.shield_mask = np.asarray(shield_mask) != 0
        if self.shield_mask.ndim != 2:
            raise ValueError(f"shield_mask must be 2-D, got shape {self.shield_mask.shape}.")
        self.curve_logger = curve_logger

    def _on_step(self) -> bool:
        actions = np.asarray(self.locals.get("actions", []), dtype=np.int64).reshape(-1)
        algo = self.locals.get("self", self.model)
        obs = np.asarray(getattr(algo, "_last_obs", []), dtype=np.int64).reshape(-1)
        if actions.size == 0 or obs.size == 0:
            return True
        if obs.size == 1 and actions.size > 1:
            obs = np.full(actions.shape, int(obs[0]), dtype=np.int64)
        if actions.size == 1 and obs.size > 1:
            actions = np.full(obs.shape, int(actions[0]), dtype=np.int64)
        if obs.shape != actions.shape:
            raise ValueError(f"states and actions must be broadcastable, got {obs.shape} vs {actions.shape}.")
        unsafe = ~self.shield_mask[obs, actions]
        self.curve_logger.log_exploration_unsafe(
            timestep=int(self.num_timesteps),
            unsafe_this_step=int(np.count_nonzero(unsafe)),
            checked_this_step=int(unsafe.size),
        )
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train plain SB3 PPO on an unshielded Gymnasium env.",
    )
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--env-kwargs", default=None, help="JSON object passed to gym.make.")
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--cost-limit", type=float, default=0.0)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--shield-path",
        type=Path,
        default=None,
        help="Optional shield_q.pt used only to audit unsafe raw action proposals.",
    )
    parser.add_argument("--shield-key", default="shield")
    parser.add_argument("--shield-source", choices=("shield", "action_risk"), default="shield")
    parser.add_argument("--risk-threshold", type=float, default=None)
    parser.add_argument("--early-stop-eval-freq", type=int, default=0)
    parser.add_argument("--early-stop-eval-episodes", type=int, default=20)
    parser.add_argument("--early-stop-success-rate", type=float, default=1.0)
    parser.add_argument("--success-reward-threshold", type=float, default=0.0)
    parser.add_argument(
        "--tensorboard-log-dir",
        type=Path,
        default=None,
        help="TensorBoard log directory for learning curves. Defaults to <run-dir>/tensorboard.",
    )
    parser.add_argument(
        "--curve-eval-freq",
        type=int,
        default=None,
        help=(
            "Evaluate and log total reward every N timesteps. "
            "Defaults to --early-stop-eval-freq when positive, otherwise --n-steps. Use 0 to disable."
        ),
    )
    parser.add_argument("--curve-eval-episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    env_kwargs = _parse_env_kwargs(args.env_kwargs)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    curve_logger = LearningCurveLogger(
        curve_dir=run_dir / "learning_curves",
        tensorboard_log_dir=args.tensorboard_log_dir or run_dir / "tensorboard",
    )
    curve_eval_freq = _resolve_curve_eval_freq(args)
    shield_mask = None
    if args.shield_path is not None:
        shield_mask = load_shield_mask(
            args.shield_path,
            shield_key=args.shield_key,
            source=args.shield_source,
            risk_threshold=args.risk_threshold,
        )

    train_env = make_unshielded_env(
        args.env_id,
        env_kwargs=env_kwargs,
        max_episode_steps=args.max_episode_steps,
        cost_limit=args.cost_limit,
        record_episodes=True,
    )
    try:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
        reward_curve = UnshieldedRewardCurveCallback(
            env_factory=lambda: make_unshielded_env(
                args.env_id,
                env_kwargs=env_kwargs,
                max_episode_steps=args.max_episode_steps,
                cost_limit=args.cost_limit,
                record_episodes=False,
            ),
            curve_logger=curve_logger,
            eval_freq=curve_eval_freq,
            eval_episodes=args.curve_eval_episodes,
            seed=args.seed + 30_000,
            reward_threshold=args.success_reward_threshold,
            shield_mask=shield_mask,
        )
        early_stop = EarlyStopOnUnshieldedSuccessCallback(
            env_id=args.env_id,
            env_kwargs=env_kwargs,
            max_episode_steps=args.max_episode_steps,
            cost_limit=args.cost_limit,
            eval_freq=args.early_stop_eval_freq,
            eval_episodes=args.early_stop_eval_episodes,
            success_rate=args.early_stop_success_rate,
            seed=args.seed + 20_000,
            reward_threshold=args.success_reward_threshold,
        )
        print(f"[{ALGORITHM_NAME}] training for {args.total_timesteps} timesteps")
        callbacks: list[BaseCallback] = [reward_curve, early_stop]
        if shield_mask is not None:
            callbacks.append(ShieldUnsafeActionProposalCallback(shield_mask, curve_logger))
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
        final_curve_evaluation = reward_curve.record_final_evaluation()
        training_records = list(train_env.episodes)
        model.save(run_dir / "model.zip")
    finally:
        train_env.close()
        curve_logger.close()

    eval_env = make_unshielded_env(
        args.env_id,
        env_kwargs=env_kwargs,
        max_episode_steps=args.max_episode_steps,
        cost_limit=args.cost_limit,
        record_episodes=True,
    )
    try:
        for episode in range(args.eval_episodes):
            obs, _ = eval_env.reset(seed=int(args.seed) + 10_000 + episode)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, _info = eval_env.step(action)
                done = bool(terminated or truncated)
        eval_records = list(eval_env.episodes)
    finally:
        eval_env.close()

    config = {
        "algorithm": ALGORITHM_NAME,
        "env_id": args.env_id,
        "env_kwargs": env_kwargs,
        "max_episode_steps": args.max_episode_steps,
        "cost_limit": float(args.cost_limit),
        "shield_path": None if args.shield_path is None else str(args.shield_path),
        "shield_source": args.shield_source,
        "shield_key": args.shield_key,
        "risk_threshold": args.risk_threshold,
        "shield_shape": None if shield_mask is None else list(shield_mask.shape),
        "total_timesteps": int(args.total_timesteps),
        "training_hyperparameters": {
            "learning_rate": float(args.learning_rate),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "n_epochs": int(args.n_epochs),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "clip_range": float(args.clip_range),
            "ent_coef": float(args.ent_coef),
            "vf_coef": float(args.vf_coef),
            "max_grad_norm": float(args.max_grad_norm),
        },
        "eval_episodes": int(args.eval_episodes),
        "early_stop_eval_policy": "unshielded",
        "early_stop_eval_freq": int(args.early_stop_eval_freq),
        "early_stop_eval_episodes": int(args.early_stop_eval_episodes),
        "early_stop_success_rate": float(args.early_stop_success_rate),
        "success_reward_threshold": float(args.success_reward_threshold),
        "tensorboard_log_dir": str(curve_logger.tensorboard_log_dir),
        "learning_curve_dir": str(curve_logger.curve_dir),
        "curve_eval_freq": int(curve_eval_freq),
        "curve_eval_episodes": int(args.curve_eval_episodes),
        "seed": int(args.seed),
    }
    write_json(run_dir / "config.json", config)

    _write_episode_csv(run_dir / "training_episodes.csv", _training_rows(training_records), include_end_timestep=True)
    _write_episode_csv(run_dir / "episodes.csv", _episode_rows(eval_records))
    _write_early_stop_evaluations(run_dir / "early_stop_evaluations.csv", early_stop.evaluations)

    summary = {
        "algorithm": ALGORITHM_NAME,
        "model_path": str(run_dir / "model.zip"),
        "final_timesteps": int(model.num_timesteps),
        "total_exploration_steps": int(model.num_timesteps),
        "unsafe_proposed_actions_during_exploration": int(curve_logger.cumulative_unsafe),
        "unshielded_eval_unsafe_action_count": (
            0 if final_curve_evaluation is None else int(final_curve_evaluation.get("unsafe_proposed_action_count", 0))
        ),
        "unshielded_eval_safety_rate": (
            0.0 if final_curve_evaluation is None else float(final_curve_evaluation.get("safety_rate", 0.0))
        ),
        "unshielded_eval_success_rate": (
            0.0 if final_curve_evaluation is None else float(final_curve_evaluation.get("success_rate", 0.0))
        ),
        "unshielded_eval_mean_total_reward": (
            0.0 if final_curve_evaluation is None else float(final_curve_evaluation.get("mean_total_reward", 0.0))
        ),
        "early_stop_triggered": bool(early_stop.stop_triggered),
        "last_early_stop_evaluation": early_stop.evaluations[-1] if early_stop.evaluations else None,
        "training": aggregate_training_violations(training_records),
        "evaluation": aggregate_violations(_records_to_metrics(eval_records)),
        "evaluation_policy": "unshielded",
        "learning_curves": {
            "curve_dir": str(curve_logger.curve_dir),
            "tensorboard_log_dir": str(curve_logger.tensorboard_log_dir),
            "reward_evaluations": reward_curve.evaluations,
        },
    }
    write_json(run_dir / "summary.json", summary)
    print(f"Artifacts written to {run_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
