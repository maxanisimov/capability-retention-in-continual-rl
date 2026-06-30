"""Train PPO-Lagrangian safe-RL baselines on a local MASA-style tabular environment."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path
from typing import Any


import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

from projects.safe_policy_optimisation.utils.learning_curves import (  # noqa: E402
    CallableUnshieldedRewardCurveCallback,
    LearningCurveLogger,
)
from projects.safe_policy_optimisation.utils.cpu_allocation import (  # noqa: E402
    apply_cpu_affinity,
    available_cpu_ids,
    cpu_affinity_supported,
    normalise_cpu_ids,
    parse_cpu_ids,
    resolve_worker_count,
    worker_thread_count,
)
from projects.safe_policy_optimisation.utils.cli import add_ppo_hyperparameter_args  # noqa: E402
from projects.safe_policy_optimisation.utils.envs import env_kwargs_from_args  # noqa: E402
from projects.safe_policy_optimisation.utils.io import (  # noqa: E402
    episode_rows,
    training_episode_rows,
    write_episode_csv,
    write_json,
    write_training_episode_csv,
)
from projects.safe_policy_optimisation.utils.safe_rl import (  # noqa: E402
    ALGORITHM_NAMES,
    PPO_LAGRANGIAN_ALGORITHM_NAMES,
    DEFAULT_TOTAL_TIMESTEPS,
    SAFE_RL_BASELINE_HYPERPARAMS,
    aggregate_training_violations,
    aggregate_violations,
    build_safe_rl_baseline,
    evaluate_policy,
    make_safe_rl_env,
    save_checkpoint,
)
from projects.safe_policy_optimisation.utils.metrics import summarise_evaluation  # noqa: E402
from projects.safe_policy_optimisation.utils.shield import load_shield_mask  # noqa: E402
from projects.safe_policy_optimisation.utils.log import log_info  # noqa: E402

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "projects"
    / "safe_policy_optimisation"
    / "artifacts"
    / "ppo_lagrangian"
)


def build_parser(
    *,
    algorithm_names: tuple[str, ...] = PPO_LAGRANGIAN_ALGORITHM_NAMES,
    default_algorithms: list[str] | None = None,
    description: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    algorithm_help: str = "PPO-Lagrangian algorithms to train.",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description
        or "Train PPO-Lagrangian safe-RL baselines on a MASA-style Gymnasium env and report cost violations."
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=algorithm_names,
        default=list(default_algorithms or algorithm_names),
        help=algorithm_help,
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override training timesteps for every selected algorithm.",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=0.0,
        help="Episode cost budget; an episode violates the constraint when cost > limit.",
    )
    parser.add_argument("--eval-episodes", type=int, default=100, help="Evaluation episodes per algorithm.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--env-id", default=None, help="Gymnasium env id to train on.")
    parser.add_argument("--env-kwargs", type=_parse_json_dict, default=None, help="JSON object passed to gym.make.")
    parser.add_argument("--max-episode-steps", type=int, default=100, help="Gymnasium time limit.")
    parser.add_argument(
        "--ghost-rand-prob",
        type=float,
        default=0.0,
        help="Backward-compatible MiniPacman shortcut used only when --env-kwargs is omitted.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device passed to the baselines.")
    add_ppo_hyperparameter_args(parser)
    parser.add_argument("--cost-gamma", type=float, default=0.99)
    parser.add_argument("--cost-gae-lambda", type=float, default=0.95)
    parser.add_argument("--lagrangian-multiplier-init", type=float, default=0.0)
    parser.add_argument(
        "--shield-path",
        type=Path,
        default=None,
        help="Optional shield_q.pt used only to audit unsafe raw action proposals during exploration.",
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
        help="TensorBoard root directory for baseline learning curves. Defaults to <run-dir>/tensorboard.",
    )
    parser.add_argument(
        "--curve-eval-freq",
        type=int,
        default=None,
        help=(
            "Evaluate and log unshielded total reward every N timesteps. "
            "Defaults to --early-stop-eval-freq when positive, otherwise --n-steps. Use 0 to disable."
        ),
    )
    parser.add_argument("--curve-eval-episodes", type=int, default=20)
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help=(
            "Number of baseline algorithms to train in parallel. "
            "Default 0 uses one CPU core per selected algorithm, capped by available CPUs."
        ),
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=None,
        help="Per-worker Torch/BLAS thread cap. Defaults to 1 when parallel jobs are used.",
    )
    parser.add_argument(
        "--cpu-ids",
        type=parse_cpu_ids,
        default=None,
        help="Optional comma-separated CPU ids to allocate across baseline algorithm workers.",
    )
    parser.add_argument("--output-dir", type=Path, default=output_dir, help="Artifact root directory.")
    parser.add_argument("--run-id", default=None, help="Optional artifact run id.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Optional directory for per-algorithm worker logs when --jobs > 1.",
    )
    return parser


def _parse_json_dict(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}.")
    return parsed


def _baseline_hyperparameters_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: getattr(args, key) for key in SAFE_RL_BASELINE_HYPERPARAMS}


def _env_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return env_kwargs_from_args(args)


def _algorithm_timesteps(algorithm: str, override: int | None) -> int:
    return int(override if override is not None else DEFAULT_TOTAL_TIMESTEPS[algorithm])


def _resolve_curve_eval_freq(curve_eval_freq: int | None, *, early_stop_eval_freq: int, n_steps: int) -> int:
    if curve_eval_freq is not None:
        return int(curve_eval_freq)
    if int(early_stop_eval_freq) > 0:
        return int(early_stop_eval_freq)
    return int(n_steps)


def _episode_succeeded(total_reward: float, infos: list[dict[str, Any]], *, reward_threshold: float) -> bool:
    for key in ("is_success", "success"):
        for info in reversed(infos):
            if key in info:
                return bool(info[key])
    return float(total_reward) > float(reward_threshold)


def evaluate_success_rate(
    model: Any,
    *,
    env_id: str,
    env_kwargs: dict[str, Any],
    max_episode_steps: int | None,
    episodes: int,
    seed: int,
    reward_threshold: float,
) -> dict[str, Any]:
    successes = 0
    rewards: list[float] = []
    lengths: list[int] = []
    eval_env = make_safe_rl_env(
        env_id,
        max_episode_steps=max_episode_steps,
        env_kwargs=env_kwargs,
    )
    try:
        for episode in range(int(episodes)):
            obs, _ = eval_env.reset(seed=int(seed) + episode)
            done = False
            total_reward = 0.0
            length = 0
            infos: list[dict[str, Any]] = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(int(action))
                infos.append(dict(info))
                total_reward += float(reward)
                length += 1
                done = bool(terminated or truncated)
            rewards.append(total_reward)
            lengths.append(length)
            successes += int(
                _episode_succeeded(total_reward, infos, reward_threshold=reward_threshold)
            )
    finally:
        eval_env.close()
    return {
        "episodes": int(episodes),
        "success_count": int(successes),
        "success_rate": float(successes / int(episodes)) if episodes else 0.0,
        "mean_reward": float(sum(rewards) / len(rewards)) if rewards else 0.0,
        "mean_length": float(sum(lengths) / len(lengths)) if lengths else 0.0,
    }


class SuccessEarlyStopper:
    def __init__(
        self,
        *,
        algorithm: str,
        env_id: str,
        env_kwargs: dict[str, Any],
        max_episode_steps: int | None,
        eval_freq: int,
        eval_episodes: int,
        success_rate: float,
        seed: int,
        reward_threshold: float,
    ) -> None:
        self.algorithm = algorithm
        self.env_id = str(env_id)
        self.env_kwargs = dict(env_kwargs)
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.target_success_rate = float(success_rate)
        self.seed = int(seed)
        self.reward_threshold = float(reward_threshold)
        self.evaluations: list[dict[str, Any]] = []
        self.stop_triggered = False
        self._next_eval_timestep = self.eval_freq if self.eval_freq > 0 else None

    def __call__(self, model: Any) -> bool:
        if self._next_eval_timestep is None:
            return True
        if int(model.num_timesteps) < self._next_eval_timestep:
            return True
        metrics = evaluate_success_rate(
            model,
            env_id=self.env_id,
            env_kwargs=self.env_kwargs,
            max_episode_steps=self.max_episode_steps,
            episodes=self.eval_episodes,
            seed=self.seed + int(model.num_timesteps),
            reward_threshold=self.reward_threshold,
        )
        row = {
            "algorithm": self.algorithm,
            "timesteps": int(model.num_timesteps),
            **metrics,
        }
        self.evaluations.append(row)
        self._next_eval_timestep += self.eval_freq
        if float(metrics["success_rate"]) >= self.target_success_rate:
            self.stop_triggered = True
            return False
        return True


class CallbackList:
    def __init__(self, callbacks: list[Any]) -> None:
        self.callbacks = list(callbacks)

    def __call__(self, model: Any) -> bool:
        continue_training = True
        for callback in self.callbacks:
            continue_training = bool(callback(model)) and continue_training
        return continue_training


class ShieldUnsafeActionProposalLogger:
    def __init__(self, shield_mask: np.ndarray, curve_logger: LearningCurveLogger) -> None:
        mask = np.asarray(shield_mask) != 0
        if mask.ndim != 2:
            raise ValueError(f"shield_mask must be 2-D, got shape {mask.shape}.")
        self.shield_mask = mask
        self.curve_logger = curve_logger

    def __call__(self, *, timestep: int, obs: Any, action: Any) -> None:
        states = np.asarray(obs).astype(np.int64).reshape(-1)
        actions = np.asarray(action).astype(np.int64).reshape(-1)
        if states.shape != actions.shape:
            if states.size == 1:
                states = np.full(actions.shape, int(states[0]), dtype=np.int64)
            elif actions.size == 1:
                actions = np.full(states.shape, int(actions[0]), dtype=np.int64)
            else:
                raise ValueError(f"states and actions must be broadcastable, got {states.shape} vs {actions.shape}.")
        unsafe = ~self.shield_mask[states, actions]
        self.curve_logger.log_exploration_unsafe(
            timestep=int(timestep),
            unsafe_this_step=int(np.count_nonzero(unsafe)),
            checked_this_step=int(unsafe.size),
        )


def _worker_thread_count(jobs: int, explicit: int | None) -> int | None:
    return worker_thread_count(jobs, explicit)


def _configure_worker_threads(torch_num_threads: int | None) -> None:
    if torch_num_threads is None:
        return
    count = str(max(1, int(torch_num_threads)))
    os.environ["OMP_NUM_THREADS"] = count
    os.environ["MKL_NUM_THREADS"] = count
    try:
        import torch

        torch.set_num_threads(int(count))
    except Exception:
        pass


def _train_algorithm_impl(job: dict[str, Any]) -> dict[str, Any]:
    cpu_id = job.get("cpu_id")
    applied_cpu_ids = apply_cpu_affinity([cpu_id]) if cpu_id is not None else None
    _configure_worker_threads(job.get("torch_num_threads"))
    algorithm = str(job["algorithm"])
    offset = int(job["offset"])
    seed = int(job["seed"])
    train_seed = seed + offset
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(job["env_kwargs"])
    tensorboard_root = (
        Path(job["tensorboard_log_dir"])
        if job.get("tensorboard_log_dir") is not None
        else output_dir / "tensorboard"
    )
    curve_logger = LearningCurveLogger(
        curve_dir=output_dir / "learning_curves" / algorithm,
        tensorboard_log_dir=tensorboard_root / algorithm,
    )
    reward_curve = CallableUnshieldedRewardCurveCallback(
        env_factory=lambda: make_safe_rl_env(
            str(job["env_id"]),
            max_episode_steps=job["max_episode_steps"],
            env_kwargs=env_kwargs,
        ),
        curve_logger=curve_logger,
        eval_freq=int(job["curve_eval_freq"]),
        eval_episodes=int(job["curve_eval_episodes"]),
        seed=seed + 30_000 + offset * 1_000,
        reward_threshold=float(job["success_reward_threshold"]),
        shield_mask=None if job.get("shield_mask") is None else np.asarray(job["shield_mask"]),
    )

    train_env = make_safe_rl_env(
        str(job["env_id"]),
        max_episode_steps=job["max_episode_steps"],
        env_kwargs=env_kwargs,
    )
    try:
        train_env.action_space.seed(train_seed)
        train_env.reset(seed=train_seed)
        model = build_safe_rl_baseline(
            algorithm,
            train_env,
            cost_limit=float(job["cost_limit"]),
            seed=train_seed,
            device=str(job["device"]),
            **dict(job["baseline_hyperparameters"]),
        )
        shield_mask = job.get("shield_mask")
        if shield_mask is not None and hasattr(model, "set_exploration_action_callback"):
            model.set_exploration_action_callback(
                ShieldUnsafeActionProposalLogger(np.asarray(shield_mask), curve_logger)
            )
        timesteps = _algorithm_timesteps(algorithm, job["total_timesteps"])
        early_stop = SuccessEarlyStopper(
            algorithm=algorithm,
            env_id=str(job["env_id"]),
            env_kwargs=env_kwargs,
            max_episode_steps=job["max_episode_steps"],
            eval_freq=int(job["early_stop_eval_freq"]),
            eval_episodes=int(job["early_stop_eval_episodes"]),
            success_rate=float(job["early_stop_success_rate"]),
            seed=seed + 20_000 + offset * 1_000,
            reward_threshold=float(job["success_reward_threshold"]),
        )
        log_info(f"[{algorithm}] training for {timesteps} timesteps")
        model.learn(total_timesteps=timesteps, callback=CallbackList([reward_curve, early_stop]))
        final_curve_evaluation = reward_curve.record_final_evaluation(model)

        eval_env = make_safe_rl_env(
            str(job["env_id"]),
            max_episode_steps=job["max_episode_steps"],
            env_kwargs=env_kwargs,
        )
        try:
            episodes = evaluate_policy(
                model,
                eval_env,
                cost_limit=float(job["cost_limit"]),
                episodes=int(job["eval_episodes"]),
                seed=seed + 10_000 + offset * 1_000,
            )
        finally:
            eval_env.close()

        metrics = aggregate_violations(episodes)
        training_records = list(getattr(model, "training_episodes", []))
        training_metrics = aggregate_training_violations(training_records)
        training_stats = dict(getattr(model, "last_stats", {}))
        summary = {
            **metrics,
            **training_metrics,
            "cpu_ids": applied_cpu_ids,
            "total_timesteps": float(timesteps),
            "training_stats": training_stats,
            "final_timesteps": float(model.num_timesteps),
            "total_exploration_steps": int(model.num_timesteps),
            "unsafe_proposed_actions_during_exploration": int(curve_logger.cumulative_unsafe),
            "unshielded_eval_unsafe_action_count": (
                0
                if final_curve_evaluation is None
                else int(final_curve_evaluation.get("unsafe_proposed_action_count", 0))
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
            "early_stop_evaluations": early_stop.evaluations,
            "learning_curves": {
                "curve_dir": str(curve_logger.curve_dir),
                "tensorboard_log_dir": str(curve_logger.tensorboard_log_dir),
                "unshielded_reward_evaluations": reward_curve.evaluations,
            },
        }
        save_checkpoint(
            output_dir / f"{algorithm}.pt",
            model,
            algorithm=algorithm,
            metadata={
                "cost_limit": float(job["cost_limit"]),
                "env_id": str(job["env_id"]),
                "env_kwargs": env_kwargs,
                "max_episode_steps": None if job["max_episode_steps"] is None else int(job["max_episode_steps"]),
                "seed": train_seed,
                "total_timesteps": timesteps,
                "final_timesteps": int(model.num_timesteps),
                "early_stop_triggered": bool(early_stop.stop_triggered),
                "cpu_ids": applied_cpu_ids,
            },
        )
        log_info(
            "[{algorithm}] eval violations: {count:.0f}/{episodes:.0f} ({pct:.2f}%)".format(
                algorithm=algorithm,
                count=metrics["violation_count"],
                episodes=metrics["episodes"],
                pct=metrics["violation_percentage"],
            )
        )
        log_info(
            "[{algorithm}] training exploration violations: "
            "{count:.0f}/{episodes:.0f} ({pct:.2f}%)".format(
                algorithm=algorithm,
                count=training_metrics["training_violation_count"],
                episodes=training_metrics["training_episode_count"],
                pct=training_metrics["training_violation_percentage"],
            )
        )
        return {
            "algorithm": algorithm,
            "summary": summary,
            "episode_rows": episode_rows(algorithm, episodes),
            "training_rows": training_episode_rows(algorithm, training_records),
        }
    finally:
        train_env.close()
        curve_logger.close()


def _train_algorithm(job: dict[str, Any]) -> dict[str, Any]:
    log_path_value = job.get("log_path")
    if not log_path_value:
        return _train_algorithm_impl(job)
    log_path = Path(log_path_value)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            return _train_algorithm_impl(job)


def _run_parallel_algorithm_jobs(
    job_payloads: list[dict[str, Any]],
    *,
    jobs: int,
    cpu_ids: list[int],
) -> list[dict[str, Any]]:
    if not job_payloads:
        return []
    max_workers = min(int(jobs), len(job_payloads), len(cpu_ids))
    if max_workers <= 1:
        payload = dict(job_payloads[0])
        payload["cpu_id"] = cpu_ids[0] if cpu_ids else None
        results = [_train_algorithm(payload)]
        for pending in job_payloads[1:]:
            payload = dict(pending)
            payload["cpu_id"] = cpu_ids[0] if cpu_ids else None
            results.append(_train_algorithm(payload))
        return results

    context = mp.get_context("spawn")
    pending_index = 0
    results: list[dict[str, Any]] = []
    future_to_cpu: dict[concurrent.futures.Future[dict[str, Any]], int] = {}

    def submit_next(
        executor: concurrent.futures.ProcessPoolExecutor,
        cpu_id: int,
    ) -> None:
        nonlocal pending_index
        if pending_index >= len(job_payloads):
            return
        payload = dict(job_payloads[pending_index])
        payload["cpu_id"] = int(cpu_id)
        pending_index += 1
        future_to_cpu[executor.submit(_train_algorithm, payload)] = int(cpu_id)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
    ) as executor:
        for cpu_id in cpu_ids[:max_workers]:
            submit_next(executor, int(cpu_id))
        while future_to_cpu:
            for future in concurrent.futures.as_completed(future_to_cpu):
                cpu_id = future_to_cpu.pop(future)
                results.append(future.result())
                submit_next(executor, cpu_id)
                break
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.env_id is None:
        raise ValueError("--env-id is required for PPO-Lagrangian training.")
    unsupported = sorted(set(args.algorithms) - set(ALGORITHM_NAMES))
    if unsupported:
        raise ValueError(f"Unknown algorithms: {unsupported}")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = _env_kwargs_from_args(args)
    requested_jobs = int(args.jobs)
    cpu_ids = normalise_cpu_ids(args.cpu_ids) if args.cpu_ids is not None else available_cpu_ids()
    jobs = resolve_worker_count(
        requested_jobs,
        method_count=len(args.algorithms),
        cpu_ids=cpu_ids,
    )
    worker_cpu_ids = list(cpu_ids[:jobs])
    torch_num_threads = _worker_thread_count(jobs, args.torch_num_threads)
    _configure_worker_threads(torch_num_threads if jobs <= 1 else None)
    shield_mask = None
    if args.shield_path is not None:
        shield_mask = load_shield_mask(
            args.shield_path,
            shield_key=args.shield_key,
            source=args.shield_source,
            risk_threshold=args.risk_threshold,
        )
    curve_eval_freq = _resolve_curve_eval_freq(
        args.curve_eval_freq,
        early_stop_eval_freq=args.early_stop_eval_freq,
        n_steps=args.n_steps,
    )

    config = {
        "algorithms": list(args.algorithms),
        "cost_limit": float(args.cost_limit),
        "eval_episodes": int(args.eval_episodes),
        "seed": int(args.seed),
        "env_id": args.env_id,
        "env_kwargs": env_kwargs,
        "max_episode_steps": None if args.max_episode_steps is None else int(args.max_episode_steps),
        "device": args.device,
        "total_timesteps": {
            algorithm: _algorithm_timesteps(algorithm, args.total_timesteps)
            for algorithm in args.algorithms
        },
        "early_stop_eval_freq": int(args.early_stop_eval_freq),
        "early_stop_eval_episodes": int(args.early_stop_eval_episodes),
        "early_stop_success_rate": float(args.early_stop_success_rate),
        "success_reward_threshold": float(args.success_reward_threshold),
        "shield_path": None if args.shield_path is None else str(args.shield_path),
        "shield_source": args.shield_source,
        "shield_key": args.shield_key,
        "risk_threshold": args.risk_threshold,
        "tensorboard_log_dir": None if args.tensorboard_log_dir is None else str(args.tensorboard_log_dir),
        "learning_curve_dir": str(output_dir / "learning_curves"),
        "curve_eval_freq": int(curve_eval_freq),
        "curve_eval_episodes": int(args.curve_eval_episodes),
        "requested_jobs": requested_jobs,
        "jobs": int(jobs),
        "torch_num_threads": torch_num_threads,
        "available_cpu_ids": list(cpu_ids),
        "worker_cpu_ids": worker_cpu_ids,
        "cpu_affinity_supported": cpu_affinity_supported(),
        "baseline_hyperparameters": _baseline_hyperparameters_from_args(args),
    }
    write_json(output_dir / "config.json", config)

    job_payloads = [
        {
            "algorithm": algorithm,
            "offset": offset,
            "seed": int(args.seed),
            "output_dir": str(output_dir),
            "env_id": args.env_id,
            "env_kwargs": env_kwargs,
            "max_episode_steps": args.max_episode_steps,
            "device": args.device,
            "cost_limit": args.cost_limit,
            "eval_episodes": args.eval_episodes,
            "total_timesteps": args.total_timesteps,
            "early_stop_eval_freq": args.early_stop_eval_freq,
            "early_stop_eval_episodes": args.early_stop_eval_episodes,
            "early_stop_success_rate": args.early_stop_success_rate,
            "success_reward_threshold": args.success_reward_threshold,
            "shield_mask": shield_mask,
            "tensorboard_log_dir": None if args.tensorboard_log_dir is None else str(args.tensorboard_log_dir),
            "curve_eval_freq": curve_eval_freq,
            "curve_eval_episodes": args.curve_eval_episodes,
            "torch_num_threads": torch_num_threads,
            "baseline_hyperparameters": _baseline_hyperparameters_from_args(args),
            "log_path": None if args.log_dir is None else str(Path(args.log_dir) / f"{algorithm}.log"),
        }
        for offset, algorithm in enumerate(args.algorithms)
    ]
    results = _run_parallel_algorithm_jobs(
        job_payloads,
        jobs=jobs,
        cpu_ids=worker_cpu_ids,
    )

    result_by_algorithm = {str(result["algorithm"]): result for result in results}
    summary: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    all_training_rows: list[dict[str, Any]] = []
    for algorithm in args.algorithms:
        result = result_by_algorithm[algorithm]
        summary[algorithm] = result["summary"]
        all_rows.extend(result["episode_rows"])
        all_training_rows.extend(result["training_rows"])

    metrics = {
        algorithm: summarise_evaluation(
            result_by_algorithm[algorithm]["episode_rows"],
            success_reward_threshold=float(getattr(args, "success_reward_threshold", 0.0)),
            cost_limit=float(args.cost_limit),
            algorithm=algorithm,
        )
        for algorithm in args.algorithms
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "metrics.json", metrics)
    write_episode_csv(output_dir / "episodes.csv", all_rows)
    write_training_episode_csv(output_dir / "training_episodes.csv", all_training_rows)
    log_info(f"Artifacts written to {output_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
