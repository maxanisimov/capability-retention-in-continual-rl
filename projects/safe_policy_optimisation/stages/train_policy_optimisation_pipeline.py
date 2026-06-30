"""Train deterministic safe-policy baselines, shielded PPO, and Rashomon PPO."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

from projects.safe_policy_optimisation.stages import (  # noqa: E402
    compute_shield_rashomon_set,
    train_cpo,
    train_ppo,
    train_ppo_lagrangian,
    train_rashomon_shielded_policy,
    train_discrete_shielded_policy,
)
from projects.safe_policy_optimisation.utils.config import (  # noqa: E402
    PIPELINES_FILE,
    TASKS_FILE,
    apply_settings_to_namespace,
    cli_supplied_flags,
    compose_pipeline_settings,
    load_and_apply_settings,
    load_yaml_settings,
    registry_source_file,
)
from projects.safe_policy_optimisation.utils.cpu_allocation import (  # noqa: E402
    apply_cpu_affinity,
    available_cpu_ids,
    cpu_affinity_supported,
    format_cpu_ids,
    normalise_cpu_ids,
    parse_cpu_ids,
    resolve_worker_count,
    worker_thread_count,
)
from projects.safe_policy_optimisation.utils.io import write_json  # noqa: E402
from projects.safe_policy_optimisation.utils.safe_rl import (  # noqa: E402
    ALGORITHM_NAMES,
    PPO_LAGRANGIAN_ALGORITHM_NAMES,
)
from projects.safe_policy_optimisation.utils.log import log_info  # noqa: E402

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "projects"
    / "safe_policy_optimisation"
    / "artifacts"
    / "policy_optimisation_pipeline"
)
DEFAULT_PIPELINES_FILE = PIPELINES_FILE
DEFAULT_TASKS_FILE = TASKS_FILE


def _parse_stage_args(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    return parser.parse_args(argv)


def _env_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.env_kwargs is not None:
        return dict(args.env_kwargs)
    if args.env_id == "CustomMiniPacman-v0":
        return {"ghost_rand_prob": float(args.ghost_rand_prob)}
    return {}


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _append_optional_arg(argv: list[str], flag: str, value: Any) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _stage_tensorboard_dir(args: argparse.Namespace, run_dir: Path, stage: str) -> Path | None:
    if args.tensorboard_log_dir is None:
        return None
    return Path(args.tensorboard_log_dir) / stage


def _append_monitoring_args(argv: list[str], args: argparse.Namespace, run_dir: Path, stage: str) -> None:
    _append_optional_arg(argv, "--curve-eval-freq", args.curve_eval_freq)
    argv.extend(["--curve-eval-episodes", str(args.curve_eval_episodes)])
    _append_optional_arg(argv, "--tensorboard-log-dir", _stage_tensorboard_dir(args, run_dir, stage))


def _rashomon_bounds_path(rashomon_dir: Path) -> Path:
    return Path(rashomon_dir) / "rashomon_param_bounds.pt"


def _parse_json_dict(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}.")
    return parsed


def _cli_supplied_flags(argv: list[str]) -> set[str]:
    return cli_supplied_flags(argv)


def _load_yaml_settings(path: Path) -> dict[str, Any]:
    return load_yaml_settings(path)


def _settings_file_for_args(args: argparse.Namespace) -> Path | None:
    if args.settings_file is not None:
        return Path(args.settings_file)
    if args.rashomon_dir is not None:
        candidate = Path(args.rashomon_dir) / "training_settings.yaml"
        return candidate if candidate.exists() else None
    return None


def apply_training_settings(
    args: argparse.Namespace,
    *,
    explicit_flags: set[str] | None = None,
) -> argparse.Namespace:
    settings_file = _settings_file_for_args(args)
    if settings_file is None:
        if args.pipeline is None:
            raise ValueError("Direct stage execution requires --pipeline or --settings-file.")
        settings, _pipeline_settings, _task_settings = compose_pipeline_settings(
            args.pipeline,
            task_name=args.task,
            tasks_file=DEFAULT_TASKS_FILE,
            pipelines_file=DEFAULT_PIPELINES_FILE,
        )
        selected_task = str(settings["task"])
        pipeline_source = registry_source_file(
            DEFAULT_PIPELINES_FILE,
            filename="pipelines.yaml",
            root_key="pipelines",
            name=args.pipeline,
        )
        task_source = registry_source_file(
            DEFAULT_TASKS_FILE,
            filename="tasks.yaml",
            root_key="tasks",
            name=selected_task,
        )
        args = apply_settings_to_namespace(
            args,
            settings,
            settings_file=pipeline_source,
            explicit_flags=explicit_flags,
        )
        args.training_task_settings_file = task_source
        args.training_pipeline_settings_file = pipeline_source
        return args
    return load_and_apply_settings(args, settings_file, explicit_flags=explicit_flags)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic safe-policy training pipeline: safe-RL baselines, "
            "shielded PPO, Rashomon-set synthesis, and Rashomon-shielded PPO. "
            "Rashomon early stopping evaluates the unshielded/raw policy."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pipeline",
        default=None,
        help="Pipeline registry key used when this stage is invoked directly without --settings-file.",
    )
    parser.add_argument(
        "--settings-file",
        type=Path,
        default=None,
        help=(
            "YAML settings file for this pipeline. If omitted, the script reads "
            "<rashomon-dir>/training_settings.yaml when present."
        ),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--env-kwargs", type=_parse_json_dict, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help=(
            "Number of independent policy optimisation jobs to run concurrently. "
            "Default 0 uses one CPU core per active method, capped by available CPUs."
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
        help="Optional comma-separated CPU ids to allocate across policy optimisation workers.",
    )
    parser.add_argument("--constraint", default="PCTL")
    parser.add_argument("--constraint-kwargs", type=_parse_json_dict, default={})
    parser.add_argument("--init-safety-bound", type=float, default=1e-12)
    parser.add_argument("--theta", type=float, default=1e-12)
    parser.add_argument("--max-vi-steps", type=int, default=2000)
    parser.add_argument("--granularity", type=int, default=10)
    parser.add_argument("--unsafe-cost-threshold", type=float, default=0.5)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument(
        "--ghost-rand-prob",
        type=float,
        default=0.0,
        help="MiniPacman ghost random-action probability. Default 0.0 gives deterministic dynamics.",
    )
    parser.add_argument("--cost-limit", type=float, default=0.0)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHM_NAMES,
        default=list(ALGORITHM_NAMES),
        help=(
            "Safe-RL baseline algorithms requested by legacy configs. "
            "PPO-Lagrangian methods run in train_ppo_lagrangian.py; CPO runs in train_cpo.py."
        ),
    )
    parser.add_argument(
        "--shield-path",
        type=Path,
        default=None,
        help="Saved shield_q.pt used by shielded and Rashomon policy stages.",
    )
    parser.add_argument(
        "--rashomon-dir",
        type=Path,
        default=None,
        help=(
            "Existing Rashomon artifact directory. If omitted, the pipeline computes "
            "a Rashomon set from --shield-path."
        ),
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="Shared training budget for every method before success-based early stopping.",
    )
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
    parser.add_argument("--cost-gamma", type=float, default=0.99)
    parser.add_argument("--cost-gae-lambda", type=float, default=0.95)
    parser.add_argument("--lagrangian-multiplier-init", type=float, default=0.0)
    parser.add_argument("--rashomon-n-iters", type=int, default=2000)
    parser.add_argument("--rashomon-checkpoint", type=int, default=100)
    parser.add_argument("--rashomon-batch-size", type=int, default=500)
    parser.add_argument("--certificate-samples", type=int, default=1000)
    parser.add_argument("--early-stop-eval-freq", type=int, default=5_000)
    parser.add_argument("--early-stop-eval-episodes", type=int, default=20)
    parser.add_argument("--early-stop-success-rate", type=float, default=1.0)
    parser.add_argument("--success-reward-threshold", type=float, default=0.0)
    parser.add_argument(
        "--tensorboard-log-dir",
        type=Path,
        default=None,
        help=(
            "Root TensorBoard log directory for policy learning curves. "
            "Each stage writes under a stage-specific subdirectory."
        ),
    )
    parser.add_argument(
        "--curve-eval-freq",
        type=int,
        default=None,
        help=(
            "Evaluate and log unshielded total reward every N timesteps. "
            "Defaults inside each stage to --early-stop-eval-freq when positive, otherwise --n-steps. Use 0 to disable."
        ),
    )
    parser.add_argument("--curve-eval-episodes", type=int, default=20)
    parser.add_argument(
        "--shielded-evaluation-policy",
        choices=("unshielded", "shielded"),
        default="unshielded",
        help="Final evaluation policy for the shielded PPO stage.",
    )
    parser.add_argument(
        "--rashomon-evaluation-policy",
        choices=("unshielded", "shielded"),
        default="unshielded",
        help="Final evaluation policy for the Rashomon shielded PPO stage.",
    )
    parser.add_argument("--skip-ppo-policy", action="store_true")
    parser.add_argument("--skip-ppo-lagrangian", action="store_true")
    parser.add_argument("--skip-cpo-policy", action="store_true")
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Compatibility alias that skips PPO-Lagrangian/PPO-PID-Lagrangian and CPO.",
    )
    parser.add_argument("--skip-shielded-policy", action="store_true")
    parser.add_argument("--skip-rashomon-policy", action="store_true")
    return parser


def _worker_thread_count(jobs: int, explicit: int | None) -> int | None:
    return worker_thread_count(jobs, explicit)


def _policy_optimisation_method_count(args: argparse.Namespace) -> int:
    count = 0
    if not args.skip_ppo_policy:
        count += 1
    if _ppo_lagrangian_algorithms(args):
        count += len(_ppo_lagrangian_algorithms(args))
    if _cpo_enabled(args):
        count += 1
    if not args.skip_shielded_policy:
        count += 1
    if not args.skip_rashomon_policy:
        count += 1
    return count


def _independent_stage_slot_count(args: argparse.Namespace) -> int:
    return (
        int(not args.skip_ppo_policy)
        + int(bool(_ppo_lagrangian_algorithms(args)))
        + int(_cpo_enabled(args))
        + int(not args.skip_shielded_policy)
        + int(not args.skip_rashomon_policy)
    )


def _ppo_lagrangian_algorithms(args: argparse.Namespace) -> list[str]:
    if args.skip_baselines or args.skip_ppo_lagrangian:
        return []
    return [algorithm for algorithm in args.algorithms if algorithm in PPO_LAGRANGIAN_ALGORITHM_NAMES]


def _cpo_enabled(args: argparse.Namespace) -> bool:
    return not args.skip_baselines and not args.skip_cpo_policy and "cpo" in set(args.algorithms)


def _ppo_lagrangian_worker_count(args: argparse.Namespace, *, jobs: int) -> int:
    algorithms = _ppo_lagrangian_algorithms(args)
    if not algorithms:
        return 0
    other_stage_slots = max(0, _independent_stage_slot_count(args) - 1)
    spare_slots = max(1, int(jobs) - other_stage_slots)
    return min(len(algorithms), spare_slots)


def _baseline_worker_count(args: argparse.Namespace, *, jobs: int) -> int:
    """Compatibility alias for older tests/imports."""

    return _ppo_lagrangian_worker_count(args, jobs=jobs)


def _pipeline_cpu_allocation(
    args: argparse.Namespace,
    *,
    jobs: int,
    baseline_jobs: int,
    cpu_ids: list[int],
) -> dict[str, Any]:
    slots = list(cpu_ids[: int(jobs)])
    return {
        "strategy": "dynamic_no_overlap",
        "worker_cpu_ids": slots,
        "ppo_lagrangian_worker_count": int(baseline_jobs),
        "safe_rl_baseline_worker_count": int(baseline_jobs),
        "independent_stage_slot_count": int(_independent_stage_slot_count(args)),
        "policy_optimisation_method_count": int(_policy_optimisation_method_count(args)),
    }


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


def _resolve_device(device: str) -> str:
    if str(device) != "auto":
        return str(device)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _stage_module(stage: str) -> Any:
    modules = {
        "ppo_policy": train_ppo,
        "ppo_lagrangian": train_ppo_lagrangian,
        "cpo": train_cpo,
        "shielded_policy": train_discrete_shielded_policy,
        "rashomon_set": compute_shield_rashomon_set,
        "rashomon_policy": train_rashomon_shielded_policy,
    }
    return modules[stage]


def _run_stage_worker(job: dict[str, Any]) -> dict[str, Any]:
    applied_cpu_ids = apply_cpu_affinity(job.get("cpu_ids"))
    _configure_worker_threads(job.get("torch_num_threads"))
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            log_info(f"Starting stage {job['stage']} on CPU ids {applied_cpu_ids}")
            module = _stage_module(str(job["stage"]))
            stage_args = _parse_stage_args(module.build_parser(), list(job["argv"]))
            summary = module.run(stage_args)
            log_info(f"Finished stage {job['stage']}")
    finished_at = time.time()
    return {
        "stage": str(job["stage"]),
        "summary": summary,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": float(finished_at - started_at),
        "log_path": str(log_path),
        "cpu_ids": applied_cpu_ids,
    }


def _run_stage_inline(
    stage: str,
    argv: list[str],
    *,
    torch_num_threads: int | None,
    cpu_ids: list[int] | None = None,
) -> dict[str, Any]:
    applied_cpu_ids = apply_cpu_affinity(cpu_ids) if cpu_ids is not None else None
    _configure_worker_threads(torch_num_threads)
    started_at = time.time()
    module = _stage_module(stage)
    stage_args = _parse_stage_args(module.build_parser(), argv)
    summary = module.run(stage_args)
    finished_at = time.time()
    return {
        "stage": stage,
        "summary": summary,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": float(finished_at - started_at),
        "log_path": None,
        "cpu_ids": applied_cpu_ids,
    }


def _stage_job(
    *,
    stage: str,
    argv: list[str],
    log_dir: Path,
    torch_num_threads: int | None,
    cpu_ids: list[int] | None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "argv": list(argv),
        "log_path": str(log_dir / f"{stage}.log"),
        "torch_num_threads": torch_num_threads,
        "cpu_ids": None if cpu_ids is None else list(cpu_ids),
    }


def _ppo_argv(args: argparse.Namespace, run_dir: Path, shield_path: Path, env_kwargs: str) -> list[str]:
    ppo_argv = [
        "--output-dir",
        str(run_dir),
        "--run-id",
        "ppo_policy",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--env-id",
        args.env_id,
        "--env-kwargs",
        env_kwargs,
        "--cost-limit",
        str(args.cost_limit),
        "--shield-path",
        str(shield_path),
        "--eval-episodes",
        str(args.eval_episodes),
        "--total-timesteps",
        str(args.total_timesteps),
        "--learning-rate",
        str(args.learning_rate),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--gamma",
        str(args.gamma),
        "--gae-lambda",
        str(args.gae_lambda),
        "--clip-range",
        str(args.clip_range),
        "--ent-coef",
        str(args.ent_coef),
        "--vf-coef",
        str(args.vf_coef),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--early-stop-eval-freq",
        str(args.early_stop_eval_freq),
        "--early-stop-eval-episodes",
        str(args.early_stop_eval_episodes),
        "--early-stop-success-rate",
        str(args.early_stop_success_rate),
        "--success-reward-threshold",
        str(args.success_reward_threshold),
    ]
    _append_monitoring_args(ppo_argv, args, run_dir, "ppo_policy")
    _append_optional_arg(ppo_argv, "--max-episode-steps", args.max_episode_steps)
    return ppo_argv


def _ppo_lagrangian_argv(
    args: argparse.Namespace,
    run_dir: Path,
    env_kwargs: str,
    *,
    baseline_jobs: int,
    cpu_ids: list[int] | None = None,
    log_dir: Path | None = None,
    stage_name: str = "ppo_lagrangian",
    algorithms: list[str] | None = None,
) -> list[str]:
    selected_algorithms = list(_ppo_lagrangian_algorithms(args) if algorithms is None else algorithms)
    baseline_argv = [
        "--output-dir",
        str(run_dir),
        "--run-id",
        stage_name,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--env-id",
        args.env_id,
        "--env-kwargs",
        env_kwargs,
        "--ghost-rand-prob",
        str(args.ghost_rand_prob),
        "--cost-limit",
        str(args.cost_limit),
        "--shield-path",
        str(args.shield_path),
        "--eval-episodes",
        str(args.eval_episodes),
        "--total-timesteps",
        str(args.total_timesteps),
        "--learning-rate",
        str(args.learning_rate),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--gamma",
        str(args.gamma),
        "--gae-lambda",
        str(args.gae_lambda),
        "--clip-range",
        str(args.clip_range),
        "--ent-coef",
        str(args.ent_coef),
        "--vf-coef",
        str(args.vf_coef),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--cost-gamma",
        str(args.cost_gamma),
        "--cost-gae-lambda",
        str(args.cost_gae_lambda),
        "--lagrangian-multiplier-init",
        str(args.lagrangian_multiplier_init),
        "--early-stop-eval-freq",
        str(args.early_stop_eval_freq),
        "--early-stop-eval-episodes",
        str(args.early_stop_eval_episodes),
        "--early-stop-success-rate",
        str(args.early_stop_success_rate),
        "--success-reward-threshold",
        str(args.success_reward_threshold),
        "--jobs",
        str(max(1, int(baseline_jobs))),
        "--algorithms",
        *selected_algorithms,
    ]
    _append_monitoring_args(baseline_argv, args, run_dir, stage_name)
    _append_optional_arg(baseline_argv, "--log-dir", log_dir)
    _append_optional_arg(baseline_argv, "--torch-num-threads", args.torch_num_threads)
    _append_optional_arg(baseline_argv, "--cpu-ids", format_cpu_ids(cpu_ids))
    _append_optional_arg(baseline_argv, "--max-episode-steps", args.max_episode_steps)
    return baseline_argv


def _cpo_argv(args: argparse.Namespace, run_dir: Path, env_kwargs: str) -> list[str]:
    return _ppo_lagrangian_argv(
        args,
        run_dir,
        env_kwargs,
        baseline_jobs=1,
        cpu_ids=None,
        log_dir=None,
        stage_name="cpo",
        algorithms=["cpo"],
    )


def _shielded_argv(args: argparse.Namespace, run_dir: Path, shield_path: Path, env_kwargs: str) -> list[str]:
    shielded_argv = [
        "--output-dir",
        str(run_dir),
        "--run-id",
        "shielded_policy",
        "--shield-path",
        str(shield_path),
        "--env-id",
        args.env_id,
        "--env-kwargs",
        env_kwargs,
        "--cost-limit",
        str(args.cost_limit),
        "--total-timesteps",
        str(args.total_timesteps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--seed",
        str(args.seed),
        "--learning-rate",
        str(args.learning_rate),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--gamma",
        str(args.gamma),
        "--gae-lambda",
        str(args.gae_lambda),
        "--clip-range",
        str(args.clip_range),
        "--ent-coef",
        str(args.ent_coef),
        "--vf-coef",
        str(args.vf_coef),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--device",
        args.device,
        "--early-stop-eval-freq",
        str(args.early_stop_eval_freq),
        "--early-stop-eval-episodes",
        str(args.early_stop_eval_episodes),
        "--early-stop-success-rate",
        str(args.early_stop_success_rate),
        "--success-reward-threshold",
        str(args.success_reward_threshold),
        "--evaluation-policy",
        args.shielded_evaluation_policy,
    ]
    _append_monitoring_args(shielded_argv, args, run_dir, "shielded_policy")
    _append_optional_arg(shielded_argv, "--max-episode-steps", args.max_episode_steps)
    return shielded_argv


def _rashomon_set_argv(
    args: argparse.Namespace,
    run_dir: Path,
    shield_path: Path,
    rashomon_dir: Path | None,
) -> tuple[list[str], Path, str]:
    rashomon_output_dir = run_dir
    rashomon_run_id = "shield_rashomon"
    if rashomon_dir is not None:
        rashomon_output_dir = Path(rashomon_dir).parent
        rashomon_run_id = Path(rashomon_dir).name
    return (
        [
            "--output-dir",
            str(rashomon_output_dir),
            "--run-id",
            str(rashomon_run_id),
            "--shield-path",
            str(shield_path),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--rashomon-n-iters",
            str(args.rashomon_n_iters),
            "--rashomon-checkpoint",
            str(args.rashomon_checkpoint),
            "--rashomon-batch-size",
            str(args.rashomon_batch_size),
            "--certificate-samples",
            str(args.certificate_samples),
        ],
        rashomon_output_dir,
        str(rashomon_run_id),
    )


def _rashomon_policy_argv(
    args: argparse.Namespace,
    run_dir: Path,
    shield_path: Path,
    rashomon_dir: Path,
    env_kwargs: str,
) -> list[str]:
    rashomon_policy_argv = [
        "--output-dir",
        str(run_dir),
        "--run-id",
        "rashomon_policy",
        "--rashomon-dir",
        str(rashomon_dir),
        "--shield-path",
        str(shield_path),
        "--env-id",
        args.env_id,
        "--env-kwargs",
        env_kwargs,
        "--cost-limit",
        str(args.cost_limit),
        "--total-timesteps",
        str(args.total_timesteps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--seed",
        str(args.seed),
        "--learning-rate",
        str(args.learning_rate),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--gamma",
        str(args.gamma),
        "--gae-lambda",
        str(args.gae_lambda),
        "--clip-range",
        str(args.clip_range),
        "--ent-coef",
        str(args.ent_coef),
        "--vf-coef",
        str(args.vf_coef),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--device",
        args.device,
        "--early-stop-eval-policy",
        "unshielded",
        "--evaluation-policy",
        args.rashomon_evaluation_policy,
        "--early-stop-eval-freq",
        str(args.early_stop_eval_freq),
        "--early-stop-eval-episodes",
        str(args.early_stop_eval_episodes),
        "--early-stop-success-rate",
        str(args.early_stop_success_rate),
        "--success-reward-threshold",
        str(args.success_reward_threshold),
    ]
    _append_monitoring_args(rashomon_policy_argv, args, run_dir, "rashomon_policy")
    _append_optional_arg(rashomon_policy_argv, "--max-episode-steps", args.max_episode_steps)
    return rashomon_policy_argv


def run(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_started_at = time.time()
    missing = [
        flag
        for flag, value in (
            ("--task", args.task),
            ("--env-id", args.env_id),
            ("--shield-path", args.shield_path),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Missing required pipeline settings after configuration: "
            + ", ".join(missing)
            + ". Use run_experiment.py --pipeline ... or pass --pipeline/--settings-file directly."
        )
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"

    shield_path = Path(args.shield_path)
    if not shield_path.exists():
        raise FileNotFoundError(
            f"Shield not found: {shield_path}. Provide --shield-path or synthesise the shield first."
        )

    requested_device = str(args.device)
    args.device = _resolve_device(args.device)
    rashomon_dir = args.rashomon_dir
    rashomon_bounds_exist = rashomon_dir is not None and _rashomon_bounds_path(Path(rashomon_dir)).exists()
    requested_jobs = int(args.jobs)
    cpu_ids = normalise_cpu_ids(args.cpu_ids) if args.cpu_ids is not None else available_cpu_ids()
    policy_method_count = _policy_optimisation_method_count(args)
    jobs = resolve_worker_count(
        requested_jobs,
        method_count=policy_method_count,
        cpu_ids=cpu_ids,
    )
    baseline_jobs = _baseline_worker_count(args, jobs=jobs)
    cpu_allocation = _pipeline_cpu_allocation(
        args,
        jobs=jobs,
        baseline_jobs=baseline_jobs,
        cpu_ids=cpu_ids,
    )
    args.torch_num_threads = _worker_thread_count(jobs, args.torch_num_threads)
    _configure_worker_threads(args.torch_num_threads if jobs <= 1 else None)

    task_env_kwargs = _env_kwargs_from_args(args)
    env_kwargs = json.dumps(task_env_kwargs)
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "training_settings_file": (
            None if getattr(args, "training_settings_file", None) is None
            else str(args.training_settings_file)
        ),
        "task": args.task,
        "task_settings_file": (
            None if getattr(args, "training_task_settings_file", None) is None
            else str(args.training_task_settings_file)
        ),
        "deterministic": all(float(value) == 0.0 for value in task_env_kwargs.values() if isinstance(value, (int, float))),
        "env_id": args.env_id,
        "env_kwargs": task_env_kwargs,
        "max_episode_steps": _optional_int(args.max_episode_steps),
        "requested_device": requested_device,
        "device": args.device,
        "requested_jobs": requested_jobs,
        "jobs": int(jobs),
        "torch_num_threads": args.torch_num_threads,
        "policy_optimisation_method_count": int(policy_method_count),
        "available_cpu_ids": list(cpu_ids),
        "cpu_affinity_supported": cpu_affinity_supported(),
        "cpu_allocation": cpu_allocation,
        "total_timesteps_budget": int(args.total_timesteps),
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
            "cost_gamma": float(args.cost_gamma),
            "cost_gae_lambda": float(args.cost_gae_lambda),
            "lagrangian_multiplier_init": float(args.lagrangian_multiplier_init),
        },
        "early_stop_eval_freq": int(args.early_stop_eval_freq),
        "early_stop_eval_episodes": int(args.early_stop_eval_episodes),
        "early_stop_success_rate": float(args.early_stop_success_rate),
        "success_reward_threshold": float(args.success_reward_threshold),
        "monitoring": {
            "tensorboard_log_dir": None if args.tensorboard_log_dir is None else str(args.tensorboard_log_dir),
            "curve_eval_freq": None if args.curve_eval_freq is None else int(args.curve_eval_freq),
            "curve_eval_episodes": int(args.curve_eval_episodes),
        },
        "shielded_evaluation_policy": args.shielded_evaluation_policy,
        "rashomon_evaluation_policy": args.rashomon_evaluation_policy,
        "shield_path": str(shield_path),
        "stages": {},
    }

    def record_stage(
        stage_result: dict[str, Any],
        *,
        run_path: Path,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "run_dir": str(run_path.resolve()),
            "summary": stage_result["summary"],
            "started_at": stage_result["started_at"],
            "finished_at": stage_result["finished_at"],
            "elapsed_seconds": stage_result["elapsed_seconds"],
        }
        if stage_result.get("log_path") is not None:
            payload["log_path"] = stage_result["log_path"]
        if stage_result.get("cpu_ids") is not None:
            payload["cpu_ids"] = stage_result["cpu_ids"]
        if extra:
            payload.update(extra)
        summary["stages"][stage_result["stage"]] = payload

    if jobs > 1:
        pending_specs: list[dict[str, Any]] = []
        if not args.skip_ppo_policy:
            pending_specs.append({"stage": "ppo_policy", "cpu_count": 1})
        if _ppo_lagrangian_algorithms(args):
            pending_specs.append({"stage": "ppo_lagrangian", "cpu_count": max(1, int(baseline_jobs))})
        if _cpo_enabled(args):
            pending_specs.append({"stage": "cpo", "cpu_count": 1})
        if not args.skip_shielded_policy:
            pending_specs.append({"stage": "shielded_policy", "cpu_count": 1})
        if rashomon_bounds_exist:
            summary["stages"]["rashomon_set"] = {
                "run_dir": str(Path(rashomon_dir).resolve()),
                "reused": True,
            }
            if not args.skip_rashomon_policy:
                pending_specs.append(
                    {
                        "stage": "rashomon_policy",
                        "cpu_count": 1,
                        "rashomon_dir": Path(rashomon_dir),
                    }
                )
        elif not args.skip_rashomon_policy:
            rashomon_argv, _rashomon_output_dir, _rashomon_run_id = _rashomon_set_argv(
                args,
                run_dir,
                shield_path,
                rashomon_dir,
            )
            pending_specs.append(
                {
                    "stage": "rashomon_set",
                    "cpu_count": 1,
                    "argv": rashomon_argv,
                }
            )

        def make_stage_job(spec: dict[str, Any], assigned_cpu_ids: list[int]) -> dict[str, Any]:
            stage = str(spec["stage"])
            if stage == "ppo_policy":
                return _stage_job(
                    stage=stage,
                    argv=_ppo_argv(args, run_dir, shield_path, env_kwargs),
                    log_dir=log_dir,
                    torch_num_threads=args.torch_num_threads,
                    cpu_ids=assigned_cpu_ids,
                )
            if stage == "ppo_lagrangian":
                return _stage_job(
                    stage=stage,
                    argv=_ppo_lagrangian_argv(
                        args,
                        run_dir,
                        env_kwargs,
                        baseline_jobs=max(1, len(assigned_cpu_ids)),
                        cpu_ids=assigned_cpu_ids,
                        log_dir=log_dir / "ppo_lagrangian",
                    ),
                    log_dir=log_dir,
                    torch_num_threads=args.torch_num_threads,
                    cpu_ids=assigned_cpu_ids,
                )
            if stage == "cpo":
                return _stage_job(
                    stage=stage,
                    argv=_cpo_argv(args, run_dir, env_kwargs),
                    log_dir=log_dir,
                    torch_num_threads=args.torch_num_threads,
                    cpu_ids=assigned_cpu_ids,
                )
            if stage == "shielded_policy":
                return _stage_job(
                    stage=stage,
                    argv=_shielded_argv(args, run_dir, shield_path, env_kwargs),
                    log_dir=log_dir,
                    torch_num_threads=args.torch_num_threads,
                    cpu_ids=assigned_cpu_ids,
                )
            if stage == "rashomon_set":
                return _stage_job(
                    stage=stage,
                    argv=list(spec["argv"]),
                    log_dir=log_dir,
                    torch_num_threads=args.torch_num_threads,
                    cpu_ids=assigned_cpu_ids,
                )
            if stage == "rashomon_policy":
                return _stage_job(
                    stage=stage,
                    argv=_rashomon_policy_argv(
                        args,
                        run_dir,
                        shield_path,
                        Path(spec["rashomon_dir"]),
                        env_kwargs,
                    ),
                    log_dir=log_dir,
                    torch_num_threads=args.torch_num_threads,
                    cpu_ids=assigned_cpu_ids,
                )
            raise ValueError(f"Unknown pending stage {stage!r}.")

        free_cpu_ids = list(cpu_ids[:jobs])
        future_to_job: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}

        def submit_ready(executor: concurrent.futures.ProcessPoolExecutor) -> None:
            index = 0
            while index < len(pending_specs):
                spec = pending_specs[index]
                cpu_count = int(spec["cpu_count"])
                if len(free_cpu_ids) < cpu_count:
                    index += 1
                    continue
                assigned_cpu_ids = free_cpu_ids[:cpu_count]
                del free_cpu_ids[:cpu_count]
                job = make_stage_job(spec, assigned_cpu_ids)
                future_to_job[executor.submit(_run_stage_worker, job)] = {
                    "stage": str(spec["stage"]),
                    "cpu_ids": assigned_cpu_ids,
                }
                pending_specs.pop(index)

        if pending_specs:
            context = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max(1, min(jobs, len(pending_specs))),
                mp_context=context,
            ) as executor:
                submit_ready(executor)
                while future_to_job:
                    for future in concurrent.futures.as_completed(future_to_job):
                        job_info = future_to_job.pop(future)
                        stage = str(job_info["stage"])
                        assigned_cpu_ids = list(job_info["cpu_ids"])
                        stage_result = future.result()
                        free_cpu_ids.extend(assigned_cpu_ids)
                        if stage == "ppo_policy":
                            record_stage(
                                stage_result,
                                run_path=run_dir / "ppo_policy",
                            )
                        elif stage == "ppo_lagrangian":
                            record_stage(
                                stage_result,
                                run_path=run_dir / "ppo_lagrangian",
                            )
                        elif stage == "cpo":
                            record_stage(stage_result, run_path=run_dir / "cpo")
                        elif stage == "shielded_policy":
                            record_stage(
                                stage_result,
                                run_path=run_dir / "shielded_policy",
                            )
                        elif stage == "rashomon_set":
                            rashomon_dir = Path(stage_result["summary"]["run_dir"])
                            record_stage(stage_result, run_path=Path(rashomon_dir))
                            if not args.skip_rashomon_policy:
                                pending_specs.insert(
                                    0,
                                    {
                                        "stage": "rashomon_policy",
                                        "cpu_count": 1,
                                        "rashomon_dir": Path(rashomon_dir),
                                    },
                                )
                        elif stage == "rashomon_policy":
                            record_stage(
                                stage_result,
                                run_path=run_dir / "rashomon_policy",
                                extra={
                                    "early_stop_eval_policy": "unshielded",
                                    "evaluation_policy": args.rashomon_evaluation_policy,
                                },
                            )
                        submit_ready(executor)
                        break
    else:
        if not args.skip_ppo_policy:
            ppo_result = _run_stage_inline(
                "ppo_policy",
                _ppo_argv(args, run_dir, shield_path, env_kwargs),
                torch_num_threads=args.torch_num_threads,
                cpu_ids=list(cpu_ids[:1]),
            )
            record_stage(ppo_result, run_path=run_dir / "ppo_policy")

        if _ppo_lagrangian_algorithms(args):
            baseline_result = _run_stage_inline(
                "ppo_lagrangian",
                _ppo_lagrangian_argv(
                    args,
                    run_dir,
                    env_kwargs,
                    baseline_jobs=1,
                    cpu_ids=list(cpu_ids[:1]),
                ),
                torch_num_threads=args.torch_num_threads,
                cpu_ids=list(cpu_ids[:1]),
            )
            record_stage(
                baseline_result,
                run_path=run_dir / "ppo_lagrangian",
            )

        if _cpo_enabled(args):
            cpo_result = _run_stage_inline(
                "cpo",
                _cpo_argv(args, run_dir, env_kwargs),
                torch_num_threads=args.torch_num_threads,
                cpu_ids=list(cpu_ids[:1]),
            )
            record_stage(cpo_result, run_path=run_dir / "cpo")

        if not args.skip_shielded_policy:
            shielded_result = _run_stage_inline(
                "shielded_policy",
                _shielded_argv(args, run_dir, shield_path, env_kwargs),
                torch_num_threads=args.torch_num_threads,
                cpu_ids=list(cpu_ids[:1]),
            )
            record_stage(
                shielded_result,
                run_path=run_dir / "shielded_policy",
            )

        if not rashomon_bounds_exist and not args.skip_rashomon_policy:
            rashomon_argv, _rashomon_output_dir, _rashomon_run_id = _rashomon_set_argv(
                args,
                run_dir,
                shield_path,
                rashomon_dir,
            )
            rashomon_result = _run_stage_inline(
                "rashomon_set",
                rashomon_argv,
                torch_num_threads=args.torch_num_threads,
                cpu_ids=list(cpu_ids[:1]),
            )
            rashomon_dir = Path(rashomon_result["summary"]["run_dir"])
            record_stage(rashomon_result, run_path=Path(rashomon_dir))
        elif rashomon_bounds_exist:
            summary["stages"]["rashomon_set"] = {
                "run_dir": str(Path(rashomon_dir).resolve()),
                "reused": True,
                "cpu_ids": list(cpu_ids[:1]),
            }

        if not args.skip_rashomon_policy:
            if rashomon_dir is None:
                raise RuntimeError("Rashomon directory was not created or provided.")
            rashomon_policy_result = _run_stage_inline(
                "rashomon_policy",
                _rashomon_policy_argv(args, run_dir, shield_path, Path(rashomon_dir), env_kwargs),
                torch_num_threads=args.torch_num_threads,
                cpu_ids=list(cpu_ids[:1]),
            )
            record_stage(
                rashomon_policy_result,
                run_path=run_dir / "rashomon_policy",
                extra={
                    "early_stop_eval_policy": "unshielded",
                    "evaluation_policy": args.rashomon_evaluation_policy,
                },
            )

    summary["started_at"] = pipeline_started_at
    summary["finished_at"] = time.time()
    summary["elapsed_seconds"] = float(summary["finished_at"] - pipeline_started_at)
    write_json(run_dir / "summary.json", summary)
    log_info(f"Deterministic safe-policy pipeline artifacts written to {run_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(raw_argv)
    args = apply_training_settings(args, explicit_flags=_cli_supplied_flags(raw_argv))
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
