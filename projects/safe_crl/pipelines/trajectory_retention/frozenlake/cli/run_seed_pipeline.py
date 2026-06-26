"""Run source training plus the requested adaptation methods for one seed.

Each step is one subprocess (the existing core/methods/*.py scripts read argv
from sys.argv directly, so they're invoked the same way the old launchers
invoked them); this script just sequences them for a single seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from projects.safe_crl.pipelines.trajectory_retention.frozenlake.core.orchestration.run_paths import (
    RL_CHOICES,
    SOURCE_MODE,
    default_adapt_ewc_settings_file,
    default_adapt_ppo_settings_file,
    default_adapt_rashomon_settings_file,
    default_downstream_envs_file,
    default_outputs_root,
    default_source_envs_file,
    default_train_source_settings_file,
    is_mode_complete,
    pipeline_root,
    resolve_default_source_run_dir,
    validate_rl,
)


METHOD_TO_MODE = {
    "unconstrained": "downstream_unconstrained",
    "ewc": "downstream_ewc",
    "rashomon": "downstream_rashomon",
}
METHOD_ORDER = ("unconstrained", "ewc", "rashomon")
MODE_TO_CLI = {
    SOURCE_MODE: "train_source.py",
    "downstream_unconstrained": "adapt_unconstrained.py",
    "downstream_ewc": "adapt_ewc.py",
    "downstream_rashomon": "adapt_rashomon.py",
}
MODE_TO_RUN_SUBDIR = {
    "downstream_unconstrained": "downstream_unconstrained",
    "downstream_ewc": "downstream_ewc",
    "downstream_rashomon": "downstream_rashomon",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source training plus selected adaptation methods for one seed.",
    )
    parser.add_argument("--pipeline", "--layout", type=str, dest="layout", default="diagonal_30x30")
    parser.add_argument("--rl", type=str, default="ppo", choices=RL_CHOICES)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=[])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--activation", choices=["tanh", "relu"], default="relu")
    parser.add_argument("--source-env-file", type=Path, default=default_source_envs_file())
    parser.add_argument("--downstream-env-file", type=Path, default=default_downstream_envs_file())
    parser.add_argument("--source-settings-file", type=Path, default=default_train_source_settings_file())
    parser.add_argument("--adapt-settings-file", type=Path, default=default_adapt_ppo_settings_file())
    parser.add_argument("--ewc-settings-file", type=Path, default=default_adapt_ewc_settings_file())
    parser.add_argument("--rashomon-settings-file", type=Path, default=default_adapt_rashomon_settings_file())
    parser.add_argument("--outputs-root", type=Path, default=default_outputs_root())
    parser.add_argument(
        "--source-run-root",
        type=Path,
        default=None,
        help="Optional root containing pre-trained source checkpoints, if different from --outputs-root.",
    )
    parser.add_argument("--resume-policy", choices=["skip_completed", "rerun_all"], default="skip_completed")
    parser.add_argument("--disable-task-neutralization", action="store_true")
    parser.add_argument("--total-timesteps-override", type=int, default=None)
    parser.add_argument("--ewc-lambda-override", type=float, default=None)
    parser.add_argument("--fisher-sample-size", type=int, default=10_000)
    parser.add_argument("--ewc-apply-to-critic", action="store_true")
    parser.add_argument("--rashomon-n-iters", type=int, default=None)
    parser.add_argument("--rashomon-min-hard-spec", type=float, default=None)
    parser.add_argument("--rashomon-surrogate-aggregation", choices=["mean", "min"], default=None)
    parser.add_argument("--inverse-temp-start", type=int, default=None)
    parser.add_argument("--inverse-temp-max", type=int, default=None)
    parser.add_argument("--rashomon-checkpoint", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _worker_script(mode: str) -> Path:
    return pipeline_root() / "cli" / MODE_TO_CLI[mode]


def _common_args(args: argparse.Namespace) -> list[str]:
    return [
        "--pipeline", args.layout,
        "--seed", str(args.seed),
        "--device", args.device,
        "--activation", args.activation,
        "--source-env-file", str(args.source_env_file),
        "--downstream-env-file", str(args.downstream_env_file),
    ]


def _build_source_cmd(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, str(_worker_script(SOURCE_MODE)),
        *_common_args(args),
        "--settings-file", str(args.source_settings_file),
        "--adapt-settings-file", str(args.adapt_settings_file),
        "--output-dir", str(args.outputs_root),
    ]


def _build_method_cmd(args: argparse.Namespace, method: str) -> list[str]:
    mode = METHOD_TO_MODE[method]
    cmd = [
        sys.executable, str(_worker_script(mode)),
        *_common_args(args),
        "--source-settings-file", str(args.source_settings_file),
        "--adapt-settings-file", str(args.adapt_settings_file),
        "--outputs-root", str(args.outputs_root),
        "--run-subdir", MODE_TO_RUN_SUBDIR[mode],
    ]
    if args.source_run_root is not None:
        source_run_dir = resolve_default_source_run_dir(args.source_run_root, args.layout, args.seed, rl=args.rl)
        cmd.extend(["--source-run-dir", str(source_run_dir)])
    if args.disable_task_neutralization:
        cmd.append("--disable-task-neutralization")
    if args.total_timesteps_override is not None:
        cmd.extend(["--total-timesteps-override", str(args.total_timesteps_override)])
    if method == "ewc":
        cmd.extend(["--ewc-settings-file", str(args.ewc_settings_file)])
        cmd.extend(["--fisher-sample-size", str(args.fisher_sample_size)])
        if args.ewc_lambda_override is not None:
            cmd.extend(["--ewc-lambda-override", str(args.ewc_lambda_override)])
        if args.ewc_apply_to_critic:
            cmd.append("--ewc-apply-to-critic")
    if method == "rashomon":
        cmd.extend(["--rashomon-settings-file", str(args.rashomon_settings_file)])
        if args.rashomon_n_iters is not None:
            cmd.extend(["--rashomon-n-iters", str(args.rashomon_n_iters)])
        if args.rashomon_min_hard_spec is not None:
            cmd.extend(["--rashomon-min-hard-spec", str(args.rashomon_min_hard_spec)])
        if args.rashomon_surrogate_aggregation is not None:
            cmd.extend(["--rashomon-surrogate-aggregation", args.rashomon_surrogate_aggregation])
        if args.inverse_temp_start is not None:
            cmd.extend(["--inverse-temp-start", str(args.inverse_temp_start)])
        if args.inverse_temp_max is not None:
            cmd.extend(["--inverse-temp-max", str(args.inverse_temp_max)])
        if args.rashomon_checkpoint is not None:
            cmd.extend(["--rashomon-checkpoint", str(args.rashomon_checkpoint)])
    return cmd


def _run_source(args: argparse.Namespace) -> None:
    if args.source_run_root is not None:
        print(f"Using pre-trained source checkpoints from {args.source_run_root}; skipping source training.")
        return
    if args.resume_policy == "skip_completed" and is_mode_complete(
        args.outputs_root, args.layout, args.seed, SOURCE_MODE, rl=args.rl,
    ):
        print(f"[skip] source already complete for seed={args.seed}")
        return
    cmd = _build_source_cmd(args)
    print(f"[run] source seed={args.seed}: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode:
        raise RuntimeError(f"Source training failed for seed={args.seed} (rc={result.returncode}).")


def _run_method(args: argparse.Namespace, method: str) -> None:
    mode = METHOD_TO_MODE[method]
    if args.resume_policy == "skip_completed" and is_mode_complete(
        args.outputs_root, args.layout, args.seed, mode, rl=args.rl,
    ):
        print(f"[skip] {mode} already complete for seed={args.seed}")
        return
    cmd = _build_method_cmd(args, method)
    print(f"[run] {mode} seed={args.seed}: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode:
        raise RuntimeError(f"{mode} failed for seed={args.seed} (rc={result.returncode}).")


def run_seed_pipeline(args: argparse.Namespace) -> None:
    validate_rl(args.rl)
    requested = [method for method in METHOD_ORDER if method in args.methods]
    _run_source(args)
    for method in requested:
        _run_method(args, method)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
        validate_rl(args.rl)
        requested = [method for method in METHOD_ORDER if method in args.methods]
        print(f"Dry run: seed={args.seed} layout={args.layout} methods=[source, {', '.join(requested)}]")
        return 0
    run_seed_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
