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

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.trajectory_retention.lunarlander.core.orchestration.run_paths import (
    RL_CHOICES,
    SOURCE_MODE,
    default_adapt_ewc_settings_file,
    default_adapt_ppo_settings_file,
    default_adapt_rashomon_settings_file,
    default_outputs_root,
    default_task_settings_file,
    is_mode_complete,
    pipeline_root,
    resolve_default_source_run_dir,
    validate_rl,
)


METHOD_TO_MODE = {
    "unconstrained": "downstream_unconstrained",
    "ewc": "downstream_ewc",
    "rashomon": "downstream_rashomon",
    "rashomon_expanded": "downstream_rashomon_expanded",
    "rashomon_plus": "downstream_rashomon_plus",
}
METHOD_ORDER = ("unconstrained", "ewc", "rashomon", "rashomon_expanded", "rashomon_plus")

MODE_TO_CLI = {
    SOURCE_MODE: "train_source.py",
    "downstream_unconstrained": "adapt_unconstrained.py",
    "downstream_ewc": "adapt_ewc.py",
    "downstream_rashomon": "adapt_rashomon.py",
    "downstream_rashomon_expanded": "adapt_rashomon_expanded.py",
    "downstream_rashomon_plus": "adapt_rashomon_plus.py",
}
MODE_TO_RUN_SUBDIR = {
    "downstream_unconstrained": "downstream_unconstrained",
    "downstream_ewc": "downstream_ewc",
    "downstream_rashomon": "downstream_rashomon",
    "downstream_rashomon_expanded": "downstream_rashomon_expanded",
    "downstream_rashomon_plus": "downstream_rashomon_plus",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LunarLander source training plus requested adaptation methods for one seed.",
    )
    parser.add_argument("--pipeline", "--task-setting", type=str, dest="task_setting", default="default")
    parser.add_argument("--rl", type=str, default="ppo", choices=RL_CHOICES)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=list(METHOD_ORDER))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--task-settings-file", type=Path, default=default_task_settings_file())
    parser.add_argument("--adapt-settings-file", type=Path, default=default_adapt_ppo_settings_file())
    parser.add_argument("--ewc-settings-file", type=Path, default=default_adapt_ewc_settings_file())
    parser.add_argument("--rashomon-settings-file", type=Path, default=default_adapt_rashomon_settings_file())
    parser.add_argument("--outputs-root", type=Path, default=default_outputs_root())
    parser.add_argument("--source-run-root", type=Path, default=None)
    parser.add_argument("--resume-policy", choices=["skip_completed", "rerun_all"], default="skip_completed")
    parser.add_argument("--disable-task-neutralization", action="store_true")
    parser.add_argument("--total-timesteps-override", type=int, default=None)
    parser.add_argument("--ewc-lambda-override", type=float, default=None)
    parser.add_argument("--fisher-sample-size", type=int, default=10_000)
    parser.add_argument("--ewc-apply-to-critic", action="store_true")
    parser.add_argument("--rashomon-n-iters", type=int, default=None)
    parser.add_argument("--second-rashomon-n-iters", type=int, default=None)
    parser.add_argument("--rashomon-min-hard-spec", type=float, default=None)
    parser.add_argument("--rashomon-surrogate-aggregation", choices=["mean", "min"], default=None)
    parser.add_argument("--inverse-temp-start", type=int, default=None)
    parser.add_argument("--inverse-temp-max", type=int, default=None)
    parser.add_argument("--rashomon-checkpoint", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _worker_script(mode: str) -> Path:
    return pipeline_root() / "cli" / MODE_TO_CLI[mode]


def _common_downstream_args(args: argparse.Namespace, mode: str) -> list[str]:
    source_run_root = args.source_run_root if args.source_run_root is not None else args.outputs_root
    source_run_dir = resolve_default_source_run_dir(source_run_root, args.task_setting, args.seed)
    return [
        "--pipeline", args.task_setting,
        "--seed", str(args.seed),
        "--device", args.device,
        "--task-settings-file", str(args.task_settings_file),
        "--outputs-root", str(args.outputs_root),
        "--source-run-dir", str(source_run_dir),
        "--run-subdir", MODE_TO_RUN_SUBDIR[mode],
    ]


def _build_source_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(_worker_script(SOURCE_MODE)),
        "--seed", str(args.seed),
        "--task-role", "source",
        "--pipeline", args.task_setting,
        "--task-settings-file", str(args.task_settings_file),
        "--output-dir", str(args.outputs_root),
        "--device", args.device,
    ]
    return cmd


def _build_unconstrained_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(_worker_script("downstream_unconstrained"))]
    cmd.extend(_common_downstream_args(args, "downstream_unconstrained"))
    cmd.extend(["--adapt-settings-file", str(args.adapt_settings_file)])
    if args.disable_task_neutralization:
        cmd.append("--disable-task-neutralization")
    if args.total_timesteps_override is not None:
        cmd.extend(["--total-timesteps-override", str(args.total_timesteps_override)])
    return cmd


def _build_ewc_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(_worker_script("downstream_ewc"))]
    cmd.extend(_common_downstream_args(args, "downstream_ewc"))
    cmd.extend(["--adapt-settings-file", str(args.adapt_settings_file)])
    cmd.extend(["--ewc-settings-file", str(args.ewc_settings_file)])
    if args.disable_task_neutralization:
        cmd.append("--disable-task-neutralization")
    if args.total_timesteps_override is not None:
        cmd.extend(["--total-timesteps-override", str(args.total_timesteps_override)])
    if args.ewc_lambda_override is not None:
        cmd.extend(["--ewc-lambda-override", str(args.ewc_lambda_override)])
    cmd.extend(["--fisher-sample-size", str(args.fisher_sample_size)])
    if args.ewc_apply_to_critic:
        cmd.append("--ewc-apply-to-critic")
    return cmd


def _add_rashomon_overrides(cmd: list[str], args: argparse.Namespace) -> None:
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


def _build_rashomon_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(_worker_script("downstream_rashomon"))]
    cmd.extend(_common_downstream_args(args, "downstream_rashomon"))
    cmd.extend(["--adapt-settings-file", str(args.adapt_settings_file)])
    cmd.extend(["--rashomon-settings-file", str(args.rashomon_settings_file)])
    if args.disable_task_neutralization:
        cmd.append("--disable-task-neutralization")
    if args.total_timesteps_override is not None:
        cmd.extend(["--total-timesteps-override", str(args.total_timesteps_override)])
    _add_rashomon_overrides(cmd, args)
    return cmd


def _build_rashomon_expanded_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(_worker_script("downstream_rashomon_expanded"))]
    cmd.extend(_common_downstream_args(args, "downstream_rashomon_expanded"))
    if args.disable_task_neutralization:
        cmd.append("--disable-task-neutralization")
    if args.total_timesteps_override is not None:
        cmd.extend(["--total-timesteps", str(args.total_timesteps_override)])
    _add_rashomon_overrides(cmd, args)
    return cmd


def _build_rashomon_plus_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(_worker_script("downstream_rashomon_plus"))]
    cmd.extend(_common_downstream_args(args, "downstream_rashomon_plus"))
    if args.disable_task_neutralization:
        cmd.append("--disable-task-neutralization")
    if args.total_timesteps_override is not None:
        cmd.extend(["--total-timesteps", str(args.total_timesteps_override)])
    _add_rashomon_overrides(cmd, args)
    if args.second_rashomon_n_iters is not None:
        cmd.extend(["--second-rashomon-n-iters", str(args.second_rashomon_n_iters)])
    return cmd


METHOD_TO_CMD_BUILDER = {
    "unconstrained": _build_unconstrained_cmd,
    "ewc": _build_ewc_cmd,
    "rashomon": _build_rashomon_cmd,
    "rashomon_expanded": _build_rashomon_expanded_cmd,
    "rashomon_plus": _build_rashomon_plus_cmd,
}


def _run_source(args: argparse.Namespace) -> None:
    if args.source_run_root is not None:
        print(f"Skipping source training; using --source-run-root={args.source_run_root}")
        return
    if args.resume_policy == "skip_completed" and is_mode_complete(
        args.outputs_root, args.task_setting, args.seed, SOURCE_MODE, rl=args.rl,
    ):
        print(f"Skipping source training for seed={args.seed}; already complete.")
        return

    cmd = _build_source_cmd(args)
    print("Running source training:\n  " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Source training failed for seed={args.seed} (rc={result.returncode}).")


def _run_method(args: argparse.Namespace, method: str) -> None:
    mode = METHOD_TO_MODE[method]
    if args.resume_policy == "skip_completed" and is_mode_complete(
        args.outputs_root, args.task_setting, args.seed, mode, rl=args.rl,
    ):
        print(f"Skipping method={method} for seed={args.seed}; already complete.")
        return

    cmd = METHOD_TO_CMD_BUILDER[method](args)
    print(f"Running method={method}:\n  " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Method={method} failed for seed={args.seed} (rc={result.returncode}).")


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
        print(f"Dry run: seed={args.seed} pipeline={args.task_setting} methods=[source, {', '.join(requested)}]")
        return 0

    run_seed_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
