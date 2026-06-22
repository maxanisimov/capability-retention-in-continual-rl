"""Run LunarLander training/adaptation across multiple seeds with CPU pinning."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from experiments.pipelines._shared.multi_seed_launcher import (
    dedupe_preserve_order,
    resolve_core_pool,
    run_seed_pool,
)
from experiments.pipelines.lunarlander.core.orchestration.run_paths import (
    NOADAPT_POLICY_SUBDIR,
    default_adapt_ewc_settings_file,
    default_adapt_ppo_settings_file,
    default_outputs_root,
    default_task_settings_file,
    pipeline_root,
    resolve_default_source_run_dir,
)


MODE_TO_CLI = {
    "source": "train_source.py",
    "downstream_unconstrained": "adapt_unconstrained.py",
    "downstream_ewc": "adapt_ewc.py",
    "downstream_rashomon": "adapt_rashomon.py",
    "downstream_rashomon_nonconvex": "adapt_rashomon_nonconvex.py",
    "downstream_rashomon_expanded": "adapt_rashomon_expanded.py",
    "downstream_rashomon_plus": "adapt_rashomon_plus.py",
    "expand_rashomon_set": "expand_rashomon_set.py",
}

MODE_TO_DEFAULT_RUN_SUBDIR = {
    "downstream_unconstrained": "downstream_unconstrained",
    "downstream_ewc": "downstream_ewc",
    "downstream_rashomon": "downstream_rashomon",
    "downstream_rashomon_nonconvex": "downstream_rashomon_nonconvex",
    "downstream_rashomon_expanded": "downstream_rashomon_expanded",
    "downstream_rashomon_plus": "downstream_rashomon_plus",
    "expand_rashomon_set": "downstream_rashomon_union_expanded",
}


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run one LunarLander mode across multiple seeds while pinning each "
            "active run to a unique CPU core."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=sorted(MODE_TO_CLI.keys()),
        help="Experiment mode to launch across seeds.",
    )
    parser.add_argument("--pipeline", type=str, dest="task_setting", default="default", help="Pipeline name.")
    parser.add_argument("--task-setting", type=str, dest="task_setting", help=argparse.SUPPRESS)
    parser.add_argument(
        "--task-settings-file",
        type=Path,
        default=default_task_settings_file(),
        help="Path to task pipeline settings YAML (legacy monolithic task settings YAML is also supported).",
    )
    parser.add_argument(
        "--adapt-settings-file",
        type=Path,
        default=default_adapt_ppo_settings_file(),
        help="Path to downstream adaptation PPO settings YAML.",
    )
    parser.add_argument(
        "--ewc-settings-file",
        type=Path,
        default=default_adapt_ewc_settings_file(),
        help="Path to downstream adaptation EWC settings YAML.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Seed list to run (default: 0 1 2 3 4 5 6 7 8 9).",
    )
    parser.add_argument(
        "--cores",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional explicit CPU core IDs to use. If omitted, all cores from the current "
            "affinity mask are used."
        ),
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=default_outputs_root(),
        help="Output root for downstream modes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_outputs_root(),
        help="Output root for source mode.",
    )
    parser.add_argument(
        "--source-run-root",
        type=Path,
        default=None,
        help=(
            "Optional root used to derive per-seed source checkpoints as "
            "<source-run-root>/<pipeline>/seed_<seed>/noadapt for downstream modes."
        ),
    )
    parser.add_argument(
        "--run-subdir",
        type=str,
        default=None,
        help="Override downstream run subdirectory under outputs/<pipeline>/seed_<seed>/.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device passed to worker runs.")
    parser.add_argument(
        "--disable-task-neutralization",
        action="store_true",
        help=(
            "Disable task neutralization in downstream modes. "
            "For expand_rashomon_set mode this forwards --no-task-neutralization; "
            "for other downstream modes it forwards --disable-task-neutralization."
        ),
    )
    parser.add_argument(
        "--total-timesteps-override",
        type=int,
        default=None,
        help="Forward --total-timesteps-override to downstream modes.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for per-seed launcher logs (defaults to per-mode location under outputs).",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="Polling interval for child process completion checks.",
    )
    return parser.parse_known_args()


def _worker_script(mode: str) -> Path:
    return pipeline_root() / "cli" / MODE_TO_CLI[mode]


def _build_worker_cmd(
    args: argparse.Namespace,
    *,
    seed: int,
    passthrough: list[str],
) -> list[str]:
    cmd: list[str] = [sys.executable, str(_worker_script(args.mode))]

    if args.mode == "source":
        cmd.extend(
            [
                "--seed",
                str(seed),
                "--task-role",
                "source",
                "--pipeline",
                str(args.task_setting),
                "--task-settings-file",
                str(args.task_settings_file),
                "--output-dir",
                str(args.output_dir),
                "--device",
                str(args.device),
            ],
        )
    else:
        cmd.extend(
            [
                "--pipeline",
                str(args.task_setting),
                "--seed",
                str(seed),
                "--device",
                str(args.device),
                "--task-settings-file",
                str(args.task_settings_file),
                "--outputs-root",
                str(args.outputs_root),
            ],
        )

        if args.mode in {"downstream_unconstrained", "downstream_ewc", "expand_rashomon_set"}:
            cmd.extend(["--adapt-settings-file", str(args.adapt_settings_file)])
        if args.mode == "downstream_ewc":
            cmd.extend(["--ewc-settings-file", str(args.ewc_settings_file)])

        run_subdir = args.run_subdir or MODE_TO_DEFAULT_RUN_SUBDIR.get(args.mode)
        if run_subdir:
            cmd.extend(["--run-subdir", str(run_subdir)])

        if args.source_run_root is not None:
            source_run_dir = resolve_default_source_run_dir(args.source_run_root, args.task_setting, seed)
            cmd.extend(["--source-run-dir", str(source_run_dir)])
        if args.disable_task_neutralization:
            if args.mode == "expand_rashomon_set":
                cmd.append("--no-task-neutralization")
            else:
                cmd.append("--disable-task-neutralization")
        if args.total_timesteps_override is not None:
            cmd.extend(["--total-timesteps-override", str(args.total_timesteps_override)])

    cmd.extend(passthrough)
    return cmd


def _default_log_dir(args: argparse.Namespace) -> Path:
    mode_suffix = NOADAPT_POLICY_SUBDIR if args.mode == "source" else args.mode
    base = args.output_dir if args.mode == "source" else args.outputs_root
    return base / args.task_setting / "multi_seed_logs" / mode_suffix


def main() -> int:
    args, passthrough = _parse_args()

    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError(
            "CPU affinity pinning requires os.sched_setaffinity/os.sched_getaffinity support.",
        )

    seeds = dedupe_preserve_order(args.seeds)
    if not seeds:
        raise ValueError("No seeds provided. Pass at least one seed via --seeds.")

    core_pool = resolve_core_pool(args.cores)
    if not core_pool:
        raise RuntimeError("No CPU cores available for scheduling.")

    log_dir = args.log_dir or _default_log_dir(args)

    print(
        f"Launching {len(seeds)} runs for mode={args.mode} pipeline={args.task_setting} "
        f"with {len(core_pool)} available core(s): {core_pool}",
    )
    if len(core_pool) < len(seeds):
        print("Note: fewer cores than seeds; runs will execute in waves with one run per core.")
    if passthrough:
        print(f"Forwarding extra args to worker scripts: {' '.join(passthrough)}")

    return run_seed_pool(
        seeds=seeds,
        cores=core_pool,
        build_cmd=lambda seed: _build_worker_cmd(args, seed=seed, passthrough=passthrough),
        log_dir=log_dir,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
