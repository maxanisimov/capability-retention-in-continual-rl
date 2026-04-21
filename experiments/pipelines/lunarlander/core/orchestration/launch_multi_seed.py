"""Run LunarLander training/adaptation across multiple seeds with CPU pinning."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import TextIO

from experiments.pipelines.lunarlander.core.orchestration.run_paths import (
    default_adapt_ewc_settings_file,
    default_adapt_ppo_settings_file,
    default_outputs_root,
    default_task_settings_file,
    pipeline_root,
)


MODE_TO_CLI = {
    "source": "train_source.py",
    "downstream_unconstrained": "adapt_unconstrained.py",
    "downstream_ewc": "adapt_ewc.py",
    "downstream_rashomon": "adapt_rashomon.py",
}

MODE_TO_DEFAULT_RUN_SUBDIR = {
    "downstream_unconstrained": "downstream_unconstrained",
    "downstream_ewc": "downstream_ewc",
    "downstream_rashomon": "downstream_rashomon",
}


@dataclass
class SeedRun:
    seed: int
    core: int
    process: subprocess.Popen[bytes]
    log_path: Path
    log_handle: TextIO


def _dedupe_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


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
    parser.add_argument(
        "--task-setting",
        type=str,
        default="default",
        help="Task-setting name from task_settings.yaml.",
    )
    parser.add_argument(
        "--task-settings-file",
        type=Path,
        default=default_task_settings_file(),
        help="Path to LunarLander task settings YAML.",
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
            "<source-run-root>/<task-setting>/seed_<seed>/source for downstream modes."
        ),
    )
    parser.add_argument(
        "--run-subdir",
        type=str,
        default=None,
        help="Override downstream run subdirectory under outputs/<task-setting>/seed_<seed>/.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device passed to worker runs.")
    parser.add_argument(
        "--disable-task-neutralization",
        action="store_true",
        help="Forward --disable-task-neutralization to downstream modes.",
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


def _resolve_core_pool(requested_cores: list[int] | None) -> list[int]:
    available_cores = sorted(os.sched_getaffinity(0))

    if requested_cores is None:
        return available_cores

    requested_set = set(requested_cores)
    available_set = set(available_cores)
    invalid = sorted(requested_set - available_set)
    if invalid:
        raise ValueError(
            f"Requested --cores {invalid} are not available in current affinity mask {available_cores}.",
        )
    return _dedupe_preserve_order(requested_cores)


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
                "--task-setting",
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
                "--task-setting",
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

        if args.mode in {"downstream_unconstrained", "downstream_ewc"}:
            cmd.extend(["--adapt-settings-file", str(args.adapt_settings_file)])
        if args.mode == "downstream_ewc":
            cmd.extend(["--ewc-settings-file", str(args.ewc_settings_file)])

        run_subdir = args.run_subdir or MODE_TO_DEFAULT_RUN_SUBDIR.get(args.mode)
        if run_subdir:
            cmd.extend(["--run-subdir", str(run_subdir)])

        if args.source_run_root is not None:
            source_run_dir = args.source_run_root / args.task_setting / f"seed_{seed}" / "source"
            cmd.extend(["--source-run-dir", str(source_run_dir)])
        if args.disable_task_neutralization:
            cmd.append("--disable-task-neutralization")
        if args.total_timesteps_override is not None:
            cmd.extend(["--total-timesteps-override", str(args.total_timesteps_override)])

    cmd.extend(passthrough)
    return cmd


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    # Keep each process single-threaded at BLAS/OpenMP level so per-core pinning
    # actually prevents CPU core contention.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["BLIS_NUM_THREADS"] = "1"
    return env


def _start_seed_run(
    *,
    seed: int,
    cmd: list[str],
    core: int,
    log_path: Path,
) -> SeedRun:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")

    def _pin_to_core() -> None:
        os.sched_setaffinity(0, {core})

    process = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=_worker_env(),
        preexec_fn=_pin_to_core,
    )
    return SeedRun(
        seed=seed,
        core=core,
        process=process,
        log_path=log_path,
        log_handle=log_handle,
    )


def _default_log_dir(args: argparse.Namespace) -> Path:
    mode_suffix = "source" if args.mode == "source" else args.mode
    base = args.output_dir if args.mode == "source" else args.outputs_root
    return base / args.task_setting / "multi_seed_logs" / mode_suffix


def main() -> int:
    args, passthrough = _parse_args()

    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError(
            "CPU affinity pinning requires os.sched_setaffinity/os.sched_getaffinity support.",
        )

    seeds = _dedupe_preserve_order(args.seeds)
    if not seeds:
        raise ValueError("No seeds provided. Pass at least one seed via --seeds.")

    core_pool = _resolve_core_pool(args.cores)
    if not core_pool:
        raise RuntimeError("No CPU cores available for scheduling.")

    log_dir = args.log_dir or _default_log_dir(args)

    pending: deque[int] = deque(seeds)
    free_cores: deque[int] = deque(core_pool)
    active: list[SeedRun] = []
    failures: list[tuple[int, int, int, Path]] = []

    print(
        f"Launching {len(seeds)} runs for mode={args.mode} task-setting={args.task_setting} "
        f"with {len(core_pool)} available core(s): {core_pool}",
    )
    if len(core_pool) < len(seeds):
        print("Note: fewer cores than seeds; runs will execute in waves with one run per core.")
    if passthrough:
        print(f"Forwarding extra args to worker scripts: {' '.join(passthrough)}")

    while pending or active:
        while pending and free_cores:
            seed = pending.popleft()
            core = free_cores.popleft()
            cmd = _build_worker_cmd(args, seed=seed, passthrough=passthrough)
            log_path = log_dir / f"seed_{seed}.log"
            run = _start_seed_run(seed=seed, cmd=cmd, core=core, log_path=log_path)
            active.append(run)
            print(f"[start] seed={seed} core={core} pid={run.process.pid} log={log_path}")

        if not active:
            continue

        time.sleep(args.poll_seconds)
        still_active: list[SeedRun] = []
        for run in active:
            return_code = run.process.poll()
            if return_code is None:
                still_active.append(run)
                continue

            run.log_handle.close()
            free_cores.append(run.core)
            if return_code == 0:
                print(f"[done] seed={run.seed} core={run.core} rc=0")
            else:
                print(
                    f"[fail] seed={run.seed} core={run.core} rc={return_code} "
                    f"log={run.log_path}",
                )
                failures.append((run.seed, run.core, return_code, run.log_path))
        active = still_active

    if failures:
        print("\nOne or more runs failed:")
        for seed, core, rc, log_path in failures:
            print(f"  seed={seed} core={core} rc={rc} log={log_path}")
        return 1

    print("\nAll runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
