"""Run LunarLander EWC downstream adaptation across multiple seeds with CPU pinning."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import TextIO


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
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Run downstream_adaptation_ewc.py for one task setting across multiple seeds, "
            "pinned one active run per CPU core."
        ),
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
        default=script_dir / "settings" / "task_settings.yaml",
        help="Path to LunarLander task settings YAML.",
    )
    parser.add_argument(
        "--adapt-settings-file",
        type=Path,
        default=script_dir / "settings" / "downstream_adaptation_settings_ppo.yaml",
        help="Path to shared downstream adaptation settings YAML.",
    )
    parser.add_argument(
        "--ewc-settings-file",
        type=Path,
        default=script_dir / "settings" / "downstream_adaptation_settings_ewc.yaml",
        help="Path to EWC-specific downstream adaptation settings YAML.",
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
        default=script_dir / "outputs",
        help="Outputs root directory forwarded to adaptation runs.",
    )
    parser.add_argument(
        "--source-run-root",
        type=Path,
        default=None,
        help=(
            "Optional root used to derive per-seed source checkpoints as "
            "<source-run-root>/<task-setting>/seed_<seed>/source."
        ),
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for adaptation runs.")
    parser.add_argument(
        "--run-subdir",
        type=str,
        default="downstream_ewc",
        help="Subdirectory name used by downstream_adaptation_ewc.py.",
    )
    parser.add_argument(
        "--disable-task-neutralization",
        action="store_true",
        help="Forward --disable-task-neutralization to each seed run.",
    )
    parser.add_argument(
        "--total-timesteps-override",
        type=int,
        default=None,
        help="Optional override for downstream PPO total timesteps.",
    )
    parser.add_argument(
        "--ewc-lambda-override",
        type=float,
        default=None,
        help="Optional override for EWC lambda.",
    )
    parser.add_argument(
        "--fisher-sample-size",
        type=int,
        default=10_000,
        help="Maximum number of source states used to estimate Fisher diagonal.",
    )
    parser.add_argument(
        "--ewc-apply-to-critic",
        action="store_true",
        help="Also apply EWC regularization to critic parameters.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-seed launcher logs. "
            "Default: <outputs-root>/<task-setting>/multi_seed_logs/downstream_ewc."
        ),
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


def _build_worker_cmd(
    args: argparse.Namespace,
    seed: int,
    passthrough: list[str],
) -> list[str]:
    script_path = Path(__file__).resolve().parent / "downstream_adaptation_ewc.py"
    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "--task-setting",
        str(args.task_setting),
        "--seed",
        str(seed),
        "--device",
        str(args.device),
        "--task-settings-file",
        str(args.task_settings_file),
        "--adapt-settings-file",
        str(args.adapt_settings_file),
        "--ewc-settings-file",
        str(args.ewc_settings_file),
        "--outputs-root",
        str(args.outputs_root),
        "--run-subdir",
        str(args.run_subdir),
        "--fisher-sample-size",
        str(args.fisher_sample_size),
    ]

    if args.source_run_root is not None:
        source_run_dir = args.source_run_root / args.task_setting / f"seed_{seed}" / "source"
        cmd.extend(["--source-run-dir", str(source_run_dir)])

    if args.disable_task_neutralization:
        cmd.append("--disable-task-neutralization")

    if args.total_timesteps_override is not None:
        cmd.extend(["--total-timesteps-override", str(args.total_timesteps_override)])

    if args.ewc_lambda_override is not None:
        cmd.extend(["--ewc-lambda-override", str(args.ewc_lambda_override)])

    if args.ewc_apply_to_critic:
        cmd.append("--ewc-apply-to-critic")

    cmd.extend(passthrough)
    return cmd


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    # Force single-threaded BLAS/OpenMP to avoid cross-core contention.
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

    log_dir = args.log_dir or (
        args.outputs_root / args.task_setting / "multi_seed_logs" / "downstream_ewc"
    )
    pending: deque[int] = deque(seeds)
    free_cores: deque[int] = deque(core_pool)
    active: list[SeedRun] = []
    failures: list[tuple[int, int, int, Path]] = []

    print(
        f"Launching {len(seeds)} downstream-EWC runs for task-setting={args.task_setting} "
        f"with {len(core_pool)} available core(s): {core_pool}",
    )
    if len(core_pool) < len(seeds):
        print("Note: fewer cores than seeds; runs will execute in waves with one active run per core.")
    if passthrough:
        print(f"Forwarding extra args to downstream_adaptation_ewc.py: {' '.join(passthrough)}")

    while pending or active:
        while pending and free_cores:
            seed = pending.popleft()
            core = free_cores.popleft()
            cmd = _build_worker_cmd(args, seed, passthrough)
            log_path = log_dir / f"seed_{seed}.log"
            run = _start_seed_run(seed=seed, cmd=cmd, core=core, log_path=log_path)
            active.append(run)
            print(f"[launch] seed={seed} core={core} log={log_path}")

        finished: list[SeedRun] = []
        for run in active:
            return_code = run.process.poll()
            if return_code is None:
                continue
            run.log_handle.close()
            free_cores.append(run.core)
            if return_code == 0:
                print(f"[ok] seed={run.seed} core={run.core} log={run.log_path}")
            else:
                print(f"[failed] seed={run.seed} core={run.core} exit={return_code} log={run.log_path}")
                failures.append((run.seed, run.core, return_code, run.log_path))
            finished.append(run)

        for run in finished:
            active.remove(run)

        if pending or active:
            time.sleep(max(args.poll_seconds, 0.1))

    if failures:
        print("\nCompleted with failures:")
        for seed, core, return_code, log_path in failures:
            print(f"  - seed={seed}, core={core}, exit={return_code}, log={log_path}")
        return 1

    print("\nAll downstream EWC seed runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
