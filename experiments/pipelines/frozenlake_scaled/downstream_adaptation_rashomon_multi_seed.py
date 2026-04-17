"""Run Rashomon downstream adaptation across multiple seeds with CPU-core pinning."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
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


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Run downstream_adaptation_rashomon.py for one layout across multiple seeds, "
            "pinned one active run per CPU core."
        ),
    )
    parser.add_argument("--layout", type=str, default="diagonal_30x30")
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
        help="Optional explicit CPU core IDs to use. Must be within this process affinity mask.",
    )
    parser.add_argument(
        "--source-env-file",
        type=Path,
        default=script_dir / "settings" / "source_envs.yaml",
    )
    parser.add_argument(
        "--downstream-env-file",
        type=Path,
        default=script_dir / "settings" / "downstream_envs.yaml",
    )
    parser.add_argument(
        "--source-settings-file",
        type=Path,
        default=script_dir / "settings" / "train_source_policy_settings.yaml",
    )
    parser.add_argument(
        "--adapt-settings-file",
        type=Path,
        default=script_dir / "settings" / "downstream_adaptation_settings_ppo.yaml",
    )
    parser.add_argument(
        "--rashomon-settings-file",
        type=Path,
        default=script_dir / "settings" / "downstream_adaptation_settings_rashomon.yaml",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=script_dir / "outputs",
    )
    parser.add_argument(
        "--source-run-root",
        type=Path,
        default=None,
        help=(
            "Optional root used to derive per-seed source checkpoint paths as "
            "<source-run-root>/<layout>/seed_<seed>/source."
        ),
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["tanh", "relu"],
        default="relu",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--run-subdir",
        type=str,
        default="downstream_rashomon",
        help="Output subdirectory under outputs/<layout>/seed_<seed>/ for Rashomon artifacts.",
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
        "--rashomon-n-iters",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--rashomon-min-acc-limit",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--rashomon-aggregation",
        type=str,
        choices=["mean", "min"],
        default=None,
    )
    parser.add_argument(
        "--inverse-temp-start",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--inverse-temp-max",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--rashomon-checkpoint",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-seed launcher logs. "
            "Default: <outputs-root>/<layout>/multi_seed_logs/downstream_rashomon."
        ),
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="Polling interval for child process completion checks.",
    )
    return parser.parse_args()


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


def _build_worker_cmd(args: argparse.Namespace, seed: int) -> list[str]:
    script_path = Path(__file__).resolve().parent / "downstream_adaptation_rashomon.py"
    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "--layout",
        args.layout,
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--activation",
        args.activation,
        "--source-env-file",
        str(args.source_env_file),
        "--downstream-env-file",
        str(args.downstream_env_file),
        "--source-settings-file",
        str(args.source_settings_file),
        "--adapt-settings-file",
        str(args.adapt_settings_file),
        "--rashomon-settings-file",
        str(args.rashomon_settings_file),
        "--outputs-root",
        str(args.outputs_root),
        "--run-subdir",
        args.run_subdir,
    ]

    if args.source_run_root is not None:
        source_run_dir = args.source_run_root / args.layout / f"seed_{seed}" / "source"
        cmd.extend(["--source-run-dir", str(source_run_dir)])

    if args.disable_task_neutralization:
        cmd.append("--disable-task-neutralization")

    if args.total_timesteps_override is not None:
        cmd.extend(["--total-timesteps-override", str(args.total_timesteps_override)])

    if args.rashomon_n_iters is not None:
        cmd.extend(["--rashomon-n-iters", str(args.rashomon_n_iters)])
    if args.rashomon_min_acc_limit is not None:
        cmd.extend(["--rashomon-min-acc-limit", str(args.rashomon_min_acc_limit)])
    if args.rashomon_aggregation is not None:
        cmd.extend(["--rashomon-aggregation", args.rashomon_aggregation])
    if args.inverse_temp_start is not None:
        cmd.extend(["--inverse-temp-start", str(args.inverse_temp_start)])
    if args.inverse_temp_max is not None:
        cmd.extend(["--inverse-temp-max", str(args.inverse_temp_max)])
    if args.rashomon_checkpoint is not None:
        cmd.extend(["--rashomon-checkpoint", str(args.rashomon_checkpoint)])

    return cmd


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("BLIS_NUM_THREADS", "1")
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
    args = _parse_args()
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError(
            "CPU affinity pinning requires os.sched_setaffinity/os.sched_getaffinity support on this platform.",
        )

    seeds = _dedupe_preserve_order(args.seeds)
    if not seeds:
        raise ValueError("No seeds provided. Pass at least one seed via --seeds.")

    core_pool = _resolve_core_pool(args.cores)
    if not core_pool:
        raise RuntimeError("No CPU cores available for scheduling.")

    log_dir = args.log_dir or (args.outputs_root / args.layout / "multi_seed_logs" / "downstream_rashomon")
    pending: deque[int] = deque(seeds)
    free_cores: deque[int] = deque(core_pool)
    active: list[SeedRun] = []
    failures: list[tuple[int, int, int, Path]] = []

    print(
        f"Launching {len(seeds)} Rashomon downstream runs for layout={args.layout} "
        f"with {len(core_pool)} available core(s): {core_pool}",
    )
    if len(core_pool) < len(seeds):
        print("Note: fewer cores than seeds; runs will execute in waves with one active run per core.")

    while pending or active:
        while pending and free_cores:
            seed = pending.popleft()
            core = free_cores.popleft()
            cmd = _build_worker_cmd(args, seed)
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

    print("\nAll Rashomon downstream seed runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
