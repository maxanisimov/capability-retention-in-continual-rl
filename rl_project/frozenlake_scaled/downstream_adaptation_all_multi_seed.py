"""Run all downstream adaptation methods across multiple seeds with CPU-core pinning.

Each seed-method pair is treated as one independent job:
  - unconstrained
  - ewc
  - rashomon

At most one active job is placed on each CPU core, so concurrently running jobs
never clash on a core.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, TypeVar


ALL_METHODS: tuple[str, ...] = ("unconstrained", "ewc", "rashomon")
T = TypeVar("T")


@dataclass
class JobSpec:
    method: str
    seed: int


@dataclass
class ActiveRun:
    job: JobSpec
    core: int
    process: subprocess.Popen[bytes]
    log_path: Path
    log_handle: TextIO


def _dedupe_preserve_order(values: list[T]) -> list[T]:
    seen: set[T] = set()
    out: list[T] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Run downstream adaptation methods (unconstrained, EWC, Rashomon) for one layout "
            "across multiple seeds, pinned one active job per CPU core."
        ),
    )
    parser.add_argument("--layout", type=str, default="diagonal_30x30")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=list(ALL_METHODS),
        default=list(ALL_METHODS),
        help="Subset/order of methods to run (default: unconstrained ewc rashomon).",
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
        "--ewc-settings-file",
        type=Path,
        default=script_dir / "settings" / "downstream_adaptation_settings_ewc.yaml",
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
        "--disable-task-neutralization",
        action="store_true",
        help="Forward --disable-task-neutralization to each job.",
    )

    # Unconstrained overrides.
    parser.add_argument(
        "--unconstrained-total-timesteps-override",
        type=int,
        default=None,
        help="Optional override for unconstrained downstream PPO total timesteps.",
    )

    # EWC overrides.
    parser.add_argument(
        "--ewc-run-subdir",
        type=str,
        default="downstream_ewc",
        help="Output subdirectory under outputs/<layout>/seed_<seed>/ for EWC artifacts.",
    )
    parser.add_argument(
        "--ewc-total-timesteps-override",
        type=int,
        default=None,
        help="Optional override for EWC downstream PPO total timesteps.",
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

    # Rashomon overrides.
    parser.add_argument(
        "--rashomon-run-subdir",
        type=str,
        default="downstream_rashomon",
        help="Output subdirectory under outputs/<layout>/seed_<seed>/ for Rashomon artifacts.",
    )
    parser.add_argument("--rashomon-n-iters", type=int, default=None)
    parser.add_argument(
        "--rashomon-min-hard-spec",
        type=float,
        default=None,
        help="Optional override for minimum hard specification level.",
    )
    parser.add_argument(
        "--rashomon-surrogate-aggregation",
        type=str,
        choices=["mean", "min"],
        default=None,
        help="Optional override for surrogate aggregation mode.",
    )
    parser.add_argument("--inverse-temp-start", type=int, default=None)
    parser.add_argument("--inverse-temp-max", type=int, default=None)
    parser.add_argument("--rashomon-checkpoint", type=int, default=None)

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=(
            "Directory for launcher logs. "
            "Default: <outputs-root>/<layout>/multi_seed_logs/downstream_all "
            "(with per-method subdirectories)."
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


def _source_run_dir(args: argparse.Namespace, seed: int) -> Path | None:
    if args.source_run_root is None:
        return None
    return args.source_run_root / args.layout / f"seed_{seed}" / "source"


def _base_cmd(args: argparse.Namespace, script_name: str, seed: int) -> list[str]:
    script_path = Path(__file__).resolve().parent / script_name
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
        "--outputs-root",
        str(args.outputs_root),
    ]

    source_run_dir = _source_run_dir(args, seed)
    if source_run_dir is not None:
        cmd.extend(["--source-run-dir", str(source_run_dir)])

    if args.disable_task_neutralization:
        cmd.append("--disable-task-neutralization")

    return cmd


def _build_job_cmd(args: argparse.Namespace, job: JobSpec) -> list[str]:
    if job.method == "unconstrained":
        cmd = _base_cmd(args, "downstream_adaptation_unconstrained.py", job.seed)
        if args.unconstrained_total_timesteps_override is not None:
            cmd.extend(
                [
                    "--total-timesteps-override",
                    str(args.unconstrained_total_timesteps_override),
                ],
            )
        return cmd

    if job.method == "ewc":
        cmd = _base_cmd(args, "downstream_adaptation_ewc.py", job.seed)
        cmd.extend(
            [
                "--ewc-settings-file",
                str(args.ewc_settings_file),
                "--run-subdir",
                args.ewc_run_subdir,
                "--fisher-sample-size",
                str(args.fisher_sample_size),
            ],
        )
        if args.ewc_total_timesteps_override is not None:
            cmd.extend(["--total-timesteps-override", str(args.ewc_total_timesteps_override)])
        if args.ewc_lambda_override is not None:
            cmd.extend(["--ewc-lambda-override", str(args.ewc_lambda_override)])
        if args.ewc_apply_to_critic:
            cmd.append("--ewc-apply-to-critic")
        return cmd

    if job.method == "rashomon":
        cmd = _base_cmd(args, "downstream_adaptation_rashomon.py", job.seed)
        cmd.extend(
            [
                "--rashomon-settings-file",
                str(args.rashomon_settings_file),
                "--run-subdir",
                args.rashomon_run_subdir,
            ],
        )
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

    raise ValueError(f"Unsupported method '{job.method}'.")


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("BLIS_NUM_THREADS", "1")
    return env


def _start_job(
    *,
    job: JobSpec,
    cmd: list[str],
    core: int,
    log_path: Path,
) -> ActiveRun:
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
    return ActiveRun(
        job=job,
        core=core,
        process=process,
        log_path=log_path,
        log_handle=log_handle,
    )


def _make_jobs(methods: list[str], seeds: list[int]) -> list[JobSpec]:
    # Seed-major ordering: for each seed, launch selected methods in requested order.
    jobs: list[JobSpec] = []
    for seed in seeds:
        for method in methods:
            jobs.append(JobSpec(method=method, seed=seed))
    return jobs


def main() -> int:
    args = _parse_args()
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError(
            "CPU affinity pinning requires os.sched_setaffinity/os.sched_getaffinity support on this platform.",
        )

    seeds = _dedupe_preserve_order(args.seeds)
    methods = _dedupe_preserve_order(args.methods)
    if not seeds:
        raise ValueError("No seeds provided. Pass at least one seed via --seeds.")
    if not methods:
        raise ValueError("No methods selected. Pass at least one method via --methods.")

    core_pool = _resolve_core_pool(args.cores)
    if not core_pool:
        raise RuntimeError("No CPU cores available for scheduling.")

    jobs = _make_jobs(methods=methods, seeds=seeds)
    log_dir = args.log_dir or (args.outputs_root / args.layout / "multi_seed_logs" / "downstream_all")

    pending: deque[JobSpec] = deque(jobs)
    free_cores: deque[int] = deque(core_pool)
    active: list[ActiveRun] = []
    failures: list[tuple[str, int, int, int, Path]] = []

    print(
        f"Launching {len(jobs)} downstream jobs for layout={args.layout} "
        f"(methods={methods}, seeds={seeds}) with {len(core_pool)} available core(s): {core_pool}",
    )
    if len(core_pool) < len(jobs):
        print("Note: fewer cores than jobs; runs will execute in waves with one active run per core.")

    while pending or active:
        while pending and free_cores:
            job = pending.popleft()
            core = free_cores.popleft()
            cmd = _build_job_cmd(args, job)
            log_path = log_dir / job.method / f"seed_{job.seed}.log"
            run = _start_job(job=job, cmd=cmd, core=core, log_path=log_path)
            active.append(run)
            print(
                f"[launch] method={job.method} seed={job.seed} core={core} "
                f"log={log_path}",
            )

        finished: list[ActiveRun] = []
        for run in active:
            return_code = run.process.poll()
            if return_code is None:
                continue
            run.log_handle.close()
            free_cores.append(run.core)
            if return_code == 0:
                print(
                    f"[ok] method={run.job.method} seed={run.job.seed} "
                    f"core={run.core} log={run.log_path}",
                )
            else:
                print(
                    f"[failed] method={run.job.method} seed={run.job.seed} "
                    f"core={run.core} exit={return_code} log={run.log_path}",
                )
                failures.append((run.job.method, run.job.seed, run.core, return_code, run.log_path))
            finished.append(run)

        for run in finished:
            active.remove(run)

        if pending or active:
            time.sleep(max(args.poll_seconds, 0.1))

    if failures:
        print("\nCompleted with failures:")
        for method, seed, core, return_code, log_path in failures:
            print(
                f"  - method={method}, seed={seed}, core={core}, "
                f"exit={return_code}, log={log_path}",
            )
        return 1

    print("\nAll downstream method-seed jobs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
