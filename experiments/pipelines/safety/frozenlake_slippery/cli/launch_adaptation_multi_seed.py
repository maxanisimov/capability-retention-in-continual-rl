"""Run one FrozenLake slippery shield safety adaptation mode across seeds with CPU pinning."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, TextIO

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.safety.frozenlake_slippery.core.paths import (
    default_outputs_root,
    mode_run_dir,
    pipeline_root,
    resolve_source_run_dir,
)


ADAPTATION_MODES = (
    "downstream_unconstrained",
    "downstream_ewc",
    "downstream_rashomon",
    "downstream_safe_line_search",
    "downstream_lagrangian",
)

MODE_TO_REQUIRED_ARTIFACTS = {
    "downstream_unconstrained": ("actor.pt", "critic.pt", "training_data.pt", "run_summary.yaml"),
    "downstream_ewc": ("actor.pt", "critic.pt", "training_data.pt", "run_summary.yaml", "ewc_state.pt"),
    "downstream_rashomon": (
        "actor.pt",
        "critic.pt",
        "training_data.pt",
        "run_summary.yaml",
        "rashomon_dataset.pt",
        "rashomon_param_bounds.pt",
    ),
    "downstream_safe_line_search": (
        "actor.pt",
        "critic.pt",
        "training_data.pt",
        "run_summary.yaml",
        "source_safety_dataset.pt",
    ),
    "downstream_lagrangian": (
        "actor.pt",
        "critic.pt",
        "training_data.pt",
        "run_summary.yaml",
        "source_safety_dataset.pt",
    ),
}

JOB_PENDING = "PENDING"
JOB_RUNNING = "RUNNING"
JOB_SUCCEEDED = "SUCCEEDED"
JOB_FAILED = "FAILED"
JOB_SKIPPED = "SKIPPED"
JOB_BLOCKED = "BLOCKED"


@dataclass
class JobRecord:
    seed: int
    mode: str
    state: str = JOB_PENDING
    core: int | None = None
    scheduled_wave: int | None = None
    command: list[str] = field(default_factory=list)
    log_path: str | None = None
    pid: int | None = None
    return_code: int | None = None
    runtime_seconds: float | None = None
    blocked_reason: str | None = None
    process: subprocess.Popen[bytes] | None = field(default=None, repr=False)
    log_handle: TextIO | None = field(default=None, repr=False)
    start_time: float | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "mode": str(self.mode),
            "state": str(self.state),
            "core": self.core,
            "scheduled_wave": self.scheduled_wave,
            "command": list(self.command),
            "log_path": self.log_path,
            "pid": self.pid,
            "return_code": self.return_code,
            "runtime_seconds": self.runtime_seconds,
            "blocked_reason": self.blocked_reason,
        }


def _dedupe_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run one FrozenLake slippery shield safety adaptation method across multiple seeds "
            "while pinning each active seed run to a unique CPU core."
        ),
    )
    parser.add_argument("--mode", required=True, choices=ADAPTATION_MODES)
    parser.add_argument("--pipeline", "--layout", type=str, dest="layout", default="diagonal_4x4")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--cores",
        type=int,
        nargs="+",
        default=None,
        help="CPU core IDs to use. Defaults to all cores in the current affinity mask.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Optional cap on simultaneous runs. Must be <= selected unique core count.",
    )
    parser.add_argument("--outputs-root", type=Path, default=default_outputs_root())
    parser.add_argument(
        "--source-run-root",
        type=Path,
        default=None,
        help=(
            "Optional root containing <pipeline>/seed_<seed>/noadapt source artifacts. "
            "Defaults to --outputs-root."
        ),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--success-rate",
        type=float,
        default=None,
        help=(
            "Probability that the requested action is executed in slippery FrozenLake. "
            "Forwarded to the adaptation job."
        ),
    )
    parser.add_argument("--resume-policy", choices=["skip_completed", "rerun_all"], default="skip_completed")
    parser.add_argument("--failure-policy", choices=["continue", "fail_fast"], default="continue")
    parser.add_argument("--total-timesteps-override", type=int, default=None)
    parser.add_argument("--ewc-lambda", type=float, default=None)
    parser.add_argument("--fisher-sample-size", type=int, default=None)
    parser.add_argument("--rashomon-n-iters", type=int, default=None)
    parser.add_argument("--inverse-temp-start", type=int, default=None)
    parser.add_argument("--inverse-temp-max", type=int, default=None)
    parser.add_argument("--rashomon-checkpoint", type=int, default=None)
    parser.add_argument("--safe-line-search-max-backtracks", type=int, default=None)
    parser.add_argument("--safe-line-search-backtrack-coef", type=float, default=None)
    parser.add_argument("--lagrangian-lambda-init", type=float, default=None)
    parser.add_argument("--lagrangian-lambda-lr", type=float, default=None)
    parser.add_argument("--lagrangian-lambda-max", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args(argv)


def _resolve_core_pool(requested_cores: list[int] | None, max_parallel: int | None) -> list[int]:
    available_cores = sorted(os.sched_getaffinity(0))
    if requested_cores is None:
        core_pool = available_cores
    else:
        core_pool = _dedupe_preserve_order(list(requested_cores))
        invalid = sorted(set(core_pool) - set(available_cores))
        if invalid:
            raise ValueError(
                f"Requested --cores {invalid} are not available in current affinity mask {available_cores}.",
            )

    if max_parallel is not None:
        if max_parallel <= 0:
            raise ValueError(f"--max-parallel must be > 0, got {max_parallel}.")
        if max_parallel > len(core_pool):
            raise ValueError(
                f"--max-parallel={max_parallel} exceeds selected unique core count {len(core_pool)}.",
            )
        core_pool = core_pool[:max_parallel]

    if not core_pool:
        raise RuntimeError("No CPU cores available for scheduling.")
    return core_pool


def _is_job_complete(outputs_root: Path, layout: str, seed: int, mode: str) -> bool:
    run_dir = mode_run_dir(outputs_root, layout, seed, mode)
    return all((run_dir / artifact).exists() for artifact in MODE_TO_REQUIRED_ARTIFACTS[mode])


def _default_log_root(args: argparse.Namespace) -> Path:
    return args.outputs_root / args.layout / "multi_seed_logs" / "adaptation_parallel" / args.mode


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["BLIS_NUM_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLBACKEND", "Agg")
    return env


def _build_command(args: argparse.Namespace, *, seed: int, passthrough: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        str(pipeline_root() / "run_experiment.py"),
        "--mode",
        str(args.mode),
        "--pipeline",
        str(args.layout),
        "--seed",
        str(seed),
        "--device",
        str(args.device),
        "--outputs-root",
        str(args.outputs_root),
    ]
    if args.source_run_root is not None:
        source_run_dir = resolve_source_run_dir(args.source_run_root, args.layout, seed)
        cmd.extend(["--source-run-dir", str(source_run_dir)])
    if args.total_timesteps_override is not None:
        cmd.extend(["--total-timesteps-override", str(args.total_timesteps_override)])
    if args.success_rate is not None:
        cmd.extend(["--success-rate", str(args.success_rate)])
    if args.mode == "downstream_ewc":
        if args.ewc_lambda is not None:
            cmd.extend(["--ewc-lambda", str(args.ewc_lambda)])
        if args.fisher_sample_size is not None:
            cmd.extend(["--fisher-sample-size", str(args.fisher_sample_size)])
    if args.mode in {"downstream_rashomon", "downstream_safe_line_search", "downstream_lagrangian"}:
        if args.inverse_temp_start is not None:
            cmd.extend(["--inverse-temp-start", str(args.inverse_temp_start)])
        if args.inverse_temp_max is not None:
            cmd.extend(["--inverse-temp-max", str(args.inverse_temp_max)])
    if args.mode == "downstream_rashomon":
        if args.rashomon_n_iters is not None:
            cmd.extend(["--rashomon-n-iters", str(args.rashomon_n_iters)])
        if args.rashomon_checkpoint is not None:
            cmd.extend(["--rashomon-checkpoint", str(args.rashomon_checkpoint)])
    if args.mode == "downstream_safe_line_search":
        if args.safe_line_search_max_backtracks is not None:
            cmd.extend(["--safe-line-search-max-backtracks", str(args.safe_line_search_max_backtracks)])
        if args.safe_line_search_backtrack_coef is not None:
            cmd.extend(["--safe-line-search-backtrack-coef", str(args.safe_line_search_backtrack_coef)])
    if args.mode == "downstream_lagrangian":
        if args.lagrangian_lambda_init is not None:
            cmd.extend(["--lagrangian-lambda-init", str(args.lagrangian_lambda_init)])
        if args.lagrangian_lambda_lr is not None:
            cmd.extend(["--lagrangian-lambda-lr", str(args.lagrangian_lambda_lr)])
        if args.lagrangian_lambda_max is not None:
            cmd.extend(["--lagrangian-lambda-max", str(args.lagrangian_lambda_max)])
    cmd.extend(passthrough)
    return cmd


def _start_job(job: JobRecord, *, core: int, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")

    def _pin_to_core() -> None:
        os.sched_setaffinity(0, {core})

    process = subprocess.Popen(
        job.command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=_worker_env(),
        preexec_fn=_pin_to_core,
    )
    job.core = int(core)
    job.log_path = str(log_path)
    job.pid = int(process.pid)
    job.process = process
    job.log_handle = log_handle
    job.start_time = time.time()
    job.state = JOB_RUNNING


def _write_summary(
    *,
    args: argparse.Namespace,
    jobs: list[JobRecord],
    core_pool: list[int],
    summary_path: Path,
) -> Path:
    payload = {
        "run_settings": {
            "mode": str(args.mode),
            "layout": str(args.layout),
            "seeds": [int(seed) for seed in _dedupe_preserve_order(list(args.seeds))],
            "outputs_root": str(args.outputs_root),
            "source_run_root": None if args.source_run_root is None else str(args.source_run_root),
            "core_pool": [int(core) for core in core_pool],
            "max_parallel": int(len(core_pool)),
            "resume_policy": str(args.resume_policy),
            "failure_policy": str(args.failure_policy),
            "dry_run": bool(args.dry_run),
            "success_rate": args.success_rate,
            "inverse_temp_start": args.inverse_temp_start,
            "inverse_temp_max": args.inverse_temp_max,
            "safe_line_search_max_backtracks": args.safe_line_search_max_backtracks,
            "safe_line_search_backtrack_coef": args.safe_line_search_backtrack_coef,
            "lagrangian_lambda_init": args.lagrangian_lambda_init,
            "lagrangian_lambda_lr": args.lagrangian_lambda_lr,
            "lagrangian_lambda_max": args.lagrangian_lambda_max,
        },
        "jobs": [job.as_dict() for job in jobs],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return summary_path


def _prepare_jobs(
    *,
    args: argparse.Namespace,
    seeds: list[int],
    log_root: Path,
    passthrough: list[str],
) -> tuple[list[JobRecord], deque[JobRecord]]:
    jobs: list[JobRecord] = []
    pending: deque[JobRecord] = deque()
    for seed in seeds:
        job = JobRecord(seed=seed, mode=args.mode)
        job.command = _build_command(args, seed=seed, passthrough=passthrough)
        job.log_path = str(log_root / f"seed_{seed}.log")
        if args.resume_policy == "skip_completed" and _is_job_complete(args.outputs_root, args.layout, seed, args.mode):
            job.state = JOB_SKIPPED
        else:
            pending.append(job)
        jobs.append(job)
    return jobs, pending


def _mark_dry_run_jobs(pending: deque[JobRecord], core_pool: list[int]) -> None:
    for index, job in enumerate(pending):
        job.core = int(core_pool[index % len(core_pool)])
        job.scheduled_wave = int(index // len(core_pool))
        job.state = JOB_SUCCEEDED
        job.return_code = 0
        job.runtime_seconds = 0.0


def _finish_job(job: JobRecord, return_code: int) -> None:
    if job.log_handle is not None:
        job.log_handle.close()
        job.log_handle = None
    job.process = None
    job.return_code = int(return_code)
    if job.start_time is not None:
        job.runtime_seconds = float(time.time() - job.start_time)
    job.state = JOB_SUCCEEDED if return_code == 0 else JOB_FAILED


def main(argv: list[str] | None = None) -> int:
    args, passthrough = _parse_args(argv)
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError(
            "CPU affinity pinning requires os.sched_setaffinity/os.sched_getaffinity support.",
        )

    seeds = _dedupe_preserve_order(list(args.seeds))
    if not seeds:
        raise ValueError("No seeds provided. Pass at least one seed via --seeds.")

    core_pool = _resolve_core_pool(args.cores, args.max_parallel)
    log_root = args.log_dir or _default_log_root(args)
    summary_path = log_root / "summary.yaml"
    jobs, pending = _prepare_jobs(args=args, seeds=seeds, log_root=log_root, passthrough=passthrough)

    print(
        f"Launching mode={args.mode} for {len(seeds)} seed(s) on "
        f"{len(core_pool)} unique CPU core(s): {core_pool}",
    )
    if len(pending) < len(jobs):
        print(f"Skipping {len(jobs) - len(pending)} completed run(s).")
    if len(core_pool) < len(pending):
        print("Fewer cores than pending seeds; runs will execute in waves with one active seed per core.")
    if passthrough:
        print(f"Forwarding extra args to run_experiment.py: {' '.join(passthrough)}")

    if args.dry_run:
        _mark_dry_run_jobs(pending, core_pool)
        for job in jobs:
            print(f"[dry-run] seed={job.seed} state={job.state} core={job.core} log={job.log_path}")
            if job.state == JOB_SUCCEEDED:
                print("  " + " ".join(job.command))
        summary = _write_summary(args=args, jobs=jobs, core_pool=core_pool, summary_path=summary_path)
        print(f"Summary written to {summary}")
        return 0

    free_cores: deque[int] = deque(core_pool)
    active: list[JobRecord] = []
    had_failure = False
    stop_launching = False

    while pending or active:
        while pending and free_cores and not stop_launching:
            job = pending.popleft()
            core = free_cores.popleft()
            assert job.log_path is not None
            _start_job(job, core=core, log_path=Path(job.log_path))
            active.append(job)
            print(f"[start] seed={job.seed} core={core} pid={job.pid} log={job.log_path}")
            _write_summary(args=args, jobs=jobs, core_pool=core_pool, summary_path=summary_path)

        if not active:
            break

        time.sleep(max(float(args.poll_seconds), 0.1))
        still_active: list[JobRecord] = []
        for job in active:
            assert job.process is not None
            return_code = job.process.poll()
            if return_code is None:
                still_active.append(job)
                continue

            assert job.core is not None
            free_cores.append(job.core)
            _finish_job(job, int(return_code))
            print(f"[{'done' if return_code == 0 else 'fail'}] seed={job.seed} core={job.core} rc={return_code}")
            if return_code != 0:
                had_failure = True
                if args.failure_policy == "fail_fast":
                    stop_launching = True
        active = still_active
        _write_summary(args=args, jobs=jobs, core_pool=core_pool, summary_path=summary_path)

    if stop_launching:
        for job in pending:
            job.state = JOB_BLOCKED
            job.blocked_reason = "fail_fast_triggered"
        pending.clear()

    summary = _write_summary(args=args, jobs=jobs, core_pool=core_pool, summary_path=summary_path)
    print(f"Summary written to {summary}")
    if had_failure:
        print("One or more runs failed. Check the per-seed logs above.")
        return 1
    print("All scheduled runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
