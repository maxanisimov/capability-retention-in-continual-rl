"""Launch source plus downstream FrozenLake shield safety modes across seeds."""

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

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.frozenlake_shield_safety.core.paths import default_outputs_root, mode_run_dir, pipeline_root


SOURCE_MODE = "source"
DOWNSTREAM_MODES = (
    "downstream_unconstrained",
    "downstream_ewc",
    "downstream_rashomon",
    "downstream_safe_line_search",
    "downstream_lagrangian",
)
ALL_MODES = (SOURCE_MODE, *DOWNSTREAM_MODES)

JOB_PENDING = "PENDING"
JOB_RUNNING = "RUNNING"
JOB_SUCCEEDED = "SUCCEEDED"
JOB_FAILED = "FAILED"
JOB_SKIPPED = "SKIPPED"
JOB_BLOCKED = "BLOCKED"

MODE_TO_REQUIRED_ARTIFACTS = {
    "source": (
        "actor.pt",
        "critic.pt",
        "training_data.pt",
        "run_summary.yaml",
        "rashomon_dataset.pt",
        "shield.pt",
        "shield_info.pt",
    ),
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
            "seed": self.seed,
            "mode": self.mode,
            "state": self.state,
            "core": self.core,
            "scheduled_wave": self.scheduled_wave,
            "command": self.command,
            "log_path": self.log_path,
            "pid": self.pid,
            "return_code": self.return_code,
            "runtime_seconds": self.runtime_seconds,
            "blocked_reason": self.blocked_reason,
        }


@dataclass
class SeedRun:
    seed: int
    jobs: list[JobRecord]
    cursor: int = 0
    core: int | None = None
    scheduled_wave: int | None = None
    active_job: JobRecord | None = None


def _dedupe_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source plus downstream modes for the FrozenLake shield-safety pipeline.",
    )
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
        help="Optional cap on simultaneous seed pipelines. Must be <= selected unique core count.",
    )
    parser.add_argument("--outputs-root", type=Path, default=default_outputs_root())
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--resume-policy", choices=["skip_completed", "rerun_all"], default="skip_completed")
    parser.add_argument("--failure-policy", choices=["continue", "fail_fast", "stop_seed"], default="continue")
    parser.add_argument("--total-timesteps-override", type=int, default=None)
    parser.add_argument("--rashomon-n-iters", type=int, default=None)
    parser.add_argument("--inverse-temp-start", type=int, default=None)
    parser.add_argument("--inverse-temp-max", type=int, default=None)
    parser.add_argument("--safe-line-search-max-backtracks", type=int, default=None)
    parser.add_argument("--safe-line-search-backtrack-coef", type=float, default=None)
    parser.add_argument("--lagrangian-lambda-init", type=float, default=None)
    parser.add_argument("--lagrangian-lambda-lr", type=float, default=None)
    parser.add_argument("--lagrangian-lambda-max", type=float, default=None)
    parser.add_argument("--shield-type", choices=["deterministic", "probabilistic"], default=None)
    parser.add_argument("--shield-risk-threshold", type=float, default=None)
    parser.add_argument("--shield-theta", type=float, default=None)
    parser.add_argument("--shield-max-vi-steps", type=int, default=None)
    parser.add_argument("--unsafe-cost-threshold", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


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


def _build_command(args: argparse.Namespace, *, seed: int, mode: str) -> list[str]:
    cmd = [
        sys.executable,
        str(pipeline_root() / "run_experiment.py"),
        "--mode",
        mode,
        "--pipeline",
        args.layout,
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--outputs-root",
        str(args.outputs_root),
    ]
    if args.total_timesteps_override is not None:
        cmd.extend(["--total-timesteps-override", str(args.total_timesteps_override)])
    if mode == "source":
        if args.shield_type is not None:
            cmd.extend(["--shield-type", str(args.shield_type)])
        if args.shield_risk_threshold is not None:
            cmd.extend(["--shield-risk-threshold", str(args.shield_risk_threshold)])
        if args.shield_theta is not None:
            cmd.extend(["--shield-theta", str(args.shield_theta)])
        if args.shield_max_vi_steps is not None:
            cmd.extend(["--shield-max-vi-steps", str(args.shield_max_vi_steps)])
        if args.unsafe_cost_threshold is not None:
            cmd.extend(["--unsafe-cost-threshold", str(args.unsafe_cost_threshold)])
    if mode == "downstream_rashomon" and args.rashomon_n_iters is not None:
        cmd.extend(["--rashomon-n-iters", str(args.rashomon_n_iters)])
    if mode in {"downstream_rashomon", "downstream_safe_line_search", "downstream_lagrangian"}:
        if args.inverse_temp_start is not None:
            cmd.extend(["--inverse-temp-start", str(args.inverse_temp_start)])
        if args.inverse_temp_max is not None:
            cmd.extend(["--inverse-temp-max", str(args.inverse_temp_max)])
    if mode == "downstream_safe_line_search":
        if args.safe_line_search_max_backtracks is not None:
            cmd.extend(["--safe-line-search-max-backtracks", str(args.safe_line_search_max_backtracks)])
        if args.safe_line_search_backtrack_coef is not None:
            cmd.extend(["--safe-line-search-backtrack-coef", str(args.safe_line_search_backtrack_coef)])
    if mode == "downstream_lagrangian":
        if args.lagrangian_lambda_init is not None:
            cmd.extend(["--lagrangian-lambda-init", str(args.lagrangian_lambda_init)])
        if args.lagrangian_lambda_lr is not None:
            cmd.extend(["--lagrangian-lambda-lr", str(args.lagrangian_lambda_lr)])
        if args.lagrangian_lambda_max is not None:
            cmd.extend(["--lagrangian-lambda-max", str(args.lagrangian_lambda_max)])
    return cmd


def _log_path(args: argparse.Namespace, *, seed: int, mode: str) -> Path:
    log_root = args.log_dir or (args.outputs_root / args.layout / "multi_seed_logs" / "full_pipeline")
    return log_root / mode / f"seed_{seed}.log"


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


def _write_summary(args: argparse.Namespace, jobs: list[JobRecord], core_pool: list[int]) -> Path:
    summary_path = args.outputs_root / args.layout / "multi_seed_logs" / "full_pipeline" / "summary.yaml"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_settings": {
            "layout": args.layout,
            "seeds": [int(seed) for seed in _dedupe_preserve_order(list(args.seeds))],
            "outputs_root": str(args.outputs_root),
            "core_pool": [int(core) for core in core_pool],
            "max_parallel": int(len(core_pool)),
            "resume_policy": str(args.resume_policy),
            "failure_policy": str(args.failure_policy),
            "dry_run": bool(args.dry_run),
            "inverse_temp_start": args.inverse_temp_start,
            "inverse_temp_max": args.inverse_temp_max,
            "safe_line_search_max_backtracks": args.safe_line_search_max_backtracks,
            "safe_line_search_backtrack_coef": args.safe_line_search_backtrack_coef,
            "lagrangian_lambda_init": args.lagrangian_lambda_init,
            "lagrangian_lambda_lr": args.lagrangian_lambda_lr,
            "lagrangian_lambda_max": args.lagrangian_lambda_max,
            "shield_type": args.shield_type,
            "shield_risk_threshold": args.shield_risk_threshold,
            "shield_theta": args.shield_theta,
            "shield_max_vi_steps": args.shield_max_vi_steps,
            "unsafe_cost_threshold": args.unsafe_cost_threshold,
        },
        "jobs": [job.as_dict() for job in jobs],
    }
    summary_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return summary_path


def _start_job(job: JobRecord, *, core: int, scheduled_wave: int, log_path: Path) -> None:
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
    job.scheduled_wave = int(scheduled_wave)
    job.log_path = str(log_path)
    job.pid = int(process.pid)
    job.process = process
    job.log_handle = log_handle
    job.start_time = time.time()
    job.state = JOB_RUNNING


def _finish_job(job: JobRecord, return_code: int) -> None:
    if job.log_handle is not None:
        job.log_handle.close()
        job.log_handle = None
    job.process = None
    job.return_code = int(return_code)
    if job.start_time is not None:
        job.runtime_seconds = float(time.time() - job.start_time)
    job.state = JOB_SUCCEEDED if return_code == 0 else JOB_FAILED


def _prepare_seed_runs(args: argparse.Namespace, seeds: list[int]) -> tuple[list[JobRecord], deque[SeedRun]]:
    jobs: list[JobRecord] = []
    pending_seed_runs: deque[SeedRun] = deque()
    for seed in seeds:
        seed_jobs: list[JobRecord] = []
        for mode in ALL_MODES:
            job = JobRecord(seed=seed, mode=mode)
            job.command = _build_command(args, seed=seed, mode=mode)
            job.log_path = str(_log_path(args, seed=seed, mode=mode))
            if args.resume_policy == "skip_completed" and _is_job_complete(args.outputs_root, args.layout, seed, mode):
                job.state = JOB_SKIPPED
            seed_jobs.append(job)
            jobs.append(job)
        if any(job.state == JOB_PENDING for job in seed_jobs):
            pending_seed_runs.append(SeedRun(seed=seed, jobs=seed_jobs))
    return jobs, pending_seed_runs


def _block_pending_jobs(seed_run: SeedRun, reason: str) -> None:
    for job in seed_run.jobs[seed_run.cursor :]:
        if job.state == JOB_PENDING:
            job.state = JOB_BLOCKED
            job.blocked_reason = reason


def _start_next_job(seed_run: SeedRun) -> bool:
    assert seed_run.core is not None
    assert seed_run.scheduled_wave is not None
    while seed_run.cursor < len(seed_run.jobs):
        job = seed_run.jobs[seed_run.cursor]
        if job.state == JOB_SKIPPED:
            seed_run.cursor += 1
            continue
        if job.state != JOB_PENDING:
            seed_run.cursor += 1
            continue
        if job.log_path is None:
            raise RuntimeError(f"Missing log path for seed={job.seed}, mode={job.mode}.")
        _start_job(
            job,
            core=seed_run.core,
            scheduled_wave=seed_run.scheduled_wave,
            log_path=Path(job.log_path),
        )
        seed_run.active_job = job
        return True
    seed_run.active_job = None
    return False


def _mark_dry_run_jobs(seed_runs: list[SeedRun], core_pool: list[int]) -> None:
    scheduled_seed_index = 0
    for seed_run in seed_runs:
        if not any(job.state == JOB_PENDING for job in seed_run.jobs):
            continue
        core = int(core_pool[scheduled_seed_index % len(core_pool)])
        scheduled_wave = int(scheduled_seed_index // len(core_pool))
        scheduled_seed_index += 1
        for job in seed_run.jobs:
            if job.state == JOB_PENDING:
                job.core = core
                job.scheduled_wave = scheduled_wave
                job.state = JOB_SUCCEEDED
                job.return_code = 0
                job.runtime_seconds = 0.0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError(
            "CPU affinity pinning requires os.sched_setaffinity/os.sched_getaffinity support.",
        )

    seeds = _dedupe_preserve_order(list(args.seeds))
    if not seeds:
        raise ValueError("No seeds provided. Pass at least one seed via --seeds.")

    core_pool = _resolve_core_pool(args.cores, args.max_parallel)
    jobs, pending_seed_runs = _prepare_seed_runs(args, seeds)

    print(
        f"Launching full pipeline for {len(seeds)} seed(s) on "
        f"{len(core_pool)} unique CPU core(s): {core_pool}",
    )
    skipped_count = sum(1 for job in jobs if job.state == JOB_SKIPPED)
    if skipped_count:
        print(f"Skipping {skipped_count} completed job(s).")
    if len(core_pool) < len(pending_seed_runs):
        print("Fewer cores than pending seeds; seed pipelines will execute in waves with no shared active core.")

    if args.dry_run:
        _mark_dry_run_jobs([*pending_seed_runs], core_pool)
        for job in jobs:
            print(
                f"[dry-run] seed={job.seed} mode={job.mode} state={job.state} "
                f"core={job.core} log={job.log_path}",
            )
            if job.state == JOB_SUCCEEDED:
                print("  " + " ".join(job.command))
        summary_path = _write_summary(args, jobs, core_pool)
        print(f"Summary written to {summary_path}")
        return 0

    free_cores: deque[int] = deque(core_pool)
    active_seed_runs: list[SeedRun] = []
    had_failure = False
    stop_launching = False
    scheduled_seed_count = 0

    while pending_seed_runs or active_seed_runs:
        while pending_seed_runs and free_cores and not stop_launching:
            seed_run = pending_seed_runs.popleft()
            seed_run.core = int(free_cores.popleft())
            seed_run.scheduled_wave = int(scheduled_seed_count // len(core_pool))
            scheduled_seed_count += 1
            if _start_next_job(seed_run):
                active_seed_runs.append(seed_run)
                assert seed_run.active_job is not None
                print(
                    f"[start] seed={seed_run.seed} mode={seed_run.active_job.mode} "
                    f"core={seed_run.core} pid={seed_run.active_job.pid} log={seed_run.active_job.log_path}",
                )
            else:
                assert seed_run.core is not None
                free_cores.append(seed_run.core)
            _write_summary(args, jobs, core_pool)

        if not active_seed_runs:
            break

        time.sleep(max(float(args.poll_seconds), 0.1))
        still_active: list[SeedRun] = []
        for seed_run in active_seed_runs:
            job = seed_run.active_job
            assert job is not None
            assert job.process is not None
            return_code = job.process.poll()
            if return_code is None:
                still_active.append(seed_run)
                continue

            _finish_job(job, int(return_code))
            print(
                f"[{'done' if return_code == 0 else 'fail'}] seed={job.seed} "
                f"mode={job.mode} core={job.core} rc={return_code}",
            )
            seed_run.cursor += 1
            seed_run.active_job = None

            if return_code != 0:
                had_failure = True
                if job.mode == SOURCE_MODE:
                    _block_pending_jobs(seed_run, "source_failed")
                elif args.failure_policy == "stop_seed":
                    _block_pending_jobs(seed_run, f"{job.mode}_failed")
                if args.failure_policy == "fail_fast":
                    stop_launching = True
                    _block_pending_jobs(seed_run, "fail_fast_triggered")

            if stop_launching:
                _block_pending_jobs(seed_run, "fail_fast_triggered")

            if any(job.state == JOB_PENDING for job in seed_run.jobs[seed_run.cursor :]):
                if not stop_launching and _start_next_job(seed_run):
                    assert seed_run.active_job is not None
                    still_active.append(seed_run)
                    print(
                        f"[start] seed={seed_run.seed} mode={seed_run.active_job.mode} "
                        f"core={seed_run.core} pid={seed_run.active_job.pid} log={seed_run.active_job.log_path}",
                    )
                    continue

            assert seed_run.core is not None
            free_cores.append(seed_run.core)
        active_seed_runs = still_active
        _write_summary(args, jobs, core_pool)

    if stop_launching:
        for seed_run in pending_seed_runs:
            _block_pending_jobs(seed_run, "fail_fast_triggered")
        pending_seed_runs.clear()

    summary_path = _write_summary(args, jobs, core_pool)
    print(f"Summary written to {summary_path}")
    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
