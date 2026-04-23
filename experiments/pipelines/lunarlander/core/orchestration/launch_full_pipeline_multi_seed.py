"""Run full LunarLander pipeline across seeds with dependency-aware scheduling.

Pipeline per seed:
  source -> {downstream_unconstrained, downstream_ewc, downstream_rashomon}

This launcher uses one global CPU-core pool and can overlap downstream jobs for
completed seeds while other seeds are still training source policies.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import TextIO

import yaml

from experiments.pipelines.lunarlander.core.orchestration.run_paths import (
    default_adapt_ewc_settings_file,
    default_adapt_ppo_settings_file,
    default_outputs_root,
    default_task_settings_file,
    pipeline_root,
    seed_run_dir,
)


SOURCE_MODE = "source"
DOWNSTREAM_MODES = (
    "downstream_unconstrained",
    "downstream_ewc",
    "downstream_rashomon",
)
ALL_MODES = (SOURCE_MODE, *DOWNSTREAM_MODES)

MODE_TO_CLI = {
    SOURCE_MODE: "train_source.py",
    "downstream_unconstrained": "adapt_unconstrained.py",
    "downstream_ewc": "adapt_ewc.py",
    "downstream_rashomon": "adapt_rashomon.py",
}

MODE_TO_DEFAULT_RUN_SUBDIR = {
    "downstream_unconstrained": "downstream_unconstrained",
    "downstream_ewc": "downstream_ewc",
    "downstream_rashomon": "downstream_rashomon",
}

MODE_TO_REQUIRED_ARTIFACTS = {
    SOURCE_MODE: ("actor.pt", "critic.pt", "training_data.pt", "run_summary.yaml"),
    "downstream_unconstrained": ("actor.pt", "critic.pt", "training_data.pt", "run_summary.yaml"),
    "downstream_ewc": ("actor.pt", "critic.pt", "training_data.pt", "run_summary.yaml", "ewc_state.pt"),
    "downstream_rashomon": (
        "actor.pt",
        "critic.pt",
        "training_data.pt",
        "run_summary.yaml",
        "rashomon_dataset.pt",
        "rashomon_bounded_model.pt",
        "rashomon_param_bounds.pt",
        "rashomon_rollout_stats.yaml",
    ),
}

MODE_PRIORITY = {mode: idx for idx, mode in enumerate(ALL_MODES)}

JOB_PENDING = "PENDING"
JOB_READY = "READY"
JOB_RUNNING = "RUNNING"
JOB_SUCCEEDED = "SUCCEEDED"
JOB_FAILED = "FAILED"
JOB_SKIPPED = "SKIPPED"
JOB_BLOCKED = "BLOCKED"

FINAL_STATES = {JOB_SUCCEEDED, JOB_FAILED, JOB_SKIPPED, JOB_BLOCKED}

RESUME_POLICIES = ("skip_completed", "rerun_all", "skip_source_only")
FAILURE_POLICIES = ("continue", "fail_fast", "stop_seed")
DISPATCH_POLICIES = ("balanced", "source_priority", "downstream_priority")


@dataclass
class Job:
    seed: int
    mode: str
    deps: list[str] = field(default_factory=list)
    state: str = JOB_PENDING
    blocked_reason: str | None = None
    command: list[str] = field(default_factory=list)
    log_path: Path | None = None
    core: int | None = None
    process: subprocess.Popen[bytes] | None = None
    log_handle: TextIO | None = None
    start_time: float | None = None
    end_time: float | None = None
    return_code: int | None = None

    @property
    def key(self) -> str:
        return _job_key(self.mode, self.seed)

    @property
    def runtime_seconds(self) -> float | None:
        if self.start_time is None or self.end_time is None:
            return None
        return float(self.end_time - self.start_time)


def _job_key(mode: str, seed: int) -> str:
    return f"{mode}:{seed}"


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
        description=(
            "Run full LunarLander pipeline (source + all downstream methods) across "
            "multiple seeds with dependency-aware scheduling and CPU pinning."
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
        help="Output root for all pipeline modes.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device passed to worker runs.")
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="Polling interval for child process completion checks.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=(
            "Base log directory for this launcher. Defaults to "
            "<outputs-root>/<task-setting>/multi_seed_logs/full_pipeline."
        ),
    )
    parser.add_argument(
        "--resume-policy",
        type=str,
        choices=RESUME_POLICIES,
        default="skip_completed",
        help="Resume behavior for already completed runs.",
    )
    parser.add_argument(
        "--failure-policy",
        type=str,
        choices=FAILURE_POLICIES,
        default="continue",
        help="Failure behavior when one job fails.",
    )
    parser.add_argument(
        "--dispatch-policy",
        type=str,
        choices=DISPATCH_POLICIES,
        default="balanced",
        help="How to prioritize source vs downstream jobs when both are ready.",
    )
    parser.add_argument(
        "--disable-task-neutralization",
        action="store_true",
        help="Forward --disable-task-neutralization to downstream modes.",
    )
    parser.add_argument(
        "--total-timesteps-override",
        type=int,
        default=None,
        help="Forward --total-timesteps-override to downstream_unconstrained/downstream_ewc.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve, schedule, and print commands without launching worker processes.",
    )
    parser.add_argument(
        "--aggregate-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run aggregate_layout_metrics.py after successful completion. "
            "Enabled by default; use --no-aggregate-metrics to disable."
        ),
    )
    parser.add_argument(
        "--aggregate-task-setting",
        type=str,
        default=None,
        help=(
            "Task setting passed to aggregate_layout_metrics.py. "
            "Defaults to --task-setting when omitted."
        ),
    )
    return parser.parse_args(argv)


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


def _job_output_dir(outputs_root: Path, task_setting: str, seed: int, mode: str) -> Path:
    root = seed_run_dir(outputs_root, task_setting, seed)
    if mode == SOURCE_MODE:
        return root / "source"
    return root / MODE_TO_DEFAULT_RUN_SUBDIR[mode]


def _job_log_path(log_root: Path, seed: int, mode: str) -> Path:
    return log_root / mode / f"seed_{seed}.log"


def _is_job_complete(outputs_root: Path, task_setting: str, seed: int, mode: str) -> bool:
    job_dir = _job_output_dir(outputs_root, task_setting, seed, mode)
    required = MODE_TO_REQUIRED_ARTIFACTS[mode]
    return all((job_dir / artifact).exists() for artifact in required)


def _build_worker_cmd(args: argparse.Namespace, *, seed: int, mode: str) -> list[str]:
    cmd: list[str] = [sys.executable, str(_worker_script(mode))]

    if mode == SOURCE_MODE:
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
                str(args.outputs_root),
                "--device",
                str(args.device),
            ],
        )
        return cmd

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
            "--source-run-dir",
            str(_job_output_dir(args.outputs_root, args.task_setting, seed, SOURCE_MODE)),
        ],
    )
    if mode in {"downstream_unconstrained", "downstream_ewc"}:
        cmd.extend(["--adapt-settings-file", str(args.adapt_settings_file)])
    if mode == "downstream_ewc":
        cmd.extend(["--ewc-settings-file", str(args.ewc_settings_file)])
    if args.disable_task_neutralization:
        cmd.append("--disable-task-neutralization")
    if args.total_timesteps_override is not None and mode in {"downstream_unconstrained", "downstream_ewc"}:
        cmd.extend(["--total-timesteps-override", str(args.total_timesteps_override)])
    return cmd


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["BLIS_NUM_THREADS"] = "1"
    return env


def _spawn_subprocess(cmd: list[str], *, core: int, log_handle: TextIO) -> subprocess.Popen[bytes]:
    def _pin_to_core() -> None:
        os.sched_setaffinity(0, {core})

    return subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=_worker_env(),
        preexec_fn=_pin_to_core,
    )


def _create_job_graph(seeds: list[int]) -> dict[str, Job]:
    jobs: dict[str, Job] = {}
    for seed in seeds:
        source_key = _job_key(SOURCE_MODE, seed)
        jobs[source_key] = Job(seed=seed, mode=SOURCE_MODE, deps=[])
        for mode in DOWNSTREAM_MODES:
            key = _job_key(mode, seed)
            jobs[key] = Job(seed=seed, mode=mode, deps=[source_key])
    return jobs


def _mark_blocked(job: Job, *, reason: str) -> bool:
    if job.state in FINAL_STATES or job.state == JOB_RUNNING:
        return False
    job.state = JOB_BLOCKED
    job.blocked_reason = reason
    return True


def _refresh_ready_states(jobs: dict[str, Job]) -> bool:
    changed = False
    for key in sorted(jobs.keys()):
        job = jobs[key]
        if job.state != JOB_PENDING:
            continue
        if not job.deps:
            job.state = JOB_READY
            changed = True
            continue

        dep_states = [jobs[dep].state for dep in job.deps]
        if any(state in {JOB_FAILED, JOB_BLOCKED} for state in dep_states):
            job.state = JOB_BLOCKED
            job.blocked_reason = "dependency_failed_or_blocked"
            changed = True
        elif all(state in {JOB_SUCCEEDED, JOB_SKIPPED} for state in dep_states):
            job.state = JOB_READY
            changed = True
    return changed


def _apply_resume_policy(jobs: dict[str, Job], args: argparse.Namespace) -> None:
    if args.resume_policy == "rerun_all":
        return

    for job in jobs.values():
        if args.resume_policy == "skip_source_only" and job.mode != SOURCE_MODE:
            continue
        if _is_job_complete(args.outputs_root, args.task_setting, job.seed, job.mode):
            job.state = JOB_SKIPPED


def _sorted_jobs_by_seed_mode(jobs: dict[str, Job]) -> list[Job]:
    return sorted(
        jobs.values(),
        key=lambda j: (j.seed, MODE_PRIORITY.get(j.mode, 999)),
    )


def _jobs_by_state(jobs: dict[str, Job], state: str, *, mode: str | None = None) -> list[Job]:
    out = [job for job in jobs.values() if job.state == state and (mode is None or job.mode == mode)]
    out.sort(key=lambda j: (j.seed, MODE_PRIORITY.get(j.mode, 999)))
    return out


def _source_jobs_remaining(jobs: dict[str, Job]) -> bool:
    return any(job.mode == SOURCE_MODE and job.state not in FINAL_STATES for job in jobs.values())


def _select_jobs_to_launch(
    *,
    ready_sources: list[Job],
    ready_downstream: list[Job],
    free_slots: int,
    dispatch_policy: str,
    total_cores: int,
    active_source_count: int,
    sources_remaining: bool,
) -> list[Job]:
    if free_slots <= 0:
        return []
    if not ready_sources and not ready_downstream:
        return []

    if dispatch_policy == "source_priority":
        selected = ready_sources[:free_slots]
        remaining = free_slots - len(selected)
        if remaining > 0:
            selected.extend(ready_downstream[:remaining])
        return selected

    if dispatch_policy == "downstream_priority":
        selected = ready_downstream[:free_slots]
        remaining = free_slots - len(selected)
        if remaining > 0:
            selected.extend(ready_sources[:remaining])
        return selected

    # dispatch_policy == "balanced"
    if not ready_sources:
        return ready_downstream[:free_slots]
    if not ready_downstream:
        return ready_sources[:free_slots]
    if not sources_remaining:
        return ready_downstream[:free_slots]

    reserve = max(1, total_cores // 2)
    source_slots = max(0, reserve - active_source_count)
    selected_sources = ready_sources[: min(free_slots, source_slots)]
    selected: list[Job] = list(selected_sources)

    remaining_slots = free_slots - len(selected)
    selected_downstream = ready_downstream[:remaining_slots]
    selected.extend(selected_downstream)
    remaining_slots = free_slots - len(selected)

    if remaining_slots > 0:
        leftover_sources = ready_sources[len(selected_sources) :]
        take = min(remaining_slots, len(leftover_sources))
        selected.extend(leftover_sources[:take])
        remaining_slots = free_slots - len(selected)

    if remaining_slots > 0:
        leftover_downstream = ready_downstream[len(selected_downstream) :]
        take = min(remaining_slots, len(leftover_downstream))
        selected.extend(leftover_downstream[:take])

    return selected


def _apply_failure_policy(
    jobs: dict[str, Job],
    *,
    failed_job: Job,
    failure_policy: str,
) -> bool:
    stop_launching = False

    # Source failure always blocks that seed's downstream jobs.
    if failed_job.mode == SOURCE_MODE:
        for mode in DOWNSTREAM_MODES:
            dep_job = jobs[_job_key(mode, failed_job.seed)]
            _mark_blocked(dep_job, reason="source_failed")

    if failure_policy == "stop_seed":
        for job in jobs.values():
            if job.seed != failed_job.seed:
                continue
            if job.key == failed_job.key:
                continue
            _mark_blocked(job, reason="seed_stopped_after_failure")

    if failure_policy == "fail_fast":
        stop_launching = True
        for job in jobs.values():
            if job.key == failed_job.key:
                continue
            _mark_blocked(job, reason="fail_fast_triggered")

    return stop_launching


def _default_log_root(args: argparse.Namespace) -> Path:
    return args.outputs_root / args.task_setting / "multi_seed_logs" / "full_pipeline"


def _run_aggregate_metrics(args: argparse.Namespace) -> bool:
    task_setting = args.aggregate_task_setting or args.task_setting
    cmd = [
        sys.executable,
        str(pipeline_root() / "cli" / "aggregate_layout_metrics.py"),
        "--task-setting",
        str(task_setting),
        "--outputs-root",
        str(args.outputs_root),
    ]
    print("\nRunning aggregate metrics export:")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if int(result.returncode) != 0:
        print(
            f"Aggregate metrics export failed for task_setting={task_setting} "
            f"(rc={int(result.returncode)}).",
        )
        return False
    return True


def _start_job(
    *,
    job: Job,
    cmd: list[str],
    core: int,
    log_path: Path,
    dry_run: bool,
) -> None:
    now = time.time()
    job.command = list(cmd)
    job.log_path = log_path
    job.core = core
    job.start_time = now

    if dry_run:
        job.state = JOB_SUCCEEDED
        job.end_time = now
        job.return_code = 0
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    process = _spawn_subprocess(cmd, core=core, log_handle=log_handle)
    job.log_handle = log_handle
    job.process = process
    job.state = JOB_RUNNING


def _finalize_running_job(job: Job, *, return_code: int) -> None:
    job.return_code = int(return_code)
    job.end_time = time.time()
    job.state = JOB_SUCCEEDED if return_code == 0 else JOB_FAILED
    if job.log_handle is not None:
        job.log_handle.close()
        job.log_handle = None
    job.process = None


def _summarize_to_yaml(*, jobs: dict[str, Job], summary_path: Path, args: argparse.Namespace) -> None:
    payload = {
        "run_settings": {
            "task_setting": str(args.task_setting),
            "task_settings_file": str(args.task_settings_file),
            "adapt_settings_file": str(args.adapt_settings_file),
            "ewc_settings_file": str(args.ewc_settings_file),
            "outputs_root": str(args.outputs_root),
            "device": str(args.device),
            "seeds": [int(s) for s in _dedupe_preserve_order(list(args.seeds))],
            "cores": (None if args.cores is None else [int(c) for c in args.cores]),
            "resume_policy": str(args.resume_policy),
            "failure_policy": str(args.failure_policy),
            "dispatch_policy": str(args.dispatch_policy),
            "disable_task_neutralization": bool(args.disable_task_neutralization),
            "total_timesteps_override": (
                int(args.total_timesteps_override)
                if args.total_timesteps_override is not None
                else None
            ),
            "aggregate_metrics": bool(args.aggregate_metrics),
            "aggregate_task_setting": (
                str(args.aggregate_task_setting)
                if args.aggregate_task_setting is not None
                else None
            ),
            "dry_run": bool(args.dry_run),
        },
        "jobs": [
            {
                "seed": int(job.seed),
                "mode": str(job.mode),
                "state": str(job.state),
                "blocked_reason": job.blocked_reason,
                "runtime_seconds": (None if job.runtime_seconds is None else float(job.runtime_seconds)),
                "return_code": (None if job.return_code is None else int(job.return_code)),
                "log_path": (None if job.log_path is None else str(job.log_path)),
                "command": list(job.command),
            }
            for job in _sorted_jobs_by_seed_mode(jobs)
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _print_summary_table(jobs: dict[str, Job]) -> None:
    print("\nJob Summary:")
    print(f"{'seed':>4}  {'mode':<24} {'state':<10} {'runtime_s':>10} {'rc':>4}  log")
    for job in _sorted_jobs_by_seed_mode(jobs):
        runtime = "-" if job.runtime_seconds is None else f"{job.runtime_seconds:.2f}"
        rc = "-" if job.return_code is None else str(job.return_code)
        log_path = "-" if job.log_path is None else str(job.log_path)
        print(
            f"{job.seed:>4}  {job.mode:<24} {job.state:<10} {runtime:>10} {rc:>4}  {log_path}",
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError(
            "CPU affinity pinning requires os.sched_setaffinity/os.sched_getaffinity support.",
        )

    seeds = _dedupe_preserve_order(list(args.seeds))
    if not seeds:
        raise ValueError("No seeds provided. Pass at least one seed via --seeds.")

    core_pool = _resolve_core_pool(args.cores)
    if not core_pool:
        raise RuntimeError("No CPU cores available for scheduling.")

    log_root = args.log_dir or _default_log_root(args)
    log_root.mkdir(parents=True, exist_ok=True)
    summary_path = log_root / "summary.yaml"

    jobs = _create_job_graph(seeds)
    _apply_resume_policy(jobs, args)
    _refresh_ready_states(jobs)

    free_cores: deque[int] = deque(core_pool)
    active_keys: list[str] = []
    stop_launching = False
    first_failure_seen = False

    print(
        f"Launching full pipeline for {len(seeds)} seed(s) with {len(core_pool)} core(s): {core_pool}",
    )
    print(
        f"Policies: resume={args.resume_policy} | failure={args.failure_policy} | "
        f"dispatch={args.dispatch_policy}",
    )
    if args.dry_run:
        print("Dry-run mode enabled; commands will be resolved and scheduled but not executed.")

    while True:
        _refresh_ready_states(jobs)
        if all(job.state in FINAL_STATES for job in jobs.values()):
            break

        if not stop_launching:
            while free_cores:
                ready_sources = _jobs_by_state(jobs, JOB_READY, mode=SOURCE_MODE)
                ready_downstream = [
                    job
                    for job in _jobs_by_state(jobs, JOB_READY)
                    if job.mode in DOWNSTREAM_MODES
                ]
                if not ready_sources and not ready_downstream:
                    break

                active_source_count = sum(
                    1 for job in jobs.values() if job.mode == SOURCE_MODE and job.state == JOB_RUNNING
                )
                selected = _select_jobs_to_launch(
                    ready_sources=ready_sources,
                    ready_downstream=ready_downstream,
                    free_slots=len(free_cores),
                    dispatch_policy=args.dispatch_policy,
                    total_cores=len(core_pool),
                    active_source_count=active_source_count,
                    sources_remaining=_source_jobs_remaining(jobs),
                )
                if not selected:
                    break

                for job in selected:
                    if not free_cores:
                        break
                    core = free_cores.popleft()
                    cmd = _build_worker_cmd(args, seed=job.seed, mode=job.mode)
                    log_path = _job_log_path(log_root, seed=job.seed, mode=job.mode)
                    _start_job(job=job, cmd=cmd, core=core, log_path=log_path, dry_run=args.dry_run)

                    joined_cmd = " ".join(cmd)
                    if args.dry_run:
                        print(
                            f"[dry-run] seed={job.seed} mode={job.mode} core={core} "
                            f"log={log_path}\n          cmd: {joined_cmd}",
                        )
                        free_cores.append(core)
                    else:
                        assert job.process is not None
                        active_keys.append(job.key)
                        print(
                            f"[start] seed={job.seed} mode={job.mode} core={core} "
                            f"pid={job.process.pid} log={log_path}",
                        )

            if args.dry_run:
                # Dry-run jobs transition immediately to final states; loop back to unlock deps.
                continue

        if not active_keys:
            # No running jobs and no new launches. Prevent deadlock.
            unresolved = [job for job in jobs.values() if job.state in {JOB_PENDING, JOB_READY}]
            for job in unresolved:
                _mark_blocked(job, reason="unschedulable_or_stopped")
            break

        time.sleep(max(0.0, float(args.poll_seconds)))
        still_active: list[str] = []
        for job_key in active_keys:
            job = jobs[job_key]
            if job.process is None:
                continue
            rc = job.process.poll()
            if rc is None:
                still_active.append(job_key)
                continue

            _finalize_running_job(job, return_code=int(rc))
            if job.core is not None:
                free_cores.append(job.core)
            if job.state == JOB_SUCCEEDED:
                print(f"[done] seed={job.seed} mode={job.mode} core={job.core} rc=0")
                continue

            print(
                f"[fail] seed={job.seed} mode={job.mode} core={job.core} rc={job.return_code} "
                f"log={job.log_path}",
            )
            if not first_failure_seen:
                first_failure_seen = True
            stop_launching = _apply_failure_policy(
                jobs,
                failed_job=job,
                failure_policy=args.failure_policy,
            ) or stop_launching
            if args.failure_policy == "fail_fast":
                stop_launching = True
        active_keys = still_active

    _refresh_ready_states(jobs)
    _print_summary_table(jobs)
    _summarize_to_yaml(jobs=jobs, summary_path=summary_path, args=args)
    print(f"\nSaved summary: {summary_path}")

    has_fail_or_block = any(job.state in {JOB_FAILED, JOB_BLOCKED} for job in jobs.values())
    aggregate_ok = True
    if bool(args.aggregate_metrics) and (not bool(args.dry_run)):
        if has_fail_or_block:
            print(
                "Skipping aggregate metrics export because one or more jobs failed or were blocked.",
            )
        else:
            aggregate_ok = _run_aggregate_metrics(args)
    if has_fail_or_block:
        print("One or more jobs failed or were blocked.")
        return 1
    if not aggregate_ok:
        return 1

    print("All jobs completed successfully (or were skipped).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
