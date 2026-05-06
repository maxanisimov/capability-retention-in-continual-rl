"""Run the full FrozenLake pipeline across seeds with dependency-aware scheduling."""

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

from experiments.pipelines.frozenlake.core.orchestration.run_paths import (
    NOADAPT_POLICY_SUBDIR,
    default_adapt_ewc_settings_file,
    default_adapt_ppo_settings_file,
    default_adapt_rashomon_settings_file,
    default_downstream_envs_file,
    default_outputs_root,
    default_source_envs_file,
    default_train_source_settings_file,
    legacy_outputs_root,
    pipeline_root,
    resolve_default_source_run_dir,
    seed_run_dir,
)


SOURCE_MODE = "source"
DOWNSTREAM_MODES = ("downstream_unconstrained", "downstream_ewc", "downstream_rashomon")
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
        description="Run source plus all FrozenLake downstream methods across multiple seeds.",
    )
    parser.add_argument("--pipeline", "--layout", type=str, dest="layout", default="diagonal_30x30")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--cores", type=int, nargs="+", default=None)
    parser.add_argument("--source-env-file", type=Path, default=default_source_envs_file())
    parser.add_argument("--downstream-env-file", type=Path, default=default_downstream_envs_file())
    parser.add_argument("--source-settings-file", type=Path, default=default_train_source_settings_file())
    parser.add_argument("--adapt-settings-file", type=Path, default=default_adapt_ppo_settings_file())
    parser.add_argument("--ewc-settings-file", type=Path, default=default_adapt_ewc_settings_file())
    parser.add_argument("--rashomon-settings-file", type=Path, default=default_adapt_rashomon_settings_file())
    parser.add_argument("--outputs-root", type=Path, default=default_outputs_root())
    parser.add_argument("--activation", choices=["tanh", "relu"], default="relu")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument(
        "--resume-policy",
        choices=["skip_completed", "rerun_all", "skip_source_only"],
        default="skip_completed",
    )
    parser.add_argument("--failure-policy", choices=["continue", "fail_fast", "stop_seed"], default="continue")
    parser.add_argument("--disable-task-neutralization", action="store_true")
    parser.add_argument("--total-timesteps-override", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-metrics", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _resolve_core_pool(requested_cores: list[int] | None) -> list[int]:
    available_cores = sorted(os.sched_getaffinity(0))
    if requested_cores is None:
        return available_cores
    invalid = sorted(set(requested_cores) - set(available_cores))
    if invalid:
        raise ValueError(f"Requested --cores {invalid} are not available in affinity mask {available_cores}.")
    return _dedupe_preserve_order(requested_cores)


def _worker_script(mode: str) -> Path:
    return pipeline_root() / "cli" / MODE_TO_CLI[mode]


def _job_output_dir(outputs_root: Path, layout: str, seed: int, mode: str) -> Path:
    root = seed_run_dir(outputs_root, layout, seed)
    if mode == SOURCE_MODE:
        return root / NOADAPT_POLICY_SUBDIR
    return root / MODE_TO_DEFAULT_RUN_SUBDIR[mode]


def _job_output_dir_candidates(outputs_root: Path, layout: str, seed: int, mode: str) -> list[Path]:
    canonical = _job_output_dir(outputs_root, layout, seed, mode)
    candidates = [canonical]
    if mode == SOURCE_MODE:
        legacy_source = seed_run_dir(outputs_root, layout, seed) / "source"
        if legacy_source != canonical:
            candidates.append(legacy_source)
    if outputs_root != legacy_outputs_root():
        global_legacy = _job_output_dir(legacy_outputs_root(), layout, seed, mode)
        if global_legacy not in candidates:
            candidates.append(global_legacy)
        if mode == SOURCE_MODE:
            global_legacy_source = seed_run_dir(legacy_outputs_root(), layout, seed) / "source"
            if global_legacy_source not in candidates:
                candidates.append(global_legacy_source)
    return candidates


def _is_job_complete(outputs_root: Path, layout: str, seed: int, mode: str) -> bool:
    required = MODE_TO_REQUIRED_ARTIFACTS[mode]
    return any(all((job_dir / artifact).exists() for artifact in required) for job_dir in _job_output_dir_candidates(outputs_root, layout, seed, mode))


def _job_log_path(log_root: Path, seed: int, mode: str) -> Path:
    mode_dir = NOADAPT_POLICY_SUBDIR if mode == SOURCE_MODE else mode
    return log_root / mode_dir / f"seed_{seed}.log"


def _build_worker_cmd(args: argparse.Namespace, *, seed: int, mode: str) -> list[str]:
    cmd = [
        sys.executable,
        str(_worker_script(mode)),
        "--pipeline",
        str(args.layout),
        "--seed",
        str(seed),
        "--device",
        str(args.device),
        "--activation",
        str(args.activation),
        "--source-env-file",
        str(args.source_env_file),
        "--downstream-env-file",
        str(args.downstream_env_file),
    ]
    if mode == SOURCE_MODE:
        cmd.extend(
            [
                "--settings-file",
                str(args.source_settings_file),
                "--adapt-settings-file",
                str(args.adapt_settings_file),
                "--output-dir",
                str(args.outputs_root),
            ],
        )
        return cmd

    cmd.extend(
        [
            "--source-settings-file",
            str(args.source_settings_file),
            "--adapt-settings-file",
            str(args.adapt_settings_file),
            "--outputs-root",
            str(args.outputs_root),
            "--run-subdir",
            MODE_TO_DEFAULT_RUN_SUBDIR[mode],
            "--source-run-dir",
            str(resolve_default_source_run_dir(args.outputs_root, args.layout, seed)),
        ],
    )
    if mode == "downstream_ewc":
        cmd.extend(["--ewc-settings-file", str(args.ewc_settings_file)])
    if mode == "downstream_rashomon":
        cmd.extend(["--rashomon-settings-file", str(args.rashomon_settings_file)])
    if args.disable_task_neutralization:
        cmd.append("--disable-task-neutralization")
    if args.total_timesteps_override is not None:
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

    return subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=_worker_env(), preexec_fn=_pin_to_core)


def _create_job_graph(seeds: list[int]) -> dict[str, Job]:
    jobs: dict[str, Job] = {}
    for seed in seeds:
        source_key = _job_key(SOURCE_MODE, seed)
        jobs[source_key] = Job(seed=seed, mode=SOURCE_MODE)
        for mode in DOWNSTREAM_MODES:
            jobs[_job_key(mode, seed)] = Job(seed=seed, mode=mode, deps=[source_key])
    return jobs


def _refresh_ready_states(jobs: dict[str, Job]) -> bool:
    changed = False
    for key in sorted(jobs):
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
        if _is_job_complete(args.outputs_root, args.layout, job.seed, job.mode):
            job.state = JOB_SKIPPED


def _mark_blocked(job: Job, *, reason: str) -> bool:
    if job.state in FINAL_STATES or job.state == JOB_RUNNING:
        return False
    job.state = JOB_BLOCKED
    job.blocked_reason = reason
    return True


def _apply_failure_policy(jobs: dict[str, Job], *, failed_job: Job, failure_policy: str) -> bool:
    stop_launching = False
    if failed_job.mode == SOURCE_MODE:
        for mode in DOWNSTREAM_MODES:
            _mark_blocked(jobs[_job_key(mode, failed_job.seed)], reason="source_failed")
    if failure_policy == "stop_seed":
        for job in jobs.values():
            if job.seed == failed_job.seed and job.key != failed_job.key:
                _mark_blocked(job, reason="seed_stopped_after_failure")
    if failure_policy == "fail_fast":
        stop_launching = True
        for job in jobs.values():
            if job.key != failed_job.key:
                _mark_blocked(job, reason="fail_fast_triggered")
    return stop_launching


def _sorted_jobs(jobs: dict[str, Job]) -> list[Job]:
    return sorted(jobs.values(), key=lambda job: (job.seed, MODE_PRIORITY[job.mode]))


def _default_log_root(args: argparse.Namespace) -> Path:
    return args.outputs_root / args.layout / "multi_seed_logs" / "full_pipeline"


def _summarize_to_yaml(*, jobs: dict[str, Job], summary_path: Path, args: argparse.Namespace) -> None:
    payload = {
        "run_settings": {
            "layout": str(args.layout),
            "outputs_root": str(args.outputs_root),
            "seeds": [int(seed) for seed in _dedupe_preserve_order(list(args.seeds))],
            "resume_policy": str(args.resume_policy),
            "failure_policy": str(args.failure_policy),
            "dry_run": bool(args.dry_run),
        },
        "jobs": [
            {
                "seed": int(job.seed),
                "mode": str(job.mode),
                "state": str(job.state),
                "blocked_reason": job.blocked_reason,
                "runtime_seconds": job.runtime_seconds,
                "return_code": job.return_code,
                "log_path": None if job.log_path is None else str(job.log_path),
                "command": list(job.command),
            }
            for job in _sorted_jobs(jobs)
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _run_aggregate_metrics(args: argparse.Namespace) -> bool:
    cmd = [
        sys.executable,
        str(pipeline_root() / "cli" / "aggregate_layout_metrics.py"),
        "--pipeline",
        str(args.layout),
        "--outputs-root",
        str(args.outputs_root),
    ]
    print("\nRunning aggregate metrics export:")
    print("  " + " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode == 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("CPU affinity pinning requires os.sched_setaffinity/os.sched_getaffinity support.")
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

    while any(job.state not in FINAL_STATES for job in jobs.values()):
        _refresh_ready_states(jobs)
        ready = [job for job in _sorted_jobs(jobs) if job.state == JOB_READY]
        while ready and free_cores and not stop_launching:
            job = ready.pop(0)
            core = free_cores.popleft()
            cmd = _build_worker_cmd(args, seed=job.seed, mode=job.mode)
            log_path = _job_log_path(log_root, job.seed, job.mode)
            now = time.time()
            job.command = cmd
            job.log_path = log_path
            job.core = core
            job.start_time = now
            if args.dry_run:
                job.state = JOB_SUCCEEDED
                job.end_time = now
                job.return_code = 0
                free_cores.append(core)
            else:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("w", encoding="utf-8")
                job.log_handle = log_handle
                job.process = _spawn_subprocess(cmd, core=core, log_handle=log_handle)
                job.state = JOB_RUNNING
                active_keys.append(job.key)
                print(f"[start] seed={job.seed} mode={job.mode} core={core} log={log_path}")

        if args.dry_run:
            continue
        if not active_keys:
            break

        time.sleep(max(args.poll_seconds, 0.1))
        still_active: list[str] = []
        for key in active_keys:
            job = jobs[key]
            assert job.process is not None
            rc = job.process.poll()
            if rc is None:
                still_active.append(key)
                continue
            if job.log_handle is not None:
                job.log_handle.close()
                job.log_handle = None
            job.process = None
            job.end_time = time.time()
            job.return_code = int(rc)
            job.state = JOB_SUCCEEDED if rc == 0 else JOB_FAILED
            assert job.core is not None
            free_cores.append(job.core)
            print(f"[{'done' if rc == 0 else 'fail'}] seed={job.seed} mode={job.mode} rc={rc}")
            if rc != 0:
                stop_launching = _apply_failure_policy(jobs, failed_job=job, failure_policy=args.failure_policy)
        active_keys = still_active

    _summarize_to_yaml(jobs=jobs, summary_path=summary_path, args=args)
    failed_or_blocked = [job for job in jobs.values() if job.state in {JOB_FAILED, JOB_BLOCKED}]
    if failed_or_blocked:
        return 1
    if args.aggregate_metrics and not args.dry_run:
        if not _run_aggregate_metrics(args):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
