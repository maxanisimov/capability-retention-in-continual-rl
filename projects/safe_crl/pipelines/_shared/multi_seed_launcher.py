"""Shared CPU-pinned multi-seed subprocess scheduler used by per-environment launch_multi_seed.py scripts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time
from typing import Callable, TextIO


@dataclass
class SeedRun:
    seed: int
    core: int
    process: subprocess.Popen[bytes]
    log_path: Path
    log_handle: TextIO


def dedupe_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def resolve_core_pool(requested_cores: list[int] | None) -> list[int]:
    available_cores = sorted(os.sched_getaffinity(0))
    if requested_cores is None:
        return available_cores
    invalid = sorted(set(requested_cores) - set(available_cores))
    if invalid:
        raise ValueError(f"Requested --cores {invalid} are not available in affinity mask {available_cores}.")
    return dedupe_preserve_order(requested_cores)


def worker_env() -> dict[str, str]:
    """Single-threaded BLAS/OpenMP env so per-core pinning actually prevents CPU contention."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["BLIS_NUM_THREADS"] = "1"
    return env


def start_seed_run(*, seed: int, cmd: list[str], core: int, log_path: Path) -> SeedRun:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")

    def _pin_to_core() -> None:
        os.sched_setaffinity(0, {core})

    process = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=worker_env(),
        preexec_fn=_pin_to_core,
    )
    return SeedRun(seed=seed, core=core, process=process, log_path=log_path, log_handle=log_handle)


def run_seed_pool(
    *,
    seeds: list[int],
    cores: list[int],
    build_cmd: Callable[[int], list[str]],
    log_dir: Path,
    poll_seconds: float = 1.0,
) -> int:
    """Run one subprocess per seed, CPU-pinned, at most one seed per core at a time.

    seeds must already be deduplicated by the caller (callers print the run count
    before invoking this, so deduplication has to happen there, not here).
    """
    if not seeds:
        raise ValueError("No seeds provided.")
    if not cores:
        raise RuntimeError("No CPU cores available for scheduling.")

    pending: deque[int] = deque(seeds)
    free_cores: deque[int] = deque(cores)
    active: list[SeedRun] = []
    failures: list[tuple[int, int, int, Path]] = []

    while pending or active:
        while pending and free_cores:
            seed = pending.popleft()
            core = free_cores.popleft()
            cmd = build_cmd(seed)
            run = start_seed_run(seed=seed, cmd=cmd, core=core, log_path=log_dir / f"seed_{seed}.log")
            active.append(run)
            print(f"[start] seed={seed} core={core} pid={run.process.pid} log={run.log_path}")

        time.sleep(poll_seconds)
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
                print(f"[fail] seed={run.seed} core={run.core} rc={return_code} log={run.log_path}")
                failures.append((run.seed, run.core, int(return_code), run.log_path))
        active = still_active

    if failures:
        print("\nOne or more runs failed:")
        for seed, core, rc, log_path in failures:
            print(f"  seed={seed} core={core} rc={rc} log={log_path}")
        return 1
    print("\nAll runs completed successfully.")
    return 0
