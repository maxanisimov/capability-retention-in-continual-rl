"""CPU-core allocation and a bounded core-slot job scheduler.

Shared by the parallel launchers (``scripts/run_parallel_pipelines.py`` and
``scripts/run_seed_sweep.py``) so concurrent subprocesses are confined to
disjoint sets of CPU cores and never contend for the same cores.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def available_cores() -> list[int]:
    """Cores this process may use (respects an outer taskset / cgroup)."""

    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux fallback
        return list(range(os.cpu_count() or 1))


def partition_cores(
    cores: list[int], n_groups: int, *, cores_per_group: int | None
) -> list[list[int]]:
    """Split ``cores`` into ``n_groups`` disjoint, contiguous slices."""

    if n_groups <= 0:
        raise ValueError("Need at least one group.")
    if cores_per_group is not None:
        if cores_per_group <= 0:
            raise ValueError("cores_per_group must be positive.")
        needed = cores_per_group * n_groups
        if needed > len(cores):
            raise SystemExit(
                f"Need {needed} cores ({cores_per_group} x {n_groups}) but only "
                f"{len(cores)} are available: {cores}."
            )
        return [
            cores[i * cores_per_group : (i + 1) * cores_per_group]
            for i in range(n_groups)
        ]
    if n_groups > len(cores):
        raise SystemExit(
            f"Cannot give {n_groups} groups a disjoint core each: only "
            f"{len(cores)} cores available ({cores})."
        )
    base, extra = divmod(len(cores), n_groups)
    groups: list[list[int]] = []
    start = 0
    for i in range(n_groups):
        size = base + (1 if i < extra else 0)
        groups.append(cores[start : start + size])
        start += size
    return groups


@dataclass
class Job:
    """One subprocess to schedule on a core slot.

    ``make_command`` receives the assigned core ids and returns the process argv
    (without any ``taskset`` prefix - the scheduler adds that). ``meta`` carries
    arbitrary caller data (e.g. pipeline/seed) used by ``on_complete``.
    """

    name: str
    make_command: Callable[[list[int]], list[str]]
    log_path: Path
    meta: dict[str, Any] = field(default_factory=dict)


def run_core_slot_jobs(
    initial_jobs: list[Job],
    *,
    slots: list[list[int]],
    env: dict[str, str] | None = None,
    use_taskset: bool = True,
    on_complete: Callable[[Job, int], list[Job] | None] | None = None,
    on_poll: Callable[[], list[Job] | None] | None = None,
    poll_seconds: float = 0.5,
) -> dict[str, int]:
    """Run jobs across a fixed pool of disjoint core slots.

    At most ``len(slots)`` jobs run at once; each is pinned to one slot's cores.
    When a job exits, its slot is recycled and ``on_complete(job, returncode)``
    may return follow-up jobs to enqueue (used for warmup -> seed dependencies).

    ``on_poll``, if given, is called once per scheduling tick (every
    ``poll_seconds``) regardless of whether any job just finished. Use it to
    enqueue jobs gated on an external condition - e.g. a shared artifact a
    still-running warmup job has already written to disk - instead of waiting
    for that job's exit, which conflates "shared setup is done" with "this
    job's own, unrelated work is also done".

    Returns ``{job.name: returncode}``.
    """

    if not slots:
        raise ValueError("Need at least one core slot.")
    queue: deque[Job] = deque(initial_jobs)
    free_slots = list(range(len(slots)))
    running: dict[int, tuple[Job, int, Any]] = {}  # pid -> (job, slot_idx, handle)
    procs: dict[int, subprocess.Popen] = {}
    results: dict[str, int] = {}

    def _terminate_all(*_: object) -> None:
        for proc in procs.values():
            if proc.poll() is None:
                proc.terminate()

    previous = {
        signal.SIGINT: signal.signal(signal.SIGINT, _terminate_all),
        signal.SIGTERM: signal.signal(signal.SIGTERM, _terminate_all),
    }
    try:
        while queue or running:
            while free_slots and queue:
                slot_idx = free_slots.pop(0)
                job = queue.popleft()
                core_ids = slots[slot_idx]
                cmd = list(job.make_command(core_ids))
                if use_taskset:
                    cmd = ["taskset", "-c", ",".join(str(c) for c in core_ids)] + cmd
                job.log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = job.log_path.open("w", encoding="utf-8")
                proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
                running[proc.pid] = (job, slot_idx, handle)
                procs[proc.pid] = proc
                print(
                    f"[{time.strftime('%H:%M:%S')}] start {job.name} "
                    f"cores={core_ids} pid={proc.pid} -> {job.log_path}"
                )

            if not running:
                continue
            time.sleep(poll_seconds)
            if on_poll is not None:
                for job in on_poll() or []:
                    queue.append(job)
            for pid, proc in list(procs.items()):
                ret = proc.poll()
                if ret is None:
                    continue
                job, slot_idx, handle = running.pop(pid)
                procs.pop(pid)
                handle.close()
                results[job.name] = ret
                free_slots.append(slot_idx)
                status = "OK" if ret == 0 else f"FAILED (exit {ret})"
                print(f"[{time.strftime('%H:%M:%S')}] done  {job.name}: {status}")
                if on_complete is not None:
                    for follow_up in on_complete(job, ret) or []:
                        queue.append(follow_up)
    finally:
        signal.signal(signal.SIGINT, previous[signal.SIGINT])
        signal.signal(signal.SIGTERM, previous[signal.SIGTERM])
    return results
