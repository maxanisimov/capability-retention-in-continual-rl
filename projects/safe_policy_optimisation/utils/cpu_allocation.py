"""CPU worker allocation helpers for experiment pipelines."""

from __future__ import annotations

import os
from collections.abc import Iterable


def cpu_affinity_supported() -> bool:
    return hasattr(os, "sched_setaffinity")


def available_cpu_ids() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(cpu_id) for cpu_id in os.sched_getaffinity(0))
    return list(range(max(1, int(os.cpu_count() or 1))))


def normalise_cpu_ids(cpu_ids: Iterable[int] | None) -> list[int]:
    if cpu_ids is None:
        return available_cpu_ids()
    seen: set[int] = set()
    normalised: list[int] = []
    for value in cpu_ids:
        cpu_id = int(value)
        if cpu_id < 0:
            raise ValueError(f"CPU ids must be non-negative, got {cpu_id}.")
        if cpu_id in seen:
            continue
        seen.add(cpu_id)
        normalised.append(cpu_id)
    if not normalised:
        raise ValueError("At least one CPU id is required.")
    return normalised


def parse_cpu_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = [part.strip() for part in str(value).split(",")]
    if not any(parts):
        raise ValueError("Expected a comma-separated list of CPU ids.")
    return normalise_cpu_ids(int(part) for part in parts if part)


def format_cpu_ids(cpu_ids: Iterable[int] | None) -> str | None:
    if cpu_ids is None:
        return None
    values = [str(int(cpu_id)) for cpu_id in cpu_ids]
    return ",".join(values) if values else None


def apply_cpu_affinity(cpu_ids: Iterable[int] | None) -> list[int] | None:
    if cpu_ids is None:
        return None
    normalised = normalise_cpu_ids(cpu_ids)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(normalised))
    return normalised


def resolve_worker_count(
    requested_jobs: int,
    *,
    method_count: int,
    cpu_ids: Iterable[int],
) -> int:
    available = len(list(cpu_ids))
    if available <= 0:
        raise ValueError("At least one CPU id is required.")
    count = max(0, int(method_count))
    if count <= 0:
        return 1
    upper_bound = max(1, min(count, available))
    if int(requested_jobs) <= 0:
        return upper_bound
    return max(1, min(int(requested_jobs), upper_bound))


def worker_thread_count(jobs: int, explicit: int | None) -> int | None:
    if explicit is not None:
        return max(1, int(explicit))
    if int(jobs) <= 1:
        return None
    return 1
