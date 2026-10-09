#!/usr/bin/env python3
"""Limit startup memory peaks without limiting concurrent FrozenLake training.

Reuse the existing segment sweep and worker unchanged. Admit at most K workers
until each saves config.json, which occurs after model initialization, safety
checks, and shaping attachment. Then release its slot while training continues.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "core")]

from projects.safe_policy_optimisation.scripts import (  # noqa: E402
    run_frozenlake_shaping_sweep as sweep,
)


class InitialisationGate:
    """Hold admission only through initialization, not through model.learn()."""

    def __init__(self, original_run_job, slots: int):
        if slots <= 0:
            raise ValueError("Initialization slots must be positive")
        self.original_run_job = original_run_job
        self.slots = threading.BoundedSemaphore(slots)

    def __call__(self, job: dict, output: Path) -> dict:
        with ThreadPoolExecutor(max_workers=1) as worker:
            with self.slots:
                print(f"Initializing {job['id']}", flush=True)
                result = worker.submit(self.original_run_job, job, output)
                config = Path(job["directory"]) / "config.json"
                while not result.done() and not config.exists():
                    time.sleep(0.5)
                print(f"Releasing initialization slot for {job['id']}", flush=True)
            # Crucially this wait is OUTSIDE the semaphore. Already initialized
            # workers continue training while the next initialization starts.
            return result.result()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervise", type=Path, required=True)
    parser.add_argument("--max-initialisers", type=int, required=True)
    args = parser.parse_args()
    manifest = sweep.read_json(args.supervise)
    if manifest.get("pspo_variant") != "line_segment_verify_first":
        raise ValueError("This launcher requires line-segment verify-first")
    if manifest.get("max_concurrent_initialisers") != args.max_initialisers:
        raise ValueError("Initialization admission must match the saved manifest")
    relative = str(Path(__file__).resolve().relative_to(REPO))
    if (
        manifest["source_sha256"].get(relative)
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("The gated launcher must be included in the source guard")
    original = sweep.run_job
    sweep.run_job = InitialisationGate(original, args.max_initialisers)
    try:
        return sweep.supervise(args.supervise)
    finally:
        sweep.run_job = original


if __name__ == "__main__":
    raise SystemExit(main())
