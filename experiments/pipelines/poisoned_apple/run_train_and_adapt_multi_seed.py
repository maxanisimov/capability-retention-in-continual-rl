#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO


@dataclass
class RunningJob:
    seed: int
    core_id: int | None
    process: subprocess.Popen[str]
    log_file: Path
    stream: TextIO


def _parse_seeds(seed_tokens: list[str]) -> list[int]:
    seeds: list[int] = []
    seen: set[int] = set()

    for token in seed_tokens:
        for part in token.split(","):
            text = part.strip()
            if not text:
                continue

            if "-" in text:
                left, right = text.split("-", maxsplit=1)
                start = int(left.strip())
                end = int(right.strip())
                step = 1 if end >= start else -1
                for seed in range(start, end + step, step):
                    if seed not in seen:
                        seen.add(seed)
                        seeds.append(seed)
                continue

            seed = int(text)
            if seed not in seen:
                seen.add(seed)
                seeds.append(seed)

    if not seeds:
        raise ValueError("No seeds parsed. Pass values like '--seeds 0 1 2' or '--seeds 0-9'.")
    return seeds


def _available_cpu_ids() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    cpu_count = os.cpu_count() or 1
    return list(range(cpu_count))


def _child_env() -> dict[str, str]:
    env = dict(os.environ)

    # Keep each run single-threaded at the BLAS/OpenMP layer so we do not
    # oversubscribe cores when running multiple seeds in parallel.
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "TORCH_NUM_THREADS",
    ):
        env[key] = "1"

    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("SDL_AUDIODRIVER", "dummy")
    env.setdefault("ALSA_CONFIG_PATH", "/dev/null")
    return env


def _prepare_parallelism(
    seeds: list[int],
    reserve_cores: int,
    max_parallel: int | None,
) -> tuple[list[int], int]:
    cpu_ids = _available_cpu_ids()
    if reserve_cores < 0:
        raise ValueError(f"--reserve-cores must be >= 0, got {reserve_cores}.")

    usable_count = max(1, len(cpu_ids) - reserve_cores)
    usable_cores = cpu_ids[:usable_count]

    desired = len(seeds) if max_parallel is None else max_parallel
    if desired <= 0:
        raise ValueError(f"--max-parallel must be >= 1, got {desired}.")

    worker_count = min(desired, len(usable_cores), len(seeds))
    if worker_count < 1:
        worker_count = 1

    return usable_cores, worker_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run poisoned_apple/run_train_and_adapt.py for multiple seeds in parallel "
            "without CPU oversubscription (single-threaded workers + optional core pinning)."
        ),
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Config key passed to run_train_and_adapt.py",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["0-9"],
        metavar="SEED_OR_RANGE",
        help="Seed list/ranges, e.g. --seeds 0 1 2 or --seeds 0-9 or --seeds 0,1,2",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Max concurrent runs (default: min(#usable_cores, #seeds)).",
    )
    parser.add_argument(
        "--reserve-cores",
        type=int,
        default=1,
        help="Keep this many cores free for system/interactive use (default: 1).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between child-process status checks.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used for child runs.",
    )
    parser.add_argument(
        "--run-script",
        type=str,
        default=None,
        help="Path to run_train_and_adapt.py (default: sibling script).",
    )
    parser.add_argument(
        "--logs-root",
        type=str,
        default=None,
        help="Directory for per-seed logs (default: <this_folder>/logs/multi_seed).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved child commands and exit without launching processes.",
    )
    parser.add_argument(
        "--no-cpu-affinity",
        action="store_true",
        help="Disable taskset-based CPU pinning (single-thread limits are still enforced).",
    )
    parser.add_argument(
        "child_args",
        nargs=argparse.REMAINDER,
        help=(
            "Extra args forwarded to run_train_and_adapt.py. "
            "Use '--' before forwarded args."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    run_script = Path(args.run_script) if args.run_script else script_dir / "run_train_and_adapt.py"
    logs_root = Path(args.logs_root) if args.logs_root else script_dir / "logs" / "multi_seed"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = logs_root / args.cfg / run_stamp
    log_dir.mkdir(parents=True, exist_ok=True)

    if not run_script.exists():
        raise FileNotFoundError(f"Could not find run script: {run_script}")

    seeds = _parse_seeds(args.seeds)

    forwarded = list(args.child_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    if args.dry_run:
        print("=" * 90)
        print("PoisonedApple multi-seed runner (dry run)")
        print(f"Config: {args.cfg}")
        print(f"Seeds: {seeds}")
        print(f"Run script: {run_script}")
        print("Commands:")
        for seed in seeds:
            cmd = [
                args.python_bin,
                str(run_script),
                "--cfg",
                args.cfg,
                "--seed",
                str(seed),
                *forwarded,
            ]
            print("  " + " ".join(cmd))
        print("=" * 90)
        return 0

    usable_cores, worker_count = _prepare_parallelism(
        seeds=seeds,
        reserve_cores=args.reserve_cores,
        max_parallel=args.max_parallel,
    )

    has_taskset = shutil.which("taskset") is not None
    use_affinity = has_taskset and not args.no_cpu_affinity
    core_pool = deque(usable_cores[:worker_count]) if use_affinity else deque([None] * worker_count)

    env = _child_env()
    pending = deque(seeds)
    running: list[RunningJob] = []
    results: list[tuple[int, int, Path]] = []

    print("=" * 90)
    print("PoisonedApple multi-seed runner")
    print(f"Config: {args.cfg}")
    print(f"Seeds: {seeds}")
    print(f"Run script: {run_script}")
    print(f"Logs: {log_dir}")
    print(f"Usable cores: {usable_cores}")
    print(f"Concurrent workers: {worker_count}")
    print(f"CPU affinity: {'enabled' if use_affinity else 'disabled'}")
    if not has_taskset and not args.no_cpu_affinity:
        print("Note: taskset was not found; running without explicit core pinning.")
    print("=" * 90)

    try:
        while pending or running:
            while pending and len(running) < worker_count:
                seed = pending.popleft()
                core_id = core_pool.popleft()
                log_path = log_dir / f"seed_{seed}.log"
                stream = log_path.open("w", encoding="utf-8")

                cmd = [
                    args.python_bin,
                    str(run_script),
                    "--cfg",
                    args.cfg,
                    "--seed",
                    str(seed),
                ]
                cmd.extend(forwarded)

                if core_id is not None:
                    launch_cmd = ["taskset", "-c", str(core_id)] + cmd
                else:
                    launch_cmd = cmd

                print(
                    f"[start] seed={seed} "
                    + (f"core={core_id} " if core_id is not None else "")
                    + f"log={log_path}",
                )
                process = subprocess.Popen(
                    launch_cmd,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
                running.append(
                    RunningJob(
                        seed=seed,
                        core_id=core_id,
                        process=process,
                        log_file=log_path,
                        stream=stream,
                    ),
                )

            time.sleep(max(0.1, args.poll_interval))

            still_running: list[RunningJob] = []
            for job in running:
                return_code = job.process.poll()
                if return_code is None:
                    still_running.append(job)
                    continue

                job.stream.close()
                if job.core_id is not None:
                    core_pool.append(job.core_id)
                results.append((job.seed, return_code, job.log_file))
                status = "ok" if return_code == 0 else "fail"
                print(f"[{status}] seed={job.seed} rc={return_code} log={job.log_file}")

            running = still_running

    except KeyboardInterrupt:
        print("\nInterrupted. Terminating active runs...")
        for job in running:
            job.process.terminate()
        for job in running:
            try:
                job.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                job.process.kill()
            job.stream.close()
        return 130

    failed = [(seed, rc, log) for seed, rc, log in results if rc != 0]
    succeeded = [seed for seed, rc, _ in results if rc == 0]

    print("=" * 90)
    print("PoisonedApple multi-seed summary")
    print(f"Config: {args.cfg}")
    print(f"Seeds requested: {len(seeds)}")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")
    if succeeded:
        print(f"Succeeded seeds: {sorted(succeeded)}")
    if failed:
        print("Failed seeds:")
        for seed, rc, log_path in sorted(failed, key=lambda x: x[0]):
            print(f"  seed={seed} rc={rc} log={log_path}")
    print(f"Logs directory: {log_dir}")
    print("=" * 90)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
