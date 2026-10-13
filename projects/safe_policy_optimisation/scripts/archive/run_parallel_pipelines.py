#!/usr/bin/env python
"""Run several safe-policy-optimisation pipelines at once on disjoint CPU cores.

Each pipeline is launched as a separate ``run_experiment.py`` process that is
confined to its own slice of CPU cores, so concurrently running pipelines never
contend for the same cores. Isolation is enforced two ways:

1. ``taskset -c <ids>`` (when available) pins the whole process tree - including
   shield synthesis and any BLAS threads - to that pipeline's cores.
2. ``--cpu-ids <ids>`` tells the pipeline which cores to spread its internal
   policy-optimisation workers across, and ``--jobs``/``--torch-num-threads`` are
   sized to that slice.

Available cores are taken from this process's CPU affinity and split into
contiguous, non-overlapping groups - one per pipeline.

Examples
--------
Split all available cores evenly across two pipelines::

    python projects/safe_policy_optimisation/scripts/run_parallel_pipelines.py \
        --pipeline deterministic_minipacman \
        --pipeline deterministic_colour_bomb

Give each pipeline a fixed number of cores and forward extra launcher args::

    python .../run_parallel_pipelines.py \
        --pipeline deterministic_minipacman \
        --pipeline deterministic_bridge_crossing \
        --cores-per-pipeline 8 \
        -- --total-timesteps 2000

Preview the core assignment without launching anything::

    python .../run_parallel_pipelines.py --pipeline a --pipeline b --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from projects.safe_policy_optimisation.utils.parallel import (  # noqa: E402
    available_cores,
    partition_cores,
)

RUN_EXPERIMENT = Path(__file__).resolve().parent.parent / "run_experiment.py"


def build_command(
    pipeline: str,
    group: list[int],
    *,
    jobs: int | None,
    torch_threads: int,
    use_taskset: bool,
    extra_args: list[str],
) -> list[str]:
    ids_csv = ",".join(str(c) for c in group)
    n_jobs = jobs if jobs is not None else len(group)
    cmd: list[str] = []
    if use_taskset:
        cmd += ["taskset", "-c", ids_csv]
    cmd += [
        sys.executable,
        str(RUN_EXPERIMENT),
        "--pipeline",
        pipeline,
        "--cpu-ids",
        ids_csv,
        "--jobs",
        str(max(1, n_jobs)),
        "--torch-num-threads",
        str(torch_threads),
    ]
    cmd += extra_args
    return cmd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch multiple run_experiment.py pipelines on disjoint CPU cores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipeline",
        dest="pipelines",
        action="append",
        required=True,
        metavar="NAME",
        help="Pipeline name to run (repeat for each pipeline).",
    )
    parser.add_argument(
        "--cores-per-pipeline",
        type=int,
        default=None,
        help="Cores to give each pipeline. Default: split all available cores evenly.",
    )
    parser.add_argument(
        "--reserve-cores",
        type=int,
        default=0,
        help="Leave this many cores unassigned (e.g. for the OS / other work).",
    )
    parser.add_argument(
        "--jobs-per-pipeline",
        type=int,
        default=None,
        help="Override --jobs passed to each pipeline. Default: one per assigned core.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Per-worker Torch/BLAS thread cap (also sets OMP/MKL_NUM_THREADS). Default 1.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for per-pipeline stdout/stderr logs. Default: a timestamped dir under ./parallel_runs.",
    )
    parser.add_argument(
        "--no-taskset",
        action="store_true",
        help="Do not wrap launches in taskset (rely on --cpu-ids affinity only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned core assignment and commands, then exit.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args after `--` forwarded verbatim to every run_experiment.py call.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    pipelines: list[str] = args.pipelines
    if len(set(pipelines)) != len(pipelines):
        print(
            "Warning: duplicate pipeline names will write to the same output dir; "
            "pass distinct --run-id via `-- --run-id ...` to avoid clobbering.",
            file=sys.stderr,
        )

    # `extra_args` includes a leading "--" from REMAINDER; drop it.
    extra_args = [a for a in args.extra_args if a != "--"]

    cores = available_cores()
    if args.reserve_cores > 0:
        if args.reserve_cores >= len(cores):
            raise SystemExit(f"--reserve-cores {args.reserve_cores} leaves no cores (have {len(cores)}).")
        cores = cores[: len(cores) - args.reserve_cores]

    groups = partition_cores(cores, len(pipelines), cores_per_group=args.cores_per_pipeline)

    use_taskset = (not args.no_taskset) and shutil.which("taskset") is not None
    if not args.no_taskset and not use_taskset:
        print("Note: taskset not found; relying on --cpu-ids affinity only.", file=sys.stderr)

    log_dir = args.log_dir or (
        Path.cwd() / "parallel_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    plans = []
    for pipeline, group in zip(pipelines, groups):
        cmd = build_command(
            pipeline,
            group,
            jobs=args.jobs_per_pipeline,
            torch_threads=args.torch_threads,
            use_taskset=use_taskset,
            extra_args=extra_args,
        )
        plans.append((pipeline, group, cmd))

    print(f"Available cores: {cores}")
    for pipeline, group, cmd in plans:
        print(f"  {pipeline:40s} cores={group}")
        print(f"      $ {' '.join(cmd)}")
    if args.dry_run:
        print("Dry run - nothing launched.")
        return 0
    print(f"Logs: {log_dir}")

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(args.torch_threads)
    env["MKL_NUM_THREADS"] = str(args.torch_threads)

    procs: list[tuple[str, subprocess.Popen, object]] = []
    for pipeline, _group, cmd in plans:
        log_path = log_dir / f"{pipeline}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
        procs.append((pipeline, proc, log_handle))
        print(f"Launched {pipeline} (pid {proc.pid}) -> {log_path}")

    def _terminate_all(*_: object) -> None:
        print("\nInterrupted - terminating pipelines...", file=sys.stderr)
        for _name, proc, _h in procs:
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGINT, _terminate_all)
    signal.signal(signal.SIGTERM, _terminate_all)

    results: dict[str, int] = {}
    try:
        for pipeline, proc, handle in procs:
            ret = proc.wait()
            handle.close()
            results[pipeline] = ret
            status = "OK" if ret == 0 else f"FAILED (exit {ret})"
            print(f"[{time.strftime('%H:%M:%S')}] {pipeline}: {status}")
    finally:
        for _name, proc, handle in procs:
            if not handle.closed:
                handle.close()

    failed = {name: code for name, code in results.items() if code != 0}
    if failed:
        print(f"\n{len(failed)} pipeline(s) failed: {sorted(failed)}", file=sys.stderr)
        return 1
    print(f"\nAll {len(results)} pipeline(s) completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
