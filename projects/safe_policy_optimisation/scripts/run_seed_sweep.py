#!/usr/bin/env python
"""Run pipelines across multiple seeds (in parallel, core-isolated) and aggregate.

Covers two cases with one CLI:

1. one pipeline across N seeds, with the metrics aggregated across seeds;
2. several pipelines, each across N seeds.

All runs execute concurrently on disjoint CPU-core slots (no core clashes). For
each pipeline the shield + Rashomon set are built once, by the base seed. If a
``--rashomon-dir`` is forwarded (after ``--``), the remaining seeds fan out as
soon as that shared set exists on disk - not once the base seed's own training
finishes too, since that training has nothing to do with what the other seeds
need. Without ``--rashomon-dir`` (or once the base seed's own run exits, as a
fallback) the remaining seeds fan out then instead. After a pipeline's seeds
finish, the per-stage ``metrics.json`` files (success / total reward / safety)
are aggregated to mean +/- std across seeds.

Examples
--------
One pipeline, 5 seeds::

    python projects/safe_policy_optimisation/scripts/run_seed_sweep.py \
        --pipeline deterministic_minipacman --n-seeds 5

Several pipelines, explicit seeds, 8 cores each::

    python .../run_seed_sweep.py \
        --pipeline deterministic_minipacman \
        --pipeline deterministic_colour_bomb \
        --seeds 0 1 2 3 --cores-per-job 8

Preview the (pipeline x seed) grid and core-slot plan::

    python .../run_seed_sweep.py --pipeline a --n-seeds 3 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from projects.safe_policy_optimisation.utils.metrics import (
    aggregate_seed_metrics,
    seed_metrics_rows,
)
from projects.safe_policy_optimisation.utils.parallel import (
    Job,
    available_cores,
    partition_cores,
    run_core_slot_jobs,
)

RUN_EXPERIMENT = Path(__file__).resolve().parent.parent / "run_experiment.py"

# Headline metrics shown in the terminal summary (matched as a key suffix).
HEADLINE_SUFFIXES = (
    "success.success_rate",
    "reward.mean_total_reward",
    "safety.safety_rate",
    "safety.violation_percentage",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pipelines across seeds on disjoint cores and aggregate metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipeline", dest="pipelines", action="append", required=True, metavar="NAME",
        help="Pipeline name (repeat for several pipelines).",
    )
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", type=int, nargs="+", help="Explicit list of seeds.")
    seeds.add_argument("--n-seeds", type=int, help="Number of seeds (with --base-seed).")
    parser.add_argument("--base-seed", type=int, default=0, help="First seed for --n-seeds (default 0).")
    parser.add_argument(
        "--cores-per-job", type=int, default=None,
        help="CPU cores per run. Default: divide cores by the number of parallel slots.",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=None,
        help="Max concurrent runs. Default: as many disjoint slots as fit.",
    )
    parser.add_argument(
        "--torch-threads", type=int, default=1,
        help="Per-worker Torch/BLAS thread cap (also sets OMP/MKL_NUM_THREADS). Default 1.",
    )
    parser.add_argument(
        "--sweep-root", type=Path, default=None,
        help="Root dir for sweep outputs. Default: outputs/_sweeps/<timestamp>.",
    )
    parser.add_argument("--no-taskset", action="store_true", help="Do not pin with taskset.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose summary.json already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    parser.add_argument(
        "extra_args", nargs=argparse.REMAINDER,
        help="Args after `--` forwarded verbatim to every run_experiment.py call.",
    )
    return parser.parse_args(argv)


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds is not None:
        return list(dict.fromkeys(args.seeds))  # de-dup, keep order
    n = args.n_seeds if args.n_seeds is not None else 1
    return [args.base_seed + i for i in range(n)]


def _safe_name(pipeline: str) -> str:
    return pipeline.replace("/", "_")


def run_dir_for(sweep_root: Path, pipeline: str, seed: int) -> Path:
    return sweep_root / _safe_name(pipeline) / f"seed{seed}"


def is_complete(run_dir: Path) -> bool:
    return (run_dir / "summary.json").exists()


def _flag_value(argv: list[str], flag: str) -> str | None:
    """Return the value following ``flag`` in an argv-style list, or None."""
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def rashomon_artifacts_ready(rashomon_dir: Path | None) -> bool:
    """True once the shared Rashomon set the non-warmup seeds depend on is
    fully built and safe to reuse - not just started.

    ``summary.json`` is compute_shield_rashomon_set.py's last write, strictly
    after every ``.pt`` artifact in the same directory (dataset, base policy,
    bounded model, param bounds) - see that module for the exact write order.
    Gating on it (rather than e.g. the warmup process exiting) lets the other
    seeds start as soon as the shared box exists, instead of waiting for the
    warmup seed's own, unrelated policy training to also finish.
    """
    return rashomon_dir is not None and (rashomon_dir / "summary.json").exists()


def make_seed_job(
    pipeline: str, seed: int, *, sweep_root: Path, torch_threads: int,
    extra_args: list[str], log_dir: Path, is_warmup: bool, rest_seeds: list[int],
) -> Job:
    pdir = sweep_root / _safe_name(pipeline)
    run_dir = pdir / f"seed{seed}"

    def make_command(core_ids: list[int]) -> list[str]:
        ids = ",".join(str(c) for c in core_ids)
        return [
            sys.executable, str(RUN_EXPERIMENT),
            "--pipeline", pipeline,
            "--seed", str(seed),
            "--output-dir", str(pdir),
            "--run-id", f"seed{seed}",
            "--cpu-ids", ids,
            "--jobs", str(max(1, len(core_ids))),
            "--torch-num-threads", str(torch_threads),
            *extra_args,
        ]

    tag = "warmup" if is_warmup else "seed"
    return Job(
        name=f"{pipeline}:seed{seed}" + (":warmup" if is_warmup else ""),
        make_command=make_command,
        log_path=log_dir / f"{_safe_name(pipeline)}_seed{seed}_{tag}.log",
        meta={"pipeline": pipeline, "seed": seed, "warmup": is_warmup, "rest_seeds": rest_seeds, "run_dir": run_dir},
    )


def write_aggregate(pipeline: str, sweep_root: Path, seeds: list[int]) -> dict | None:
    seed_to_dir = {s: run_dir_for(sweep_root, pipeline, s) for s in seeds}
    present = {s: d for s, d in seed_to_dir.items() if d.is_dir()}
    if not present:
        return None
    aggregate = aggregate_seed_metrics(present)
    out_dir = sweep_root / _safe_name(pipeline) / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    (out_dir / "aggregated_metrics.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = seed_metrics_rows(aggregate)
    with (out_dir / "aggregated_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "mean", "std", "n", "min", "max"])
        writer.writeheader()
        writer.writerows(rows)
    return aggregate


def print_summary(pipeline: str, aggregate: dict) -> None:
    print(f"\n=== {pipeline}: aggregated over seeds {aggregate['seeds']} ===")
    metrics = aggregate["metrics"]
    headline = [k for k in sorted(metrics) if k.endswith(HEADLINE_SUFFIXES)]
    if not headline:
        headline = sorted(metrics)
    for key in headline:
        stat = metrics[key]
        print(f"  {key:55s} {stat['mean']:.4f} +/- {stat['std']:.4f}  (n={stat['n']})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pipelines = list(dict.fromkeys(args.pipelines))
    seeds = resolve_seeds(args)
    if not seeds:
        raise SystemExit("No seeds to run.")
    extra_args = [a for a in args.extra_args if a != "--"]

    total_jobs = len(pipelines) * len(seeds)
    cores = available_cores()

    # Decide slot count and cores-per-slot.
    if args.cores_per_job is not None:
        cores_per_job = args.cores_per_job
        n_slots = max(1, len(cores) // cores_per_job)
    else:
        n_slots = min(total_jobs, len(cores))
        if args.max_parallel is not None:
            n_slots = min(n_slots, args.max_parallel)
        n_slots = max(1, n_slots)
        cores_per_job = max(1, len(cores) // n_slots)
    if args.max_parallel is not None:
        n_slots = min(n_slots, args.max_parallel)
    n_slots = max(1, min(n_slots, len(cores) // cores_per_job or 1))
    slots = partition_cores(cores, n_slots, cores_per_group=cores_per_job)

    sweep_root = args.sweep_root or (Path.cwd() / "outputs" / "_sweeps" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_dir = sweep_root / "_logs"

    use_taskset = (not args.no_taskset) and shutil.which("taskset") is not None
    if not args.no_taskset and not use_taskset:
        print("Note: taskset not found; relying on --cpu-ids affinity only.", file=sys.stderr)

    print(f"Pipelines: {pipelines}")
    print(f"Seeds: {seeds}  ->  {total_jobs} runs total")
    print(f"Cores: {len(cores)} available; {n_slots} parallel slot(s) x {cores_per_job} core(s)")
    for i, slot in enumerate(slots):
        print(f"  slot {i}: cores {slot}")
    print(f"Per pipeline: base seed {seeds[0]} runs first and builds the shield + Rashomon "
          f"set; seeds {seeds[1:]} fan out as soon as that shared set exists on disk "
          f"(not once the base seed's own run also finishes).")
    print(f"Sweep root: {sweep_root}")

    if args.dry_run:
        for pipeline in pipelines:
            for seed in seeds:
                role = "warmup" if seed == seeds[0] else "seed"
                cmd = make_seed_job(
                    pipeline, seed, sweep_root=sweep_root, torch_threads=args.torch_threads,
                    extra_args=extra_args, log_dir=log_dir, is_warmup=(role == "warmup"),
                    rest_seeds=seeds[1:],
                ).make_command(slots[0])
                prefix = ["taskset", "-c", ",".join(map(str, slots[0]))] if use_taskset else []
                print(f"  [{pipeline} seed{seed} {role}] $ {' '.join(prefix + cmd)}")
        print("Dry run - nothing launched.")
        return 0

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(args.torch_threads)
    env["MKL_NUM_THREADS"] = str(args.torch_threads)

    # Build initial jobs: a warmup per pipeline (base seed), unless artifacts already
    # exist (skip-existing + complete base run) in which case enqueue the rest directly.
    #
    # `triggered` guards against launching a pipeline's non-warmup seeds twice:
    # they can now start two ways - early, via on_poll, as soon as the shared
    # Rashomon set appears on disk (while the warmup job's own training is
    # still running), or via on_complete when the warmup job exits (the only
    # path before, and still the fallback whenever there's no --rashomon-dir
    # to watch, or the sweep is reusing an existing complete base run).
    initial_jobs: list[Job] = []
    triggered: dict[str, bool] = {pipeline: False for pipeline in pipelines}
    pending_warmups: dict[str, Job] = {}
    for pipeline in pipelines:
        base, rest = seeds[0], seeds[1:]
        base_complete = args.skip_existing and is_complete(run_dir_for(sweep_root, pipeline, base))
        if base_complete:
            print(f"Skipping completed base run {pipeline} seed{base}; assuming shared artifacts exist.")
            triggered[pipeline] = True
            for seed in rest:
                if args.skip_existing and is_complete(run_dir_for(sweep_root, pipeline, seed)):
                    continue
                initial_jobs.append(make_seed_job(
                    pipeline, seed, sweep_root=sweep_root, torch_threads=args.torch_threads,
                    extra_args=extra_args, log_dir=log_dir, is_warmup=False, rest_seeds=[]))
        else:
            warmup_job = make_seed_job(
                pipeline, base, sweep_root=sweep_root, torch_threads=args.torch_threads,
                extra_args=extra_args, log_dir=log_dir, is_warmup=True, rest_seeds=rest)
            initial_jobs.append(warmup_job)
            pending_warmups[pipeline] = warmup_job

    rashomon_dir_arg = _flag_value(extra_args, "--rashomon-dir")
    rashomon_dir = Path(rashomon_dir_arg) if rashomon_dir_arg else None

    def follow_up_jobs(pipeline: str, rest_seeds: list[int]) -> list[Job]:
        follow_ups = []
        for seed in rest_seeds:
            if args.skip_existing and is_complete(run_dir_for(sweep_root, pipeline, seed)):
                continue
            follow_ups.append(make_seed_job(
                pipeline, seed, sweep_root=sweep_root, torch_threads=args.torch_threads,
                extra_args=extra_args, log_dir=log_dir, is_warmup=False, rest_seeds=[]))
        return follow_ups

    def on_poll() -> list[Job]:
        # extra_args (and so --rashomon-dir) is one shared block forwarded to
        # every pipeline in this sweep, so a single readiness check covers all
        # still-pending warmups.
        if not rashomon_artifacts_ready(rashomon_dir):
            return []
        new_jobs: list[Job] = []
        for pipeline, warmup_job in pending_warmups.items():
            if triggered[pipeline]:
                continue
            triggered[pipeline] = True
            print(f"[{time.strftime('%H:%M:%S')}] shared Rashomon set ready for {pipeline} "
                  f"({rashomon_dir}); launching seeds {warmup_job.meta['rest_seeds']} without "
                  f"waiting for the warmup run's own training to finish.")
            new_jobs.extend(follow_up_jobs(pipeline, warmup_job.meta["rest_seeds"]))
        return new_jobs

    def on_complete(job: Job, returncode: int) -> list[Job]:
        if not job.meta.get("warmup"):
            return []
        pipeline = job.meta["pipeline"]
        if returncode != 0:
            if not triggered[pipeline]:
                print(f"Warmup for {pipeline} failed (exit {returncode}); skipping its remaining seeds.",
                      file=sys.stderr)
            return []
        if triggered[pipeline]:
            return []  # already launched early via on_poll
        triggered[pipeline] = True
        return follow_up_jobs(pipeline, job.meta["rest_seeds"])

    results = run_core_slot_jobs(
        initial_jobs, slots=slots, env=env, use_taskset=use_taskset,
        on_complete=on_complete, on_poll=on_poll,
    )

    # Aggregate per pipeline and report.
    for pipeline in pipelines:
        aggregate = write_aggregate(pipeline, sweep_root, seeds)
        if aggregate is None:
            print(f"\n{pipeline}: no metrics found to aggregate.", file=sys.stderr)
        else:
            print_summary(pipeline, aggregate)

    failed = {name: code for name, code in results.items() if code != 0}
    print(f"\nSweep complete: {len(results)} run(s), {len(failed)} failed. Outputs under {sweep_root}")
    if failed:
        print(f"Failed runs: {sorted(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
