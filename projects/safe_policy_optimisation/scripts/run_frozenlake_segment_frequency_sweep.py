#!/usr/bin/env python3
"""Launch segment verify-first FrozenLake with a configurable train-phase cadence.

Reuse source-locked runners without changing them. Frequency 1 certifies the
aggregate update after every full rollout/optimisation phase, not each SGD step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "core")]

from projects.safe_policy_optimisation.scripts import (  # noqa: E402
    run_frozenlake_segment_shaping as segment,
)

sweep = segment.sweep


def set_frequency(manifest: dict, frequency: int) -> dict:
    if frequency <= 0:
        raise ValueError("Enforcement frequency must be positive")
    for job in manifest["jobs"]:
        command = job["command"]
        if command.count("--safety-frequency") != 1 or job["method"] != "pspo":
            raise ValueError("Require exactly one cadence flag per PSPO job")
        command[command.index("--safety-frequency") + 1] = str(frequency)
    manifest["pspo_safety_frequency_rollouts"] = frequency
    manifest["enforcement_semantics"] = (
        "After every frequency complete rollout-level PPO train phases; "
        "not after each internal minibatch SGD step."
    )
    return manifest


def active_reserved_cpus(runs_root: Path) -> set[int]:
    """Exclude queued as well as running jobs belonging to live supervisors."""
    reserved = set()
    for manifest_path in runs_root.glob("*/_orchestrator/launch_manifest.json"):
        supervisor = manifest_path.parent / "supervisor.json"
        if not supervisor.exists():
            continue
        pid = int(sweep.read_json(supervisor)["pid"])
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        # Avoid treating a reused process identifier as an experiment supervisor.
        if str(manifest_path).encode() not in cmdline:
            continue
        manifest = sweep.read_json(manifest_path)
        for job in manifest["jobs"]:
            record = (
                Path(manifest["output_dir"])
                / "_process"
                / (job["id"].replace("/", "_") + ".json")
            )
            if record.exists() and sweep.read_json(record)["status"] in {
                "complete",
                "failed",
            }:
                continue
            reserved.add(int(job["cpu"]))
    return reserved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--total-timesteps", type=int, default=204800)
    parser.add_argument("--safety-frequency", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument(
        "--screen-session", default="frozenlake16-pspo-segment-vf-freq1"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--supervise", type=Path)
    args = parser.parse_args()
    if args.supervise:
        manifest = sweep.read_json(args.supervise)
        frequency = manifest["pspo_safety_frequency_rollouts"]
        if manifest["pspo_variant"] != "line_segment_verify_first" or frequency <= 0:
            raise ValueError("Unexpected variant/cadence")
        for job in manifest["jobs"]:
            command = job["command"]
            if (
                command.count("--safety-frequency") != 1
                or int(command[command.index("--safety-frequency") + 1]) != frequency
            ):
                raise ValueError("Worker cadence differs from manifest")
        return sweep.supervise(args.supervise)
    if (
        args.output_dir is None
        or args.size < 16
        or min(args.total_timesteps, args.safety_frequency, args.eval_episodes) <= 0
    ):
        parser.error("Require --output-dir, size >= 16 and positive budgets/cadence")
    args.verify_first = True
    args.methods = ("pspo",)
    cpus, evidence = sweep.baseline.free_physical_cpus(95.0)
    reserved = active_reserved_cpus(args.output_dir.resolve().parent)
    cpus = sorted(set(cpus) - reserved, key=lambda cpu: (cpu < 32, cpu))
    original_sources = sweep.all_source_paths
    sweep.all_source_paths = lambda: sorted(
        set(original_sources()) | {Path(__file__).resolve()}
    )
    try:
        manifest = set_frequency(segment.prepare(args, cpus), args.safety_frequency)
    finally:
        sweep.all_source_paths = original_sources
    manifest["cpu_availability_evidence"] = evidence
    manifest["excluded_active_reserved_cpus"] = sorted(reserved)
    path = args.output_dir.resolve() / "_orchestrator/launch_manifest.json"
    sweep.atomic_json(path, manifest)
    print(
        f"Prepared {len(manifest['jobs'])} jobs, frequency {args.safety_frequency}: {path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
