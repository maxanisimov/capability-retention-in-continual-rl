#!/usr/bin/env python3
"""Run reward-shaped FrozenLake PSPO with a certified line-segment LID.

Reuse the shaping runner's safety-only initialization and post-enforcement raw
evaluation without editing source-locked files used by ongoing experiments.
Only region geometry changes by default; --verify-first additionally selects
exact proposal verification before computing a segment. This process-local
adapter records the actual variant and includes itself in source snapshots.
"""

from __future__ import annotations

import argparse
import os
import sys
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

from provably_safe_policy_optimisation import AdaptiveSafePPO  # noqa: E402

from projects.safe_policy_optimisation.scripts import (  # noqa: E402
    run_frozenlake_pspo_shaping as pspo,
)
from projects.safe_policy_optimisation.scripts import (  # noqa: E402
    run_frozenlake_shaping_sweep as sweep,
)


def worker_parser() -> argparse.ArgumentParser:
    parser = pspo.build_parser()
    parser.description = __doc__
    parser.set_defaults(size=16, total_timesteps=204800)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--verify-first", action="store_true")
    parser.add_argument("--segment-tolerance", type=float, default=1e-3)
    parser.add_argument("--segment-splits", type=int, default=4)
    parser.add_argument("--segment-max-splits", type=int, default=8)
    return parser


def run_worker(args: argparse.Namespace) -> dict:
    original_model = pspo.AdaptiveSafePPOV2
    original_write = pspo.write_json
    original_sources = pspo.source_paths
    verify_first = bool(getattr(args, "verify_first", False))
    model_cls = AdaptiveSafePPO if verify_first else original_model
    variant = "line_segment_verify_first" if verify_first else "line_segment"
    settings = {
        "safe_region_shape": "segment",
        "segment_tolerance": args.segment_tolerance,
        "segment_splits": args.segment_splits,
        "segment_max_splits": args.segment_max_splits,
    }

    def segment_model(*positional, **kwargs):
        kwargs.update(settings)
        if verify_first:
            # This setting belongs only to the region-first V2 implementation.
            kwargs.pop("region_update_mode")
        model = model_cls(*positional, **kwargs)
        if type(model) is not model_cls:
            raise AssertionError("Unexpected PSPO implementation")
        return model

    def write_variant(path, data):
        if path.name in {
            "config.json",
            "summary.json",
            "metrics.json",
            "certificate.json",
        }:
            data = {**data, "pspo_variant": variant}
        if path.name == "config.json":
            data["safety_enforcement"] = {
                **data["safety_enforcement"],
                **settings,
                "verify_first": verify_first,
                "implementation": model_cls.__name__,
                "variant": "segment_verify_first"
                if verify_first
                else "segment_region_first",
            }
        if path.name == "summary.json":
            diagnostics = data["adaptive_diagnostics"]
            if diagnostics["safe_region_shape"] != "segment" or diagnostics.get(
                "audit_candidates_exactly", False
            ):
                raise AssertionError("The requested segment geometry was not used")
            if verify_first and diagnostics["verifications_run"] < 1:
                raise AssertionError("Verify-first must verify a training proposal")
        return original_write(path, data)

    pspo.AdaptiveSafePPOV2 = segment_model
    pspo.write_json = write_variant
    pspo.source_paths = lambda: [*original_sources(), Path(__file__).resolve()]
    try:
        return pspo.run(args)
    finally:
        pspo.AdaptiveSafePPOV2 = original_model
        pspo.write_json = original_write
        pspo.source_paths = original_sources


def prepare(args: argparse.Namespace, cpus: list[int]) -> dict:
    verify_first = bool(getattr(args, "verify_first", False))
    # Include this adapter in both the supervisor's hash guard and its snapshot.
    original_sources = sweep.all_source_paths
    sweep.all_source_paths = lambda: sorted(
        set(original_sources()) | {Path(__file__).resolve()}
    )
    try:
        manifest = sweep.build_manifest(args, cpus)
    finally:
        sweep.all_source_paths = original_sources
    for job in manifest["jobs"]:
        command = job["command"]
        if command[5] != str(Path(pspo.__file__).resolve()):
            raise AssertionError("Unexpected worker command")
        command[5] = str(Path(__file__).resolve())
        command.append("--worker")
        if verify_first:
            command.append("--verify-first")
    manifest.update(
        pspo_variant="line_segment_verify_first" if verify_first else "line_segment",
        safe_region_shape="segment",
        verify_first=verify_first,
        segment_tolerance=1e-3,
        segment_splits=4,
        segment_max_splits=8,
    )
    return manifest


def main() -> int:
    if "--worker" in sys.argv:
        run_worker(worker_parser().parse_args())
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--total-timesteps", type=int, default=204800)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--screen-session", default="frozenlake16-pspo-segment")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify-first", action="store_true")
    parser.add_argument("--supervise", type=Path)
    args = parser.parse_args()
    if args.supervise:
        manifest = sweep.read_json(args.supervise)
        if manifest.get("pspo_variant") not in {
            "line_segment",
            "line_segment_verify_first",
        }:
            raise ValueError("Refusing to supervise a different PSPO variant")
        return sweep.supervise(args.supervise)
    if (
        args.output_dir is None
        or args.size < 16
        or min(args.total_timesteps, args.eval_episodes) <= 0
    ):
        parser.error("Require --output-dir, size >= 16, and positive budgets")
    args.methods = ("pspo",)
    cpus, evidence = sweep.baseline.free_physical_cpus(95.0)
    cpus = sorted(cpus, key=lambda cpu: (cpu < 32, cpu))
    manifest = prepare(args, cpus)
    manifest["cpu_availability_evidence"] = evidence
    path = args.output_dir.resolve() / "_orchestrator/launch_manifest.json"
    sweep.atomic_json(path, manifest)
    print(
        f"Prepared {len(manifest['jobs'])} {manifest['pspo_variant']} jobs: {path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
