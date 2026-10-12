#!/usr/bin/env python3
"""Wait for both matched ten-seed sweeps, then update the comparison figure.

This is postprocessing only. Never substitutes partial runs or the old pilot.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stats(values):
    if len(values) != 10:
        raise ValueError("Final reporting requires all ten seeds")
    return {
        "mean": statistics.mean(values),
        "two_standard_errors": 2 * statistics.stdev(values) / math.sqrt(10),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-root", type=Path, required=True)
    parser.add_argument("--pspo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-hours", type=float, default=6)
    args = parser.parse_args()
    roots = {"ppo": args.ppo_root.resolve(), "pspo": args.pspo_root.resolve()}
    for method, root in roots.items():
        manifest = read(root / "_orchestrator/launch_manifest.json")
        if (
            manifest["size"] != 64
            or manifest["seeds"] != list(range(10))
            or manifest["methods"] != [method]
            or manifest["requested_timesteps_per_run"] != 200000
        ):
            raise ValueError(f"Not the requested matched short sweep: {root}")
        if method == "pspo" and (
            manifest["pspo_variant"] != "line_segment_verify_first"
            or manifest["pspo_safety_frequency_rollouts"] != 1
        ):
            raise ValueError("Require frequency-1 segment verify-first PSPO")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "short64_comparison_status.json"
    deadline = time.monotonic() + args.timeout_hours * 3600
    while True:
        statuses = {m: read(r / "status.json") for m, r in roots.items()}
        if any(s["failed"] for s in statuses.values()):
            save(status_path, {"status": "failed", "training_status": statuses})
            raise RuntimeError(
                "A requested training seed failed; no partial figure produced"
            )
        if all(s["complete"] == 10 for s in statuses.values()):
            break
        if time.monotonic() >= deadline:
            save(status_path, {"status": "timeout", "training_status": statuses})
            raise TimeoutError("Training still incomplete; no partial figure produced")
        save(
            status_path, {"status": "waiting_for_training", "training_status": statuses}
        )
        time.sleep(30)
    save(
        status_path,
        {"status": "generating_matched_figure", "training_status": statuses},
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "plot_frozenlake_freq1_reward_safety_curves.py"),
            "--ppo64-root",
            str(roots["ppo"]),
            "--pspo64-root",
            str(roots["pspo"]),
            "--layout64-max-steps",
            "200704",
            "--output-dir",
            str(args.output_dir),
        ],
        cwd=REPO,
        check=True,
    )
    seed_results = []
    for method, root in roots.items():
        for seed in range(10):
            directory = root / method / f"seed{seed}"
            summary = read(directory / "summary.json")
            metrics = read(directory / "metrics.json")
            if summary["final_timesteps"] != 200704 or metrics["eval_episodes"] != 100:
                raise ValueError("Unexpected final training/evaluation budget")
            seed_results.append(
                {
                    "method": method,
                    "seed": seed,
                    "standard_total_reward": metrics["success"]["success_rate"],
                    "goal_success": metrics["success"]["success_rate"],
                    "safety_rate": metrics["safety"]["safety_rate"],
                    "step_penalised_total_reward": metrics["reward"][
                        "mean_total_reward"
                    ],
                    "training_seconds": summary[
                        "training_seconds_excluding_curve_evaluation"
                    ],
                    "lid_seconds": summary.get("lid_computation_seconds", 0),
                }
            )
    metric_names = [k for k in seed_results[0] if k not in {"method", "seed"}]
    report = {
        "requested_training_steps": 200000,
        "actual_training_steps": 200704,
        "seeds_per_method": 10,
        "evaluation_episodes_per_seed": 100,
        "reward_definition": "Standard FrozenLake reward: 1 at goal, 0 otherwise; no step cost or shaping.",
        "uncertainty": "mean +/- two sample standard errors across training seeds",
        "training_time_definition": "Excludes curve evaluations, final evaluation and model initialisation; includes safety enforcement.",
        "source_roots": {m: str(r) for m, r in roots.items()},
        "seed_results": seed_results,
        "methods": {
            method: {
                key: stats([r[key] for r in seed_results if r["method"] == method])
                for key in metric_names
            }
            for method in roots
        },
    }
    save(args.output_dir / "short64_comparison_results.json", report)
    save(status_path, {"status": "complete", "training_status": statuses})
    print(json.dumps(report["methods"], indent=2), flush=True)


if __name__ == "__main__":
    main()
