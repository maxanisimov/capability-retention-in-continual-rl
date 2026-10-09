#!/usr/bin/env python3
"""Automatically report a baseline sweep alongside completed PPO/PSPO references."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

METHOD_LABELS = {
    "ppo": "PPO",
    "ppo_lagrangian": "PPO-Lagrangian",
    "ppo_pid_lagrangian": "PPO-PID-Lagrangian",
    "cpo": "CPO",
    "ppo_shield": "PPO-Shield (shield on)",
    "ppo_shield_nominal": "PPO-Shield (shield off)",
    "pspo": "PSPO",
}
KEYS = ("total_reward", "safety_rate", "goal_success", "training_seconds")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def seed_rows(root: Path) -> list[dict]:
    path = root / "seed_results.csv"
    if not path.exists():
        return []
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        for key in (*KEYS, "lid_seconds"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError(f"Nonfinite result in {path}")
    return rows


def aggregate_rows(rows: list[dict]) -> dict:
    aggregate = {}
    for method in METHOD_LABELS:
        group = [row for row in rows if row["method"] == method]
        if not group:
            continue
        seeds = [row["seed"] for row in group]
        if len(set(seeds)) != len(seeds) or set(seeds) - set(range(10)):
            raise ValueError(f"Duplicate or unexpected training seeds for {method}")
        aggregate[method] = {"seed_count": len(group)}
        for key in (*KEYS, "lid_seconds"):
            xs = [row[key] for row in group]
            aggregate[method][key] = {
                "mean": statistics.mean(xs),
                "two_standard_errors": 2 * statistics.stdev(xs) / math.sqrt(len(xs))
                if len(xs) > 1
                else None,
            }
    return aggregate


def verify_final_protocol(root: Path, reference: Path, rows: list[dict]) -> None:
    """Fail closed rather than reporting a mixed-budget or shaped evaluation."""
    control = read_json(reference / "ppo/seed0/config.json")
    for row in rows:
        source = reference if row["method"] in ("ppo", "pspo") else root
        method = (
            "ppo_shield" if row["method"] == "ppo_shield_nominal" else row["method"]
        )
        directory = source / method / f"seed{row['seed']}"
        config = read_json(directory / "config.json")
        summary = read_json(directory / "summary.json")
        metrics_name = (
            "metrics_nominal.json"
            if row["method"] == "ppo_shield_nominal"
            else "metrics.json"
        )
        metrics = read_json(directory / metrics_name)
        for key in ("env_kwargs", "max_episode_steps", "requested_timesteps"):
            if config[key] != control[key]:
                raise ValueError(f"Unmatched {key}: {directory}")
        for key, value in control["architecture"].items():
            if config["architecture"][key] != value:
                raise ValueError(f"Unmatched architecture {key}: {directory}")
        for key, value in control["training_hyperparameters"].items():
            if config["training_hyperparameters"][key] != value:
                raise ValueError(f"Unmatched {key}: {directory}")
        for key in ("enabled", "scale", "gamma", "timeout_mode", "training_only"):
            if config["shaping"][key] != control["shaping"][key]:
                raise ValueError(f"Unmatched shaping {key}: {directory}")
        if summary["final_timesteps"] != 401408 or metrics["eval_episodes"] != 100:
            raise ValueError(f"Incomplete production budget or evaluation: {directory}")
        if metrics["reward_shaping_enabled"]:
            raise ValueError(f"Shaped final evaluation: {directory}")
        expected_policy = (
            "greedy_runtime_shield"
            if row["method"] == "ppo_shield"
            else "greedy_unshielded"
        )
        if metrics["evaluation_policy"] != expected_policy:
            raise ValueError(f"Incorrect final evaluation policy: {directory}")
        checks = {
            "total_reward": metrics["reward"]["mean_total_reward"],
            "safety_rate": metrics["safety"]["safety_rate"],
            "goal_success": metrics["success"]["success_rate"],
            "training_seconds": summary["training_seconds_excluding_curve_evaluation"],
        }
        if any(
            not math.isclose(row[key], value, abs_tol=1e-9)
            for key, value in checks.items()
        ):
            raise ValueError(f"CSV and final artifacts disagree: {directory}")


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_comparison(root: Path, reference: Path) -> bool:
    status = read_json(root / "status.json")
    reference_status = read_json(reference / "status.json")
    if reference_status["complete"] != 20 or reference_status["failed"]:
        raise ValueError("Reference PPO/PSPO cohort is not fully complete")
    rows = seed_rows(reference) + seed_rows(root)
    aggregate = aggregate_rows(rows)
    terminal = status["complete"] + status["failed"] == status["total"]
    final = terminal and status["failed"] == 0
    if final:
        if set(aggregate) != set(METHOD_LABELS) or any(
            a["seed_count"] != 10 for a in aggregate.values()
        ):
            raise ValueError("Completed sweep does not contain every ten-seed result")
        verify_final_protocol(root, reference, rows)
    report = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "final": final,
        "baseline_status": status,
        "reference_cohort": str(reference),
        "uncertainty": "mean +/- two sample standard errors over training seeds",
        "protocol_verified": final,
        "methods": aggregate,
    }
    atomic_text(root / "comparison.json", json.dumps(report, indent=2) + "\n")
    csv_text = io.StringIO()
    writer = csv.DictWriter(csv_text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(root / "comparison_seed_results.csv", csv_text.getvalue())
    lines = [
        "# FrozenLake 128x128: shaped training, 400k-step comparison",
        "",
        "FINAL: all forty baseline jobs completed; matched protocol verified."
        if final
        else f"PARTIAL: {status['complete']}/{status['total']} baseline jobs complete, {status['failed']} failed; missing seeds are not final results.",
        "",
        "Values are mean +/- two standard errors across training seeds.",
        "Final evaluation uses original rewards, 100 episodes per seed.",
        "",
        "| Method | Seeds | Total reward | Safety (%) | Goal (%) | Training (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, label in METHOD_LABELS.items():
        a = aggregate.get(method)
        if a is None:
            lines.append(f"| {label} | 0/10 | pending | pending | pending | pending |")
            continue
        cells = []
        for key in KEYS:
            scale = 100 if key in ("safety_rate", "goal_success") else 1
            digits = 3 if key == "total_reward" else 2
            mean = a[key]["mean"] * scale
            se2 = a[key]["two_standard_errors"]
            cells.append(
                f"{mean:.{digits}f} +/- {se2 * scale:.{digits}f}"
                if se2 is not None
                else f"{mean:.{digits}f} (SE unavailable)"
            )
        lines.append(f"| {label} | {a['seed_count']}/10 | " + " | ".join(cells) + " |")
    lines += [
        "",
        "PPO/PSPO are reused from the completed reference cohort, not retrained.",
        "PPO-Shield shield-on and shield-off rows evaluate the same learned checkpoint; its training time is shared.",
        "Training times exclude periodic and final evaluation.",
        f"Updated UTC: {report['updated_utc']}.",
        "",
    ]
    atomic_text(root / "comparison.md", "\n".join(lines))
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            terminal = write_comparison(args.root, args.reference)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as error:
            if not args.wait:
                raise
            # Retry only while the controller is still writing an unfinished sweep.
            status = read_json(args.root / "status.json")
            if status["complete"] + status["failed"] == status["total"]:
                raise
            print(f"Waiting for consistent completion records: {error}", flush=True)
            terminal = False
        if terminal or not args.wait:
            print((args.root / "comparison.md").read_text(), flush=True)
            return int(read_json(args.root / "status.json")["failed"] > 0)
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
