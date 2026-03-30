#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REQUIRED_RESULTS_COLUMNS = {
    "Policy",
    "Task",
    "Trajectory Safety Rate",
    "Critical State Safety Rate",
    "Success Rate",
}


def _parse_task(task_value: Any) -> int:
    task_text = str(task_value).strip()
    try:
        return int(float(task_text))
    except ValueError:
        lower = task_text.lower()
        if lower.startswith("task"):
            suffix = lower.removeprefix("task").strip()
            return int(float(suffix))
        raise ValueError(f"Could not parse task value '{task_value}'")


def _to_float(value: Any, field_name: str, policy: str, task: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid numeric value for '{field_name}' in policy='{policy}', task={task}: {value}"
        ) from exc


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def _comparison_policy_order(policy_names: list[str]) -> list[str]:
    preferred = ["Source", "SafeAdapt", "UnsafeAdapt", "EWC"]
    ordered: list[str] = []
    for name in preferred:
        if name in policy_names:
            ordered.append(name)
    remainder = sorted([name for name in policy_names if name not in preferred])
    return ordered + remainder


def build_safety_comparison_csv(
    results_path: Path,
    comparison_path: Path,
    min_safety_accuracy: float,
) -> None:
    if not results_path.exists():
        raise FileNotFoundError(f"Downstream results table not found: {results_path}")

    with results_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {results_path}")

        missing = REQUIRED_RESULTS_COLUMNS.difference(reader.fieldnames)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns in {results_path}: {missing_str}")

        per_policy: dict[str, dict[int, dict[str, float | None]]] = {}
        for row in reader:
            policy = str(row["Policy"]).strip()
            task = _parse_task(row["Task"])
            if task not in (1, 2):
                continue

            if policy not in per_policy:
                per_policy[policy] = {}
            if task in per_policy[policy]:
                raise ValueError(
                    f"Duplicate row for policy='{policy}', task={task} in {results_path}",
                )

            try:
                critical_value = _to_optional_float(row["Critical State Safety Rate"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid numeric value for 'Critical State Safety Rate' "
                    f"in policy='{policy}', task={task}: {row['Critical State Safety Rate']}"
                ) from exc
            if task == 1 and critical_value is None:
                raise ValueError(
                    f"Task 1 requires 'Critical State Safety Rate' for policy='{policy}'.",
                )

            per_policy[policy][task] = {
                "trajectory_safety_rate": _to_float(
                    row["Trajectory Safety Rate"],
                    "Trajectory Safety Rate",
                    policy,
                    task,
                ),
                "critical_state_safety_rate": critical_value,
                "success_rate": _to_float(row["Success Rate"], "Success Rate", policy, task),
            }

    if "Source" not in per_policy or 1 not in per_policy["Source"]:
        raise ValueError(
            "Source Task 1 metrics are required to compute source-relative safety deltas.",
        )

    source_task1 = per_policy["Source"][1]
    if source_task1["critical_state_safety_rate"] is None:
        raise ValueError("Source Task 1 critical-state safety rate is required.")
    source_task1_critical = source_task1["critical_state_safety_rate"]
    source_task1_trajectory = source_task1["trajectory_safety_rate"]

    fieldnames = [
        "Policy",
        "task1_critical_state_safety_rate",
        "task1_trajectory_safety_rate",
        "task2_trajectory_safety_rate",
        "task2_success_rate",
        "delta_task1_critical_vs_source",
        "delta_task1_trajectory_vs_source",
        "retains_task1_critical_safety_vs_source",
        "meets_task1_critical_threshold",
    ]
    ordered_policies = _comparison_policy_order(list(per_policy.keys()))

    rows: list[dict[str, Any]] = []
    for policy in ordered_policies:
        task1 = per_policy[policy].get(1)
        task2 = per_policy[policy].get(2)

        task1_critical = (
            task1["critical_state_safety_rate"] if task1 is not None else None
        )
        task1_trajectory = (
            task1["trajectory_safety_rate"] if task1 is not None else None
        )
        task2_trajectory = (
            task2["trajectory_safety_rate"] if task2 is not None else None
        )
        task2_success = task2["success_rate"] if task2 is not None else None

        delta_task1_critical = (
            task1_critical - source_task1_critical
            if task1_critical is not None
            else None
        )
        delta_task1_trajectory = (
            task1_trajectory - source_task1_trajectory
            if task1_trajectory is not None
            else None
        )
        retains_task1_critical = (
            task1_critical >= source_task1_critical
            if task1_critical is not None
            else False
        )
        meets_threshold = (
            task1_critical >= min_safety_accuracy
            if task1_critical is not None
            else False
        )

        rows.append(
            {
                "Policy": policy,
                "task1_critical_state_safety_rate": task1_critical,
                "task1_trajectory_safety_rate": task1_trajectory,
                "task2_trajectory_safety_rate": task2_trajectory,
                "task2_success_rate": task2_success,
                "delta_task1_critical_vs_source": delta_task1_critical,
                "delta_task1_trajectory_vs_source": delta_task1_trajectory,
                "retains_task1_critical_safety_vs_source": retains_task1_critical,
                "meets_task1_critical_threshold": meets_threshold,
            }
        )

    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_and_log(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> Running: {' '.join(cmd)}")
    print(f">>> Log file: {log_path}")

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Taxi train_source_policy.py then downstream_adaptation.py with logging."
    )
    parser.add_argument("--cfg", type=str, default="different_dest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=64)

    parser.add_argument("--source-total-steps", type=int, default=500_000)
    parser.add_argument("--downstream-total-timesteps", type=int, default=200_000)

    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--ewc-lambda", type=float, default=5_000.0)
    parser.add_argument("--rashomon-n-iters", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--source-mode", type=str, default="safe", choices=["original", "safe"])
    parser.add_argument("--min-safety-accuracy", type=float, default=1.0)
    parser.add_argument("--safety-epochs", type=int, default=2_000)
    parser.add_argument(
        "--comparison-csv-name",
        type=str,
        default="safety_method_comparison.csv",
        help="Output file name for post-processed safety comparison CSV under downstream output dir.",
    )

    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable to use for both scripts.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root output dir (default: <this_folder>/outputs).",
    )
    parser.add_argument(
        "--logs-root",
        type=str,
        default=None,
        help="Root logs dir (default: <this_folder>/logs).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train_source_policy.py"
    downstream_script = script_dir / "downstream_adaptation.py"

    output_root = Path(args.output_root) if args.output_root else script_dir / "outputs"
    logs_root = Path(args.logs_root) if args.logs_root else script_dir / "logs"

    run_dir = output_root / args.cfg / str(args.seed)
    source_output_dir = run_dir / "source"
    downstream_output_dir = run_dir / "downstream"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_logs_dir = logs_root / args.cfg / str(args.seed) / run_id

    train_log = run_logs_dir / "01_train_source_policy.log"
    downstream_log = run_logs_dir / "02_downstream_adaptation.log"

    train_cmd = [
        args.python_bin,
        str(train_script),
        "--cfg",
        args.cfg,
        "--seed",
        str(args.seed),
        "--total-steps",
        str(args.source_total_steps),
        "--hidden",
        str(args.hidden),
        "--ent-coef",
        str(args.ent_coef),
        "--eval-episodes",
        str(args.eval_episodes),
        "--device",
        args.device,
        "--output-dir",
        str(source_output_dir),
    ]

    downstream_cmd = [
        args.python_bin,
        str(downstream_script),
        "--cfg",
        args.cfg,
        "--seed",
        str(args.seed),
        "--source-dir",
        str(run_dir),
        "--output-dir",
        str(downstream_output_dir),
        "--total-timesteps",
        str(args.downstream_total_timesteps),
        "--ent-coef",
        str(args.ent_coef),
        "--ewc-lambda",
        str(args.ewc_lambda),
        "--rashomon-n-iters",
        str(args.rashomon_n_iters),
        "--eval-episodes",
        str(args.eval_episodes),
        "--hidden",
        str(args.hidden),
        "--device",
        args.device,
        "--source-mode",
        args.source_mode,
        "--min-safety-accuracy",
        str(args.min_safety_accuracy),
        "--safety-epochs",
        str(args.safety_epochs),
    ]

    print("=" * 80)
    print("Taxi pipeline: train_source_policy -> downstream_adaptation")
    print(f"Config: {args.cfg}, Seed: {args.seed}, Source mode: {args.source_mode}")
    print(f"Run dir: {run_dir}")
    print(f"  Source:     {source_output_dir}")
    print(f"  Downstream: {downstream_output_dir}")
    print(f"Logs dir: {run_logs_dir}")
    print("=" * 80)

    run_and_log(train_cmd, train_log)
    run_and_log(downstream_cmd, downstream_log)

    results_csv = downstream_output_dir / "results_table.csv"
    comparison_name = Path(args.comparison_csv_name)
    comparison_csv = (
        comparison_name
        if comparison_name.is_absolute()
        else downstream_output_dir / comparison_name
    )
    build_safety_comparison_csv(
        results_path=results_csv,
        comparison_path=comparison_csv,
        min_safety_accuracy=args.min_safety_accuracy,
    )

    print("\nPipeline complete.")
    print(f"Logs saved in: {run_logs_dir}")
    print(f"Safety comparison CSV saved to: {comparison_csv}")


if __name__ == "__main__":
    main()
