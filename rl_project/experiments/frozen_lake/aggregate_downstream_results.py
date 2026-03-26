#!/usr/bin/env python3
"""Aggregate FrozenLake downstream results across seeds.

Expected per-seed file structure:
  outputs/standard_4x4/<seed>/downstream/results_table.csv

This script reads all available seed tables, aligns rows by (Policy, Task), and
computes mean/std across seeds for each metric cell.

Examples:
  python aggregate_downstream_results.py
  python aggregate_downstream_results.py --seeds 42 43 44
  python aggregate_downstream_results.py --base-dir outputs/standard_4x4 --output-dir outputs/standard_4x4/aggregated
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_base = script_dir / "outputs" / "standard_4x4_v2"
    default_out = default_base / "aggregated"

    parser = argparse.ArgumentParser(description="Aggregate downstream results across seeds.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_base,
        help="Directory containing seed folders (default: outputs/standard_4x4).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit seed list. If omitted, all numeric seed folders are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help="Directory for aggregated outputs.",
    )
    return parser.parse_args()


def _discover_seeds(base_dir: Path, explicit_seeds: list[int] | None) -> list[int]:
    if explicit_seeds is not None and len(explicit_seeds) > 0:
        return sorted(set(explicit_seeds))

    seeds: list[int] = []
    for child in base_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            seeds.append(int(child.name))
    return sorted(seeds)


def _candidate_result_paths(seed_dir: Path) -> Iterable[Path]:
    # Support both possible names to be resilient to user naming.
    yield seed_dir / "results_table.csv"
    yield seed_dir / "results.table.csv"


def _load_seed_table(base_dir: Path, seed: int) -> pd.DataFrame | None:
    seed_dir = base_dir / str(seed)
    for path in _candidate_result_paths(seed_dir):
        if path.exists():
            df = pd.read_csv(path)
            df["seed"] = seed
            return df
    return None


def _validate_columns(df: pd.DataFrame, path_hint: str) -> None:
    required_cols = {
        "Policy",
        "Task",
        "Avg Total Reward",
        # "Avg Reward",
        "Success Rate",
        "Trajectory Safety Rate",
        "Critical State Safety Rate",
        "Avg Steps",
        "seed",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        missing_sorted = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in {path_hint}: {missing_sorted}")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_dir = base_dir / "aggregated"
    os.makedirs(output_dir, exist_ok=True)
    cfg_name = args.base_dir.name

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    seeds = _discover_seeds(base_dir, args.seeds)
    num_seeds = len(seeds)
    if not seeds:
        raise RuntimeError(f"No seed folders found under: {base_dir}")

    frames: list[pd.DataFrame] = []
    missing_seeds: list[int] = []

    for seed in seeds:
        df = _load_seed_table(base_dir, seed)
        if df is None:
            missing_seeds.append(seed)
            continue
        _validate_columns(df, f"seed={seed}")
        frames.append(df)

    if not frames:
        raise RuntimeError(
            "No result tables were found for the requested seeds. "
            f"Checked seeds: {seeds}"
        )

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.rename(columns={'Avg Reward': 'Avg Total Reward'})  # In case of old naming

    group_cols = ["Policy", "Task"]
    metric_cols = [
        "Avg Total Reward",
        "Success Rate",
        "Trajectory Safety Rate",
        "Critical State Safety Rate",
        "Avg Steps",
    ]

    means = all_df.groupby(group_cols, as_index=False)[metric_cols].mean().round(2)
    stds = all_df.groupby(group_cols, as_index=False)[metric_cols].std(ddof=0).round(2)

    # Human-readable combined table: "mean +- std" in each metric cell.
    summary = means[group_cols].copy()
    for col in metric_cols:
        summary[col] = means[col].map(lambda x: f"{x:.2f}") + " +- " + stds[col].map(lambda x: f"{x:.2f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mean_path = output_dir / "downstream_aggregated_mean.csv"
    std_path = output_dir / "downstream_aggregated_std.csv"
    summary_path = output_dir / "downstream_aggregated_mean_std.csv"

    means.to_csv(mean_path, index=False)
    stds.to_csv(std_path, index=False)
    summary.to_csv(summary_path, index=False)

    # Per-task LaTeX tables
    env_name = 'FrozenLake'
    latex_cols_dct = {
        1: ["Critical State Safety Rate", "Trajectory Safety Rate", "Avg Total Reward"], # source taks metrics 
        2: ["Avg Total Reward", "Success Rate"], # downstream task metrics
    }
    latex_policy_order = ['Source', 'UnsafeAdapt', 'EWC', 'SafeAdapt']
    for task in sorted(all_df["Task"].unique()):
        latex_cols = latex_cols_dct.get(task, ["Critical State Safety Rate", "Trajectory Safety Rate", "Avg Total Reward"])
        task_means = means[means["Task"] == task].drop(columns=["Task"])
        task_stds = stds[stds["Task"] == task].drop(columns=["Task"])
        # task_name = 'source' if task == 1 else 'downstream'

        # Build "mean ± std" table for LaTeX
        task_summary = task_means[["Policy"]].copy()
        for col in latex_cols:
            task_summary[col] = (
                task_means[col].values.astype(str)
                + r" $\pm$ "
                + task_stds[col].values.astype(str)
            )
        task_summary = task_summary.set_index("Policy").reindex(latex_policy_order).reset_index()

        # Make sure the style is what you need:
        task_summary["Policy"] = task_summary["Policy"].replace("SafeAdapt", r"\textsc{SafeAdapt} (ours)")
        latex_path = output_dir / f"task{task}_table.tex"
        analysis_type = 'stability' if task == 1 else 'plasticity'
        label = f"tab:{env_name}_{cfg_name}_task{task}"
        caption_tail = f" {analysis_type} analysis: Task {task} metrics across {num_seeds} seeds (mean" + r" $\pm$ " + "std)"
        task_summary.to_latex(
            latex_path,
            index=False,
            escape=False,
            caption=env_name + r"(\texttt{" + cfg_name.replace('_', r'\_') + r"})" + caption_tail,
            label=label,
            column_format='lccc'
        )
        print(f"LaTeX table     : {latex_path}")

    used_seeds = sorted(all_df["seed"].unique().tolist())

    print("=" * 80)
    print("DOWNSTREAM RESULTS AGGREGATION")
    print("=" * 80)
    print(f"Base dir        : {base_dir}")
    print(f"Requested seeds : {seeds}")
    print(f"Used seeds      : {used_seeds}")
    if missing_seeds:
        print(f"Missing seeds   : {missing_seeds}")
    print(f"Rows aggregated : {len(all_df)}")
    print("-" * 80)
    print(f"Mean table      : {mean_path}")
    print(f"Std table       : {std_path}")
    print(f"Mean+-Std table : {summary_path}")


if __name__ == "__main__":
    main()
