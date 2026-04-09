#!/usr/bin/env python3
"""Aggregate FrozenLake SafeAdapt vs PerfAdapt metrics across seeds.

This script expects a per-configuration directory with numeric seed folders, e.g.
`outputs/standard_4x4/<seed>/...`.

For each seed, it extracts:
- `SafeAdapt` from the standard result table locations
- `PerfAdapt` from `<seed>/downstream/results_table_perfadapt.csv`
and then aggregates across seeds.

Aggregated metrics:
- Task 1 trajectory safety
- Task 1 total reward
- Task 2 total reward
- Task 2 success rate

Examples:
  python safeadapt_vs_perfadapt.py --cfg standard_4x4
  python safeadapt_vs_perfadapt.py --base-dir outputs/diagonal_8x8 --seeds 0 1 2 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Aggregate SafeAdapt vs PerfAdapt metrics across seeds."
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="standard_4x4",
        help="Environment configuration name under outputs/ (ignored when --base-dir is set).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Directory containing numeric seed folders.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit seed list. If omitted, all numeric seed folders are used.",
    )
    parser.add_argument(
        "--safeadapt-label",
        type=str,
        default="SafeAdapt",
        help="Policy label to use for SafeAdapt rows.",
    )
    parser.add_argument(
        "--perfadapt-label",
        type=str,
        default="PerfAdapt",
        help="Policy label to use for PerfAdapt rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for aggregated output files (default: <base-dir>/aggregated).",
    )

    args = parser.parse_args()
    if args.base_dir is None:
        args.base_dir = script_dir / "outputs" / args.cfg
    if args.output_dir is None:
        args.output_dir = args.base_dir / "aggregated"
    return args


def _discover_seeds(base_dir: Path, explicit_seeds: list[int] | None) -> list[int]:
    if explicit_seeds:
        return sorted(set(explicit_seeds))

    seeds: list[int] = []
    for child in base_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            seeds.append(int(child.name))
    return sorted(seeds)


def _candidate_result_paths(seed_dir: Path) -> list[Path]:
    return [
        seed_dir / "downstream" / "results_table.csv",
        seed_dir / "results_table.csv",
        seed_dir / "downstream" / "results.table.csv",
        seed_dir / "results.table.csv",
    ]


def _load_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "Avg Total Reward" not in df.columns and "Avg Reward" in df.columns:
        df = df.rename(columns={"Avg Reward": "Avg Total Reward"})
    if "Task" in df.columns:
        df["Task"] = pd.to_numeric(df["Task"], errors="coerce")
    return df


def _extract_policy_metrics(df: pd.DataFrame, policy_label: str) -> dict | None:
    required_cols = {
        "Policy",
        "Task",
        "Trajectory Safety Rate",
        "Avg Total Reward",
        "Success Rate",
    }
    if not required_cols.issubset(df.columns):
        return None

    policy_df = df[df["Policy"] == policy_label].copy()
    if policy_df.empty:
        return None

    task1_df = policy_df[policy_df["Task"] == 1]
    task2_df = policy_df[policy_df["Task"] == 2]
    if task1_df.empty or task2_df.empty:
        return None

    task1 = task1_df.iloc[0]
    task2 = task2_df.iloc[0]

    return {
        "Task 1 Trajectory Safety": float(task1["Trajectory Safety Rate"]),
        "Task 1 Total Reward": float(task1["Avg Total Reward"]),
        "Task 2 Total Reward": float(task2["Avg Total Reward"]),
        "Task 2 Success Rate": float(task2["Success Rate"]),
    }


def _extract_for_seed(
    seed_dir: Path,
    seed: int,
    safeadapt_label: str,
    perfadapt_label: str,
) -> list[dict]:
    rows: list[dict] = []

    # SafeAdapt: read from standard per-seed result-table locations.
    safeadapt_found = None
    for path in _candidate_result_paths(seed_dir):
        df = _load_table(path)
        if df is None:
            continue
        metrics = _extract_policy_metrics(df, safeadapt_label)
        if metrics is not None:
            safeadapt_found = metrics
            break
    if safeadapt_found is not None:
        rows.append(
            {
                "seed": seed,
                "Policy": safeadapt_label,
                **safeadapt_found,
            }
        )

    # PerfAdapt: must come from <seed>/downstream/results_table_perfadapt.csv.
    perfadapt_path = seed_dir / "downstream" / "results_table_perfadapt.csv"
    perfadapt_df = _load_table(perfadapt_path)
    if perfadapt_df is not None:
        perfadapt_found = _extract_policy_metrics(perfadapt_df, perfadapt_label)
        if perfadapt_found is not None:
            rows.append(
                {
                    "seed": seed,
                    "Policy": perfadapt_label,
                    **perfadapt_found,
                }
            )

    return rows


def _format_mean_std(mean: float, std: float) -> str:
    if pd.isna(mean) or pd.isna(std):
        return "NA"
    return f"{mean:.3f} +- {std:.3f}"


def _escape_latex(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _write_latex_comparison_table(
    latex_path: Path,
    summary_df: pd.DataFrame,
    metrics: list[str],
    caption: str,
    label: str,
) -> None:
    columns = ["Policy"] + metrics
    col_format = "l" + "c" * len(metrics)

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_format}}}")
    lines.append(r"\hline")
    header = " & ".join(_escape_latex(col) for col in columns) + r" \\"
    lines.append(header)
    lines.append(r"\hline")

    for _, row in summary_df.iterrows():
        values: list[str] = []
        policy = _escape_latex(str(row["Policy"]))
        values.append(policy)
        for col in metrics:
            raw = str(row[col]).replace("+-", r"$\pm$")
            values.append(raw)
        lines.append(" & ".join(values) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    seeds = _discover_seeds(base_dir, args.seeds)
    if not seeds:
        raise RuntimeError(f"No numeric seed folders found under: {base_dir}")

    per_seed_rows: list[dict] = []
    missing_by_policy: dict[str, list[int]] = {
        args.safeadapt_label: [],
        args.perfadapt_label: [],
    }

    for seed in seeds:
        seed_dir = base_dir / str(seed)
        rows = _extract_for_seed(
            seed_dir=seed_dir,
            seed=seed,
            safeadapt_label=args.safeadapt_label,
            perfadapt_label=args.perfadapt_label,
        )

        found_labels = {row["Policy"] for row in rows}
        for label in missing_by_policy:
            if label not in found_labels:
                missing_by_policy[label].append(seed)

        per_seed_rows.extend(rows)

    if not per_seed_rows:
        raise RuntimeError(
            "No SafeAdapt/PerfAdapt rows were found in any seed tables. "
            f"Checked base dir: {base_dir}"
        )

    per_seed_df = pd.DataFrame(per_seed_rows)
    metrics = [
        "Task 1 Trajectory Safety",
        "Task 1 Total Reward",
        "Task 2 Total Reward",
        "Task 2 Success Rate",
    ]

    policy_order = [args.safeadapt_label, args.perfadapt_label]
    grouped_means = per_seed_df.groupby("Policy")[metrics].mean()
    grouped_stds = per_seed_df.groupby("Policy")[metrics].std(ddof=0)
    grouped_counts = per_seed_df.groupby("Policy")["seed"].nunique()

    mean_rows: list[dict] = []
    std_rows: list[dict] = []
    for policy in policy_order:
        count = int(grouped_counts.get(policy, 0))
        mean_row = {"Policy": policy, "Num Seeds": count}
        std_row = {"Policy": policy, "Num Seeds": count}
        for col in metrics:
            mean_row[col] = float(grouped_means.loc[policy, col]) if policy in grouped_means.index else float("nan")
            std_row[col] = float(grouped_stds.loc[policy, col]) if policy in grouped_stds.index else float("nan")
        mean_rows.append(mean_row)
        std_rows.append(std_row)

    means = pd.DataFrame(mean_rows)
    stds = pd.DataFrame(std_rows)

    summary = means[["Policy", "Num Seeds"]].copy()
    for col in metrics:
        summary[col] = [
            _format_mean_std(m, s)
            for m, s in zip(means[col].tolist(), stds[col].tolist())
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = output_dir / "safeadapt_vs_perfadapt_per_seed.csv"
    mean_path = output_dir / "safeadapt_vs_perfadapt_aggregated_mean.csv"
    std_path = output_dir / "safeadapt_vs_perfadapt_aggregated_std.csv"
    summary_path = output_dir / "safeadapt_vs_perfadapt_aggregated_mean_std.csv"
    latex_path = output_dir / "safeadapt_vs_perfadapt_table.tex"

    per_seed_df.to_csv(per_seed_path, index=False)
    means.round(4).to_csv(mean_path, index=False)
    stds.round(4).to_csv(std_path, index=False)
    summary.to_csv(summary_path, index=False)

    # LaTeX table (mean +- std for each metric)
    latex_df = summary[["Policy"] + metrics].copy()
    cfg_for_caption = _escape_latex(base_dir.name)
    caption = (
        r"FrozenLake (\texttt{" + cfg_for_caption + r"}) SafeAdapt vs PerfAdapt "
        r"across seeds (mean $\pm$ std)."
    )
    label = _escape_latex(
        f"tab:frozenlake_{base_dir.name}_safeadapt_vs_perfadapt".replace("-", "_")
    )
    _write_latex_comparison_table(
        latex_path=latex_path,
        summary_df=latex_df,
        metrics=metrics,
        caption=caption,
        label=label,
    )

    print("=" * 80)
    print("SAFEADAPT VS PERFADAPT AGGREGATION")
    print("=" * 80)
    print(f"Base dir             : {base_dir}")
    print(f"Seeds requested      : {seeds}")
    print(f"Rows aggregated      : {len(per_seed_df)}")
    print("-" * 80)
    for label in (args.safeadapt_label, args.perfadapt_label):
        used = sorted(per_seed_df.loc[per_seed_df["Policy"] == label, "seed"].unique().tolist())
        missing = missing_by_policy.get(label, [])
        print(f"{label} used seeds     : {used}")
        if missing:
            print(f"{label} missing seeds  : {missing}")
    print("-" * 80)
    print(summary.to_string(index=False))
    print("-" * 80)
    print(f"Per-seed table       : {per_seed_path}")
    print(f"Mean table           : {mean_path}")
    print(f"Std table            : {std_path}")
    print(f"Mean+-Std table      : {summary_path}")
    print(f"LaTeX table          : {latex_path}")


if __name__ == "__main__":
    main()
