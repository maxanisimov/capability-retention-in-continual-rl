#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_POLICIES = ["Source", "UnsafeAdapt", "EWC", "SafeAdapt"]


def _escape_latex_text(text: object) -> str:
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def _to_label_token(text: object) -> str:
    s = str(text).strip().lower()
    # Keep labels compact and LaTeX-friendly.
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "unknown"


def _parse_seed_tokens(seed_tokens: list[str] | None) -> list[int] | None:
    if not seed_tokens:
        return None

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
            else:
                seed = int(text)
                if seed not in seen:
                    seen.add(seed)
                    seeds.append(seed)
    return seeds


def _discover_seed_dirs(cfg_root: Path) -> list[int]:
    seeds: list[int] = []
    if not cfg_root.exists():
        return seeds
    for child in sorted(cfg_root.iterdir(), key=lambda p: p.name):
        if child.is_dir() and child.name.isdigit():
            seeds.append(int(child.name))
    return seeds


def _safe_float(x: object) -> float:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return math.nan
    return val


def _mean_std_min_max(series: pd.Series) -> tuple[float, float, float, float]:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    return float(np.mean(vals)), float(np.std(vals, ddof=0)), float(np.min(vals)), float(np.max(vals))


def _aggregate_task(
    per_seed_df: pd.DataFrame,
    task: int,
    policies: list[str],
    metric_columns: list[tuple[str, str]],
    include_min_max: bool,
) -> pd.DataFrame:
    task_df = per_seed_df[per_seed_df["task"] == task].copy()

    rows: list[dict[str, object]] = []
    for policy in policies:
        policy_df = task_df[task_df["policy"] == policy]
        row: dict[str, object] = {
            "Policy": policy,
            "Num Seeds": int(policy_df["seed"].nunique()),
        }
        for out_prefix, col in metric_columns:
            mean, std, min_v, max_v = _mean_std_min_max(policy_df[col])
            row[f"{out_prefix} Mean"] = mean
            row[f"{out_prefix} Std"] = std
            if include_min_max:
                row[f"{out_prefix} Min"] = min_v
                row[f"{out_prefix} Max"] = max_v
        rows.append(row)

    return pd.DataFrame(rows)


def _to_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    precision: int,
    caption_is_latex: bool = False,
) -> str:
    def _fmt_value(value: object) -> str:
        if value is None:
            return "NA"
        if isinstance(value, str):
            # Keep explicit math-mode strings untouched.
            if value.startswith("$") and value.endswith("$"):
                return value
            return _escape_latex_text(value)
        if isinstance(value, (np.integer, int)):
            return f"${int(value)}$"
        if isinstance(value, (np.floating, float)):
            if np.isnan(float(value)):
                return "NA"
            return f"${float(value):.{precision}f}$"
        return _escape_latex_text(value)

    columns = list(df.columns)
    col_spec = "l" + "r" * (len(columns) - 1)

    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    caption_content = caption if caption_is_latex else _escape_latex_text(caption)
    lines.append(rf"\caption{{{caption_content}}}")
    lines.append(rf"\label{{{_escape_latex_text(label)}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(_escape_latex_text(c) for c in columns) + r" \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        line = " & ".join(_fmt_value(row[c]) for c in columns) + r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _latex_view_with_mean_pm_std(
    df: pd.DataFrame,
    include_min_max: bool,
    precision: int,
) -> pd.DataFrame:
    """Build LaTeX-facing view where metric mean/std are merged as 'mean +/- std'."""
    out = pd.DataFrame()
    if "Policy" in df.columns:
        out["Policy"] = df["Policy"]

    mean_cols = [c for c in df.columns if c.endswith(" Mean")]
    for mean_col in mean_cols:
        prefix = mean_col[: -len(" Mean")]
        std_col = f"{prefix} Std"
        if std_col not in df.columns:
            continue

        merged_col = prefix
        merged_vals: list[str] = []
        for mean_v, std_v in zip(df[mean_col], df[std_col]):
            mean_num = pd.to_numeric(pd.Series([mean_v]), errors="coerce").iloc[0]
            std_num = pd.to_numeric(pd.Series([std_v]), errors="coerce").iloc[0]
            if pd.isna(mean_num) or pd.isna(std_num):
                merged_vals.append("NA")
            else:
                merged_vals.append(
                    f"${float(mean_num):.{precision}f} \\pm {float(std_num):.{precision}f}$"
                )
        out[merged_col] = merged_vals

        if include_min_max:
            min_col = f"{prefix} Min"
            max_col = f"{prefix} Max"
            if min_col in df.columns:
                out[min_col] = df[min_col]
            if max_col in df.columns:
                out[max_col] = df[max_col]

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate PoisonedApple metrics across seeds for a config. "
            "Reads downstream/results_table.csv per seed and writes CSV + LaTeX tables."
        )
    )
    parser.add_argument("--cfg", required=True, type=str, help="Configuration name, e.g. simple_6x6")
    parser.add_argument(
        "--outputs-root",
        type=str,
        default=None,
        help="Root outputs dir (default: <this_folder>/outputs)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        metavar="SEED_OR_RANGE",
        help="Optional seed subset, e.g. --seeds 0 1 2 or --seeds 0-9",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default=",".join(DEFAULT_POLICIES),
        help="Comma-separated policy names to include.",
    )
    parser.add_argument(
        "--task2-success-column",
        type=str,
        default="Avg Performance Success",
        help=(
            "Which downstream results_table column to use as Task-2 success rate. "
            "Default: 'Avg Performance Success'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where tables are written (default: <outputs-root>/<cfg>/aggregated_metrics)",
    )
    parser.add_argument(
        "--environment-name",
        type=str,
        default="PoisonedApple",
        help="Environment name used in LaTeX captions.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal precision used in LaTeX/CSV outputs.",
    )
    parser.add_argument(
        "--latex-precision",
        type=int,
        default=2,
        help="Decimal precision used in LaTeX table values (default: 2).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested seed is missing metrics files.",
    )
    parser.add_argument(
        "--include-min-max",
        action="store_true",
        help=(
            "Include per-metric Min/Max columns in aggregated tables. "
            "Default is disabled (only Mean/Std)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    outputs_root = Path(args.outputs_root) if args.outputs_root else script_dir / "outputs"
    cfg_root = outputs_root / args.cfg
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    if not policies:
        raise ValueError("No policies provided.")

    requested_seeds = _parse_seed_tokens(args.seeds)
    seeds = requested_seeds if requested_seeds is not None else _discover_seed_dirs(cfg_root)
    if not seeds:
        raise FileNotFoundError(
            f"No seed directories found for config '{args.cfg}' under {cfg_root}"
        )

    rows: list[dict[str, object]] = []
    missing_files: list[Path] = []
    for seed in seeds:
        results_path = cfg_root / str(seed) / "downstream" / "results_table.csv"
        if not results_path.exists():
            missing_files.append(results_path)
            continue

        seed_df = pd.read_csv(results_path)
        required_columns = {
            "Policy",
            "Task",
            "Trajectory Safety Rate",
            "Critical State Safety Rate",
            "Avg Total Reward",
            args.task2_success_column,
        }
        missing_cols = required_columns.difference(seed_df.columns)
        if missing_cols:
            missing_cols_str = ", ".join(sorted(missing_cols))
            raise ValueError(
                f"{results_path} is missing required columns: {missing_cols_str}"
            )

        filtered = seed_df[
            seed_df["Policy"].isin(policies) & seed_df["Task"].isin([1, 2])
        ].copy()
        for _, row in filtered.iterrows():
            rows.append(
                {
                    "seed": int(seed),
                    "policy": str(row["Policy"]),
                    "task": int(row["Task"]),
                    "trajectory_safety": _safe_float(row["Trajectory Safety Rate"]),
                    "critical_state_safety": _safe_float(row["Critical State Safety Rate"]),
                    "total_reward": _safe_float(row["Avg Total Reward"]),
                    "success_rate": _safe_float(row[args.task2_success_column]),
                }
            )

    if missing_files and args.strict:
        preview = "\n".join(str(p) for p in missing_files[:10])
        raise FileNotFoundError(
            f"Missing results files for {len(missing_files)} seed(s). First entries:\n{preview}"
        )

    if not rows:
        raise RuntimeError(
            "No rows loaded from downstream results_table.csv files. "
            "Check --cfg, --outputs-root, and --seeds."
        )

    per_seed_df = pd.DataFrame(rows)
    per_seed_df = per_seed_df.sort_values(["seed", "policy", "task"]).reset_index(drop=True)
    num_seeds_loaded = int(per_seed_df["seed"].nunique())

    task1_agg = _aggregate_task(
        per_seed_df=per_seed_df,
        task=1,
        policies=policies,
        include_min_max=args.include_min_max,
        metric_columns=[
            ("Critical State Safety", "critical_state_safety"),
            ("Trajectory Safety", "trajectory_safety"),
            ("Total Reward", "total_reward"),
        ],
    )
    task2_agg = _aggregate_task(
        per_seed_df=per_seed_df,
        task=2,
        policies=policies,
        include_min_max=args.include_min_max,
        metric_columns=[
            ("Total Reward", "total_reward"),
            ("Success Rate", "success_rate"),
        ],
    )

    output_dir = Path(args.output_dir) if args.output_dir else cfg_root / "aggregated_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_csv = output_dir / "per_seed_metrics.csv"
    task1_csv = output_dir / "task1_aggregated_metrics.csv"
    task2_csv = output_dir / "task2_aggregated_metrics.csv"
    task1_tex = output_dir / "task1_aggregated_metrics.tex"
    task2_tex = output_dir / "task2_aggregated_metrics.tex"

    per_seed_df.to_csv(per_seed_csv, index=False, float_format=f"%.{args.precision}f")
    task1_agg.to_csv(task1_csv, index=False, float_format=f"%.{args.precision}f")
    task2_agg.to_csv(task2_csv, index=False, float_format=f"%.{args.precision}f")

    env_name_tex = _escape_latex_text(args.environment_name)
    cfg_name_tex = _escape_latex_text(args.cfg)
    env_name_label = _to_label_token(args.environment_name)
    cfg_label = _to_label_token(args.cfg)

    task1_latex = _to_latex_table(
        df=_latex_view_with_mean_pm_std(
            task1_agg,
            include_min_max=args.include_min_max,
            precision=args.latex_precision,
        ),
        caption=(
            f"{env_name_tex} ({cfg_name_tex}): Task 1 aggregated metrics "
            f"(mean $\\pm$ std) across {num_seeds_loaded} seeds."
        ),
        label=f"tab:{env_name_label}_{cfg_label}_task1_aggregated",
        precision=args.latex_precision,
        caption_is_latex=True,
    )
    task2_latex = _to_latex_table(
        df=_latex_view_with_mean_pm_std(
            task2_agg,
            include_min_max=args.include_min_max,
            precision=args.latex_precision,
        ),
        caption=(
            f"{env_name_tex} ({cfg_name_tex}): Task 2 aggregated metrics "
            f"(mean $\\pm$ std) across {num_seeds_loaded} seeds."
        ),
        label=f"tab:{env_name_label}_{cfg_label}_task2_aggregated",
        precision=args.latex_precision,
        caption_is_latex=True,
    )
    task1_tex.write_text(task1_latex, encoding="utf-8")
    task2_tex.write_text(task2_latex, encoding="utf-8")

    print("=" * 90)
    print("PoisonedApple metrics aggregation complete")
    print(f"Config: {args.cfg}")
    print(f"Seeds requested: {seeds}")
    print(f"Rows loaded: {len(per_seed_df)}")
    if missing_files:
        print(f"Missing results files (skipped): {len(missing_files)}")
    print(f"Per-seed CSV  : {per_seed_csv}")
    print(f"Task-1 CSV    : {task1_csv}")
    print(f"Task-1 LaTeX  : {task1_tex}")
    print(f"Task-2 CSV    : {task2_csv}")
    print(f"Task-2 LaTeX  : {task2_tex}")
    print("=" * 90)


if __name__ == "__main__":
    main()
