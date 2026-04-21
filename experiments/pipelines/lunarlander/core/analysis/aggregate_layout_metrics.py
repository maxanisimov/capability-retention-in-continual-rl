"""Aggregate LunarLander per-policy metrics across seeds and export CSV + LaTeX."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import pstdev
from typing import Any

import yaml

from experiments.pipelines.lunarlander.core.orchestration.run_paths import default_outputs_root


POLICY_ORDER = [
    "source",
    "downstream_unconstrained",
    "downstream_ewc",
    "downstream_rashomon",
]

POLICY_RENAME = {
    "source": "Source",
    "downstream_unconstrained": "Unconstrained",
    "downstream_ewc": "EWC",
    "downstream_rashomon": "Rashomon",
}

RAW_TO_AGG_METRIC = {
    # Per-seed average total reward.
    "source_mean_reward": "source_total_reward",
    "downstream_mean_reward": "downstream_total_reward",
    # Per-seed failure rate.
    "source_failure_rate": "source_failure_rate",
    "downstream_failure_rate": "downstream_failure_rate",
}

PREFERRED_METRIC_ORDER = [
    "source_total_reward",
    "source_failure_rate",
    "downstream_total_reward",
    "downstream_failure_rate",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in YAML file: {path}")
    return data


def _seed_from_name(name: str) -> int | None:
    if not name.startswith("seed_"):
        return None
    suffix = name.removeprefix("seed_")
    if not suffix.isdigit():
        return None
    return int(suffix)


def _safe_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(as_float):
        return None
    return as_float


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(pstdev(values))


def _extract_task_metrics(summary: dict[str, Any]) -> dict[str, float]:
    results = summary.get("run_results")
    if not isinstance(results, dict):
        # Backward compatibility for older schemas that may store metrics at top level.
        results = summary

    out: dict[str, float] = {}
    for raw_key, agg_key in RAW_TO_AGG_METRIC.items():
        as_float = _safe_float(results.get(raw_key))
        if as_float is None:
            continue
        out[agg_key] = as_float
    return out


def _ordered_metrics(metric_names: set[str]) -> list[str]:
    ordered = [metric for metric in PREFERRED_METRIC_ORDER if metric in metric_names]
    remaining = sorted(metric for metric in metric_names if metric not in set(ordered))
    return ordered + remaining


def _format_metric_header(metric_name: str) -> str:
    return metric_name.replace("_", " ").title()


def _format_pm(mean_value: float, std_value: float, *, bold_mean: bool = False) -> str:
    def _r2(v: float) -> str:
        rounded = round(v, 2)
        if abs(rounded) < 0.005:
            rounded = 0.0
        return f"{rounded:.2f}"

    mean_str = _r2(mean_value)
    std_str = _r2(std_value)
    if bold_mean:
        return rf"$\mathbf{{{mean_str}}} \pm {std_str}$"
    return rf"${mean_str} \pm {std_str}$"


def _metric_selection_rule(metric_name: str) -> str | None:
    """Return selection rule for bolding best values in LaTeX columns."""
    if metric_name.endswith("total_reward"):
        return "max"
    if metric_name.endswith("failure_rate"):
        return "min"
    return None


def _escape_latex(text: str) -> str:
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
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def _build_latex_table_from_csv(csv_path: Path, *, task_setting: str) -> str:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")

    metric_names: list[str] = []
    for col in rows[0].keys():
        if col.endswith("_mean"):
            metric = col.removesuffix("_mean")
            if f"{metric}_std" in rows[0]:
                metric_names.append(metric)
    metric_names = _ordered_metrics(set(metric_names))
    if not metric_names:
        raise ValueError("No metric mean/std column pairs found in CSV.")

    # Preserve a stable, readable policy order.
    sorted_rows: list[dict[str, str]] = []
    by_policy = {row["policy"]: row for row in rows if "policy" in row}
    for policy in POLICY_ORDER:
        if policy in by_policy:
            sorted_rows.append(by_policy[policy])
    for row in rows:
        policy = row.get("policy", "")
        if policy not in POLICY_ORDER:
            sorted_rows.append(row)

    # For each metric column, select the best mean according to metric semantics:
    # highest total reward, lowest failure rate.
    best_mean_by_metric: dict[str, float] = {}
    for metric in metric_names:
        rule = _metric_selection_rule(metric)
        if rule is None:
            continue
        mean_key = f"{metric}_mean"
        values: list[float] = []
        for row in sorted_rows:
            raw = row.get(mean_key, "")
            if raw == "":
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        if not values:
            continue
        best_mean_by_metric[metric] = max(values) if rule == "max" else min(values)

    col_spec = "l" + "c" * len(metric_names)
    header_cells = ["Policy"] + [_escape_latex(_format_metric_header(m)) for m in metric_names]

    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]

    for row in sorted_rows:
        policy_raw = row.get("policy", "")
        policy_name = POLICY_RENAME.get(policy_raw, policy_raw)
        row_cells: list[str] = [_escape_latex(policy_name)]
        for metric in metric_names:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            mean_raw = row.get(mean_key, "")
            std_raw = row.get(std_key, "")
            if mean_raw == "" or std_raw == "":
                row_cells.append("-")
                continue
            mean_value = float(mean_raw)
            std_value = float(std_raw)
            cell = _format_pm(mean_value, std_value)
            best_mean = best_mean_by_metric.get(metric)
            if best_mean is not None and math.isclose(mean_value, best_mean, abs_tol=1e-12):
                cell = _format_pm(mean_value, std_value, bold_mean=True)
            row_cells.append(cell)
        lines.append(" & ".join(row_cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{LunarLander ({_escape_latex(task_setting)}): aggregated metrics across seeds.}}",
            rf"\label{{tab:lunarlander_{_escape_latex(task_setting)}_aggregated_metrics}}",
            r"\end{table}",
        ],
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate source/downstream run_summary metrics per policy across seeds "
            "for one LunarLander task setting, then export CSV and LaTeX."
        ),
    )
    parser.add_argument(
        "--task-setting",
        type=str,
        default=None,
        help="Task-setting name under outputs/.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default=None,
        help="Alias for --task-setting.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=default_outputs_root(),
        help="Outputs root containing <task-setting>/seed_*/<policy>/run_summary.yaml.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=None,
        help="Optional policy directory filter.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path. Default: outputs/<task-setting>/aggregate_layout_metrics.csv",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=None,
        help="Optional LaTeX path. Default: outputs/<task-setting>/aggregate_layout_metrics.tex",
    )
    args = parser.parse_args()

    task_setting = args.task_setting or args.layout
    if task_setting is None:
        raise ValueError("Provide --task-setting (or alias --layout).")

    layout_dir = args.outputs_root / task_setting
    if not layout_dir.exists():
        raise FileNotFoundError(f"Task-setting outputs directory not found: {layout_dir}")

    # policy -> seed -> metric -> value
    policy_seed_metrics: dict[str, dict[int, dict[str, float]]] = {}
    discovered_metric_names: set[str] = set()
    total_found = 0

    for seed_dir in sorted(layout_dir.glob("seed_*"), key=lambda p: p.name):
        if not seed_dir.is_dir():
            continue
        seed = _seed_from_name(seed_dir.name)
        if seed is None:
            continue

        for summary_path in sorted(seed_dir.glob("*/run_summary.yaml")):
            total_found += 1
            policy = summary_path.parent.name
            if args.policies is not None and policy not in args.policies:
                continue

            summary = _load_yaml(summary_path)
            metrics = _extract_task_metrics(summary)
            if not metrics:
                print(
                    "[skip] No usable reward/failure metrics "
                    f"(source/downstream total reward + failure rate) in {summary_path}",
                )
                continue

            policy_seed_metrics.setdefault(policy, {})
            if seed in policy_seed_metrics[policy]:
                print(
                    f"[warn] Duplicate summary for policy={policy} seed={seed}; "
                    f"overwriting with {summary_path}",
                )
            policy_seed_metrics[policy][seed] = metrics
            discovered_metric_names.update(metrics.keys())

    if total_found == 0:
        raise FileNotFoundError(f"No run_summary.yaml files found under {layout_dir}/seed_*/")
    if not policy_seed_metrics:
        raise RuntimeError("No usable task metrics were found in discovered run summaries.")

    policy_names = sorted(policy_seed_metrics.keys())
    if args.policies is not None:
        policy_names = [policy for policy in args.policies if policy in policy_seed_metrics]
    if not policy_names:
        raise RuntimeError("No policies matched the selection with usable metrics.")

    ordered_metrics = _ordered_metrics(discovered_metric_names)

    output_csv = args.output_csv or (layout_dir / "aggregate_layout_metrics.csv")
    output_tex = args.output_tex or (layout_dir / "aggregate_layout_metrics.tex")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_tex.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["task_setting", "policy", "num_seeds"]
    for metric in ordered_metrics:
        fieldnames.append(f"{metric}_mean")
        fieldnames.append(f"{metric}_std")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for policy in policy_names:
            seed_map = policy_seed_metrics[policy]
            row: dict[str, object] = {
                "task_setting": task_setting,
                "policy": policy,
                "num_seeds": len(seed_map),
            }
            for metric in ordered_metrics:
                values = [seed_map[s][metric] for s in sorted(seed_map) if metric in seed_map[s]]
                if values:
                    row[f"{metric}_mean"] = f"{_mean(values):.6f}"
                    row[f"{metric}_std"] = f"{_std(values):.6f}"
                else:
                    row[f"{metric}_mean"] = ""
                    row[f"{metric}_std"] = ""
            writer.writerow(row)

    latex = _build_latex_table_from_csv(output_csv, task_setting=task_setting)
    output_tex.write_text(latex + "\n", encoding="utf-8")

    print(f"Wrote aggregate metrics CSV: {output_csv}")
    print(f"Wrote aggregate metrics LaTeX table: {output_tex}")


if __name__ == "__main__":
    main()
