"""Aggregate LunarLander adaptation metrics relative to NoAdapt across seeds."""

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
    "downstream_unconstrained",
    "downstream_ewc",
    "downstream_rashomon",
]

POLICY_RENAME = {
    "noadapt": "NoAdapt",
    "source": "NoAdapt",
    "downstream_unconstrained": "Unconstrained",
    "downstream_ewc": "EWC",
    "downstream_rashomon": "Rashomon",
}

RAW_TO_AGG_METRIC = {
    "total_reward": {
        # Per-seed average total reward.
        "source_mean_reward": "source_total_reward",
        "downstream_mean_reward": "downstream_total_reward",
    },
    "success_rate": {
        # Per-seed success rate.
        "source_success_rate": "source_success_rate",
        "downstream_success_rate": "downstream_success_rate",
    },
    "failure_rate": {
        # Per-seed failure rate.
        "source_failure_rate": "source_failure_rate",
        "downstream_failure_rate": "downstream_failure_rate",
    },
}

DEFAULT_METRIC_GROUPS = ["total_reward", "success_rate"]

PREFERRED_METRIC_ORDER = [
    "relative_source_total_reward",
    "relative_source_success_rate",
    "relative_downstream_total_reward",
    "relative_downstream_success_rate",
    "relative_source_failure_rate",
    "relative_downstream_failure_rate",
]


def _normalize_policy_dir_name(policy_name: str) -> str:
    if policy_name == "source":
        return "noadapt"
    return policy_name


def _is_baseline_policy(policy_name: str) -> bool:
    return _normalize_policy_dir_name(policy_name) == "noadapt"


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


def _extract_task_metrics(
    summary: dict[str, Any],
    *,
    raw_to_agg_metric: dict[str, str],
) -> dict[str, float]:
    results = summary.get("run_results")
    if not isinstance(results, dict):
        # Backward compatibility for older schemas that may store metrics at top level.
        results = summary

    out: dict[str, float] = {}
    for raw_key, agg_key in raw_to_agg_metric.items():
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
    if metric_name.endswith("success_rate"):
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
    # highest total reward/success rate, lowest failure rate.
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
            rf"\caption{{LunarLander ({_escape_latex(task_setting)}): adaptation metrics relative to NoAdapt across seeds.}}",
            rf"\label{{tab:lunarlander_{_escape_latex(task_setting)}_relative_metrics}}",
            r"\end{table}",
        ],
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate adaptation policy metrics relative to NoAdapt per seed "
            "for one LunarLander pipeline, then export CSV and LaTeX."
        ),
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        dest="task_setting",
        default=None,
        help="Pipeline name under outputs/.",
    )
    parser.add_argument("--task-setting", type=str, dest="task_setting", help=argparse.SUPPRESS)
    parser.add_argument(
        "--layout",
        type=str,
        default=None,
        help="Alias for --pipeline.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=default_outputs_root(),
        help="Outputs root containing <pipeline>/seed_*/<policy>/run_summary.yaml.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=None,
        help="Optional adaptation policy directory filter.",
    )
    parser.add_argument(
        "--metric-groups",
        nargs="+",
        choices=sorted(RAW_TO_AGG_METRIC.keys()),
        default=list(DEFAULT_METRIC_GROUPS),
        help=(
            "Metric groups to aggregate relative to NoAdapt. "
            "Default: total_reward success_rate."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path. Default: outputs/<pipeline>/aggregate_layout_relative_metrics.csv",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=None,
        help="Optional LaTeX path. Default: outputs/<pipeline>/aggregate_layout_relative_metrics.tex",
    )
    args = parser.parse_args()

    task_setting = args.task_setting or args.layout
    if task_setting is None:
        raise ValueError("Provide --pipeline (or alias --layout).")

    layout_dir = args.outputs_root / task_setting
    if not layout_dir.exists():
        raise FileNotFoundError(f"Pipeline outputs directory not found: {layout_dir}")

    selected_policies = (
        None
        if args.policies is None
        else list(dict.fromkeys(_normalize_policy_dir_name(policy) for policy in args.policies))
    )
    if selected_policies is not None:
        selected_policies = [policy for policy in selected_policies if not _is_baseline_policy(policy)]
        if not selected_policies:
            raise ValueError("--policies must include at least one adaptation policy (not noadapt/source).")

    selected_metric_groups = list(args.metric_groups)
    raw_to_agg_metric: dict[str, str] = {}
    for metric_group in selected_metric_groups:
        raw_to_agg_metric.update(RAW_TO_AGG_METRIC[metric_group])
    base_metric_names = set(raw_to_agg_metric.values())
    selected_relative_metric_names = {f"relative_{metric_name}" for metric_name in base_metric_names}

    # Baseline (NoAdapt) per seed: seed -> metric -> value
    baseline_seed_metrics: dict[int, dict[str, float]] = {}
    # Raw adaptation per-seed metrics before relative subtraction:
    # policy -> seed -> metric -> value
    policy_seed_raw_metrics: dict[str, dict[int, dict[str, float]]] = {}
    total_found = 0

    for seed_dir in sorted(layout_dir.glob("seed_*"), key=lambda p: p.name):
        if not seed_dir.is_dir():
            continue
        seed = _seed_from_name(seed_dir.name)
        if seed is None:
            continue

        for summary_path in sorted(seed_dir.glob("*/run_summary.yaml")):
            total_found += 1
            raw_policy = summary_path.parent.name
            policy = _normalize_policy_dir_name(raw_policy)

            summary = _load_yaml(summary_path)
            metrics = _extract_task_metrics(summary, raw_to_agg_metric=raw_to_agg_metric)
            if not metrics:
                print(
                    "[skip] No usable selected metrics "
                    f"(metric_groups={selected_metric_groups}) in {summary_path}",
                )
                continue

            if _is_baseline_policy(policy):
                if seed in baseline_seed_metrics:
                    print(
                        f"[warn] Duplicate NoAdapt summary for seed={seed}; "
                        f"overwriting with {summary_path}",
                    )
                baseline_seed_metrics[seed] = metrics
                continue

            if selected_policies is not None and policy not in selected_policies:
                continue

            policy_seed_raw_metrics.setdefault(policy, {})
            if seed in policy_seed_raw_metrics[policy]:
                print(
                    f"[warn] Duplicate summary for policy={policy} seed={seed}; "
                    f"overwriting with {summary_path}",
                )
            policy_seed_raw_metrics[policy][seed] = metrics

    if total_found == 0:
        raise FileNotFoundError(f"No run_summary.yaml files found under {layout_dir}/seed_*/")
    if not baseline_seed_metrics:
        raise RuntimeError(
            "No NoAdapt/source run summaries with usable metrics were found; "
            "cannot compute relative adaptation metrics.",
        )
    if not policy_seed_raw_metrics:
        raise RuntimeError("No adaptation policies with usable metrics were found.")

    # Relative metrics per seed:
    # policy -> seed -> relative_metric -> delta
    policy_seed_relative_metrics: dict[str, dict[int, dict[str, float]]] = {}

    for policy, seed_map in sorted(policy_seed_raw_metrics.items()):
        for seed, policy_metrics in sorted(seed_map.items()):
            baseline_metrics = baseline_seed_metrics.get(seed)
            if baseline_metrics is None:
                print(
                    f"[warn] Missing NoAdapt/source baseline for seed={seed}; "
                    f"skipping policy={policy} seed={seed}",
                )
                continue

            relative_metrics: dict[str, float] = {}
            for metric_name in sorted(base_metric_names):
                policy_value = policy_metrics.get(metric_name)
                baseline_value = baseline_metrics.get(metric_name)
                if policy_value is None or baseline_value is None:
                    print(
                        "[warn] Missing metric pair for relative computation: "
                        f"policy={policy}, seed={seed}, metric={metric_name}, "
                        f"policy_has_metric={policy_value is not None}, "
                        f"baseline_has_metric={baseline_value is not None}",
                    )
                    continue
                relative_name = f"relative_{metric_name}"
                relative_metrics[relative_name] = policy_value - baseline_value

            if not relative_metrics:
                print(
                    f"[warn] No relative metrics computed for policy={policy} seed={seed}; "
                    "skipping this seed.",
                )
                continue

            policy_seed_relative_metrics.setdefault(policy, {})
            policy_seed_relative_metrics[policy][seed] = relative_metrics

    if not policy_seed_relative_metrics:
        raise RuntimeError("No relative policy metrics were computed.")

    policy_names = sorted(policy_seed_relative_metrics.keys())
    if selected_policies is not None:
        policy_names = [policy for policy in selected_policies if policy in policy_seed_relative_metrics]
    else:
        ordered: list[str] = []
        for policy in POLICY_ORDER:
            if policy in policy_seed_relative_metrics:
                ordered.append(policy)
        for policy in policy_names:
            if policy not in set(ordered):
                ordered.append(policy)
        policy_names = ordered

    if not policy_names:
        raise RuntimeError("No policies matched the selection with usable relative metrics.")

    ordered_metrics = _ordered_metrics(selected_relative_metric_names)

    output_csv = args.output_csv or (layout_dir / "aggregate_layout_relative_metrics.csv")
    output_tex = args.output_tex or (layout_dir / "aggregate_layout_relative_metrics.tex")
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
            seed_map = policy_seed_relative_metrics[policy]
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

    print(f"Wrote relative aggregate metrics CSV: {output_csv}")
    print(f"Wrote relative aggregate metrics LaTeX table: {output_tex}")


if __name__ == "__main__":
    main()
