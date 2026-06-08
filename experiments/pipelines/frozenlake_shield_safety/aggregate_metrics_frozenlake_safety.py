"""Aggregate FrozenLake shield safety metrics across seeds and export CSV + LaTeX."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import pstdev
import sys
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.frozenlake_shield_safety.core.config import LAYOUT_NAME
from experiments.pipelines.frozenlake_shield_safety.core.paths import (
    DOWNSTREAM_EWC_SUBDIR,
    DOWNSTREAM_LAGRANGIAN_SUBDIR,
    DOWNSTREAM_RASHOMON_SUBDIR,
    DOWNSTREAM_SAFE_LINE_SEARCH_SUBDIR,
    DOWNSTREAM_UNCONSTRAINED_SUBDIR,
    NOADAPT_POLICY_SUBDIR,
    default_outputs_root,
)


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    nested_path: tuple[str, ...]
    flat_key: str


@dataclass(frozen=True)
class MetricAggregate:
    n: int
    mean: float | None
    std: float | None


@dataclass(frozen=True)
class MethodAggregate:
    layout: str
    method: str
    method_label: str
    num_seeds: int
    metrics: dict[str, MetricAggregate]


METHOD_ORDER = [
    NOADAPT_POLICY_SUBDIR,
    DOWNSTREAM_UNCONSTRAINED_SUBDIR,
    DOWNSTREAM_EWC_SUBDIR,
    DOWNSTREAM_RASHOMON_SUBDIR,
    DOWNSTREAM_SAFE_LINE_SEARCH_SUBDIR,
    DOWNSTREAM_LAGRANGIAN_SUBDIR,
]

METHOD_LABELS = {
    NOADAPT_POLICY_SUBDIR: "NoAdapt",
    "source": "NoAdapt",
    DOWNSTREAM_UNCONSTRAINED_SUBDIR: "Unconstrained",
    DOWNSTREAM_EWC_SUBDIR: "EWC",
    DOWNSTREAM_RASHOMON_SUBDIR: "Rashomon",
    DOWNSTREAM_SAFE_LINE_SEARCH_SUBDIR: "SafeLineSearch",
    DOWNSTREAM_LAGRANGIAN_SUBDIR: "Lagrangian",
}

METRIC_SPECS = {
    "source_safety_critical_state_safety_rate": MetricSpec(
        key="source_safety_critical_state_safety_rate",
        label="Source Safety-Critical Safety",
        nested_path=("task_metrics", "source", "safety_critical_state_safety_rate"),
        flat_key="source_safety_critical_state_safety_rate",
    ),
    "source_greedy_trajectory_safety": MetricSpec(
        key="source_greedy_trajectory_safety",
        label="Source Greedy Trajectory Safety",
        nested_path=("task_metrics", "source", "greedy_trajectory_safety"),
        flat_key="source_greedy_trajectory_safety",
    ),
    "source_total_reward": MetricSpec(
        key="source_total_reward",
        label="Source Total Reward",
        nested_path=("task_metrics", "source", "total_reward"),
        flat_key="source_total_reward",
    ),
    "downstream_total_reward": MetricSpec(
        key="downstream_total_reward",
        label="Downstream Total Reward",
        nested_path=("task_metrics", "downstream", "total_reward"),
        flat_key="downstream_total_reward",
    ),
}

DEFAULT_METRICS = [
    "source_safety_critical_state_safety_rate",
    "source_greedy_trajectory_safety",
    "source_total_reward",
    "downstream_total_reward",
]

DEFAULT_CSV_NAME = "aggregate_metrics_frozenlake_shield_safety.csv"
DEFAULT_TEX_NAME = "aggregate_metrics_frozenlake_shield_safety.tex"


def _normalize_method(method: str) -> str:
    if method == "source":
        return NOADAPT_POLICY_SUBDIR
    return method


def _seed_from_dir_name(name: str) -> int | None:
    if not name.startswith("seed_"):
        return None
    suffix = name.removeprefix("seed_")
    if not suffix.isdigit():
        return None
    return int(suffix)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return data


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


def _dig(mapping: dict[str, Any], path: tuple[str, ...]) -> object | None:
    current: object = mapping
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _extract_metric(summary: dict[str, Any], spec: MetricSpec) -> float | None:
    results = summary.get("run_results")
    if not isinstance(results, dict):
        results = summary

    for container in (results, summary):
        value = _safe_float(_dig(container, spec.nested_path))
        if value is not None:
            return value
        value = _safe_float(container.get(spec.flat_key))
        if value is not None:
            return value
    return None


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(pstdev(values))


def _summary_path(seed_dir: Path, method: str) -> Path:
    canonical = seed_dir / method / "run_summary.yaml"
    if method == NOADAPT_POLICY_SUBDIR and not canonical.exists():
        legacy = seed_dir / "source" / "run_summary.yaml"
        if legacy.exists():
            return legacy
    return canonical


def aggregate_metrics(
    *,
    outputs_root: Path,
    layout: str,
    metric_keys: list[str] | None = None,
    methods: list[str] | None = None,
    seeds: set[int] | None = None,
) -> list[MethodAggregate]:
    """Read per-seed summaries and aggregate selected metrics per method."""

    selected_metrics = list(metric_keys or DEFAULT_METRICS)
    unknown_metrics = [metric for metric in selected_metrics if metric not in METRIC_SPECS]
    if unknown_metrics:
        raise ValueError(f"Unknown metric(s): {', '.join(unknown_metrics)}")

    selected_methods = list(methods or METHOD_ORDER)
    selected_methods = list(dict.fromkeys(_normalize_method(method) for method in selected_methods))

    layout_dir = outputs_root / layout
    if not layout_dir.exists():
        raise FileNotFoundError(f"Layout output directory not found: {layout_dir}")

    values_by_method: dict[str, dict[str, list[float]]] = {
        method: {metric: [] for metric in selected_metrics} for method in selected_methods
    }
    seeds_by_method: dict[str, set[int]] = {method: set() for method in selected_methods}
    found_summary = False

    seed_dirs = sorted(
        (path for path in layout_dir.glob("seed_*") if path.is_dir()),
        key=lambda path: (_seed_from_dir_name(path.name) is None, path.name),
    )
    for seed_dir in seed_dirs:
        seed = _seed_from_dir_name(seed_dir.name)
        if seed is None:
            continue
        if seeds is not None and seed not in seeds:
            continue

        for method in selected_methods:
            summary_path = _summary_path(seed_dir, method)
            if not summary_path.exists():
                continue

            found_summary = True
            summary = _load_yaml(summary_path)
            seeds_by_method[method].add(seed)
            for metric_key in selected_metrics:
                value = _extract_metric(summary, METRIC_SPECS[metric_key])
                if value is not None:
                    values_by_method[method][metric_key].append(value)

    if not found_summary:
        raise FileNotFoundError(f"No run_summary.yaml files found under {layout_dir}/seed_*/")

    rows: list[MethodAggregate] = []
    for method in selected_methods:
        if not seeds_by_method[method]:
            continue
        metric_aggs: dict[str, MetricAggregate] = {}
        for metric_key in selected_metrics:
            values = values_by_method[method][metric_key]
            metric_aggs[metric_key] = MetricAggregate(
                n=len(values),
                mean=_mean(values) if values else None,
                std=_std(values) if values else None,
            )
        rows.append(
            MethodAggregate(
                layout=layout,
                method=method,
                method_label=METHOD_LABELS.get(method, method),
                num_seeds=len(seeds_by_method[method]),
                metrics=metric_aggs,
            ),
        )

    if not rows:
        raise RuntimeError("No selected methods had usable run summaries.")
    return rows


def _format_number(value: float, precision: int) -> str:
    threshold = 0.5 * (10 ** -precision)
    if abs(value) < threshold:
        value = 0.0
    return f"{value:.{precision}f}"


def _format_pm_text(aggregate: MetricAggregate, precision: int) -> str:
    if aggregate.mean is None or aggregate.std is None:
        return ""
    return (
        f"{_format_number(aggregate.mean, precision)} "
        f"\u00b1 {_format_number(aggregate.std, precision)}"
    )


def _format_pm_latex(aggregate: MetricAggregate, precision: int) -> str:
    if aggregate.mean is None or aggregate.std is None:
        return "-"
    return (
        f"${_format_number(aggregate.mean, precision)} "
        rf"\pm {_format_number(aggregate.std, precision)}$"
    )


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


def _latex_label_fragment(text: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in out.split("_") if part)


def write_csv(
    rows: list[MethodAggregate],
    output_csv: Path,
    *,
    metric_keys: list[str],
    precision: int,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["layout", "method", "method_label", "num_seeds"] + metric_keys
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload: dict[str, object] = {
                "layout": row.layout,
                "method": row.method,
                "method_label": row.method_label,
                "num_seeds": row.num_seeds,
            }
            for metric_key in metric_keys:
                payload[metric_key] = _format_pm_text(row.metrics[metric_key], precision)
            writer.writerow(payload)


def build_latex_table(
    rows: list[MethodAggregate],
    *,
    metric_keys: list[str],
    precision: int,
    layout: str,
) -> str:
    column_spec = "lc" + ("c" * len(metric_keys))
    headers = ["Method", "N"] + [METRIC_SPECS[metric].label for metric in metric_keys]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        " & ".join(_escape_latex(header) for header in headers) + r" \\",
        r"\hline",
    ]
    for row in rows:
        cells = [_escape_latex(row.method_label), str(row.num_seeds)]
        cells.extend(_format_pm_latex(row.metrics[metric], precision) for metric in metric_keys)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            (
                r"\caption{FrozenLake shield safety "
                f"({_escape_latex(layout)}): aggregated metrics across seeds.}}"
            ),
            rf"\label{{tab:frozenlake_shield_safety_{_latex_label_fragment(layout)}_metrics}}",
            r"\end{table}",
        ],
    )
    return "\n".join(lines)


def write_latex_table(
    rows: list[MethodAggregate],
    output_tex: Path,
    *,
    metric_keys: list[str],
    precision: int,
    layout: str,
) -> None:
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(
        build_latex_table(
            rows,
            metric_keys=metric_keys,
            precision=precision,
            layout=layout,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate FrozenLake shield safety run_summary metrics across seeds and "
            "export a CSV plus a LaTeX table."
        ),
    )
    parser.add_argument(
        "--pipeline",
        "--layout",
        dest="layout",
        default=LAYOUT_NAME,
        help=f"Pipeline/layout name under outputs-root. Default: {LAYOUT_NAME}.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=default_outputs_root(),
        help="Root containing <pipeline>/seed_*/<method>/run_summary.yaml.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Optional method directory names to aggregate. "
            f"Default: {' '.join(METHOD_ORDER)}."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(METRIC_SPECS.keys()),
        default=list(DEFAULT_METRICS),
        help=(
            "Metrics to aggregate. Defaults to source safety-critical safety, "
            "source greedy trajectory safety, source total reward, and "
            "downstream total reward."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional seed filter.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal places for mean/std cells. Default: 3.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=f"CSV path. Default: outputs/<pipeline>/{DEFAULT_CSV_NAME}.",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=None,
        help=f"LaTeX path. Default: outputs/<pipeline>/{DEFAULT_TEX_NAME}.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metric_keys = list(args.metrics)
    layout = str(args.layout)
    outputs_root = Path(args.outputs_root)
    layout_dir = outputs_root / layout
    output_csv = args.output_csv or (layout_dir / DEFAULT_CSV_NAME)
    output_tex = args.output_tex or (layout_dir / DEFAULT_TEX_NAME)

    rows = aggregate_metrics(
        outputs_root=outputs_root,
        layout=layout,
        metric_keys=metric_keys,
        methods=args.methods,
        seeds=set(args.seeds) if args.seeds is not None else None,
    )
    write_csv(rows, output_csv, metric_keys=metric_keys, precision=int(args.precision))
    write_latex_table(
        rows,
        output_tex,
        metric_keys=metric_keys,
        precision=int(args.precision),
        layout=layout,
    )

    print(f"Wrote aggregate metrics CSV: {output_csv}")
    print(f"Wrote aggregate metrics LaTeX table: {output_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
