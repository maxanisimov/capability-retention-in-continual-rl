"""Aggregate average source/downstream task success rates for one pipeline."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import pstdev
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import yaml


MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None

METHOD_ORDER = [
    "noadapt",
    "downstream_unconstrained",
    "downstream_ewc",
    "downstream_rashomon",
    "downstream_rashomon_plus",
    "downstream_rashomon_expanded",
    "downstream_rashomon_family",
    "downstream_rashomon_nonconvex",
    "downstream_rashomon_union_expanded",
]

METHOD_LABELS = {
    "noadapt": "NoAdapt",
    "source": "NoAdapt",
    "downstream_unconstrained": "Unconstrained",
    "downstream_ewc": "EWC",
    "downstream_rashomon": "Rashomon",
    "downstream_rashomon_plus": "Rashomon+",
    "downstream_rashomon_expanded": "Rashomon Expanded",
    "downstream_rashomon_family": "Rashomon Family",
    "downstream_rashomon_nonconvex": "Rashomon Nonconvex",
    "downstream_rashomon_union_expanded": "Rashomon Union Expanded",
}

CSV_FIELDNAMES = [
    "pipeline",
    "method",
    "method_label",
    "num_seeds",
    "seeds",
    "avg_task_success_rate_mean",
    "avg_task_success_rate_std",
    "source_success_rate_mean",
    "source_success_rate_std",
    "downstream_success_rate_mean",
    "downstream_success_rate_std",
]


@dataclass(frozen=True)
class SeedTaskPerformance:
    pipeline: str
    seed: int
    method: str
    raw_method: str
    source_success_rate: float
    downstream_success_rate: float
    summary_path: Path

    @property
    def avg_task_success_rate(self) -> float:
        return (self.source_success_rate + self.downstream_success_rate) / 2.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _warn(message: str) -> None:
    print(f"[warn] {message}", file=sys.stderr)


def _normalize_method_name(method_name: str) -> str:
    if method_name == "source":
        return "noadapt"
    return method_name


def _method_label(method_name: str) -> str:
    if method_name in METHOD_LABELS:
        return METHOD_LABELS[method_name]
    return method_name.removeprefix("downstream_").replace("_", " ").title()


def _seed_from_dir_name(name: str) -> int | None:
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


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data)}.")
    return data


def _candidate_metric_mappings(summary: dict[str, Any]) -> list[dict[str, Any]]:
    mappings: list[dict[str, Any]] = []
    run_results = summary.get("run_results")
    if isinstance(run_results, dict):
        mappings.append(run_results)
    mappings.append(summary)
    return mappings


def _role_success_rate(summary: dict[str, Any], role: str) -> float | None:
    success_key = f"{role}_success_rate"
    failure_key = f"{role}_failure_rate"

    for mapping in _candidate_metric_mappings(summary):
        value = _safe_float(mapping.get(success_key))
        if value is not None:
            return value

    for mapping in _candidate_metric_mappings(summary):
        value = _safe_float(mapping.get(failure_key))
        if value is not None:
            return 1.0 - value

    final_eval_by_role = summary.get("final_eval_by_role")
    if isinstance(final_eval_by_role, dict):
        role_metrics = final_eval_by_role.get(role)
        if isinstance(role_metrics, dict):
            value = _safe_float(role_metrics.get("success_rate"))
            if value is not None:
                return value

    return None


def _extract_seed_performance(
    summary_path: Path,
    *,
    pipeline: str,
    seed: int,
    raw_method: str,
) -> SeedTaskPerformance | None:
    summary = _load_yaml_mapping(summary_path)
    source_success_rate = _role_success_rate(summary, "source")
    downstream_success_rate = _role_success_rate(summary, "downstream")

    if source_success_rate is None or downstream_success_rate is None:
        _warn(
            "Skipping summary with missing source/downstream success rates: "
            f"{summary_path}",
        )
        return None

    return SeedTaskPerformance(
        pipeline=pipeline,
        seed=seed,
        method=_normalize_method_name(raw_method),
        raw_method=raw_method,
        source_success_rate=source_success_rate,
        downstream_success_rate=downstream_success_rate,
        summary_path=summary_path,
    )


def resolve_pipeline_dir(
    *,
    pipeline: str | None,
    outputs_root: Path | None = None,
    pipeline_dir: Path | None = None,
) -> tuple[str, Path]:
    """Resolve the concrete pipeline directory to scan."""
    if pipeline_dir is not None:
        resolved = pipeline_dir.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Pipeline directory not found: {resolved}")
        return (pipeline or resolved.name), resolved

    if pipeline is None:
        raise ValueError("Provide --pipeline, or provide --pipeline-dir to infer the name.")

    if outputs_root is not None:
        resolved = (outputs_root.expanduser() / pipeline).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Pipeline outputs directory not found: {resolved}")
        return pipeline, resolved

    root = _repo_root() / "experiments" / "pipelines"
    search_patterns = [
        ("artifacts/runs", root.glob(f"*/artifacts/runs/{pipeline}")),
        ("outputs", root.glob(f"*/outputs/{pipeline}")),
    ]
    for location_name, candidates_iter in search_patterns:
        candidates = sorted({path.resolve() for path in candidates_iter if path.is_dir()})
        if len(candidates) == 1:
            return pipeline, candidates[0]
        if len(candidates) > 1:
            formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
            raise RuntimeError(
                f"Multiple {location_name} directories matched pipeline '{pipeline}'. "
                "Use --pipeline-dir or --outputs-root.\n"
                f"{formatted}",
            )

    raise FileNotFoundError(
        f"No outputs found for pipeline '{pipeline}' under experiments/pipelines/*/",
    )


def collect_seed_performances(
    pipeline_dir: Path,
    *,
    pipeline: str,
    methods: list[str] | None = None,
) -> list[SeedTaskPerformance]:
    selected_methods = (
        None
        if methods is None
        else {_normalize_method_name(method) for method in methods}
    )
    by_method_seed: dict[tuple[str, int], SeedTaskPerformance] = {}
    total_summaries = 0

    for seed_dir in sorted(pipeline_dir.glob("seed_*"), key=lambda path: path.name):
        if not seed_dir.is_dir():
            continue
        seed = _seed_from_dir_name(seed_dir.name)
        if seed is None:
            continue

        for summary_path in sorted(seed_dir.glob("*/run_summary.yaml")):
            total_summaries += 1
            raw_method = summary_path.parent.name
            method = _normalize_method_name(raw_method)
            if selected_methods is not None and method not in selected_methods:
                continue

            performance = _extract_seed_performance(
                summary_path,
                pipeline=pipeline,
                seed=seed,
                raw_method=raw_method,
            )
            if performance is None:
                continue

            key = (performance.method, performance.seed)
            existing = by_method_seed.get(key)
            if existing is None:
                by_method_seed[key] = performance
                continue
            if existing.raw_method == "source" and performance.raw_method == "noadapt":
                by_method_seed[key] = performance
                continue
            if existing.raw_method == "noadapt" and performance.raw_method == "source":
                continue
            _warn(
                "Duplicate summary for "
                f"method={performance.method} seed={performance.seed}; "
                f"using {performance.summary_path}",
            )
            by_method_seed[key] = performance

    if total_summaries == 0:
        raise FileNotFoundError(f"No run_summary.yaml files found under {pipeline_dir}/seed_*/")
    if not by_method_seed:
        raise RuntimeError("No usable source/downstream success-rate summaries were found.")

    return [
        by_method_seed[key]
        for key in sorted(
            by_method_seed,
            key=lambda item: (_method_sort_key(item[0]), item[1]),
        )
    ]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(pstdev(values))


def _method_sort_key(method: str) -> tuple[int, str]:
    if method in METHOD_ORDER:
        return (METHOD_ORDER.index(method), method)
    return (len(METHOD_ORDER), method)


def aggregate_performances(performances: list[SeedTaskPerformance]) -> list[dict[str, object]]:
    by_method: dict[str, list[SeedTaskPerformance]] = {}
    for performance in performances:
        by_method.setdefault(performance.method, []).append(performance)

    rows: list[dict[str, object]] = []
    for method in sorted(by_method, key=_method_sort_key):
        method_performances = sorted(by_method[method], key=lambda item: item.seed)
        avg_values = [item.avg_task_success_rate for item in method_performances]
        source_values = [item.source_success_rate for item in method_performances]
        downstream_values = [item.downstream_success_rate for item in method_performances]
        seeds = [item.seed for item in method_performances]
        pipeline = method_performances[0].pipeline

        rows.append(
            {
                "pipeline": pipeline,
                "method": method,
                "method_label": _method_label(method),
                "num_seeds": len(method_performances),
                "seeds": " ".join(str(seed) for seed in seeds),
                "avg_task_success_rate_mean": _mean(avg_values),
                "avg_task_success_rate_std": _std(avg_values),
                "source_success_rate_mean": _mean(source_values),
                "source_success_rate_std": _std(source_values),
                "downstream_success_rate_mean": _mean(downstream_values),
                "downstream_success_rate_std": _std(downstream_values),
            },
        )

    return rows


def write_aggregate_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _format_csv_value(row[key])
                    for key in CSV_FIELDNAMES
                },
            )


def _format_csv_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _load_pyplot():
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError(
            "matplotlib is required to write figures. Install the visualization "
            "dependencies, for example with `pip install -e .[viz]`.",
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _apply_neurips_plot_style(plt_module, dpi: int) -> None:
    plt_module.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        },
    )


def write_plot(rows: list[dict[str, object]], fig_base: Path, *, dpi: int) -> tuple[Path, Path]:
    if not rows:
        raise ValueError("Cannot plot empty aggregate rows.")

    plt_module = _load_pyplot()
    _apply_neurips_plot_style(plt_module, dpi)
    fig_base.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(row["method_label"]) for row in rows]
    means = [float(row["avg_task_success_rate_mean"]) for row in rows]
    stds = [float(row["avg_task_success_rate_std"]) for row in rows]
    x_positions = list(range(len(rows)))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]

    fig, ax = plt_module.subplots(figsize=(5.5, 2.6))
    ax.bar(
        x_positions,
        means,
        yerr=stds,
        color=[colors[i % len(colors)] for i in x_positions],
        edgecolor="black",
        linewidth=0.4,
        error_kw={"elinewidth": 0.7, "capthick": 0.7},
        capsize=2.5,
    )
    ax.set_ylabel("Average success rate")
    ax.set_xlabel("Method")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20 if max(map(len, labels), default=0) > 12 else 0, ha="right")
    ax.yaxis.grid(True, color="#D0D0D0", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.3)

    pdf_path = fig_base.with_suffix(".pdf")
    png_path = fig_base.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)
    plt_module.close(fig)
    return pdf_path, png_path


def _latex_escape(text: str) -> str:
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
    escaped = text
    for source, target in replacements.items():
        escaped = escaped.replace(source, target)
    return escaped


def _slugify_label(text: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in text)
    return "-".join(part for part in slug.split("-") if part) or "pipeline"


def write_caption_snippet(fig_base: Path, *, pipeline: str, caption: str | None) -> Path:
    caption_path = fig_base.with_name(f"{fig_base.name}_caption.tex")
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    if caption is None:
        caption_text = (
            "Average task success rate for "
            f"{_latex_escape(pipeline)}. Bars show the mean across seeds of the "
            "unweighted average of source-task and downstream-task success rates; "
            "error bars show one standard deviation across seeds."
        )
    else:
        caption_text = caption
    label = f"fig:avg-task-performance-{_slugify_label(pipeline)}"
    caption_path.write_text(
        "% Generated by analyse_avg_task_performance.py\n"
        f"\\caption{{{caption_text}}}\n"
        f"\\label{{{label}}}\n",
        encoding="utf-8",
    )
    return caption_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate each method's per-seed average source/downstream success "
            "rate for one experiment pipeline and export CSV plus paper figures."
        ),
    )
    parser.add_argument("--pipeline", type=str, default=None, help="Pipeline name.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=None,
        help="Root containing <pipeline>/seed_*/<method>/run_summary.yaml.",
    )
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=None,
        help="Concrete pipeline directory containing seed_*/<method>/run_summary.yaml.",
    )
    parser.add_argument("--methods", nargs="+", default=None, help="Optional method directory filter.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument(
        "--output-fig-base",
        type=Path,
        default=None,
        help="Optional figure path without extension. Writes .pdf, .png, and _caption.tex.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG output resolution.")
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Optional LaTeX caption text. Defaults to an auto-generated caption.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    pipeline, pipeline_dir = resolve_pipeline_dir(
        pipeline=args.pipeline,
        outputs_root=args.outputs_root,
        pipeline_dir=args.pipeline_dir,
    )
    output_csv = args.output_csv or (pipeline_dir / "avg_task_performance.csv")
    fig_base = args.output_fig_base or (pipeline_dir / "avg_task_performance")

    performances = collect_seed_performances(
        pipeline_dir,
        pipeline=pipeline,
        methods=args.methods,
    )
    rows = aggregate_performances(performances)
    write_aggregate_csv(rows, output_csv)
    pdf_path, png_path = write_plot(rows, fig_base, dpi=args.dpi)
    caption_path = write_caption_snippet(fig_base, pipeline=pipeline, caption=args.caption)

    print(f"Wrote average task performance CSV: {output_csv}")
    print(f"Wrote average task performance figure PDF: {pdf_path}")
    print(f"Wrote average task performance figure PNG: {png_path}")
    print(f"Wrote figure caption snippet: {caption_path}")


if __name__ == "__main__":
    main()
