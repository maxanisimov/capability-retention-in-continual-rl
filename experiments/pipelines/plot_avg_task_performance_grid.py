"""Create a multi-panel average task performance figure across pipelines."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

if __package__ in {None, ""}:
    import sys

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines import analyse_avg_task_performance as avg_perf


DEFAULT_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#E45756",
    "#72B7B2",
]

PIPELINE_FAMILY_LABELS = {
    "frozenlake": "Frozen Lake",
    "lunarlander": "Lunar Lander",
    "acrobot": "Acrobot",
    "highway": "Highway",
}

PIPELINE_TITLE_ALIASES = {
    ("lunarlander", "deterministic__default_to_sluggish_vehicle"): (
        "Lunar Lander: sluggish"
    ),
    ("lunarlander", "deterministic__default_to_underpowered_vehicle"): (
        "Lunar Lander: underpowered"
    ),
}


def _pipeline_family_label(family: str) -> str:
    return PIPELINE_FAMILY_LABELS.get(family, family.replace("_", " ").title())


def _parse_pipeline_spec(spec: str) -> tuple[str, str, str | None]:
    """Parse '[title=]family:pipeline' pipeline specifications."""
    title: str | None = None
    reference = spec
    if "=" in spec:
        title, reference = spec.split("=", 1)
        title = title.strip()
        if not title:
            raise ValueError(f"Pipeline spec has an empty title: {spec!r}")

    if ":" not in reference:
        raise ValueError(
            "Pipeline specs must have the form '[title=]family:pipeline', "
            f"got {spec!r}. Example: frozenlake:diagonal_10x10",
        )
    family, pipeline = reference.split(":", 1)
    family = family.strip()
    pipeline = pipeline.strip()
    if not family or not pipeline:
        raise ValueError(
            "Pipeline specs must include both family and pipeline name, "
            f"got {spec!r}.",
        )
    return family, pipeline, title


_PIPELINE_CATEGORIES = ("safety_retention", "trajectory_retention", "envs")


def _resolve_family_root(family: str, pipelines_root: Path) -> Path:
    if "/" in family:
        category, _, name = family.partition("/")
        return pipelines_root / category / name
    matches = [
        pipelines_root / category / family
        for category in _PIPELINE_CATEGORIES
        if (pipelines_root / category / family).exists()
    ]
    if not matches:
        raise FileNotFoundError(
            f"Pipeline family directory not found for '{family}' under "
            f"{', '.join(str(pipelines_root / c) for c in _PIPELINE_CATEGORIES)}.",
        )
    if len(matches) > 1:
        formatted = "\n".join(f"  - {m.relative_to(pipelines_root)}" for m in matches)
        raise RuntimeError(
            f"Pipeline family '{family}' is ambiguous across categories:\n{formatted}\n"
            "Use '<category>/<name>' (e.g. 'safety_retention/frozenlake') to disambiguate.",
        )
    return matches[0]


def _resolve_pipeline_spec(spec: str) -> tuple[str, Path]:
    family, pipeline, title = _parse_pipeline_spec(spec)
    pipelines_root = avg_perf._repo_root() / "experiments" / "pipelines"
    family_root = _resolve_family_root(family, pipelines_root)

    artifacts_dir = family_root / "artifacts" / "runs" / pipeline
    if artifacts_dir.is_dir():
        pipeline_dir = artifacts_dir.resolve()
    else:
        outputs_dir = family_root / "outputs" / pipeline
        if not outputs_dir.is_dir():
            raise FileNotFoundError(
                "Pipeline outputs directory not found. Looked for:\n"
                f"  - {artifacts_dir}\n"
                f"  - {outputs_dir}",
            )
        pipeline_dir = outputs_dir.resolve()

    display_name = (
        title
        or PIPELINE_TITLE_ALIASES.get((family, pipeline))
        or f"{_pipeline_family_label(family)}: {_title_from_pipeline(pipeline)}"
    )
    return display_name, pipeline_dir


def _resolve_pipeline_inputs(
    *,
    pipelines: list[str] | None,
    outputs_root: Path | None,
    pipeline_dirs: list[Path] | None,
    pipeline_specs: list[str] | None = None,
) -> list[tuple[str, Path]]:
    if pipeline_specs is not None:
        if (
            pipelines is not None
            or outputs_root is not None
            or pipeline_dirs is not None
        ):
            raise ValueError(
                "--pipeline-specs cannot be combined with --pipelines, "
                "--outputs-root, or --pipeline-dirs.",
            )
        return [_resolve_pipeline_spec(spec) for spec in pipeline_specs]

    if pipeline_dirs is None:
        if not pipelines:
            raise ValueError(
                "Provide --pipelines, --pipeline-dirs, or --pipeline-specs.",
            )
        return [
            avg_perf.resolve_pipeline_dir(
                pipeline=pipeline,
                outputs_root=outputs_root,
                pipeline_dir=None,
            )
            for pipeline in pipelines
        ]

    if outputs_root is not None:
        raise ValueError("--outputs-root cannot be combined with --pipeline-dirs.")
    if pipelines is not None and len(pipelines) != len(pipeline_dirs):
        raise ValueError("--pipelines and --pipeline-dirs must have the same length.")

    resolved: list[tuple[str, Path]] = []
    for index, pipeline_dir in enumerate(pipeline_dirs):
        pipeline = None if pipelines is None else pipelines[index]
        resolved.append(
            avg_perf.resolve_pipeline_dir(
                pipeline=pipeline,
                outputs_root=None,
                pipeline_dir=pipeline_dir,
            ),
        )
    return resolved


def _title_from_pipeline(pipeline: str) -> str:
    return pipeline.replace("__", " -> ").replace("_", " ")


def _collect_pipeline_rows(
    pipeline_inputs: list[tuple[str, Path]],
    *,
    methods: list[str] | None,
) -> list[tuple[str, list[dict[str, object]]]]:
    panel_rows: list[tuple[str, list[dict[str, object]]]] = []
    for pipeline, pipeline_dir in pipeline_inputs:
        performances = avg_perf.collect_seed_performances(
            pipeline_dir,
            pipeline=pipeline,
            methods=methods,
        )
        panel_rows.append((pipeline, avg_perf.aggregate_performances(performances)))
    return panel_rows


def _ordered_method_labels(
    panel_rows: list[tuple[str, list[dict[str, object]]]],
) -> dict[str, str]:
    methods = {
        str(row["method"])
        for _, rows in panel_rows
        for row in rows
    }
    return {
        method: avg_perf.METHOD_LABELS.get(method, avg_perf._method_label(method))
        for method in sorted(methods, key=avg_perf._method_sort_key)
    }


def write_grid_plot(
    panel_rows: list[tuple[str, list[dict[str, object]]]],
    fig_base: Path,
    *,
    ncols: int,
    dpi: int,
    titles: list[str] | None = None,
) -> tuple[Path, Path]:
    if not panel_rows:
        raise ValueError("No pipeline rows to plot.")
    if ncols <= 0:
        raise ValueError(f"ncols must be positive, got {ncols}.")
    if titles is not None and len(titles) != len(panel_rows):
        raise ValueError("--titles must have one title per pipeline.")

    plt_module = avg_perf._load_pyplot()
    avg_perf._apply_neurips_plot_style(plt_module, dpi)

    fig_base.parent.mkdir(parents=True, exist_ok=True)
    n_panels = len(panel_rows)
    ncols = min(ncols, n_panels)
    nrows = math.ceil(n_panels / ncols)
    width = 5.5
    height = max(1.65 * nrows, 2.4)
    fig, axes = plt_module.subplots(
        nrows,
        ncols,
        figsize=(width, height),
        sharey=True,
        squeeze=False,
    )

    method_labels = _ordered_method_labels(panel_rows)
    method_to_color = {
        method: DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
        for index, method in enumerate(method_labels)
    }

    for panel_index, (pipeline, rows) in enumerate(panel_rows):
        ax = axes[panel_index // ncols][panel_index % ncols]
        methods = [str(row["method"]) for row in rows]
        labels = [str(row["method_label"]) for row in rows]
        means = [float(row["avg_task_success_rate_mean"]) for row in rows]
        stds = [float(row["avg_task_success_rate_std"]) for row in rows]
        x_positions = list(range(len(rows)))

        ax.bar(
            x_positions,
            means,
            yerr=stds,
            color=[method_to_color[method] for method in methods],
            edgecolor="black",
            linewidth=0.4,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
            capsize=2.0,
        )
        title = (
            titles[panel_index]
            if titles is not None
            else _title_from_pipeline(pipeline)
        )
        ax.set_title(title)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.yaxis.grid(True, color="#D0D0D0", linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for empty_index in range(n_panels, nrows * ncols):
        axes[empty_index // ncols][empty_index % ncols].axis("off")

    fig.supxlabel("Method")
    fig.tight_layout(pad=0.35, h_pad=0.8, w_pad=0.55)

    pdf_path = fig_base.with_suffix(".pdf")
    png_path = fig_base.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)
    plt_module.close(fig)
    return pdf_path, png_path


def write_caption_snippet(
    fig_base: Path,
    *,
    pipelines: list[str],
    caption: str | None,
) -> Path:
    caption_path = fig_base.with_name(f"{fig_base.name}_caption.tex")
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    if caption is None:
        caption_text = (
            "Average task success rate across transfer pipelines. Each panel "
            "corresponds to one pipeline. Bars show the mean across seeds of the "
            "unweighted average of source-task and downstream-task success rates; "
            "error bars show one standard deviation across seeds."
        )
    else:
        caption_text = caption
    label = "fig:avg-task-performance-grid"
    caption_path.write_text(
        "% Generated by plot_avg_task_performance_grid.py\n"
        f"\\caption{{{caption_text}}}\n"
        f"\\label{{{label}}}\n",
        encoding="utf-8",
    )
    return caption_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create one multi-panel bar-chart figure where each subplot is a "
            "pipeline's average source/downstream task success rate by method."
        ),
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=None,
        help=(
            "Pipeline names. Required unless --pipeline-dirs or "
            "--pipeline-specs is provided."
        ),
    )
    parser.add_argument(
        "--pipeline-specs",
        nargs="+",
        default=None,
        help=(
            "Family-qualified pipeline specs of the form '[title=]family:pipeline'. "
            "Examples: frozenlake:diagonal_10x10 "
            "lunarlander:deterministic__default_to_sluggish_vehicle"
        ),
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=None,
        help="Shared root containing <pipeline>/seed_*/<method>/run_summary.yaml.",
    )
    parser.add_argument(
        "--pipeline-dirs",
        nargs="+",
        type=Path,
        default=None,
        help="Concrete pipeline directories. May be paired with --pipelines for names.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional method filter.",
    )
    parser.add_argument(
        "--titles",
        nargs="+",
        default=None,
        help="Optional subplot titles, one per pipeline.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of subplot columns. Default: 3.",
    )
    parser.add_argument(
        "--output-fig-base",
        type=Path,
        default=Path("avg_task_performance_grid"),
        help="Figure path without extension. Writes .pdf, .png, and _caption.tex.",
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
    pipeline_inputs = _resolve_pipeline_inputs(
        pipelines=args.pipelines,
        outputs_root=args.outputs_root,
        pipeline_dirs=args.pipeline_dirs,
        pipeline_specs=args.pipeline_specs,
    )
    panel_rows = _collect_pipeline_rows(pipeline_inputs, methods=args.methods)
    pdf_path, png_path = write_grid_plot(
        panel_rows,
        args.output_fig_base,
        ncols=args.ncols,
        dpi=args.dpi,
        titles=args.titles,
    )
    caption_path = write_caption_snippet(
        args.output_fig_base,
        pipelines=[pipeline for pipeline, _ in pipeline_inputs],
        caption=args.caption,
    )

    print(f"Wrote average task performance grid PDF: {pdf_path}")
    print(f"Wrote average task performance grid PNG: {png_path}")
    print(f"Wrote figure caption snippet: {caption_path}")


if __name__ == "__main__":
    main()
