"""Generate initial-frame figures for FrozenLake slippery shield-safety task pairs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.pipelines.frozenlake.core.analysis.plot_initial_frames import (
    ICML_PAIR_HEIGHT_IN,
    ICML_TWO_COLUMN_WIDTH_IN,
    make_initial_frame_figure_from_tasks,
)

import matplotlib.pyplot as plt

from experiments.pipelines.frozenlake_slippery_shield_safety.core.config import (
    get_pipeline_config,
)
from experiments.pipelines.frozenlake_slippery_shield_safety.core.paths import (
    artifacts_root,
)


def default_initial_frame_dir() -> Path:
    return artifacts_root() / "figures" / "initial_frames"


def _task_cfg_from_pipeline_map(
    env_map: tuple[str, ...],
    *,
    max_episode_steps: int,
    is_slippery: bool,
    success_rate: float,
) -> dict[str, Any]:
    return {
        "env_map": [str(row) for row in env_map],
        "is_slippery": bool(is_slippery),
        "success_rate": float(success_rate),
        "max_episode_steps": int(max_episode_steps),
    }


def source_downstream_task_cfgs(layout: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = get_pipeline_config(layout)
    source_cfg = _task_cfg_from_pipeline_map(
        cfg.source_map,
        max_episode_steps=cfg.max_episode_steps,
        is_slippery=cfg.is_slippery,
        success_rate=cfg.success_rate,
    )
    downstream_cfg = _task_cfg_from_pipeline_map(
        cfg.downstream_map,
        max_episode_steps=cfg.max_episode_steps,
        is_slippery=cfg.is_slippery,
        success_rate=cfg.success_rate,
    )
    return source_cfg, downstream_cfg


def save_initial_frame_figure(
    *,
    layout: str,
    output_dir: Path,
    basename: str | None = None,
    formats: list[str] | None = None,
    dpi: int = 600,
    figure_width: float = ICML_TWO_COLUMN_WIDTH_IN,
    figure_height: float = ICML_PAIR_HEIGHT_IN,
    seed: int = 0,
) -> list[Path]:
    if formats is None:
        formats = ["pdf", "png"]

    output_dir.mkdir(parents=True, exist_ok=True)
    source_cfg, downstream_cfg = source_downstream_task_cfgs(layout)
    fig = make_initial_frame_figure_from_tasks(
        source_cfg,
        downstream_cfg,
        dpi=dpi,
        figure_width=figure_width,
        figure_height=figure_height,
        seed=seed,
    )

    stem = basename or f"frozenlake_slippery_shield_safety_{layout}_initial_frames"
    saved_paths: list[Path] = []
    try:
        for fmt in formats:
            fmt_clean = fmt.lower().lstrip(".")
            out_path = output_dir / f"{stem}.{fmt_clean}"
            fig.savefig(out_path, dpi=dpi)
            saved_paths.append(out_path)
    finally:
        plt.close(fig)

    return saved_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a source/downstream initial-frame figure for one "
            "FrozenLake Slippery Shield Safety pipeline layout."
        ),
    )
    parser.add_argument(
        "--pipeline",
        "--layout",
        dest="layout",
        default="diagonal_4x4",
        help="Pipeline layout to render, e.g. diagonal_6x6.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_initial_frame_dir(),
        help="Directory for generated initial-frame figures.",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Output filename stem without extension.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Output formats, e.g. pdf png svg.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Raster output DPI.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed passed to env.reset() before rendering the initial frame.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=ICML_TWO_COLUMN_WIDTH_IN,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=ICML_PAIR_HEIGHT_IN,
        help="Figure height in inches.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    saved_paths = save_initial_frame_figure(
        layout=args.layout,
        output_dir=args.output_dir,
        basename=args.basename,
        formats=args.formats,
        dpi=args.dpi,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        seed=args.seed,
    )
    for path in saved_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
