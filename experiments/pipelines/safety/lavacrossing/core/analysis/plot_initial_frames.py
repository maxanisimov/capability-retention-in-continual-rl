"""Generate initial-frame figures for LavaCrossing shield-safety task pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.pipelines.safety.lavacrossing.core.config import (
    LAYOUT_NAME,
    get_pipeline_config,
)
from experiments.pipelines.safety.lavacrossing.core.env import make_env
from experiments.pipelines.safety.lavacrossing.core.paths import artifacts_root

ICML_TWO_COLUMN_WIDTH_IN = 7.0
ICML_PAIR_HEIGHT_IN = 3.2


def default_initial_frame_dir() -> Path:
    return artifacts_root() / "figures" / "initial_frames"


def _initial_frame(
    env_map: tuple[str, ...],
    *,
    task_num: float,
    max_episode_steps: int,
    slip_prob: float,
    seed: int,
):
    env = make_env(
        env_map,
        task_num=task_num,
        max_episode_steps=max_episode_steps,
        shaped=False,
        slip_prob=slip_prob,
        render_mode="rgb_array",
    )
    try:
        env.reset(seed=seed)
        frame = env.render()
    finally:
        env.close()
    if frame is None:
        raise RuntimeError("env.render() returned None; check render_mode.")
    return frame


def make_initial_frame_figure(
    *,
    layout: str,
    seed: int = 0,
    dpi: int = 600,
    figure_width: float = ICML_TWO_COLUMN_WIDTH_IN,
    figure_height: float = ICML_PAIR_HEIGHT_IN,
) -> plt.Figure:
    cfg = get_pipeline_config(layout)
    source_frame = _initial_frame(
        cfg.source_map,
        task_num=cfg.source_task_num,
        max_episode_steps=cfg.max_episode_steps,
        slip_prob=cfg.slip_prob,
        seed=seed,
    )
    downstream_frame = _initial_frame(
        cfg.downstream_map,
        task_num=cfg.downstream_task_num,
        max_episode_steps=cfg.max_episode_steps,
        slip_prob=cfg.slip_prob,
        seed=seed,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(figure_width, figure_height),
        dpi=dpi,
        constrained_layout=True,
    )
    for ax, title, frame in zip(
        axes,
        ("Source", "Downstream"),
        (source_frame, downstream_frame),
        strict=True,
    ):
        ax.imshow(frame)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{layout} ({cfg.dynamics}, slip_prob={cfg.slip_prob:g})", fontsize=10)
    return fig


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
    fig = make_initial_frame_figure(
        layout=layout,
        seed=seed,
        dpi=dpi,
        figure_width=figure_width,
        figure_height=figure_height,
    )

    stem = basename or f"lavacrossing_shield_safety_{layout}_initial_frames"
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
            "LavaCrossing Shield Safety pipeline."
        ),
    )
    parser.add_argument(
        "--pipeline",
        "--layout",
        dest="layout",
        default=LAYOUT_NAME,
        help=f"Pipeline key to render. Default: {LAYOUT_NAME}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_initial_frame_dir(),
        help="Directory for generated initial-frame figures.",
    )
    parser.add_argument("--basename", default=None, help="Output filename stem without extension.")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], help="Output formats, e.g. pdf png svg.")
    parser.add_argument("--dpi", type=int, default=600, help="Raster output DPI.")
    parser.add_argument("--seed", type=int, default=0, help="Seed passed to env.reset() before rendering.")
    parser.add_argument("--figure-width", type=float, default=ICML_TWO_COLUMN_WIDTH_IN)
    parser.add_argument("--figure-height", type=float, default=ICML_PAIR_HEIGHT_IN)
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
