"""Generate paper-ready initial-frame figures for FrozenLake task pairs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

_MPLCONFIGDIR = Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from experiments.pipelines.behaviour_retention.frozenlake.core.env.task_loading import (  # noqa: E402
    load_task_settings,
)
from experiments.pipelines.behaviour_retention.frozenlake.core.orchestration.run_paths import (
    artifacts_root,
    default_task_pipelines_file,
)  # noqa: E402

ICML_TWO_COLUMN_WIDTH_IN = 6.75
ICML_PAIR_HEIGHT_IN = 3.35


def _apply_icml_style(dpi: int) -> None:
    """Use compact publication defaults that work without a LaTeX install."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
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
            "axes.linewidth": 0.5,
            "lines.linewidth": 1.0,
        },
    )


def _validate_map(env_map: list[str], *, name: str) -> None:
    if not env_map:
        raise ValueError(f"{name} map is empty.")
    ncols = len(env_map[0])
    if ncols == 0:
        raise ValueError(f"{name} map has an empty first row.")
    for idx, row in enumerate(env_map):
        if len(row) != ncols:
            raise ValueError(
                f"{name} map row {idx} has length {len(row)}; expected {ncols}.",
            )
        unknown = sorted(set(row) - {"F", "H", "S", "G"})
        if unknown:
            raise ValueError(
                f"{name} map row {idx} contains unsupported cell codes: {unknown}.",
            )


def _task_cfg_from_map(
    env_map: list[str],
    *,
    max_episode_steps: int | None = None,
    is_slippery: bool = False,
) -> dict[str, Any]:
    _validate_map(env_map, name="FrozenLake")
    return {
        "env_map": env_map,
        "is_slippery": is_slippery,
        "max_episode_steps": max_episode_steps or 4 * len(env_map),
    }


def _make_rgb_env(task_cfg: dict[str, Any]) -> gym.Env:
    env_map = [str(row) for row in task_cfg["env_map"]]
    task_name = str(task_cfg.get("_resolved_definition_name", "FrozenLake"))
    _validate_map(env_map, name=task_name)
    return gym.make(
        "FrozenLake-v1",
        desc=env_map,
        is_slippery=bool(task_cfg.get("is_slippery", False)),
        max_episode_steps=int(task_cfg["max_episode_steps"]),
        render_mode="rgb_array",
    )


def render_initial_frame(task_cfg: dict[str, Any], *, seed: int = 0) -> np.ndarray:
    """Render the first FrozenLake frame using Gymnasium's RGB-array mode."""
    env = _make_rgb_env(task_cfg)
    try:
        env.reset(seed=seed)
        frame = env.render()
    finally:
        env.close()

    if frame is None:
        raise RuntimeError("env.render() returned None; expected render_mode='rgb_array'.")

    frame_array = np.asarray(frame)
    if frame_array.ndim != 3 or frame_array.shape[2] not in {3, 4}:
        raise ValueError(
            "Expected an RGB/RGBA frame with shape (height, width, channels); "
            f"got {frame_array.shape}.",
        )
    return frame_array[:, :, :3]


def _draw_frame(ax: plt.Axes, frame: np.ndarray, *, title: str) -> None:
    ax.imshow(frame, interpolation="nearest")

    ax.set_title(title, pad=3.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def load_source_downstream_tasks(
    *,
    task_settings_file: Path,
    layout: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_cfg = load_task_settings(task_settings_file, layout, "source")
    downstream_cfg = load_task_settings(task_settings_file, layout, "downstream")
    return source_cfg, downstream_cfg


def load_source_downstream_maps(
    *,
    task_settings_file: Path,
    layout: str,
) -> tuple[list[str], list[str]]:
    source_cfg, downstream_cfg = load_source_downstream_tasks(
        task_settings_file=task_settings_file,
        layout=layout,
    )
    return source_cfg["env_map"], downstream_cfg["env_map"]


def make_initial_frame_figure(
    source_map: list[str],
    downstream_map: list[str],
    *,
    dpi: int = 600,
    figure_width: float = ICML_TWO_COLUMN_WIDTH_IN,
    figure_height: float = ICML_PAIR_HEIGHT_IN,
    seed: int = 0,
    include_legend: bool | None = None,
) -> plt.Figure:
    del include_legend
    return make_initial_frame_figure_from_tasks(
        _task_cfg_from_map(source_map),
        _task_cfg_from_map(downstream_map),
        dpi=dpi,
        figure_width=figure_width,
        figure_height=figure_height,
        seed=seed,
    )


def make_initial_frame_figure_from_tasks(
    source_cfg: dict[str, Any],
    downstream_cfg: dict[str, Any],
    *,
    dpi: int = 600,
    figure_width: float = ICML_TWO_COLUMN_WIDTH_IN,
    figure_height: float = ICML_PAIR_HEIGHT_IN,
    seed: int = 0,
    include_legend: bool | None = None,
) -> plt.Figure:
    del include_legend
    _apply_icml_style(dpi)
    source_frame = render_initial_frame(source_cfg, seed=seed)
    downstream_frame = render_initial_frame(downstream_cfg, seed=seed)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(figure_width, figure_height),
        constrained_layout=False,
    )

    _draw_frame(axes[0], source_frame, title="Source task")
    _draw_frame(axes[1], downstream_frame, title="Downstream task")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.91, wspace=0.04)

    return fig


def save_initial_frame_figure(
    *,
    task_settings_file: Path,
    layout: str,
    output_dir: Path,
    basename: str | None = None,
    formats: list[str] | None = None,
    dpi: int = 600,
    figure_width: float = ICML_TWO_COLUMN_WIDTH_IN,
    figure_height: float = ICML_PAIR_HEIGHT_IN,
    seed: int = 0,
    include_legend: bool | None = None,
) -> list[Path]:
    del include_legend
    if formats is None:
        formats = ["pdf", "png"]
    output_dir.mkdir(parents=True, exist_ok=True)

    source_cfg, downstream_cfg = load_source_downstream_tasks(
        task_settings_file=task_settings_file,
        layout=layout,
    )
    fig = make_initial_frame_figure_from_tasks(
        source_cfg,
        downstream_cfg,
        dpi=dpi,
        figure_width=figure_width,
        figure_height=figure_height,
        seed=seed,
    )

    stem = basename or f"frozenlake_{layout}_initial_frames"
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
            "Generate an ICML-ready figure of FrozenLake source/downstream "
            "initial frames."
        ),
    )
    parser.add_argument(
        "--pipeline",
        "--layout",
        dest="layout",
        default="diagonal_10x10",
    )
    parser.add_argument(
        "--task-settings-file",
        type=Path,
        default=default_task_pipelines_file(),
        help="FrozenLake task pipeline settings file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=artifacts_root() / "figures",
        help="Directory for generated figure files.",
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
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help=(
            "Accepted for backward compatibility; RGB-array FrozenLake frames "
            "already include the rendered visual labels."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    saved_paths = save_initial_frame_figure(
        task_settings_file=args.task_settings_file,
        layout=args.layout,
        output_dir=args.output_dir,
        basename=args.basename,
        formats=args.formats,
        dpi=args.dpi,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        seed=args.seed,
        include_legend=not args.no_legend,
    )
    for path in saved_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
