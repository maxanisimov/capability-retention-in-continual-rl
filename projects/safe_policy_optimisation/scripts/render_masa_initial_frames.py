#!/usr/bin/env python3
"""Render each MASA environment's initial frame (env.render() in rgb_array mode).

Environments are built with the exact env_id and env_kwargs of the canonical
PSPO runs and reset with seed 0, before any action. Per environment this
writes the native frame as PNG and a margin-free PDF at --height inches, so a
plain \\includegraphics gives a small paper-ready figure. It also writes one row
of all six at the AAMAS text width, and the frame metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ccl-masa-frames-matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import gymnasium as gym  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import projects.safe_crl.utils.masa_tabular_envs  # noqa: E402,F401  registers Custom* ids
from _aamas_reward_safety import COLUMN_WIDTH_IN, FULL_WIDTH_IN, apply_aamas_style  # noqa: E402
from plot_pspo_variant_reward_time_aamas import CONTROLS, DEFAULT_RUNS, ENVIRONMENTS  # noqa: E402

DEFAULT_OUTPUT = REPO / "projects/safe_policy_optimisation/figures/masa_initial_frames"
RESET_SEED = 0
A4_HEIGHT_IN = 297 / 25.4
# Single-column AAMAS grids: output stem -> (rows of environments, printed
# height, layout description). The 3x2 grid is three quarters of a quarter of an
# A4 page tall; the 2x3 grid keeps the same pairs in columns and is 15% of it.
GRIDS = {
    "masa_initial_frames_grid": (
        (
            ("media_streaming", "mini_pacman"),
            ("bridge_crossing", "bridge_crossing_v2"),
            ("colour_bomb", "colour_bomb_v2"),
        ),
        0.75 * 0.25 * A4_HEIGHT_IN,
        "three rows of two: Media Streaming and MiniPacman; Bridge Crossing v1 and v2; "
        "Colour Bomb v1 and v2",
    ),
    "masa_initial_frames_grid_2x3": (
        (
            ("media_streaming", "bridge_crossing", "colour_bomb"),
            ("mini_pacman", "bridge_crossing_v2", "colour_bomb_v2"),
        ),
        0.15 * A4_HEIGHT_IN,
        "two rows of three: Media Streaming, Bridge Crossing v1 and Colour Bomb v1 above "
        "MiniPacman, Bridge Crossing v2 and Colour Bomb v2",
    ),
}


def experiment_settings(environment: str) -> dict:
    config = json.loads(
        (DEFAULT_RUNS / CONTROLS[environment] / "two_hidden" / environment / "seed0" / "config.json").read_text()
    )
    return {"env_id": config["env_id"], "env_kwargs": config["env_kwargs"]}


def initial_frame(environment: str):
    settings = experiment_settings(environment)
    env = gym.make(settings["env_id"], render_mode="rgb_array", **settings["env_kwargs"])
    try:
        env.reset(seed=RESET_SEED)
        start_state = int(env.unwrapped._state)  # noqa: SLF001
        frame = env.render()
    finally:
        env.close()
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"{environment}: unexpected rgb_array frame {None if frame is None else frame.shape}")
    return frame, {**settings, "reset_seed": RESET_SEED, "start_state": start_state,
                   "frame_shape": list(frame.shape), "actions_executed": 0}


def save_single(frame, stem: Path, height_in: float) -> float:
    """Native PNG (DPI set so its natural height is height_in) and a margin-free PDF."""
    rows, cols = frame.shape[:2]
    width_in = height_in * cols / rows
    dpi = rows / height_in
    Image.fromarray(frame).save(stem.with_suffix(".png"), dpi=(dpi, dpi))
    fig = plt.figure(figsize=(width_in, height_in))
    axis = fig.add_axes((0, 0, 1, 1))
    axis.imshow(frame, interpolation="nearest")
    axis.set_axis_off()
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi)
    plt.close(fig)
    return width_in


def save_row(frames: list, labels: list[str], stem: Path) -> tuple[float, float]:
    """All environments in one row at the AAMAS text width, names underneath."""
    apply_aamas_style()
    aspects = [frame.shape[1] / frame.shape[0] for frame in frames]
    gap_in, label_in = 0.06, 0.2
    image_height = (FULL_WIDTH_IN - gap_in * (len(frames) - 1)) / sum(aspects)
    height = image_height + label_in
    fig = plt.figure(figsize=(FULL_WIDTH_IN, height))
    left = 0.0
    for frame, aspect, label in zip(frames, aspects, labels):
        width = image_height * aspect
        axis = fig.add_axes((left / FULL_WIDTH_IN, label_in / height, width / FULL_WIDTH_IN, image_height / height))
        axis.imshow(frame, interpolation="nearest")
        axis.set_axis_off()
        fig.text((left + width / 2) / FULL_WIDTH_IN, 0.35 * label_in / height, label,
                 ha="center", va="center", fontsize=7)
        left += width + gap_in
    fig.savefig(stem.with_suffix(".pdf"), dpi=600)
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    plt.close(fig)
    return FULL_WIDTH_IN, height


def save_grid(frames: dict, labels: dict, stem: Path, rows, height_in: float) -> tuple[float, float]:
    """``rows`` as a single-column AAMAS figure, each frame as large as its cell allows."""
    apply_aamas_style()
    gap_x, gap_y, label_in = 0.08, 0.05, 0.16
    n_columns = len(rows[0])
    cell_w = (COLUMN_WIDTH_IN - (n_columns - 1) * gap_x) / n_columns
    image_h = (height_in - len(rows) * label_in - (len(rows) - 1) * gap_y) / len(rows)
    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, height_in))
    for row, environments in enumerate(rows):
        top = height_in - row * (image_h + label_in + gap_y)
        for column, environment in enumerate(environments):
            frame = frames[environment]
            aspect = frame.shape[1] / frame.shape[0]
            height = min(image_h, cell_w / aspect)
            width = height * aspect
            centre = column * (cell_w + gap_x) + cell_w / 2
            bottom = top - image_h + (image_h - height) / 2
            axis = fig.add_axes((
                (centre - width / 2) / COLUMN_WIDTH_IN, bottom / height_in,
                width / COLUMN_WIDTH_IN, height / height_in,
            ))
            axis.imshow(frame, interpolation="nearest")
            axis.set_axis_off()
            fig.text(centre / COLUMN_WIDTH_IN, (top - image_h - 0.5 * label_in) / height_in,
                     labels[environment], ha="center", va="center", fontsize=7)
    fig.savefig(stem.with_suffix(".pdf"), dpi=600)
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    plt.close(fig)
    return COLUMN_WIDTH_IN, height_in


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--height", type=float, default=1.0, help="Printed height of each single frame (inches).")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames, labels, metadata = [], [], {}
    for environment, title in ENVIRONMENTS:
        frame, info = initial_frame(environment)
        width = save_single(frame, output / environment, args.height)
        frames.append(frame)
        labels.append(title.replace("\n", " "))
        metadata[environment] = {**info, "printed_size_in": [round(width, 3), args.height]}
        print(f"{environment}: {frame.shape[1]}x{frame.shape[0]} px -> {width:.2f} x {args.height:.2f} in")
    row_size = save_row(frames, labels, output / "masa_initial_frames_row")
    by_environment = {environment: frame for (environment, _), frame in zip(ENVIRONMENTS, frames)}
    label_by_environment = {environment: label for (environment, _), label in zip(ENVIRONMENTS, labels)}
    for name, (rows, height_in, layout) in GRIDS.items():
        grid_size = save_grid(by_environment, label_by_environment, output / name, rows, height_in)
        (output / f"{name}.tex").write_text(
            "\\begin{figure}[t]\n  \\centering\n"
            f"  \\includegraphics[width={grid_size[0]:g}in]{{{name}.pdf}}\n"
            "  \\caption{Initial frames of the six MASA environments (\\texttt{env.render()} in "
            "\\texttt{rgb\\_array} mode after \\texttt{reset}, before any action).}\n"
            f"  \\label{{fig:{name.replace('masa_', 'masa-').replace('_', '-')}}}\n"
            f"  \\Description{{Six rendered environments in {layout}.}}\n"
            "\\end{figure}\n"
        )
        print(f"{name}: {grid_size[0]:.2f} x {grid_size[1]:.2f} in")
    (output / "frame_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "masa_initial_frames_row.tex").write_text(
        "\\begin{figure*}[t]\n  \\centering\n"
        "  \\includegraphics{masa_initial_frames_row.pdf}\n"
        "  \\caption{Initial frames of the six MASA environments (\\texttt{env.render()} in "
        "\\texttt{rgb\\_array} mode after \\texttt{reset}, before any action).}\n"
        "  \\label{fig:masa-initial-frames}\n"
        "  \\Description{Six rendered environments in one row: Media Streaming, Colour Bomb v1, "
        "Colour Bomb v2, Bridge Crossing v1, Bridge Crossing v2 and MiniPacman.}\n"
        "\\end{figure*}\n"
    )
    print(f"row figure: {row_size[0]:.2f} x {row_size[1]:.2f} in")
    print(f"Saved PNG/PDF frames, the row figure and metadata in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
