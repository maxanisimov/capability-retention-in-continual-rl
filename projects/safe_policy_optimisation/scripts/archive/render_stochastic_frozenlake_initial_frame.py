#!/usr/bin/env python3
"""Render the experiment's exact FrozenLake layout at reset, before any action."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ccl-frozenlake-frame-matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from projects.safe_policy_optimisation.utils.frozen_lake_experiment import (  # noqa: E402
    SparseFrozenLake,
)

DEFAULT_RUN = (
    REPO / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
    / "pspo_stochastic_frozenlake128"
)
DEFAULT_OUTPUT = REPO / "projects/safe_policy_optimisation/figures/stochastic_frozenlake128_initial_frame"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pixels", type=int, default=2048)
    args = parser.parse_args()
    layout = args.run_root / "_inputs/layout.txt"
    experiment = json.loads((args.run_root / "experiment.json").read_text())
    digest = hashlib.sha256(layout.read_bytes()).hexdigest()
    if digest != experiment["input_sha256"]["layout.txt"]:
        raise ValueError("Layout differs from the one used in the experiment.")
    if args.pixels <= 0 or args.pixels % 128:
        parser.error("--pixels must be a positive multiple of 128")
    settings = experiment["settings"]
    env = SparseFrozenLake(
        layout_path=str(layout), render_mode="rgb_array",
        success_rate=settings["success_rate"], step_penalty=settings["step_penalty"],
    )
    state, _ = env.reset(seed=0)
    if state != 0 or env.lastaction is not None:
        raise AssertionError("Frame is not the initial state.")
    # Increase rendering resolution before creating surfaces/sprites. Dynamics
    # and the saved experiment inputs are unchanged.
    env.window_size = (args.pixels, args.pixels)
    env.cell_size = (args.pixels // 128, args.pixels // 128)
    frame = env.render()
    if frame.shape != (args.pixels, args.pixels, 3):
        raise AssertionError(f"Unexpected native frame shape: {frame.shape}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(output / "initial_frame.png")
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    ax.imshow(frame, extent=(-0.5, 127.5, 127.5, -0.5), interpolation="nearest")
    ax.annotate(
        "Agent / start (0, 0)", xy=(0, 0), xytext=(9, 7), fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#15803d", "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": "#15803d"},
    )
    ax.annotate(
        "Goal (127, 127)", xy=(127, 127), xytext=(87, 120), fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#b91c1c", "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": "#b91c1c"},
    )
    ax.set_title("Stochastic FrozenLake 128×128 — initial step (t = 0)", fontsize=11)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks([0, 32, 64, 96, 127])
    ax.set_yticks([0, 32, 64, 96, 127])
    fig.savefig(output / "initial_layout.pdf")
    fig.savefig(output / "initial_layout.png", dpi=300)
    plt.close(fig)
    (output / "initial_frame.tex").write_text(
        "\\begin{frame}{Stochastic FrozenLake: initial state}\n\\centering\n"
        "\\includegraphics[height=0.76\\textheight]{initial_layout.pdf}\n\n"
        "{\\footnotesize Fixed $128\\times128$ layout; agent at $(0,0)$, goal at $(127,127)$.}"
        "\n\\end{frame}\n"
    )
    (output / "frame_metadata.json").write_text(json.dumps({
        "layout_path": str(layout.resolve()), "layout_sha256": digest,
        "time_step": 0, "reset_seed": 0, "state_id": int(state),
        "agent_row_col": [0, 0], "goal_row_col": [127, 127],
        "renderer": "Gymnasium native FrozenLake rgb_array",
        "pixels": args.pixels, "actions_executed": 0,
    }, indent=2) + "\n")
    env.close()
    print(f"Saved initial-state frame and annotated PNG/PDF in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
