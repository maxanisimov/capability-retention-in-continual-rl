"""Plot an already-synthesised safety shield over the environment's rgb_array frame.

Shared across the safety_retention environment pipelines. Loads the shield Q-function saved
by synthesise_shield.py (<safety_retention>/<env>/artifacts/shields/<task>/shield_q.pt),
rebuilds the environment with ``render_mode='rgb_array'``, and overlays each cell's
per-action eventual-safety probabilities on the frame returned by ``env.render()``. Arrows
are coloured by an eventual-safety-probability colormap (with a colorbar).

For MASA envs whose state is more than a grid coordinate (e.g. CustomMiniPacman-v0, where a
state is agent x ghost x food), the plot shows a 2-D slice: the non-position features are
held fixed via ``--fixed-features`` (a JSON dict). Pass ``--fixed-features`` multiple times
to render several slices in one run; each plot is named after the fixed-feature values.

    # FrozenLake task (one grid, no fixed features):
    python plot_shield.py --task diagonal_4x4_stochastic_source

    # MASA env, two ghost-position slices -> two plots:
    python plot_shield.py --env CustomMiniPacman-v0 --task minipacman_default \\
        --fixed-features '{"ghost_y": 1, "ghost_x": 1}' \\
        --fixed-features '{"ghost_y": 3, "ghost_x": 8}'
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

# Headless rendering for FrozenLake / MASA envs (pygame) and matplotlib.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import torch

_SAFETY_RETENTION_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from experiments.pipelines.safety_retention.task_library import environment_subdir

FROZEN_LAKE_ENV = "FrozenLake-v1"
# FrozenLake action ids -> unit (dx, dy) in image pixel coordinates (y points down).
FROZEN_LAKE_ACTION_DIRECTIONS: dict[int, tuple[int, int]] = {
    0: (-1, 0),  # LEFT
    1: (0, 1),   # DOWN
    2: (1, 0),   # RIGHT
    3: (0, -1),  # UP
}
CMAP = matplotlib.colormaps["RdYlGn"]
NORM = Normalize(vmin=0.0, vmax=1.0)
_TEXT_BOX = dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7, edgecolor="none")


def _draw_safety_action(
    ax,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
    span_w: float,
    span_h: float,
    prob: float,
) -> None:
    """Draw one action as an arrow (or a dot for a stay/no-op) coloured by its eventual-safety
    probability, with a probability label. ``dx``/``dy`` are fractions of the cell span."""
    color = CMAP(NORM(prob))
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        ax.plot(cx, cy, marker="o", markersize=6, color=color, zorder=4)
        text_x, text_y, ha, va = cx, cy, "center", "center"
    else:
        ax.annotate(
            "",
            xy=(cx + dx * span_w, cy + dy * span_h),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0 + 3.0 * prob, shrinkA=0, shrinkB=0),
            zorder=4,
        )
        text_x = cx + dx * span_w * 1.4
        text_y = cy + dy * span_h * 1.4
        ha = "center" if abs(dx) < 1e-12 else ("right" if dx < 0 else "left")
        va = "center" if abs(dy) < 1e-12 else ("bottom" if dy < 0 else "top")
    ax.text(
        text_x, text_y, f"{prob:.2f}",
        ha=ha, va=va, fontsize=7, color=color, bbox=_TEXT_BOX, zorder=5,
    )


def _add_safety_colorbar(fig, ax) -> None:
    mappable = ScalarMappable(norm=NORM, cmap=CMAP)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(eventual safety)")


def _slug(fixed_features: dict | None) -> str | None:
    if not fixed_features:
        return None
    parts = [f"{key}={fixed_features[key]}" for key in sorted(fixed_features)]
    return re.sub(r"[^A-Za-z0-9._=-]+", "-", "__".join(parts))


def _plot_path(output_dir: Path, fixed_features: dict | None) -> Path:
    slug = _slug(fixed_features)
    name = "shield_plot.png" if slug is None else f"shield_plot__{slug}.png"
    return output_dir / name


# --------------------------------------------------------------------------------------
# FrozenLake-v1 grid plot (state == cell; no fixed features)
# --------------------------------------------------------------------------------------
def plot_frozenlake(payload: dict, *, seed: int, output_path: Path) -> None:
    desc = [str(row) for row in payload["map"]]
    deterministic = bool(payload.get("deterministic", True))
    slip_probability = float(payload.get("slip_probability", 0.0))
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=not deterministic,
        success_rate=1.0 - slip_probability,
        render_mode="rgb_array",
    )
    env.reset(seed=seed)
    frame = np.asarray(env.render())
    env.close()

    q_safety = payload["q_safety"].numpy()
    state_safety = payload["state_safety"].numpy()
    holes = set(payload["holes"])
    goals = set(payload["goals"])
    nrow, ncol = len(desc), len(desc[0])
    cell_h = frame.shape[0] / nrow
    cell_w = frame.shape[1] / ncol
    arrow_frac = 0.30

    fig, ax = plt.subplots(figsize=(max(5.0, ncol * 1.4), max(5.0, nrow * 1.4)))
    ax.imshow(frame)
    ax.set_xticks([])
    ax.set_yticks([])

    for state in range(nrow * ncol):
        r, c = divmod(state, ncol)
        cx = (c + 0.5) * cell_w
        cy = (r + 0.5) * cell_h
        if state in holes or state in goals:
            ax.text(
                cx, cy, f"{state_safety[state]:.2f}",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=CMAP(NORM(state_safety[state])), bbox=_TEXT_BOX, zorder=5,
            )
            continue
        for action, (ux, uy) in FROZEN_LAKE_ACTION_DIRECTIONS.items():
            _draw_safety_action(
                ax, cx, cy, arrow_frac * ux, arrow_frac * uy, cell_w, cell_h,
                float(q_safety[state, action]),
            )

    dynamics = "deterministic" if slip_probability == 0.0 else f"slip_probability={slip_probability:g}"
    ax.set_title(
        f"{payload['task']}: probabilistic safety shield\n"
        f"per-action P(eventually avoid holes), {dynamics}",
        fontsize=11,
    )
    _add_safety_colorbar(fig, ax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# MASA env plot (slice selected by fixed features)
# --------------------------------------------------------------------------------------
def plot_masa(payload: dict, fixed_features: dict | None, *, seed: int, output_path: Path) -> None:
    from experiments.utils.masa_tabular_envs import make_custom_masa_env
    from experiments.utils.masa_tabular_envs.visualisation import build_shield_layout

    q_safety = payload["q_safety"].numpy()
    env = make_custom_masa_env(payload["env"], env_kwargs=payload.get("env_kwargs"), render_mode="rgb_array")
    env.reset(seed=seed)
    try:
        layout = build_shield_layout(env, fixed_features=fixed_features)
    finally:
        env.close()

    frame = np.asarray(layout.background)
    cell_w = frame.shape[1] / float(layout.ncol)
    cell_h = frame.shape[0] / float(layout.nrow)

    fig, ax = plt.subplots(figsize=(max(6.0, layout.ncol * 0.9), max(5.0, layout.nrow * 0.9)))
    ax.imshow(frame)
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw the env's real actions only. Some layouts (e.g. the generic grid) expose more
    # action_deltas keys than the env has actions; q_safety has exactly n_actions columns.
    n_actions = q_safety.shape[1]
    for cell in layout.cells:
        cx = float(cell.center_x) if cell.center_x is not None else (cell.col + 0.5) * cell_w
        cy = float(cell.center_y) if cell.center_y is not None else (cell.row + 0.5) * cell_h
        span_w = float(cell.span_w) if cell.span_w is not None else cell_w
        span_h = float(cell.span_h) if cell.span_h is not None else cell_h
        for action in range(n_actions):
            dx, dy = layout.action_deltas.get(action, (0.0, 0.0))
            _draw_safety_action(ax, cx, cy, float(dx), float(dy), span_w, span_h, float(q_safety[cell.state, action]))

    slice_note = "" if not fixed_features else f"\nfixed: {fixed_features}"
    ax.set_title(
        f"{payload['env']} ({payload['task']}): probabilistic safety shield\n"
        f"per-action P(eventual safety){slice_note}",
        fontsize=11,
    )
    _add_safety_colorbar(fig, ax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a synthesised safety shield over the env's rgb_array frame.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=FROZEN_LAKE_ENV,
        help="Environment id, used to locate the shield under <safety_retention>/<env>/artifacts/shields/.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Run label whose shield_q.pt to plot.",
    )
    parser.add_argument(
        "--shield-path",
        type=Path,
        default=None,
        help="Explicit path to a shield_q.pt. Overrides the --env/--task lookup.",
    )
    parser.add_argument(
        "--fixed-features",
        type=str,
        action="append",
        default=None,
        help=(
            "JSON dict of non-position state features to hold fixed for one plot slice "
            "(MASA envs). Pass multiple times to render multiple slices."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed passed to env.reset() for the rendered frame.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the plots. Default: the directory containing shield_q.pt.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    shield_path = args.shield_path or (
        _SAFETY_RETENTION_ROOT / environment_subdir(args.env) / "artifacts" / "shields" / args.task / "shield_q.pt"
    )
    if not shield_path.exists():
        raise FileNotFoundError(
            f"No shield Q-function at {shield_path}. Run synthesise_shield.py first.",
        )
    payload = torch.load(shield_path, weights_only=False)
    output_dir = args.output_dir or shield_path.parent

    if payload["env"] == FROZEN_LAKE_ENV:
        if args.fixed_features:
            print("FrozenLake state is the cell itself; --fixed-features is ignored.")
        output_path = output_dir / "shield_plot.png"
        plot_frozenlake(payload, seed=args.seed, output_path=output_path)
        print(f"Saved shield plot to {output_path}")
        return 0

    feature_sets = (
        [json.loads(spec) for spec in args.fixed_features] if args.fixed_features else [None]
    )
    for fixed_features in feature_sets:
        output_path = _plot_path(output_dir, fixed_features)
        plot_masa(payload, fixed_features, seed=args.seed, output_path=output_path)
        print(f"Saved shield plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
