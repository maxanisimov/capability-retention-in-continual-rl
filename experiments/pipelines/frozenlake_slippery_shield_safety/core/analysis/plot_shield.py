"""Shield probability plotting helpers for slippery FrozenLake."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


FROZENLAKE_ACTION_DELTAS = {
    0: (-1.0, 0.0),  # left
    1: (0.0, 1.0),   # down
    2: (1.0, 0.0),   # right
    3: (0.0, -1.0),  # up
}


def _grid_shape_from_env(env: Any) -> tuple[int, int]:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "desc"):
        return tuple(np.asarray(unwrapped.desc).shape)  # type: ignore[return-value]
    if hasattr(unwrapped, "nrow") and hasattr(unwrapped, "ncol"):
        return int(unwrapped.nrow), int(unwrapped.ncol)
    raise ValueError("Cannot infer FrozenLake grid shape from environment.")


def _safe_probability(action_risk: np.ndarray | None, state: int, action: int) -> float:
    if action_risk is None:
        return 1.0
    risk = float(action_risk[state, action])
    return float(np.clip(1.0 - risk, 0.0, 1.0))


def plot_shield_safety_probabilities(
    env: Any,
    shield: np.ndarray,
    *,
    action_risk: np.ndarray | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    seed: int = 42,
    arrow_scale: float = 0.32,
    arrow_width: float = 2.0,
    show_probability_text: bool = True,
    show_disallowed_actions: bool = False,
) -> Any:
    """Plot shield arrows coloured by probability of remaining safe.

    By default only actions allowed by the shield are drawn. If
    ``show_disallowed_actions`` is true, every state-action pair is drawn with
    rejected actions shown faintly. If ``action_risk`` is given, arrow colour is
    ``1 - action_risk[state, action]``. Without risk metadata, allowed actions
    are rendered as fully safe.
    """

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import matplotlib.patheffects as path_effects

    if getattr(env.unwrapped, "render_mode", None) != "rgb_array":
        raise ValueError("Environment must be created with render_mode='rgb_array'.")

    nrow, ncol = _grid_shape_from_env(env)
    shield_arr = np.asarray(shield)
    expected_shape = (nrow * ncol, 4)
    if shield_arr.shape != expected_shape:
        raise ValueError(f"Expected shield shape {expected_shape}, got {shield_arr.shape}.")

    risk_arr = None if action_risk is None else np.asarray(action_risk, dtype=np.float64)
    if risk_arr is not None and risk_arr.shape != expected_shape:
        raise ValueError(f"Expected action_risk shape {expected_shape}, got {risk_arr.shape}.")

    env.reset(seed=seed)
    frame = env.render()
    if frame is None:
        raise RuntimeError("env.render() returned None; check render_mode.")

    img_h, img_w = frame.shape[:2]
    cell_w = img_w / float(ncol)
    cell_h = img_h / float(nrow)

    dpi = 100
    figsize = (img_w / dpi + 1.4, img_h / dpi + 0.8)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(frame)
    ax.axis("off")
    ax.set_title("Shield safety probabilities" if title is None else title)

    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=0.0, vmax=1.0)

    for state in range(shield_arr.shape[0]):
        row = state // ncol
        col = state % ncol
        cx = (col + 0.5) * cell_w
        cy = (row + 0.5) * cell_h
        if show_disallowed_actions:
            visible_actions = range(shield_arr.shape[1])
        else:
            visible_actions = np.flatnonzero(shield_arr[state] > 0)
        for action in visible_actions:
            allowed = bool(shield_arr[state, int(action)] > 0)
            if not allowed and not show_disallowed_actions:
                continue
            dx_unit, dy_unit = FROZENLAKE_ACTION_DELTAS[int(action)]
            dx = dx_unit * cell_w * arrow_scale
            dy = dy_unit * cell_h * arrow_scale
            p_safe = _safe_probability(risk_arr, state, int(action))
            color = cmap(norm(p_safe))
            ax.annotate(
                "",
                xy=(cx + dx, cy + dy),
                xytext=(cx, cy),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "lw": arrow_width,
                    "alpha": 0.95 if allowed else 0.35,
                    "linestyle": "-" if allowed else ":",
                    "shrinkA": 0.0,
                    "shrinkB": 0.0,
                },
            )
            if show_probability_text:
                label_x = cx + dx * 0.78
                label_y = cy + dy * 0.78
                label = ax.text(
                    label_x,
                    label_y,
                    f"{p_safe:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white",
                    weight="bold",
                    alpha=1.0 if allowed else 0.55,
                )
                label.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=1.4, foreground="black"),
                        path_effects.Normal(),
                    ],
                )

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(stay safe after action)")
    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Figure saved to {path}")

    return fig
