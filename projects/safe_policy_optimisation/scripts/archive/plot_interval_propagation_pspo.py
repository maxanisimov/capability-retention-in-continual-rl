#!/usr/bin/env python
"""Draw a schematic of interval propagation for PSPO certification."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO = Path("/vol/bitbucket/ma5923/_projects/CertifiedContinualLearning")
OUT_DIR = REPO / "projects/safe_policy_optimisation/figures"
OUT_STEM = OUT_DIR / "interval_propagation_pspo"


plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
    title_color: str = "#1f2933",
    body_y_frac: float = 0.34,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height - 0.06,
        title,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * body_y_frac - 0.01,
        body,
        ha="center",
        va="center",
        fontsize=10.2,
        color="#263238",
        linespacing=1.35,
    )
    return box


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#455a64",
    label: str | None = None,
    rad: float = 0.0,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    if label:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.04,
            label,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=color,
        )


def add_interval_bars(ax: plt.Axes) -> None:
    inset = ax.inset_axes([0.725, 0.115, 0.235, 0.39])
    actions = ["a0", "a_safe", "a2"]
    lower = [0.45, 1.35, -0.20]
    upper = [0.95, 1.75, 0.30]
    colors = ["#d97361", "#3b9a63", "#d97361"]
    y_positions = [2, 1, 0]
    for y, lo, hi, color, label in zip(y_positions, lower, upper, colors, actions):
        inset.plot([lo, hi], [y, y], color=color, linewidth=8, solid_capstyle="butt")
        inset.plot([lo, lo], [y - 0.12, y + 0.12], color=color, linewidth=1.6)
        inset.plot([hi, hi], [y - 0.12, y + 0.12], color=color, linewidth=1.6)
        inset.text(-0.42, y, label, ha="right", va="center", fontsize=9)
    inset.axvline(0.95, color="#546e7a", linestyle="--", linewidth=1.2)
    inset.text(0.98, 2.45, "max unsafe\nupper", ha="left", va="top", fontsize=8.5, color="#546e7a")
    inset.text(1.36, 0.45, r"$L_{\mathrm{safe}} > \max U_{\mathrm{unsafe}}$", ha="left", va="center", fontsize=9.2, color="#1b5e20")
    inset.set_xlim(-0.45, 1.95)
    inset.set_ylim(-0.55, 2.65)
    inset.set_yticks([])
    inset.set_xlabel("logit interval")
    inset.spines[["top", "right", "left"]].set_visible(False)
    inset.grid(axis="x", alpha=0.22)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "Interval propagation certificate for PSPO",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#16212b",
    )

    add_box(
        ax,
        (0.04, 0.59),
        0.24,
        0.28,
        title="Safety-critical states",
        body=(
            "Example region R\n"
            r"$x_1 \in [0.20, 0.45]$" "\n"
            r"$x_2 \in [0.60, 0.90]$" "\n"
            "shield says: action 1 is safe"
        ),
        facecolor="#eef6ff",
        edgecolor="#467fcf",
        title_color="#184a86",
    )
    add_box(
        ax,
        (0.04, 0.17),
        0.24,
        0.27,
        title="Safe parameter box",
        body=(
            r"$\Theta = [\theta^-, \theta^+]$" "\n"
            "weights and biases vary\n"
            "within certified intervals"
        ),
        facecolor="#edf7ed",
        edgecolor="#3b8f59",
        title_color="#1f6b3c",
    )
    add_box(
        ax,
        (0.34, 0.46),
        0.17,
        0.25,
        title="Affine layer",
        body=(
            r"$z_1 = W_1 x + b_1$" "\n\n"
            r"$z_1 \in [z_1^-, z_1^+]$"
        ),
        facecolor="#fff7e8",
        edgecolor="#d9902f",
        title_color="#8a5513",
    )
    add_box(
        ax,
        (0.56, 0.46),
        0.15,
        0.25,
        title="Activation",
        body=(
            r"$h = \tanh(z_1)$" "\n\n"
            r"$h \in [h^-, h^+]$"
        ),
        facecolor="#fbf0ff",
        edgecolor="#8d5da8",
        title_color="#643a78",
    )
    add_box(
        ax,
        (0.76, 0.61),
        0.20,
        0.22,
        title="Logit bounds",
        body=(
            r"$\ell = W_2 h + b_2$" "\n"
            "one interval per action"
        ),
        facecolor="#eefaf4",
        edgecolor="#2f8f58",
        title_color="#1f6b3c",
        body_y_frac=0.35,
    )
    add_arrow(ax, (0.28, 0.72), (0.34, 0.60), label="state interval")
    add_arrow(ax, (0.28, 0.31), (0.34, 0.51), label="parameter intervals", rad=-0.18)
    add_arrow(ax, (0.51, 0.59), (0.56, 0.59), label="propagate bounds")
    add_arrow(ax, (0.71, 0.59), (0.76, 0.70), label="propagate bounds")

    add_interval_bars(ax)

    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"Wrote {OUT_STEM.with_suffix('.pdf')}")
    print(f"Wrote {OUT_STEM.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
