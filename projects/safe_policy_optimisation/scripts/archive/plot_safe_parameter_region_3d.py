#!/usr/bin/env python
"""Draw a 3D schematic of PSPO updates inside a safe parameter region."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


REPO = Path("/vol/bitbucket/ma5923/_projects/CertifiedContinualLearning")
OUT_DIR = REPO / "projects/safe_policy_optimisation/figures"
OUT_STEM = OUT_DIR / "safe_parameter_region_3d"


plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def box_faces(lower: np.ndarray, upper: np.ndarray) -> list[list[tuple[float, float, float]]]:
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


def draw_arrow(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: str,
    linestyle: str = "-",
    linewidth: float = 2.0,
    alpha: float = 1.0,
) -> None:
    delta = end - start
    ax.quiver(
        start[0],
        start[1],
        start[2],
        delta[0],
        delta[1],
        delta[2],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        arrow_length_ratio=0.12,
        length=1.0,
        normalize=False,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lower = np.array([-1.0, -0.75, -0.55])
    upper = np.array([1.0, 0.75, 0.65])
    raw_steps = np.array(
        [
            [-0.70, -0.35, -0.25],
            [-0.25, 0.20, 0.10],
            [0.30, 0.75, 0.35],
            [0.95, 1.00, 0.80],
            [1.30, 0.60, 0.90],
            [0.80, 0.10, 0.35],
        ]
    )
    safe_steps = np.clip(raw_steps, lower, upper)

    fig = plt.figure(figsize=(8.0, 6.4))
    ax = fig.add_subplot(111, projection="3d")

    safe_box = Poly3DCollection(
        box_faces(lower, upper),
        facecolor="#5aa469",
        edgecolor="#1f5f36",
        linewidths=1.2,
        alpha=0.18,
    )
    ax.add_collection3d(safe_box)

    # Lightly mark the target direction that unconstrained PPO would prefer.
    objective_start = np.array([-0.95, -0.68, -0.50])
    objective_end = np.array([1.28, 1.02, 0.92])
    draw_arrow(
        ax,
        objective_start,
        objective_end,
        color="#6c757d",
        linestyle=":",
        linewidth=1.5,
        alpha=0.45,
    )

    for i in range(len(raw_steps) - 1):
        start = safe_steps[i]
        candidate = raw_steps[i + 1]
        projected = safe_steps[i + 1]
        candidate_safe = np.allclose(candidate, projected)
        if candidate_safe:
            draw_arrow(ax, start, projected, color="#157347", linewidth=2.7)
        else:
            draw_arrow(ax, start, candidate, color="#cc5a43", linestyle="--", linewidth=1.8, alpha=0.86)
            draw_arrow(ax, candidate, projected, color="#2468b2", linewidth=2.2)

    ax.plot(
        safe_steps[:, 0],
        safe_steps[:, 1],
        safe_steps[:, 2],
        color="#157347",
        linewidth=2.1,
        marker="o",
        markersize=5,
        label="accepted safe iterates",
    )
    unsafe = np.array([raw for raw, safe in zip(raw_steps, safe_steps) if not np.allclose(raw, safe)])
    if len(unsafe):
        ax.scatter(
            unsafe[:, 0],
            unsafe[:, 1],
            unsafe[:, 2],
            marker="x",
            s=80,
            color="#cc5a43",
            linewidth=2.2,
            label="unsafe PPO candidates",
        )

    ax.scatter(
        safe_steps[0, 0],
        safe_steps[0, 1],
        safe_steps[0, 2],
        s=90,
        color="#0b3d2e",
        edgecolor="white",
        linewidth=1.0,
        label="safe base policy",
    )
    ax.scatter(
        safe_steps[-1, 0],
        safe_steps[-1, 1],
        safe_steps[-1, 2],
        s=90,
        color="#1f77b4",
        edgecolor="white",
        linewidth=1.0,
        label="latest policy",
    )

    ax.text(-0.88, -0.46, -0.30, "safe base\npolicy", color="#0b3d2e", ha="right")
    ax.text(-0.25, -0.90, 0.70, "certified safe\nparameter region", color="#1f5f36", ha="center")

    ax.set_title("Safe policy optimisation in a 3-parameter slice")
    ax.set_xlabel(r"policy parameter $\theta_1$")
    ax.set_ylabel(r"policy parameter $\theta_2$")
    ax.set_zlabel(r"policy parameter $\theta_3$")
    ax.set_xlim(-1.25, 1.45)
    ax.set_ylim(-1.00, 1.12)
    ax.set_zlim(-0.75, 1.02)
    ax.view_init(elev=24, azim=-55)
    ax.grid(True, alpha=0.25)
    ax.set_box_aspect((1.55, 1.2, 1.0))

    legend_handles = [
        plt.Line2D([0], [0], color="#157347", marker="o", linewidth=2.2, label="safe update accepted"),
        plt.Line2D([0], [0], color="#cc5a43", marker="x", linestyle="--", linewidth=1.8, label="unsafe candidate"),
        plt.Line2D([0], [0], color="#2468b2", linewidth=2.2, label="projected update"),
        plt.Line2D([0], [0], color="#6c757d", linestyle=":", linewidth=1.8, label="reward objective"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#5aa469", alpha=0.18, edgecolor="#1f5f36", label="safe region"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.96), frameon=False)

    fig.text(
        0.5,
        0.02,
        "PPO proposes reward-improving steps; PSPO accepts safe candidates and projects unsafe candidates back into the certified safe region.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"Wrote {OUT_STEM.with_suffix('.pdf')}")
    print(f"Wrote {OUT_STEM.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
