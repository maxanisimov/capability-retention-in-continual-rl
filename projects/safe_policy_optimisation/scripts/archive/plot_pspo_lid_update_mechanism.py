#!/usr/bin/env python
"""Draw a 3D schematic of the three-stage PSPO policy update.

Panel (a) proposes a PPO update, panel (b) grows the maximal Local Invariant
Domain (LID) along that update's direction, and panel (c) projects the proposal
onto the LID.

The geometry follows the implementation rather than a generic trust region: a
LID is an *axis-aligned orthotope* (``OrthotopeRegion`` in
``core/provably_safe_policy_optimisation/regions.py``), directional growth makes
it one-sided so that ``theta_k`` is a *vertex* rather than the centre
(``_directional_masks_from_update_deltas`` in ``adaptive_safe_ppo_v2.py``), the
box is grown to maximise the ``|Delta_k|``-weighted width subject to staying
inside the certified-safe set, and the projection is coordinate-wise clipping
(``projection.py``), so untouched coordinates keep their proposed value.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from projects.safe_policy_optimisation.scripts.plot_safe_parameter_region_3d import (  # noqa: E402
    box_faces,
    draw_arrow,
)

OUT_DIR = REPO / "projects/safe_policy_optimisation/figures"
OUT_STEM = OUT_DIR / "pspo_lid_update_mechanism"

SAFE_DARK = "#0b3d2e"
SAFE_EDGE = "#1f5f36"
SAFE_FILL = "#5aa469"
SAFE_LINE = "#157347"
UNSAFE = "#cc5a43"
PROJECT = "#2468b2"
LATEST = "#1f77b4"
NEUTRAL = "#6c757d"
AMBIENT_FILL = "#9dc6a8"
AMBIENT_EDGE = "#6f9a7d"
LEADER = "#8a8f94"

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

AXIS_NAMES = (r"\theta_1", r"\theta_2", r"\theta_3")

THETA_K = np.array([0.0, 0.0, 0.0])
DELTA_K = np.array([0.95, 0.80, 0.52])
THETA_BAR = THETA_K + DELTA_K

# Certified-safe set: a deliberately non-convex blob containing theta_k but not
# the proposal, so growth along Delta_k is stopped by a curved boundary.
BLOB_CENTRE = np.array([0.24, 0.10, 0.00])
BLOB_R0 = 0.60
BLOB_A = 0.20
BLOB_B = 0.10
BLOB_SCALES = np.array([1.30, 0.98, 0.92])

# Growth-objective log-volume/width mix, as in `_magnitude_weighted_objective_fn`
# (core/src/interval_utils.py). The code default is 0.0 (pure |Delta|-weighted
# width); a small log-volume term keeps every side of the drawn box visible
# without changing the geometry class.
OBJ_ALPHA = 0.35


def blob_radius(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Direction-dependent radius of the certified-safe set."""
    return BLOB_R0 * (1.0 + BLOB_A * np.cos(2.0 * u) + BLOB_B * np.sin(v))


def blob_surface(n_u: int = 96, n_v: int = 49) -> tuple[np.ndarray, ...]:
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    uu, vv = np.meshgrid(u, v)
    r = blob_radius(uu, vv)
    x = BLOB_CENTRE[0] + BLOB_SCALES[0] * r * np.sin(vv) * np.cos(uu)
    y = BLOB_CENTRE[1] + BLOB_SCALES[1] * r * np.sin(vv) * np.sin(uu)
    z = BLOB_CENTRE[2] + BLOB_SCALES[2] * r * np.cos(vv)
    return x, y, z


def blob_point(u: float, v: float) -> np.ndarray:
    """A single point on the boundary of the certified-safe set."""
    r = blob_radius(np.asarray(u), np.asarray(v))
    return BLOB_CENTRE + BLOB_SCALES * r * np.array(
        [np.sin(v) * np.cos(u), np.sin(v) * np.sin(u), np.cos(v)]
    )


def inside_blob(points: np.ndarray) -> np.ndarray:
    """Boolean mask of which points lie inside the certified-safe set."""
    local = (points - BLOB_CENTRE) / BLOB_SCALES
    radius = np.linalg.norm(local, axis=-1)
    v = np.arccos(np.clip(local[..., 2] / np.maximum(radius, 1e-12), -1.0, 1.0))
    u = np.arctan2(local[..., 1], local[..., 0])
    return radius <= blob_radius(u, v)


def unit_box_samples(n: int = 11) -> np.ndarray:
    """Dense sample of the boundary of the unit box, in [0, 1]^3."""
    grid = np.linspace(0.0, 1.0, n)
    a, b = np.meshgrid(grid, grid)
    a, b = a.ravel(), b.ravel()
    ones, zeros = np.ones_like(a), np.zeros_like(a)
    faces = []
    for fixed in (zeros, ones):
        faces.append(np.stack([fixed, a, b], axis=1))
        faces.append(np.stack([a, fixed, b], axis=1))
        faces.append(np.stack([a, b, fixed], axis=1))
    return np.concatenate(faces, axis=0)


UNIT_BOX_SAMPLES = unit_box_samples()


def max_scale_inside(shapes: np.ndarray, hi: float = 4.0, n_iter: int = 28) -> np.ndarray:
    """Per-shape largest ``t`` keeping ``[THETA_K, THETA_K + t * shape]`` certified.

    Bisects every candidate shape at once: ``shapes`` is ``(S, 3)``, result ``(S,)``.
    """
    lo = np.zeros(len(shapes))
    hi_arr = np.full(len(shapes), hi)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi_arr)
        corners = mid[:, None] * shapes
        points = THETA_K + corners[:, None, :] * UNIT_BOX_SAMPLES[None, :, :]
        safe = inside_blob(points).all(axis=1)
        lo = np.where(safe, mid, lo)
        hi_arr = np.where(safe, hi_arr, mid)
    return lo


def growth_objective(uppers: np.ndarray, alpha: float = OBJ_ALPHA) -> np.ndarray:
    """The PSPO magnitude-weighted growth objective, evaluated per box.

    Mirrors ``_magnitude_weighted_objective_fn``:
    ``[a * sum_j w_j log(r_j + 1e-6) + (1 - a) * sum_j w_j r_j] / sum_j w_j``
    with ``w_j = |Delta_{k,j}|`` and ``r_j`` the per-coordinate width.
    """
    weights = np.abs(DELTA_K)
    widths = uppers - THETA_K
    logvol = (weights * np.log(widths + 1e-6)).sum(axis=-1)
    width = (weights * widths).sum(axis=-1)
    return (alpha * logvol + (1.0 - alpha) * width) / weights.sum()


def fit_maximal_lid(n_grid: int = 41) -> np.ndarray:
    """Grid-search the box shape maximising the growth objective inside the blob.

    The shape is normalised so the dominant coordinate of ``Delta_k`` has unit
    extent and the two remaining ratios are swept, so the returned box genuinely
    touches the certified boundary rather than being placed by hand.
    """
    ratios = np.linspace(0.04, 1.0, n_grid)
    r2, r3 = np.meshgrid(ratios, ratios)
    shapes = np.stack([np.ones_like(r2).ravel(), r2.ravel(), r3.ravel()], axis=1)
    uppers = THETA_K + max_scale_inside(shapes)[:, None] * shapes
    return uppers[int(np.argmax(growth_objective(uppers)))]


def box_edges(upper: np.ndarray) -> list[np.ndarray]:
    """The twelve edges of the box ``[THETA_K, upper]`` as (2, 3) segments."""
    lo, hi = THETA_K, upper
    corners = np.array([[lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]]])
    segments = []
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for i in (0, 1):
            for j in (0, 1):
                start, end = np.empty(3), np.empty(3)
                start[axis], end[axis] = corners[axis]
                for other, index in zip(others, (i, j)):
                    start[other] = end[other] = corners[other][index]
                segments.append(np.stack([start, end]))
    return segments


def draw_wireframe(
    ax: plt.Axes,
    upper: np.ndarray,
    *,
    color: str,
    alpha: float = 1.0,
    linewidth: float = 1.0,
    linestyle: str = "-",
    zorder: int = 4,
) -> None:
    for segment in box_edges(upper):
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )


def draw_lid(ax: plt.Axes, upper: np.ndarray) -> None:
    """The certified LID: translucent fill plus a crisp wireframe."""
    ax.add_collection3d(
        Poly3DCollection(
            box_faces(THETA_K, upper),
            facecolor=SAFE_FILL,
            edgecolor="none",
            alpha=0.16,
            zorder=3,
        )
    )
    draw_wireframe(ax, upper, color=SAFE_EDGE, linewidth=1.5, zorder=5)


def draw_blob(ax: plt.Axes, *, alpha: float = 0.13) -> None:
    """The ambient safe parameter set: curved, non-convex, not axis-aligned."""
    x, y, z = blob_surface()
    ax.plot_surface(
        x, y, z, color=AMBIENT_FILL, alpha=alpha, linewidth=0.0, shade=True,
        rstride=2, cstride=2, zorder=1,
    )
    ax.plot_wireframe(
        x, y, z, color=AMBIENT_EDGE, alpha=0.30, linewidth=0.5, linestyle="--",
        rstride=8, cstride=12, zorder=2,
    )


def leader(ax: plt.Axes, start: np.ndarray, end: np.ndarray) -> None:
    """A thin line tying a text label to the object it names."""
    ax.plot(
        *zip(start, end), color=LEADER, linewidth=0.7, alpha=0.75,
        linestyle="-", zorder=9,
    )


def label_safe_set(ax: plt.Axes) -> None:
    anchor = np.array([-0.70, -0.52, 0.46])
    leader(ax, anchor, blob_point(-2.7, 1.25))
    ax.text(
        anchor[0], anchor[1], anchor[2] + 0.03,
        r"$\Theta^{\mathrm{safe}}$", color="#4a7a5c", fontsize=11,
    )


def note(ax: plt.Axes, text: str, color: str, y: float = 0.94) -> None:
    ax.text2D(0.01, y, text, transform=ax.transAxes, color=color, fontsize=8.5, va="top")


def style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, pad=2)
    ax.set_xlabel(r"$\theta_1$", labelpad=-6)
    ax.set_ylabel(r"$\theta_2$", labelpad=-6)
    ax.set_zlabel(r"$\theta_3$", labelpad=-8)
    ax.set_xlim(-0.70, 1.20)
    ax.set_ylim(-0.62, 0.95)
    ax.set_zlim(-0.60, 0.72)
    ax.set_box_aspect((1.55, 1.25, 1.05), zoom=0.88)
    ax.view_init(elev=20, azim=-58)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7, pad=-3)


def mark_theta_k(ax: plt.Axes) -> None:
    ax.scatter(*THETA_K, s=80, color=SAFE_DARK, edgecolor="white", linewidth=1.0, zorder=8)
    ax.text(-0.06, -0.04, -0.22, r"$\theta_k$", color=SAFE_DARK, ha="right", fontsize=11)


def mark_theta_bar(ax: plt.Axes) -> None:
    ax.scatter(*THETA_BAR, s=95, marker="x", color=UNSAFE, linewidth=2.4, zorder=8)
    ax.text(
        THETA_BAR[0], THETA_BAR[1] + 0.04, THETA_BAR[2] + 0.09,
        r"$\bar{\theta}_{k+1}$", color=UNSAFE, fontsize=11,
    )


def draw_panel_a(ax: plt.Axes) -> None:
    draw_blob(ax)
    draw_arrow(ax, THETA_K, THETA_BAR, color=UNSAFE, linewidth=2.4)
    mark_theta_k(ax)
    mark_theta_bar(ax)
    mid = THETA_K + 0.55 * DELTA_K
    ax.text(mid[0] - 0.06, mid[1] + 0.02, mid[2] + 0.10, r"$\Delta_k$", color=UNSAFE, fontsize=12)
    label_safe_set(ax)
    note(
        ax,
        r"PPO proposes an unconstrained step $\Delta_k$. It leaves the safe"
        "\n"
        r"parameter set $\Theta^{\mathrm{safe}}=\{\theta: H(s,\mu_\theta(s))=1\ \forall s\in\mathcal{D}_{\mathrm{sh}}\}$,"
        "\n"
        "which is curved and not axis-aligned.",
        SAFE_DARK,
    )


def draw_panel_b(ax: plt.Axes, upper: np.ndarray) -> None:
    draw_blob(ax)
    draw_arrow(ax, THETA_K, THETA_BAR, color=NEUTRAL, linestyle=":", linewidth=1.3, alpha=0.5)
    ax.scatter(*THETA_BAR, s=70, marker="x", color=UNSAFE, linewidth=1.8, alpha=0.45, zorder=8)
    draw_wireframe(
        ax, THETA_K + 1.25 * (upper - THETA_K),
        color=UNSAFE, alpha=0.9, linewidth=1.1, linestyle="--", zorder=6,
    )
    draw_lid(ax, upper)
    for scale, alpha in ((0.38, 0.55), (0.68, 0.75)):
        draw_wireframe(
            ax, THETA_K + scale * (upper - THETA_K),
            color=SAFE_DARK, alpha=alpha, linewidth=1.0, linestyle=":", zorder=7,
        )
    mark_theta_k(ax)
    label_safe_set(ax)
    anchor = np.array([upper[0] * 0.10, upper[1] + 0.30, upper[2] + 0.30])
    leader(ax, anchor, np.array([upper[0] * 0.42, upper[1], upper[2]]))
    ax.text(
        anchor[0], anchor[1], anchor[2] + 0.03,
        r"$\Theta_k^{\mathrm{LID}}=[\ell_k,u_k]$", color=SAFE_EDGE, fontsize=10,
    )
    note(
        ax,
        r"Grow an axis-aligned box anchored at $\theta_k$ into the octant"
        "\n"
        r"of $\Delta_k$, until the IBP certificate fails (dashed). The LID is"
        "\n"
        r"a certified inner approximation: $\Theta_k^{\mathrm{LID}}\subseteq\Theta^{\mathrm{safe}}$.",
        SAFE_DARK,
    )


def clipping_path(theta_next: np.ndarray) -> list[np.ndarray]:
    """Waypoints clipping one overshooting coordinate at a time."""
    point = THETA_BAR.copy()
    path = [point.copy()]
    for axis in np.flatnonzero(~np.isclose(theta_next, THETA_BAR)):
        point[axis] = theta_next[axis]
        path.append(point.copy())
    return path


def draw_panel_c(ax: plt.Axes, upper: np.ndarray, theta_next: np.ndarray) -> None:
    draw_blob(ax, alpha=0.07)
    draw_lid(ax, upper)
    for start, end in zip(*(lambda p: (p[:-1], p[1:]))(clipping_path(theta_next))):
        ax.plot(
            *zip(start, end), color=PROJECT, linestyle="--",
            linewidth=1.2, alpha=0.9, zorder=7,
        )
    draw_arrow(ax, THETA_BAR, theta_next, color=PROJECT, linewidth=2.2)
    draw_arrow(ax, THETA_K, theta_next, color=SAFE_LINE, linewidth=2.6)
    mark_theta_k(ax)
    mark_theta_bar(ax)
    ax.scatter(*theta_next, s=95, color=LATEST, edgecolor="white", linewidth=1.0, zorder=9)
    ax.text(
        theta_next[0] + 0.03, theta_next[1] + 0.21, theta_next[2] - 0.20,
        r"$\theta_{k+1}$", color=LATEST, ha="left", fontsize=11,
    )
    clipped = np.flatnonzero(~np.isclose(theta_next, THETA_BAR))
    kept = np.flatnonzero(np.isclose(theta_next, THETA_BAR))
    clipped_txt = ", ".join(f"${AXIS_NAMES[i]}$" for i in clipped)
    kept_txt = ", ".join(f"${AXIS_NAMES[i]}$" for i in kept)
    note(
        ax,
        r"$\theta_{k+1}=\mathrm{Proj}_{\Theta_k^{\mathrm{LID}}}(\bar{\theta}_{k+1})$"
        " clips coordinate-wise:\n"
        f"{clipped_txt} hit a face, {kept_txt} keeps its proposed value.",
        PROJECT,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    upper = fit_maximal_lid()
    theta_next = np.clip(THETA_BAR, THETA_K, upper)
    n_clipped = int(np.sum(~np.isclose(theta_next, THETA_BAR)))
    print(f"LID upper corner u_k: {np.round(upper, 4).tolist()}")
    print(f"theta_next: {np.round(theta_next, 4).tolist()} ({n_clipped} coordinate(s) clipped)")

    fig = plt.figure(figsize=(14.0, 4.5))
    axes = [fig.add_subplot(1, 3, i, projection="3d") for i in (1, 2, 3)]

    draw_panel_a(axes[0])
    draw_panel_b(axes[1], upper)
    draw_panel_c(axes[2], upper, theta_next)

    titles = (
        r"(a) Propose an update $\bar{\theta}_{k+1}=\theta_k+\Delta_k$",
        r"(b) Grow the maximal LID along $\Delta_k$",
        r"(c) Project $\bar{\theta}_{k+1}$ onto the LID",
    )
    for ax, title in zip(axes, titles):
        style_axes(ax, title)

    legend_handles = [
        plt.Line2D([0], [0], color=UNSAFE, linewidth=2.2, label=r"proposed update $\Delta_k$"),
        plt.Line2D([0], [0], color=UNSAFE, marker="x", linestyle="none", markersize=8,
                   label=r"uncertified proposal $\bar{\theta}_{k+1}$"),
        plt.Rectangle((0, 0), 1, 1, facecolor=AMBIENT_FILL, alpha=0.45, edgecolor=AMBIENT_EDGE,
                      linestyle="--",
                      label=r"safe set $\Theta^{\mathrm{safe}}$ (curved, not axis-aligned)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=SAFE_FILL, alpha=0.30, edgecolor=SAFE_EDGE,
                      label=r"certified LID $\Theta_k^{\mathrm{LID}}$ (axis-aligned box)"),
        plt.Line2D([0], [0], color=SAFE_DARK, linestyle=":", linewidth=1.2,
                   label="growth checkpoints"),
        plt.Line2D([0], [0], color=UNSAFE, linestyle="--", linewidth=1.2,
                   label="over-grown box (fails certificate)"),
        plt.Line2D([0], [0], color=PROJECT, linewidth=2.2, label="coordinate-wise projection"),
        plt.Line2D([0], [0], color=SAFE_LINE, linewidth=2.4,
                   label=r"accepted update $\theta_k\to\theta_{k+1}$"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.4, handlelength=1.8)
    fig.subplots_adjust(left=0.005, right=1.0, top=1.0, bottom=0.10, wspace=0.0)
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT_STEM.with_suffix(".png"), bbox_inches="tight", pad_inches=0.12, dpi=300)
    print(f"Wrote {OUT_STEM.with_suffix('.pdf')}")
    print(f"Wrote {OUT_STEM.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
