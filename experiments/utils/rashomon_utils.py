import os
import math
from pathlib import Path
import numpy as np
from scipy.fftpack import dct
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def _get_scalar_param(param_bounds: list, tensor_index: int, flat_index: int) -> float:
    return float(param_bounds[tensor_index].flatten()[flat_index].item())

def _cuboid_faces(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> list[list[tuple[float, float, float]]]:
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


def _union_surface_faces_from_cuboids(
    cuboids: list[tuple[float, float, float, float, float, float]],
) -> list[list[tuple[float, float, float]]]:
    """Build outer-surface faces of a union of axis-aligned cuboids."""
    valid = [(x0, x1, y0, y1, z0, z1) for (x0, x1, y0, y1, z0, z1) in cuboids if x1 > x0 and y1 > y0 and z1 > z0]
    if not valid:
        return []

    xs = sorted({x for x0, x1, _, _, _, _ in valid for x in (x0, x1)})
    ys = sorted({y for _, _, y0, y1, _, _ in valid for y in (y0, y1)})
    zs = sorted({z for _, _, _, _, z0, z1 in valid for z in (z0, z1)})
    if len(xs) < 2 or len(ys) < 2 or len(zs) < 2:
        return []

    x_idx = {x: i for i, x in enumerate(xs)}
    y_idx = {y: i for i, y in enumerate(ys)}
    z_idx = {z: i for i, z in enumerate(zs)}

    occ = np.zeros((len(xs) - 1, len(ys) - 1, len(zs) - 1), dtype=bool)
    for x0, x1, y0, y1, z0, z1 in valid:
        i0, i1 = x_idx[x0], x_idx[x1]
        j0, j1 = y_idx[y0], y_idx[y1]
        k0, k1 = z_idx[z0], z_idx[z1]
        occ[i0:i1, j0:j1, k0:k1] = True

    faces: list[list[tuple[float, float, float]]] = []
    nx, ny, nz = occ.shape
    filled = np.argwhere(occ)
    for i, j, k in filled:
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = ys[j], ys[j + 1]
        z0, z1 = zs[k], zs[k + 1]

        # xmin face
        if i == 0 or not occ[i - 1, j, k]:
            faces.append([(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)])
        # xmax face
        if i == nx - 1 or not occ[i + 1, j, k]:
            faces.append([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)])
        # ymin face
        if j == 0 or not occ[i, j - 1, k]:
            faces.append([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)])
        # ymax face
        if j == ny - 1 or not occ[i, j + 1, k]:
            faces.append([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)])
        # zmin face
        if k == 0 or not occ[i, j, k - 1]:
            faces.append([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)])
        # zmax face
        if k == nz - 1 or not occ[i, j, k + 1]:
            faces.append([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)])

    return faces


def _union_cells_from_rectangles(
    rectangles: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Decompose the union of axis-aligned rectangles into filled grid cells."""
    valid = [(x0, x1, y0, y1) for (x0, x1, y0, y1) in rectangles if x1 > x0 and y1 > y0]
    if not valid:
        return []

    xs = sorted({x for x0, x1, _, _ in valid for x in (x0, x1)})
    ys = sorted({y for _, _, y0, y1 in valid for y in (y0, y1)})
    if len(xs) < 2 or len(ys) < 2:
        return []

    x_idx = {x: i for i, x in enumerate(xs)}
    y_idx = {y: i for i, y in enumerate(ys)}

    occ = np.zeros((len(xs) - 1, len(ys) - 1), dtype=bool)
    for x0, x1, y0, y1 in valid:
        i0, i1 = x_idx[x0], x_idx[x1]
        j0, j1 = y_idx[y0], y_idx[y1]
        occ[i0:i1, j0:j1] = True

    cells: list[tuple[float, float, float, float]] = []
    for i, j in np.argwhere(occ):
        cells.append((xs[i], xs[i + 1], ys[j], ys[j + 1]))
    return cells


def plot_param_bounds(
    param_lower_bounds: list,
    param_upper_bounds: list,
    param_indices: list[tuple[int, int]],
    scatter_points: list[dict] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    legend_loc: str = "upper center",
    legend_bbox_to_anchor: tuple[float, float] = (0.5, -0.5),
    legend_ncol: int = 2,
):
    """Plot Rashomon bounds for 2 parameters (rectangle) or 3 parameters (cuboid)."""
    n_params = len(param_indices)
    assert n_params in (2, 3), "Expected parameter indices for either 2D or 3D plotting."

    bounds = []
    for tensor_idx, flat_idx in param_indices:
        lo = _get_scalar_param(param_lower_bounds, tensor_idx, flat_idx)
        hi = _get_scalar_param(param_upper_bounds, tensor_idx, flat_idx)
        bounds.append((lo, hi))

    labels = [f"param {idx}" for idx in param_indices]

    if n_params == 2:
        (x0, x1), (y0, y1) = bounds
        figsize = figsize or (5.0, 4.0)
        fig, ax = plt.subplots(figsize=figsize)

        if scatter_points is not None:
            # assert len(source_params) == 2, "source_params must have 2 values for 2D plots."
            for scatter_point_info in scatter_points:
                ax.scatter(
                    scatter_point_info['coordinates'][0],
                    scatter_point_info['coordinates'][1],
                    color=scatter_point_info.get('color', "tab:orange"),
                    marker="o",
                    s=60,
                    label=scatter_point_info.get('label', None),
                    zorder=5,
                )

        rect = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=True,
            facecolor="tab:blue",
            edgecolor="tab:blue",
            linewidth=1.5,
            alpha=0.2,
            label="Rashomon set",
        )
        ax.add_patch(rect)

        # x_pad = 0.05 * max(1e-12, x1 - x0)
        # y_pad = 0.05 * max(1e-12, y1 - y0)
        # ax.set_xlim(x0 - x_pad, x1 + x_pad)
        # ax.set_ylim(y0 - y_pad, y1 + y_pad)

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(title or f"Rashomon set: {labels[0]} vs {labels[1]}")
        ax.grid(alpha=0.25)
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, ncol=legend_ncol, framealpha=0.2)
        plt.tight_layout()
        plt.show()
        return

    (x0, x1), (y0, y1), (z0, z1) = bounds
    figsize = figsize or (6.0, 5.0)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    faces = _cuboid_faces(x0, x1, y0, y1, z0, z1)
    cuboid = Poly3DCollection(
        faces,
        facecolors="tab:blue",
        edgecolors="tab:blue",
        linewidths=1.0,
        alpha=0.2,
    )
    ax.add_collection3d(cuboid)

    if scatter_points is not None:
        for scatter_point_info in scatter_points:
            ax.scatter(
                scatter_point_info['coordinates'][0],
                scatter_point_info['coordinates'][1],
                scatter_point_info['coordinates'][2], # type: ignore
            color=scatter_point_info.get('color', "tab:orange"),
            marker="o",
            s=70,
            depthshade=False,
        )

    # x_pad = 0.05 * max(1e-12, x1 - x0)
    # y_pad = 0.05 * max(1e-12, y1 - y0)
    # z_pad = 0.05 * max(1e-12, z1 - z0)
    # ax.set_xlim(x0 - x_pad, x1 + x_pad)
    # ax.set_ylim(y0 - y_pad, y1 + y_pad)
    # ax.set_zlim(z0 - z_pad, z1 + z_pad)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(title or f"Rashomon set: {labels[0]} vs {labels[1]} vs {labels[2]}")

    legend_handles = [Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.2, label="Rashomon set")]
    if scatter_points is not None:
        for scatter_point_info in scatter_points:
            if 'label' in scatter_point_info and scatter_point_info['label'] is not None:
                legend_handles.append(
                    Line2D(
                        [0], [0], marker="o", color="w", 
                        markerfacecolor=scatter_point_info.get('color', "tab:orange"), markersize=8, label=scatter_point_info['label']
                    ) # type: ignore
                )
    ax.legend(
        handles=legend_handles, 
        loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, ncol=legend_ncol,
        framealpha=0.2
    )   

    # plt.tight_layout(rect=[0, 0.08, 1, 1])  # leave space at bottom for legend
    plt.tight_layout()
    plt.show()


def plot_param_bounds_multi_set(
    param_lower_bounds_sets: list[list[torch.Tensor]],
    param_upper_bounds_sets: list[list[torch.Tensor]],
    param_indices: list[tuple[int, int]],
    scatter_points: list[dict] | None = None,
    set_labels: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    alpha: float = 0.18,
    set_color: str | tuple[float, float, float] | tuple[float, float, float, float] | None = None,
    merge_cuboids: bool = False,
    legend: bool = True,
    legend_loc: str = "best",
    show: bool = True,
    ax: plt.Axes | None = None,
):
    """Plot multiple convex Rashomon sets in one 2D or 3D subplot.

    This is intended for visualizing the union returned by
    ``compute_nonconvex_rashomon_bounds`` where bounds are provided as:
    ``bounds[set_idx][param_tensor_idx]``.

    Args:
        param_lower_bounds_sets: Lower bounds per set/per parameter tensor.
        param_upper_bounds_sets: Upper bounds per set/per parameter tensor.
        param_indices: Selected parameter coordinates as ``(tensor_idx, flat_idx)``.
            Provide exactly 2 indices for 2D or 3 indices for 3D.
        scatter_points: Optional overlay points. Each dict should include
            ``coordinates`` (length 2 or 3), plus optional ``color``, ``marker``,
            ``size`` and ``label``.
        set_labels: Optional label per convex set.
        title: Optional plot title.
        figsize: Figure size when creating a new figure.
        alpha: Face alpha for rectangles/cuboids.
        set_color: Optional shared color for all Rashomon sets. If ``None``,
            sets use distinct colors from ``tab20``.
        merge_cuboids: If ``True``, render the union as one shape:
            - 2D: fills the union of rectangles without internal boundaries.
            - 3D: renders the outer union surface of all cuboids.
        legend: Whether to draw a legend.
        legend_loc: Legend location argument for Matplotlib.
        show: Whether to call ``plt.show()``.
        ax: Optional existing axis. If omitted, a new figure/axis is created.

    Returns:
        ``(fig, ax)`` for further customization.
    """
    n_params = len(param_indices)
    if n_params not in (2, 3):
        raise ValueError("plot_param_bounds_multi_set expects 2 or 3 parameter indices.")

    if len(param_lower_bounds_sets) != len(param_upper_bounds_sets):
        raise ValueError("Lower/upper set bounds lengths do not match.")
    n_sets = len(param_lower_bounds_sets)
    if n_sets == 0:
        raise ValueError("At least one convex Rashomon set is required.")

    if set_labels is not None and len(set_labels) != n_sets:
        raise ValueError(
            f"set_labels length mismatch: expected {n_sets}, got {len(set_labels)}.",
        )

    if scatter_points is not None:
        if any(len(p.get("coordinates", [])) != n_params for p in scatter_points):
            raise ValueError(
                "Each scatter point must have coordinates matching the selected parameter dimensionality.",
            )

    # Extract scalar bounds for selected coordinates for every set.
    set_bounds: list[list[tuple[float, float]]] = []
    for set_idx, (lower_set, upper_set) in enumerate(zip(param_lower_bounds_sets, param_upper_bounds_sets)):
        if len(lower_set) != len(upper_set):
            raise ValueError(
                f"Set {set_idx}: lower/upper tensor counts mismatch "
                f"(lower={len(lower_set)}, upper={len(upper_set)}).",
            )
        coord_bounds: list[tuple[float, float]] = []
        for tensor_idx, flat_idx in param_indices:
            lo = _get_scalar_param(lower_set, tensor_idx, flat_idx)
            hi = _get_scalar_param(upper_set, tensor_idx, flat_idx)
            if hi < lo:
                raise ValueError(
                    f"Set {set_idx}, param {(tensor_idx, flat_idx)} has invalid interval: "
                    f"lower={lo} > upper={hi}.",
                )
            coord_bounds.append((lo, hi))
        set_bounds.append(coord_bounds)

    mins = [min(b[p_idx][0] for b in set_bounds) for p_idx in range(n_params)]
    maxs = [max(b[p_idx][1] for b in set_bounds) for p_idx in range(n_params)]
    pads = [0.05 * max(1e-12, hi - lo) for lo, hi in zip(mins, maxs)]

    labels = [f"param {idx}" for idx in param_indices]
    labels_per_set = set_labels if set_labels is not None else [f"set #{i + 1}" for i in range(n_sets)]
    if set_color is None:
        colours = list(plt.cm.tab20(np.linspace(0, 1, max(2, n_sets))))
    else:
        colours = [set_color for _ in range(n_sets)]

    if n_params == 2:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (6.0, 5.0))
        else:
            fig = ax.figure

        if merge_cuboids:
            rectangles = [(b[0][0], b[0][1], b[1][0], b[1][1]) for b in set_bounds]
            union_cells = _union_cells_from_rectangles(rectangles)
            union_label = (
                "union of convex Rashomon sets"
                if set_labels is None
                else " | ".join(labels_per_set)
            )
            for idx, (x0, x1, y0, y1) in enumerate(union_cells):
                rect = Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=True,
                    edgecolor="none",
                    facecolor=colours[0],
                    linewidth=0.0,
                    alpha=alpha,
                    antialiased=False,
                    label=(union_label if idx == 0 else None),
                )
                ax.add_patch(rect)
        else:
            for i, bounds in enumerate(set_bounds):
                (x0, x1), (y0, y1) = bounds
                rect = Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=True,
                    edgecolor=colours[i],
                    facecolor=colours[i],
                    linewidth=1.4,
                    alpha=alpha,
                    label=labels_per_set[i],
                )
                ax.add_patch(rect)

        if scatter_points is not None:
            for p in scatter_points:
                ax.scatter(
                    p["coordinates"][0],
                    p["coordinates"][1],
                    color=p.get("color", "black"),
                    marker=p.get("marker", "o"),
                    s=p.get("size", 55),
                    label=p.get("label", None),
                    zorder=5,
                )

        ax.set_xlim(mins[0] - pads[0], maxs[0] + pads[0])
        ax.set_ylim(mins[1] - pads[1], maxs[1] + pads[1])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(title or "Convex Rashomon sets (2D projection)")
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

        if legend:
            handles, labels_found = ax.get_legend_handles_labels()
            deduped: dict[str, object] = {}
            for h, lbl in zip(handles, labels_found):
                if lbl and lbl not in deduped:
                    deduped[lbl] = h
            if deduped:
                ax.legend(deduped.values(), deduped.keys(), loc=legend_loc, framealpha=0.25)

        if show:
            plt.tight_layout()
            plt.show()
        return fig, ax

    # 3D mode
    if ax is None:
        fig = plt.figure(figsize=figsize or (7.0, 5.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
        if not hasattr(ax, "zaxis"):
            raise ValueError("For 3D plotting, provide a 3D axis (projection='3d').")

    if merge_cuboids:
        cuboids = [
            (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1], bounds[2][0], bounds[2][1])
            for bounds in set_bounds
        ]
        union_faces = _union_surface_faces_from_cuboids(cuboids)
        if len(union_faces) > 0:
            union_poly = Poly3DCollection(
                union_faces,
                facecolors=colours[0],
                edgecolors="none",
                linewidths=0.0,
                alpha=alpha,
            )
            ax.add_collection3d(union_poly)
    else:
        for i, bounds in enumerate(set_bounds):
            (x0, x1), (y0, y1), (z0, z1) = bounds
            faces = _cuboid_faces(x0, x1, y0, y1, z0, z1)
            cuboid = Poly3DCollection(
                faces,
                facecolors=colours[i],
                edgecolors=colours[i],
                linewidths=1.0,
                alpha=alpha,
            )
            ax.add_collection3d(cuboid)

    scatter_handles: list[Line2D] = []
    if scatter_points is not None:
        for p in scatter_points:
            ax.scatter(
                p["coordinates"][0],
                p["coordinates"][1],
                p["coordinates"][2],  # type: ignore[index]
                color=p.get("color", "black"),
                marker=p.get("marker", "o"),
                s=p.get("size", 60),
                depthshade=False,
            )
            label = p.get("label", None)
            if label is not None:
                scatter_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=p.get("marker", "o"),
                        color="w",
                        markerfacecolor=p.get("color", "black"),
                        markersize=8,
                        label=label,
                    ),
                )

    ax.set_xlim(mins[0] - pads[0], maxs[0] + pads[0])
    ax.set_ylim(mins[1] - pads[1], maxs[1] + pads[1])
    ax.set_zlim(mins[2] - pads[2], maxs[2] + pads[2])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(title or "Convex Rashomon sets (3D projection)")

    if legend:
        if merge_cuboids:
            set_handles = [
                Patch(
                    facecolor=colours[0],
                    edgecolor=colours[0],
                    alpha=alpha,
                    label=("union of convex Rashomon sets" if set_labels is None else " | ".join(labels_per_set)),
                ),
            ]
        else:
            set_handles = [
                Patch(facecolor=colours[i], edgecolor=colours[i], alpha=alpha, label=labels_per_set[i])
                for i in range(n_sets)
            ]
        legend_handles = set_handles + scatter_handles
        deduped: dict[str, object] = {}
        for h in legend_handles:
            lbl = h.get_label()  # type: ignore[union-attr]
            if lbl and lbl not in deduped:
                deduped[lbl] = h
        if deduped:
            ax.legend(deduped.values(), deduped.keys(), loc=legend_loc, framealpha=0.25)

    if show:
        plt.tight_layout()
        plt.show()
    return fig, ax


def _select_checkpoint_indices(n_ckpts: int, num_checkpoints_to_plot: int) -> list[int]:
    """Select checkpoint indices using first/last/linspace rules."""
    if n_ckpts <= 0:
        raise ValueError("No checkpoints are available to plot.")

    if num_checkpoints_to_plot <= 1:
        return [n_ckpts - 1]

    if n_ckpts == 1:
        return [0]

    if num_checkpoints_to_plot == 2:
        return [0, n_ckpts - 1]

    num = min(num_checkpoints_to_plot, n_ckpts)
    linspace_indices = np.linspace(0, n_ckpts - 1, num=num)
    selected = [int(round(v)) for v in linspace_indices]
    selected[0] = 0
    selected[-1] = n_ckpts - 1

    deduped: list[int] = []
    for idx in selected:
        if idx not in deduped:
            deduped.append(idx)

    return deduped


def _extract_param_bounds_per_checkpoint(
    param_bounds_l_per_checkpoint: list[list],
    param_bounds_u_per_checkpoint: list[list],
    tensor_idx: int,
    flat_idx: int,
    selected_plot_indices: list[int],
) -> list[dict[str, float]]:
    return [
        {
            "lower": param_bounds_l_per_checkpoint[i][tensor_idx].flatten()[flat_idx].item(),
            "upper": param_bounds_u_per_checkpoint[i][tensor_idx].flatten()[flat_idx].item(),
        }
        for i in selected_plot_indices
    ]


def _cuboid_faces(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> list[list[tuple[float, float, float]]]:
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


def plot_param_bounds_per_checkpoint(
    param_bounds_l_per_checkpoint: list[list],
    param_bounds_u_per_checkpoint: list[list],
    param_indices: list[tuple[int, int]],
    scatter_points: list[list[dict]] | None = None,
    # source_params: tuple[float, float] | tuple[float, float, float] | None = None,
    # source_label: str = "Source policy",
    num_checkpoints_to_plot: int = 5,
    n_rows: int = 1,
    figsize: tuple[float, float] | None = None,
):
    """Plot 2D/3D Rashomon sets per checkpoint for 2 or 3 selected parameters."""
    n_params = len(param_indices)
    if n_params not in (2, 3):
        raise ValueError("plot_param_bounds_per_checkpoint expects 2 or 3 parameter indices.")

    if len(param_bounds_l_per_checkpoint) != len(param_bounds_u_per_checkpoint):
        raise ValueError("Lower/upper checkpoint bounds lengths do not match.")

    n_ckpts = len(param_bounds_l_per_checkpoint)
    if n_ckpts == 0:
        raise ValueError("No checkpoint bounds were provided.")

    if scatter_points is not None:
        assert all (len(dct['coordinates']) == n_params for dct in scatter_points), "Each scatter point must have coordinates matching the number of parameters."

    selected_plot_indices = _select_checkpoint_indices(n_ckpts, num_checkpoints_to_plot)
    n_plots = len(selected_plot_indices)
    n_cols = math.ceil(n_plots / n_rows)

    param_bounds_to_plot: list[list[dict[str, float]]] = []
    for tensor_idx, flat_idx in param_indices:
        param_bounds_to_plot.append(
            _extract_param_bounds_per_checkpoint(
                param_bounds_l_per_checkpoint,
                param_bounds_u_per_checkpoint,
                tensor_idx,
                flat_idx,
                selected_plot_indices,
            )
        )

    mins = [min(b["lower"] for b in param_bounds) for param_bounds in param_bounds_to_plot]
    maxs = [max(b["upper"] for b in param_bounds) for param_bounds in param_bounds_to_plot]
    pads = [0.02 * max(1e-12, hi - lo) for lo, hi in zip(mins, maxs)]

    if n_params == 2:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(2.8 * n_cols, 3.2 * n_rows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
    else:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.3 * n_cols, 3.2 * n_rows),
            squeeze=False,
            subplot_kw={"projection": "3d"},
            # constrained_layout=True,
        )

    fig.suptitle(
        "Rashomon set across checkpoints\n"
        + ", ".join([f"param{i + 1}={param_indices[i]}" for i in range(n_params)]),
        fontsize=13,
        y=1.03,
    )

    for i, ax in enumerate(axes.flat):
        if i >= n_plots:
            ax.axis("off")
            continue

        ckpt_label = selected_plot_indices[i] + 1

        if n_params == 2:
            xb = param_bounds_to_plot[0][i]
            yb = param_bounds_to_plot[1][i]

            if scatter_points is not None:
                for scatter_point_info in scatter_points:
                    ax.scatter(
                        scatter_point_info['coordinates'][0],
                        scatter_point_info['coordinates'][1],
                        color=scatter_point_info.get('color', "tab:orange"),
                        marker="o",
                        s=50,
                        label=scatter_point_info.get('label', None) if i == 0 else None,
                        zorder=5,
                    )

            rect = Rectangle(
                (xb["lower"], yb["lower"]),
                xb["upper"] - xb["lower"],
                yb["upper"] - yb["lower"],
                fill=True,
                edgecolor="tab:blue",
                facecolor="tab:blue",
                linewidth=1.5,
                alpha=0.2,
                label="Rashomon set" if i == 0 else None,
            )
            ax.add_patch(rect)

            ax.set_xlim(mins[0] - pads[0], maxs[0] + pads[0])
            ax.set_ylim(mins[1] - pads[1], maxs[1] + pads[1])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.25)
            ax.set_title(f"checkpoint #{ckpt_label}", fontsize=9)

            if i // n_cols == n_rows - 1:
                ax.set_xlabel(f"param1 {param_indices[0]}", fontsize=10)
            if i % n_cols == 0:
                ax.set_ylabel(f"param2 {param_indices[1]}", fontsize=10)

        else:
            xb = param_bounds_to_plot[0][i]
            yb = param_bounds_to_plot[1][i]
            zb = param_bounds_to_plot[2][i]

            faces = _cuboid_faces(
                xb["lower"], xb["upper"],
                yb["lower"], yb["upper"],
                zb["lower"], zb["upper"],
            )
            cuboid = Poly3DCollection(
                faces,
                facecolors="tab:blue",
                edgecolors="tab:blue",
                linewidths=1.0,
                alpha=0.2,
            )
            ax.add_collection3d(cuboid)

            if scatter_points is not None:
                for scatter_point_info in scatter_points:
                    ax.scatter(
                        scatter_point_info['coordinates'][0],
                        scatter_point_info['coordinates'][1],
                        scatter_point_info['coordinates'][2], # type: ignore
                        color=scatter_point_info.get('color', "tab:orange"),
                    marker="o",
                    s=45,
                    depthshade=False,
                )

            ax.set_xlim(mins[0] - pads[0], maxs[0] + pads[0])
            ax.set_ylim(mins[1] - pads[1], maxs[1] + pads[1])
            ax.set_zlim(mins[2] - pads[2], maxs[2] + pads[2])
            ax.set_title(f"checkpoint #{ckpt_label}", fontsize=9)
            ax.set_xlabel(f"param1 {param_indices[0]}", fontsize=9)
            ax.set_ylabel(f"param2 {param_indices[1]}", fontsize=9)
            ax.set_zlabel(f"param3 {param_indices[2]}", fontsize=9, labelpad=4)
            ax.tick_params(axis="z", pad=1)
            ax.set_box_aspect((1, 1, 1))  # keep 3D box compact and balanced

    legend_handles = [Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.2, label="Rashomon set")]
    if scatter_points is not None:
        for scatter_point_info in scatter_points:
            if 'label' in scatter_point_info and scatter_point_info['label'] is not None:
                legend_handles.insert(
                    0,
                    Line2D(
                        [0], [0], marker="o", color="w", 
                        markerfacecolor=scatter_point_info.get('color', "tab:orange"),
                        markersize=7, label=scatter_point_info.get('label', None)
                    ) # type: ignore
        )
    legend_labels = [h.get_label() for h in legend_handles]
    fig.legend(
        legend_handles, 
        legend_labels, # type: ignore
        loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8
    )

    if n_params ==2:
        fig.tight_layout()
    # else:
    #     pass
    #     # fig.subplots_adjust(right=0.9, bottom=0.14)  # reserve room for right edge + legend
    plt.show()


def _figure_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    """Convert a Matplotlib figure canvas into an RGB uint8 array."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Expected HxWx4 RGBA canvas buffer, got shape {rgba.shape}.")
    return np.ascontiguousarray(rgba[..., :3]).astype(np.uint8)


def _axis_pad(lower: float, upper: float, ratio: float = 0.05) -> float:
    """Return a stable axis padding, including near-zero-span intervals."""
    span = upper - lower
    if abs(span) < 1e-12:
        return max(1e-6, ratio * max(abs(lower), abs(upper), 1.0))
    return ratio * span


def create_rashomon_set_video(
    param_bounds_l_per_checkpoint: list[list[torch.Tensor]],
    param_bounds_u_per_checkpoint: list[list[torch.Tensor]],
    param_indices: list[tuple[int, int]],
    output_path: str,
    scatter_points: list[dict] | None = None,
    checkpoint_indices: list[int] | None = None,
    checkpoint_labels: list[int | str] | None = None,
    num_checkpoints_to_plot: int | None = None,
    fps: int = 2,
    dpi: int = 140,
    fig_size_2d: tuple[float, float] = (5.0, 5.0),
    fig_size_3d: tuple[float, float] = (6.0, 5.0),
    elev: float = 20.0,
    azim: float = -60.0,
    title_prefix: str = "Rashomon set",
    repeat_last_frame: int = 0,
) -> str:
    """Create a 2D/3D Rashomon-set evolution video across checkpoints.

    The function mirrors notebook visuals:
    - 2D: one rectangle per checkpoint.
    - 3D: one cuboid per checkpoint.
    Optional scatter points (e.g., source policy params) are overlaid in all frames.

    Args:
        param_bounds_l_per_checkpoint: Lower bounds for each checkpoint.
        param_bounds_u_per_checkpoint: Upper bounds for each checkpoint.
        param_indices: Parameter indices to visualize (2 or 3 parameters).
        output_path: Video path (e.g. ``.mp4`` or ``.gif``).
        scatter_points: Optional static scatter points over all frames.
        checkpoint_indices: Optional explicit checkpoint indices to render.
        checkpoint_labels: Optional labels for all checkpoints; used in frame titles.
        num_checkpoints_to_plot: Optional evenly-spaced checkpoint count.
        fps: Output frames per second.
        dpi: Figure DPI for frame rendering.
        fig_size_2d: Frame figure size for 2D mode.
        fig_size_3d: Frame figure size for 3D mode.
        elev: 3D camera elevation.
        azim: 3D camera azimuth.
        title_prefix: Prefix for frame titles.
        repeat_last_frame: Number of extra repeats for the final frame.

    Returns:
        Absolute output path of the written video file.
    """
    import imageio.v2 as imageio

    n_params = len(param_indices)
    if n_params not in (2, 3):
        raise ValueError("create_rashomon_set_video expects 2 or 3 parameter indices.")

    if len(param_bounds_l_per_checkpoint) != len(param_bounds_u_per_checkpoint):
        raise ValueError("Lower/upper checkpoint bounds lengths do not match.")

    n_ckpts = len(param_bounds_l_per_checkpoint)
    if n_ckpts == 0:
        raise ValueError("No checkpoint bounds were provided.")

    if checkpoint_labels is not None and len(checkpoint_labels) != n_ckpts:
        raise ValueError("checkpoint_labels must be None or have one label per checkpoint.")

    if scatter_points is not None:
        if any(len(dct["coordinates"]) != n_params for dct in scatter_points):
            raise ValueError("Each scatter point must match the parameter dimensionality (2D/3D).")

    if checkpoint_indices is not None and num_checkpoints_to_plot is not None:
        raise ValueError("Provide either checkpoint_indices or num_checkpoints_to_plot, not both.")

    if checkpoint_indices is not None:
        selected_plot_indices = list(dict.fromkeys(checkpoint_indices))
        for idx in selected_plot_indices:
            if idx < 0 or idx >= n_ckpts:
                raise IndexError(f"checkpoint index {idx} out of range for {n_ckpts} checkpoints.")
    elif num_checkpoints_to_plot is not None:
        selected_plot_indices = _select_checkpoint_indices(n_ckpts, num_checkpoints_to_plot)
    else:
        selected_plot_indices = list(range(n_ckpts))

    if len(selected_plot_indices) == 0:
        raise ValueError("No checkpoints selected for rendering.")

    param_bounds_to_plot: list[list[dict[str, float]]] = []
    for tensor_idx, flat_idx in param_indices:
        param_bounds_to_plot.append(
            _extract_param_bounds_per_checkpoint(
                param_bounds_l_per_checkpoint=param_bounds_l_per_checkpoint,
                param_bounds_u_per_checkpoint=param_bounds_u_per_checkpoint,
                tensor_idx=tensor_idx,
                flat_idx=flat_idx,
                selected_plot_indices=selected_plot_indices,
            )
        )

    mins = [min(frame_bounds["lower"] for frame_bounds in per_param) for per_param in param_bounds_to_plot]
    maxs = [max(frame_bounds["upper"] for frame_bounds in per_param) for per_param in param_bounds_to_plot]
    pads = [_axis_pad(lo, hi) for lo, hi in zip(mins, maxs)]

    frames: list[np.ndarray] = []
    for frame_idx, ckpt_idx in enumerate(selected_plot_indices):
        ckpt_label: int | str
        if checkpoint_labels is None:
            ckpt_label = ckpt_idx + 1
        else:
            ckpt_label = checkpoint_labels[ckpt_idx]

        if n_params == 2:
            fig, ax = plt.subplots(figsize=fig_size_2d, dpi=dpi)
            xb = param_bounds_to_plot[0][frame_idx]
            yb = param_bounds_to_plot[1][frame_idx]

            if scatter_points is not None:
                for scatter_point_info in scatter_points:
                    ax.scatter(
                        scatter_point_info["coordinates"][0],
                        scatter_point_info["coordinates"][1],
                        color=scatter_point_info.get("color", "tab:orange"),
                        marker="o",
                        s=55,
                        label=scatter_point_info.get("label", None),
                        zorder=5,
                    )

            rect = Rectangle(
                (xb["lower"], yb["lower"]),
                xb["upper"] - xb["lower"],
                yb["upper"] - yb["lower"],
                fill=True,
                edgecolor="tab:blue",
                facecolor="tab:blue",
                linewidth=1.5,
                alpha=0.2,
                label="Rashomon set",
            )
            ax.add_patch(rect)
            ax.set_xlim(mins[0] - pads[0], maxs[0] + pads[0])
            ax.set_ylim(mins[1] - pads[1], maxs[1] + pads[1])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.25)
            ax.set_xlabel(f"param1 {param_indices[0]}")
            ax.set_ylabel(f"param2 {param_indices[1]}")
            ax.set_title(f"{title_prefix} - checkpoint #{ckpt_label}")

            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                deduped: dict[str, object] = {}
                for handle, label in zip(handles, labels):
                    if label not in deduped:
                        deduped[label] = handle
                ax.legend(deduped.values(), deduped.keys(), loc="best")
        else:
            fig = plt.figure(figsize=fig_size_3d, dpi=dpi)
            ax = fig.add_subplot(111, projection="3d")
            xb = param_bounds_to_plot[0][frame_idx]
            yb = param_bounds_to_plot[1][frame_idx]
            zb = param_bounds_to_plot[2][frame_idx]

            faces = _cuboid_faces(
                xb["lower"], xb["upper"],
                yb["lower"], yb["upper"],
                zb["lower"], zb["upper"],
            )
            cuboid = Poly3DCollection(
                faces,
                facecolors="tab:blue",
                edgecolors="tab:blue",
                linewidths=1.0,
                alpha=0.2,
            )
            ax.add_collection3d(cuboid)

            legend_handles: list[object] = [
                Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.2, label="Rashomon set"),
            ]
            if scatter_points is not None:
                for scatter_point_info in scatter_points:
                    ax.scatter(
                        scatter_point_info["coordinates"][0],
                        scatter_point_info["coordinates"][1],
                        scatter_point_info["coordinates"][2],  # type: ignore[index]
                        color=scatter_point_info.get("color", "tab:orange"),
                        marker="o",
                        s=55,
                        depthshade=False,
                    )
                    label = scatter_point_info.get("label", None)
                    if label is not None:
                        legend_handles.insert(
                            0,
                            Line2D(
                                [0], [0], marker="o", color="w",
                                markerfacecolor=scatter_point_info.get("color", "tab:orange"),
                                markersize=8,
                                label=label,
                            ),
                        )

            ax.set_xlim(mins[0] - pads[0], maxs[0] + pads[0])
            ax.set_ylim(mins[1] - pads[1], maxs[1] + pads[1])
            ax.set_zlim(mins[2] - pads[2], maxs[2] + pads[2])
            ax.set_xlabel(f"param1 {param_indices[0]}")
            ax.set_ylabel(f"param2 {param_indices[1]}")
            ax.set_zlabel(f"param3 {param_indices[2]}", labelpad=4)
            ax.tick_params(axis="z", pad=1)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{title_prefix} - checkpoint #{ckpt_label}")

            legend_labels = [handle.get_label() for handle in legend_handles]  # type: ignore[attr-defined]
            ax.legend(legend_handles, legend_labels, loc="best")  # type: ignore[arg-type]

        fig.tight_layout()
        frames.append(_figure_to_rgb_array(fig))
        plt.close(fig)

    if repeat_last_frame > 0 and len(frames) > 0:
        frames.extend([frames[-1]] * repeat_last_frame)

    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(out_path, frames, fps=fps, loop=0)
    else:
        imageio.mimwrite(out_path, frames, fps=fps, macro_block_size=1)

    return str(out_path.resolve())


def create_video_param_bounds_per_checkpoint(
    param_bounds_l_per_checkpoint: list[list[torch.Tensor]],
    param_bounds_u_per_checkpoint: list[list[torch.Tensor]],
    param_indices: list[tuple[int, int]],
    output_path: str,
    scatter_points: list[dict] | None = None,
    checkpoint_labels: list[int | str] | None = None,
    fps: int = 2,
    dpi: int = 140,
    fig_size_2d: tuple[float, float] = (5.0, 5.0),
    fig_size_3d: tuple[float, float] = (6.0, 5.0),
    elev: float = 20.0,
    azim: float = -60.0,
    title_prefix: str = "Rashomon parameter bounds",
    repeat_last_frame: int = 0,
) -> str:
    """Create a checkpoint-by-checkpoint parameter-bounds video.

    This helper always renders *all* checkpoints provided in
    ``param_bounds_l_per_checkpoint`` / ``param_bounds_u_per_checkpoint``.
    It supports both:
    - 2D visualisations (2 parameter indices), and
    - 3D visualisations (3 parameter indices).

    Args:
        param_bounds_l_per_checkpoint: Lower bounds per checkpoint.
        param_bounds_u_per_checkpoint: Upper bounds per checkpoint.
        param_indices: Parameter indices to visualise (length 2 or 3).
        output_path: Output video path (e.g. ``.mp4`` / ``.gif``).
        scatter_points: Optional static points overlaid in every frame.
        checkpoint_labels: Optional labels for each checkpoint.
        fps: Output frames per second.
        dpi: Figure DPI for rendering frames.
        fig_size_2d: Figure size for 2D frames.
        fig_size_3d: Figure size for 3D frames.
        elev: 3D elevation angle.
        azim: 3D azimuth angle.
        title_prefix: Prefix for per-frame titles.
        repeat_last_frame: Number of additional repeats for the final frame.

    Returns:
        Absolute path to the written video file.
    """
    if len(param_bounds_l_per_checkpoint) != len(param_bounds_u_per_checkpoint):
        raise ValueError("Lower/upper checkpoint bounds lengths do not match.")
    if len(param_bounds_l_per_checkpoint) == 0:
        raise ValueError("No checkpoint bounds were provided.")

    checkpoint_indices = list(range(len(param_bounds_l_per_checkpoint)))
    return create_rashomon_set_video(
        param_bounds_l_per_checkpoint=param_bounds_l_per_checkpoint,
        param_bounds_u_per_checkpoint=param_bounds_u_per_checkpoint,
        param_indices=param_indices,
        output_path=output_path,
        scatter_points=scatter_points,
        checkpoint_indices=checkpoint_indices,
        checkpoint_labels=checkpoint_labels,
        num_checkpoints_to_plot=None,
        fps=fps,
        dpi=dpi,
        fig_size_2d=fig_size_2d,
        fig_size_3d=fig_size_3d,
        elev=elev,
        azim=azim,
        title_prefix=title_prefix,
        repeat_last_frame=repeat_last_frame,
    )


def plot_parameter_bound_widths(
    param_bounds_l: list[torch.Tensor],
    param_bounds_u: list[torch.Tensor],
    layer_names: list[str] | None = None,
    title: str | None = "SafeAdapt Parameter Bound Widths",
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
    log_scale: bool = False,
):
    """Analyse and plot the interval widths of SafeAdapt parameter bounds.

    For each layer the function computes ``width = param_bounds_u − param_bounds_l``
    and displays:

    * **Top-left** – per-layer box-plot of widths (outliers shown as dots).
    * **Top-right** – per-layer histogram of widths (overlaid, semi-transparent).
    * **Bottom-left** – per-layer summary statistics table (mean, std, min, max,
      median, % zero-width).
    * **Bottom-right** – cumulative distribution of widths across all layers.

    Args:
        param_bounds_l: Lower parameter bounds (list of tensors, one per layer).
        param_bounds_u: Upper parameter bounds (same structure).
        layer_names: Optional human-readable names for each layer.  When
            *None*, layers are labelled automatically as ``W0`` / ``b0`` etc.
            based on tensor dimensionality (≥ 2-D → weight, 1-D → bias).
        title: Optional suptitle for the figure.
        figsize: ``(width, height)`` in inches.  Defaults to ``(16, 10)``.
        save_path: If given, save the figure to this path.
        log_scale: If *True*, use a log scale on the width axis where
            applicable (box-plot and CDF).

    Returns:
        ``(fig, stats_df)`` — the matplotlib Figure and a
        ``pandas.DataFrame`` of per-layer summary statistics.
    """
    import pandas as pd

    assert len(param_bounds_l) == len(param_bounds_u), (
        "param_bounds_l and param_bounds_u must have the same length"
    )

    # ------------------------------------------------------------------
    # Compute widths & auto-generate layer names
    # ------------------------------------------------------------------
    widths_per_layer: list[np.ndarray] = []
    auto_names: list[str] = []
    w_idx, b_idx = 0, 0

    for lb, ub in zip(param_bounds_l, param_bounds_u):
        w = (ub - lb).detach().cpu().numpy().ravel()
        widths_per_layer.append(w)
        if lb.ndim >= 2:
            auto_names.append(f"W{w_idx}")
            w_idx += 1
        else:
            auto_names.append(f"b{b_idx}")
            b_idx += 1

    names = layer_names if layer_names is not None else auto_names
    n_layers = len(names)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    stats_rows = []
    for name, w in zip(names, widths_per_layer):
        stats_rows.append({
            "layer": name,
            "n_params": len(w),
            "mean": float(np.mean(w)),
            "std": float(np.std(w)),
            "min": float(np.min(w)),
            "median": float(np.median(w)),
            "max": float(np.max(w)),
            "pct_zero": float((w == 0).sum() / len(w) * 100),
        })
    stats_df = pd.DataFrame(stats_rows)

    # ------------------------------------------------------------------
    # Figure layout: 2×2 grid
    # ------------------------------------------------------------------
    if figsize is None:
        figsize = (16, max(10, 2.5 * n_layers))

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # ── (0, 0) Box-plot ────────────────────────────────────────────────
    ax_box = fig.add_subplot(gs[0, 0])
    bp = ax_box.boxplot(
        widths_per_layer,
        vert=False,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.4),
        labels=names,
    )
    colours = plt.cm.tab10(np.linspace(0, 1, n_layers))
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.6)
    ax_box.set_xlabel("Interval width")
    ax_box.set_title("Per-layer bound widths (box-plot)")
    if log_scale:
        ax_box.set_xscale("symlog", linthresh=1e-8)

    # ── (0, 1) Overlaid histograms ────────────────────────────────────
    ax_hist = fig.add_subplot(gs[0, 1])
    for i, (name, w) in enumerate(zip(names, widths_per_layer)):
        if w.max() - w.min() < 1e-12:
            ax_hist.axvline(w[0], color=colours[i], label=f"{name} (const)", lw=1.5)
        else:
            ax_hist.hist(w, bins=50, alpha=0.45, color=colours[i], label=name, edgecolor="none")
    ax_hist.set_xlabel("Interval width")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Per-layer bound width distributions")
    ax_hist.legend(fontsize=8, loc="upper right")
    if log_scale:
        ax_hist.set_xscale("symlog", linthresh=1e-8)

    # ── (1, 0) Statistics table ───────────────────────────────────────
    ax_table = fig.add_subplot(gs[1, 0])
    ax_table.axis("off")
    col_labels = ["Layer", "#Params", "Mean", "Std", "Min", "Median", "Max", "% Zero"]
    cell_text = []
    for r in stats_rows:
        cell_text.append([
            r["layer"],
            f"{r['n_params']:,}",
            f"{r['mean']:.2e}",
            f"{r['std']:.2e}",
            f"{r['min']:.2e}",
            f"{r['median']:.2e}",
            f"{r['max']:.2e}",
            f"{r['pct_zero']:.1f}",
        ])
    tbl = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    # Header style
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#d9e2f3")
    ax_table.set_title("Summary statistics", fontsize=11, pad=12)

    # ── (1, 1) CDF across all layers ─────────────────────────────────
    ax_cdf = fig.add_subplot(gs[1, 1])
    for i, (name, w) in enumerate(zip(names, widths_per_layer)):
        sorted_w = np.sort(w)
        cdf = np.arange(1, len(sorted_w) + 1) / len(sorted_w)
        ax_cdf.plot(sorted_w, cdf, label=name, color=colours[i], lw=1.5)
    ax_cdf.set_xlabel("Interval width")
    ax_cdf.set_ylabel("Cumulative fraction")
    ax_cdf.set_title("CDF of bound widths")
    ax_cdf.legend(fontsize=8, loc="lower right")
    ax_cdf.set_ylim(0, 1.02)
    if log_scale:
        ax_cdf.set_xscale("symlog", linthresh=1e-8)

    # ------------------------------------------------------------------
    # Suptitle & save
    # ------------------------------------------------------------------
    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig, stats_df
