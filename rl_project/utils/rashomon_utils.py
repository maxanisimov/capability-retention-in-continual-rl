import os
import numpy as np
import torch
import matplotlib.pyplot as plt

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