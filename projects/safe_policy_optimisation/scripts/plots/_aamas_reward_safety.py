"""Compact AAMAS reward/safety figures without changing other venues' styles.

The official AAMAS 2027 sigconf template has ~7.006 inches of text width and
~3.337 inches per column. Render at native size, not oversized and scaled down.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

FULL_WIDTH_IN = 7.0
COLUMN_WIDTH_IN = 3.33
GRID_HEIGHT_IN = 2.65
SINGLE_HEIGHT_IN = 2.35
TRANSPOSED_ROW_HEIGHT_IN = 0.49
TRANSPOSED_MARGIN_HEIGHT_IN = 0.76
TRANSPOSED_SINGLE_HEIGHT_IN = 2.20
ENVIRONMENT_HEADING_X = 0.57  # Seven percent of figure width right of centre.
ENVIRONMENT_ORDER = (
    "Media Streaming", "Colour Bomb v1", "Colour Bomb v2",
    "Bridge Crossing v1", "Bridge Crossing v2", "MiniPacman",
)


@dataclass(frozen=True)
class BarMethod:
    key: str
    label: str
    color: str
    hatch: str | None = None


@dataclass(frozen=True)
class BarPanel:
    label: str
    reward_means: list[float | None]
    reward_errors: list[float]
    safety_means: list[float | None]
    safety_errors: list[float]


def canonical_label(label: str) -> str:
    return {"Colour Bomb": "Colour Bomb v1", "Bridge Crossing": "Bridge Crossing v1",
            "Mini Pacman": "MiniPacman"}.get(label, label)


def ordered_panels(panels: list[BarPanel]) -> list[BarPanel]:
    order = {label: position for position, label in enumerate(ENVIRONMENT_ORDER)}
    return sorted(panels, key=lambda panel: order.get(canonical_label(panel.label), len(order)))


def panel_title(label: str) -> str:
    label = canonical_label(label)
    if label.endswith((" v1", " v2")):
        name, version = label.rsplit(" ", 1)
        return f"{name}\n{version}"
    return "Media\nStreaming" if label == "Media Streaming" else label


def apply_aamas_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.55, "axes.grid": False,
        "xtick.major.width": 0.55, "ytick.major.width": 0.55,
        "ytick.major.size": 2, "ytick.major.pad": 2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.5,
        "savefig.bbox": None,
    })


def reward_bottom(means: list[float | None], errors: list[float]) -> float:
    lower = min(mean - error for mean, error in zip(means, errors) if mean is not None)
    return lower - 0.05 * (abs(lower) if abs(lower) > 1e-9 else 1.0)


def legend_indices(count: int, columns: int) -> list[int]:
    """Matplotlib fills columns first; retain the bars' row-major order."""
    rows = math.ceil(count / columns)
    return [row * columns + column for column in range(columns)
            for row in range(rows) if row * columns + column < count]


def compact_figure(
    panels: list[BarPanel], methods: list[BarMethod], *, single_column: bool = False,
    title: str | None = None, transpose: bool = False, legend_one_row: bool = False,
):
    if not panels or not methods:
        raise ValueError("Need at least one panel and method")
    if single_column and len(panels) != 1 and not transpose:
        raise ValueError("Single-column layout is reserved for one task")
    if legend_one_row and (not transpose or len(panels) < 2):
        raise ValueError("One-row legend requires a transposed multi-task figure")
    panels = ordered_panels(panels)
    for panel in panels:
        arrays = (panel.reward_means, panel.reward_errors, panel.safety_means, panel.safety_errors)
        if any(len(array) != len(methods) for array in arrays):
            raise ValueError(f"Wrong method count in {panel.label}")
        for array in arrays:
            if any(value is not None and not math.isfinite(value) for value in array):
                raise ValueError(f"Non-finite metric in {panel.label}")
        if any(error < 0 for error in panel.reward_errors + panel.safety_errors):
            raise ValueError(f"Negative error bar in {panel.label}")
        for metric, means in (("reward", panel.reward_means), ("safety", panel.safety_means)):
            if all(mean is None for mean in means):
                raise ValueError(f"No {metric} data in {panel.label}")
    apply_aamas_style()
    transposed_single = transpose and len(panels) == 1
    if transposed_single:
        fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, TRANSPOSED_SINGLE_HEIGHT_IN),
                                 squeeze=False)
        fig.subplots_adjust(left=0.14, right=0.59, bottom=0.12, top=0.80,
                            hspace=0.65)
    elif transpose:
        height = TRANSPOSED_ROW_HEIGHT_IN * len(panels) + TRANSPOSED_MARGIN_HEIGHT_IN
        legend_space = 0.65
        if legend_one_row:
            # Reclaim two legend rows without changing panel heights or typography.
            height -= 0.26
            legend_space -= 0.26
        # Reserve a fourth legend row when all seven legacy methods are used.
        if len(methods) > 6 and not legend_one_row:
            height += 0.13
            legend_space += 0.13
        fig, axes = plt.subplots(len(panels), 2, figsize=(COLUMN_WIDTH_IN, height),
                                 squeeze=False)
        fig.subplots_adjust(left=0.12, right=0.99, bottom=legend_space / height,
                            top=1 - 0.42 / height, wspace=0.22, hspace=0.68)
    elif single_column:
        fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN, SINGLE_HEIGHT_IN), squeeze=False)
        fig.subplots_adjust(left=0.155, right=0.99, bottom=0.285, top=0.79,
                            wspace=0.22)
    else:
        fig, axes = plt.subplots(2, len(panels), figsize=(FULL_WIDTH_IN, GRID_HEIGHT_IN), squeeze=False)
        fig.subplots_adjust(left=0.069, right=0.993, bottom=0.235, top=0.83,
                            wspace=0.53, hspace=0.40)
    for column, panel in enumerate(panels):
        for metric in range(2):
            if transposed_single:
                axis = axes[metric, 0]
            elif transpose:
                axis = axes[column, metric]
            else:
                axis = axes[0, metric] if single_column else axes[metric, column]
            means, errors = ((panel.reward_means, panel.reward_errors) if metric == 0
                             else (panel.safety_means, panel.safety_errors))
            bottom = reward_bottom(means, errors) if metric == 0 else 0.0
            upper = max(mean + error for mean, error in zip(means, errors) if mean is not None)
            if metric == 1:
                upper = max(1.0, upper)
            span = max(upper - bottom, 0.01)
            for position, method in enumerate(methods):
                mean, error = means[position], errors[position]
                if mean is None:
                    axis.text(position, bottom + 0.035 * span, "n/a", rotation=90,
                              ha="center", va="bottom", fontsize=7)
                    continue
                axis.bar(position, mean - bottom, bottom=bottom, yerr=error, width=0.76,
                         capsize=1.5 if single_column or transpose else 1.0,
                         color=method.color, edgecolor="#252525", linewidth=0.35,
                         hatch=method.hatch,
                         error_kw={"elinewidth": 0.65, "capthick": 0.65}, zorder=3)
            # Never clip uncertainty above 1.0; the *means* remain probabilities.
            low = min([0.0] + [mean - error for mean, error in zip(means, errors)
                              if mean is not None]) if metric == 1 else bottom
            axis.set_ylim(low, upper + 0.10 * span)
            axis.set_xlim(-0.60, len(methods) - 0.40)
            axis.set_xticks([])
            axis.yaxis.set_major_locator(MaxNLocator(nbins=2 if transpose else 3,
                                                     min_n_ticks=2))
            if transpose and metric == 1:
                axis.set_yticks([0.0, 0.5, 1.0] if transposed_single else [0.0, 1.0])
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            axis.grid(axis="y", color="#DADADA", linewidth=0.4, zorder=0)
            axis.set_axisbelow(True)
            if metric == 1:
                axis.axhline(1.0, color="#777777", linestyle="--", linewidth=0.5, zorder=1)
            if transposed_single:
                axis.set_title("Total reward" if metric == 0 else "Safety rate", pad=3)
            elif transpose:
                if metric == 0:
                    # Position the task name across the whole figure, not one axis.
                    # A slightly taller inter-row gutter clears the tick labels
                    # above while keeping the native overall height unchanged.
                    bounds = axis.get_position()
                    fig.text(ENVIRONMENT_HEADING_X,
                             bounds.y1 + 1.5 / (72 * fig.get_size_inches()[1]),
                             canonical_label(panel.label), ha="center", va="bottom",
                             fontsize=7.5, fontweight="bold")
            elif single_column:
                axis.set_title("Total reward" if metric == 0 else "Safety rate", pad=5)
            elif metric == 0:
                axis.set_title(panel_title(panel.label), pad=6)
    if transposed_single:
        fig.suptitle(title or panels[0].label, fontsize=8, fontweight="bold",
                     x=ENVIRONMENT_HEADING_X, y=0.97)
        columns = 1
    elif transpose:
        for metric, label in enumerate(("Total reward", "Safety rate")):
            bounds = axes[0, metric].get_position()
            fig.text((bounds.x0 + bounds.x1) / 2, 0.985, label,
                     ha="center", va="top", fontsize=8)
        columns = len(methods) if legend_one_row else 2
    elif single_column:
        fig.suptitle(title or panels[0].label, fontsize=8, fontweight="bold", y=0.96)
        columns = 2
    else:
        fig.text(0.012, 0.70, "Total reward", rotation=90, va="center", fontsize=8)
        fig.text(0.012, 0.34, "Safety rate", rotation=90, va="center", fontsize=8)
        columns = 4 if len(methods) >= 7 else 3
    indices = legend_indices(len(methods), columns)
    handles = [Patch(facecolor=method.color, edgecolor="#252525", linewidth=0.35,
                     hatch=method.hatch) for method in methods]
    labels = [method.label if transposed_single else method.label.replace("\n", " ")
              for method in methods]
    fig.legend([handles[i] for i in indices], [labels[i] for i in indices],
               loc="center left" if transposed_single else "lower center",
               bbox_to_anchor=((0.64, 0.46) if transposed_single else
                               (0.5 if legend_one_row else 0.52, 0.015)),
               ncol=columns, borderaxespad=0 if transposed_single else 0.5,
               frameon=False, fontsize=7 if single_column or transpose else 8,
               handlelength=1.0 if legend_one_row else 1.25,
               handletextpad=0.3 if legend_one_row else 0.45,
               columnspacing=0.35 if legend_one_row else 1.0, labelspacing=0.35)
    return fig


def save_compact_figure(fig, stem: Path, *, panels: list[BarPanel], methods: list[BarMethod],
                        se_multiplier: float, caption: str) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["environment", "method", "reward_mean", "reward_error",
                         "safety_mean", "safety_error", "se_multiplier"])
        for panel in ordered_panels(panels):
            for i, method in enumerate(methods):
                writer.writerow([canonical_label(panel.label), method.label.replace("\n", " "),
                                 panel.reward_means[i], panel.reward_errors[i],
                                 panel.safety_means[i], panel.safety_errors[i], se_multiplier])
    width = float(fig.get_size_inches()[0])
    environment = "figure*" if width > COLUMN_WIDTH_IN + 0.1 else "figure"
    hatched_labels = [method.label.replace("\n", " ") for method in methods if method.hatch]
    fill_description = (
        "Hatched bars identify " + ", ".join(hatched_labels) + "; other bars use solid fills. "
        if hatched_labels else
        "Methods use solid-colour bars and are identified by the shared legend. "
    )
    description = ("Grouped bar charts compare total reward and safety rate. "
                   + fill_description +
                   "Panels use task-specific reward scales and probability safety scales. "
                   f"Error bars show {se_multiplier:g} standard errors. "
                   "Reward bars are anchored at each panel's minimum rather than zero.")
    stem.with_suffix(".tex").write_text(
        f"\\begin{{{environment}}}[t]\n  \\centering\n"
        f"  \\includegraphics[width={width:g}in]{{{stem.name}.pdf}}\n"
        f"  \\caption{{{caption} Reward bars are anchored at each panel's minimum; "
        "compare the labelled axis values rather than bar lengths across tasks.}\n"
        f"  \\label{{fig:{stem.name.replace('_', '-')}}}\n"
        f"  \\Description{{{description}}}\n\\end{{{environment}}}\n"
    )
    plt.close(fig)
