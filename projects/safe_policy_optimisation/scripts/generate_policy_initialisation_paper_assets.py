#!/usr/bin/env python3
"""Generate compact ICLR assets for the PSPO initialisation ablation.

The analysis JSON is the single source of truth for both figures and tables.
The script emits three figures at the repository's native ICLR text width
(5.5 inches), three compact ``booktabs`` tables, and a LaTeX include file with
paper-ready captions and labels.

Curves and final metrics show the seed mean and one standard error. Initialiser
diagnostics are properties of the single base policy shared by all ten seeds
of a variant/environment cell and therefore have no error bars.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

REPO = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paper_style import (  # noqa: E402
    INK_SECONDARY,
    SURFACE,
    TEXT_WIDTH_IN,
    apply_paper_style,
    save_paper_figure,
)
from generate_policy_initialisation_tables import (  # noqa: E402
    DEFAULT_RESULTS,
    ENVIRONMENTS,
    VARIANTS,
    base_policy_diagnostics,
    build_base_table,
    build_main_table,
    build_paired_table,
    load_results,
    paired_index,
    seed_count,
    summary_index,
)

DEFAULT_OUTPUT_DIR = (
    REPO / "projects/safe_policy_optimisation/figures/pspo_policy_initialisation"
)

VARIANT_LABELS = {
    "control": "PSPO",
    "no_entropy": "PSPO w/o entropy",
    "ce_only": "PSPO w/ CE-only init.",
}

# PSPO stays green across the paper; the cumulative ablations use progressively
# lighter greys. Distinct lines and markers preserve legibility in greyscale.
VARIANT_COLORS = {
    "control": "#009E73",
    "no_entropy": "#4D4D4D",
    "ce_only": "#BDBDBD",
}
VARIANT_LINESTYLES = {
    "control": "solid",
    "no_entropy": (0, (4, 1.5)),
    "ce_only": (0, (1.2, 1.2)),
}
VARIANT_MARKERS = {
    "control": "o",
    "no_entropy": "s",
    "ce_only": "^",
}

ENV_TITLES = {
    "media_streaming": "Media Streaming",
    "colour_bomb": "Colour Bomb v1",
    "colour_bomb_v2": "Colour Bomb v2",
    "bridge_crossing": "Bridge Crossing v1",
    "bridge_crossing_v2": "Bridge Crossing v2",
    "mini_pacman": "MiniPacman",
}
ENV_TICKS = ("MS", "CB1", "CB2", "BC1", "BC2", "MP")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--ci-multiplier",
        type=float,
        default=1.0,
        help="Multiplier on the curve standard error (default: 1).",
    )
    parser.add_argument(
        "--curve-points",
        type=int,
        default=101,
        help="Points in the common interpolation grid (default: 101).",
    )
    return parser.parse_args(argv)


def _paper_style() -> None:
    apply_paper_style()
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": SURFACE,
            "figure.facecolor": "white",
            "axes.grid": True,
            "axes.grid.axis": "y",
            "legend.frameon": False,
        }
    )


def _resolve_seed_dir(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def _read_curve(seed_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = seed_dir / "learning_curves/evaluation_unshielded_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing learning curve: {path}")
    steps: list[float] = []
    rewards: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = float(row["timestep"])
            reward = float(row["mean_total_reward"])
            if math.isfinite(step) and math.isfinite(reward):
                steps.append(step)
                rewards.append(reward)
    if not steps:
        raise ValueError(f"no finite curve points: {path}")
    order = np.argsort(np.asarray(steps))
    return np.asarray(steps)[order], np.asarray(rewards)[order]


def load_curves(
    results: dict,
) -> dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]:
    """Read every completed initialisation-arm curve referenced by results.json."""
    curves: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(
        list
    )
    for row in results["per_seed"]:
        variant = row["variant"]
        environment = row["environment"]
        if (
            variant not in VARIANTS
            or environment not in ENVIRONMENTS
            or row["status"] != "complete"
        ):
            continue
        curves[(variant, environment)].append(
            _read_curve(_resolve_seed_dir(row["seed_dir"]))
        )
    expected = seed_count(summary_index(results))
    for environment in ENVIRONMENTS:
        for variant in VARIANTS:
            observed = len(curves[(variant, environment)])
            if observed != expected:
                raise ValueError(
                    f"expected {expected} curves for {variant}/{environment}, "
                    f"found {observed}"
                )
    return curves


def _curve_summary(
    curves: Iterable[tuple[np.ndarray, np.ndarray]],
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(
        [
            np.interp(grid, steps, rewards, left=rewards[0], right=rewards[-1])
            for steps, rewards in curves
        ]
    )
    mean = samples.mean(axis=0)
    se = samples.std(axis=0, ddof=1) / math.sqrt(samples.shape[0])
    return mean, se


def plot_learning_curves(
    results: dict,
    *,
    output_dir: Path,
    ci_multiplier: float,
    curve_points: int,
) -> None:
    curves = load_curves(results)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(TEXT_WIDTH_IN, 3.12),
        sharex=True,
        layout="constrained",
    )
    for axis, environment in zip(axes.flat, ENVIRONMENTS):
        maximum = max(
            float(steps[-1])
            for variant in VARIANTS
            for steps, _ in curves[(variant, environment)]
        )
        grid = np.linspace(0.0, maximum, curve_points)
        progress = 100.0 * grid / maximum
        plotted_values: list[np.ndarray] = []
        for variant in VARIANTS:
            mean, se = _curve_summary(curves[(variant, environment)], grid)
            error = ci_multiplier * se
            plotted_values.extend((mean - error, mean + error))
            axis.fill_between(
                progress,
                mean - error,
                mean + error,
                color=VARIANT_COLORS[variant],
                alpha=0.15,
                linewidth=0,
            )
            axis.plot(
                progress,
                mean,
                color=VARIANT_COLORS[variant],
                linestyle=VARIANT_LINESTYLES[variant],
                label=VARIANT_LABELS[variant],
                zorder=3 if variant == "control" else 2,
            )
        low = min(float(values.min()) for values in plotted_values)
        high = max(float(values.max()) for values in plotted_values)
        span = high - low
        padding = 0.06 * (span if span > 1e-9 else max(abs(high), 1.0))
        axis.set_ylim(low - padding, high + padding)
        axis.set_title(ENV_TITLES[environment], fontweight="bold", pad=2.5)
        axis.set_xlim(0.0, 100.0)
        axis.set_xticks((0, 50, 100))

    for axis in axes[-1]:
        axis.set_xlabel("Training progress (%)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Unshielded reward")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=3,
        handlelength=2.6,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    fig.get_layout_engine().set(w_pad=0.025, h_pad=0.025, wspace=0.06, hspace=0.06)
    save_paper_figure(fig, output_dir / "policy_initialisation_learning_curves")
    plt.close(fig)


def plot_final_metrics(
    results: dict,
    *,
    output_dir: Path,
    ci_multiplier: float,
) -> None:
    """Plot the two reported reward endpoints without cross-env rescaling."""
    summary = summary_index(results)
    endpoints = (
        ("final_unshielded_reward", "Final reward"),
        ("normalized_reward_auc", "Reward AUC"),
    )
    panel_titles = {
        "media_streaming": "Media\nStream.",
        "colour_bomb": "Colour Bomb\nv1",
        "colour_bomb_v2": "Colour Bomb\nv2",
        "bridge_crossing": "Bridge Cross.\nv1",
        "bridge_crossing_v2": "Bridge Cross.\nv2",
        "mini_pacman": "MiniPacman",
    }
    fig, axes = plt.subplots(
        2,
        len(ENVIRONMENTS),
        figsize=(TEXT_WIDTH_IN, 2.35),
        sharex=True,
        layout="constrained",
    )
    positions = np.arange(len(VARIANTS), dtype=float)
    for row_index, (endpoint, ylabel) in enumerate(endpoints):
        for column_index, environment in enumerate(ENVIRONMENTS):
            axis = axes[row_index, column_index]
            records = [
                summary[(variant, environment, endpoint)] for variant in VARIANTS
            ]
            means = np.asarray([float(record["mean"]) for record in records])
            errors = ci_multiplier * np.asarray(
                [float(record["standard_error"]) for record in records]
            )
            axis.plot(
                positions,
                means,
                color="#D7D7D7",
                linewidth=0.7,
                zorder=1,
            )
            for position, variant, mean, error in zip(
                positions, VARIANTS, means, errors
            ):
                axis.errorbar(
                    position,
                    mean,
                    yerr=error,
                    fmt=VARIANT_MARKERS[variant],
                    color=VARIANT_COLORS[variant],
                    markeredgecolor="#555555",
                    markeredgewidth=0.35,
                    markersize=3.7,
                    capsize=1.5,
                    elinewidth=0.8,
                    zorder=3,
                )
            low = float(np.min(means - errors))
            high = float(np.max(means + errors))
            span = high - low
            padding = 0.10 * (span if span > 1e-9 else max(abs(high), 1.0))
            axis.set_ylim(low - padding, high + padding)
            axis.set_xlim(-0.45, len(VARIANTS) - 0.55)
            axis.set_xticks(())
            axis.locator_params(axis="y", nbins=3)
            if row_index == 0:
                axis.set_title(panel_titles[environment], fontweight="bold", pad=2.5)
            if column_index == 0:
                axis.set_ylabel(ylabel)

    handles = [
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker=VARIANT_MARKERS[variant],
            color=VARIANT_COLORS[variant],
            markeredgecolor="#555555",
            markeredgewidth=0.35,
            markersize=4.2,
        )
        for variant in VARIANTS
    ]
    fig.legend(
        handles,
        [VARIANT_LABELS[variant] for variant in VARIANTS],
        loc="outside upper center",
        ncol=3,
        columnspacing=1.0,
        handletextpad=0.25,
    )
    fig.get_layout_engine().set(w_pad=0.015, h_pad=0.02, wspace=0.04, hspace=0.05)
    save_paper_figure(fig, output_dir / "policy_initialisation_final_metrics")
    plt.close(fig)


def plot_initialiser_diagnostics(results: dict, *, output_dir: Path) -> None:
    diagnostics = base_policy_diagnostics(results)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(TEXT_WIDTH_IN, 2.05),
        layout="constrained",
    )
    positions = np.arange(len(ENVIRONMENTS), dtype=float)
    offsets = {"control": -0.18, "no_entropy": 0.0, "ce_only": 0.18}
    fields = (
        ("initialization_epochs", "Initialisation epochs"),
        ("initialization_final_min_all_margin", "Min. all-safe margin"),
        ("initialization_final_safe_entropy", "Min. safe entropy"),
    )
    for axis, (field, ylabel) in zip(axes, fields):
        for variant in VARIANTS:
            values = [
                float(diagnostics[(variant, environment)][field])
                for environment in ENVIRONMENTS
            ]
            axis.scatter(
                positions + offsets[variant],
                values,
                s=19,
                marker=VARIANT_MARKERS[variant],
                facecolor=VARIANT_COLORS[variant],
                edgecolor="white",
                linewidth=0.4,
                label=VARIANT_LABELS[variant],
                zorder=3,
            )
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, ENV_TICKS)
        axis.set_xlim(-0.55, len(ENVIRONMENTS) - 0.45)

    axes[0].set_yscale("symlog", linthresh=1.0, base=10)
    axes[0].set_ylim(-0.25, 650)
    axes[0].set_yticks((0, 1, 10, 100))
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[1].axhline(0.0, color=INK_SECONDARY, linewidth=0.7, linestyle=(0, (2, 2)))
    axes[2].axhline(0.95, color=INK_SECONDARY, linewidth=0.7, linestyle=(0, (2, 2)))
    axes[2].set_ylim(-0.04, 1.05)
    axes[2].set_yticks((0.0, 0.5, 1.0))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=3,
        columnspacing=1.5,
        handletextpad=0.2,
    )
    fig.get_layout_engine().set(w_pad=0.025, h_pad=0.025, wspace=0.08)
    save_paper_figure(fig, output_dir / "policy_initialisation_diagnostics")
    plt.close(fig)


def write_tables(results: dict, *, output_dir: Path) -> None:
    summary = summary_index(results)
    paired = paired_index(results)
    seeds = seed_count(summary)
    tables = {
        "policy_initialisation_main_table.tex": build_main_table(
            summary, decimals=2, seeds=seeds
        ),
        "policy_initialisation_paired_table.tex": build_paired_table(
            paired, decimals=2, seeds=seeds
        ),
        "policy_initialisation_base_table.tex": build_base_table(
            base_policy_diagnostics(results)
        ),
    }
    for name, table in tables.items():
        (output_dir / name).write_text(table, encoding="utf-8")


def write_latex_include(*, output_dir: Path, seeds: int, ci_multiplier: float) -> None:
    if math.isclose(ci_multiplier, 1.0):
        uncertainty = "one standard error"
    else:
        uncertainty = f"{ci_multiplier:g} standard errors"
    content = rf"""% ICLR-sized PSPO policy-initialisation ablation assets.
% Generated by generate_policy_initialisation_paper_assets.py.
% Required packages: graphicx, booktabs, multirow.

\begin{{figure}}[t]
  \centering
  \includegraphics{{projects/safe_policy_optimisation/figures/pspo_policy_initialisation/policy_initialisation_learning_curves.pdf}}
  \caption{{Unshielded evaluation reward during PSPO training for three policy
  initialisers. Curves are means over {seeds} paired seeds; shading is
  {uncertainty}. PSPO combines an all-safe logit-margin loss with safe-action
  entropy. PSPO w/o entropy slows learning in five of six environments, while
  PSPO w/ CE-only init. is especially weak in Media Streaming and Colour Bomb
  v2.}}
  \label{{fig:pspo-policy-initialisation-curves}}
\end{{figure}}

\begin{{figure}}[t]
  \centering
  \includegraphics{{projects/safe_policy_optimisation/figures/pspo_policy_initialisation/policy_initialisation_final_metrics.pdf}}
  \caption{{Final unshielded reward and normalised reward AUC (mean $\pm$
  {uncertainty}, {seeds} paired seeds). Each environment has its own vertical
  scale. PSPO obtains the best or tied reward AUC in every environment. All
  three variants retain exact shield alignment and empirical evaluation
  safety of 1.00.}}
  \label{{fig:pspo-policy-initialisation-final-metrics}}
\end{{figure}}

\begin{{figure}}[t]
  \centering
  \includegraphics{{projects/safe_policy_optimisation/figures/pspo_policy_initialisation/policy_initialisation_diagnostics.pdf}}
  \caption{{Diagnostics for the shared initial policy in each experimental
  cell (MS: Media Streaming; CB: Colour Bomb; BC: Bridge Crossing; MP:
  MiniPacman). PSPO w/ CE-only init. often terminates after very little
  optimisation and can leave negative all-safe margins and near-zero
  safe-action entropy. Dashed rules mark zero margin and PSPO's 0.95 entropy
  target.}}
  \label{{fig:pspo-policy-initialisation-diagnostics}}
\end{{figure}}

\begin{{table}}[t]
  \centering
  \caption{{Policy-initialisation ablation. Entries are mean $\pm$ one s.e.
  over {seeds} seeds; bold denotes the best mean per environment. All final
  policies have exact shield alignment and empirical evaluation safety 1.00.
  Compact headers denote PSPO, PSPO w/o entropy (PSPO$-$Ent.), and PSPO w/
  CE-only init. (PSPO$-$Ent.$-$Margin).}}
  \label{{tab:pspo-policy-initialisation-main}}
  \input{{projects/safe_policy_optimisation/figures/pspo_policy_initialisation/policy_initialisation_main_table}}
\end{{table}}

\begin{{table}}[t]
  \centering
  \caption{{Paired policy-initialisation contrasts. Each entry is treatment
  minus reference with the Holm-corrected exact sign-randomisation $p$-value
  in brackets; bold denotes $p<0.05$. Correction is across the six
  environments within each contrast and endpoint.}}
  \label{{tab:pspo-policy-initialisation-paired}}
  \input{{projects/safe_policy_optimisation/figures/pspo_policy_initialisation/policy_initialisation_paired_table}}
\end{{table}}

\begin{{table}}[t]
  \centering
  \caption{{Initial-policy diagnostics. Each entry describes the single base
  policy shared by the {seeds} seeds in that cell. PSPO w/ CE-only init. uses
  any-safe stopping; its reported margin remains the stricter all-safe margin
  for comparability.}}
  \label{{tab:pspo-policy-initialisation-base}}
  \input{{projects/safe_policy_optimisation/figures/pspo_policy_initialisation/policy_initialisation_base_table}}
\end{{table}}
"""
    (output_dir / "policy_initialisation_iclr.tex").write_text(
        content, encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ci_multiplier <= 0:
        raise ValueError("--ci-multiplier must be positive")
    if args.curve_points < 2:
        raise ValueError("--curve-points must be at least 2")
    results = load_results(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _paper_style()
    plot_learning_curves(
        results,
        output_dir=args.output_dir,
        ci_multiplier=args.ci_multiplier,
        curve_points=args.curve_points,
    )
    plot_final_metrics(
        results,
        output_dir=args.output_dir,
        ci_multiplier=args.ci_multiplier,
    )
    plot_initialiser_diagnostics(results, output_dir=args.output_dir)
    write_tables(results, output_dir=args.output_dir)
    write_latex_include(
        output_dir=args.output_dir,
        seeds=seed_count(summary_index(results)),
        ci_multiplier=args.ci_multiplier,
    )
    print(f"Wrote ICLR policy-initialisation assets to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
