#!/usr/bin/env python3
"""Learning curves for PPO-Shield, its nominal policy, and PSPO.

The curve counterpart of ``plot_shield_removal_comparison.py``. Two figures are
written:

* **evaluation-time** -- three series. PPO-Shield's periodic evaluation is run
  twice from the same weights at every checkpoint, shielded and unshielded
  (``UnshieldedRewardCurveCallback`` instances differing only in
  ``apply_shield``), so the shielded/nominal split is available at every point
  on the curve, not just at the end.

* **exploration-time** -- only *two* series. Exploration is what actually
  happened in the environment during training, and PPO-Shield explores with the
  shield attached (``ProvablySafePPO`` shields inside ``collect_rollouts``).
  There is exactly one ``training_episodes.csv`` per run and it records those
  shielded episodes. A nominal exploration trace would require training a
  second time with the shield off, so that series is legitimately absent rather
  than merely unplotted.

Aggregation is imported wholesale from ``plot_extended_budget_learning_curves``
so these curves, the six-method curves, and the bar figures all read the same
runs the same way.

Note the periodic checkpoints are 20-episode evaluations -- the 100-episode
figure is the terminal one drawn by ``plot_shield_removal_comparison.py``.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paper_style import (  # noqa: E402
    METHOD_COLORS,
    METHOD_LINESTYLES,
    TEXT_WIDTH_IN,
    apply_paper_style,
    save_paper_figure,
)
from plot_extended_budget_learning_curves import (  # noqa: E402
    ENVIRONMENT_BY_KEY,
    ENVIRONMENTS,
    EXPECTED_SEEDS,
    ROLLOUT_SIZE,
    SURFACE,
    EnvironmentSpec,
    MethodSpec,
    _common_seed_support,
    _style_metric_axis,
    aggregate_evaluation,
    aggregate_exploration,
)

DEFAULT_OUTPUT_DIR = (
    REPO
    / "projects/safe_policy_optimisation/figures"
    / "pspo_vs_rl_baselines_learning_curves_all_envs"
)

_SHIELD_CURVES = "ppo_shield/learning_curves"

# Evaluation-time: three series. The first two are the same weights evaluated
# with and without the shield at every checkpoint.
EVALUATION_SERIES = (
    MethodSpec(
        "ppo_shield",
        "PPO-Shield (shield on)",
        METHOD_COLORS["ppo_shield"],
        METHOD_LINESTYLES["ppo_shield"],
        f"{_SHIELD_CURVES}/evaluation_shielded_summary.csv",
        "ppo_shield/training_episodes.csv",
        "shielded_ppo",
    ),
    MethodSpec(
        "ppo_shield_nominal",
        "PPO-Shield (shield removed)",
        METHOD_COLORS["ppo_shield_nominal"],
        METHOD_LINESTYLES["ppo_shield_nominal"],
        f"{_SHIELD_CURVES}/evaluation_unshielded_summary.csv",
        "ppo_shield/training_episodes.csv",
        "shielded_ppo",
    ),
    MethodSpec(
        "pspo",
        "PSPO (never shielded)",
        METHOD_COLORS["pspo"],
        METHOD_LINESTYLES["pspo"],
        "learning_curves/evaluation_unshielded_summary.csv",
        "training_episodes.csv",
        "shielded_ppo",
        adaptive=True,
    ),
)

# Exploration-time: two series. See the module docstring -- the nominal policy
# never explored, so there is nothing to aggregate for it.
EXPLORATION_SERIES = tuple(
    spec for spec in EVALUATION_SERIES if spec.key != "ppo_shield_nominal"
)

KINDS = ("evaluation", "exploration")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        action="append",
        choices=tuple(ENVIRONMENT_BY_KEY),
        help="Environment to include; repeat as needed (default: all six).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--kind",
        action="append",
        choices=KINDS,
        help="Which figure to write; repeat for both (default: both).",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=ROLLOUT_SIZE,
        help=f"Exploration timestep-bin width (default: {ROLLOUT_SIZE}).",
    )
    parser.add_argument(
        "--ci-multiplier",
        type=float,
        default=2.0,
        help="Multiplier applied to standard-error bands (default: 2).",
    )
    parser.add_argument(
        "--name-prefix",
        default="shield_removal",
        help="Output file stem prefix (default: shield_removal).",
    )
    return parser.parse_args(argv)


def series_for(kind: str) -> tuple[MethodSpec, ...]:
    return EVALUATION_SERIES if kind == "evaluation" else EXPLORATION_SERIES


def collect(
    kind: str,
    environments: tuple[EnvironmentSpec, ...],
    *,
    bin_size: int,
) -> pd.DataFrame:
    """Aggregate every series for every environment into one long frame."""
    parts: list[pd.DataFrame] = []
    for environment in environments:
        for spec in series_for(kind):
            if kind == "evaluation":
                frame = aggregate_evaluation(spec, environment=environment)
            else:
                frame = aggregate_exploration(
                    spec, environment=environment, bin_size=bin_size
                )
            parts.append(
                _common_seed_support(
                    frame, expected=len(EXPECTED_SEEDS), allow_incomplete=False
                )
            )
    return pd.concat(parts, ignore_index=True)


def _plot_metric(
    axis,
    aggregates: pd.DataFrame,
    *,
    series: tuple[MethodSpec, ...],
    metric: str,
    ci_multiplier: float,
) -> None:
    """Draw each series' across-seed mean and standard-error band.

    Drawn back-to-front: solid PSPO first, the broken shield curves over it.
    These series coincide exactly wherever both are optimal (safety pinned at
    1.0 in most environments), and a solid line painted last hides a dashed one
    completely -- reversing the order lets the solid show through the dashes so
    both stay readable. Legend order is set by the caller, not by this loop.
    """
    for spec in reversed(series):
        curve = aggregates.loc[aggregates["method_key"] == spec.key].sort_values(
            "timestep"
        )
        if curve.empty:
            continue
        x = curve["timestep"].to_numpy(dtype=float)
        mean = curve[f"{metric}_mean"].to_numpy(dtype=float)
        band = ci_multiplier * curve[f"{metric}_sem"].to_numpy(dtype=float)
        axis.plot(
            x, mean, color=spec.color, linewidth=1.0, linestyle=spec.linestyle,
            label=spec.label,
        )
        lower, upper = mean - band, mean + band
        if metric == "safety":
            lower = np.clip(lower, 0.0, 1.0)
            upper = np.clip(upper, 0.0, 1.0)
        axis.fill_between(x, lower, upper, color=spec.color, alpha=0.13, linewidth=0)


def save_figure(
    aggregates: pd.DataFrame,
    *,
    kind: str,
    environments: tuple[EnvironmentSpec, ...],
    output_dir: Path,
    ci_multiplier: float,
    name_prefix: str,
) -> Path:
    """2 metric rows x N environment columns, at exactly ICLR \\textwidth.

    Matches the geometry of the bar figure so the two can sit together on a
    page. sharey is off: reward scale differs per environment, while the safety
    row is equalised explicitly by ``_style_metric_axis``.
    """
    series = series_for(kind)
    fig, axes = plt.subplots(
        2, len(environments), figsize=(TEXT_WIDTH_IN, 2.75), facecolor=SURFACE,
        squeeze=False, sharex="col", sharey=False, layout="constrained",
    )
    for column, environment in enumerate(environments):
        subset = aggregates.loc[aggregates["environment_key"] == environment.key]
        leftmost = column == 0
        for row, (metric, ylabel) in enumerate(
            (("reward", "Reward"), ("safety", "Safety rate"))
        ):
            axis = axes[row][column]
            _plot_metric(
                axis, subset, series=series, metric=metric,
                ci_multiplier=ci_multiplier,
            )
            if leftmost:
                axis.set_ylabel(ylabel)
            if metric == "reward":
                # Per-environment reward scale, so every panel keeps its ticks.
                axis.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
            elif not leftmost:
                axis.tick_params(labelleft=False)
            _style_metric_axis(
                axis, environment=environment, metric=metric, x_nbins=3
            )
        axes[0][column].set_title(
            textwrap.fill(environment.label, width=11),
            fontsize=7, fontweight="bold", pad=3, linespacing=0.95,
        )

    fig.supxlabel("Training timestep", fontsize=7.5)
    # Built from SERIES rather than from the axis, whose handles follow the
    # reversed draw order used to keep coincident curves visible.
    by_label = dict(zip(*reversed(axes[0][0].get_legend_handles_labels())))
    labels = [spec.label for spec in series]
    handles = [by_label[label] for label in labels]
    # Legend above the panels: constrained_layout pins both supxlabel and an
    # "outside lower center" legend to the figure bottom without deconflicting
    # them, so they would overlap at any height.
    fig.legend(
        handles, labels, loc="outside upper center", ncol=len(labels),
        frameon=False, handlelength=1.8, columnspacing=1.2,
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.04)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{name_prefix}_{kind}_time_learning_curves"
    save_paper_figure(fig, stem)
    plt.close(fig)
    return stem


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    keys = args.environment or [environment.key for environment in ENVIRONMENTS]
    environments = tuple(ENVIRONMENT_BY_KEY[key] for key in dict.fromkeys(keys))
    kinds = tuple(dict.fromkeys(args.kind or KINDS))
    if args.bin_size <= 0:
        raise SystemExit("--bin-size must be positive")
    if args.ci_multiplier < 0:
        raise SystemExit("--ci-multiplier must be non-negative")

    apply_paper_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        aggregates = collect(kind, environments, bin_size=args.bin_size)
        csv_path = (
            args.output_dir / f"{args.name_prefix}_{kind}_curve_aggregates.csv"
        )
        aggregates.to_csv(csv_path, index=False)
        stem = save_figure(
            aggregates,
            kind=kind,
            environments=environments,
            output_dir=args.output_dir,
            ci_multiplier=args.ci_multiplier,
            name_prefix=args.name_prefix,
        )
        drawn = ", ".join(spec.label for spec in series_for(kind))
        print(f"{kind}: {drawn}")
        print(f"  saved {stem}.pdf and {stem}.png")
        print(f"  wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
