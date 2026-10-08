#!/usr/bin/env python3
"""Combine evaluation learning curves with final reward/safety metrics.

Each environment occupies one row. The first two columns show deterministic
evaluation learning curves and the final two show the 100-episode final-policy
metrics. Bands and bar error bars are mean +/- two standard errors by default.

The figure reuses the loaders and paper styling of the standalone figures so
its values, method colours, line styles, bar order, and formal-safety hatching
remain consistent with them.
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
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_final_policy_table as final_metrics  # noqa: E402
import plot_extended_budget_learning_curves as learning  # noqa: E402
import plot_final_policy_reward_safety_bars as final_bars  # noqa: E402
from _paper_style import (  # noqa: E402
    METHOD_COLORS,
    TEXT_WIDTH_IN,
    apply_paper_style,
    save_paper_figure,
)

DEFAULT_OUTPUT_DIR = (
    REPO
    / "projects/safe_policy_optimisation/figures"
    / "pspo_vs_rl_baselines_learning_curves_all_envs"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        action="append",
        choices=[environment.key for environment in learning.ENVIRONMENTS],
        help="Environment to include; repeat as needed (default: all six).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--name",
        default="final_metrics_with_evaluation_learning_curves",
        help="Output file stem.",
    )
    parser.add_argument(
        "--ci-multiplier",
        type=float,
        default=2.0,
        help="Multiplier on standard errors for bands and bars (default: 2).",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit learning-curve points supported by fewer than all ten seeds.",
    )
    return parser.parse_args(argv)


def _ordered_specs():
    """Align curve specs with the PSPO-last ordering of the final bars."""
    curve_by_key = {spec.key: spec for spec in learning.METHODS}
    final_specs = final_bars.bar_methods()
    curve_specs = tuple(curve_by_key[spec.key] for spec in final_specs)
    return final_specs, curve_specs


def _evaluation_curves(
    environment: learning.EnvironmentSpec,
    curve_specs,
    *,
    allow_incomplete: bool,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    expected = len(learning.EXPECTED_SEEDS)
    for spec in curve_specs:
        aggregate = learning.aggregate_evaluation(spec, environment=environment)
        if not allow_incomplete:
            aggregate = aggregate.loc[aggregate["seed_count"] == expected].copy()
            if aggregate.empty:
                raise RuntimeError(
                    f"No {expected}-seed curve support for "
                    f"{environment.key}/{spec.key}"
                )
        parts.append(aggregate)
    return pd.concat(parts, ignore_index=True)


def _plot_curve(
    axis,
    aggregate: pd.DataFrame,
    curve_specs,
    *,
    environment: learning.EnvironmentSpec,
    metric: str,
    ci_multiplier: float,
) -> None:
    for spec in curve_specs:
        curve = aggregate.loc[aggregate["method_key"] == spec.key].sort_values(
            "timestep"
        )
        if curve.empty:
            continue
        timestep = curve["timestep"].to_numpy(dtype=float)
        mean = curve[f"{metric}_mean"].to_numpy(dtype=float)
        error = ci_multiplier * curve[f"{metric}_sem"].to_numpy(dtype=float)
        lower, upper = mean - error, mean + error
        if metric == "safety":
            lower = np.clip(lower, 0.0, 1.0)
            upper = np.clip(upper, 0.0, 1.0)
        axis.plot(
            timestep,
            mean,
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=1.0,
        )
        axis.fill_between(
            timestep,
            lower,
            upper,
            color=spec.color,
            alpha=0.13,
            linewidth=0,
        )

    axis.set_xlim(0, environment.horizon)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
    axis.xaxis.set_major_formatter(FuncFormatter(learning._format_timesteps))
    if metric == "reward":
        axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
    else:
        axis.set_ylim(-0.025, 1.025)
        axis.set_yticks((0.0, 0.5, 1.0))


def _final_values(environment, final_specs, ci_multiplier: float):
    rewards: list[float] = []
    reward_errors: list[float] = []
    safeties: list[float] = []
    safety_errors: list[float] = []
    for spec in final_specs:
        seed_rewards, seed_safeties, _ = final_metrics.read_seed_values(
            spec, environment
        )
        reward, reward_error = final_metrics.mean_and_error(
            seed_rewards, ci_multiplier
        )
        safety, safety_error = final_metrics.mean_and_error(
            seed_safeties, ci_multiplier
        )
        rewards.append(reward)
        reward_errors.append(reward_error)
        safeties.append(safety)
        safety_errors.append(safety_error)
    return rewards, reward_errors, safeties, safety_errors


def _plot_bars(
    axis,
    means: list[float],
    errors: list[float],
    final_specs,
    *,
    metric: str,
) -> None:
    positions = list(range(len(final_specs)))
    colors = [METHOD_COLORS[spec.key] for spec in final_specs]
    hatches = [
        final_bars.FORMAL_SAFETY_HATCH
        if spec.key in final_bars.FORMALLY_SAFE_METHODS
        else None
        for spec in final_specs
    ]
    if metric == "reward":
        bottom = final_bars.bottom_with_slack(means, errors)
        heights = [mean - bottom for mean in means]
        bars = axis.bar(
            positions,
            heights,
            bottom=bottom,
            yerr=errors,
            capsize=2,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            error_kw={"linewidth": 0.8, "ecolor": "black"},
        )
        axis.set_ylim(bottom=bottom)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
    else:
        bars = axis.bar(
            positions,
            means,
            yerr=errors,
            capsize=2,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            error_kw={"linewidth": 0.8, "ecolor": "black"},
        )
        axis.set_ylim(
            bottom=min(final_bars.bottom_with_slack(means, errors), 0.0),
            top=1.05,
        )
        axis.set_yticks((0.0, 0.5, 1.0))
        axis.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, zorder=0)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axis.set_xticks(positions)
    axis.set_xticklabels([])
    axis.set_xlim(-0.6, len(positions) - 0.4)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ci_multiplier < 0:
        raise SystemExit("--ci-multiplier must be non-negative")

    requested = set(args.environment or [e.key for e in learning.ENVIRONMENTS])
    environments = tuple(
        environment
        for environment in learning.ENVIRONMENTS
        if environment.key in requested
    )
    final_specs, curve_specs = _ordered_specs()

    if not args.allow_incomplete:
        for environment in environments:
            learning.validate_sources(environment)

    apply_paper_style()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(
        len(environments),
        4,
        figsize=(TEXT_WIDTH_IN, 1.08 * len(environments) + 1.0),
        squeeze=False,
        sharex=False,
        sharey=False,
        layout="constrained",
    )

    titles = (
        "Learning: reward",
        "Learning: safety",
        "Final reward",
        "Final safety",
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=7.2, fontweight="bold", pad=4)

    for row, environment in enumerate(environments):
        aggregate = _evaluation_curves(
            environment, curve_specs, allow_incomplete=args.allow_incomplete
        )
        _plot_curve(
            axes[row, 0],
            aggregate,
            curve_specs,
            environment=environment,
            metric="reward",
            ci_multiplier=args.ci_multiplier,
        )
        _plot_curve(
            axes[row, 1],
            aggregate,
            curve_specs,
            environment=environment,
            metric="safety",
            ci_multiplier=args.ci_multiplier,
        )

        rewards, reward_errors, safeties, safety_errors = _final_values(
            environment, final_specs, args.ci_multiplier
        )
        _plot_bars(
            axes[row, 2], rewards, reward_errors, final_specs, metric="reward"
        )
        _plot_bars(
            axes[row, 3], safeties, safety_errors, final_specs, metric="safety"
        )

        axes[row, 0].set_ylabel(
            textwrap.fill(final_metrics.ENV_LABELS[environment.key], width=12),
            fontsize=6.7,
            fontweight="bold",
            linespacing=0.95,
        )

    axes[-1, 0].set_xlabel("Training timestep")
    axes[-1, 1].set_xlabel("Training timestep")

    hatches = [
        final_bars.FORMAL_SAFETY_HATCH
        if spec.key in final_bars.FORMALLY_SAFE_METHODS
        else None
        for spec in final_specs
    ]
    handles = [
        Patch(
            facecolor=METHOD_COLORS[spec.key],
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
        )
        for spec, hatch in zip(final_specs, hatches)
    ]
    fig.legend(
        handles,
        [spec.label for spec in final_specs],
        loc="outside upper center",
        ncol=len(final_specs),
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.45,
        handlelength=1.2,
    )
    fig.align_ylabels(axes[:, 0])
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06, hspace=0.10)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.name
    save_paper_figure(fig, stem)
    plt.close(fig)
    print(f"Saved {stem}.pdf and {stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
