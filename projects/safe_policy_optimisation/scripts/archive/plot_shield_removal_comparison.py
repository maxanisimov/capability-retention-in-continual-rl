#!/usr/bin/env python3
"""Does the safety belong to the policy, or to the shield?

PPO-Shield is evaluated twice from the same trained weights: with its runtime
shield attached, and with the shield removed. PSPO is never shielded at
evaluation. Plotting the three side by side shows whether a reported safety
rate is a property of the exported policy or of a runtime wrapper.

Values are read through ``generate_final_policy_table`` so this figure, the
LaTeX table and the headline bar figure cannot disagree. Both PPO-Shield
variants come from one 100-episode deterministic evaluation of the final
policy, recorded under ``metrics.json['shielded']`` and ``['nominal']``.

Rendered at ICLR single-column width via ``_paper_style``; the seed count and
error definition belong in the LaTeX caption, not in the figure.

PPO-Shield and PSPO use the same diagonal hatch as the main reward/safety
figure to mark their formal safety guarantees. PPO-Shield with its shield
removed uses a cross-hatch so it cannot be mistaken for a guaranteed method.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paper_style import (  # noqa: E402
    METHOD_COLORS,
    TEXT_HEIGHT_IN,
    TEXT_WIDTH_IN,
    apply_paper_style,
    save_paper_figure,
)
from generate_final_policy_table import (  # noqa: E402
    ENV_LABELS,
    MethodSpec,
    mean_and_error,
    read_seed_values,
)
from plot_extended_budget_learning_curves import ENVIRONMENTS  # noqa: E402
from plot_final_policy_reward_safety_bars import (  # noqa: E402
    FORMAL_SAFETY_HATCH,
    bottom_with_slack,
)

DEFAULT_OUTPUT_DIR = (
    REPO
    / "projects/safe_policy_optimisation/figures"
    / "pspo_vs_rl_baselines_learning_curves_all_envs"
)

# The three evaluated variants. The first two are the SAME trained weights,
# differing only in whether the runtime shield is attached at evaluation, so
# they are drawn as a colour pair (purple / plum). The guaranteed methods share
# the main figure's diagonal hatch; the nominal variant uses a cross-hatch.
SERIES = (
    (
        MethodSpec(
            "ppo_shield", "PPO-Shield (shield on)", "ppo_shield/metrics.json",
            ("shielded",),
        ),
        METHOD_COLORS["ppo_shield"],
        FORMAL_SAFETY_HATCH,
    ),
    (
        MethodSpec(
            "ppo_shield_nominal", "PPO-Shield (shield removed)",
            "ppo_shield/metrics.json", ("nominal",),
        ),
        METHOD_COLORS["ppo_shield_nominal"],
        "xx",
    ),
    (
        MethodSpec("pspo", "PSPO (never shielded)", "metrics.json", (), True),
        METHOD_COLORS["pspo"],
        FORMAL_SAFETY_HATCH,
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        action="append",
        choices=[environment.key for environment in ENVIRONMENTS],
        help="Environment to include; repeat as needed (default: all six).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--name",
        default="shield_removal_comparison",
        help="Output file stem (default: shield_removal_comparison).",
    )
    parser.add_argument("--ci-multiplier", type=float, default=2.0)
    return parser.parse_args(argv)


def _apply_style() -> None:
    """Paper typography, shared with the other bar figure via ``_paper_style``."""
    apply_paper_style()
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    keys = args.environment or [environment.key for environment in ENVIRONMENTS]
    environments = [e for e in ENVIRONMENTS if e.key in set(keys)]

    _apply_style()
    columns = len(environments)
    # Fixed ICLR \textwidth and 20% of \textheight, matching the compact main
    # final-metrics chart. Adding environments narrows the panels instead of
    # growing the figure off the page.
    fig, axes = plt.subplots(
        2, columns, figsize=(TEXT_WIDTH_IN, 0.20 * TEXT_HEIGHT_IN), squeeze=False,
        layout="constrained",
    )
    positions = list(range(len(SERIES)))
    colors = [color for _spec, color, _hatch in SERIES]
    hatches = [hatch for _spec, _color, hatch in SERIES]

    for column, environment in enumerate(environments):
        rewards: list[float] = []
        reward_errors: list[float] = []
        safeties: list[float] = []
        safety_errors: list[float] = []
        for spec, _color, _hatch in SERIES:
            seed_rewards, seed_safeties, _ = read_seed_values(spec, environment)
            reward_mean, reward_error = mean_and_error(
                seed_rewards, args.ci_multiplier
            )
            safety_mean, safety_error = mean_and_error(
                seed_safeties, args.ci_multiplier
            )
            rewards.append(reward_mean)
            reward_errors.append(reward_error)
            safeties.append(safety_mean)
            safety_errors.append(safety_error)
            print(
                f"{environment.label:20s} {spec.label:28s} "
                f"R {reward_mean:8.2f} +/- {reward_error:5.2f}   "
                f"S {safety_mean:5.2f} +/- {safety_error:4.2f}"
            )

        # Reward bars anchored at the panel minimum, not zero, so that bar
        # height stays monotonic in the mean even where rewards are negative.
        axis = axes[0][column]
        bottom = bottom_with_slack(rewards, reward_errors)
        bars = axis.bar(
            positions,
            [value - bottom for value in rewards],
            bottom=bottom,
            yerr=reward_errors,
            capsize=2,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            error_kw={"linewidth": 0.8, "ecolor": "black"},
        )
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
        axis.set_ylim(bottom=bottom)
        # Environment named once per column; metric named once per row. Wrapped
        # so long names fit a ~0.8 in column without running into the
        # neighbouring panel's title.
        axis.set_title(
            textwrap.fill(ENV_LABELS[environment.key], width=11),
            fontsize=7,
            fontweight="bold",
            pad=3,
            linespacing=0.95,
        )
        axis.set_xticks(positions)
        axis.set_xticklabels([])
        if column == 0:
            axis.set_ylabel("Total reward")

        axis = axes[1][column]
        bars = axis.bar(
            positions,
            safeties,
            yerr=safety_errors,
            capsize=2,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            error_kw={"linewidth": 0.8, "ecolor": "black"},
        )
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
        axis.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, zorder=0)
        axis.set_ylim(-0.03, 1.12)
        axis.set_xticks(positions)
        axis.set_xticklabels([])
        if column == 0:
            axis.set_ylabel("Safety rate")
        # Call out collapses: any safety that lost more than a hair. These are
        # the payload of the figure, so they stay even at this size.
        for index, mean in enumerate(safeties):
            if mean < 0.995:
                axis.annotate(
                    f"{mean:.2f}",
                    (index, mean + safety_errors[index]),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    fontsize=5.5,
                    fontweight="bold",
                    color="#b3261e",
                )

    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5,
            hatch=hatch,
        )
        for _spec, color, hatch in SERIES
    ]
    fig.legend(
        handles,
        [spec.label for spec, _color, _hatch in SERIES],
        loc="outside lower center",
        ncol=len(SERIES),
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
        handlelength=1.2,
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.05)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.name
    save_paper_figure(fig, stem)
    plt.close(fig)
    print(f"\nSaved {stem}.pdf and {stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
