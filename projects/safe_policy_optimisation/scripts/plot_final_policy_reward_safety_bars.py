#!/usr/bin/env python3
"""Compact AAMAS final-policy reward/safety bars (mean +/- two SE).

Data are shared with generate_final_policy_table. Original figures are retained;
new PDFs, previews, data and LaTeX snippets go to figures/aamas.
``--include-shield-off`` adds PPO-Shield's weights evaluated without the runtime
shield, between PPO-Shield and PSPO, and writes to a separate ``_shield_off`` stem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paper_style import METHOD_COLORS  # noqa: E402
from _aamas_reward_safety import BarMethod, BarPanel, compact_figure, save_compact_figure
from generate_final_policy_table import (  # noqa: E402
    ENV_LABELS,
    SHIELD_OFF_SPEC,
    bar_order,
    mean_and_error,
    read_seed_values,
)
from plot_extended_budget_learning_curves import (  # noqa: E402
    ENVIRONMENTS,
    EXPECTED_SEEDS,
)

DEFAULT_OUTPUT_DIR = (
    REPO
    / "projects/safe_policy_optimisation/figures"
    / "aamas"
)

FORMALLY_SAFE_METHODS = frozenset({"ppo_shield", "pspo"})
FORMAL_SAFETY_HATCH = "//"

# Pink here, not _paper_style's plum, which the learning curves keep using.
SHIELD_OFF_COLOR = "pink"

# Shared with generate_final_policy_table so the table rows follow the bars.
bar_methods = bar_order


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--ci-multiplier",
        type=float,
        default=2.0,
        help="Multiplier on the standard error (default: 2).",
    )
    parser.add_argument(
        "--layout",
        choices=("env-columns", "env-rows"),
        default="env-columns",
        help="env-columns: full width; env-rows: transposed, compact single-column figure.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output file stem (default includes _transposed for env-rows).",
    )
    parser.add_argument(
        "--include-shield-off",
        action="store_true",
        help="Add PPO Shield (shield off) between PPO-Shield and PSPO (default stem gains _shield_off).",
    )
    return parser.parse_args(argv)


def bottom_with_slack(
    means: list[float], errors: list[float], slack_fraction: float = 0.05
) -> float:
    """Axis bottom = min(mean - error) minus slack sized to that bound.

    Sizing the slack off the panel's full span instead pushes the bottom well
    past the minimum bar whenever another bar sits near zero.
    """
    lows = [mean - error for mean, error in zip(means, errors)]
    minimum = min(lows)
    magnitude = abs(minimum) if abs(minimum) > 1e-9 else 1.0
    return minimum - slack_fraction * magnitude


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ci_multiplier < 0:
        raise SystemExit("--ci-multiplier must be nonnegative")
    specs = bar_methods(args.include_shield_off)
    colors = {**METHOD_COLORS, SHIELD_OFF_SPEC.key: SHIELD_OFF_COLOR}
    methods = [BarMethod(spec.key, spec.label, colors[spec.key]) for spec in specs]
    panels = []
    for environment in ENVIRONMENTS:
        reward_means, reward_errors, safety_means, safety_errors = [], [], [], []
        for spec in specs:
            rewards, safeties, _ = read_seed_values(spec, environment)
            reward_mean, reward_error = mean_and_error(rewards, args.ci_multiplier)
            safety_mean, safety_error = mean_and_error(safeties, args.ci_multiplier)
            reward_means.append(reward_mean)
            reward_errors.append(reward_error)
            safety_means.append(safety_mean)
            safety_errors.append(safety_error)
        panels.append(BarPanel(ENV_LABELS[environment.key], reward_means, reward_errors,
                               safety_means, safety_errors))
    transpose = args.layout == "env-rows"
    fig = compact_figure(panels, methods, transpose=transpose)
    name = args.name or ("final_policy_reward_safety_bars"
                         + ("_shield_off" if args.include_shield_off else "")
                         + ("_transposed" if transpose else ""))
    stem = args.output_dir / name
    caption = ("Final greedy-policy total reward and safety rate "
               f"(mean $\\pm$ {args.ci_multiplier:g} standard errors across "
               f"{len(EXPECTED_SEEDS)} seeds).")
    if args.include_shield_off:
        caption += (" PPO Shield (shield off) is the trained PPO-Shield policy evaluated "
                    "without its runtime shield.")
    save_compact_figure(
        fig, stem, panels=panels, methods=methods, se_multiplier=args.ci_multiplier,
        caption=caption,
    )
    print(f"Saved {stem}.pdf, .png, .csv and .tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
