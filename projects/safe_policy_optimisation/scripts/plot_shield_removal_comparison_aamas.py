#!/usr/bin/env python3
"""Compact single-column AAMAS shield-removal reward/safety comparison.

Reuse the existing experiment definitions and terminal 100-episode evaluations,
not learning-curve checkpoints. PPO-Shield's on/off bars use the same trained
weights. PSPO uses no runtime shield. Existing figures are not overwritten.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _aamas_reward_safety import BarMethod, BarPanel, compact_figure, save_compact_figure
from generate_final_policy_table import ENV_LABELS, mean_and_error, read_seed_values
from plot_extended_budget_learning_curves import ENVIRONMENTS, EXPECTED_SEEDS
from plot_shield_removal_comparison import SERIES

DEFAULT_OUTPUT_DIR = REPO / "projects/safe_policy_optimisation/figures/aamas"
DISPLAY_LABELS = {"ppo_shield": "PPO-Shield (shield on)",
                  "ppo_shield_nominal": "PPO-Shield (shield off)", "pspo": "PSPO"}


def bar_methods() -> list[BarMethod]:
    return [BarMethod(spec.key, DISPLAY_LABELS[spec.key], color,
                      "///" if spec.key == "ppo_shield_nominal" else None)
            for spec, color, _hatch in SERIES]


def load_panels(environments, ci_multiplier: float = 2.0) -> list[BarPanel]:
    if not math.isfinite(ci_multiplier) or ci_multiplier < 0:
        raise ValueError("The standard-error multiplier must be finite and nonnegative")
    panels = []
    for environment in environments:
        rewards, reward_errors, safeties, safety_errors = [], [], [], []
        for spec, _color, _hatch in SERIES:
            seed_rewards, seed_safeties, episode_counts = read_seed_values(spec, environment)
            if len(seed_rewards) != len(EXPECTED_SEEDS) or len(seed_safeties) != len(EXPECTED_SEEDS):
                raise ValueError(f"Incomplete seed cohort: {environment.key}, {spec.key}")
            if episode_counts != {100}:
                raise ValueError(f"Expected 100 terminal evaluation episodes: "
                                 f"{environment.key}, {spec.key}: {episode_counts}")
            reward_mean, reward_error = mean_and_error(seed_rewards, ci_multiplier)
            safety_mean, safety_error = mean_and_error(seed_safeties, ci_multiplier)
            rewards.append(reward_mean)
            reward_errors.append(reward_error)
            safeties.append(safety_mean)
            safety_errors.append(safety_error)
        panels.append(BarPanel(ENV_LABELS[environment.key], rewards, reward_errors,
                               safeties, safety_errors))
    return panels


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="shield_removal_comparison_transposed")
    parser.add_argument("--ci-multiplier", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    panels = load_panels(ENVIRONMENTS, args.ci_multiplier)
    methods = bar_methods()
    fig = compact_figure(panels, methods, transpose=True, legend_one_row=True)
    stem = args.output_dir / args.name
    save_compact_figure(
        fig, stem, panels=panels, methods=methods, se_multiplier=args.ci_multiplier,
        caption=("Shield-removal experiment: final-policy total reward and safety rate "
                 f"(mean $\\pm$ {args.ci_multiplier:g} standard errors across "
                 f"{len(EXPECTED_SEEDS)} seeds; 100 evaluation episodes per seed). "
                 "PPO-Shield (shield on) and PPO-Shield (shield off) are the same trained weights "
                 "evaluated with and without the runtime shield. "
                 "The shield-off bars are diagonally hatched. "
                 "PSPO is evaluated without a runtime shield."),
    )
    print(f"Saved {stem}.pdf, .png, .csv and .tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
