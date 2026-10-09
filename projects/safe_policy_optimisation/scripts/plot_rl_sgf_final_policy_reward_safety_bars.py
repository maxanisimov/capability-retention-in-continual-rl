#!/usr/bin/env python3
"""Compact AAMAS final-policy reward/safety bars with RL-SGF (mean +/- two SE).

Same layout, sizing and data sources as ``plot_final_policy_reward_safety_bars.py
--layout env-rows --include-shield-off`` (the transposed single-column figure),
with RL-SGF (safe initialisation) added. Bar order:

PPO, PPO-Lagrangian, PPO-PID-Lagrangian, CPO, RL-SGF, PPO-Shield (shield on),
PPO-Shield (shield off), PSPO.

Colours follow the learning curves (``_paper_style``; RL-SGF brown). PPO-Shield
(shield on) is deep purple; PPO-Shield (shield off) is light purple with black
diagonal hatching so the shield-removed evaluation stands out.

Axes are zoomed to separate methods: in every panel the bottom limit is
``min mean - 0.25 * |min mean|`` (reward and safety alike) and the top limit is
the reference layout's.

RL-SGF runs are read from ``--rl-sgf-root`` (default: the runs directory's
``rl_sgf_safe_initialised``), ``<env>/rl_sgf/seed<k>/metrics.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paper_style import METHOD_COLORS  # noqa: E402
from _aamas_reward_safety import (  # noqa: E402
    BarMethod,
    BarPanel,
    compact_figure,
    ordered_panels,
    save_compact_figure,
)
from generate_final_policy_table import (  # noqa: E402
    ENV_LABELS,
    METHODS,
    SHIELD_OFF_SPEC,
    MethodSpec,
    mean_and_error,
    read_seed_values,
)
from plot_extended_budget_learning_curves import ENVIRONMENTS, EXPECTED_SEEDS, RUNS  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO / "projects/safe_policy_optimisation/figures/aamas"
DEFAULT_RL_SGF_ROOT = RUNS / "rl_sgf_safe_initialised"
DEFAULT_NAME = "final_policy_reward_safety_bars_rl_sgf_transposed"

RL_SGF_SPEC = MethodSpec("rl_sgf", "RL-SGF", "metrics.json", ("rl_sgf",))
SHIELD_ON_COLOR = "#4B0082"   # deep purple
SHIELD_OFF_COLOR = "#C9A0DC"  # light purple
SHIELD_OFF_HATCH = "////"     # drawn in the bar edge colour (near-black)


def method_order() -> list[tuple[MethodSpec, BarMethod]]:
    by_key = {spec.key: spec for spec in METHODS}
    shield_on = dataclasses.replace(by_key["ppo_shield"], label="PPO-Shield (shield on)")
    shield_off = dataclasses.replace(SHIELD_OFF_SPEC, label="PPO-Shield (shield off)")
    return [
        (by_key["ppo_policy"], BarMethod("ppo_policy", "PPO", METHOD_COLORS["ppo_policy"])),
        (by_key["ppo_lagrangian"],
         BarMethod("ppo_lagrangian", "PPO-Lagrangian", METHOD_COLORS["ppo_lagrangian"])),
        (by_key["ppo_pid_lagrangian"],
         BarMethod("ppo_pid_lagrangian", "PPO-PID-Lagrangian", METHOD_COLORS["ppo_pid_lagrangian"])),
        (by_key["cpo"], BarMethod("cpo", "CPO", METHOD_COLORS["cpo"])),
        (RL_SGF_SPEC, BarMethod("rl_sgf", "RL-SGF", "brown")),
        (shield_on, BarMethod("ppo_shield", shield_on.label, SHIELD_ON_COLOR)),
        (shield_off, BarMethod("ppo_shield_nominal", shield_off.label, SHIELD_OFF_COLOR,
                               hatch=SHIELD_OFF_HATCH)),
        (by_key["pspo"], BarMethod("pspo", "PSPO", METHOD_COLORS["pspo"])),
    ]


def zoomed_bottom(means: list[float | None], fraction: float = 0.25) -> float:
    """Bottom limit = min mean - fraction * |min mean|."""
    lowest = min(mean for mean in means if mean is not None)
    return lowest - fraction * abs(lowest)


def zoom_axes(fig, panels: list[BarPanel]) -> None:
    """Raise each panel's bottom limit to just below its lowest mean; keep its top limit.

    ``compact_figure`` lays the transposed figure out as one (reward, safety) axis
    pair per panel, in ``ordered_panels`` order. Bars keep their original anchors
    (reward: the panel minimum; safety: 0) and are simply cut off at the new
    bottom, as are error bars that reach below it.
    """
    axes = fig.axes
    for index, panel in enumerate(ordered_panels(panels)):
        for axis, means in ((axes[2 * index], panel.reward_means),
                            (axes[2 * index + 1], panel.safety_means)):
            _, top = axis.get_ylim()
            bottom = zoomed_bottom(means)
            axis.set_ylim(bottom, top)
        # Reward bars are anchored at the reference layout's panel minimum; when the
        # zoomed bottom lies below it (negative rewards), re-anchor them at the new
        # bottom so every bar starts at the axis instead of floating above it.
        reward_axis = axes[2 * index]
        bottom = reward_axis.get_ylim()[0]
        for bar in reward_axis.patches:
            if isinstance(bar, Rectangle):
                upper = bar.get_y() + bar.get_height()
                bar.set_y(bottom)
                bar.set_height(upper - bottom)
        # The reference fixes safety ticks at 0 and 1. Keep 1.0 labelled and add the
        # lowest round value inside the zoomed range; limits are left untouched.
        safety_axis = axes[2 * index + 1]
        bottom, top = safety_axis.get_ylim()
        candidates = MaxNLocator(nbins=4).tick_values(bottom, 1.0)
        inside = [tick for tick in candidates if bottom <= tick < 1.0 - 1e-9]
        safety_axis.set_yticks(([min(inside)] if inside else []) + [1.0])
        safety_axis.set_ylim(bottom, top)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rl-sgf-root", type=Path, default=DEFAULT_RL_SGF_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--ci-multiplier", type=float, default=2.0,
                        help="Multiplier on the standard error (default: 2).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ci_multiplier < 0:
        raise SystemExit("--ci-multiplier must be nonnegative")
    order = method_order()
    panels = []
    for environment in ENVIRONMENTS:
        reward_means, reward_errors, safety_means, safety_errors = [], [], [], []
        for spec, _method in order:
            source = environment
            if spec is RL_SGF_SPEC:
                source = dataclasses.replace(
                    environment, baseline_root=args.rl_sgf_root / environment.key / "rl_sgf"
                )
            rewards, safeties, _ = read_seed_values(spec, source)
            reward_mean, reward_error = mean_and_error(rewards, args.ci_multiplier)
            safety_mean, safety_error = mean_and_error(safeties, args.ci_multiplier)
            reward_means.append(reward_mean)
            reward_errors.append(reward_error)
            safety_means.append(safety_mean)
            safety_errors.append(safety_error)
        panels.append(BarPanel(ENV_LABELS[environment.key], reward_means, reward_errors,
                               safety_means, safety_errors))
    methods = [method for _spec, method in order]
    fig = compact_figure(panels, methods, transpose=True)
    zoom_axes(fig, panels)
    stem = args.output_dir / args.name
    caption = (
        "Final greedy-policy total reward and safety rate "
        f"(mean $\\pm$ {args.ci_multiplier:g} standard errors across {len(EXPECTED_SEEDS)} seeds). "
        "RL-SGF and PSPO start from the same safe initial policy; the other methods start "
        "from random weights. PPO-Shield (shield on) is evaluated with its runtime shield; "
        "PPO-Shield (shield off) is the same trained policy evaluated without it. "
        "Both axes are truncated: each panel starts 25\\% of its lowest mean below that mean."
    )
    save_compact_figure(fig, stem, panels=panels, methods=methods,
                        se_multiplier=args.ci_multiplier, caption=caption)
    print(f"Saved {stem}.pdf, .png, .csv and .tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
