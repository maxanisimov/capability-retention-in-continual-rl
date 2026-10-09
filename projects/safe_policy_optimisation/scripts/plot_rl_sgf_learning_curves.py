"""Exploration-time and greedy-evaluation-time learning curves with RL-SGF added.

Reuses ``plot_extended_budget_learning_curves`` (same budget-matched roots, 2048-
step grid, exploration binning, mean +/- 2 standard errors, paper style) and
plots the safe-RL baselines, PSPO and RL-SGF (safe initialisation):

* PPO-Lagrangian, PPO-PID-Lagrangian, CPO, PPO-Shield: default (not
  warm-started) baselines from the canonical roots. PPO-Shield's evaluation
  curve is the shielded/deployed one, as in the canonical figure; its training
  rollouts are shielded.
* PSPO: the default runs.
* RL-SGF: ``RL_SGF_ROOT`` (warm-started from the same safe policy as PSPO).

RL-SGF updates after whole batches of episodes, so its periodic evaluations
land at the first update boundary past each 2048-step multiple, not on the
grid. Each RL-SGF evaluation is therefore held (last observation carried
forward) onto the shared grid: the value at grid step g is the most recent
evaluation at or before g. Grid points before a seed's first evaluation are
left out, and holding past the last evaluation is only allowed when that
evaluation is at or beyond the nominal budget. Exploration curves need no
alignment: completed training episodes are binned by end timestep exactly as
for the other methods.

RL-SGF is brown, the colour the user chose for it, with its own dash pattern.

Unlike the canonical figure, each safety panel is zoomed: its y-axis runs from the
lowest mean safety rate plotted in that panel (minus a small pad) to just above 1,
so methods that are all close to 1 can be told apart. Panels therefore have
different safety scales, and each prints its own tick labels.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_extended_budget_learning_curves as canon  # noqa: E402
from _paper_style import METHOD_COLORS, METHOD_LINESTYLES  # noqa: E402

RL_SGF_ROOT = canon.RUNS / "rl_sgf_safe_initialised"
DEFAULT_OUTPUT_DIR = canon.REPO / "projects/safe_policy_optimisation/figures/rl_sgf_learning_curves"

RL_SGF = canon.MethodSpec(
    "rl_sgf",
    "RL-SGF",
    "brown",
    (0, (7, 1.5, 1, 1.5, 1, 1.5)),
    "learning_curves/rl_sgf/evaluation_unshielded_summary.csv",
    "training_episodes.csv",
    "rl_sgf",
)
BY_KEY = {spec.key: spec for spec in canon.METHODS}
# Plotting order (and legend order): penalty/trust-region baselines, shield, PSPO, RL-SGF.
SELECTED_METHODS = (
    BY_KEY["ppo_lagrangian"],
    BY_KEY["ppo_pid_lagrangian"],
    BY_KEY["cpo"],
    BY_KEY["ppo_shield"],
    BY_KEY["pspo"],
    RL_SGF,
)
assert "rl_sgf" not in METHOD_COLORS and "rl_sgf" not in METHOD_LINESTYLES

# Safety-panel y-range: from the lowest *mean* safety rate drawn in the panel (minus a
# small pad so that curve does not sit on the spine) to just above 1. The +/- 2SE band
# may extend below the bottom and is then cut off. A panel whose means are all 1.0
# gets ALL_SAFE_SPAN of headroom so the flat line stays visible.
SAFETY_PAD_FRACTION = 0.04
ALL_SAFE_SPAN = 0.05


def zoom_safety_axis(axis) -> None:
    """Rescale one safety panel to the minimum plotted mean, with its own tick labels."""
    means = [np.asarray(line.get_ydata(), dtype=float) for line in axis.get_lines()]
    means = [values[np.isfinite(values)] for values in means if np.size(values)]
    lowest = min(float(values.min()) for values in means if values.size)
    span = max(1.0 - lowest, ALL_SAFE_SPAN)
    pad = SAFETY_PAD_FRACTION * span
    axis.set_ylim(min(lowest, 1.0 - ALL_SAFE_SPAN) - pad, 1.0 + pad)
    axis.yaxis.set_major_locator(canon.MaxNLocator(nbins=3, min_n_ticks=2))
    axis.tick_params(labelleft=True)


_original_draw_row = canon._draw_environment_row
_original_draw_column = canon._draw_environment_column


def _draw_row_zoomed(axes, *args, **kwargs) -> None:
    _original_draw_row(axes, *args, **kwargs)
    zoom_safety_axis(axes[1])


def _draw_column_zoomed(axes, *args, **kwargs) -> None:
    _original_draw_column(axes, *args, **kwargs)
    zoom_safety_axis(axes[1])


def rl_sgf_environment(environment: canon.EnvironmentSpec, root: Path) -> canon.EnvironmentSpec:
    """The same environment with RL-SGF's run directory as the (baseline) source root."""
    return dataclasses.replace(environment, baseline_root=root / environment.key / "rl_sgf")


def aggregate_rl_sgf_evaluation(environment: canon.EnvironmentSpec) -> pd.DataFrame:
    """Hold each seed's off-grid evaluations onto the 2048-step grid, then summarise."""
    grid = np.arange(canon.ROLLOUT_SIZE, environment.horizon + 1, canon.ROLLOUT_SIZE)
    frames: list[pd.DataFrame] = []
    for seed, seed_dir in canon.discover_seeds(environment.baseline_root):
        raw = pd.read_csv(
            seed_dir / RL_SGF.evaluation_path,
            usecols=["timestep", "mean_total_reward", "safety_rate"],
        ).sort_values("timestep")
        times = raw["timestep"].to_numpy(dtype=np.int64)
        # Index of the last evaluation at or before each grid step (-1 = none yet).
        last = np.searchsorted(times, grid, side="right") - 1
        valid = last >= 0
        if times[-1] < environment.nominal_budget:
            valid &= grid <= times[-1]
        held = raw.iloc[last[valid]]
        frames.append(
            pd.DataFrame(
                {
                    "timestep": grid[valid],
                    "seed": seed,
                    "reward": held["mean_total_reward"].to_numpy(dtype=float),
                    "safety": held["safety_rate"].to_numpy(dtype=float),
                    "source_timestep": held["timestep"].to_numpy(dtype=np.int64),
                }
            )
        )
    if not frames:
        raise FileNotFoundError(f"No RL-SGF evaluation curves under {environment.baseline_root}")
    per_seed = pd.concat(frames, ignore_index=True)
    aggregate = canon._summary_stats(per_seed, ["timestep"])
    lag = (
        per_seed.assign(hold_lag=per_seed["timestep"] - per_seed["source_timestep"])
        .groupby("timestep")["hold_lag"]
        .max()
        .rename("max_hold_lag_timesteps")
        .reset_index()
    )
    aggregate = aggregate.merge(lag, on="timestep", validate="one_to_one")
    aggregate.insert(0, "method", RL_SGF.label)
    aggregate.insert(0, "method_key", RL_SGF.key)
    aggregate["source_root"] = str(environment.baseline_root.resolve())
    aggregate["source_rel_path"] = RL_SGF.evaluation_path + " (held onto 2048-step grid)"
    return canon._add_environment_columns(aggregate, environment)


def validate_rl_sgf(environment: canon.EnvironmentSpec) -> None:
    seeds = dict(canon.discover_seeds(environment.baseline_root))
    missing = sorted(set(canon.EXPECTED_SEEDS) - set(seeds))
    if missing:
        raise RuntimeError(f"{environment.key}/rl_sgf: missing seeds {missing}")
    for seed, seed_dir in seeds.items():
        for rel in (RL_SGF.evaluation_path, RL_SGF.exploration_path):
            if not (seed_dir / rel).is_file():
                raise RuntimeError(f"missing {seed_dir / rel}")
        final = int(pd.read_csv(seed_dir / RL_SGF.evaluation_path, usecols=["timestep"])["timestep"].max())
        if final < environment.nominal_budget:
            raise RuntimeError(
                f"{environment.key}/rl_sgf/seed{seed}: final evaluation {final} is before the "
                f"nominal budget {environment.nominal_budget}"
            )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rl-sgf-root", type=Path, default=RL_SGF_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bin-size", type=int, default=canon.ROLLOUT_SIZE)
    parser.add_argument("--ci-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--layout", action="append", choices=canon.LAYOUTS,
        help="Figure layout; repeat for several (default: both).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    layouts = tuple(dict.fromkeys(args.layout or canon.LAYOUTS))
    expected = len(canon.EXPECTED_SEEDS)
    evaluation_parts: list[pd.DataFrame] = []
    exploration_parts: list[pd.DataFrame] = []
    for environment in canon.ENVIRONMENTS:
        canon.validate_sources(environment)
        rl_env = rl_sgf_environment(environment, args.rl_sgf_root)
        validate_rl_sgf(rl_env)
        for spec in SELECTED_METHODS:
            if spec is RL_SGF:
                evaluation = aggregate_rl_sgf_evaluation(rl_env)
                exploration = canon.aggregate_exploration(spec, environment=rl_env, bin_size=args.bin_size)
                # Report under the real environment roots, not the RL-SGF substitute.
                evaluation["plot_horizon_timesteps"] = environment.horizon
            else:
                evaluation = canon.aggregate_evaluation(spec, environment=environment)
                exploration = canon.aggregate_exploration(spec, environment=environment, bin_size=args.bin_size)
            evaluation_parts.append(canon._common_seed_support(evaluation, expected=expected, allow_incomplete=False))
            exploration_parts.append(canon._common_seed_support(exploration, expected=expected, allow_incomplete=False))
    evaluation = pd.concat(evaluation_parts, ignore_index=True)
    exploration = pd.concat(exploration_parts, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(args.output_dir / "evaluation_curve_aggregates.csv", index=False)
    exploration.to_csv(args.output_dir / "exploration_curve_aggregates.csv", index=False)
    metadata = {
        "environments": {
            env.key: {
                "label": env.label,
                "nominal_budget_timesteps": env.nominal_budget,
                "plot_horizon_timesteps": env.horizon,
                "baseline_root": str(env.baseline_root.resolve()),
                "pspo_root": str(env.adaptive_root.resolve()),
                "rl_sgf_root": str((args.rl_sgf_root / env.key / "rl_sgf").resolve()),
            }
            for env in canon.ENVIRONMENTS
        },
        "methods": {
            spec.key: {
                "label": spec.label,
                "evaluation_variant": "shielded/deployed" if spec.key == "ppo_shield" else "unshielded/nominal",
                "initialisation": "safe policy (warm start)" if spec.key in {"pspo", "rl_sgf"} else "random",
            }
            for spec in SELECTED_METHODS
        },
        "expected_seeds": list(canon.EXPECTED_SEEDS),
        "band": f"mean +/- {args.ci_multiplier:g} standard errors across seeds",
        "evaluation_definition": "Periodic deterministic (greedy) policy evaluation, 20 episodes per point.",
        "rl_sgf_evaluation_alignment": (
            "Last observation carried forward onto the 2048-step grid; see "
            "max_hold_lag_timesteps in evaluation_curve_aggregates.csv."
        ),
        "exploration_definition": "Per-seed means of completed training episodes binned by end timestep.",
        "exploration_safety_definition": "safe_trajectory when recorded; otherwise logical negation of violated.",
        "common_support_note": "Every plotted point contains all ten seeds.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # The canonical plotting helpers iterate the module-level METHODS tuple; swap it
    # only now, after validate_sources has checked the canonical methods' inputs.
    canon.METHODS = SELECTED_METHODS
    # Per-panel safety zoom: wrap the canonical panel drawers (the canonical script's own
    # figures keep their fixed 0-1 safety axis).
    canon._draw_environment_row = _draw_row_zoomed
    canon._draw_environment_column = _draw_column_zoomed
    canon._apply_style()
    for layout in layouts:
        for frame, kind in ((evaluation, "evaluation"), (exploration, "exploration")):
            canon.save_figure(
                frame,
                kind=kind,
                environments=canon.ENVIRONMENTS,
                output_dir=args.output_dir,
                ci_multiplier=args.ci_multiplier,
                layout=layout,
            )
    print(f"Saved learning curves to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
