#!/usr/bin/env python3
"""MiniPacman PSPO reward/safety learning curves around safety projections.

Reward and safety share one axis because both lie in [0, 1] in MiniPacman.
Vertical markers show train-phase safety enforcement, which is configured once
per 100 PPO updates. Curves and bands are the across-seed mean +/- two standard
errors by default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paper_style import (  # noqa: E402
    METHOD_COLORS,
    TEXT_WIDTH_IN,
    apply_paper_style,
    save_paper_figure,
)
from plot_extended_budget_learning_curves import (  # noqa: E402
    ENVIRONMENT_BY_KEY,
    EXPECTED_SEEDS,
    _format_timesteps,
)

ENVIRONMENT = ENVIRONMENT_BY_KEY["mini_pacman"]
DEFAULT_RUN_ROOT = ENVIRONMENT.adaptive_root
DEFAULT_OUTPUT_DIR = (
    REPO
    / "projects/safe_policy_optimisation/figures"
    / "pspo_vs_rl_baselines_learning_curves_all_envs"
)
CURVE_REL_PATH = Path("learning_curves/evaluation_unshielded_summary.csv")

REWARD_COLOR = METHOD_COLORS["pspo"]
SAFETY_COLOR = METHOD_COLORS["ppo_shield"]
PROJECTION_COLOR = "#6b6a65"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--name",
        default="minipacman_pspo_reward_safety_projection_dynamics",
        help="Output file stem.",
    )
    parser.add_argument(
        "--ci-multiplier",
        type=float,
        default=2.0,
        help="Multiplier on standard errors (default: 2).",
    )
    return parser.parse_args(argv)


def _load_seed_curves(run_root: Path) -> tuple[pd.DataFrame, dict, int]:
    frames: list[pd.DataFrame] = []
    reference_config: dict | None = None
    projection_counts: set[int] = set()

    for seed in EXPECTED_SEEDS:
        seed_dir = run_root / f"seed{seed}"
        curve_path = seed_dir / CURVE_REL_PATH
        config_path = seed_dir / "config.json"
        summary_path = seed_dir / "summary.json"
        if not curve_path.is_file():
            raise FileNotFoundError(f"missing {curve_path}")
        if not config_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"missing config/summary under {seed_dir}")

        config = json.loads(config_path.read_text())
        if reference_config is None:
            reference_config = config
        adaptive = config["adaptive"]
        reference_adaptive = reference_config["adaptive"]
        if (
            int(adaptive["frequency"]) != int(reference_adaptive["frequency"])
            or adaptive["granularity"] != reference_adaptive["granularity"]
            or int(config["training_hyperparameters"]["n_steps"])
            != int(reference_config["training_hyperparameters"]["n_steps"])
        ):
            raise ValueError("MiniPacman seeds use inconsistent enforcement schedules")

        diagnostics = json.loads(summary_path.read_text())["adaptive_diagnostics"]
        projection_counts.add(int(diagnostics["phase_projections"]))

        frame = pd.read_csv(
            curve_path,
            usecols=["timestep", "mean_total_reward", "safety_rate"],
        )
        frame["seed"] = seed
        frames.append(frame)

    if len(projection_counts) != 1:
        raise ValueError(f"inconsistent phase-projection counts: {projection_counts}")
    assert reference_config is not None
    return pd.concat(frames, ignore_index=True), reference_config, projection_counts.pop()


def _aggregate_curves(seed_curves: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        seed_curves.groupby("timestep", sort=True)
        .agg(
            reward_mean=("mean_total_reward", "mean"),
            reward_sd=("mean_total_reward", "std"),
            safety_mean=("safety_rate", "mean"),
            safety_sd=("safety_rate", "std"),
            seed_count=("seed", "nunique"),
        )
        .reset_index()
    )
    expected = len(EXPECTED_SEEDS)
    if not aggregate["seed_count"].eq(expected).all():
        raise RuntimeError("not every MiniPacman checkpoint has all ten seeds")
    aggregate["reward_sem"] = aggregate["reward_sd"].fillna(0.0) / math.sqrt(
        expected
    )
    aggregate["safety_sem"] = aggregate["safety_sd"].fillna(0.0) / math.sqrt(
        expected
    )
    return aggregate


def _projection_schedule(config: dict, max_timestep: int) -> tuple[list[int], int]:
    adaptive = config["adaptive"]
    if adaptive["granularity"] != "train_phase":
        raise ValueError("expected MiniPacman safety granularity to be train_phase")
    frequency = int(adaptive["frequency"])
    rollout_steps = int(config["training_hyperparameters"]["n_steps"])
    interval = frequency * rollout_steps
    scheduled = list(range(interval, max_timestep + 1, interval))
    return scheduled, interval


def _post_projection_changes(
    seed_curves: pd.DataFrame,
    projection_timesteps: list[int],
    eval_interval: int,
    ci_multiplier: float,
) -> tuple[float, float, float, float]:
    """Mean per-seed next-checkpoint change and its scaled SE over seeds."""
    per_seed: list[tuple[float, float]] = []
    for seed in EXPECTED_SEEDS:
        curve = seed_curves.loc[seed_curves["seed"] == seed].set_index("timestep")
        reward_changes: list[float] = []
        safety_changes: list[float] = []
        for timestep in projection_timesteps:
            after = timestep + eval_interval
            if timestep not in curve.index or after not in curve.index:
                continue
            reward_changes.append(
                float(curve.loc[after, "mean_total_reward"])
                - float(curve.loc[timestep, "mean_total_reward"])
            )
            safety_changes.append(
                float(curve.loc[after, "safety_rate"])
                - float(curve.loc[timestep, "safety_rate"])
            )
        if not reward_changes:
            raise RuntimeError(f"no projection-adjacent evaluations for seed {seed}")
        per_seed.append((np.mean(reward_changes), np.mean(safety_changes)))

    values = np.asarray(per_seed, dtype=float)
    means = values.mean(axis=0)
    errors = ci_multiplier * values.std(axis=0, ddof=1) / math.sqrt(len(values))
    return float(means[0]), float(errors[0]), float(means[1]), float(errors[1])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ci_multiplier < 0:
        raise SystemExit("--ci-multiplier must be non-negative")

    seed_curves, config, projection_count = _load_seed_curves(args.run_root)
    aggregate = _aggregate_curves(seed_curves)
    timestep = aggregate["timestep"].to_numpy(dtype=float)
    max_timestep = int(timestep.max())
    projections, projection_interval = _projection_schedule(config, max_timestep)
    eval_interval = int(config["curve_eval_freq"])
    if projection_count != len(projections) + 1:
        raise RuntimeError(
            f"expected {len(projections)} scheduled projections plus one final "
            f"flush, but summaries report {projection_count} projections"
        )

    reward_change, reward_change_2se, safety_change, safety_change_2se = (
        _post_projection_changes(
            seed_curves, projections, eval_interval, args.ci_multiplier
        )
    )

    apply_paper_style()
    fig, axis = plt.subplots(
        figsize=(TEXT_WIDTH_IN, 3.15), layout="constrained", facecolor="#fcfcfb"
    )

    reward_mean = aggregate["reward_mean"].to_numpy(dtype=float)
    reward_error = args.ci_multiplier * aggregate["reward_sem"].to_numpy(dtype=float)
    safety_mean = aggregate["safety_mean"].to_numpy(dtype=float)
    safety_error = args.ci_multiplier * aggregate["safety_sem"].to_numpy(dtype=float)

    axis.fill_between(
        timestep,
        np.clip(reward_mean - reward_error, 0.0, 1.0),
        np.clip(reward_mean + reward_error, 0.0, 1.0),
        color=REWARD_COLOR,
        alpha=0.14,
        linewidth=0,
    )
    axis.plot(
        timestep,
        reward_mean,
        color=REWARD_COLOR,
        linewidth=1.25,
        label="Mean total reward",
        zorder=3,
    )
    axis.fill_between(
        timestep,
        np.clip(safety_mean - safety_error, 0.0, 1.0),
        np.clip(safety_mean + safety_error, 0.0, 1.0),
        color=SAFETY_COLOR,
        alpha=0.13,
        linewidth=0,
    )
    axis.plot(
        timestep,
        safety_mean,
        color=SAFETY_COLOR,
        linewidth=1.15,
        linestyle=(0, (4, 1.5)),
        label="Safety rate",
        zorder=3,
    )

    for projection_timestep in projections:
        axis.axvline(
            projection_timestep,
            color=PROJECTION_COLOR,
            linestyle=(0, (1, 2)),
            linewidth=0.65,
            alpha=0.65,
            zorder=1,
        )
        axis.plot(
            projection_timestep,
            1.035,
            marker="v",
            markersize=3.0,
            color=PROJECTION_COLOR,
            clip_on=False,
            zorder=4,
        )

    axis.set_xlim(0, max_timestep)
    axis.set_ylim(-0.02, 1.075)
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axis.xaxis.set_major_formatter(FuncFormatter(_format_timesteps))
    axis.set_xlabel("Training timestep")
    axis.set_ylabel("Evaluation mean / rate")
    axis.set_title(
        "MiniPacman PSPO: reward and safety between projections",
        fontsize=9,
        fontweight="bold",
        pad=8,
    )
    axis.text(
        0.5,
        1.015,
        f"Safety enforcement every 100 PPO updates ({projection_interval:,} steps)",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#52514e",
    )

    delta_text = (
        "Next evaluation after scheduled projection\n"
        rf"$\Delta$ reward = {reward_change:+.2f} $\pm$ {reward_change_2se:.2f}"
        "     "
        rf"$\Delta$ safety = {safety_change:+.2f} $\pm$ {safety_change_2se:.2f}"
    )
    axis.text(
        0.018,
        0.045,
        delta_text,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#c3c2b7",
            "alpha": 0.92,
            "linewidth": 0.5,
        },
        zorder=5,
    )

    handles = [
        Line2D([], [], color=REWARD_COLOR, linewidth=1.5, label="Mean total reward"),
        Line2D(
            [], [], color=SAFETY_COLOR, linewidth=1.4, linestyle=(0, (4, 1.5)),
            label="Safety rate",
        ),
        Line2D(
            [], [], color=PROJECTION_COLOR, linewidth=0.8,
            linestyle=(0, (1, 2)), marker="v", markersize=3,
            label="Scheduled projection",
        ),
    ]
    axis.legend(
        handles=handles,
        loc="lower right",
        ncol=1,
        frameon=True,
        framealpha=0.92,
        edgecolor="#c3c2b7",
        handlelength=2.2,
    )
    axis.spines[["top", "right"]].set_visible(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.name
    save_paper_figure(fig, stem)
    plt.close(fig)
    print(
        f"Saved {stem}.pdf and {stem}.png; "
        f"scheduled projections={len(projections)}, final flush=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
