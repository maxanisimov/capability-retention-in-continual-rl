#!/usr/bin/env python3
"""Reproduce raw-reward curves from completed, reward-shaped FrozenLake runs.

No training or evaluation is run by this script. Time-axis points compare the
same training progress: x is mean elapsed time across seeds, and the reward
band is across seed means at that progress (not a fixed-wall-time interval).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ccl-frozenlake-efficiency-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[3]
PROJECT = REPO / "projects/safe_policy_optimisation"
RUNS = PROJECT / "artifacts/paper_2503_07671/runs"
OUTPUT = PROJECT / "figures/frozenlake_ppo_pspo_reward_efficiency_20261008"
COHORTS = {
    16: {
        "ppo": "frozenlake16_shaping_ppo_t204800_20261008T150953Z",
        "pspo": "frozenlake16_shaping_pspo_segment_verify_first_t204800_20261008T123000Z",
    },
    32: {
        "ppo": "frozenlake32_shaping_ppo_t204800_20261008T151817Z",
        "pspo": "frozenlake32_shaping_pspo_segment_verify_first_t204800_20261008T123000Z",
    },
    128: {
        "ppo": "frozenlake128_shaping_ppo_pspo_t400k_20261008T085100Z",
        "pspo": "frozenlake128_shaping_pspo_segment_verify_first_t1024000_20261008T124400Z",
    },
}
HORIZONS = {16: 204800, 32: 204800, 128: 400000}
COLORS = {"ppo": "#0072B2", "pspo": "#D55E00"}
LABELS = {"ppo": "PPO", "pspo": "PSPO (line segment, verify-first)"}


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def flag(value):
    return float(str(value).lower() in ("true", "1"))


def mean_two_se(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Require at least two finite seed-level values")
    return {
        "mean": float(values.mean()),
        "two_se": float(2 * values.std(ddof=1) / np.sqrt(len(values))),
        "seed_count": len(values),
    }


def rollout_times(text, horizon, final_time):
    """Approximate completed-window timestamps from SB3's integer-second log.

    The measured final curve time anchors the final point, including PSPO's
    final certification. Intermediate episodes have no saved wall timestamps.
    """
    matches = re.findall(
        r"\|\s*time_elapsed\s*\|\s*(\d+)\s*\|\s*\n"
        r"\|\s*total_timesteps\s*\|\s*(\d+)\s*\|",
        text,
    )
    points = [(0, 0.0)] + [
        (int(step), float(seconds))
        for seconds, step in matches
        if 0 < int(step) < horizon
    ]
    points.append((horizon, float(final_time)))
    steps, seconds = np.asarray(points, dtype=float).T
    if len(matches) < 2 or np.any(np.diff(steps) <= 0) or np.any(np.diff(seconds) < 0):
        raise ValueError("Incomplete or nonmonotone rollout timing log")
    return steps, seconds


def bin_training(rows, edges):
    """Per-seed raw completed-episode means in left-open/right-closed windows."""
    output = []
    for left, right in zip(edges[:-1], edges[1:]):
        selected = [r for r in rows if left < int(r["end_timestep"]) <= right]
        if not selected:
            raise ValueError(f"No completed episodes in window ({left}, {right}]")
        output.append(
            {
                "channel": "exploration",
                "timestep": right,
                "window_start": left,
                "window_end": right,
                "episodes": len(selected),
                "reward": float(np.mean([float(r["raw_reward"]) for r in selected])),
                "goal": float(np.mean([flag(r["goal_reached"]) for r in selected])),
                "safety": float(
                    np.mean([flag(r["safe_trajectory"]) for r in selected])
                ),
            }
        )
    return output


def checked_metrics(path):
    metrics = read_json(path)
    if (
        metrics["reward_shaping_enabled"]
        or metrics["evaluation_policy"] != "greedy_unshielded"
    ):
        raise ValueError(f"Not raw, unshielded greedy evaluation: {path}")
    return {
        "episodes": metrics["eval_episodes"],
        "reward": float(metrics["reward"]["mean_total_reward"]),
        "goal": float(metrics["success"]["success_rate"]),
        "safety": float(metrics["safety"]["safety_rate"]),
    }


def load_data(runs=RUNS):
    points, details, hashes = [], {}, {}
    for size, methods in COHORTS.items():
        horizon = HORIZONS[size]
        reference = None
        details[size] = {}
        for method, cohort in methods.items():
            directory = runs / cohort / method
            seed_details = []
            for seed in range(10):
                source = directory / f"seed{seed}"
                config = read_json(source / "config.json")
                summary = read_json(source / "summary.json")
                total = int(summary["final_timesteps"])
                if (
                    config["seed"] != seed
                    or config["env_kwargs"]["size"] != size
                    or total < horizon
                ):
                    raise ValueError(f"Unexpected run settings: {source}")
                comparable = {
                    "architecture": {
                        k: config["architecture"][k] for k in ("hidden_dim", "n_hidden")
                    },
                    "env_kwargs": config["env_kwargs"],
                    "max_episode_steps": config["max_episode_steps"],
                    "training_hyperparameters": config["training_hyperparameters"],
                    "shaping": {
                        k: config["shaping"][k]
                        for k in (
                            "enabled",
                            "gamma",
                            "scale",
                            "timeout_mode",
                            "training_only",
                        )
                    },
                    "curve_eval_freq": config["curve_eval_freq"],
                    "curve_eval_episodes": config["curve_eval_episodes"],
                }
                if reference is None:
                    reference = comparable
                if comparable != reference:
                    raise ValueError(f"Unmatched environment/hyperparameters: {source}")
                if method == "pspo":
                    if config["pspo_variant"] != "line_segment_verify_first":
                        raise ValueError(f"Unexpected PSPO variant: {source}")
                    if summary["final_exact_all_state_alignment"] != 1.0:
                        raise ValueError(f"Uncertified final PSPO checkpoint: {source}")
                if size in (16, 32) and total != horizon:
                    raise ValueError(f"Unexpected matched training budget: {source}")
                inputs = [
                    source / name
                    for name in (
                        "config.json",
                        "summary.json",
                        "initial_metrics.json",
                        "metrics.json",
                        "training_episodes.csv",
                        "learning_curves/evaluation_unshielded_summary.csv",
                    )
                ]
                log = runs / cohort / "_logs" / f"{method}_seed{seed}.log"
                inputs.append(log)
                for path in inputs:
                    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                training = read_csv(source / "training_episodes.csv")
                for row in training:
                    expected = flag(row["goal_reached"]) - config["env_kwargs"][
                        "step_penalty"
                    ] * int(row["length"])
                    if not np.isclose(
                        float(row["raw_reward"]), expected, atol=1e-9, rtol=0
                    ):
                        raise ValueError(f"Original reward identity failed: {source}")
                periodic = read_csv(
                    source / "learning_curves/evaluation_unshielded_summary.csv"
                )
                periodic = [r for r in periodic if int(r["timestep"]) <= horizon]
                expected_steps = list(range(20000, horizon + 1, 20000))
                if size in (16, 32):
                    expected_steps.append(horizon)
                if [int(r["timestep"]) for r in periodic] != expected_steps:
                    raise ValueError(f"Missing/duplicate periodic evaluation: {source}")
                final_time = float(periodic[-1]["training_wall_time_s"])
                log_steps, log_seconds = rollout_times(
                    log.read_text(), horizon, final_time
                )
                edges = [0, *range(20000, horizon, 20000), horizon]
                run_points = bin_training(training, edges)
                for point in run_points:
                    point["elapsed_seconds"] = float(
                        np.interp(point["timestep"], log_steps, log_seconds)
                    )
                for row in periodic:
                    # Do not duplicate the final checkpoint's 10- and 100-episode scores.
                    if size in (16, 32) and int(row["timestep"]) == horizon:
                        continue
                    if int(row["episodes"]) != 10:
                        raise ValueError("Require ten-episode periodic evaluation")
                    run_points.append(
                        {
                            "channel": "evaluation",
                            "timestep": int(row["timestep"]),
                            "elapsed_seconds": float(row["training_wall_time_s"]),
                            "episodes": int(row["episodes"]),
                            "reward": float(row["mean_total_reward"]),
                            "goal": float(row["success_rate"]),
                            "safety": float(row["safety_rate"]),
                        }
                    )
                run_points.append(
                    {
                        "channel": "evaluation",
                        "timestep": 0,
                        "elapsed_seconds": 0.0,
                        **checked_metrics(source / "initial_metrics.json"),
                    }
                )
                final = checked_metrics(source / "metrics.json")
                if final["episodes"] != 100:
                    raise ValueError(
                        "Require 100 episodes in the separate final evaluation"
                    )
                if size in (16, 32):
                    run_points.append(
                        {
                            "channel": "final",
                            "timestep": horizon,
                            "elapsed_seconds": final_time,
                            **final,
                        }
                    )
                points.extend(
                    {"size": size, "method": method, "seed": seed, **p}
                    for p in run_points
                )
                seed_details.append(
                    {
                        "seed": seed,
                        "source": str(source),
                        "full_training_steps": total,
                        "requested_timesteps": config["requested_timesteps"],
                        "full_training_seconds_excluding_curve_evaluation": summary[
                            "training_seconds_excluding_curve_evaluation"
                        ],
                        "elapsed_seconds_at_plotted_horizon": final_time,
                        "full_budget_final_metrics": final,
                    }
                )
            details[size][method] = {
                "comparable_settings": reference,
                "seeds": seed_details,
            }
    return points, details, hashes


def aggregate(points):
    grouped = {}
    for point in points:
        key = tuple(point[k] for k in ("size", "method", "channel", "timestep"))
        grouped.setdefault(key, []).append(point)
    output = []
    for (size, method, channel, step), selected in sorted(grouped.items()):
        if {p["seed"] for p in selected} != set(range(10)) or len(selected) != 10:
            raise ValueError("Every plotted point must contain all ten distinct seeds")
        row = {"size": size, "method": method, "channel": channel, "timestep": step}
        for metric in ("reward", "goal", "safety", "elapsed_seconds"):
            for key, value in mean_two_se([p[metric] for p in selected]).items():
                row[f"{metric}_{key}"] = value
        output.append(row)
    return output


def draw_panel(ax, rows, size, channel, clock=False):
    for method in ("ppo", "pspo"):
        selected = [
            r
            for r in rows
            if r["size"] == size and r["method"] == method and r["channel"] == channel
        ]
        x = np.array(
            [
                r["elapsed_seconds_mean"] / 60 if clock else r["timestep"] / 1000
                for r in selected
            ]
        )
        y = np.array([r["reward_mean"] for r in selected])
        error = np.array([r["reward_two_se"] for r in selected])
        ax.fill_between(
            x, y - error, y + error, color=COLORS[method], alpha=0.16, linewidth=0
        )
        ax.plot(
            x,
            y,
            color=COLORS[method],
            linewidth=1.6,
            linestyle="--" if method == "ppo" else "-",
            marker="o",
            markersize=2.6,
        )
        if channel == "evaluation":
            for row in rows:
                if (
                    row["size"] == size
                    and row["method"] == method
                    and row["channel"] == "final"
                ):
                    ax.errorbar(
                        row["elapsed_seconds_mean"] / 60
                        if clock
                        else row["timestep"] / 1000,
                        row["reward_mean"],
                        yerr=row["reward_two_se"],
                        color=COLORS[method],
                        marker="D",
                        markersize=7 if method == "ppo" else 4.5,
                        capsize=3,
                        linestyle="none",
                        zorder=6,
                        markerfacecolor="none" if method == "ppo" else COLORS[method],
                        markeredgecolor=COLORS[method],
                        markeredgewidth=1,
                    )
    ax.axhline(0, color="0.6", linewidth=0.6, zorder=0)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(
        "Mean elapsed time incl. monitoring (min)"
        if clock
        else "Training environment steps (thousands)"
    )
    ax.set_ylabel(f"{size}×{size}\nOriginal total reward")
    ax.set_ylim(bottom={16: -0.72, 32: -1.4, 128: -5.45}[size], top=1.05)
    if not clock:
        ax.set_xlim(0, HORIZONS[size] / 1000 * 1.045)
    else:
        ax.set_xlim(left=0)


def save_plot(rows, output, sizes, clock, name, prefix=False):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7), layout="constrained")
    for i in range(2):
        for j, channel in enumerate(("exploration", "evaluation")):
            size = sizes[i] if not prefix else 128
            draw_panel(
                axes[i, j], rows, size, channel, clock=bool(i) if prefix else clock
            )
            if i == 0:
                axes[i, j].set_title(
                    "Exploration (training episodes)"
                    if j == 0
                    else "Evaluation (greedy, no shield)"
                )
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[m],
            linestyle="--" if m == "ppo" else "-",
            label=LABELS[m],
        )
        for m in ("ppo", "pspo")
    ]
    if not prefix:
        handles.append(
            Line2D(
                [0],
                [0],
                color="0.35",
                marker="D",
                linestyle="none",
                label="Final 100-episode evaluation",
            )
        )
    legend_title = (
        "FrozenLake 128×128 — shared first 400,000 steps (different full budgets)"
        if prefix
        else "FrozenLake — matched 204,800 steps per method"
    ) + "\nOriginal reward; ten seeds, mean ± 2 SE"
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=2 if not prefix else 2,
        frameon=False,
        title=legend_title,
        title_fontsize=9,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=200)
    plt.close(fig)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def report(rows, details):
    result = {}
    for size in COHORTS:
        result[size] = {}
        for method in ("ppo", "pspo"):
            selected = [r for r in rows if r["size"] == size and r["method"] == method]
            evaluations = [r for r in selected if r["channel"] == "evaluation"]
            exploration = [r for r in selected if r["channel"] == "exploration"]
            threshold = next((r for r in evaluations if r["reward_mean"] >= 0.8), None)
            finals = [r for r in selected if r["channel"] == "final"]
            result[size][method] = {
                "initial_evaluation": evaluations[0],
                "last_periodic_evaluation": evaluations[-1],
                "last_exploration_window": exploration[-1],
                "first_periodic_mean_reward_at_least_0_8": threshold,
                "final_100_episode_evaluation_at_matched_budget": finals[0]
                if finals
                else None,
            }
            if size in (16, 32):
                result[size][method]["training_seconds_excluding_curve_evaluation"] = (
                    mean_two_se(
                        [
                            s["full_training_seconds_excluding_curve_evaluation"]
                            for s in details[size][method]["seeds"]
                        ]
                    )
                )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    points, details, hashes = load_data(args.runs_root)
    rows = aggregate(points)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    save_plot(rows, args.output_dir, [16, 32], False, "frozenlake16_32_reward_vs_steps")
    save_plot(
        rows, args.output_dir, [16, 32], True, "frozenlake16_32_reward_vs_walltime"
    )
    save_plot(
        rows,
        args.output_dir,
        [128, 128],
        False,
        "frozenlake128_shared_prefix_reward_efficiency",
        prefix=True,
    )
    write_csv(args.output_dir / "seed_curve_points.csv", points)
    write_csv(args.output_dir / "aggregate_curve_points.csv", rows)
    analysis = {
        "statistics": "Mean +/- two sample standard errors over ten seed-level means at each shared training progress point.",
        "pspo_variant": "line_segment_verify_first",
        "horizons": HORIZONS,
        "cohorts": details,
        "results": report(rows, details),
        "source_sha256": hashes,
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n"
    )
    (
        args.output_dir / "README.md"
    ).write_text("""# FrozenLake PPO/PSPO raw-reward learning curves

Reproduce from the repository root:
```
.venv/bin/python projects/safe_policy_optimisation/scripts/plot_frozenlake_ppo_pspo_reward_efficiency.py
```

## What is compared

All curves use original, **unshaped, undiscounted episode total reward**,
R = goal indicator - 0.001 × episode length. Potential-based reward shaping
is used only for training. PSPO here means **line segment, verify-first**,
not the earlier orthotope or deleted goal-guided initialisation cohorts.
PPO has random initialisation and unshielded training; PSPO has safety-only
initialisation and shielded exploration. This compares the complete methods.

16×16 and 32×32 have matched 204,800 actual training steps and seeds 0–9.
The 128×128 supplementary figure compares only the shared first 400,000
steps: PPO's full run has 401,408 steps, PSPO's has 1,024,000. Full-budget
PSPO metrics are retained in analysis.json for provenance but NOT plotted.
There is no matching completed ten-seed shaped PPO cohort for 64×64 or
256×256, so these layouts are not included. Environment, architecture width,
training hyperparameters, shaping settings, and evaluation schedule are checked.

## Curves and uncertainty

Exploration: mean reward of episodes completed in non-overlapping 20,000-step
windows, first averaged separately within each seed. The final matched-layout
window is shorter (200,000–204,800). A point is plotted at its window END;
unfinished episodes are not included and empty windows are never zero-filled.
This is executed training reward (shielded for PSPO), not the shaped objective.

Evaluation: initial ten-episode evaluation plus periodic ten-episode greedy,
unshielded evaluations every 20,000 steps. At the matched-layout final checkpoint,
a separate diamond shows the saved **100-episode** evaluation; the duplicate
ten-episode final evaluation is omitted. PSPO final diamonds are post-enforcement
and all-winning-state certified. Periodic proposals need not be certified and
can fluctuate; the plotted line does not imply certification between checkpoints.
Initial/final evaluations and periodic evaluations use different reset-seed ranges.
Shaded bands and diamond error bars are mean ± TWO sample standard errors across
ten training-seed means (sample SD / sqrt(10) × 2), never pooled episodes.
No additional smoothing, clipping of rewards, or interpolation of rewards is used.

## Time axis: limitations

The x coordinate is the **mean elapsed time to the same training progress**
across ten seeds. The reward band is at that progress, NOT a point-wise band
at identical wall-clock time. It includes optimisation, safety enforcement,
and periodic monitoring evaluations; it excludes initialisation and the
separate final 100-episode evaluation. It is NOT the evaluation/inference time.
Evaluation elapsed times come from training_wall_time_s in saved curve CSVs.
Exploration has no episode timestamps: window-end elapsed times are approximate,
interpolated from per-rollout console logs (integer-second resolution), anchored
at the saved final curve timestamp. No extrapolation beyond the plotted horizon
is used. The final timestamp also includes final PSPO safety enforcement.
The sweeps ran separately under different shared-host workloads: these figures
are observational wall-clock comparisons, not a controlled speed benchmark.

The JSON records all cohorts, settings, per-seed full budgets, derived summaries,
and hashes of source files. CSVs retain seed-level and aggregated plot points.
""")
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "results": analysis["results"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
