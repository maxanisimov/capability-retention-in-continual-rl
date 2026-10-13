#!/usr/bin/env python3
"""Show raw FrozenLake learning and distinguish PSPO proposals from projection.

Statistics are across training-seed means, never pooled evaluation episodes.
Training uses completed-episode means in non-overlapping 20k-step bins.
Periodic evaluations and the final 100-episode evaluation are kept distinct.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ccl-frozenlake-curves-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / (
    "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs/"
    "frozenlake128_shaping_ppo_pspo_t400k_20261008T085100Z"
)
DEFAULT_OUTPUT = (
    REPO
    / "projects/safe_policy_optimisation/figures/frozenlake128_shaping_learning_curves_20261008"
)
COLORS = {"ppo": "#424242", "pspo": "#009E73"}
METRICS = ("reward", "goal", "safety")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def flag(value) -> float:
    return float(str(value).lower() in ("true", "1"))


def mean_two_se(values) -> dict:
    xs = np.asarray(values, dtype=float)
    if not xs.size or not np.isfinite(xs).all():
        raise ValueError("Require nonempty finite seed-level observations")
    return {
        "mean": float(xs.mean()),
        "two_standard_errors": float(2 * xs.std(ddof=1) / np.sqrt(xs.size))
        if xs.size > 1
        else None,
        "seed_count": int(xs.size),
    }


def metrics_values(metrics: dict) -> dict:
    return {
        "reward": float(metrics["reward"]["mean_total_reward"]),
        "goal": float(metrics["success"]["success_rate"]),
        "safety": float(metrics["safety"]["safety_rate"]),
    }


def bin_training(rows: list[dict], edges: list[int]) -> list[dict]:
    """Average completed episodes per seed; never zero-fill an empty window."""
    output = []
    for left, right in zip(edges[:-1], edges[1:]):
        selected = [r for r in rows if left < int(r["end_timestep"]) <= right]
        if not selected:
            continue
        output.append(
            {
                "timestep": (left + right) / 2,
                "window_start": left,
                "window_end": right,
                "episodes": len(selected),
                "reward": float(np.mean([float(r["raw_reward"]) for r in selected])),
                "shaped_reward": float(
                    np.mean([float(r["shaped_reward"]) for r in selected])
                ),
                "goal": float(np.mean([flag(r["goal_reached"]) for r in selected])),
                "safety": float(
                    np.mean([flag(r["safe_trajectory"]) for r in selected])
                ),
            }
        )
    return output


def aggregate_points(points: list[dict]) -> list[dict]:
    output = []
    keys = sorted({(p["method"], p["channel"], p["timestep"]) for p in points})
    for method, channel, timestep in keys:
        selected = [
            p
            for p in points
            if (p["method"], p["channel"], p["timestep"]) == (method, channel, timestep)
        ]
        if len({p["seed"] for p in selected}) != len(selected):
            raise ValueError("Duplicate seed at a learning-curve point")
        row = {
            "method": method,
            "channel": channel,
            "timestep": timestep,
            "seed_count": len(selected),
        }
        for metric in METRICS:
            stats = mean_two_se([p[metric] for p in selected])
            row[metric + "_mean"] = stats["mean"]
            row[metric + "_two_se"] = stats["two_standard_errors"]
        output.append(row)
    return output


def load_data(root: Path) -> tuple[list[dict], dict]:
    status = read_json(root / "status.json")
    if status["complete"] != 20 or status["failed"]:
        raise ValueError("Require the fully completed twenty-run PPO/PSPO cohort")
    points, details = (
        [],
        {
            "root": str(root),
            "uncertainty": "mean +/- two sample standard errors over training seeds",
        },
    )
    for method in ("ppo", "pspo"):
        seed_details = []
        for seed in range(10):
            directory = root / method / f"seed{seed}"
            config = read_json(directory / "config.json")
            summary = read_json(directory / "summary.json")
            total = int(summary["final_timesteps"])
            if (
                total != 401408
                or config["requested_timesteps"] != 400000
                or config["env_kwargs"]["size"] != 128
            ):
                raise ValueError(f"Unmatched production run: {directory}")
            edges = [*range(0, 400000, 20000), total]
            training = read_csv(directory / "training_episodes.csv")
            for row in training:
                expected = flag(row["goal_reached"]) - config["env_kwargs"][
                    "step_penalty"
                ] * int(row["length"])
                if not np.isclose(float(row["raw_reward"]), expected, atol=1e-9):
                    raise ValueError(f"Raw reward identity failed: {directory}")
            for row in bin_training(training, edges):
                points.append(
                    {"method": method, "seed": seed, "channel": "exploration", **row}
                )
            periodic = read_csv(
                directory / "learning_curves/evaluation_unshielded_summary.csv"
            )
            timepoints = [int(r["timestep"]) for r in periodic]
            if len(set(timepoints)) != len(timepoints):
                raise ValueError(f"Duplicate evaluation timestep: {directory}")
            for row in periodic:
                if int(row["timestep"]) >= total:
                    continue  # Final ten-episode curve is not the final 100-episode score.
                points.append(
                    {
                        "method": method,
                        "seed": seed,
                        "channel": "evaluation",
                        "timestep": int(row["timestep"]),
                        "episodes": int(row["episodes"]),
                        "reward": float(row["mean_total_reward"]),
                        "goal": float(row["success_rate"]),
                        "safety": float(row["safety_rate"]),
                    }
                )
            for name, channel, timestep in (
                ("initial_metrics.json", "evaluation", 0),
                ("metrics.json", "final_100_episodes", total),
            ):
                metrics = read_json(directory / name)
                if (
                    metrics["reward_shaping_enabled"]
                    or metrics["evaluation_policy"] != "greedy_unshielded"
                ):
                    raise ValueError(f"Unexpected evaluation: {directory / name}")
                if name == "metrics.json" and metrics["eval_episodes"] != 100:
                    raise ValueError("Final evaluation must use 100 episodes")
                points.append(
                    {
                        "method": method,
                        "seed": seed,
                        "channel": channel,
                        "timestep": timestep,
                        "episodes": metrics["eval_episodes"],
                        **metrics_values(metrics),
                    }
                )
            late = [r for r in periodic if 300000 <= int(r["timestep"]) <= 380000]
            if len(late) != 5:
                raise ValueError("Require five shared late periodic evaluations")
            slope = float(
                np.polyfit(
                    [int(r["timestep"]) / 100000 for r in late],
                    [float(r["mean_total_reward"]) for r in late],
                    1,
                )[0]
            )
            seed_detail = {"seed": seed, "late_evaluation_slope_per_100k": slope}
            if method == "pspo":
                events = summary["adaptive_diagnostics"]["safety_update_events"]
                if [e["timestep"] for e in events] != [204800, total]:
                    raise ValueError("Unexpected PSPO safety checkpoint schedule")
                before = read_json(directory / "pre_finalization_metrics.json")
                if before["eval_episodes"] != 10:
                    raise ValueError("Require ten paired pre-projection episodes")
                paired = read_csv(directory / "episodes.csv")[:10]
                if [int(r["episode"]) for r in paired] != list(range(10)):
                    raise ValueError("Final paired episodes are not the first ten")
                after = {
                    "reward": float(
                        np.mean([float(r["total_reward"]) for r in paired])
                    ),
                    "goal": float(np.mean([flag(r["success"]) for r in paired])),
                    "safety": float(
                        np.mean([flag(r["safe_trajectory"]) for r in paired])
                    ),
                }
                before_values = metrics_values(before)
                points.append(
                    {
                        "method": method,
                        "seed": seed,
                        "channel": "evaluation",
                        "timestep": total,
                        "episodes": 10,
                        **before_values,
                    }
                )
                points.append(
                    {
                        "method": method,
                        "seed": seed,
                        "channel": "paired_post_projection",
                        "timestep": total,
                        "episodes": 10,
                        **after,
                    }
                )
                seed_detail.update(
                    {
                        "before_projection": before_values,
                        "after_projection_paired": after,
                        "projection_decision": events[-1]["decision"],
                    }
                )
            seed_details.append(seed_detail)
        details[method] = {
            "seeds": seed_details,
            "late_evaluation_slope_per_100k": mean_two_se(
                [r["late_evaluation_slope_per_100k"] for r in seed_details]
            ),
        }
        for channel, label in (
            ("final_100_episodes", "final_evaluation"),
            ("exploration", "last_training_bin"),
        ):
            selected = [
                p for p in points if p["method"] == method and p["channel"] == channel
            ]
            last = max(p["timestep"] for p in selected)
            selected = [p for p in selected if p["timestep"] == last]
            details[method][label] = {
                m: mean_two_se([p[m] for p in selected]) for m in METRICS
            }
    seed_details = details["pspo"]["seeds"]
    details["pspo"]["final_projection_paired"] = {
        stage: {m: mean_two_se([r[field][m] for r in seed_details]) for m in METRICS}
        for stage, field in (
            ("before", "before_projection"),
            ("after", "after_projection_paired"),
        )
    }
    details["pspo"]["final_projection_paired"]["change"] = {
        m: mean_two_se(
            [
                r["after_projection_paired"][m] - r["before_projection"][m]
                for r in seed_details
            ]
        )
        for m in METRICS
    }
    return points, details


def select(aggregate: list[dict], method: str, channel: str) -> list[dict]:
    return sorted(
        [r for r in aggregate if r["method"] == method and r["channel"] == channel],
        key=lambda r: r["timestep"],
    )


def plot_curve(ax, aggregate, method, channel, metric="reward", *, shade=True):
    rows = select(aggregate, method, channel)
    if any(r["seed_count"] != 10 for r in rows):
        raise ValueError("The displayed curves require all ten seeds at every point")
    x = np.array([r["timestep"] for r in rows]) / 1000
    y = np.array([r[metric + "_mean"] for r in rows])
    error = np.array([r[metric + "_two_se"] for r in rows])
    ax.plot(x, y, color=COLORS[method], lw=1.55, ls="-" if method == "pspo" else "--")
    if shade:
        ax.fill_between(
            x, y - error, y + error, color=COLORS[method], alpha=0.15, linewidth=0
        )


def mark_checkpoints(ax):
    for step in (204.8, 401.408):
        ax.axvline(step, color="#969696", lw=0.8, ls=":", zorder=0)
    ax.set_xlim(0, 415)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xlabel("Training steps (thousands)")


def save_figure(fig, output, name):
    fig.savefig(output / (name + ".pdf"), facecolor="white")
    fig.savefig(output / (name + ".png"), dpi=220, facecolor="white")
    plt.close(fig)


def create_figures(
    output: Path, points: list[dict], aggregate: list[dict], details: dict
):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e3e3e3",
            "grid.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(10.4, 3.55), gridspec_kw={"width_ratios": [1, 1.25, 0.7]}
    )
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.25, top=0.82, wspace=0.32)
    for method in ("ppo", "pspo"):
        plot_curve(axes[0], aggregate, method, "exploration")
        plot_curve(axes[1], aggregate, method, "evaluation")
        row = select(aggregate, method, "final_100_episodes")[0]
        axes[1].errorbar(
            row["timestep"] / 1000,
            row["reward_mean"],
            yerr=row["reward_two_se"],
            fmt="D",
            color=COLORS[method],
            ms=4.5,
            capsize=2,
            lw=1,
            zorder=5,
        )
    for ax in axes[:2]:
        mark_checkpoints(ax)
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_ylabel("Original total reward")
    axes[0].set_title("(a) Exploration / training")
    axes[1].set_title("(b) Greedy, unshielded evaluation")
    before = details["pspo"]["final_projection_paired"]["before"]["reward"]
    final = details["pspo"]["final_evaluation"]["reward"]
    axes[1].plot(
        401.408,
        before["mean"],
        "o",
        color=COLORS["pspo"],
        mfc="white",
        ms=4.5,
        zorder=6,
    )
    axes[1].annotate(
        "",
        xy=(401.408, final["mean"]),
        xytext=(401.408, before["mean"]),
        arrowprops={"arrowstyle": "->", "color": COLORS["pspo"], "lw": 1.2},
    )
    axes[1].text(
        385,
        -2.35,
        "Final safety\nprojection",
        fontsize=7,
        ha="right",
        color=COLORS["pspo"],
    )
    axes[1].set_ylim(-5.8, 1.05)
    # A zoom makes late positive-reward trends legible without hiding the drop.
    inset = axes[1].inset_axes([0.15, 0.14, 0.48, 0.32])
    for method in ("ppo", "pspo"):
        plot_curve(inset, aggregate, method, "evaluation", shade=False)
    inset.set(
        xlim=(240, 402),
        ylim=(-0.6, 0.65),
        xticks=[250, 350, 400],
        yticks=[-0.5, 0, 0.5],
    )
    inset.tick_params(labelsize=5.6, pad=1)
    inset.set_title("Late proposals (zoom)", fontsize=6.4, pad=2)
    pairs = details["pspo"]["seeds"]
    for seed in pairs:
        axes[2].plot(
            [0, 1],
            [
                seed["before_projection"]["reward"],
                seed["after_projection_paired"]["reward"],
            ],
            color=COLORS["pspo"],
            lw=0.65,
            alpha=0.3,
            marker="o",
            ms=2.5,
        )
    paired = details["pspo"]["final_projection_paired"]
    axes[2].errorbar(
        [0, 1],
        [paired[s]["reward"]["mean"] for s in ("before", "after")],
        yerr=[paired[s]["reward"]["two_standard_errors"] for s in ("before", "after")],
        color=COLORS["pspo"],
        lw=1.8,
        marker="D",
        ms=5,
        capsize=3,
        zorder=5,
    )
    axes[2].set_title("(c) PSPO final projection")
    axes[2].set(
        xlim=(-0.25, 1.25),
        ylim=(-5.8, 1.05),
        xticks=[0, 1],
        xticklabels=["Before", "After"],
    )
    axes[2].set_ylabel("Original total reward")
    axes[2].set_xlabel("Same 10 episodes per seed")
    axes[2].text(
        0.5,
        0.91,
        "Safety: 55% → 100%",
        transform=axes[2].transAxes,
        ha="center",
        fontsize=7,
    )
    handles = [
        Line2D([], [], color=COLORS["ppo"], ls="--", lw=1.5, label="PPO"),
        Line2D([], [], color=COLORS["pspo"], lw=1.5, label="PSPO (orthotope)"),
        Line2D(
            [],
            [],
            color="black",
            marker="D",
            ls="none",
            ms=4,
            label="Final checkpoint: 100 episodes/seed",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "FrozenLake 128×128 — shaping during training; original rewards shown",
        y=0.995,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.11,
        "Mean ± 2 SE over 10 training seeds. Exploration: completed-episode means in 20k-step windows; PSPO's training shield is on.",
        ha="center",
        fontsize=7,
    )
    fig.text(
        0.5,
        0.05,
        "Periodic evaluation: 10 episodes/seed. PSPO intermediate actors are uncertified; dotted lines mark safety enforcement (204.8k and 401.4k).",
        ha="center",
        fontsize=7,
    )
    save_figure(fig, output, "frozenlake128_ppo_pspo_reward_learning_curves")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.23, top=0.8, wspace=0.3)
    for ax, metric, title in zip(
        axes, ("goal", "safety"), ("Goal success", "Trajectory safety")
    ):
        for method in ("ppo", "pspo"):
            plot_curve(ax, aggregate, method, "evaluation", metric=metric)
            final_point = select(aggregate, method, "final_100_episodes")[0]
            ax.errorbar(
                final_point["timestep"] / 1000,
                final_point[metric + "_mean"],
                yerr=final_point[metric + "_two_se"],
                fmt="D",
                ms=4.5,
                capsize=2,
                color=COLORS[method],
                zorder=5,
            )
        mark_checkpoints(ax)
        ax.set_title(title + " — unshielded evaluation")
        ax.set_ylim(-0.07, 1.15)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.text(
        0.5,
        0.08,
        "Mean ± 2 SE across 10 seeds; diamonds use 100 episodes/seed. Intermediate PSPO proposals are not certified safe.",
        ha="center",
        fontsize=7,
    )
    save_figure(fig, output, "frozenlake128_ppo_pspo_goal_safety_learning_curves")


def write_csv(path: Path, rows: list[dict]):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    points, details = load_data(args.root.resolve())
    aggregate = aggregate_points(points)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "learning_curve_seed_points.csv", points)
    write_csv(args.output_dir / "learning_curve_mean_two_se.csv", aggregate)
    details["plot_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    (args.output_dir / "learning_curve_analysis.json").write_text(
        json.dumps(details, indent=2) + "\n"
    )
    create_figures(args.output_dir, points, aggregate, details)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "ppo_last_training_bin": details["ppo"]["last_training_bin"],
                "pspo_last_training_bin": details["pspo"]["last_training_bin"],
                "pspo_final_projection_paired": details["pspo"][
                    "final_projection_paired"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
