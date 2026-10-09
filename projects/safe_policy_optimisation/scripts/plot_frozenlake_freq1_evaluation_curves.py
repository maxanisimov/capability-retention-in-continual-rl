#!/usr/bin/env python3
"""Compare standard FrozenLake evaluation return: frequency-1 PSPO versus PPO.

The command-line plot uses ten seeds per method for layouts 16/32/64/128.
64 uses the matched short-budget repeats; 128 crops PPO's longer saved run.
The legacy three-layout loader remains available to the reward/safety script.
The default return is goal=1, otherwise=0, with no step penalty. The recorded
step-penalised returns are retained for audit and optional companion figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from projects.safe_policy_optimisation.scripts import (  # noqa: E402
    plot_frozenlake_ppo_pspo_reward_efficiency as previous,
)

plt = previous.plt
np = previous.np
PROJECT = previous.PROJECT
RUNS = previous.RUNS
OUTPUT = PROJECT / "figures/frozenlake_pspo_freq1_ppo_evaluation_curves_20261008"
COLORS = previous.COLORS
PSPO_COHORTS = {
    16: "frozenlake16_shaping_pspo_segment_verify_first_freq1_t204800_20261008T155100Z",
    32: "frozenlake32_shaping_pspo_segment_verify_first_freq1_t204800_20261008T160100Z",
    64: "frozenlake64_shaping_pspo_segment_verify_first_freq1_t409600_20261008T160100Z",
}
EXPECTED_SEEDS = {
    (size, method): ([0] if size == 64 and method == "ppo" else list(range(10)))
    for size in PSPO_COHORTS
    for method in ("ppo", "pspo")
}
EXPECTED_STEPS = {
    (size, method): (
        200704 if size == 64 and method == "ppo" else 409600 if size == 64 else 204800
    )
    for size in PSPO_COHORTS
    for method in ("ppo", "pspo")
}
FOUR_LAYOUT_COHORTS = {
    16: {"ppo": previous.COHORTS[16]["ppo"], "pspo": PSPO_COHORTS[16]},
    32: {"ppo": previous.COHORTS[32]["ppo"], "pspo": PSPO_COHORTS[32]},
    64: {
        "ppo": "frozenlake64_shaping_ppo_t200000_20261008T172454Z",
        "pspo": "frozenlake64_shaping_pspo_segment_verify_first_freq1_t200000_20261008T172800Z",
    },
    128: {
        "ppo": previous.COHORTS[128]["ppo"],
        "pspo": "frozenlake128_shaping_pspo_segment_verify_first_freq1_t200000_20261008T173822Z",
    },
}
FOUR_LAYOUT_STEPS = {
    (size, method): (
        204800
        if size in (16, 32)
        else 401408
        if size == 128 and method == "ppo"
        else 200704
    )
    for size in FOUR_LAYOUT_COHORTS
    for method in ("ppo", "pspo")
}
FOUR_LAYOUT_SEEDS = {
    (size, method): list(range(10))
    for size in FOUR_LAYOUT_COHORTS
    for method in ("ppo", "pspo")
}


def seed_statistics(values):
    if len(values) == 1:
        if not np.isfinite(values[0]):
            raise ValueError("Nonfinite single-seed reward")
        return {"mean": float(values[0]), "two_se": None, "seed_count": 1}
    return previous.mean_two_se(values)


def seed_paths(runs, size, method):
    if method == "pspo":
        return [
            (seed, runs / PSPO_COHORTS[size] / "pspo" / f"seed{seed}")
            for seed in range(10)
        ]
    if size == 64:
        return [(0, runs / "ppo_shaping_frozenlake64_20261008T070700Z/shaped_seed0")]
    return [
        (seed, runs / previous.COHORTS[size]["ppo"] / "ppo" / f"seed{seed}")
        for seed in range(10)
    ]


def evaluation_points(periodic, initial, final, total, *, periodic_stop=None):
    # PSPO suppresses a periodic evaluation once the requested budget has
    # been reached and evaluates the certified, rollout-rounded endpoint.
    stop = total if periodic_stop is None else periodic_stop
    if not 0 < stop <= total:
        raise ValueError("Invalid periodic-evaluation stopping budget")
    expected = [*range(20000, stop, 20000), total]
    if [int(p["timestep"]) for p in periodic] != expected:
        raise ValueError("Missing, duplicated or unexpected evaluation checkpoints")
    if initial["episodes"] != 10 or final["episodes"] != 100:
        raise ValueError("Require ten initial and 100 final episodes")
    points = [{"channel": "periodic", "timestep": 0, **initial}]
    for row in periodic:
        if int(row["episodes"]) != 10:
            raise ValueError("Require ten episodes per periodic evaluation")
        if int(row["timestep"]) == total:
            continue  # Keep the separate 100-episode endpoint, not a duplicate.
        points.append(
            {
                "channel": "periodic",
                "timestep": int(row["timestep"]),
                "episodes": int(row["episodes"]),
                "reward": float(row["mean_total_reward"]),
                "goal": float(row["success_rate"]),
                "safety": float(row["safety_rate"]),
            }
        )
    points.append({"channel": "final", "timestep": total, **final})
    return points


def with_reward_definition(points, definition="standard"):
    """Rescore saved evaluations, not training, using the exact goal indicator.

    No shifting, clipping or min/max normalisation: standard episode return is
    binary. Its seed-level mean is exactly the seed's goal-success proportion.
    """
    if definition not in {"standard", "step-penalised"}:
        raise ValueError("Unknown evaluation reward definition")
    result = []
    for point in points:
        goal = float(point["goal"])
        if not np.isfinite(goal) or not 0 <= goal <= 1:
            raise ValueError("Invalid saved goal-success proportion")
        base_reward = point.get("step_penalised_reward", point["reward"])
        result.append(
            {
                **point,
                "step_penalised_reward": base_reward,
                "reward": goal if definition == "standard" else base_reward,
            }
        )
    return result


def trim_ppo128(points, max_steps=200000):
    """Retain a recorded learning-curve prefix, not a new final checkpoint."""
    if max_steps <= 0:
        raise ValueError("The PPO128 cutoff must be positive")
    return [
        p
        for p in points
        if not (p["size"] == 128 and p["method"] == "ppo" and p["timestep"] > max_steps)
    ]


def checked_base_metrics(path, step_penalty):
    point = previous.checked_metrics(path)
    saved = previous.read_json(path)
    recovered = point["reward"] + step_penalty * saved["mean_episode_length"]
    if not np.isclose(recovered, point["goal"], rtol=0, atol=1e-8):
        raise ValueError(
            f"Saved reward plus step cost does not match goal success: {path}"
        )
    if path.name == "metrics.json":
        episodes = previous.read_csv(path.parent / "episodes.csv")
        safety = np.mean([previous.flag(e["safe_trajectory"]) for e in episodes])
        if len(episodes) != 100 or not np.isclose(
            safety, point["safety"], rtol=0, atol=1e-10
        ):
            raise ValueError(
                f"Final safety does not match saved episode rollouts: {path}"
            )
    return point


def load_data(runs=RUNS, *, matched64_roots=None, cohorts=None, expected_steps=None):
    if (cohorts is None) != (expected_steps is None):
        raise ValueError("Explicit cohorts require explicit expected training budgets")
    if cohorts is not None and matched64_roots is not None:
        raise ValueError("Choose explicit cohorts or the legacy matched64 override")
    if matched64_roots is not None and set(matched64_roots) != {"ppo", "pspo"}:
        raise ValueError("Supply both matched 64x64 method roots")
    points, sources, hashes = [], [], {}
    for size in PSPO_COHORTS if cohorts is None else cohorts:
        reference = None
        for method in ("ppo", "pspo"):
            matched64 = size == 64 and matched64_roots is not None
            if cohorts is not None:
                paths = [
                    (seed, runs / cohorts[size][method] / method / f"seed{seed}")
                    for seed in range(10)
                ]
            elif matched64:
                paths = [
                    (seed, Path(matched64_roots[method]) / method / f"seed{seed}")
                    for seed in range(10)
                ]
            else:
                paths = seed_paths(runs, size, method)
            expected_total = (
                expected_steps[size, method]
                if cohorts is not None
                else 200704
                if matched64
                else EXPECTED_STEPS[size, method]
            )
            for seed, directory in paths:
                config = previous.read_json(directory / "config.json")
                summary = previous.read_json(directory / "summary.json")
                total = summary["final_timesteps"]
                if (
                    config["seed"] != seed
                    or config["env_kwargs"]["size"] != size
                    or total != expected_total
                    or (matched64 and config["requested_timesteps"] != 200000)
                ):
                    raise ValueError(f"Unexpected cohort/seed/budget: {directory}")
                settings = {
                    "env_kwargs": config["env_kwargs"],
                    "max_episode_steps": config["max_episode_steps"],
                    "training_hyperparameters": config["training_hyperparameters"],
                    "architecture": {
                        k: config["architecture"][k] for k in ("hidden_dim", "n_hidden")
                    },
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
                    reference = settings
                if settings != reference:
                    raise ValueError(
                        f"Incompatible environment/training settings: {directory}"
                    )
                if method == "pspo":
                    diagnostics = summary["adaptive_diagnostics"]
                    if (
                        config["pspo_variant"] != "line_segment_verify_first"
                        or config["safety_enforcement"]["frequency_rollouts"] != 1
                    ):
                        raise ValueError(
                            "Require frequency-1 line-segment verify-first PSPO"
                        )
                    if (
                        diagnostics["frequency"] != 1
                        or diagnostics["pending_adaptive_update"]
                        or summary["final_exact_all_state_alignment"] != 1.0
                    ):
                        raise ValueError(
                            "Unexpected frequency or uncertified final checkpoint"
                        )
                    if [
                        e["timestep"] for e in diagnostics["safety_update_events"]
                    ] != list(range(2048, total + 1, 2048)):
                        raise ValueError(
                            "Require enforcement after every complete train phase"
                        )
                periodic = previous.read_csv(
                    directory / "learning_curves/evaluation_unshielded_summary.csv"
                )
                episodes = previous.read_csv(
                    directory / "learning_curves/evaluation_unshielded_episodes.csv"
                )
                episode_groups = {}
                for row in episodes:
                    expected_reward = previous.flag(row["success"]) - config[
                        "env_kwargs"
                    ]["step_penalty"] * int(row["length"])
                    if not np.isclose(
                        float(row["total_reward"]), expected_reward, rtol=0, atol=1e-8
                    ):
                        raise ValueError(
                            "Saved evaluation reward does not match goal minus step cost"
                        )
                    episode_groups.setdefault(int(row["timestep"]), []).append(row)
                for row in periodic:
                    group = episode_groups[int(row["timestep"])]
                    if len(group) != 10 or not np.isclose(
                        np.mean([float(e["total_reward"]) for e in group]),
                        float(row["mean_total_reward"]),
                        rtol=0,
                        atol=1e-10,
                    ):
                        raise ValueError(
                            "Periodic summary does not match saved evaluation episodes"
                        )
                    recovered = np.mean(
                        [
                            float(e["total_reward"])
                            + config["env_kwargs"]["step_penalty"] * int(e["length"])
                            for e in group
                        ]
                    )
                    if not np.isclose(
                        recovered, float(row["success_rate"]), rtol=0, atol=1e-8
                    ):
                        raise ValueError(
                            "Periodic standard reward does not match saved goal success"
                        )
                    empirical_safety = np.mean(
                        [previous.flag(e["safe_trajectory"]) for e in group]
                    )
                    if not np.isclose(
                        empirical_safety, float(row["safety_rate"]), rtol=0, atol=1e-10
                    ):
                        raise ValueError(
                            "Periodic safety does not match saved trajectory flags"
                        )
                initial = checked_base_metrics(
                    directory / "initial_metrics.json",
                    config["env_kwargs"]["step_penalty"],
                )
                final = checked_base_metrics(
                    directory / "metrics.json", config["env_kwargs"]["step_penalty"]
                )
                run_points = evaluation_points(
                    periodic,
                    initial,
                    final,
                    total,
                    periodic_stop=config["requested_timesteps"]
                    if method == "pspo"
                    else None,
                )
                points.extend(
                    {"size": size, "method": method, "seed": seed, **p}
                    for p in run_points
                )
                sources.append(
                    {
                        "size": size,
                        "method": method,
                        "seed": seed,
                        "directory": str(directory),
                        "actual_training_steps": total,
                        "settings": settings,
                        "frequency": 1 if method == "pspo" else None,
                    }
                )
                for name in (
                    "config.json",
                    "summary.json",
                    "initial_metrics.json",
                    "metrics.json",
                    "episodes.csv",
                    "learning_curves/evaluation_unshielded_summary.csv",
                    "learning_curves/evaluation_unshielded_episodes.csv",
                ):
                    path = directory / name
                    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return points, sources, hashes


def aggregate(points, *, matched64=False, expected_seeds=None):
    groups = {}
    for row in points:
        key = tuple(row[k] for k in ("size", "method", "channel", "timestep"))
        groups.setdefault(key, []).append(row)
    rows = []
    for (size, method, channel, step), group in sorted(groups.items()):
        expected = (
            expected_seeds[size, method]
            if expected_seeds is not None
            else list(range(10))
            if matched64 and size == 64
            else EXPECTED_SEEDS[size, method]
        )
        if sorted(r["seed"] for r in group) != expected:
            raise ValueError(
                "Every point must contain the same expected distinct seeds"
            )
        row = {
            "size": size,
            "method": method,
            "channel": channel,
            "timestep": step,
            "seed_count": len(group),
        }
        for metric in ("reward", "goal", "safety"):
            stats = seed_statistics([p[metric] for p in group])
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_two_se"] = stats["two_se"]
        if all("step_penalised_reward" in p for p in group):
            stats = seed_statistics([p["step_penalised_reward"] for p in group])
            row["step_penalised_reward_mean"] = stats["mean"]
            row["step_penalised_reward_two_se"] = stats["two_se"]
        rows.append(row)
    return rows


def draw_panel(ax, rows, size, reward_definition="standard", metric="reward"):
    if metric not in {"reward", "safety"}:
        raise ValueError("Choose evaluation reward or trajectory safety")
    for method in ("ppo", "pspo"):
        selected = [
            r
            for r in rows
            if r["size"] == size
            and r["method"] == method
            and r["channel"] == "periodic"
        ]
        x = np.array([r["timestep"] / 1000 for r in selected])
        y = np.array([r[f"{metric}_mean"] for r in selected])
        if selected[0]["seed_count"] > 1:
            error = np.array([r[f"{metric}_two_se"] for r in selected])
            ax.fill_between(
                x, y - error, y + error, color=COLORS[method], alpha=0.16, linewidth=0
            )
        ax.plot(
            x,
            y,
            color=COLORS[method],
            linestyle="--" if method == "ppo" else "-",
            linewidth=1.5,
            marker="o",
            markersize=2.4,
        )
        final = next(
            (
                r
                for r in rows
                if r["size"] == size
                and r["method"] == method
                and r["channel"] == "final"
            ),
            None,
        )
        if final is not None:
            ax.errorbar(
                final["timestep"] / 1000,
                final[f"{metric}_mean"],
                yerr=final[f"{metric}_two_se"],
                color=COLORS[method],
                marker="D",
                markersize=6 if method == "ppo" else 4,
                markerfacecolor="none" if method == "ppo" else COLORS[method],
                markeredgecolor=COLORS[method],
                markeredgewidth=1,
                capsize=2.5,
                linestyle="none",
                zorder=6,
            )
    pilot = any(
        r["size"] == size and r["method"] == "ppo" and r["seed_count"] == 1
        for r in rows
    )
    ax.set_title(
        f"FrozenLake {size}×{size}"
        + ("\nPPO pilot: 1 seed, shorter run" if pilot else "\nTen seeds per method"),
        fontsize=8,
    )
    ax.set_xlabel("Training steps (thousands)")
    max_step = max(r["timestep"] for r in rows if r["size"] == size)
    ax.set_xlim(0, max_step / 1000 * 1.045)
    if reward_definition == "standard":
        ax.set_ylim(-0.08, 1.08)  # Retain the complete, untruncated two-SE bands.
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    else:
        ax.set_ylim({16: -0.72, 32: -1.4, 64: -2.75, 128: -5.5}[size], 1.08)
    ax.axhline(0, color=".6", linewidth=0.6, zorder=0)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if pilot and max_step >= 200704:
        ax.axvline(
            200704 / 1000, color=COLORS["ppo"], alpha=0.4, linewidth=0.7, linestyle=":"
        )


def legend(fig, reward_definition="standard", *, ncol=1):
    handles = [
        previous.Line2D([0], [0], color=COLORS["ppo"], linestyle="--", label="PPO"),
        previous.Line2D(
            [0],
            [0],
            color=COLORS["pspo"],
            label="PSPO (frequency 1; segment verify-first)",
        ),
        previous.Line2D(
            [0],
            [0],
            color=".35",
            marker="D",
            linestyle="none",
            label="Final evaluation (100 episodes)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=ncol,
        frameon=False,
        fontsize=7,
        title="Standard reward; step 0 = initial policy; mean ± 2 SE"
        if reward_definition == "standard"
        else "Return with step penalty; mean ± 2 SE",
        title_fontsize=7,
    )


def save_figure(fig, output, name):
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=220)
    plt.close(fig)


def build_reward_figure(rows, reward_definition="standard"):
    """Four compact panels; preserve each method's actual recorded endpoint."""
    fig, axes = plt.subplots(
        1, 4, figsize=(7.1, 2.65), sharey=True, layout="constrained"
    )
    for ax, size in zip(axes, FOUR_LAYOUT_COHORTS):
        draw_panel(ax, rows, size, reward_definition)
        if size == 128:
            ppo_end = max(
                r["timestep"] for r in rows if r["size"] == 128 and r["method"] == "ppo"
            )
            ax.set_title(
                f"FrozenLake 128×128\nPPO prefix ≤{ppo_end / 1000:g}k", fontsize=8
            )
    axes[0].set_ylabel(
        "Standard total reward"
        if reward_definition == "standard"
        else "Return with step penalty"
    )
    legend(fig, reward_definition, ncol=3)
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--ppo128-max-steps", type=int, default=200000)
    parser.add_argument(
        "--reward-definition",
        choices=("standard", "step-penalised"),
        default="standard",
    )
    args = parser.parse_args()
    points, sources, hashes = load_data(
        args.runs_root, cohorts=FOUR_LAYOUT_COHORTS, expected_steps=FOUR_LAYOUT_STEPS
    )
    points = with_reward_definition(points, args.reward_definition)
    omitted_points = [
        p
        for p in points
        if p["size"] == 128
        and p["method"] == "ppo"
        and p["timestep"] > args.ppo128_max_steps
    ]
    points = trim_ppo128(points, args.ppo128_max_steps)
    rows = aggregate(points, expected_seeds=FOUR_LAYOUT_SEEDS)
    suffix = "" if args.reward_definition == "standard" else "_step_penalised"
    ylabel = (
        "Evaluation return (standard reward)"
        if args.reward_definition == "standard"
        else "Evaluation return (with step penalty)"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, _ = build_reward_figure(rows, args.reward_definition)
    save_figure(
        fig, args.output_dir, "frozenlake_pspo_freq1_vs_ppo_evaluation_reward" + suffix
    )
    for size in FOUR_LAYOUT_COHORTS:
        fig, ax = plt.subplots(figsize=(3.35, 2.8), layout="constrained")
        draw_panel(ax, rows, size, args.reward_definition)
        if size == 128:
            ppo_end = max(
                r["timestep"] for r in rows if r["size"] == 128 and r["method"] == "ppo"
            )
            ax.set_title(
                f"FrozenLake 128×128\nPPO prefix ≤{ppo_end / 1000:g}k", fontsize=8
            )
        ax.set_ylabel(ylabel)
        legend(fig, args.reward_definition)
        save_figure(
            fig,
            args.output_dir,
            f"frozenlake{size}_pspo_freq1_vs_ppo_evaluation_reward" + suffix,
        )
    previous.write_csv(
        args.output_dir / ("seed_evaluation_points" + suffix + ".csv"), points
    )
    previous.write_csv(
        args.output_dir / ("aggregate_evaluation_points" + suffix + ".csv"), rows
    )
    (args.output_dir / ("analysis" + suffix + ".json")).write_text(
        json.dumps(
            {
                "reward_definition": args.reward_definition,
                "reward": "Standard FrozenLake: 1 on reaching goal, 0 otherwise; no step penalty, shaping or discounting."
                if args.reward_definition == "standard"
                else "Base experiment return: goal indicator minus 0.001 per step, without potential shaping or discounting.",
                "standard_reward_reconstruction": "Exactly the saved goal-success proportion, validated against saved return plus 0.001 times episode length. SE recomputed across seed proportions, not reused from step-penalised returns.",
                "uncertainty": "Mean +/- two sample standard errors across ten training-seed means for every method and layout.",
                "layouts": list(FOUR_LAYOUT_COHORTS),
                "actual_training_budgets": {
                    str(size): {
                        method: FOUR_LAYOUT_STEPS[size, method]
                        for method in ("ppo", "pspo")
                    }
                    for size in FOUR_LAYOUT_COHORTS
                },
                "ppo128_display_cutoff_steps": args.ppo128_max_steps,
                "omitted_ppo128_seed_points": omitted_points,
                "unequal_budget_note": "128x128 PPO was trained for 401408 steps, but only its recorded prefix at or below the display cutoff is plotted. Its later 100-episode final evaluation is omitted, never moved to 200000. PSPO retains its actual 200704-step final evaluation. This is not a new matched-budget PPO checkpoint or final evaluation.",
                "step_zero_definition": "Saved initial_metrics.json: ten greedy unshielded evaluation episodes per seed, measured after actor construction/shaping attachment but before model.learn and before any RL update. Not an artificial zero or the first 20000-step evaluation.",
                "sources": sources,
                "source_sha256": hashes,
                "final_evaluations": [r for r in rows if r["channel"] == "final"],
            },
            indent=2,
        )
        + "\n"
    )
    (
        args.output_dir / ("README" + suffix + ".md")
    ).write_text("""# Frequency-1 PSPO versus the existing PPO evaluation learning curves

Default figures use STANDARD FrozenLake reward: 1 on reaching the goal, 0
otherwise, with NO step penalty, potential shaping or discounting. Therefore
mean evaluation return is exactly goal-success rate. This is a change in
reported evaluation reward, NOT in training: all original trained checkpoints,
trajectories, training shaping, step costs and seed/budget choices are unchanged.
Standard return is recovered exactly from the saved success indicators and
checked against saved penalised return + 0.001 × episode length; no reruns,
normalisation, clipping or fabricated rewards are involved. Uncertainty is
recomputed from the per-seed standard-return means, not the old reward SEs.

Companion files with `_step_penalised` in their names retain the earlier reward
definition (goal indicator minus 0.001 × episode length). `analysis*.json`
states which reward definition each set uses; CSVs retain both definitions.

The combined figure shows 16×16, 32×32, 64×64 and 128×128 in one compact
7.1 × 2.65 inch row (vector PDF and PNG). Individual layout figures are
3.35 inches wide. Every method/layout uses ten completed seeds, 0–9.

PSPO is line-segment verify-first, enforcing each complete 2,048-step PPO
training phase. Its actor initialisation is safety-only, and its exploration
is runtime-shielded. PPO uses random initialisation and unshielded exploration.
Both methods use potential-based reward shaping ONLY during training. These
plots show greedy, unshielded EVALUATION rescored with standard FrozenLake reward.
The x axis is training environment steps, not evaluation duration.

16×16 and 32×32 use the EXACT SAME PPO cohorts and evaluation trajectories as
the earlier frequency-100 plots: ten seeds 0–9, 204,800 steps each. No new
training or evaluation was launched. These frequency-1 figures are updated in
place; earlier frequency-100 figures are not changed by this script.

64×64 now uses the matched ten-seed PPO and PSPO short-budget repeats, both
requested 200,000 steps and trained for 200,704 complete-rollout steps. The
earlier single-seed PPO pilot and 409,600-step PSPO cohort are NOT used here.

128×128 uses ten PSPO seeds at 200,704 steps (requested 200,000), and ten PPO
seeds at 401,408 steps (requested 400,000). Both methods have matching layout,
architecture, shaping and optimiser settings, but unequal total budgets.
Only PPO's recorded prefix through --ppo128-max-steps (default 200,000) is
plotted. Its later observations and 401,408-step final diamond are omitted;
no checkpoint is retrained, relabelled or evaluated at a different timestep.
The underlying full PPO run remains intact. The PPO endpoint at 200,000 is
a periodic TEN-episode evaluation per seed, not a 100-episode final evaluation.
PSPO retains its separate final 100-episode evaluation at 200,704. Do not
claim a new matched-budget PPO final checkpoint from this curve crop.
Compare periodic curves at shared timesteps through 180,000 for learning efficiency;
PPO also has a periodic point at 200,000. PSPO skips the periodic evaluation
once its requested budget is reached and evaluates the certified final model
at 200,704; no 200,000-step PSPO observation is invented or interpolated.

Lines contain initial ten-episode evaluation and periodic ten-episode
evaluations every 20,000 training steps. Diamonds show the separate final
100-episode evaluation at the actual final budget. The duplicate ten-episode
final evaluation is not plotted. All plotted PSPO policies have frequency-1
enforcement; final certificates and every scheduled safety event are checked.
Bands and diamond error bars are mean ± TWO sample standard errors over seed
means: 2 × sample SD / sqrt(10). All plotted points have ten seed-level means.
There is no reward smoothing, clipping or interpolation. Evaluation summary
rewards are checked against saved episode returns and both reward definitions.

Step zero is the MEASURED initial-policy evaluation from initial_metrics.json,
after constructing the policy but before model.learn() and before any RL update.
It is not an artificial zero and not the first periodic evaluation at 20,000.
PSPO's initial actor uses only the safety mask (no goal/reward information),
and its parameters are checked unchanged by attaching the training shaping.
All four layouts and both methods have zero observed standard initial reward
in these ten-episode-per-seed evaluations. The next point at 20,000 reflects
training, which DOES use goal-distance potential shaping for both methods.
Differences in curves therefore measure environment-step learning efficiency
under the respective full setups, not isolated initialization effects or a
claim of faster training wall-clock time.

The CSVs retain seed-level and aggregated plotted observations; analysis.json
records source directories, settings, final results and input file hashes.

Reproduce from the repository root:
```
.venv/bin/python projects/safe_policy_optimisation/scripts/plot_frozenlake_freq1_evaluation_curves.py
```

Regenerate the step-penalised companions with `--reward-definition step-penalised`.
""")
    print(f"Saved figures and audit data to {args.output_dir}")


if __name__ == "__main__":
    main()
