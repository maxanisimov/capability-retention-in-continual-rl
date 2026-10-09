#!/usr/bin/env python3
"""Rank PSPO ablations using globally shifted, seed-paired final rewards.

The minimum is taken across every selected method, environment and seed.
Each score is (R_variant - minimum) / (R_PSPO_same_seed - minimum).
All six environments are retained, including Colour Bomb v1. The reported
two-standard-error interval is 2 * sample_SD / sqrt(N) across the pooled
environment-seed ratios (N=60 for the current six-environment benchmark).

This is a separate analysis: the historical gain-retained assets are preserved.
No initialisation rewards or learning-curve rewards enter this calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_pspo_ablation_gain_retained import (  # noqa: E402
    ENVIRONMENTS,
    SEEDS,
    VARIANTS,
    final_reward,
    seed_dir,
)

DEFAULT_OUTPUT = (
    REPO / "projects/safe_policy_optimisation/results/pspo/ablation_reward_ratios"
)
LABELS = {
    "pspo": "PSPO",
    "ce_only": "CE-only initialisation",
    "no_entropy": "No initialisation entropy",
    "fixed_lid": "Fixed LID",
    "no_gradient": "No directional growth",
}


def paired_ratios(rewards):
    """Return scores and the single global minimum; reject undefined ratios."""
    if "pspo" not in rewards or not rewards["pspo"]:
        raise ValueError("Need a PSPO reference and at least one environment")
    environments = tuple(rewards["pspo"])
    arrays = {}
    sample_count = None
    for variant, per_environment in rewards.items():
        if set(per_environment) != set(environments):
            raise ValueError(f"Unmatched environments for {variant}")
        arrays[variant] = {}
        for environment in environments:
            values = np.asarray(per_environment[environment], dtype=float)
            if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
                raise ValueError(f"Need finite seed rewards: {variant}/{environment}")
            if sample_count is None:
                sample_count = len(values)
            if len(values) != sample_count:
                raise ValueError(f"Unmatched seed counts: {variant}/{environment}")
            arrays[variant][environment] = values
    minimum = min(
        float(values.min())
        for per_environment in arrays.values()
        for values in per_environment.values()
    )
    denominators = {
        environment: arrays["pspo"][environment] - minimum
        for environment in environments
    }
    for environment, denominator in denominators.items():
        bad = np.flatnonzero(denominator <= 0)
        if bad.size:
            raise ValueError(
                f"Undefined PSPO adjusted reward for {environment}, seed positions "
                f"{bad.tolist()}; global minimum={minimum}. No seeds are dropped "
                "and no epsilon is added."
            )
    scores = {
        variant: {
            environment: (values - minimum) / denominators[environment]
            for environment, values in per_environment.items()
        }
        for variant, per_environment in arrays.items()
    }
    for per_environment in scores.values():
        for values in per_environment.values():
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Invalid adjusted reward ratio")
    if not all((values == 1).all() for values in scores["pspo"].values()):
        raise AssertionError("PSPO must have score exactly one in every seed")
    return scores, minimum


def mean_two_se(values):
    """Plug-in mean and two SE of the supplied ratios, without clipping."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite observations")
    return {
        "mean": float(values.mean()),
        "two_se": float(2 * values.std(ddof=1) / math.sqrt(len(values))),
        "n": len(values),
    }


def pooled_summary(scores):
    """Equal weighting of all environment-seed ratios, not ratio of means."""
    return {
        variant: mean_two_se(np.concatenate(list(per_environment.values())))
        for variant, per_environment in scores.items()
    }


def load_rewards():
    rewards, paths = {}, {}
    for variant in VARIANTS:
        rewards[variant.key], paths[variant.key] = {}, {}
        for environment in ENVIRONMENTS:
            values, files = [], []
            for seed in SEEDS:
                path = seed_dir(variant.key, environment, seed)
                config = json.loads((path / "config.json").read_text())
                summary = json.loads((path / "summary.json").read_text())
                if config.get("seed") != seed:
                    raise ValueError(f"Unmatched seed: {path}")
                if config.get("evaluation_policy") != "unshielded":
                    raise ValueError(f"Expected unshielded final evaluation: {path}")
                if summary.get("early_stop_triggered") is not False:
                    raise ValueError(f"Incomplete or early-stopped run: {path}")
                values.append(final_reward(path))  # Enforces 100 eval episodes.
                files.append(str((path / "metrics.json").relative_to(REPO)))
            rewards[variant.key][environment] = np.asarray(values)
            paths[variant.key][environment] = files
    return rewards, paths


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_results(output, rewards, paths, scores, minimum, summary):
    output.mkdir(parents=True, exist_ok=True)
    seed_rows, environment_rows = [], []
    for variant, per_environment in scores.items():
        for environment, values in per_environment.items():
            for index, seed in enumerate(SEEDS):
                reward = float(rewards[variant][environment][index])
                reference = float(rewards["pspo"][environment][index])
                seed_rows.append(
                    {
                        "environment": environment,
                        "variant": variant,
                        "seed": seed,
                        "total_reward": reward,
                        "global_minimum_reward": minimum,
                        "adjusted_reward": reward - minimum,
                        "paired_pspo_reward": reference,
                        "paired_pspo_adjusted_reward": reference - minimum,
                        "ratio": float(values[index]),
                        "source_metrics": paths[variant][environment][index],
                    }
                )
            environment_rows.append(
                {"environment": environment, "variant": variant, **mean_two_se(values)}
            )
    aggregate_rows = [
        {"variant": variant, "label": LABELS[variant], **stats}
        for variant, stats in summary.items()
    ]
    write_csv(output / "per_seed.csv", seed_rows)
    write_csv(output / "per_environment.csv", environment_rows)
    write_csv(output / "summary.csv", aggregate_rows)
    metadata = {
        "formula": "(R_variant_e_seed - minimum) / (R_pspo_e_seed - minimum)",
        "minimum_scope": "All variants, environments and seed-level final means",
        "global_minimum_reward": minimum,
        "shift": -minimum,
        "environments": list(ENVIRONMENTS),
        "seeds": list(SEEDS),
        "evaluation": "100 greedy unshielded episodes per seed",
        "aggregation": "Arithmetic mean of all 60 paired environment-seed ratios",
        "two_se_formula": "2 * sample_SD(all 60 ratios, ddof=1) / sqrt(60)",
        "uncertainty_scope": (
            "Pooled ratios; includes between-environment variation. Conditional on "
            "the observed global minimum; does not bootstrap its estimation."
        ),
        "sanity_check": summary["pspo"],
        "summary": summary,
    }
    (output / "results.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# PSPO ablations: shifted, seed-paired reward ratios",
        "",
        "Each reward R is the final mean total reward over 100 greedy, unshielded "
        "evaluation episodes, not a learning-curve checkpoint.",
        "",
        f"The global minimum across all 300 seed-level rewards is {minimum:.2f}; "
        f"adjusted rewards are R + {-minimum:.2f}.",
        "For each environment and seed, score = adjusted ablation reward / adjusted "
        "PSPO reward for that same environment and seed.",
        "All six environments and ten seeds are retained. PSPO is exactly 1 for "
        "every environment-seed pair.",
        "",
        "Entries are mean +/- 2 standard errors over 60 pooled ratios per method. "
        "Two SE = 2 * sample SD / sqrt(60). Unlike the historical stratified SE, "
        "this includes between-environment variation.",
        "",
        "| Method | Mean | 2 SE | n |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['label']} | {row['mean']:.6f} | {row['two_se']:.6f} | {row['n']} |"
        )
    lines += [
        "",
        "Subtracting the minimum makes rewards non-negative, not strictly positive: "
        "the minimum itself becomes zero. All observed PSPO denominators are positive.",
        "Ratios are not clipped and can exceed one.",
        f"The common {-minimum:+.2f} shift makes differences in the originally [0,1]-reward "
        "environments small relative to the shifted reward magnitude.",
        "",
        "`per_seed.csv` records the numerator, matched denominator and source file "
        "for every ratio. `per_environment.csv` and `summary.csv` contain aggregates.",
        "The old gain-retained figure/table assets are preserved and use a different metric.",
        "",
        "Reproduce from the repository root:",
        "",
        "```bash",
        "PYTHONPATH=core:. .venv/bin/python "
        "projects/safe_policy_optimisation/scripts/compute_pspo_ablation_reward_ratios.py",
        "```",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    rewards, paths = load_rewards()
    scores, minimum = paired_ratios(rewards)
    summary = pooled_summary(scores)
    if summary["pspo"] != {"mean": 1.0, "two_se": 0.0, "n": 60}:
        raise AssertionError("Expected PSPO 1 +/- 0 over all 60 paired scores")
    save_results(args.output_dir, rewards, paths, scores, minimum, summary)
    print(f"Global minimum = {minimum:.2f}; shift = {-minimum:.2f}")
    for variant, stats in summary.items():
        print(
            f"{LABELS[variant]:<27} {stats['mean']:.6f} +/- "
            f"{stats['two_se']:.6f} (2 SE, n={stats['n']})"
        )
    print(f"Saved results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
