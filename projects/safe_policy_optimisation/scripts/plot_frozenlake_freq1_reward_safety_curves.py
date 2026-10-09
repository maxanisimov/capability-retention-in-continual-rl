#!/usr/bin/env python3
"""One six-panel figure: standard evaluation reward and unshielded safety.

Columns are FrozenLake 16/32/64, rows are standard reward and trajectory safety.
Reuse the existing frequency-1 PSPO and PPO cohorts without running experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from projects.safe_policy_optimisation.scripts import (  # noqa: E402
    plot_frozenlake_freq1_evaluation_curves as curves,
)

NAME = "frozenlake_pspo_freq1_vs_ppo_evaluation_reward_safety"
CAPTION = """# Standard evaluation reward and unshielded safety: PSPO frequency 1 vs PPO

One figure, with layout sizes 16×16, 32×32 and 64×64 as columns. Top row:
standard FrozenLake episode total reward (1 at goal, 0 otherwise), averaged
over evaluation episodes. Bottom row: trajectory safety rate (proportion
of episodes with no hole visit); a safe timeout counts as safe but has zero
standard return. Both methods are evaluated greedily WITHOUT a runtime shield.

Lines show initial ten-episode evaluations plus periodic ten-episode
evaluations every 20,000 training environment steps. Diamonds show separate
100-episode final evaluations, not the duplicate ten-episode final checkpoint.
Bands and endpoint error bars are mean ± TWO sample standard errors over
training-seed means. Standard return is exactly goal success, not the base
environment return with its 0.001-per-step cost. The same trained policies
and saved trajectories are rescored; neither training nor evaluation is rerun.
Training retained its step cost and potential-based shaping. Safety summaries
are validated against the saved per-episode trajectory flags.

16×16 and 32×32: ten seeds per method and 204,800 training steps each.
64×64 is cropped at the requested comparison budget (default 200,000 steps).
Points beyond the cutoff, including the long-run PSPO final evaluation, are
discarded, never relabelled as short-budget results. By default this still uses
the existing ten-seed PSPO cohort and SINGLE-SEED PPO pilot; no PPO uncertainty
band or statistical superiority claim is possible. Supply both --ppo64-root
and --pspo64-root after the matched ten-seed 200,000-step repeats finish to
replace this pilot comparison. Those repeats finish at 200,704 steps due to
complete 2,048-step rollouts; use --layout64-max-steps 200704 to retain their
separate 100-episode final evaluations. See the JSON for actual sources/counts.

PSPO is line-segment verify-first with frequency 1: exact verification or
certified projection/reversion after every full 2,048-step PPO training phase.
All scheduled enforcement events and final all-winning-state certificates
are validated. Its initialisation is safety-only and training exploration is
shielded; PPO uses random initialisation and unshielded exploration.

Means lie within [0,1]. Bands are untruncated mean ± 2 SE and can extend
slightly outside these bounds; they are not additional observed rate values.
No smoothing or interpolation is used. The PDF is vector, 7 × 4.2 inches.
The JSON contains plotted seed means, aggregates, settings and source hashes.

Reproduce from the repository root:
```
.venv/bin/python projects/safe_policy_optimisation/scripts/plot_frozenlake_freq1_reward_safety_curves.py
```
"""


def build_figure(rows):
    plt = curves.plt
    fig, axes = plt.subplots(
        2, 3, figsize=(7.0, 4.2), sharex="col", sharey=True, layout="constrained"
    )
    for column, size in enumerate(curves.PSPO_COHORTS):
        for row, metric in enumerate(("reward", "safety")):
            ax = axes[row, column]
            curves.draw_panel(ax, rows, size, metric=metric)
            # Show both coincident curves: blue dashes over the orange solid line.
            for line in ax.get_lines():
                if (
                    line.get_color() == curves.COLORS["ppo"]
                    and line.get_linestyle() == "--"
                ):
                    line.set_zorder(3)
            ax.set_ylim(-0.10, 1.08)  # Include the complete, untruncated two-SE bands.
            if row == 0:
                ax.set_xlabel("")
            else:
                ax.set_title("")
    axes[0, 0].set_ylabel("Standard total reward")
    axes[1, 0].set_ylabel("Safety rate")
    handles = [
        curves.previous.Line2D(
            [0], [0], color=curves.COLORS["ppo"], linestyle="--", label="PPO"
        ),
        curves.previous.Line2D(
            [0],
            [0],
            color=curves.COLORS["pspo"],
            label="PSPO (enforcement frequency 1)",
        ),
        curves.previous.Line2D(
            [0],
            [0],
            color=".35",
            marker="D",
            linestyle="none",
            label="Final 100-episode evaluation",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=3,
        frameon=False,
        fontsize=7,
        title="Unshielded evaluation; mean ± 2 SE",
        title_fontsize=8,
    )
    return fig, axes


def trim_layout64(points, max_steps):
    if max_steps <= 0:
        raise ValueError("The 64x64 cutoff must be positive")
    return [p for p in points if p["size"] != 64 or p["timestep"] <= max_steps]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=curves.RUNS)
    parser.add_argument("--output-dir", type=Path, default=curves.OUTPUT)
    parser.add_argument("--layout64-max-steps", type=int, default=200000)
    parser.add_argument("--ppo64-root", type=Path)
    parser.add_argument("--pspo64-root", type=Path)
    args = parser.parse_args()
    if bool(args.ppo64_root) != bool(args.pspo64_root):
        parser.error("Supply both --ppo64-root and --pspo64-root")
    matched = (
        {"ppo": args.ppo64_root, "pspo": args.pspo64_root} if args.ppo64_root else None
    )
    points, sources, hashes = curves.load_data(args.runs_root, matched64_roots=matched)
    points = trim_layout64(points, args.layout64_max_steps)
    points = curves.with_reward_definition(points, "standard")
    rows = curves.aggregate(points, matched64=matched is not None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves.plt.rcParams.update(
        {"font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42}
    )
    fig, _ = build_figure(rows)
    curves.save_figure(fig, args.output_dir, NAME)
    (args.output_dir / f"{NAME}_analysis.json").write_text(
        json.dumps(
            {
                "reward_definition": "Standard FrozenLake: 1 at goal, 0 otherwise; no step penalty or shaping.",
                "safety_definition": "Fraction of greedy unshielded trajectories with no hole visit, including safe timeouts.",
                "uncertainty": "Mean +/- two sample standard errors across training-seed means; SE unavailable for a single seed.",
                "layout64_max_steps": args.layout64_max_steps,
                "matched64_ten_seed_repeats": matched is not None,
                "seed_points": points,
                "aggregate_points": rows,
                "sources": sources,
                "source_sha256": hashes,
            },
            indent=2,
        )
        + "\n"
    )
    (args.output_dir / "README_reward_safety.md").write_text(CAPTION)
    print(f"Saved {args.output_dir / (NAME + '.pdf')} and PNG preview")


if __name__ == "__main__":
    main()
