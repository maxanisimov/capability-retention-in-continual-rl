#!/usr/bin/env python3
"""Final half-A4-width reward/safety figure from audited FrozenLake cohorts."""

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

OUTPUT = curves.PROJECT / "figures/frozenlake_scalability_20261008"
NAME = "frozenlake_scalability"
COLORS = {"ppo": "#000000", "pspo": "#009E73"}
LABELS = {"ppo": "PPO", "pspo": "PSPO-VF"}
WIDTH_INCHES = 4.0  # 101.6 mm: below half the 210 mm width of A4.
# Every curve uses the ~200k-step budget: 16/32 ran 100 full 2048-step rollouts
# (204,800 steps); 64/128 requested 200,000 and ended at 200,704.
COHORTS = {
    **curves.FOUR_LAYOUT_COHORTS,
    128: {
        "ppo": "frozenlake128_shaping_ppo_t200000_20261008T210643Z",
        "pspo": curves.FOUR_LAYOUT_COHORTS[128]["pspo"],
    },
}
STEPS = {
    (size, method): 204800 if size in (16, 32) else 200704
    for size in COHORTS
    for method in ("ppo", "pspo")
}


def build_figure(rows):
    plt, np = curves.plt, curves.np
    style = {
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            2, 4, figsize=(WIDTH_INCHES, 2.7), sharex="col", sharey=True
        )
        fig.subplots_adjust(
            left=0.115, right=0.99, bottom=0.165, top=0.765, wspace=0.15, hspace=0.20
        )
        for column, size in enumerate(COHORTS):
            for row_number, metric in enumerate(("reward", "safety")):
                ax = axes[row_number, column]
                for method in ("pspo", "ppo"):
                    group = sorted(
                        (
                            r
                            for r in rows
                            if r["size"] == size
                            and r["method"] == method
                            and r["channel"] == "periodic"
                        ),
                        key=lambda r: r["timestep"],
                    )
                    if not group or any(r["seed_count"] != 10 for r in group):
                        raise ValueError(
                            "Every curve requires ten seed means at every point"
                        )
                    x = np.array([r["timestep"] / 1000 for r in group])
                    y = np.array([r[f"{metric}_mean"] for r in group])
                    error = np.array([r[f"{metric}_two_se"] for r in group])
                    ax.fill_between(
                        x,
                        y - error,
                        y + error,
                        color=COLORS[method],
                        alpha=0.12,
                        linewidth=0,
                        zorder=1,
                    )
                    ax.plot(
                        x,
                        y,
                        color=COLORS[method],
                        linestyle="--" if method == "ppo" else "-",
                        linewidth=1.05,
                        marker="o",
                        markersize=1.6,
                        label=LABELS[method],
                        zorder=4 if method == "ppo" else 3,
                    )
                    finals = [
                        r
                        for r in rows
                        if r["size"] == size
                        and r["method"] == method
                        and r["channel"] == "final"
                    ]
                    if len(finals) > 1:
                        raise ValueError("Duplicate final evaluation")
                    if finals:
                        final = finals[0]
                        if final["timestep"] <= group[-1]["timestep"]:
                            raise ValueError(
                                "Final evaluation must follow the last periodic point"
                            )
                        ax.plot(
                            [x[-1], final["timestep"] / 1000],
                            [y[-1], final[f"{metric}_mean"]],
                            color=COLORS[method],
                            linestyle="--" if method == "ppo" else "-",
                            linewidth=1.05,
                            label=f"_final_connector_{method}",
                            zorder=4 if method == "ppo" else 3,
                        )
                        ax.errorbar(
                            final["timestep"] / 1000,
                            final[f"{metric}_mean"],
                            yerr=final[f"{metric}_two_se"],
                            color=COLORS[method],
                            marker="D",
                            markersize=3.8 if method == "ppo" else 2.8,
                            markerfacecolor="none"
                            if method == "ppo"
                            else COLORS[method],
                            markeredgewidth=0.75,
                            capsize=1.6,
                            elinewidth=0.65,
                            linestyle="none",
                            zorder=6,
                        )
                # A title wrap preserves readable lettering at half-A4 width.
                ax.set_title(
                    f"FrozenLake\n{size}×{size}" if row_number == 0 else "", pad=4
                )
                ax.set_xlim(0, 215)
                ax.set_ylim(-0.10, 1.08)  # Unclipped mean +/- 2 SE, not clipped rates.
                ax.set_xticks([0, 100, 200])
                ax.set_yticks([0, 0.5, 1])
                ax.tick_params(length=2.3, width=0.55, pad=2)
                ax.grid(color="0.85", alpha=0.6, linewidth=0.4)
                ax.spines[["top", "right"]].set_visible(False)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.55)
        axes[0, 0].set_ylabel("Total reward", labelpad=3)
        axes[1, 0].set_ylabel("Safety rate", labelpad=3)
        fig.supxlabel("Training steps (thousands)", fontsize=8.5, y=0.045)
        handles = [
            curves.previous.Line2D(
                [0],
                [0],
                color=COLORS["ppo"],
                linestyle="--",
                linewidth=1.05,
                label="PPO",
            ),
            curves.previous.Line2D(
                [0], [0], color=COLORS["pspo"], linewidth=1.05, label=LABELS["pspo"]
            ),
            curves.previous.Line2D(
                [0],
                [0],
                color="0.35",
                marker="D",
                markersize=3.4,
                linestyle="none",
                label="Test-time evaluation",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.54, 0.995),
            ncol=3,
            frameon=False,
            fontsize=8,
            handlelength=1.6,
            handletextpad=0.5,
            columnspacing=1.0,
            borderaxespad=0.2,
        )
        # Deliberately no suptitle or legend title.
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=curves.RUNS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    points, sources, hashes = curves.load_data(
        args.runs_root,
        cohorts=COHORTS,
        expected_steps=STEPS,
    )
    points = curves.with_reward_definition(points, "standard")
    rows = curves.aggregate(points, expected_seeds=curves.FOUR_LAYOUT_SEEDS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, _ = build_figure(rows)
    with curves.plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        curves.save_figure(fig, args.output_dir, NAME)
    curves.previous.write_csv(args.output_dir / "seed_evaluation_points.csv", points)
    curves.previous.write_csv(args.output_dir / "aggregate_evaluation_points.csv", rows)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(
            {
                "figure_size_inches": [WIDTH_INCHES, 2.7],
                "figure_size_mm": [WIDTH_INCHES * 25.4, 2.7 * 25.4],
                "layouts": list(COHORTS),
                "actual_training_steps": {str(size): STEPS[size, "ppo"] for size in COHORTS},
                "reward_definition": "Standard FrozenLake: 1 at goal, 0 otherwise; no shaping or step penalty in reported reward.",
                "safety_definition": "Greedy unshielded trajectory safety (no hole visit); safe timeouts count as safe.",
                "uncertainty": "Mean +/- two sample standard errors across ten training-seed means at every plotted point.",
                "step_zero": "Measured ten-episode initial-policy evaluation before model.learn(), not an inserted zero.",
                "pspo_variant": "Line-segment verify-first, enforcement frequency 1 after every complete PPO train phase.",
                "training_budget": "Every method and layout uses the ~200k-step budget: 204800 steps (100 rollouts) for 16/32; requested 200000, trained 200704 for 64/128. PPO128 is the dedicated 200k run, not a crop of the 400k run.",
                "evaluation_episodes": "Ten per seed for initial/periodic points, 100 per seed for test-time diamonds. PSPO64/128 skip the 200000-step periodic point and retain the separate test-time evaluation at 200704.",
                "line_connections": "Final diamonds are connected to the preceding recorded periodic mean with the method's line colour/style. Connections are visual guides, not extra observations or relocated evaluations.",
                "training_reward_caveat": "Both methods train with goal-distance potential shaping, gamma=0.999 and a 0.001-per-step cost. PSPO explores with a runtime shield; PPO does not. Training optimises a different reward objective from the plotted standard reward.",
                "sources": sources,
                "source_sha256": hashes,
                "seed_points": points,
                "aggregate_points": rows,
            },
            indent=2,
        )
        + "\n"
    )
    (args.output_dir / "README.md").write_text("""# Final FrozenLake scalability figure

`frozenlake_scalability.pdf` is a vector figure, 101.6 × 68.58 mm, below
half-A4 width (105 mm). The PNG is a 220-dpi preview. Set the LaTeX width to
the available column/half-page width; it is already designed at this size.
The four layout titles wrap after FrozenLake to keep the lettering legible.
The x label is shared across all panels; no suptitle or legend title is used.

Columns: 16×16, 32×32, 64×64, 128×128. Top: standard total reward (goal=1,
otherwise=0). Bottom: greedy unshielded trajectory safety rate. Both curves
use ten seeds, with mean ± two sample standard errors. Bands are not clipped
at [0,1]. Black dashed: PPO. Green solid: PSPO-VF. Diamonds: separate test-time
100-episode evaluations. Initial/periodic observations use ten episodes per
seed; step zero is a measured evaluation before any RL training update.
Every test-time diamond is connected to the preceding recorded mean using the
method's line colour/style. Straight connecting segments are visual guides,
not new evaluations; test-time diamonds retain their own 100-episode error bars.

PSPO-VF is PSPO with line-segment LIDs, verify-first, enforcement frequency 1. Its actor
initialisation is safety-only; its exploration is shielded. PPO starts from
random parameters and explores unshielded. BOTH train with potential shaping,
gamma 0.999 and a 0.001-per-step cost. The plotted standard reward is not
the shaped or step-penalised training reward. These are comparisons of the
full setups, not isolated tests of the PSPO update mechanism.

Every curve uses the ~200k-step budget. 16/32 use 204,800 steps per method
(100 full rollouts). 64 and 128 use matched runs that requested 200,000 steps
and trained for 200,704; PPO128 is the dedicated 200k run
(`frozenlake128_shaping_ppo_t200000_20261008T210643Z`), not a crop of the 400k
run, so it has its own 100-episode test-time evaluation at 200,704.
PSPO64/128 have no periodic observation at exactly 200,000; their test-time
diamonds at 200,704 remain at their actual timesteps. No data are fabricated,
smoothed, moved to different timesteps or extrapolated. Full run data remain
unchanged. 256×256 is excluded because this comparison is not complete.

`analysis.json` and the two CSVs retain plotted data, source cohorts and hashes.
All reward/safety summaries and PSPO certificates are audited by the shared
loader. Regenerate from the repository root:

```
.venv/bin/python projects/safe_policy_optimisation/scripts/plot_frozenlake_scalability.py
```
""")
    print(f"Saved {args.output_dir / (NAME + '.pdf')} and PNG preview", flush=True)


if __name__ == "__main__":
    main()
