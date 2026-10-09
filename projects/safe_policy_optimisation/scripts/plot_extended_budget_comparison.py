"""Reward/safety bars for the extended-budget PSPO vs RL-baseline comparison.

Reads `docs/extended_budget_comparison/extended_budget_summary.csv` (written by
`compare_extended_budget_pspo_vs_baselines.py`) and draws, per environment, the
standard-budget and extended-budget result side by side for every method.

Colors follow the project's standing method colormap; keep METHODS in sync with
`plot_reward_safety_bars.py`. A method with no extended-budget result is drawn
with its standard bar only and marked `n/a` so a missing sweep stays visible.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[3]
DOC_DIR = REPO / "projects/safe_policy_optimisation/docs/extended_budget_comparison"

# (label, color) in bar order. Standing colormap - do not substitute a generic palette.
METHODS = [
    ("PSPO", "green"),
    ("PPO", "grey"),
    ("PPO-Lagrangian", "red"),
    ("PPO-PID-Lagrangian", "orange"),
    ("CPO", "yellow"),
    ("PPO-Shield", "blue"),
    ("PPO-Shield-Nominal", "lightblue"),
]
ENVIRONMENTS = ["Bridge Crossing v2", "MiniPacman"]

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_summary(path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    table: dict[tuple[str, str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table[(row["environment"], row["setting"], row["method"])] = {
                "reward": float(row["total_reward_mean"]),
                "reward_err": float(row["total_reward_2sem"]),
                "safety": float(row["safety_rate_mean"]),
                "safety_err": float(row["safety_rate_2sem"]),
                "budget": int(row["budget_timesteps"]),
            }
    return table


def draw(table, out_stub: Path) -> None:
    fig, axes = plt.subplots(len(ENVIRONMENTS), 2, figsize=(11.0, 3.4 * len(ENVIRONMENTS)))
    width = 0.38
    xs = list(range(len(METHODS)))

    for row_index, env in enumerate(ENVIRONMENTS):
        for col_index, (metric, err_key, title) in enumerate(
            [("reward", "reward_err", "Mean total reward"), ("safety", "safety_err", "Safety rate")]
        ):
            ax = axes[row_index][col_index]
            missing_labels = []
            for x, (method, color) in zip(xs, METHODS):
                for offset, setting, alpha, hatch in (
                    (-width / 2, "standard", 0.45, "//"),
                    (width / 2, "extended", 1.0, None),
                ):
                    entry = table.get((env, setting, method))
                    if entry is None:
                        if setting == "extended":
                            missing_labels.append(x)
                        continue
                    ax.bar(
                        x + offset,
                        entry[metric],
                        width,
                        yerr=entry[err_key],
                        color=color,
                        alpha=alpha,
                        hatch=hatch,
                        edgecolor="black",
                        linewidth=0.6,
                        capsize=2.5,
                        error_kw={"elinewidth": 0.9},
                    )
            for x in missing_labels:
                ax.text(
                    x + width / 2, 0.02, "n/a",
                    ha="center", va="bottom", fontsize=8, rotation=90, color="dimgrey",
                )
            ax.set_xticks(xs)
            ax.set_xticklabels([m for m, _ in METHODS], rotation=30, ha="right")
            ax.set_title(f"{env} — {title}")
            ax.set_ylim(0.0, 1.08)
            ax.axhline(0.0, color="black", linewidth=0.6)
            ax.grid(axis="y", alpha=0.25, linewidth=0.6)
            ax.set_axisbelow(True)

    fig.legend(
        handles=[
            Patch(facecolor="white", edgecolor="black", hatch="//", alpha=0.6, label="standard budget"),
            Patch(facecolor="dimgrey", edgecolor="black", label="extended budget"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out_stub.with_suffix(".png"), dpi=200)
    fig.savefig(out_stub.with_suffix(".pdf"))
    print(f"Wrote {out_stub.with_suffix('.png')} and {out_stub.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DOC_DIR / "extended_budget_summary.csv")
    parser.add_argument("--out-stub", type=Path, default=DOC_DIR / "extended_budget_comparison")
    args = parser.parse_args()
    draw(load_summary(args.summary_csv), args.out_stub)


if __name__ == "__main__":
    main()
