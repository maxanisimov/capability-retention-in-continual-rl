"""Tabular reward/safety bar chart across completed paper sweeps.

Colour Bomb v1 was run in two parts: baselines/PPO-Shield in
``_sweeps_tabular_colour_bomb_no_pspo`` and PSPO methods in
``_sweeps_tabular_colour_bomb_pspo``. Bridge Crossing v2 has a later
high-iteration PSPO rerun in ``_sweeps_tabular_hiter``. This script merges
those aggregate files per method while keeping the rest of the tabular
environments on the common ``_sweeps_tabular`` output root.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/vol/bitbucket/ma5923/_projects/CertifiedContinualLearning")
OUT_DIR = REPO / "projects/safe_policy_optimisation/figures"

SWEEPS = {
    "Media Streaming": [
        REPO / "outputs/_sweeps_tabular/paper_2503_07671_media_streaming/aggregate/aggregated_metrics.json",
    ],
    "Colour Bomb": [
        REPO / "outputs/_sweeps_tabular_colour_bomb_no_pspo/paper_2503_07671_colour_bomb/aggregate/aggregated_metrics.json",
        REPO / "outputs/_sweeps_tabular_colour_bomb_pspo/paper_2503_07671_colour_bomb/aggregate/aggregated_metrics.json",
    ],
    "Bridge Crossing v1": [
        REPO / "outputs/_sweeps_tabular/paper_2503_07671_bridge_crossing/aggregate/aggregated_metrics.json",
    ],
    "Bridge Crossing v2": [
        REPO / "outputs/_sweeps_tabular/paper_2503_07671_bridge_crossing_v2/aggregate/aggregated_metrics.json",
        REPO / "outputs/_sweeps_tabular_hiter/paper_2503_07671_bridge_crossing_v2/aggregate/aggregated_metrics.json",
    ],
    "Colour Bomb v2": [
        REPO / "outputs/_sweeps_tabular/paper_2503_07671_colour_bomb_v2/aggregate/aggregated_metrics.json",
        REPO / "outputs/_sweeps_tabular_colour_bomb_v2_pspo_precomputed_10k_margin0p5/paper_2503_07671_colour_bomb_v2/aggregate/aggregated_metrics.json",
    ],
    "MiniPacman": [
        REPO / "outputs/_sweeps_tabular/paper_2503_07671_minipacman/aggregate/aggregated_metrics.json",
    ],
}

METHODS = [
    ("ppo_policy", "PPO", "grey"),
    ("ppo_lagrangian/ppo_lagrangian", "PPO-Lagrangian", "red"),
    ("ppo_lagrangian/ppo_pid_lagrangian", "PPO-PID-Lagrangian", "orange"),
    ("cpo/cpo", "CPO", "yellow"),
    ("ppo_shield/shielded", "PPO-Shield", "blue"),
    ("ppo_shield/nominal", "PPO-Shield-Nominal", "lightblue"),
    ("rashomon_policy", "PSPO", "green"),
]
COLORS = [color for _key, _label, color in METHODS]

plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_metrics(paths: list[Path]) -> dict:
    merged: dict = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            merged.update(json.load(handle)["metrics"])
    return merged


def bottom_with_slack(means: list[float], errs: list[float], slack_frac: float = 0.05) -> float:
    los = [m - e for m, e in zip(means, errs)]
    min_lo = min(los)
    magnitude = abs(min_lo) if abs(min_lo) > 1e-9 else 1.0
    return min_lo - slack_frac * magnitude


n_envs = len(SWEEPS)
fig, axes = plt.subplots(n_envs, 2, figsize=(9.6, 2.8 * n_envs))
x = list(range(len(METHODS)))

for row, (env_name, paths) in enumerate(SWEEPS.items()):
    agg = load_metrics(paths)
    means_r, err_r, means_s, err_s = [], [], [], []
    for key, _label, _color in METHODS:
        r = agg[f"{key}.reward.mean_total_reward"]
        s = agg[f"{key}.safety.safety_rate"]
        n = r["n"]
        means_r.append(r["mean"])
        err_r.append(r["std"] / math.sqrt(n) if n > 1 else 0.0)
        means_s.append(s["mean"])
        err_s.append(s["std"] / math.sqrt(n) if n > 1 else 0.0)

    ax = axes[row, 0]
    bottom_r = bottom_with_slack(means_r, err_r)
    ax.bar(x, [m - bottom_r for m in means_r], bottom=bottom_r, yerr=err_r, capsize=3,
           color=COLORS, edgecolor="black", linewidth=0.5,
           error_kw={"linewidth": 1.0, "ecolor": "black"})
    ax.set_ylim(bottom=bottom_r)
    ax.set_ylabel("Total reward")
    ax.set_title(f"{env_name} - Total Reward", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([])

    ax = axes[row, 1]
    ax.bar(x, means_s, yerr=err_s, capsize=3, color=COLORS,
           edgecolor="black", linewidth=0.5, error_kw={"linewidth": 1.0, "ecolor": "black"})
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.0, zorder=0)
    ax.set_ylim(bottom=bottom_with_slack(means_s, err_s), top=1.05)
    ax.set_ylabel("Safety rate")
    ax.set_title(f"{env_name} - Safety Rate", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([])

handles_all = [
    plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[i], edgecolor="black", linewidth=0.5)
    for i in range(len(METHODS))
]
labels_all = [label for _key, label, _color in METHODS]
row_major_order = [0, 4, 1, 5, 2, 6, 3]
handles = [handles_all[i] for i in row_major_order]
labels = [labels_all[i] for i in row_major_order]
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.012), columnspacing=1.4, handletextpad=0.6)

fig.suptitle("Tabular Actor-Critic, Index Encoding: Total Reward and Safety Rate (mean +/- s.e., n=10 seeds)",
             fontsize=13, y=0.998)
fig.tight_layout(rect=[0, 0.043, 1, 0.978])

fig.savefig(OUT_DIR / "tabular_reward_safety.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "tabular_reward_safety.png", bbox_inches="tight", dpi=300)
print("Saved to", OUT_DIR)
