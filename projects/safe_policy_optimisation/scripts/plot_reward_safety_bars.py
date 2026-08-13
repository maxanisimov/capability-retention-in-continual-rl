"""Reward/safety bar chart for a set of sweeps, in the project's standing colormap.

Each panel's y-axis bottom is set to the minimum (mean - stderr) across that
panel's bars, minus a small slack margin for visibility - there is no fixed
axis ceiling.

To reuse for another environment or architecture: edit SWEEPS (env label ->
path to that sweep's aggregate/aggregated_metrics.json), then rerun.
METHODS/colors should stay in sync across all such figures.
"""

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/vol/bitbucket/ma5923/_projects/CertifiedContinualLearning"

# Latest completed sweeps using the 2-hidden-layer actor-critic architecture and
# index (one-hot) state encoding. The later 20260724_2325xx reruns have complete
# curve checkpoints but incomplete final adaptive PSPO metrics, so they are not
# used for the final-metric bars.
SWEEPS = {
    "Media Streaming": f"{REPO}/outputs/_sweeps/20260723_204829/paper_2503_07671_media_streaming/aggregate/aggregated_metrics.json",
    "Colour Bomb": f"{REPO}/outputs/_sweeps/20260723_215050/paper_2503_07671_colour_bomb/aggregate/aggregated_metrics.json",
    "Bridge Crossing": f"{REPO}/outputs/_sweeps/20260724_124311/paper_2503_07671_bridge_crossing/aggregate/aggregated_metrics.json",
    "Bridge Crossing v2": f"{REPO}/outputs/_sweeps/20260724_152054/paper_2503_07671_bridge_crossing_v2/aggregate/aggregated_metrics.json",
    "Colour Bomb v2": f"{REPO}/outputs/_sweeps/20260724_164416/paper_2503_07671_colour_bomb_v2/aggregate/aggregated_metrics.json",
}

LOW_N_CAVEATS = {}

# (metrics-key, display label, color), in the order bars are drawn.
# Colormap is a standing user preference for every reward/safety bar chart in
# this project - keep this mapping in sync if new methods are added.
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
    "pdf.fonttype": 42,  # embed as real fonts, not paths (AAAI/ACM friendly)
    "ps.fonttype": 42,
})

n_envs = len(SWEEPS)
fig, axes = plt.subplots(n_envs, 2, figsize=(9.2, 3.1 * n_envs))

x = list(range(len(METHODS)))


def bottom_with_slack(means, errs, slack_frac=0.05):
    """Axis bottom = min(mean - err) - slack.

    Slack is sized to the *minimum bound's own magnitude*, not the panel's
    full span - sizing it off the span pushes the bottom well past the
    minimum bar whenever another bar in the same panel sits near zero
    (e.g. media streaming's PPO-Shield-Nominal at ~0 next to CPO at ~-24),
    making the axis bottom look disconnected from the actual minimum bar.
    """
    los = [m - e for m, e in zip(means, errs)]
    min_lo = min(los)
    magnitude = abs(min_lo) if abs(min_lo) > 1e-9 else 1.0
    return min_lo - slack_frac * magnitude


for row, env_name in enumerate(SWEEPS):
    with open(SWEEPS[env_name]) as f:
        agg = json.load(f)["metrics"]

    means_r, err_r, ns_r = [], [], []
    means_s, err_s = [], []
    for key, _label, _color in METHODS:
        r = agg[f"{key}.reward.mean_total_reward"]
        s = agg[f"{key}.safety.safety_rate"]
        n = r["n"]
        means_r.append(r["mean"])
        err_r.append(r["std"] / math.sqrt(n) if n > 1 else 0.0)
        ns_r.append(n)
        means_s.append(s["mean"])
        err_s.append(s["std"] / math.sqrt(n) if n > 1 else 0.0)

    # --- Total reward panel ---
    # Bars are anchored at the panel's own minimum (bottom_r), not at 0: with
    # bottom=0, a bar's length is |mean|, so for all-negative rewards (e.g.
    # media streaming) the WORST method (most negative) draws the longest bar.
    # Anchoring at bottom_r makes bar height = mean - bottom_r, which is
    # monotonic in mean regardless of sign - higher reward -> taller bar.
    ax = axes[row, 0]
    bottom_r = bottom_with_slack(means_r, err_r)
    ax.bar(x, [m - bottom_r for m in means_r], bottom=bottom_r, yerr=err_r, capsize=3, color=COLORS,
           edgecolor="black", linewidth=0.5, error_kw={"linewidth": 1.0, "ecolor": "black"})
    ax.set_ylim(bottom=bottom_r)
    ax.set_ylabel("Total reward")
    ax.set_title(f"{env_name} — Total Reward", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    for i, (key, _label, _color) in enumerate(METHODS):
        caveat = LOW_N_CAVEATS.get((env_name, key))
        if caveat:
            ax.annotate(caveat, (i, means_r[i]), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=6.5, color="black", rotation=90, va="bottom")

    # --- Safety rate panel ---
    ax = axes[row, 1]
    ax.bar(x, means_s, yerr=err_s, capsize=3, color=COLORS,
           edgecolor="black", linewidth=0.5, error_kw={"linewidth": 1.0, "ecolor": "black"})
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.0, zorder=0)
    ax.set_ylim(bottom=bottom_with_slack(means_s, err_s), top=1.05)
    ax.set_ylabel("Safety rate")
    ax.set_title(f"{env_name} — Safety Rate", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([])

# matplotlib fills multi-column legends column-major; reorder the handles so a
# 2-row x 4-col legend reads left-to-right, top-to-bottom in the same order as
# the bars (row-major), instead of matplotlib's default column-major fill.
handles_all = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[i], edgecolor="black", linewidth=0.5)
               for i in range(len(METHODS))]
labels_all = [label for _key, label, _color in METHODS]
row_major_order = [0, 4, 1, 5, 2, 6, 3]
handles = [handles_all[i] for i in row_major_order]
labels = [labels_all[i] for i in row_major_order]
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.012), columnspacing=1.4, handletextpad=0.6)

fig.suptitle("2-Hidden-Layer Actor-Critic, Index Encoding: Total Reward and Safety Rate (mean ± s.e., n=10 seeds)",
             fontsize=13, y=0.998)
fig.tight_layout(rect=[0, 0.045, 1, 0.975])

OUT_DIR = f"{REPO}/projects/safe_policy_optimisation/figures"
fig.savefig(f"{OUT_DIR}/two_hidden_reward_safety.pdf", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/two_hidden_reward_safety.png", bbox_inches="tight", dpi=300)
print("Saved to", OUT_DIR)
