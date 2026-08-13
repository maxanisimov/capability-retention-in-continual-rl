"""One-hidden-layer reward/safety bar charts.

Baselines are read from the completed one-hidden sweep aggregates. PSPO is read
from the best completed true one-hidden precomputed PSPO run for each
environment where one exists. Adaptive PSPO is intentionally excluded.
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

BASELINE_SWEEPS = {
    "Bridge Crossing v1": REPO / "outputs/_sweeps_1hidden_bridge_crossing_v1_baselines_only/paper_2503_07671_bridge_crossing/aggregate/aggregated_metrics.json",
    "Bridge Crossing v2": REPO / "outputs/_sweeps_1hidden_missing_all_methods_pspo30000_margin5_no_adaptive/paper_2503_07671_bridge_crossing_v2/aggregate/aggregated_metrics.json",
    "Colour Bomb v1": REPO / "outputs/_sweeps_1hidden_missing_all_methods_pspo30000_margin5_no_adaptive/paper_2503_07671_colour_bomb/aggregate/aggregated_metrics.json",
    "Colour Bomb v2": REPO / "outputs/_sweeps_1hidden/paper_2503_07671_colour_bomb_v2/aggregate/aggregated_metrics.json",
    "Media Streaming": REPO / "outputs/_sweeps_1hidden_media_streaming_baselines_only/paper_2503_07671_media_streaming/aggregate/aggregated_metrics.json",
    "MiniPacman": REPO / "outputs/_sweeps_1hidden_missing_all_methods_pspo30000_margin5_no_adaptive/paper_2503_07671_minipacman/aggregate/aggregated_metrics.json",
}

PSPO_SWEEPS = {
    "Bridge Crossing v1": (
        REPO / "outputs/_pspo_hparam/bridge_crossing_v1_1hidden_precomputed/precomputed/iters_10000__margin_0p5/aggregate/aggregated_metrics.json",
        None,
    ),
    "Bridge Crossing v2": (
        REPO / "outputs/_sweeps_bridge_crossing_v2_pspo_precomputed_iters30000_margin0p5/paper_2503_07671_bridge_crossing_v2/aggregate/aggregated_metrics.json",
        "rashomon_policy",
    ),
    "Colour Bomb v1": (
        REPO / "outputs/_pspo_hparam/colour_bomb_1hidden_precomputed_iters10k20k30k_margins0p5_2_5/precomputed/iters_10000__margin_0p5/aggregate/aggregated_metrics.json",
        None,
    ),
    "Colour Bomb v2": (
        REPO / "outputs/_pspo_hparam/colour_bomb_v2_1hidden_precomputed_iters10k20k30k_margins0p5_2_5/precomputed/iters_10000__margin_0p5/aggregate/aggregated_metrics.json",
        None,
    ),
    "Media Streaming": (
        REPO / "outputs/_pspo_hparam/media_streaming_1hidden_precomputed_iters10k20k30k_margins0p5_2_5/precomputed/iters_20000__margin_0p5/aggregate/aggregated_metrics.json",
        None,
    ),
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


def load_metric(aggregate: Path, metric: str, prefix: str | None = None) -> tuple[float, float, int] | None:
    if not aggregate.exists():
        return None
    with aggregate.open() as f:
        metrics = json.load(f)["metrics"]
    key = f"{prefix}.{metric}" if prefix else metric
    if key not in metrics:
        return None
    value = metrics[key]
    n = int(value["n"])
    sem = float(value["std"]) / math.sqrt(n) if n > 1 else 0.0
    return float(value["mean"]), sem, n


def bottom_with_slack(values: list[tuple[float, float]], slack_frac: float = 0.05) -> float:
    min_lo = min(mean - err for mean, err in values)
    magnitude = abs(min_lo) if abs(min_lo) > 1e-9 else 1.0
    return min_lo - slack_frac * magnitude


n_envs = len(BASELINE_SWEEPS)
fig, axes = plt.subplots(n_envs, 2, figsize=(9.4, 3.1 * n_envs))
x = list(range(len(METHODS)))

for row, (env_name, baseline_aggregate) in enumerate(BASELINE_SWEEPS.items()):
    reward_values: list[tuple[float | None, float, int | None]] = []
    safety_values: list[tuple[float | None, float, int | None]] = []

    for key, _label, _color in METHODS:
        if key == "rashomon_policy":
            pspo_source = PSPO_SWEEPS.get(env_name)
            if pspo_source is None:
                reward_values.append((None, 0.0, None))
                safety_values.append((None, 0.0, None))
                continue
            aggregate, prefix = pspo_source
        else:
            aggregate, prefix = baseline_aggregate, key

        reward = load_metric(aggregate, "reward.mean_total_reward", prefix)
        safety = load_metric(aggregate, "safety.safety_rate", prefix)
        reward_values.append(reward if reward is not None else (None, 0.0, None))
        safety_values.append(safety if safety is not None else (None, 0.0, None))

    present_rewards = [(mean, err) for mean, err, _n in reward_values if mean is not None]
    present_safety = [(mean, err) for mean, err, _n in safety_values if mean is not None]

    ax_r = axes[row, 0]
    bottom_r = bottom_with_slack(present_rewards)
    for i, (_key, _label, color) in enumerate(METHODS):
        mean, err, _n = reward_values[i]
        if mean is None:
            ax_r.text(i, bottom_r, "n/a", ha="center", va="bottom", rotation=90, fontsize=7)
            continue
        ax_r.bar(i, mean - bottom_r, bottom=bottom_r, yerr=err, capsize=3, color=color,
                 edgecolor="black", linewidth=0.5, error_kw={"linewidth": 1.0, "ecolor": "black"})
    ax_r.set_ylim(bottom=bottom_r)
    ax_r.set_ylabel("Total reward")
    ax_r.set_title(f"{env_name} - Total Reward", fontsize=11)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([])

    ax_s = axes[row, 1]
    bottom_s = bottom_with_slack(present_safety)
    for i, (_key, _label, color) in enumerate(METHODS):
        mean, err, _n = safety_values[i]
        if mean is None:
            ax_s.text(i, bottom_s, "n/a", ha="center", va="bottom", rotation=90, fontsize=7)
            continue
        ax_s.bar(i, mean, yerr=err, capsize=3, color=color,
                 edgecolor="black", linewidth=0.5, error_kw={"linewidth": 1.0, "ecolor": "black"})
    ax_s.axhline(1.0, color="grey", linestyle="--", linewidth=1.0, zorder=0)
    ax_s.set_ylim(bottom=bottom_s, top=1.05)
    ax_s.set_ylabel("Safety rate")
    ax_s.set_title(f"{env_name} - Safety Rate", fontsize=11)
    ax_s.set_xticks(x)
    ax_s.set_xticklabels([])

handles_all = [
    plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5)
    for _key, _label, color in METHODS
]
labels_all = [label for _key, label, _color in METHODS]
row_major_order = [0, 4, 1, 5, 2, 6, 3]
handles = [handles_all[i] for i in row_major_order]
labels = [labels_all[i] for i in row_major_order]
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.012), columnspacing=1.4, handletextpad=0.6)

fig.suptitle("One-Hidden-Layer Actor-Critic, Index Encoding: Total Reward and Safety Rate (mean +/- s.e., n=10 seeds)",
             fontsize=13, y=0.998)
fig.tight_layout(rect=[0, 0.045, 1, 0.975])

OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "one_hidden_reward_safety.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "one_hidden_reward_safety.png", bbox_inches="tight", dpi=300)
print("Saved to", OUT_DIR)
