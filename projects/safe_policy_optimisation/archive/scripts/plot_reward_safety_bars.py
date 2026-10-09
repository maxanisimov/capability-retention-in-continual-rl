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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from _aamas_reward_safety import BarMethod, BarPanel, compact_figure, save_compact_figure

REPO = str(Path(__file__).resolve().parents[3])

# Latest completed sweeps using the 2-hidden-layer actor-critic architecture and
# index (one-hot) state encoding. The later 20260724_2325xx reruns have complete
# curve checkpoints but incomplete final PSPO metrics, so they are not
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
panels = []
for env_name, aggregate in SWEEPS.items():
    with open(aggregate) as handle:
        agg = json.load(handle)["metrics"]
    means_r, err_r, means_s, err_s = [], [], [], []
    for key, _label, _color in METHODS:
        r = agg[f"{key}.reward.mean_total_reward"]
        s = agg[f"{key}.safety.safety_rate"]
        n = r["n"]
        means_r.append(r["mean"])
        err_r.append(r["std"] / math.sqrt(n) if n > 1 else 0.0)
        means_s.append(s["mean"])
        err_s.append(s["std"] / math.sqrt(n) if n > 1 else 0.0)
    panels.append(BarPanel(env_name, means_r, err_r, means_s, err_s))

methods = [BarMethod(*method) for method in METHODS]
fig = compact_figure(panels, methods)
stem = Path(REPO) / "projects/safe_policy_optimisation/figures/aamas/two_hidden_reward_safety"
save_compact_figure(
    fig, stem, panels=panels, methods=methods, se_multiplier=1.0,
    caption=("Two-hidden-layer actor--critic, index encoding: total reward and "
             "safety rate (mean $\\pm$ one standard error across seeds)."),
)
print("Saved to", stem.parent)
