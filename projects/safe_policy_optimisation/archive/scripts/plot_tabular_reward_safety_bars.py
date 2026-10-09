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
from _aamas_reward_safety import BarMethod, BarPanel, compact_figure, save_compact_figure

REPO = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
OUT_DIR = REPO / "projects/safe_policy_optimisation/figures/aamas"

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
def load_metrics(paths: list[Path]) -> dict:
    merged: dict = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            merged.update(json.load(handle)["metrics"])
    return merged


panels = []
for env_name, paths in SWEEPS.items():
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

    panels.append(BarPanel(env_name, means_r, err_r, means_s, err_s))

methods = [BarMethod(*method) for method in METHODS]
fig = compact_figure(panels, methods)
save_compact_figure(
    fig, OUT_DIR / "tabular_reward_safety", panels=panels, methods=methods,
    se_multiplier=1.0,
    caption=("Tabular actor--critic, index encoding: total reward and safety rate "
             "(mean $\\pm$ one standard error across seeds)."),
)
print("Saved to", OUT_DIR)
