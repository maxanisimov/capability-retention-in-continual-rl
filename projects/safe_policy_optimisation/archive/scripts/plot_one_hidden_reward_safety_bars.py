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
from _aamas_reward_safety import BarMethod, BarPanel, compact_figure, save_compact_figure

REPO = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
OUT_DIR = REPO / "projects/safe_policy_optimisation/figures/aamas"

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


panels = []
for env_name, baseline_aggregate in BASELINE_SWEEPS.items():
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

    panels.append(BarPanel(
        env_name, [v[0] for v in reward_values], [v[1] for v in reward_values],
        [v[0] for v in safety_values], [v[1] for v in safety_values],
    ))

methods = [BarMethod(*method) for method in METHODS]
fig = compact_figure(panels, methods)
save_compact_figure(
    fig, OUT_DIR / "one_hidden_reward_safety", panels=panels, methods=methods,
    se_multiplier=1.0,
    caption=("One-hidden-layer actor--critic, index encoding: total reward and "
             "safety rate (mean $\\pm$ one standard error across seeds). "
             "PSPO uses true one-hidden precomputed runs, not adaptive runs; "
             "the MiniPacman PSPO result is unavailable, not zero."),
)
print("Saved to", OUT_DIR)
