"""One-hidden-layer learning curves.

Baselines are read from completed one-hidden sweeps. PSPO is read from the best
completed true one-hidden precomputed PSPO run where available. Adaptive PSPO is
intentionally excluded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path("/vol/bitbucket/ma5923/_projects/CertifiedContinualLearning")
OUT_DIR = REPO / "projects/safe_policy_optimisation/figures"

BASELINE_SWEEPS = {
    "Bridge Crossing v1": REPO / "outputs/_sweeps_1hidden_bridge_crossing_v1_baselines_only/paper_2503_07671_bridge_crossing",
    "Bridge Crossing v2": REPO / "outputs/_sweeps_1hidden_missing_all_methods_pspo30000_margin5_no_adaptive/paper_2503_07671_bridge_crossing_v2",
    "Colour Bomb v1": REPO / "outputs/_sweeps_1hidden_missing_all_methods_pspo30000_margin5_no_adaptive/paper_2503_07671_colour_bomb",
    "Colour Bomb v2": REPO / "outputs/_sweeps_1hidden/paper_2503_07671_colour_bomb_v2",
    "Media Streaming": REPO / "outputs/_sweeps_1hidden_media_streaming_baselines_only/paper_2503_07671_media_streaming",
    "MiniPacman": REPO / "outputs/_sweeps_1hidden_missing_all_methods_pspo30000_margin5_no_adaptive/paper_2503_07671_minipacman",
}

PSPO_SWEEPS = {
    "Bridge Crossing v1": (
        REPO / "outputs/_pspo_hparam/bridge_crossing_v1_1hidden_precomputed/precomputed/iters_10000__margin_0p5/runs",
        "learning_curves/evaluation_unshielded_summary.csv",
    ),
    "Bridge Crossing v2": (
        REPO / "outputs/_sweeps_bridge_crossing_v2_pspo_precomputed_iters30000_margin0p5/paper_2503_07671_bridge_crossing_v2",
        "rashomon_policy/learning_curves/evaluation_unshielded_summary.csv",
    ),
    "Colour Bomb v1": (
        REPO / "outputs/_pspo_hparam/colour_bomb_1hidden_precomputed_iters10k20k30k_margins0p5_2_5/precomputed/iters_10000__margin_0p5/runs",
        "learning_curves/evaluation_unshielded_summary.csv",
    ),
    "Colour Bomb v2": (
        REPO / "outputs/_pspo_hparam/colour_bomb_v2_1hidden_precomputed_iters10k20k30k_margins0p5_2_5/precomputed/iters_10000__margin_0p5/runs",
        "learning_curves/evaluation_unshielded_summary.csv",
    ),
    "Media Streaming": (
        REPO / "outputs/_pspo_hparam/media_streaming_1hidden_precomputed_iters10k20k30k_margins0p5_2_5/precomputed/iters_20000__margin_0p5/runs",
        "learning_curves/evaluation_unshielded_summary.csv",
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

CURVE_FILES = {
    "ppo_policy": "ppo_policy/learning_curves/evaluation_unshielded_summary.csv",
    "ppo_lagrangian/ppo_lagrangian": "ppo_lagrangian/learning_curves/ppo_lagrangian/evaluation_unshielded_summary.csv",
    "ppo_lagrangian/ppo_pid_lagrangian": "ppo_lagrangian/learning_curves/ppo_pid_lagrangian/evaluation_unshielded_summary.csv",
    "cpo/cpo": "cpo/learning_curves/cpo/evaluation_unshielded_summary.csv",
    "ppo_shield/shielded": "ppo_shield/learning_curves/evaluation_shielded_summary.csv",
    "ppo_shield/nominal": "ppo_shield/learning_curves/evaluation_unshielded_summary.csv",
}

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


def load_curve(run_dir: Path, rel_path: str) -> pd.DataFrame | None:
    if not run_dir.exists():
        return None
    seed_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("seed"))
    frames = []
    for seed_dir in seed_dirs:
        csv_path = seed_dir / rel_path
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, usecols=["eval_index", "timestep", "mean_total_reward", "safety_rate"])
        df["seed"] = seed_dir.name
        frames.append(df)
    if not frames:
        return None
    all_df = pd.concat(frames, ignore_index=True)
    grouped = all_df.groupby("eval_index").agg(
        timestep=("timestep", "first"),
        reward_mean=("mean_total_reward", "mean"),
        reward_sem=("mean_total_reward", "sem"),
        safety_mean=("safety_rate", "mean"),
        safety_sem=("safety_rate", "sem"),
        n=("seed", "nunique"),
    ).reset_index()
    grouped[["reward_sem", "safety_sem"]] = grouped[["reward_sem", "safety_sem"]].fillna(0.0)
    return grouped


n_envs = len(BASELINE_SWEEPS)
fig, axes = plt.subplots(n_envs, 2, figsize=(9.6, 3.1 * n_envs))

for row, (env_name, baseline_run_dir) in enumerate(BASELINE_SWEEPS.items()):
    ax_r, ax_s = axes[row, 0], axes[row, 1]

    for key, label, color in METHODS:
        if key == "rashomon_policy":
            pspo_source = PSPO_SWEEPS.get(env_name)
            if pspo_source is None:
                continue
            run_dir, rel_path = pspo_source
        else:
            run_dir, rel_path = baseline_run_dir, CURVE_FILES[key]

        curve = load_curve(run_dir, rel_path)
        if curve is None or curve.empty:
            continue
        n_seeds = int(curve["n"].max())
        curve_label = label if n_seeds > 1 else f"{label} (n=1)"

        ax_r.plot(curve["timestep"], curve["reward_mean"], color=color, linewidth=1.3, label=curve_label)
        ax_r.fill_between(curve["timestep"], curve["reward_mean"] - curve["reward_sem"],
                          curve["reward_mean"] + curve["reward_sem"], color=color, alpha=0.15, linewidth=0)

        ax_s.plot(curve["timestep"], curve["safety_mean"], color=color, linewidth=1.3, label=curve_label)
        ax_s.fill_between(curve["timestep"], curve["safety_mean"] - curve["safety_sem"],
                          curve["safety_mean"] + curve["safety_sem"], color=color, alpha=0.15, linewidth=0)

    ax_r.set_title(f"{env_name} - Total Reward (training curve)", fontsize=11)
    ax_r.set_ylabel("Mean episode reward")
    ax_r.set_xlabel("Timestep")
    if env_name not in PSPO_SWEEPS:
        ax_r.text(0.99, 0.05, "PSPO n/a", transform=ax_r.transAxes,
                  ha="right", va="bottom", fontsize=7)

    ax_s.axhline(1.0, color="grey", linestyle="--", linewidth=1.0, zorder=0)
    ax_s.set_title(f"{env_name} - Safety Rate (training curve)", fontsize=11)
    ax_s.set_ylabel("Safety rate")
    ax_s.set_xlabel("Timestep")
    if env_name not in PSPO_SWEEPS:
        ax_s.text(0.99, 0.05, "PSPO n/a", transform=ax_s.transAxes,
                  ha="right", va="bottom", fontsize=7)

handles, labels = axes[0, 0].get_legend_handles_labels()
for r in range(n_envs):
    h, l = axes[r, 0].get_legend_handles_labels()
    if len(l) >= len(METHODS):
        handles, labels = h, l
        break
if len(labels) == len(METHODS):
    row_major_order = [0, 4, 1, 5, 2, 6, 3]
    handles = [handles[i] for i in row_major_order]
    labels = [labels[i] for i in row_major_order]
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.012), columnspacing=1.4, handletextpad=0.6)

fig.suptitle("One-Hidden-Layer Actor-Critic, Index Encoding: Learning Curves (mean +/- s.e. across seeds)",
             fontsize=13, y=0.998)
fig.tight_layout(rect=[0, 0.045, 1, 0.975])

OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "one_hidden_learning_curves.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "one_hidden_learning_curves.png", bbox_inches="tight", dpi=300)
print("Saved to", OUT_DIR)
