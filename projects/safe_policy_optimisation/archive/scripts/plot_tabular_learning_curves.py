"""Tabular reward/safety learning curves across completed paper sweeps."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path("/vol/bitbucket/ma5923/_projects/CertifiedContinualLearning")
OUT_DIR = REPO / "projects/safe_policy_optimisation/figures"

RUN_DIRS = {
    "Media Streaming": REPO / "outputs/_sweeps_tabular/paper_2503_07671_media_streaming",
    "Colour Bomb": REPO / "outputs/_sweeps_tabular_colour_bomb_no_pspo/paper_2503_07671_colour_bomb",
    "Bridge Crossing v1": REPO / "outputs/_sweeps_tabular/paper_2503_07671_bridge_crossing",
    "Bridge Crossing v2": REPO / "outputs/_sweeps_tabular/paper_2503_07671_bridge_crossing_v2",
    "Colour Bomb v2": REPO / "outputs/_sweeps_tabular/paper_2503_07671_colour_bomb_v2",
    "MiniPacman": REPO / "outputs/_sweeps_tabular/paper_2503_07671_minipacman",
}

COLOUR_BOMB_PSPO_DIR = REPO / "outputs/_sweeps_tabular_colour_bomb_pspo/paper_2503_07671_colour_bomb"
BRIDGE_CROSSING_V2_PSPO_DIR = REPO / "outputs/_sweeps_tabular_hiter/paper_2503_07671_bridge_crossing_v2"
COLOUR_BOMB_V2_PSPO_DIR = (
    REPO
    / "outputs/_sweeps_tabular_colour_bomb_v2_pspo_precomputed_10k_margin0p5/paper_2503_07671_colour_bomb_v2"
)

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
    "rashomon_policy": "rashomon_policy/learning_curves/evaluation_unshielded_summary.csv",
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


def run_dir_for(env_name: str, method_key: str) -> Path:
    if env_name == "Colour Bomb" and method_key == "rashomon_policy":
        return COLOUR_BOMB_PSPO_DIR
    if env_name == "Bridge Crossing v2" and method_key == "rashomon_policy":
        return BRIDGE_CROSSING_V2_PSPO_DIR
    if env_name == "Colour Bomb v2" and method_key == "rashomon_policy":
        return COLOUR_BOMB_V2_PSPO_DIR
    return RUN_DIRS[env_name]


def load_curve(run_dir: Path, rel_path: str) -> pd.DataFrame | None:
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


n_envs = len(RUN_DIRS)
fig, axes = plt.subplots(n_envs, 2, figsize=(9.8, 2.8 * n_envs))

for row, env_name in enumerate(RUN_DIRS):
    ax_r, ax_s = axes[row, 0], axes[row, 1]

    for key, label, color in METHODS:
        curve = load_curve(run_dir_for(env_name, key), CURVE_FILES[key])
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

    ax_s.axhline(1.0, color="grey", linestyle="--", linewidth=1.0, zorder=0)
    ax_s.set_title(f"{env_name} - Safety Rate (training curve)", fontsize=11)
    ax_s.set_ylabel("Safety rate")
    ax_s.set_xlabel("Timestep")

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

fig.suptitle("Tabular Actor-Critic, Index Encoding: Learning Curves (mean +/- s.e. across seeds)",
             fontsize=13, y=0.998)
fig.tight_layout(rect=[0, 0.043, 1, 0.978])

fig.savefig(OUT_DIR / "tabular_learning_curves.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "tabular_learning_curves.png", bbox_inches="tight", dpi=300)
print("Saved to", OUT_DIR)
