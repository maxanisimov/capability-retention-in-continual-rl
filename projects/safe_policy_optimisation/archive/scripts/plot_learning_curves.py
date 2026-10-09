"""Per-environment learning curves (reward, safety) across seeds, standing colormap.

For each sweep in SWEEPS, reads every seed's per-stage
learning_curves/evaluation_*_summary.csv, aggregates mean_total_reward and
safety_rate across seeds at each eval checkpoint (mean line + shaded +/- stderr
band vs. timestep), and plots all non-adaptive methods overlaid per
environment.

To reuse for another environment or architecture: edit SWEEPS (env label ->
path to that sweep's run directory, i.e. the dir containing seed0, seed1, ...),
then rerun. METHODS/CURVE_FILES/colors should stay in sync with
plot_reward_safety_bars.py.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = "/vol/bitbucket/ma5923/_projects/CertifiedContinualLearning"

# Latest completed sweeps using the 2-hidden-layer actor-critic architecture and
# index (one-hot) state encoding.
SWEEPS = {
    "Media Streaming": f"{REPO}/outputs/_sweeps/20260723_204829/paper_2503_07671_media_streaming",
    "Colour Bomb": f"{REPO}/outputs/_sweeps/20260723_215050/paper_2503_07671_colour_bomb",
    "Bridge Crossing": f"{REPO}/outputs/_sweeps/20260724_124311/paper_2503_07671_bridge_crossing",
    "Bridge Crossing v2": f"{REPO}/outputs/_sweeps/20260724_152054/paper_2503_07671_bridge_crossing_v2",
    "Colour Bomb v2": f"{REPO}/outputs/_sweeps/20260724_164416/paper_2503_07671_colour_bomb_v2",
}

# (metrics-key, display label, color) - keep in sync with plot_reward_safety_bars.py
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

# metrics-key -> path (relative to a seed dir) of its evaluation summary CSV.
# ppo_lagrangian/cpo are "baseline stages" that nest one subfolder per algorithm;
# ppo_shield is a single stage with two eval modes (shielded/nominal) side by side.
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


n_envs = len(SWEEPS)
fig, axes = plt.subplots(n_envs, 2, figsize=(9.6, 3.1 * n_envs))

for row, (env_name, run_dir_str) in enumerate(SWEEPS.items()):
    run_dir = Path(run_dir_str)
    ax_r, ax_s = axes[row, 0], axes[row, 1]

    for key, label, color in METHODS:
        curve = load_curve(run_dir, CURVE_FILES[key])
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

    ax_r.set_title(f"{env_name} — Total Reward (training curve)", fontsize=11)
    ax_r.set_ylabel("Mean episode reward")
    ax_r.set_xlabel("Timestep")

    ax_s.axhline(1.0, color="grey", linestyle="--", linewidth=1.0, zorder=0)
    ax_s.set_title(f"{env_name} — Safety Rate (training curve)", fontsize=11)
    ax_s.set_ylabel("Safety rate")
    ax_s.set_xlabel("Timestep")

handles, labels = axes[0, 0].get_legend_handles_labels()
# fall back to a row with a full-method legend in case row 0 is missing a method
for r in range(n_envs):
    h, l = axes[r, 0].get_legend_handles_labels()
    if len(l) >= len(METHODS):
        handles, labels = h, l
        break
# matplotlib fills multi-column legends column-major; reorder to row-major so a
# 2-row legend reads left-to-right, top-to-bottom - matches plot_reward_safety_bars.py.
if len(labels) == len(METHODS):
    row_major_order = [0, 4, 1, 5, 2, 6, 3]
    handles = [handles[i] for i in row_major_order]
    labels = [labels[i] for i in row_major_order]
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.012), columnspacing=1.4, handletextpad=0.6)

fig.suptitle("2-Hidden-Layer Actor-Critic, Index Encoding: Learning Curves (mean ± s.e. across seeds)",
             fontsize=13, y=0.998)
fig.tight_layout(rect=[0, 0.045, 1, 0.975])

OUT_DIR = f"{REPO}/projects/safe_policy_optimisation/figures"
fig.savefig(f"{OUT_DIR}/two_hidden_learning_curves.pdf", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/two_hidden_learning_curves.png", bbox_inches="tight", dpi=300)
print("Saved to", OUT_DIR)
