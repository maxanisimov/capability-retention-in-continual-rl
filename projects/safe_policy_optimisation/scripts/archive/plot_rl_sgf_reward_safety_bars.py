"""Test-time reward/safety bars: RL-SGF (safe init) vs the default safe-RL baselines and PSPO.

Same style and standing colormap as ``plot_reward_safety_bars.py``; RL-SGF is
brown. One row per environment, total reward on the left and safety rate on the
right, mean +/- s.e. over 10 seeds of 100-episode greedy evaluations.

Sources (all final ``metrics.json`` test-time evaluations):

* PPO-Lagrangian, PPO-PID-Lagrangian, CPO, PPO-Shield: the canonical default
  (not safe-initialised) two-hidden baselines in
  ``docs/two_hidden_safe_rl_baselines/safe_rl_baseline_per_seed.csv``; for
  Bridge Crossing v2 and MiniPacman the extended-budget runs in
  ``docs/extended_budget_comparison/extended_budget_per_seed.csv`` instead,
  whose budgets are closest to PSPO's. PPO-Shield is evaluated with its runtime
  shield attached, as in those tables.
* PSPO: the default runs named in ``docs/pspo_best_settings.yaml``.
* RL-SGF: the safe-initialised runs (``RL_SGF_ROOT``), step size 0.1, alpha 1,
  20 episodes per update, warm-started from the same safe policy as PSPO.
"""

import csv
import json
import math
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "projects/safe_policy_optimisation/docs"
RL_SGF_ROOT = (
    REPO
    / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
    / "rl_sgf_safe_initialised"
)
OUT_DIR = Path(__file__).resolve().parents[1] / "figures"

# display label -> (baseline CSV environment name, pspo_best_settings key, RL-SGF task dir,
#                   use extended-budget baselines)
ENVIRONMENTS = {
    "Media Streaming": ("Media Streaming", "media_streaming", "media_streaming", False),
    "Colour Bomb": ("Colour Bomb v1", "colour_bomb", "colour_bomb", False),
    "Bridge Crossing": ("Bridge Crossing v1", "bridge_crossing_v1", "bridge_crossing", False),
    "Bridge Crossing v2": ("Bridge Crossing v2", "bridge_crossing_v2", "bridge_crossing_v2", True),
    "Colour Bomb v2": ("Colour Bomb v2", "colour_bomb_v2", "colour_bomb_v2", False),
    "MiniPacman": ("MiniPacman", "mini_pacman", "mini_pacman", True),
}

# (display label, color) in bar order; colors follow the project's standing colormap.
METHODS = [
    ("PPO-Lagrangian", "red"),
    ("PPO-PID-Lagrangian", "orange"),
    ("CPO", "yellow"),
    ("PPO-Shield", "blue"),
    ("PSPO", "green"),
    ("RL-SGF", "brown"),
]
COLORS = [color for _label, color in METHODS]
BASELINES = {"PPO-Lagrangian", "PPO-PID-Lagrangian", "CPO", "PPO-Shield"}

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


def baseline_rows() -> dict[tuple[str, str], list[tuple[float, float]]]:
    """(environment, method) -> per-seed (reward, safety) for default and extended baselines."""

    standard: dict[tuple[str, str], list[tuple[float, float]]] = {}
    with open(DOCS / "two_hidden_safe_rl_baselines/safe_rl_baseline_per_seed.csv") as handle:
        for row in csv.DictReader(handle):
            standard.setdefault((row["environment"], row["method"]), []).append(
                (float(row["total_reward"]), float(row["safety_rate"]))
            )
    extended: dict[tuple[str, str], list[tuple[float, float]]] = {}
    with open(DOCS / "extended_budget_comparison/extended_budget_per_seed.csv") as handle:
        for row in csv.DictReader(handle):
            if row["setting"] == "extended":
                extended.setdefault((row["environment"], row["method"]), []).append(
                    (float(row["total_reward"]), float(row["safety_rate"]))
                )
    return standard, extended


def metrics_rows(seed_dirs: list[Path], key: str | None = None) -> list[tuple[float, float]]:
    rows = []
    for seed_dir in seed_dirs:
        metrics = json.loads((seed_dir / "metrics.json").read_text())
        metrics = metrics[key] if key else metrics
        rows.append((metrics["reward"]["mean_total_reward"], metrics["safety"]["safety_rate"]))
    return rows


def seed_dirs(root: Path) -> list[Path]:
    dirs = sorted(root.glob("seed[0-9]*"), key=lambda p: int(p.name[4:]))
    return [d for d in dirs if (d / "metrics.json").is_file()]


def mean_se(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var / len(values))


def bottom_with_slack(means, errs, slack_frac=0.05):
    """Axis bottom = min(mean - err) - slack sized to that bound's own magnitude."""
    min_lo = min(m - e for m, e in zip(means, errs))
    magnitude = abs(min_lo) if abs(min_lo) > 1e-9 else 1.0
    return min_lo - slack_frac * magnitude


def collect() -> dict[str, dict[str, list[tuple[float, float]]]]:
    standard, extended = baseline_rows()
    best = yaml.safe_load((DOCS / "pspo_best_settings.yaml").read_text())["environments"]
    data: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for label, (csv_env, pspo_key, task, use_extended) in ENVIRONMENTS.items():
        source = extended if use_extended else standard
        per_method = {method: source[(csv_env, method)] for method in BASELINES}
        run = best[pspo_key]["source_run"]
        per_method["PSPO"] = metrics_rows(seed_dirs(Path(run["path"]) / run["architecture_subdir"]))
        per_method["RL-SGF"] = metrics_rows(seed_dirs(RL_SGF_ROOT / task / "rl_sgf"), key="rl_sgf")
        for method, rows in per_method.items():
            if len(rows) != 10:
                raise ValueError(f"{label}/{method}: expected 10 seeds, found {len(rows)}.")
        data[label] = per_method
    return data


def main() -> None:
    data = collect()
    n_envs = len(data)
    fig, axes = plt.subplots(n_envs, 2, figsize=(9.2, 3.1 * n_envs))
    x = list(range(len(METHODS)))
    summary = {}
    for row, (env_name, per_method) in enumerate(data.items()):
        stats_r = [mean_se([r for r, _s in per_method[m]]) for m, _c in METHODS]
        stats_s = [mean_se([s for _r, s in per_method[m]]) for m, _c in METHODS]
        summary[env_name] = {
            m: {"reward": stats_r[i], "safety": stats_s[i]} for i, (m, _c) in enumerate(METHODS)
        }
        means_r, err_r = [m for m, _e in stats_r], [e for _m, e in stats_r]
        means_s, err_s = [m for m, _e in stats_s], [e for _m, e in stats_s]

        # Reward bars are anchored at the panel minimum so taller always means more reward,
        # including for all-negative rewards (Media Streaming).
        ax = axes[row, 0]
        bottom_r = bottom_with_slack(means_r, err_r)
        ax.bar(x, [m - bottom_r for m in means_r], bottom=bottom_r, yerr=err_r, capsize=3,
               color=COLORS, edgecolor="black", linewidth=0.5,
               error_kw={"linewidth": 1.0, "ecolor": "black"})
        ax.set_ylim(bottom=bottom_r)
        ax.set_ylabel("Total reward")
        ax.set_title(f"{env_name} — Total Reward", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([])

        ax = axes[row, 1]
        ax.bar(x, means_s, yerr=err_s, capsize=3, color=COLORS, edgecolor="black",
               linewidth=0.5, error_kw={"linewidth": 1.0, "ecolor": "black"})
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.0, zorder=0)
        ax.set_ylim(bottom=bottom_with_slack(means_s, err_s), top=1.05)
        ax.set_ylabel("Safety rate")
        ax.set_title(f"{env_name} — Safety Rate", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([])

    # 2 rows x 3 columns, filled row-major in bar order.
    handles_all = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", linewidth=0.5)
                   for c in COLORS]
    labels_all = [label for label, _c in METHODS]
    order = [0, 3, 1, 4, 2, 5]
    fig.legend([handles_all[i] for i in order], [labels_all[i] for i in order],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.012),
               columnspacing=1.4, handletextpad=0.6)
    fig.suptitle("Test-time Total Reward and Safety Rate: RL-SGF (safe init) vs Safe-RL Baselines "
                 "and PSPO\n(mean ± s.e., n=10 seeds, 100 greedy episodes each)",
                 fontsize=12, y=0.998)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "rl_sgf_reward_safety.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "rl_sgf_reward_safety.png", bbox_inches="tight", dpi=300)
    (OUT_DIR / "rl_sgf_reward_safety_summary.json").write_text(json.dumps(summary, indent=1))
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
