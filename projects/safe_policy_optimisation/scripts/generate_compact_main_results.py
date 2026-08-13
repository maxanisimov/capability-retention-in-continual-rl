#!/usr/bin/env python
"""Generate compact main-paper tables and a small tabular learning-curve figure."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/vol/bitbucket/ma5923/_projects/CertifiedContinualLearning")
SCRIPT_DIR = REPO / "projects/safe_policy_optimisation/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_architecture_results as result_plots  # noqa: E402


DOC_DIR = REPO / "projects/safe_policy_optimisation/docs/pspo_precomputed"
OUT_DIR = REPO / "projects/safe_policy_optimisation/figures"
MAIN_TABLE = DOC_DIR / "tabular_main_results_compact.tex"
ABLATION_TABLE = DOC_DIR / "tabular_safe_update_ablation_compact.tex"
SNIPPET = DOC_DIR / "tabular_compact_experiments_snippet.tex"
COMPACT_FIGURE = OUT_DIR / "tabular_compact_learning_curves.pdf"
COMPACT_FIGURE_PNG = OUT_DIR / "tabular_compact_learning_curves.png"
ABLATION_CSV = DOC_DIR / "pspo_safe_update_ablation_current.csv"

ENV_LABELS = {
    "Media Streaming": "Media",
    "Colour Bomb": "Colour v1",
    "Bridge Crossing v1": "Bridge v1",
    "Bridge Crossing v2": "Bridge v2",
    "Colour Bomb v2": "Colour v2",
    "MiniPacman": "MiniPacman",
}

METHOD_LABELS = {
    "ppo_policy": "PPO",
    "ppo_lagrangian/ppo_lagrangian": "PPO-Lag",
    "ppo_lagrangian/ppo_pid_lagrangian": "PID-Lag",
    "cpo/cpo": "CPO",
    "ppo_shield/shielded": "PPO-Shield",
}

COMPACT_CURVE_METHODS = [
    ("ppo_policy", "PPO", "grey"),
    ("ppo_shield/shielded", "PPO-Shield", "blue"),
    ("rashomon_policy", "PSPO", "green"),
]

plt.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


@dataclass
class MethodMetrics:
    label: str
    reward_mean: float
    reward_sem: float
    safety_mean: float
    safety_sem: float


def fmt_number(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def fmt_reward(mean: float, sem: float) -> str:
    return rf"{fmt_number(mean)}{{\scriptscriptstyle \pm {fmt_number(sem)}}}"


def fmt_safety(mean: float) -> str:
    return f"{mean:.2f}"


def fmt_rs(metric: MethodMetrics, *, include_label: bool = False) -> str:
    value = rf"${fmt_reward(metric.reward_mean, metric.reward_sem)} / {fmt_safety(metric.safety_mean)}$"
    if include_label:
        return rf"\shortstack[l]{{{metric.label}\\{value}}}"
    return value


def fmt_delta(value: float) -> str:
    sign = "+" if value > 0 else ""
    return rf"${sign}{fmt_number(value)}$"


def metric_from_baseline(run_dir: Path, method_key: str) -> MethodMetrics:
    reward = result_plots.load_baseline_metric(run_dir, method_key, "reward.mean_total_reward")
    safety = result_plots.load_baseline_metric(run_dir, method_key, "safety.safety_rate")
    if reward is None or safety is None:
        raise ValueError(f"Missing metrics for {method_key} in {run_dir}")
    return MethodMetrics(METHOD_LABELS[method_key], reward.mean, reward.sem, safety.mean, safety.sem)


def metric_from_pspo(record: result_plots.PspoRecord) -> MethodMetrics:
    return MethodMetrics("PSPO", record.reward.mean, record.reward.sem, record.safety.mean, record.safety.sem)


def best_safe_non_pspo(run_dir: Path) -> MethodMetrics:
    candidates: list[MethodMetrics] = []
    for method_key, _label, _color in result_plots.METHODS:
        if method_key == "rashomon_policy":
            continue
        metric = metric_from_baseline(run_dir, method_key)
        if metric.safety_mean >= 1.0 - 1e-12:
            candidates.append(metric)
    if not candidates:
        raise ValueError(f"No non-PSPO method with safety rate 1.0 in {run_dir}")
    return max(candidates, key=lambda metric: metric.reward_mean)


def write_main_table(selected: dict[str, dict[str, result_plots.PspoRecord]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Environment & PPO $R/S$ & Best non-PSPO safe $R/S$ & PSPO $R/S$ \\",
        r"\midrule",
    ]
    config = result_plots.ARCHITECTURES["tabular"]
    for environment in result_plots.ENV_ORDER:
        run_dir = config["baseline_runs"][environment]
        ppo = metric_from_baseline(run_dir, "ppo_policy")
        best_safe = best_safe_non_pspo(run_dir)
        pspo = metric_from_pspo(selected["tabular"][environment])
        lines.append(
            " & ".join([
                ENV_LABELS[environment],
                fmt_rs(ppo),
                fmt_rs(best_safe, include_label=True),
                fmt_rs(pspo),
            ])
            + r" \\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Compact tabular final performance. Each entry reports mean total reward / safety rate over $n=10$ seeds; reward includes standard error. The safe baseline column selects the highest-reward non-PSPO method with reported final safety rate $1.0$ and excludes PPO-Shield-Nominal.}",
        r"\label{tab:tabular-main-compact}",
        r"\end{table*}",
        "",
    ])
    MAIN_TABLE.write_text("\n".join(lines), encoding="utf-8")


def read_tabular_ablation_rows() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with ABLATION_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["architecture"] != "tabular":
                continue
            rows.setdefault(row["environment"], {})[row["method"]] = row
    return rows


def row_metric(row: dict[str, str], label: str) -> MethodMetrics:
    return MethodMetrics(
        label=label,
        reward_mean=float(row["total_reward_mean"]),
        reward_sem=float(row["total_reward_sem"]),
        safety_mean=float(row["safety_rate_mean"]),
        safety_sem=float(row["safety_rate_sem"]),
    )


def write_ablation_table() -> None:
    rows = read_tabular_ablation_rows()
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Environment & PPO-Shield-Nominal $R/S$ & PSPO $R/S$ & $\Delta R$ & $\Delta S$ \\",
        r"\midrule",
    ]
    for environment in result_plots.ENV_ORDER:
        nominal_row = rows[environment]["PPO-Shield-Nominal"]
        pspo_row = rows[environment]["PSPO"]
        nominal = row_metric(nominal_row, "PPO-Shield-Nominal")
        pspo = row_metric(pspo_row, "PSPO")
        lines.append(
            " & ".join([
                ENV_LABELS[environment],
                fmt_rs(nominal),
                fmt_rs(pspo),
                fmt_delta(float(pspo_row["pspo_minus_nominal_reward"])),
                fmt_delta(float(pspo_row["pspo_minus_nominal_safety_rate"])),
            ])
            + r" \\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Safe-update ablation in the tabular setting. PPO-Shield-Nominal evaluates the shield-trained PPO policy without applying the shield. Deltas are PSPO minus PPO-Shield-Nominal.}",
        r"\label{tab:tabular-safe-update-ablation}",
        r"\end{table*}",
        "",
    ])
    ABLATION_TABLE.write_text("\n".join(lines), encoding="utf-8")


def plot_compact_curves(selected: dict[str, dict[str, result_plots.PspoRecord]]) -> None:
    environments = ["Colour Bomb v2", "MiniPacman"]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 3.8), squeeze=False)
    config = result_plots.ARCHITECTURES["tabular"]

    for row, environment in enumerate(environments):
        ax_reward = axes[row][0]
        ax_safety = axes[row][1]
        for method_key, label, color in COMPACT_CURVE_METHODS:
            if method_key == "rashomon_policy":
                record = selected["tabular"][environment]
                seed_root = record.seed_root
                rel_path = record.curve_rel_path
            else:
                seed_root = config["baseline_runs"][environment]
                rel_path = result_plots.CURVE_FILES[method_key]
            curve = result_plots.load_curve(seed_root, rel_path)
            if curve is None or curve.empty:
                continue
            ax_reward.plot(curve["timestep"], curve["reward_mean"], color=color, linewidth=1.2, label=label)
            ax_reward.fill_between(
                curve["timestep"],
                curve["reward_mean"] - curve["reward_sem"],
                curve["reward_mean"] + curve["reward_sem"],
                color=color,
                alpha=0.14,
                linewidth=0,
            )
            ax_safety.plot(curve["timestep"], curve["safety_mean"], color=color, linewidth=1.2, label=label)
            ax_safety.fill_between(
                curve["timestep"],
                curve["safety_mean"] - curve["safety_sem"],
                curve["safety_mean"] + curve["safety_sem"],
                color=color,
                alpha=0.14,
                linewidth=0,
            )

        ax_reward.set_title(f"Total Reward - {environment}")
        ax_reward.set_ylabel("Reward")
        ax_reward.set_xlabel("Timestep")
        ax_safety.set_title(f"Safety Rate - {environment}")
        ax_safety.set_ylabel("Safety")
        ax_safety.set_xlabel("Timestep")
        ax_safety.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, zorder=0)
        ax_safety.set_ylim(top=1.0)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPACT_FIGURE, bbox_inches="tight")
    fig.savefig(COMPACT_FIGURE_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_snippet() -> None:
    lines = [
        r"% Compact replacement for the figure-heavy tabular Results block.",
        r"\input{projects/safe_policy_optimisation/docs/pspo_precomputed/tabular_main_results_compact}",
        r"\input{projects/safe_policy_optimisation/docs/pspo_precomputed/tabular_safe_update_ablation_compact}",
        "",
        r"\begin{figure*}[t]",
        r"    \centering",
        r"    \includegraphics[width=0.82\linewidth]{figures/tabular_compact_learning_curves.pdf}",
        r"    \caption{Representative tabular learning curves for Colour Bomb v2 and MiniPacman. Colour Bomb v2 shows the case where PSPO improves safe reward; MiniPacman shows the main tabular reward-cost case. Full learning curves for all environments and architectures are in the supplement.}",
        r"    \label{fig:tabular-compact-learning-curves}",
        r"\end{figure*}",
        "",
        r"\Cref{tab:tabular-main-compact} reports compact final metrics for the tabular experiments. PSPO reaches safety rate $1.0$ in all six environments. It matches the best safe return in Colour Bomb v1 and both Bridge Crossing tasks, improves over PPO-Shield in Media Streaming and Colour Bomb v2, and pays its clearest reward cost in MiniPacman. \Cref{tab:tabular-safe-update-ablation} isolates the safe-update mechanism: PPO-Shield-Nominal is often unsafe when evaluated without the shield, whereas PSPO remains safe under unshielded greedy evaluation.",
        "",
        r"\Cref{fig:tabular-compact-learning-curves} gives representative learning dynamics. Full bar charts, complete learning curves, action-safety plots, and one-hidden/two-hidden architecture results are reported in the supplement. For larger actor-critic architectures, PSPO remains safe but gives lower total reward in several environments because verification becomes more conservative.",
        "",
    ]
    SNIPPET.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    selected = result_plots.best_pspo_records()
    write_main_table(selected)
    write_ablation_table()
    plot_compact_curves(selected)
    write_snippet()
    print(f"Wrote {MAIN_TABLE}")
    print(f"Wrote {ABLATION_TABLE}")
    print(f"Wrote {SNIPPET}")
    print(f"Wrote {COMPACT_FIGURE}")
    print(f"Wrote {COMPACT_FIGURE_PNG}")


if __name__ == "__main__":
    main()
