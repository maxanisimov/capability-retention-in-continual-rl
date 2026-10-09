#!/usr/bin/env python
"""Compare PSPO with the no-safe-update ablation, PPO-Shield-Nominal."""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/vol/bitbucket/ma5923/_projects/CertifiedContinualLearning")
SCRIPT_DIR = REPO / "projects/safe_policy_optimisation/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_architecture_results as result_plots  # noqa: E402


OUT_DIR = REPO / "projects/safe_policy_optimisation/figures"
DOC_DIR = REPO / "projects/safe_policy_optimisation/docs/pspo_precomputed"
CSV_PATH = DOC_DIR / "pspo_safe_update_ablation_current.csv"
JSON_PATH = DOC_DIR / "pspo_safe_update_ablation_current.json"
MD_PATH = DOC_DIR / "pspo_safe_update_ablation_current.md"

METHODS = [
    ("ppo_shield/nominal", "PPO-Shield-Nominal", "lightblue"),
    ("rashomon_policy", "PSPO", "green"),
]
NOMINAL_CURVE = "ppo_shield/learning_curves/evaluation_unshielded_summary.csv"

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


@dataclass
class AblationRecord:
    architecture: str
    environment: str
    method: str
    total_reward_mean: float
    total_reward_sem: float
    safety_rate_mean: float
    safety_rate_sem: float
    success_rate_mean: float | None
    success_rate_sem: float | None
    n_seeds: int
    pspo_minus_nominal_reward: float
    pspo_minus_nominal_safety_rate: float
    pspo_minus_nominal_success_rate: float | None
    rashomon_n_iters: int | None
    bc_target_margin: float | None
    certification_method: str | None
    source: str


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def metric_to_tuple(metric: result_plots.Metric | None) -> tuple[float | None, float | None, int | None]:
    if metric is None:
        return None, None, None
    return metric.mean, metric.sem, metric.n


def load_nominal_metrics(run_dir: Path) -> tuple[result_plots.Metric, result_plots.Metric, result_plots.Metric | None]:
    reward = result_plots.load_baseline_metric(run_dir, "ppo_shield/nominal", "reward.mean_total_reward")
    safety = result_plots.load_baseline_metric(run_dir, "ppo_shield/nominal", "safety.safety_rate")
    success = result_plots.load_baseline_metric(run_dir, "ppo_shield/nominal", "success.success_rate")
    if reward is None or safety is None:
        raise ValueError(f"{run_dir} is missing PPO-Shield-Nominal reward or safety metrics")
    return reward, safety, success


def collect_records(selected_pspo: dict[str, dict[str, result_plots.PspoRecord]]) -> list[AblationRecord]:
    records: list[AblationRecord] = []

    for architecture, config in result_plots.ARCHITECTURES.items():
        for environment in result_plots.ENV_ORDER:
            if environment not in config["baseline_runs"]:
                continue
            pspo = selected_pspo.get(architecture, {}).get(environment)
            if pspo is None:
                continue

            nominal_reward, nominal_safety, nominal_success = load_nominal_metrics(
                config["baseline_runs"][environment]
            )
            pspo_success_mean, pspo_success_sem, _pspo_success_n = metric_to_tuple(pspo.success)
            nominal_success_mean, nominal_success_sem, _nominal_success_n = metric_to_tuple(nominal_success)
            success_delta = (
                pspo_success_mean - nominal_success_mean
                if pspo_success_mean is not None and nominal_success_mean is not None
                else None
            )
            reward_delta = pspo.reward.mean - nominal_reward.mean
            safety_delta = pspo.safety.mean - nominal_safety.mean

            records.append(
                AblationRecord(
                    architecture=architecture,
                    environment=environment,
                    method="PPO-Shield-Nominal",
                    total_reward_mean=nominal_reward.mean,
                    total_reward_sem=nominal_reward.sem,
                    safety_rate_mean=nominal_safety.mean,
                    safety_rate_sem=nominal_safety.sem,
                    success_rate_mean=nominal_success_mean,
                    success_rate_sem=nominal_success_sem,
                    n_seeds=nominal_reward.n,
                    pspo_minus_nominal_reward=reward_delta,
                    pspo_minus_nominal_safety_rate=safety_delta,
                    pspo_minus_nominal_success_rate=success_delta,
                    rashomon_n_iters=None,
                    bc_target_margin=None,
                    certification_method=None,
                    source=rel(config["baseline_runs"][environment]),
                )
            )
            records.append(
                AblationRecord(
                    architecture=architecture,
                    environment=environment,
                    method="PSPO",
                    total_reward_mean=pspo.reward.mean,
                    total_reward_sem=pspo.reward.sem,
                    safety_rate_mean=pspo.safety.mean,
                    safety_rate_sem=pspo.safety.sem,
                    success_rate_mean=pspo_success_mean,
                    success_rate_sem=pspo_success_sem,
                    n_seeds=pspo.reward.n,
                    pspo_minus_nominal_reward=reward_delta,
                    pspo_minus_nominal_safety_rate=safety_delta,
                    pspo_minus_nominal_success_rate=success_delta,
                    rashomon_n_iters=pspo.n_iters,
                    bc_target_margin=pspo.bc_target_margin,
                    certification_method=pspo.certification_method,
                    source=rel(pspo.source),
                )
            )

    return records


def write_tables(records: list[AblationRecord]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(AblationRecord.__dataclass_fields__)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    payload: dict[str, Any] = {
        "definitions": {
            "PPO-Shield-Nominal": (
                "PPO-Shield policy evaluated without the shield. This is used "
                "as the no-safe-update ablation for PSPO."
            ),
            "PSPO": "Best completed PSPO-precomputed result by total reward for each architecture/environment.",
            "deltas": "PSPO metric mean minus PPO-Shield-Nominal metric mean for the same architecture/environment.",
        },
        "records": [asdict(record) for record in records],
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bottom_with_slack(values: list[tuple[float, float]], slack_frac: float = 0.05) -> float:
    min_lo = min(mean - err for mean, err in values)
    magnitude = abs(min_lo) if abs(min_lo) > 1e-9 else 1.0
    return min_lo - slack_frac * magnitude


def plot_bars(architecture: str, records: list[AblationRecord]) -> None:
    config = result_plots.ARCHITECTURES[architecture]
    environments = [env for env in result_plots.ENV_ORDER if env in config["baseline_runs"]]
    fig, axes = plt.subplots(
        len(environments),
        2,
        figsize=(6.6, 1.22 * len(environments)),
        squeeze=False,
    )
    x = list(range(len(METHODS)))
    record_map = {
        (record.environment, record.method): record
        for record in records
        if record.architecture == architecture
    }

    for row, environment in enumerate(environments):
        reward_values: list[tuple[float | None, float]] = []
        safety_values: list[tuple[float | None, float]] = []
        for _method_key, label, _color in METHODS:
            record = record_map.get((environment, label))
            reward_values.append((record.total_reward_mean, record.total_reward_sem) if record else (None, 0.0))
            safety_values.append((record.safety_rate_mean, record.safety_rate_sem) if record else (None, 0.0))

        present_reward = [(mean, err) for mean, err in reward_values if mean is not None]
        present_safety = [(mean, err) for mean, err in safety_values if mean is not None]

        ax_r = axes[row][0]
        bottom_r = bottom_with_slack(present_reward)
        for index, (_method_key, label, color) in enumerate(METHODS):
            mean, err = reward_values[index]
            if mean is None:
                ax_r.text(index, bottom_r, "n/a", ha="center", va="bottom", rotation=90, fontsize=7)
                continue
            ax_r.bar(
                index,
                mean - bottom_r,
                bottom=bottom_r,
                yerr=err,
                capsize=3,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                error_kw={"linewidth": 1.0, "ecolor": "black"},
            )
        ax_r.set_ylim(bottom=bottom_r)
        ax_r.set_ylabel("Total reward", fontsize=8, labelpad=2)
        ax_r.set_title(f"Total Reward - {environment}", fontsize=8.5, pad=2)
        ax_r.set_xticks(x)
        ax_r.set_xticklabels([])
        ax_r.tick_params(axis="both", labelsize=7, pad=1)

        ax_s = axes[row][1]
        bottom_s = min(0.0, bottom_with_slack(present_safety))
        for index, (_method_key, label, color) in enumerate(METHODS):
            mean, err = safety_values[index]
            if mean is None:
                ax_s.text(index, bottom_s, "n/a", ha="center", va="bottom", rotation=90, fontsize=7)
                continue
            ax_s.bar(
                index,
                mean,
                yerr=err,
                capsize=3,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                error_kw={"linewidth": 1.0, "ecolor": "black"},
            )
        ax_s.axhline(1.0, color="grey", linestyle="--", linewidth=1.0, zorder=0)
        ax_s.set_ylim(bottom=bottom_s, top=1.0)
        ax_s.set_ylabel("Safety rate", fontsize=8, labelpad=2)
        ax_s.set_title(f"Safety Rate - {environment}", fontsize=8.5, pad=2)
        ax_s.set_xticks(x)
        ax_s.set_xticklabels([])
        ax_s.tick_params(axis="both", labelsize=7, pad=1)

    add_legend(fig)
    fig.tight_layout(rect=[0, 0.07, 1, 1], h_pad=0.35, w_pad=1.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = result_plots.ARCHITECTURES[architecture]["filename"]
    fig.savefig(OUT_DIR / f"{filename}_pspo_safe_update_ablation_reward_safety.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{filename}_pspo_safe_update_ablation_reward_safety.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_learning_curves(
    architecture: str,
    selected_pspo: dict[str, dict[str, result_plots.PspoRecord]],
) -> None:
    config = result_plots.ARCHITECTURES[architecture]
    environments = [env for env in result_plots.ENV_ORDER if env in config["baseline_runs"]]
    fig, axes = plt.subplots(
        len(environments),
        2,
        figsize=(6.6, 1.32 * len(environments)),
        squeeze=False,
    )

    for row, environment in enumerate(environments):
        ax_r = axes[row][0]
        ax_s = axes[row][1]
        pspo = selected_pspo.get(architecture, {}).get(environment)
        curve_specs = [
            ("PPO-Shield-Nominal", "lightblue", config["baseline_runs"][environment], NOMINAL_CURVE),
        ]
        if pspo is not None:
            curve_specs.append(("PSPO", "green", pspo.seed_root, pspo.curve_rel_path))

        for label, color, seed_root, rel_path in curve_specs:
            curve = result_plots.load_curve(seed_root, rel_path)
            if curve is None or curve.empty:
                continue
            n_seeds = int(curve["n"].max())
            curve_label = label if n_seeds > 1 else f"{label} (n=1)"
            ax_r.plot(curve["timestep"], curve["reward_mean"], color=color, linewidth=1.0, label=curve_label)
            ax_r.fill_between(
                curve["timestep"],
                curve["reward_mean"] - curve["reward_sem"],
                curve["reward_mean"] + curve["reward_sem"],
                color=color,
                alpha=0.15,
                linewidth=0,
            )
            ax_s.plot(curve["timestep"], curve["safety_mean"], color=color, linewidth=1.0, label=curve_label)
            ax_s.fill_between(
                curve["timestep"],
                curve["safety_mean"] - curve["safety_sem"],
                curve["safety_mean"] + curve["safety_sem"],
                color=color,
                alpha=0.15,
                linewidth=0,
            )

        ax_r.set_ylabel("Mean reward", fontsize=8, labelpad=2)
        ax_r.set_xlabel("Timestep", fontsize=8, labelpad=2)
        ax_r.set_title(f"Total Reward - {environment}", fontsize=8.5, pad=2)
        ax_r.tick_params(axis="both", labelsize=7, pad=1)
        ax_s.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, zorder=0)
        ax_s.set_ylim(top=1.0)
        ax_s.set_ylabel("Safety rate", fontsize=8, labelpad=2)
        ax_s.set_xlabel("Timestep", fontsize=8, labelpad=2)
        ax_s.set_title(f"Safety Rate - {environment}", fontsize=8.5, pad=2)
        ax_s.tick_params(axis="both", labelsize=7, pad=1)

    handles, labels = axes[0][0].get_legend_handles_labels()
    add_legend(fig, handles, labels)
    fig.tight_layout(rect=[0, 0.07, 1, 1], h_pad=0.35, w_pad=1.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = result_plots.ARCHITECTURES[architecture]["filename"]
    fig.savefig(OUT_DIR / f"{filename}_pspo_safe_update_ablation_learning_curves.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{filename}_pspo_safe_update_ablation_learning_curves.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def add_legend(
    fig: plt.Figure,
    handles: list[Any] | None = None,
    labels: list[str] | None = None,
) -> None:
    if handles is None or labels is None:
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5)
            for _key, _label, color in METHODS
        ]
        labels = [label for _key, label, _color in METHODS]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.012),
        columnspacing=1.4,
        handletextpad=0.6,
    )


def write_markdown(records: list[AblationRecord]) -> None:
    by_arch_env: dict[tuple[str, str], dict[str, AblationRecord]] = {}
    for record in records:
        by_arch_env.setdefault((record.architecture, record.environment), {})[record.method] = record

    lines = [
        "# PSPO Safe-Update Ablation",
        "",
        "Comparison: PSPO vs PPO-Shield-Nominal. PPO-Shield-Nominal is treated as the ablation where PSPO's safe policy updates are removed and the resulting policy is evaluated without the shield.",
        "",
    ]
    for architecture in result_plots.ARCHITECTURES:
        rows = [
            methods
            for (arch, _env), methods in by_arch_env.items()
            if arch == architecture and "PSPO" in methods and "PPO-Shield-Nominal" in methods
        ]
        if not rows:
            continue
        reward_improved = [methods["PSPO"].environment for methods in rows if methods["PSPO"].pspo_minus_nominal_reward > 0]
        safety_improved = [
            methods["PSPO"].environment
            for methods in rows
            if methods["PSPO"].pspo_minus_nominal_safety_rate > 0
        ]
        safety_matched = [
            methods["PSPO"].environment
            for methods in rows
            if abs(methods["PSPO"].pspo_minus_nominal_safety_rate) <= 1e-12
        ]
        competitive_nominal = [
            methods["PSPO"].environment
            for methods in rows
            if methods["PSPO"].pspo_minus_nominal_reward <= 0
            and methods["PSPO"].pspo_minus_nominal_safety_rate <= 0
        ]

        title = result_plots.ARCHITECTURES[architecture]["title"]
        lines.extend([
            f"## {title}",
            "",
            f"- PSPO improves total reward in {len(reward_improved)}/{len(rows)} environments: {', '.join(reward_improved) if reward_improved else 'none'}.",
            f"- PSPO improves safety rate in {len(safety_improved)}/{len(rows)} environments: {', '.join(safety_improved) if safety_improved else 'none'}.",
            f"- PSPO matches nominal safety in {len(safety_matched)}/{len(rows)} environments: {', '.join(safety_matched) if safety_matched else 'none'}.",
            f"- Nominal remains competitive or better on both reward and safety in {len(competitive_nominal)}/{len(rows)} environments: {', '.join(competitive_nominal) if competitive_nominal else 'none'}.",
            "",
            "| Environment | PSPO Reward | Nominal Reward | Delta Reward | PSPO Safety | Nominal Safety | Delta Safety |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for environment in result_plots.ENV_ORDER:
            methods = by_arch_env.get((architecture, environment))
            if not methods or "PSPO" not in methods or "PPO-Shield-Nominal" not in methods:
                continue
            pspo = methods["PSPO"]
            nominal = methods["PPO-Shield-Nominal"]
            lines.append(
                f"| {environment} | {pspo.total_reward_mean:.4g} | {nominal.total_reward_mean:.4g} | "
                f"{pspo.pspo_minus_nominal_reward:.4g} | {pspo.safety_rate_mean:.4g} | "
                f"{nominal.safety_rate_mean:.4g} | {pspo.pspo_minus_nominal_safety_rate:.4g} |"
            )
        lines.append("")

    MD_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    selected_pspo = result_plots.best_pspo_records()
    records = collect_records(selected_pspo)
    write_tables(records)
    write_markdown(records)
    for architecture in result_plots.ARCHITECTURES:
        plot_bars(architecture, records)
        plot_learning_curves(architecture, selected_pspo)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {MD_PATH}")
    print(f"Wrote PSPO safe-update ablation figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
