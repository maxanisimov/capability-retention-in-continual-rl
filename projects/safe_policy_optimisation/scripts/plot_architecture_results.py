#!/usr/bin/env python
"""Regenerate architecture-specific result figures with best PSPO precomputed.

For each actor-critic architecture, this script:

* finds the best completed PSPO-precomputed result per environment by mean total
  reward;
* writes those selections to a JSON manifest;
* plots total-reward/safety bars and learning curves, using the selected PSPO
  run as the "PSPO" method.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO = Path("/vol/bitbucket/ma5923/_projects/CertifiedContinualLearning")
OUT_DIR = REPO / "projects/safe_policy_optimisation/figures"
DOC_DIR = REPO / "projects/safe_policy_optimisation/docs/pspo_precomputed"
BEST_JSON = DOC_DIR / "best_pspo_precomputed_by_architecture_current.json"

ENV_ORDER = [
    "Media Streaming",
    "Colour Bomb",
    "Bridge Crossing v1",
    "Bridge Crossing v2",
    "Colour Bomb v2",
    "MiniPacman",
]

ARCHITECTURES = {
    "tabular": {
        "title": "Tabular Actor-Critic, Index Encoding",
        "filename": "tabular",
        "baseline_runs": {
            "Media Streaming": REPO / "outputs/_sweeps_tabular/paper_2503_07671_media_streaming",
            "Colour Bomb": REPO / "outputs/_sweeps_tabular_colour_bomb_no_pspo/paper_2503_07671_colour_bomb",
            "Bridge Crossing v1": REPO / "outputs/_sweeps_tabular/paper_2503_07671_bridge_crossing",
            "Bridge Crossing v2": REPO / "outputs/_sweeps_tabular/paper_2503_07671_bridge_crossing_v2",
            "Colour Bomb v2": REPO / "outputs/_sweeps_tabular/paper_2503_07671_colour_bomb_v2",
            "MiniPacman": REPO / "outputs/_sweeps_tabular/paper_2503_07671_minipacman",
        },
    },
    "one_hidden": {
        "title": "One-Hidden-Layer Actor-Critic, Index Encoding",
        "filename": "one_hidden",
        "baseline_runs": {
            "Bridge Crossing v1": REPO / "outputs/_sweeps_1hidden_bridge_crossing_v1_baselines_only/paper_2503_07671_bridge_crossing",
            "Bridge Crossing v2": REPO / "outputs/_sweeps_1hidden_missing_all_methods_pspo30000_margin5_no_adaptive/paper_2503_07671_bridge_crossing_v2",
            "Colour Bomb": REPO / "outputs/_sweeps_1hidden_missing_all_methods_pspo30000_margin5_no_adaptive/paper_2503_07671_colour_bomb",
            "Colour Bomb v2": REPO / "outputs/_sweeps_1hidden/paper_2503_07671_colour_bomb_v2",
            "Media Streaming": REPO / "outputs/_sweeps_1hidden_media_streaming_baselines_only/paper_2503_07671_media_streaming",
            "MiniPacman": REPO / "outputs/_sweeps_1hidden_missing_all_methods_pspo30000_margin5_no_adaptive/paper_2503_07671_minipacman",
        },
    },
    "two_hidden": {
        "title": "Two-Hidden-Layer Actor-Critic, Index Encoding",
        "filename": "two_hidden",
        "baseline_runs": {
            "Media Streaming": REPO / "outputs/_sweeps/20260723_204829/paper_2503_07671_media_streaming",
            "Colour Bomb": REPO / "outputs/_sweeps/20260723_215050/paper_2503_07671_colour_bomb",
            "Bridge Crossing v1": REPO / "outputs/_sweeps/20260724_124311/paper_2503_07671_bridge_crossing",
            "Bridge Crossing v2": REPO / "outputs/_sweeps/20260724_152054/paper_2503_07671_bridge_crossing_v2",
            "Colour Bomb v2": REPO / "outputs/_sweeps/20260724_164416/paper_2503_07671_colour_bomb_v2",
        },
    },
}

METHODS = [
    ("ppo_policy", "PPO", "grey"),
    ("ppo_lagrangian/ppo_lagrangian", "PPO-Lagrangian", "red"),
    ("ppo_lagrangian/ppo_pid_lagrangian", "PPO-PID-Lagrangian", "orange"),
    ("cpo/cpo", "CPO", "yellow"),
    ("ppo_shield/shielded", "PPO-Shield", "blue"),
    ("rashomon_policy", "PSPO", "green"),
]

CURVE_FILES = {
    "ppo_policy": "ppo_policy/learning_curves/evaluation_unshielded_summary.csv",
    "ppo_lagrangian/ppo_lagrangian": "ppo_lagrangian/learning_curves/ppo_lagrangian/evaluation_unshielded_summary.csv",
    "ppo_lagrangian/ppo_pid_lagrangian": "ppo_lagrangian/learning_curves/ppo_pid_lagrangian/evaluation_unshielded_summary.csv",
    "cpo/cpo": "cpo/learning_curves/cpo/evaluation_unshielded_summary.csv",
    "ppo_shield/shielded": "ppo_shield/learning_curves/evaluation_shielded_summary.csv",
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


@dataclass
class Metric:
    mean: float
    std: float
    n: int

    @property
    def sem(self) -> float:
        return self.std / math.sqrt(self.n) if self.n > 1 else 0.0


@dataclass
class PspoRecord:
    environment: str
    architecture: str
    reward: Metric
    safety: Metric
    success: Metric | None
    n_iters: int | None
    bc_target_margin: float | None
    certification_method: str
    source: Path
    seed_root: Path
    curve_rel_path: str
    hyperparameters: dict[str, Any]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def metric_from_aggregate(entry: dict[str, Any] | None) -> Metric | None:
    if entry is None:
        return None
    return Metric(float(entry["mean"]), float(entry.get("std") or 0.0), int(entry.get("n") or 0))


def env_name(env_id: str | None, path: Path | str) -> str:
    text = f"{env_id or ''} {path}".lower()
    if "bridge_crossing_v2" in text or "bridgecrossingv2" in text:
        return "Bridge Crossing v2"
    if "bridge_crossing" in text or "bridgecrossing" in text:
        return "Bridge Crossing v1"
    if "colour_bomb_v2" in text or "colourbombgridworldv3" in text:
        return "Colour Bomb v2"
    if "colour_bomb" in text or "colourbombgridworld" in text:
        return "Colour Bomb"
    if "media_streaming" in text or "mediastreaming" in text:
        return "Media Streaming"
    if "mini_pacman" in text or "minipacman" in text:
        return "MiniPacman"
    return env_id or "Unknown"


def arch_name(n_hidden: Any, path: Path | str) -> str:
    if n_hidden is None:
        text = str(path).lower()
        if "tabular" in text:
            n_hidden = 0
        elif "1hidden" in text or "one_hidden" in text:
            n_hidden = 1
        elif "2hidden" in text or "two_hidden" in text or "outputs/_sweeps/202607" in text:
            n_hidden = 2
    try:
        n_hidden = int(n_hidden)
    except (TypeError, ValueError):
        return "unknown"
    return {0: "tabular", 1: "one_hidden", 2: "two_hidden"}.get(n_hidden, f"{n_hidden}_hidden")


def setting_from_text(text: str) -> tuple[int | None, float | None]:
    match = re.search(r"iters_(\d+)__margin_([0-9p]+)", text)
    if match:
        return int(match.group(1)), float(match.group(2).replace("p", "."))
    n_iters = None
    margin = None
    match = re.search(r"iters(\d+)", text)
    if match:
        n_iters = int(match.group(1))
    match = re.search(r"margin([0-9]+p?[0-9]*)", text)
    if match:
        margin = float(match.group(1).replace("p", "."))
    return n_iters, margin


def resolve_repo_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def metric_key(metrics: dict[str, Any], method_key: str, suffix: str) -> dict[str, Any] | None:
    candidates = [
        f"{method_key}.{suffix}",
        f"{method_key}/{method_key}.{suffix}",
        f"{method_key}/{suffix}",
    ]
    for key in candidates:
        if key in metrics:
            return metrics[key]
    method_prefix = f"{method_key}/"
    for key, value in metrics.items():
        if key.startswith(method_prefix) and key.endswith(suffix):
            return value
    return None


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def hparam_records() -> list[PspoRecord]:
    records: list[PspoRecord] = []
    for precomputed_dir in (REPO / "outputs/_pspo_hparam").glob("**/precomputed"):
        for setting_dir in precomputed_dir.iterdir():
            if not setting_dir.is_dir() or not setting_dir.name.startswith("iters_"):
                continue
            metric_files = sorted((setting_dir / "runs").glob("seed*/metrics.json"))
            if len(metric_files) < 10:
                continue

            rewards: list[float] = []
            safeties: list[float] = []
            successes: list[float] = []
            config: dict[str, Any] | None = None
            for metric_file in metric_files:
                data = load_json(metric_file)
                if data is None:
                    continue
                rewards.append(float(data["reward"]["mean_total_reward"]))
                safeties.append(float(data["safety"]["safety_rate"]))
                successes.append(float(data["success"]["success_rate"]))
                if config is None:
                    config = load_json(metric_file.parent / "config.json")
            if len(rewards) < 10:
                continue

            reward_mean, reward_std = mean_std(rewards)
            safety_mean, safety_std = mean_std(safeties)
            success_mean, success_std = mean_std(successes)
            n_iters, margin = setting_from_text(setting_dir.name)
            summary = load_json(setting_dir / "set" / "summary.json") or {}
            base_policy = summary.get("base_policy") or {}
            rashomon = summary.get("rashomon") or {}
            dataset = summary.get("dataset") or {}
            config = config or {}
            cert = (
                config.get("certification_method")
                or config.get("growth_method")
                or rashomon.get("certification_method")
                or ("CROWN" if "crown" in str(setting_dir).lower() else "IBP")
            )
            hp = {
                "n_hidden": config.get("n_hidden"),
                "hidden_dim": config.get("hidden_dim"),
                "state_representation": config.get("state_representation") or dataset.get("state_representation"),
                "rashomon_n_iters": rashomon.get("n_iters", n_iters),
                "bc_target_margin": base_policy.get("target_margin", margin),
                "bc_final_min_margin": base_policy.get("final_min_margin"),
                "rashomon_batch_size": rashomon.get("batch_size"),
                "certificate_samples": rashomon.get("certificate_samples"),
                "checkpoint": rashomon.get("checkpoint"),
                "certification_method": cert,
                "selected_certificate": rashomon.get("selected_certificate"),
                "rashomon_dir": rel(setting_dir / "set"),
                "source": rel(setting_dir),
            }
            records.append(PspoRecord(
                environment=env_name(config.get("env_id"), setting_dir),
                architecture=arch_name(config.get("n_hidden"), setting_dir),
                reward=Metric(reward_mean, reward_std, len(rewards)),
                safety=Metric(safety_mean, safety_std, len(safeties)),
                success=Metric(success_mean, success_std, len(successes)),
                n_iters=hp["rashomon_n_iters"],
                bc_target_margin=hp["bc_target_margin"],
                certification_method=cert,
                source=setting_dir,
                seed_root=setting_dir / "runs",
                curve_rel_path="learning_curves/evaluation_unshielded_summary.csv",
                hyperparameters={key: value for key, value in hp.items() if value is not None},
            ))
    return records


def aggregate_records() -> list[PspoRecord]:
    records: list[PspoRecord] = []
    for aggregate_path in (REPO / "outputs").glob("**/aggregate/aggregated_metrics.json"):
        if "_pspo_hparam" in str(aggregate_path):
            continue
        data = load_json(aggregate_path)
        if data is None:
            continue
        metrics = data.get("metrics") or {}
        reward = metric_from_aggregate(metric_key(metrics, "rashomon_policy", "reward.mean_total_reward"))
        safety = metric_from_aggregate(metric_key(metrics, "rashomon_policy", "safety.safety_rate"))
        success = metric_from_aggregate(metric_key(metrics, "rashomon_policy", "success.success_rate"))
        if reward is None or safety is None or reward.n < 10:
            continue

        pipeline_dir = aggregate_path.parents[1]
        config = load_json(pipeline_dir / "seed0" / "rashomon_policy" / "config.json") or {}
        if "config" in config:
            config = config["config"]
        architecture_config = config.get("base_policy_architecture") or {}
        policy_kwargs = config.get("policy_kwargs") or {}
        rashomon_dir = str(config.get("rashomon_dir") or "")
        rashomon_summary = load_json((resolve_repo_path(rashomon_dir) or Path("")) / "summary.json") or {}
        rashomon_info = rashomon_summary.get("rashomon") or {}
        base_policy_info = rashomon_summary.get("base_policy") or {}
        dataset_info = rashomon_summary.get("dataset") or {}
        inferred_iters, inferred_margin = setting_from_text(f"{aggregate_path} {rashomon_dir}")
        n_iters = config.get("rashomon_n_iters") or rashomon_info.get("n_iters") or inferred_iters
        margin = config.get("bc_target_margin")
        if margin is None:
            margin = base_policy_info.get("target_margin", inferred_margin)
        cert = (
            config.get("certification_method")
            or config.get("growth_method")
            or rashomon_info.get("certification_method")
            or ("CROWN" if "crown" in f"{aggregate_path} {rashomon_dir}".lower() else "IBP")
        )
        hp = {
            "n_hidden": config.get("n_hidden") or architecture_config.get("n_hidden"),
            "hidden_dim": config.get("hidden_dim") or architecture_config.get("hidden_dim"),
            "net_arch": policy_kwargs.get("net_arch"),
            "input_dim": architecture_config.get("input_dim") or dataset_info.get("feature_dim"),
            "state_representation": (
                config.get("state_representation")
                or architecture_config.get("state_representation")
                or dataset_info.get("state_representation")
            ),
            "rashomon_n_iters": n_iters,
            "bc_target_margin": margin,
            "bc_final_min_margin": base_policy_info.get("final_min_margin"),
            "rashomon_batch_size": rashomon_info.get("batch_size"),
            "certificate_samples": rashomon_info.get("certificate_samples"),
            "checkpoint": rashomon_info.get("checkpoint"),
            "certification_method": cert,
            "selected_certificate": rashomon_info.get("selected_certificate"),
            "rashomon_dir": rashomon_dir or None,
            "source_aggregate": rel(aggregate_path),
        }
        records.append(PspoRecord(
            environment=env_name(config.get("env_id"), aggregate_path),
            architecture=arch_name(config.get("n_hidden"), aggregate_path),
            reward=reward,
            safety=safety,
            success=success,
            n_iters=n_iters,
            bc_target_margin=margin,
            certification_method=cert,
            source=aggregate_path,
            seed_root=pipeline_dir,
            curve_rel_path="rashomon_policy/learning_curves/evaluation_unshielded_summary.csv",
            hyperparameters={key: value for key, value in hp.items() if value is not None},
        ))
    return records


def best_pspo_records() -> dict[str, dict[str, PspoRecord]]:
    grouped: dict[tuple[str, str], list[PspoRecord]] = defaultdict(list)
    for record in hparam_records() + aggregate_records():
        if record.environment == "Unknown" or record.architecture == "unknown":
            continue
        grouped[(record.architecture, record.environment)].append(record)

    selected: dict[str, dict[str, PspoRecord]] = defaultdict(dict)
    for (architecture, environment), records in grouped.items():
        records.sort(
            key=lambda item: (
                item.reward.mean,
                item.safety.mean,
                item.reward.n,
                1 if item.n_iters is not None else 0,
                1 if item.bc_target_margin is not None else 0,
                1 if "_pspo_hparam" in str(item.source) else 0,
                -len(str(item.source)),
            ),
            reverse=True,
        )
        selected[architecture][environment] = records[0]
    return selected


def write_best_manifest(selected: dict[str, dict[str, PspoRecord]]) -> None:
    manifest: dict[str, Any] = {
        "metric": "reward.mean_total_reward",
        "selection": "Best completed PSPO-precomputed result per environment and actor-critic architecture.",
        "minimum_completed_seeds": 10,
        "architectures": {},
    }
    for architecture in ("tabular", "one_hidden", "two_hidden"):
        manifest["architectures"][architecture] = {}
        for environment in ENV_ORDER:
            record = selected.get(architecture, {}).get(environment)
            if record is None:
                manifest["architectures"][architecture][environment] = None
                continue
            manifest["architectures"][architecture][environment] = {
                "metrics": {
                    "reward_mean": record.reward.mean,
                    "reward_std": record.reward.std,
                    "reward_sem": record.reward.sem,
                    "safety_rate_mean": record.safety.mean,
                    "safety_rate_std": record.safety.std,
                    "safety_rate_sem": record.safety.sem,
                    "success_rate_mean": record.success.mean if record.success else None,
                    "success_rate_std": record.success.std if record.success else None,
                    "success_rate_sem": record.success.sem if record.success else None,
                    "n_seeds": record.reward.n,
                },
                "hyperparameters": record.hyperparameters,
                "source": rel(record.source),
                "seed_root": rel(record.seed_root),
                "curve_rel_path": record.curve_rel_path,
            }
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    BEST_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_baseline_metric(run_dir: Path, method_key: str, suffix: str) -> Metric | None:
    aggregate_path = run_dir / "aggregate/aggregated_metrics.json"
    data = load_json(aggregate_path)
    if data is None:
        return None
    return metric_from_aggregate(metric_key(data.get("metrics") or {}, method_key, suffix))


def bottom_with_slack(values: list[tuple[float, float]], slack_frac: float = 0.05) -> float:
    min_lo = min(mean - err for mean, err in values)
    magnitude = abs(min_lo) if abs(min_lo) > 1e-9 else 1.0
    return min_lo - slack_frac * magnitude


def plot_bars(architecture: str, selected: dict[str, dict[str, PspoRecord]]) -> None:
    config = ARCHITECTURES[architecture]
    envs = [env for env in ENV_ORDER if env in config["baseline_runs"]]
    fig, axes = plt.subplots(
        len(envs),
        2,
        figsize=(7.2, 1.22 * len(envs)),
        squeeze=False,
    )
    x = list(range(len(METHODS)))

    for row, environment in enumerate(envs):
        baseline_run = config["baseline_runs"][environment]
        reward_values: list[tuple[float | None, float]] = []
        safety_values: list[tuple[float | None, float]] = []
        for method_key, _label, _color in METHODS:
            if method_key == "rashomon_policy":
                record = selected.get(architecture, {}).get(environment)
                reward = record.reward if record else None
                safety = record.safety if record else None
            else:
                reward = load_baseline_metric(baseline_run, method_key, "reward.mean_total_reward")
                safety = load_baseline_metric(baseline_run, method_key, "safety.safety_rate")
            reward_values.append((reward.mean, reward.sem) if reward else (None, 0.0))
            safety_values.append((safety.mean, safety.sem) if safety else (None, 0.0))

        present_reward = [(mean, err) for mean, err in reward_values if mean is not None]
        present_safety = [(mean, err) for mean, err in safety_values if mean is not None]

        ax_r = axes[row][0]
        bottom_r = bottom_with_slack(present_reward)
        for index, (_method_key, _label, color) in enumerate(METHODS):
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
        bottom_s = bottom_with_slack(present_safety)
        for index, (_method_key, _label, color) in enumerate(METHODS):
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

    add_legend(fig, ncol=6)
    fig.tight_layout(rect=[0, 0.07, 1, 1], h_pad=0.35, w_pad=1.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{config['filename']}_reward_safety.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{config['filename']}_reward_safety.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def load_curve(seed_root: Path, rel_path: str) -> pd.DataFrame | None:
    if not seed_root.exists():
        return None
    seed_dirs = sorted(path for path in seed_root.iterdir() if path.is_dir() and path.name.startswith("seed"))
    frames = []
    for seed_dir in seed_dirs:
        csv_path = seed_dir / rel_path
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, usecols=["eval_index", "timestep", "mean_total_reward", "safety_rate"])
        frame["seed"] = seed_dir.name
        frames.append(frame)
    if not frames:
        return None
    all_frames = pd.concat(frames, ignore_index=True)
    grouped = all_frames.groupby("eval_index").agg(
        timestep=("timestep", "first"),
        reward_mean=("mean_total_reward", "mean"),
        reward_sem=("mean_total_reward", "sem"),
        safety_mean=("safety_rate", "mean"),
        safety_sem=("safety_rate", "sem"),
        n=("seed", "nunique"),
    ).reset_index()
    grouped[["reward_sem", "safety_sem"]] = grouped[["reward_sem", "safety_sem"]].fillna(0.0)
    return grouped


def plot_learning_curves(architecture: str, selected: dict[str, dict[str, PspoRecord]]) -> None:
    config = ARCHITECTURES[architecture]
    envs = [env for env in ENV_ORDER if env in config["baseline_runs"]]
    fig, axes = plt.subplots(
        len(envs),
        2,
        figsize=(7.2, 1.32 * len(envs)),
        squeeze=False,
    )

    for row, environment in enumerate(envs):
        ax_r, ax_s = axes[row][0], axes[row][1]
        baseline_run = config["baseline_runs"][environment]
        pspo_plotted = False
        for method_key, label, color in METHODS:
            if method_key == "rashomon_policy":
                record = selected.get(architecture, {}).get(environment)
                if record is None:
                    continue
                seed_root = record.seed_root
                rel_path = record.curve_rel_path
            else:
                seed_root = baseline_run
                rel_path = CURVE_FILES[method_key]

            curve = load_curve(seed_root, rel_path)
            if curve is None or curve.empty:
                continue
            if method_key == "rashomon_policy":
                pspo_plotted = True
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

        if not pspo_plotted:
            ax_r.text(0.99, 0.05, "PSPO n/a", transform=ax_r.transAxes, ha="right", va="bottom", fontsize=7)
            ax_s.text(0.99, 0.05, "PSPO n/a", transform=ax_s.transAxes, ha="right", va="bottom", fontsize=7)
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
    for row in range(len(envs)):
        row_handles, row_labels = axes[row][0].get_legend_handles_labels()
        if len(row_labels) >= len(labels):
            handles, labels = row_handles, row_labels
    add_legend(fig, handles, labels, ncol=6)
    fig.tight_layout(rect=[0, 0.07, 1, 1], h_pad=0.35, w_pad=1.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{config['filename']}_learning_curves.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{config['filename']}_learning_curves.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def add_legend(
    fig: plt.Figure,
    handles: list[Any] | None = None,
    labels: list[str] | None = None,
    ncol: int | None = None,
) -> None:
    if handles is None or labels is None:
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5)
            for _key, _label, color in METHODS
        ]
        labels = [label for _key, label, _color in METHODS]
    if len(labels) == len(METHODS):
        order = [0, 3, 1, 4, 2, 5]
        handles = [handles[index] for index in order]
        labels = [labels[index] for index in order]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol or (3 if len(labels) == 6 else 4),
        frameon=False,
        bbox_to_anchor=(0.5, -0.012),
        columnspacing=1.4,
        handletextpad=0.6,
    )


def main() -> None:
    selected = best_pspo_records()
    write_best_manifest(selected)
    for architecture in ("tabular", "one_hidden", "two_hidden"):
        plot_bars(architecture, selected)
        plot_learning_curves(architecture, selected)
    print(f"Wrote {BEST_JSON}")
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
