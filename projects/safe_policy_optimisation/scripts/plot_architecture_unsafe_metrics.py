#!/usr/bin/env python
"""Plot exploration/evaluation safety metrics for architecture sweeps.

This script uses the same environment/method/result selection as
``plot_architecture_results.py``: baselines from the architecture-specific
sweeps and PSPO from the best completed PSPO-precomputed run for that
architecture/environment.
"""

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
UNSAFE_CSV_PATH = DOC_DIR / "unsafe_metrics_by_architecture_current.csv"
UNSAFE_JSON_PATH = DOC_DIR / "unsafe_metrics_by_architecture_current.json"
SAFETY_CSV_PATH = DOC_DIR / "safety_rates_by_architecture_current.csv"
SAFETY_JSON_PATH = DOC_DIR / "safety_rates_by_architecture_current.json"

ACTION_METRICS = [
    ("exploration_safe_action_percentage", "Exploration Safe Actions (%)"),
    ("evaluation_safe_action_percentage", "Evaluation Safe Actions (%)"),
]
SAFETY_METRICS = [
    ("exploration_safety_rate", "Exploration Safety Rate (%)"),
    ("evaluation_safety_rate", "Evaluation Safety Rate (%)"),
]

TRAINING_FILES = {
    "ppo_policy": ("ppo_policy/training_episodes.csv", "plain_ppo"),
    "ppo_lagrangian/ppo_lagrangian": ("ppo_lagrangian/training_episodes.csv", "ppo_lagrangian"),
    "ppo_lagrangian/ppo_pid_lagrangian": ("ppo_lagrangian/training_episodes.csv", "ppo_pid_lagrangian"),
    "cpo/cpo": ("cpo/training_episodes.csv", "cpo"),
    "ppo_shield/shielded": ("ppo_shield/training_episodes.csv", "shielded_ppo"),
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
class UnsafeMetricRecord:
    architecture: str
    environment: str
    method_key: str
    method_label: str
    n_seeds: int
    exploration_checked_actions: int
    exploration_unsafe_actions: int
    exploration_safe_actions: int
    exploration_unsafe_action_percentage: float
    exploration_safe_action_percentage: float
    n_eval_checkpoints: int
    evaluation_proposed_action_checks: int
    evaluation_unsafe_proposed_action_count: int
    evaluation_safe_proposed_action_count: int
    evaluation_unsafe_action_percentage: float
    evaluation_safe_action_percentage: float
    source_root: str
    exploration_rel_path: str
    evaluation_rel_path: str


@dataclass
class SafetyRateRecord:
    architecture: str
    environment: str
    method_key: str
    method_label: str
    n_exploration_seeds: int
    exploration_episodes: int
    exploration_safe_episodes: int
    exploration_safety_rate: float
    n_evaluation_seeds: int
    evaluation_episodes: int
    evaluation_safe_episodes: int
    evaluation_safety_rate: float
    source_root: str
    training_rel_path: str
    evaluation_rel_path: str


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def sem_from_binary_percentage(count: int, total: int) -> float:
    if total <= 1:
        return 0.0
    p = count / total
    return 100.0 * math.sqrt(p * (1.0 - p) / total)


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def replace_filename(path: str, filename: str) -> str:
    return str(Path(path).with_name(filename))


def pspo_training_rel_path(curve_rel_path: str) -> str:
    path = Path(curve_rel_path)
    if len(path.parts) >= 3 and path.parts[0] == "rashomon_policy":
        return "rashomon_policy/training_episodes.csv"
    return "training_episodes.csv"


def seed_name_from_path(path: Path) -> str | None:
    seed_dir = next((parent for parent in path.parents if parent.name.startswith("seed")), None)
    return seed_dir.name if seed_dir is not None else None


def last_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))

    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        buffer = b""
        while position > 0:
            read_size = min(8192, position)
            position -= read_size
            handle.seek(position)
            buffer = handle.read(read_size) + buffer
            lines = [line for line in buffer.splitlines() if line.strip()]
            if len(lines) >= 2:
                last_line = lines[-1].decode("utf-8")
                values = next(csv.reader([last_line]))
                return dict(zip(header, values))
    return {}


def collect_exploration_unsafe(seed_root: Path, exploration_rel_path: str) -> tuple[int, int, set[str]]:
    checked = 0
    unsafe = 0
    seed_names: set[str] = set()
    for curve_file in sorted(seed_root.glob(f"seed*/{exploration_rel_path}")):
        seed_name = seed_name_from_path(curve_file)
        if seed_name is not None:
            seed_names.add(seed_name)
        row = last_csv_row(curve_file)
        if {"cumulative_checked", "cumulative_unsafe"}.issubset(row):
            checked += int(float(row.get("cumulative_checked") or 0))
            unsafe += int(float(row.get("cumulative_unsafe") or 0))
            continue
        with curve_file.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for event in reader:
                checked += int(float(event.get("checked_this_step") or 0))
                unsafe += int(float(event.get("unsafe_this_step") or 0))
    return checked, unsafe, seed_names


def collect_evaluation_summary(seed_root: Path, evaluation_rel_path: str) -> tuple[int, int, int, int, set[str]]:
    proposed_checks = 0
    unsafe_proposed = 0
    eval_episodes = 0
    eval_safe_episodes = 0
    n_checkpoints = 0
    seed_names: set[str] = set()
    for curve_file in sorted(seed_root.glob(f"seed*/{evaluation_rel_path}")):
        seed_name = seed_name_from_path(curve_file)
        if seed_name is not None:
            seed_names.add(seed_name)
        with curve_file.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            required = {
                "proposed_action_checks",
                "unsafe_proposed_action_count",
                "episodes",
                "safe_trajectory_count",
            }
            if not required.issubset(fieldnames):
                raise ValueError(f"{curve_file} is missing required fields {sorted(required - fieldnames)}")
            for row in reader:
                proposed_checks += int(float(row.get("proposed_action_checks") or 0))
                unsafe_proposed += int(float(row.get("unsafe_proposed_action_count") or 0))
                eval_episodes += int(float(row.get("episodes") or 0))
                eval_safe_episodes += int(float(row.get("safe_trajectory_count") or 0))
                n_checkpoints += 1
    return proposed_checks, unsafe_proposed, eval_episodes, eval_safe_episodes, n_checkpoints, seed_names


def collect_training_safety(
    seed_root: Path,
    training_rel_path: str,
    algorithm: str | None,
) -> tuple[int, int, set[str]]:
    episodes = 0
    safe_episodes = 0
    seed_names: set[str] = set()
    for training_file in sorted(seed_root.glob(f"seed*/{training_rel_path}")):
        seed_name = seed_name_from_path(training_file)
        if seed_name is not None:
            seed_names.add(seed_name)
        with training_file.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if algorithm is not None and row.get("algorithm") != algorithm:
                    continue
                safe_value = parse_bool(row.get("safe_trajectory"))
                if safe_value is None:
                    violated = parse_bool(row.get("violated"))
                    safe_value = not violated if violated is not None else False
                episodes += 1
                safe_episodes += int(safe_value)
    return episodes, safe_episodes, seed_names


def method_paths(
    method_key: str,
    curve_rel_path: str,
) -> tuple[str, str, str, str | None]:
    exploration_rel_path = replace_filename(curve_rel_path, "exploration_unsafe_actions.csv")
    evaluation_rel_path = curve_rel_path
    if method_key == "rashomon_policy":
        training_rel_path = pspo_training_rel_path(curve_rel_path)
        algorithm = None
    else:
        training_rel_path, algorithm = TRAINING_FILES[method_key]
    return exploration_rel_path, evaluation_rel_path, training_rel_path, algorithm


def compute_records(
    *,
    architecture: str,
    environment: str,
    method_key: str,
    method_label: str,
    seed_root: Path,
    curve_rel_path: str,
) -> tuple[UnsafeMetricRecord | None, SafetyRateRecord | None]:
    exploration_rel_path, evaluation_rel_path, training_rel_path, algorithm = method_paths(
        method_key,
        curve_rel_path,
    )
    exploration_files = sorted(seed_root.glob(f"seed*/{exploration_rel_path}"))
    evaluation_files = sorted(seed_root.glob(f"seed*/{evaluation_rel_path}"))
    training_files = sorted(seed_root.glob(f"seed*/{training_rel_path}"))
    if not exploration_files and not evaluation_files and not training_files:
        return None, None

    exploration_checked, exploration_unsafe, exploration_unsafe_seeds = collect_exploration_unsafe(
        seed_root,
        exploration_rel_path,
    )
    (
        evaluation_checks,
        evaluation_unsafe,
        evaluation_episodes,
        evaluation_safe_episodes,
        n_eval_checkpoints,
        evaluation_seeds,
    ) = collect_evaluation_summary(seed_root, evaluation_rel_path)
    exploration_episodes, exploration_safe_episodes, exploration_safety_seeds = collect_training_safety(
        seed_root,
        training_rel_path,
        algorithm,
    )

    exploration_safe = exploration_checked - exploration_unsafe
    evaluation_safe = evaluation_checks - evaluation_unsafe
    unsafe_record = UnsafeMetricRecord(
        architecture=architecture,
        environment=environment,
        method_key=method_key,
        method_label=method_label,
        n_seeds=len(exploration_unsafe_seeds | evaluation_seeds),
        exploration_checked_actions=exploration_checked,
        exploration_unsafe_actions=exploration_unsafe,
        exploration_safe_actions=exploration_safe,
        exploration_unsafe_action_percentage=pct(exploration_unsafe, exploration_checked),
        exploration_safe_action_percentage=pct(exploration_safe, exploration_checked),
        n_eval_checkpoints=n_eval_checkpoints,
        evaluation_proposed_action_checks=evaluation_checks,
        evaluation_unsafe_proposed_action_count=evaluation_unsafe,
        evaluation_safe_proposed_action_count=evaluation_safe,
        evaluation_unsafe_action_percentage=pct(evaluation_unsafe, evaluation_checks),
        evaluation_safe_action_percentage=pct(evaluation_safe, evaluation_checks),
        source_root=rel(seed_root),
        exploration_rel_path=exploration_rel_path,
        evaluation_rel_path=evaluation_rel_path,
    )
    safety_record = SafetyRateRecord(
        architecture=architecture,
        environment=environment,
        method_key=method_key,
        method_label=method_label,
        n_exploration_seeds=len(exploration_safety_seeds),
        exploration_episodes=exploration_episodes,
        exploration_safe_episodes=exploration_safe_episodes,
        exploration_safety_rate=pct(exploration_safe_episodes, exploration_episodes),
        n_evaluation_seeds=len(evaluation_seeds),
        evaluation_episodes=evaluation_episodes,
        evaluation_safe_episodes=evaluation_safe_episodes,
        evaluation_safety_rate=pct(evaluation_safe_episodes, evaluation_episodes),
        source_root=rel(seed_root),
        training_rel_path=training_rel_path,
        evaluation_rel_path=evaluation_rel_path,
    )
    return unsafe_record, safety_record


def collect_records() -> tuple[list[UnsafeMetricRecord], list[SafetyRateRecord]]:
    selected_pspo = result_plots.best_pspo_records()
    unsafe_records: list[UnsafeMetricRecord] = []
    safety_records: list[SafetyRateRecord] = []
    for architecture, config in result_plots.ARCHITECTURES.items():
        for environment in result_plots.ENV_ORDER:
            if environment not in config["baseline_runs"]:
                continue
            for method_key, method_label, _color in result_plots.METHODS:
                if method_key == "rashomon_policy":
                    pspo_record = selected_pspo.get(architecture, {}).get(environment)
                    if pspo_record is None:
                        continue
                    seed_root = pspo_record.seed_root
                    curve_rel_path = pspo_record.curve_rel_path
                else:
                    seed_root = config["baseline_runs"][environment]
                    curve_rel_path = result_plots.CURVE_FILES[method_key]
                unsafe_record, safety_record = compute_records(
                    architecture=architecture,
                    environment=environment,
                    method_key=method_key,
                    method_label=method_label,
                    seed_root=seed_root,
                    curve_rel_path=curve_rel_path,
                )
                if unsafe_record is not None:
                    unsafe_records.append(unsafe_record)
                if safety_record is not None:
                    safety_records.append(safety_record)
    return unsafe_records, safety_records


def write_tables(
    unsafe_records: list[UnsafeMetricRecord],
    safety_records: list[SafetyRateRecord],
) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    with UNSAFE_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(UnsafeMetricRecord.__dataclass_fields__))
        writer.writeheader()
        for record in unsafe_records:
            writer.writerow(asdict(record))

    unsafe_payload: dict[str, Any] = {
        "definitions": {
            "exploration_unsafe_action_percentage": (
                "100 * sum(unsafe_this_step) / sum(checked_this_step), using "
                "exploration_unsafe_actions.csv over training/exploration."
            ),
            "exploration_safe_action_percentage": (
                "100 * (sum(checked_this_step) - sum(unsafe_this_step)) / "
                "sum(checked_this_step), using exploration_unsafe_actions.csv "
                "over training/exploration."
            ),
            "evaluation_unsafe_action_percentage": (
                "100 * sum(unsafe_proposed_action_count) / "
                "sum(proposed_action_checks), using periodic evaluation summary rows."
            ),
            "evaluation_safe_action_percentage": (
                "100 * (sum(proposed_action_checks) - "
                "sum(unsafe_proposed_action_count)) / sum(proposed_action_checks), "
                "using periodic evaluation summary rows."
            ),
        },
        "records": [asdict(record) for record in unsafe_records],
    }
    UNSAFE_JSON_PATH.write_text(json.dumps(unsafe_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with SAFETY_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SafetyRateRecord.__dataclass_fields__))
        writer.writeheader()
        for record in safety_records:
            writer.writerow(asdict(record))

    safety_payload: dict[str, Any] = {
        "definitions": {
            "exploration_safety_rate": (
                "100 * count(safe training episodes) / count(training episodes), "
                "using training_episodes.csv over the exploration/training run."
            ),
            "evaluation_safety_rate": (
                "100 * sum(safe_trajectory_count) / sum(episodes), using periodic "
                "evaluation summary rows rather than final post-training episodes."
            ),
        },
        "records": [asdict(record) for record in safety_records],
    }
    SAFETY_JSON_PATH.write_text(json.dumps(safety_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_architecture(
    *,
    architecture: str,
    records: list[Any],
    metrics: list[tuple[str, str]],
    output_suffix: str,
    error_fields: dict[str, tuple[str, str]],
) -> None:
    config = result_plots.ARCHITECTURES[architecture]
    environments = [env for env in result_plots.ENV_ORDER if env in config["baseline_runs"]]
    fig, axes = plt.subplots(
        len(environments),
        len(metrics),
        figsize=(7.2, 1.22 * len(environments)),
        squeeze=False,
    )
    methods = result_plots.METHODS
    x = list(range(len(methods)))
    record_map = {
        (record.environment, record.method_key): record
        for record in records
        if record.architecture == architecture
    }

    for row, environment in enumerate(environments):
        for col, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row][col]
            for index, (method_key, _label, color) in enumerate(methods):
                record = record_map.get((environment, method_key))
                if record is None:
                    ax.text(index, 0.0, "n/a", ha="center", va="bottom", rotation=90, fontsize=7)
                    continue
                numerator_field, denominator_field = error_fields[metric_key]
                numerator = int(getattr(record, numerator_field))
                denominator = int(getattr(record, denominator_field))
                ax.bar(
                    index,
                    float(getattr(record, metric_key)),
                    yerr=sem_from_binary_percentage(numerator, denominator),
                    capsize=3,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                    error_kw={"linewidth": 1.0, "ecolor": "black"},
                )

            ax.set_ylim(0.0, 100.0)
            ax.set_ylabel("Percentage", fontsize=8, labelpad=2)
            ax.set_title(f"{metric_label} - {environment}", fontsize=8.5, pad=2)
            ax.set_xticks(x)
            ax.set_xticklabels([])
            ax.tick_params(axis="both", labelsize=7, pad=1)

    result_plots.add_legend(fig, ncol=6)
    fig.tight_layout(rect=[0, 0.07, 1, 1], h_pad=0.35, w_pad=1.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = config["filename"]
    fig.savefig(OUT_DIR / f"{filename}_{output_suffix}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{filename}_{output_suffix}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    unsafe_records, safety_records = collect_records()
    write_tables(unsafe_records, safety_records)
    for architecture in result_plots.ARCHITECTURES:
        plot_architecture(
            architecture=architecture,
            records=unsafe_records,
            metrics=ACTION_METRICS,
            output_suffix="unsafe_metrics",
            error_fields={
                "exploration_safe_action_percentage": (
                    "exploration_safe_actions",
                    "exploration_checked_actions",
                ),
                "evaluation_safe_action_percentage": (
                    "evaluation_safe_proposed_action_count",
                    "evaluation_proposed_action_checks",
                ),
            },
        )
        plot_architecture(
            architecture=architecture,
            records=safety_records,
            metrics=SAFETY_METRICS,
            output_suffix="exploration_evaluation_safety_rate",
            error_fields={
                "exploration_safety_rate": (
                    "exploration_safe_episodes",
                    "exploration_episodes",
                ),
                "evaluation_safety_rate": (
                    "evaluation_safe_episodes",
                    "evaluation_episodes",
                ),
            },
        )
    print(f"Wrote {UNSAFE_CSV_PATH}")
    print(f"Wrote {UNSAFE_JSON_PATH}")
    print(f"Wrote {SAFETY_CSV_PATH}")
    print(f"Wrote {SAFETY_JSON_PATH}")
    print(f"Wrote safe-action and safety-rate figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
