#!/usr/bin/env python
"""Summarise completed PSPO-precomputed hyperparameter results.

The script reads aggregate metric files from ``outputs/`` and writes:

* a JSON file with the best PSPO-precomputed settings per environment and
  actor-critic architecture;
* a Markdown table with the best observed reward for each unique
  hyperparameter setting, again separated by architecture.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO / "projects" / "safe_policy_optimisation" / "docs" / "pspo_precomputed"
DEFAULT_AGGREGATE_ROOT = REPO / "outputs"

ENV_NAMES = {
    "paper_2503_07671_bridge_crossing": "Bridge Crossing v1",
    "paper_2503_07671_bridge_crossing_v2": "Bridge Crossing v2",
    "paper_2503_07671_colour_bomb": "Colour Bomb v1",
    "paper_2503_07671_colour_bomb_v2": "Colour Bomb v2",
    "paper_2503_07671_media_streaming": "Media Streaming",
    "paper_2503_07671_minipacman": "MiniPacman",
}


@dataclass(frozen=True)
class Metric:
    mean: float
    sem: float
    std: float
    n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-root", type=Path, default=DEFAULT_AGGREGATE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--best-json", default="pspo_precomputed_best_hyperparameters.json")
    parser.add_argument("--reward-csv", default="pspo_precomputed_hparam_rewards.csv")
    parser.add_argument("--reward-table", default="pspo_precomputed_hparam_rewards.md")
    parser.add_argument("--shield-gap-table", default="pspo_precomputed_below_ppo_shield.md")
    return parser.parse_args()


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


def metric_from_entry(entry: dict[str, Any] | None) -> Metric | None:
    if entry is None:
        return None
    n = int(entry.get("n") or 0)
    std = float(entry.get("std") or 0.0)
    sem = std / math.sqrt(n) if n > 0 else 0.0
    return Metric(mean=float(entry["mean"]), sem=sem, std=std, n=n)


def metric_entry(metrics: dict[str, Any], suffix: str, *, single_method: bool) -> dict[str, Any] | None:
    if single_method:
        return metrics.get(suffix)
    return (
        metrics.get(f"rashomon_policy.{suffix}")
        or metrics.get(f"rashomon_policy/rashomon_policy.{suffix}")
    )


def environment_from_path(path: Path) -> str | None:
    for part in path.parts:
        if part in ENV_NAMES:
            return ENV_NAMES[part]
    path_text = str(path)
    if "bridge_crossing_v1_1hidden_precomputed" in path_text:
        return "Bridge Crossing v1"
    if "bridge_crossing_v2_tabular_fixed" in path_text:
        return "Bridge Crossing v2"
    return None


def setting_from_path(path: Path) -> tuple[int | None, float | None]:
    match = re.search(r"iters_(\d+)__margin_([^/]+)", str(path))
    if not match:
        return None, None
    return int(match.group(1)), float(match.group(2).replace("p", "."))


def resolve_rashomon_dir(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def artifact_summary(rashomon_dir: str | None) -> dict[str, Any]:
    directory = resolve_rashomon_dir(rashomon_dir)
    if directory is None:
        return {}
    return load_json(directory / "summary.json") or {}


def architecture_name(n_hidden: int | None, net_arch: Any) -> str:
    if n_hidden == 0 or net_arch == []:
        return "tabular"
    if n_hidden == 1 or net_arch == [64]:
        return "one_hidden"
    if n_hidden == 2 or net_arch == [64, 64]:
        return "two_hidden"
    if n_hidden is None:
        return "unknown"
    return f"{n_hidden}_hidden"


def architecture_from_path(path: Path) -> str:
    path_text = str(path)
    if "_pspo_hparam/bridge_crossing_v1_1hidden_precomputed" in path_text:
        return "one_hidden"
    if "_sweeps_1hidden" in path_text:
        return "one_hidden"
    if "_sweeps_tabular" in path_text or "bridge_crossing_v2_tabular_fixed" in path_text:
        return "tabular"
    if "outputs/_sweeps/" in path_text:
        return "two_hidden"
    return "unknown"


def normal_sweep_config(aggregate_path: Path) -> dict[str, Any]:
    pipeline_dir = aggregate_path.parents[1]
    config = load_json(pipeline_dir / "seed0" / "rashomon_policy" / "config.json") or {}
    architecture = config.get("base_policy_architecture") or {}
    policy_kwargs = config.get("policy_kwargs") or {}
    training = config.get("training_hyperparameters") or {}
    env_kwargs = config.get("env_kwargs") or {}
    rashomon_dir = config.get("rashomon_dir")
    summary = artifact_summary(rashomon_dir)
    rashomon = summary.get("rashomon") or {}
    base_policy = summary.get("base_policy") or {}
    dataset = summary.get("dataset") or {}

    path_text = str(aggregate_path)
    inferred_iters, inferred_margin = None, None
    if "precomputed_10k_margin5" in path_text:
        inferred_iters, inferred_margin = 10000, 5.0
    elif "precomputed_10k" in path_text:
        inferred_iters = 10000
    elif "pspo30000_margin5" in path_text:
        inferred_iters, inferred_margin = 30000, 5.0

    n_hidden = architecture.get("n_hidden")
    net_arch = policy_kwargs.get("net_arch")
    state_representation = (
        env_kwargs.get("state_representation")
        or architecture.get("state_representation")
        or dataset.get("state_representation")
    )
    return {
        "actor_critic_architecture": architecture_name(n_hidden, net_arch),
        "n_hidden": n_hidden,
        "hidden_dim": architecture.get("hidden_dim"),
        "net_arch": net_arch,
        "input_dim": architecture.get("input_dim") or dataset.get("feature_dim"),
        "state_representation": state_representation,
        "rashomon_dir": rashomon_dir,
        "rashomon_n_iters": rashomon.get("n_iters", inferred_iters),
        "bc_target_margin": base_policy.get("target_margin", inferred_margin),
        "bc_final_min_margin": base_policy.get("final_min_margin"),
        "rashomon_batch_size": rashomon.get("batch_size"),
        "certificate_samples": rashomon.get("certificate_samples"),
        "checkpoint": rashomon.get("checkpoint"),
        "certification_method": rashomon.get("certification_method"),
        "selected_certificate": rashomon.get("selected_certificate"),
        "total_timesteps": config.get("total_timesteps"),
        "cost_limit": config.get("cost_limit"),
        "learning_rate": training.get("learning_rate"),
        "n_steps": training.get("n_steps"),
        "batch_size": training.get("batch_size"),
        "n_epochs": training.get("n_epochs"),
    }


def ppo_shield_record(aggregate_path: Path) -> dict[str, Any] | None:
    data = load_json(aggregate_path)
    if data is None:
        return None
    environment = environment_from_path(aggregate_path)
    if environment is None:
        return None
    metrics = data.get("metrics") or {}
    reward = metric_from_entry(
        metrics.get("ppo_shield/shielded.reward.mean_total_reward")
        or metrics.get("ppo_shield.reward.mean_total_reward")
    )
    if reward is None:
        return None

    config = load_json(aggregate_path.parents[1] / "seed0" / "ppo_shield" / "config.json") or {}
    architecture = config.get("base_policy_architecture") or {}
    policy_kwargs = config.get("policy_kwargs") or {}
    inferred_architecture = architecture_name(architecture.get("n_hidden"), policy_kwargs.get("net_arch"))
    if inferred_architecture == "unknown":
        inferred_architecture = architecture_from_path(aggregate_path)
    return {
        "environment": environment,
        "actor_critic_architecture": inferred_architecture,
        "reward_mean": reward.mean,
        "reward_sem": reward.sem,
        "reward_std": reward.std,
        "n_seeds": reward.n,
        "source_aggregate": rel(aggregate_path),
    }


def hparam_sweep_config(aggregate_path: Path) -> dict[str, Any]:
    summary = load_json(aggregate_path.parents[1] / "set" / "summary.json") or {}
    rashomon = summary.get("rashomon") or {}
    base_policy = summary.get("base_policy") or {}
    dataset = summary.get("dataset") or {}
    rashomon_n_iters, bc_target_margin = setting_from_path(aggregate_path)
    path_text = str(aggregate_path)
    n_hidden = 1 if "1hidden" in path_text else 0 if "tabular" in path_text else None
    net_arch = [64] if n_hidden == 1 else [] if n_hidden == 0 else None
    return {
        "actor_critic_architecture": architecture_name(n_hidden, net_arch),
        "n_hidden": n_hidden,
        "hidden_dim": 64 if n_hidden == 1 else None,
        "net_arch": net_arch,
        "input_dim": dataset.get("feature_dim"),
        "state_representation": dataset.get("state_representation"),
        "rashomon_dir": rel(aggregate_path.parents[1] / "set"),
        "rashomon_n_iters": rashomon.get("n_iters", rashomon_n_iters),
        "bc_target_margin": base_policy.get("target_margin", bc_target_margin),
        "bc_final_min_margin": base_policy.get("final_min_margin"),
        "rashomon_batch_size": rashomon.get("batch_size"),
        "certificate_samples": rashomon.get("certificate_samples"),
        "checkpoint": rashomon.get("checkpoint"),
        "certification_method": rashomon.get("certification_method"),
        "selected_certificate": rashomon.get("selected_certificate"),
        "total_timesteps": None,
        "cost_limit": None,
        "learning_rate": None,
        "n_steps": None,
        "batch_size": None,
        "n_epochs": None,
    }


def aggregate_record(aggregate_path: Path) -> dict[str, Any] | None:
    data = load_json(aggregate_path)
    if data is None:
        return None
    metrics = data.get("metrics") or {}
    environment = environment_from_path(aggregate_path)
    if environment is None:
        return None

    path_text = str(aggregate_path)
    single_method_precomputed = "_pspo_hparam" in path_text and "/precomputed/" in path_text
    single_method_adaptive = "_pspo_hparam" in path_text and "/adaptive/" in path_text
    if single_method_adaptive:
        return None

    reward = metric_from_entry(
        metric_entry(metrics, "reward.mean_total_reward", single_method=single_method_precomputed)
    )
    safety = metric_from_entry(
        metric_entry(metrics, "safety.safety_rate", single_method=single_method_precomputed)
    )
    success = metric_from_entry(
        metric_entry(metrics, "success.success_rate", single_method=single_method_precomputed)
    )
    if reward is None:
        return None

    config = hparam_sweep_config(aggregate_path) if single_method_precomputed else normal_sweep_config(aggregate_path)
    return {
        "environment": environment,
        **config,
        "reward_mean": reward.mean,
        "reward_sem": reward.sem,
        "reward_std": reward.std,
        "safety_rate_mean": safety.mean if safety else None,
        "safety_rate_sem": safety.sem if safety else None,
        "success_rate_mean": success.mean if success else None,
        "success_rate_sem": success.sem if success else None,
        "n_seeds": reward.n,
        "source_aggregate": rel(aggregate_path),
    }


def table_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["actor_critic_architecture"],
        record["environment"],
        record.get("n_hidden"),
        record.get("hidden_dim"),
        json.dumps(record.get("net_arch"), sort_keys=True),
        record.get("state_representation"),
        record.get("rashomon_n_iters"),
        record.get("bc_target_margin"),
        record.get("total_timesteps"),
        record.get("cost_limit"),
        record.get("learning_rate"),
        record.get("n_steps"),
        record.get("batch_size"),
        record.get("n_epochs"),
        record.get("rashomon_dir"),
    )


def sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(record["actor_critic_architecture"]),
        str(record["environment"]),
        -(record["reward_mean"] or 0),
        str(record.get("rashomon_n_iters")),
        str(record.get("bc_target_margin")),
        str(record.get("source_aggregate")),
    )


def compact_hyperparameters(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "n_hidden",
        "hidden_dim",
        "net_arch",
        "input_dim",
        "state_representation",
        "rashomon_n_iters",
        "bc_target_margin",
        "bc_final_min_margin",
        "rashomon_batch_size",
        "certificate_samples",
        "checkpoint",
        "certification_method",
        "selected_certificate",
        "total_timesteps",
        "cost_limit",
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "rashomon_dir",
    )
    return {key: record.get(key) for key in keys if record.get(key) is not None}


def build_best(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        arch = record["actor_critic_architecture"]
        env = record["environment"]
        grouped.setdefault(arch, {}).setdefault(env, []).append(record)

    out: dict[str, Any] = {
        "metric": "reward.mean_total_reward",
        "method": "PSPO precomputed",
        "selection": (
            "For each actor-critic architecture and environment, keep all settings "
            "tied for the maximum completed mean total reward."
        ),
        "architectures": {},
    }
    for arch, envs in sorted(grouped.items()):
        out["architectures"][arch] = {}
        for env, env_records in sorted(envs.items()):
            best_reward = max(record["reward_mean"] for record in env_records)
            best_records = [
                record for record in env_records if math.isclose(record["reward_mean"], best_reward)
            ]
            best_records.sort(key=lambda item: str(item["source_aggregate"]))
            first = best_records[0]
            out["architectures"][arch][env] = {
                "best_reward_mean": best_reward,
                "best_reward_sem": first["reward_sem"],
                "safety_rate_mean": first["safety_rate_mean"],
                "safety_rate_sem": first["safety_rate_sem"],
                "success_rate_mean": first["success_rate_mean"],
                "success_rate_sem": first["success_rate_sem"],
                "n_tied_settings": len(best_records),
                "best_settings": [
                    {
                        "hyperparameters": compact_hyperparameters(record),
                        "metrics": {
                            "reward_mean": record["reward_mean"],
                            "reward_sem": record["reward_sem"],
                            "reward_std": record["reward_std"],
                            "safety_rate_mean": record["safety_rate_mean"],
                            "safety_rate_sem": record["safety_rate_sem"],
                            "success_rate_mean": record["success_rate_mean"],
                            "success_rate_sem": record["success_rate_sem"],
                            "n_seeds": record["n_seeds"],
                        },
                        "source_aggregate": record["source_aggregate"],
                    }
                    for record in best_records
                ],
            }
    return out


def write_reward_table(path: Path, records: list[dict[str, Any]]) -> None:
    best_by_setting: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = table_key(record)
        current = best_by_setting.get(key)
        if current is None or (
            record["reward_mean"],
            record["safety_rate_mean"] or -1.0,
            record["n_seeds"],
        ) > (
            current["reward_mean"],
            current["safety_rate_mean"] or -1.0,
            current["n_seeds"],
        ):
            best_by_setting[key] = record

    fieldnames = [
        "actor_critic_architecture",
        "environment",
        "reward_mean",
        "reward_sem",
        "reward_std",
        "safety_rate_mean",
        "safety_rate_sem",
        "success_rate_mean",
        "success_rate_sem",
        "n_seeds",
        "n_hidden",
        "hidden_dim",
        "net_arch",
        "input_dim",
        "state_representation",
        "rashomon_n_iters",
        "bc_target_margin",
        "bc_final_min_margin",
        "rashomon_batch_size",
        "certificate_samples",
        "checkpoint",
        "certification_method",
        "selected_certificate",
        "total_timesteps",
        "cost_limit",
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "rashomon_dir",
        "source_aggregate",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(best_by_setting.values(), key=sort_key):
            row = {field: record.get(field) for field in fieldnames}
            row["net_arch"] = json.dumps(row["net_arch"])
            writer.writerow(row)


def markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return "`" + json.dumps(value) + "`"
    text = str(value)
    return text.replace("|", "\\|")


def write_reward_markdown(path: Path, records: list[dict[str, Any]]) -> None:
    best_by_setting: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = table_key(record)
        current = best_by_setting.get(key)
        if current is None or (
            record["reward_mean"],
            record["safety_rate_mean"] or -1.0,
            record["n_seeds"],
        ) > (
            current["reward_mean"],
            current["safety_rate_mean"] or -1.0,
            current["n_seeds"],
        ):
            best_by_setting[key] = record

    columns = [
        ("Architecture", "actor_critic_architecture"),
        ("Environment", "environment"),
        ("Reward", "reward_mean"),
        ("Reward SEM", "reward_sem"),
        ("Safety", "safety_rate_mean"),
        ("Success", "success_rate_mean"),
        ("n", "n_seeds"),
        ("n_hidden", "n_hidden"),
        ("hidden_dim", "hidden_dim"),
        ("net_arch", "net_arch"),
        ("state", "state_representation"),
        ("rashomon_n_iters", "rashomon_n_iters"),
        ("bc_target_margin", "bc_target_margin"),
        ("total_timesteps", "total_timesteps"),
        ("cost_limit", "cost_limit"),
        ("rashomon_dir", "rashomon_dir"),
        ("source", "source_aggregate"),
    ]
    lines = [
        "# PSPO Precomputed Hyperparameter Rewards",
        "",
        (
            "Best observed total reward for each unique PSPO-precomputed "
            "hyperparameter setting, grouped by actor-critic architecture."
        ),
        "",
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in sorted(best_by_setting.values(), key=sort_key):
        lines.append("| " + " | ".join(markdown_value(record.get(key)) for _, key in columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def best_record_by_environment_architecture(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (record["environment"], record["actor_critic_architecture"])
        current = best.get(key)
        if current is None or (
            record["reward_mean"],
            record.get("safety_rate_mean") or -1.0,
            record["n_seeds"],
            -len(record["source_aggregate"]),
        ) > (
            current["reward_mean"],
            current.get("safety_rate_mean") or -1.0,
            current["n_seeds"],
            -len(current["source_aggregate"]),
        ):
            best[key] = record
    return best


def write_shield_gap_markdown(
    path: Path,
    pspo_records: list[dict[str, Any]],
    shield_records: list[dict[str, Any]],
) -> None:
    best_pspo = best_record_by_environment_architecture(pspo_records)
    best_shield = best_record_by_environment_architecture(shield_records)
    rows = []
    for key in sorted(set(best_pspo) & set(best_shield)):
        pspo = best_pspo[key]
        shield = best_shield[key]
        if pspo["reward_mean"] < shield["reward_mean"]:
            rows.append(
                {
                    "environment": key[0],
                    "actor_critic_architecture": key[1],
                    "pspo_reward_mean": pspo["reward_mean"],
                    "pspo_reward_sem": pspo["reward_sem"],
                    "ppo_shield_reward_mean": shield["reward_mean"],
                    "ppo_shield_reward_sem": shield["reward_sem"],
                    "reward_gap": pspo["reward_mean"] - shield["reward_mean"],
                    "rashomon_n_iters": pspo.get("rashomon_n_iters"),
                    "bc_target_margin": pspo.get("bc_target_margin"),
                    "pspo_source_aggregate": pspo["source_aggregate"],
                    "ppo_shield_source_aggregate": shield["source_aggregate"],
                }
            )

    columns = [
        ("Environment", "environment"),
        ("Architecture", "actor_critic_architecture"),
        ("PSPO reward", "pspo_reward_mean"),
        ("PSPO SEM", "pspo_reward_sem"),
        ("PPO-Shield reward", "ppo_shield_reward_mean"),
        ("PPO-Shield SEM", "ppo_shield_reward_sem"),
        ("Gap", "reward_gap"),
        ("rashomon_n_iters", "rashomon_n_iters"),
        ("bc_target_margin", "bc_target_margin"),
        ("PSPO source", "pspo_source_aggregate"),
        ("PPO-Shield source", "ppo_shield_source_aggregate"),
    ]
    lines = [
        "# PSPO Precomputed Results Below PPO-Shield",
        "",
        (
            "Environment/architecture pairs where the best completed "
            "PSPO-precomputed total reward is below the best completed "
            "PPO-Shield total reward for the same inferred actor-critic architecture."
        ),
        "",
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_value(row.get(key)) for _, key in columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    aggregate_root = args.aggregate_root if args.aggregate_root.is_absolute() else REPO / args.aggregate_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    records = [
        record
        for path in aggregate_root.glob("**/aggregate/aggregated_metrics.json")
        if (record := aggregate_record(path)) is not None
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    best_path = output_dir / args.best_json
    table_path = output_dir / args.reward_csv
    markdown_path = output_dir / args.reward_table
    shield_gap_path = output_dir / args.shield_gap_table
    best_path.write_text(json.dumps(build_best(records), indent=2, sort_keys=True) + "\n")
    write_reward_table(table_path, records)
    write_reward_markdown(markdown_path, records)
    shield_records = [
        record
        for path in aggregate_root.glob("**/aggregate/aggregated_metrics.json")
        if (record := ppo_shield_record(path)) is not None
    ]
    write_shield_gap_markdown(shield_gap_path, records, shield_records)
    print(f"Wrote {rel(best_path)}")
    print(f"Wrote {rel(table_path)}")
    print(f"Wrote {rel(markdown_path)}")
    print(f"Wrote {rel(shield_gap_path)}")
    print(f"Summarised {len(records)} completed PSPO-precomputed aggregate(s).")


if __name__ == "__main__":
    main()
