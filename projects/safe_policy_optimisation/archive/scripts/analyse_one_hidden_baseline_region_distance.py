#!/usr/bin/env python
"""Compare safer/higher-reward one-hidden baselines to PSPO safe regions."""

from __future__ import annotations

import csv
import json
import math
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


REPO = Path("/vol/bitbucket/ma5923/_projects/CertifiedContinualLearning")
DOC_DIR = REPO / "projects/safe_policy_optimisation/docs/pspo_precomputed"
MANIFEST = DOC_DIR / "best_pspo_precomputed_by_architecture_current.json"
OUT_CSV = DOC_DIR / "one_hidden_baseline_region_distance.csv"
OUT_JSON = DOC_DIR / "one_hidden_baseline_region_distance.json"

SAFETY_TOL = 1e-9
REWARD_TOL = 1e-12

BASELINE_METHODS = [
    ("ppo_policy", "PPO"),
    ("ppo_lagrangian/ppo_lagrangian", "PPO-Lagrangian"),
    ("ppo_lagrangian/ppo_pid_lagrangian", "PPO-PID-Lagrangian"),
    ("cpo/cpo", "CPO"),
    ("ppo_shield/shielded", "PPO-Shield"),
]

CHECKPOINTS = {
    "ppo_policy": ("ppo_policy/model.zip", "sb3"),
    "ppo_lagrangian/ppo_lagrangian": ("ppo_lagrangian/ppo_lagrangian.pt", "custom"),
    "ppo_lagrangian/ppo_pid_lagrangian": ("ppo_lagrangian/ppo_pid_lagrangian.pt", "custom"),
    "cpo/cpo": ("cpo/cpo.pt", "custom"),
    "ppo_shield/shielded": ("ppo_shield/model.zip", "sb3"),
}

BASELINE_RUNS = {
    "Bridge Crossing v1": REPO
    / "outputs/_sweeps_1hidden_bridge_crossing_v1_baselines_only/paper_2503_07671_bridge_crossing",
    "Bridge Crossing v2": REPO
    / "outputs/_sweeps_1hidden_missing_all_methods_pspo30000_margin5_no_adaptive/paper_2503_07671_bridge_crossing_v2",
    "Colour Bomb": REPO
    / "outputs/_sweeps_1hidden_missing_all_methods_pspo30000_margin5_no_adaptive/paper_2503_07671_colour_bomb",
    "Colour Bomb v2": REPO / "outputs/_sweeps_1hidden/paper_2503_07671_colour_bomb_v2",
    "Media Streaming": REPO
    / "outputs/_sweeps_1hidden_media_streaming_baselines_only/paper_2503_07671_media_streaming",
    "MiniPacman": REPO
    / "outputs/_sweeps_1hidden_missing_all_methods_pspo30000_margin5_no_adaptive/paper_2503_07671_minipacman",
}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_per_seed(metrics: dict[str, Any], *keys: str) -> dict[str, float]:
    for key in keys:
        record = metrics.get(key)
        if isinstance(record, dict) and isinstance(record.get("per_seed"), dict):
            return {str(seed): float(value) for seed, value in record["per_seed"].items()}
    raise KeyError(f"None of the metric keys were found: {keys}")


def pspo_aggregate_path(record: dict[str, Any]) -> Path:
    source = resolve_path(record["source"])
    if source.is_file():
        return source
    return source / "aggregate" / "aggregated_metrics.json"


def pspo_seed_metrics(record: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    aggregate = load_json(pspo_aggregate_path(record))["metrics"]
    reward = metric_per_seed(
        aggregate,
        "reward.mean_total_reward",
        "rashomon_policy.reward.mean_total_reward",
    )
    safety = metric_per_seed(
        aggregate,
        "safety.safety_rate",
        "rashomon_policy.safety.safety_rate",
    )
    return reward, safety


def baseline_seed_metrics(run_dir: Path, method_key: str) -> tuple[dict[str, float], dict[str, float]]:
    metrics = load_json(run_dir / "aggregate" / "aggregated_metrics.json")["metrics"]
    reward = metric_per_seed(metrics, f"{method_key}.reward.mean_total_reward")
    safety = metric_per_seed(metrics, f"{method_key}.safety.safety_rate")
    return reward, safety


def base_parameter_names(architecture: dict[str, Any]) -> list[str]:
    n_hidden = int(architecture["n_hidden"])
    names: list[str] = []
    for hidden_idx in range(n_hidden):
        layer_idx = hidden_idx * 2
        names.extend([f"{layer_idx}.weight", f"{layer_idx}.bias"])
    final_idx = n_hidden * 2
    names.extend([f"{final_idx}.weight", f"{final_idx}.bias"])
    return names


def sb3_name(base_name: str, architecture: dict[str, Any]) -> str:
    final_idx = int(architecture["n_hidden"]) * 2
    if base_name == f"{final_idx}.weight":
        return "action_net.weight"
    if base_name == f"{final_idx}.bias":
        return "action_net.bias"
    layer_idx, suffix = base_name.split(".", 1)
    return f"mlp_extractor.policy_net.{layer_idx}.{suffix}"


def flatten(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.detach().cpu().reshape(-1).to(torch.float64) for tensor in tensors])


def load_pspo_region(rashomon_dir: Path) -> dict[str, Any]:
    base = torch.load(rashomon_dir / "base_policy.pt", map_location="cpu")
    bounds = torch.load(rashomon_dir / "rashomon_param_bounds.pt", map_location="cpu")
    architecture = dict(base["architecture"])
    names = base_parameter_names(architecture)
    base_state = base["state_dict"]
    lower = list(bounds["param_bounds_l"])
    upper = list(bounds["param_bounds_u"])
    if len(lower) != len(names) or len(upper) != len(names):
        raise ValueError(f"Bound count does not match architecture in {rashomon_dir}")
    base_tensors = [base_state[name] for name in names]
    return {
        "architecture": architecture,
        "names": names,
        "base": flatten(base_tensors),
        "lower": flatten(lower),
        "upper": flatten(upper),
        "num_parameters": int(sum(t.numel() for t in base_tensors)),
    }


def load_custom_actor(path: Path, names: list[str]) -> torch.Tensor:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["actor_state_dict"]
    return flatten([state[name] for name in names])


def load_sb3_actor(path: Path, names: list[str], architecture: dict[str, Any]) -> torch.Tensor:
    with zipfile.ZipFile(path) as archive:
        state = torch.load(archive.open("policy.pth"), map_location="cpu", weights_only=False)
    return flatten([state[sb3_name(name, architecture)] for name in names])


def load_baseline_actor(path: Path, kind: str, names: list[str], architecture: dict[str, Any]) -> torch.Tensor:
    if kind == "custom":
        return load_custom_actor(path, names)
    if kind == "sb3":
        return load_sb3_actor(path, names, architecture)
    raise ValueError(f"Unknown checkpoint kind: {kind}")


def distance_row(theta: torch.Tensor, region: dict[str, Any]) -> dict[str, float | int | bool]:
    lower = region["lower"]
    upper = region["upper"]
    base = region["base"]
    violation = torch.maximum(torch.maximum(lower - theta, theta - upper), torch.zeros_like(theta))
    inside = bool(torch.all(violation <= 0).item())
    dist_region_l2 = float(torch.linalg.vector_norm(violation, ord=2).item())
    dist_region_linf = float(torch.max(violation).item()) if violation.numel() else 0.0
    dist_initial = torch.abs(theta - base)
    box_half_width = 0.5 * (upper - lower)
    box_l2_radius = float(torch.linalg.vector_norm(box_half_width, ord=2).item())
    box_linf_radius = float(torch.max(box_half_width).item()) if box_half_width.numel() else 0.0
    violating = int((violation > 0).sum().item())
    total = int(violation.numel())
    return {
        "inside_safe_region": inside,
        "distance_to_region_l2": dist_region_l2,
        "distance_to_region_linf": dist_region_linf,
        "violating_param_count": violating,
        "violating_param_fraction": float(violating / total) if total else math.nan,
        "distance_to_initial_l2": float(torch.linalg.vector_norm(dist_initial, ord=2).item()),
        "distance_to_initial_linf": float(torch.max(dist_initial).item()) if dist_initial.numel() else 0.0,
        "box_l2_radius": box_l2_radius,
        "box_linf_radius": box_linf_radius,
        "distance_to_region_l2_over_box_l2_radius": (
            dist_region_l2 / box_l2_radius if box_l2_radius > 0 else math.nan
        ),
        "distance_to_region_linf_over_box_linf_radius": (
            dist_region_linf / box_linf_radius if box_linf_radius > 0 else math.nan
        ),
        "distance_to_initial_l2_over_box_l2_radius": (
            float(torch.linalg.vector_norm(dist_initial, ord=2).item()) / box_l2_radius
            if box_l2_radius > 0
            else math.nan
        ),
        "distance_to_initial_linf_over_box_linf_radius": (
            float(torch.max(dist_initial).item()) / box_linf_radius
            if box_linf_radius > 0 and dist_initial.numel()
            else math.nan
        ),
    }


def blank_distance_fields() -> dict[str, str]:
    return {
        "inside_safe_region": "",
        "distance_to_region_l2": "",
        "distance_to_region_linf": "",
        "violating_param_count": "",
        "violating_param_fraction": "",
        "distance_to_initial_l2": "",
        "distance_to_initial_linf": "",
        "box_l2_radius": "",
        "box_linf_radius": "",
        "distance_to_region_l2_over_box_l2_radius": "",
        "distance_to_region_linf_over_box_linf_radius": "",
        "distance_to_initial_l2_over_box_l2_radius": "",
        "distance_to_initial_linf_over_box_linf_radius": "",
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["environment"], row["method_label"])].append(row)
    summary: list[dict[str, Any]] = []
    for (environment, method_label), group in sorted(grouped.items()):
        valid = [row for row in group if row["comparison_status"] == "computed"]
        summary.append({
            "environment": environment,
            "method_label": method_label,
            "candidate_seeds": len(group),
            "computed_seeds": len(valid),
            "inside_safe_region_count": sum(bool(row["inside_safe_region"]) for row in valid),
            "mean_distance_to_region_l2": (
                statistics.fmean(float(row["distance_to_region_l2"]) for row in valid) if valid else None
            ),
            "median_distance_to_region_l2": (
                statistics.median(float(row["distance_to_region_l2"]) for row in valid) if valid else None
            ),
            "mean_distance_to_initial_l2": (
                statistics.fmean(float(row["distance_to_initial_l2"]) for row in valid) if valid else None
            ),
            "median_distance_to_initial_l2": (
                statistics.median(float(row["distance_to_initial_l2"]) for row in valid) if valid else None
            ),
            "statuses": sorted(set(str(row["comparison_status"]) for row in group)),
        })
    return summary


def main() -> None:
    manifest = load_json(MANIFEST)["architectures"]["one_hidden"]
    rows: list[dict[str, Any]] = []

    for environment, pspo_record in manifest.items():
        run_dir = BASELINE_RUNS.get(environment)
        if run_dir is None:
            continue
        pspo_reward, pspo_safety = pspo_seed_metrics(pspo_record)
        rashomon_dir = resolve_path(pspo_record["hyperparameters"]["rashomon_dir"])
        region = load_pspo_region(rashomon_dir)
        architecture = region["architecture"]
        architecture_status = (
            "compatible_one_hidden"
            if int(architecture.get("n_hidden", -1)) == 1
            else f"incompatible_n_hidden_{architecture.get('n_hidden')}"
        )

        for method_key, method_label in BASELINE_METHODS:
            baseline_reward, baseline_safety = baseline_seed_metrics(run_dir, method_key)
            rel_checkpoint, kind = CHECKPOINTS[method_key]
            for seed in sorted(set(pspo_reward) & set(pspo_safety) & set(baseline_reward) & set(baseline_safety), key=int):
                if abs(baseline_safety[seed] - pspo_safety[seed]) > SAFETY_TOL:
                    continue
                if baseline_reward[seed] <= pspo_reward[seed] + REWARD_TOL:
                    continue
                checkpoint_path = run_dir / f"seed{seed}" / rel_checkpoint
                base_row: dict[str, Any] = {
                    "environment": environment,
                    "seed": int(seed),
                    "method_key": method_key,
                    "method_label": method_label,
                    "baseline_reward": baseline_reward[seed],
                    "baseline_safety_rate": baseline_safety[seed],
                    "pspo_reward": pspo_reward[seed],
                    "pspo_safety_rate": pspo_safety[seed],
                    "reward_delta_baseline_minus_pspo": baseline_reward[seed] - pspo_reward[seed],
                    "safety_delta_baseline_minus_pspo": baseline_safety[seed] - pspo_safety[seed],
                    "rashomon_dir": str(rashomon_dir),
                    "pspo_region_n_hidden": int(architecture.get("n_hidden", -1)),
                    "pspo_region_num_parameters": region["num_parameters"],
                    "checkpoint_path": str(checkpoint_path),
                }
                if architecture_status != "compatible_one_hidden":
                    base_row.update({"comparison_status": "architecture_mismatch"})
                    base_row.update(blank_distance_fields())
                    rows.append(base_row)
                    continue
                if not checkpoint_path.exists():
                    base_row.update({"comparison_status": "missing_checkpoint"})
                    base_row.update(blank_distance_fields())
                    rows.append(base_row)
                    continue
                try:
                    theta = load_baseline_actor(checkpoint_path, kind, region["names"], architecture)
                    if theta.shape != region["base"].shape:
                        raise ValueError(f"shape mismatch: baseline={tuple(theta.shape)}, region={tuple(region['base'].shape)}")
                    base_row.update({"comparison_status": "computed"})
                    base_row.update(distance_row(theta, region))
                except Exception as exc:  # noqa: BLE001 - keep analysis row instead of aborting.
                    base_row.update({"comparison_status": f"load_or_shape_error: {exc}"})
                    base_row.update(blank_distance_fields())
                rows.append(base_row)

    fieldnames = [
        "environment",
        "seed",
        "method_key",
        "method_label",
        "baseline_reward",
        "baseline_safety_rate",
        "pspo_reward",
        "pspo_safety_rate",
        "reward_delta_baseline_minus_pspo",
        "safety_delta_baseline_minus_pspo",
        "comparison_status",
        "inside_safe_region",
        "distance_to_region_l2",
        "distance_to_region_linf",
        "violating_param_count",
        "violating_param_fraction",
        "distance_to_initial_l2",
        "distance_to_initial_linf",
        "box_l2_radius",
        "box_linf_radius",
        "distance_to_region_l2_over_box_l2_radius",
        "distance_to_region_linf_over_box_linf_radius",
        "distance_to_initial_l2_over_box_l2_radius",
        "distance_to_initial_linf_over_box_linf_radius",
        "pspo_region_n_hidden",
        "pspo_region_num_parameters",
        "rashomon_dir",
        "checkpoint_path",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    OUT_JSON.write_text(
        json.dumps(
            {
                "selection": (
                    "Per-seed baseline rows where safety_rate matches PSPO and total reward is higher."
                ),
                "safety_tolerance": SAFETY_TOL,
                "reward_tolerance": REWARD_TOL,
                "csv": str(OUT_CSV.relative_to(REPO)),
                "rows": rows,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    print("\nSummary:")
    for item in summary:
        mean_region = item["mean_distance_to_region_l2"]
        mean_initial = item["mean_distance_to_initial_l2"]
        print(
            f"- {item['environment']} / {item['method_label']}: "
            f"candidate={item['candidate_seeds']}, computed={item['computed_seeds']}, "
            f"inside={item['inside_safe_region_count']}, "
            f"mean_d_region_l2={mean_region}, mean_d_init_l2={mean_initial}, "
            f"statuses={','.join(item['statuses'])}"
        )


if __name__ == "__main__":
    main()
