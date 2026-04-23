"""Expand a Rashomon set via sampled-bound reference policy and union-PGD adaptation.

For one LunarLander task setting and seed, this script:
1) Loads source/unconstrained/EWC/Rashomon actors + Rashomon bounded model + dataset.
2) Computes pairwise actor-parameter distances.
3) Checks whether unconstrained and EWC actors lie outside the original Rashomon set,
   and measures distance to the set in parameter space.
4) Samples a new reference actor by independently choosing lower/upper bound per parameter.
5) Recomputes a new Rashomon set from that sampled reference actor.
6) Builds an updated interval set as a multi-interval union of old/new intervals
   (without collapsing them into one min/max envelope).
7) Runs downstream PPO adaptation with PGD projection onto the updated interval set.
8) Evaluates adapted policy on source and downstream tasks.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import sys
from typing import Any

os.environ["SDL_AUDIODRIVER"] = "dummy"

import gymnasium as gym
import torch
from torch.utils.data import TensorDataset
import yaml

# Allow running this file directly from experiments/pipelines/lunarlander.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.lunarlander.core.env.env_factory import _make_lunarlander_env
from experiments.pipelines.lunarlander.core.env.task_loading import (
    _load_task_settings,
    _resolve_lunarlander_dynamics,
)
from experiments.pipelines.lunarlander.core.eval.evaluate_policy import (
    _build_actor_from_state_dict,
    _resolve_actor_path,
)
from experiments.pipelines.lunarlander.core.methods.adapt_rashomon import (
    _load_source_hidden_size,
    compute_rashomon_bounds,
    neutralize_task_feature,
)
from experiments.pipelines.lunarlander.core.methods.source_train import build_actor_critic
from experiments.pipelines.lunarlander.core.orchestration.run_paths import (
    default_adapt_ppo_settings_file,
    default_adapt_rashomon_settings_file,
    default_outputs_root,
    default_task_settings_file,
    resolve_default_source_run_dir as _resolve_default_source_run_dir,
    resolve_policy_dir as _resolve_policy_dir,
    seed_run_dir as _seed_run_dir,
)
from experiments.utils.ppo_utils import PPOConfig, evaluate_with_success, ppo_train


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data)}.")
    return data


def _resolve_setting_cfg(
    settings: dict[str, Any],
    setting_name: str,
    *,
    settings_name: str,
) -> dict[str, Any]:
    if setting_name in settings:
        cfg = settings[setting_name]
    elif "default" in settings:
        cfg = settings["default"]
    else:
        raise ValueError(
            f"Setting '{setting_name}' not found in {settings_name}, and no 'default' key exists.",
        )
    if not isinstance(cfg, dict):
        raise ValueError(
            f"Expected mapping for setting '{setting_name}' in {settings_name}, got {type(cfg)}.",
        )
    return cfg


def _torch_load_any(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _discover_rashomon_run_dir(
    *,
    task_setting: str,
    seed: int,
    outputs_root: Path,
    preferred_subdir: str | None,
) -> Path:
    seed_roots = [
        outputs_root / task_setting / f"seed_{seed}",
        outputs_root / f"seed_{seed}",
    ]
    candidate_dirs: list[Path] = []

    if preferred_subdir:
        for root in seed_roots:
            candidate_dirs.append(root / preferred_subdir)

    for root in seed_roots:
        candidate_dirs.append(root / "downstream_rashomon")
        if root.exists():
            candidate_dirs.extend(sorted(p for p in root.glob("downstream_rashomon*") if p.is_dir()))

    seen: set[Path] = set()
    dedup_dirs: list[Path] = []
    for d in candidate_dirs:
        if d in seen:
            continue
        seen.add(d)
        dedup_dirs.append(d)

    for run_dir in dedup_dirs:
        model_path = run_dir / "rashomon_bounded_model.pt"
        dataset_path = run_dir / "rashomon_dataset.pt"
        if model_path.exists() and dataset_path.exists():
            return run_dir

    tried = [str(d) for d in dedup_dirs]
    raise FileNotFoundError(
        "Could not find a Rashomon run directory containing both "
        "`rashomon_bounded_model.pt` and `rashomon_dataset.pt`. Tried:\n- "
        + "\n- ".join(tried),
    )


def _load_actor_from_policy_dir(
    *,
    policy_name: str,
    policy_dir: Path,
    device: str,
) -> dict[str, Any]:
    if not policy_dir.exists():
        raise FileNotFoundError(f"Policy directory does not exist for '{policy_name}': {policy_dir}")
    actor_path = _resolve_actor_path(policy_dir, policy_name)
    state_dict = _torch_load_any(actor_path)
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected actor checkpoint to be a state_dict dict for '{policy_name}', got {type(state_dict)}.",
        )
    actor = _build_actor_from_state_dict(state_dict).to(device)
    actor.eval()
    return {
        "actor": actor,
        "actor_path": actor_path,
        "policy_dir": policy_dir,
        "num_params": _count_params(actor),
    }


def _actor_param_vector(param_list: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in param_list], dim=0)


def _pairwise_actor_l2_distances(
    actor_parameters_dct: dict[str, list[torch.Tensor]],
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    actor_names = sorted(actor_parameters_dct.keys())
    actor_vectors = {name: _actor_param_vector(actor_parameters_dct[name]) for name in actor_names}

    sizes = {name: vec.numel() for name, vec in actor_vectors.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Actors have different flattened parameter sizes: {sizes}")

    matrix: dict[str, dict[str, float]] = {name: {} for name in actor_names}
    pairs: list[dict[str, Any]] = []
    for i, name_i in enumerate(actor_names):
        for j, name_j in enumerate(actor_names):
            dist = float(torch.norm(actor_vectors[name_i] - actor_vectors[name_j], p=2).item())
            matrix[name_i][name_j] = dist
            if i < j:
                pairs.append(
                    {
                        "actor_a": name_i,
                        "actor_b": name_j,
                        "l2_distance": dist,
                    },
                )
    pairs.sort(key=lambda x: float(x["l2_distance"]), reverse=True)
    return matrix, pairs


def _analyze_distance_to_rashomon_interval(
    *,
    actor_name: str,
    actor_param_list: list[torch.Tensor],
    bounds_l_intervals: list[list[torch.Tensor]],
    bounds_u_intervals: list[list[torch.Tensor]],
    top_k: int,
) -> dict[str, Any]:
    if len(bounds_l_intervals) != len(bounds_u_intervals):
        raise ValueError(
            "Interval-set lower/upper lengths mismatch: "
            f"lower={len(bounds_l_intervals)} upper={len(bounds_u_intervals)}",
        )
    if len(bounds_l_intervals) == 0:
        raise ValueError("At least one interval set is required.")

    n_intervals = len(bounds_l_intervals)
    n_param_tensors = len(actor_param_list)
    for int_idx, (bounds_l, bounds_u) in enumerate(zip(bounds_l_intervals, bounds_u_intervals)):
        if len(bounds_l) != len(bounds_u):
            raise ValueError(
                f"Interval {int_idx}: lower/upper tensor count mismatch "
                f"(lower={len(bounds_l)}, upper={len(bounds_u)}).",
            )
        if len(bounds_l) != n_param_tensors:
            raise ValueError(
                f"Interval {int_idx}: expected {n_param_tensors} parameter tensors, "
                f"got {len(bounds_l)}.",
            )

    # Project to each convex Rashomon set (axis-aligned box in full parameter space),
    # then take the closest one. This preserves cross-parameter coupling.
    best_interval_idx: int | None = None
    best_dist_sq = float("inf")
    n_total_params = sum(int(p.detach().numel()) for p in actor_param_list)

    for int_idx, (bounds_l, bounds_u) in enumerate(zip(bounds_l_intervals, bounds_u_intervals)):
        dist_sq = 0.0
        for tensor_idx, p_t in enumerate(actor_param_list):
            p_flat = p_t.detach().flatten().cpu()
            l_flat = bounds_l[tensor_idx].detach().flatten().cpu()
            u_flat = bounds_u[tensor_idx].detach().flatten().cpu()
            if p_flat.shape != l_flat.shape or p_flat.shape != u_flat.shape:
                raise ValueError(
                    f"{actor_name}: shape mismatch at tensor {tensor_idx}, interval {int_idx}: "
                    f"param={tuple(p_flat.shape)}, lower={tuple(l_flat.shape)}, upper={tuple(u_flat.shape)}",
                )
            projected = torch.maximum(torch.minimum(p_flat, u_flat), l_flat)
            delta = projected - p_flat
            dist_sq += float((delta * delta).sum().item())

        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_interval_idx = int_idx

    if best_interval_idx is None:
        raise RuntimeError(f"{actor_name}: failed to select nearest interval set.")

    chosen_l = bounds_l_intervals[best_interval_idx]
    chosen_u = bounds_u_intervals[best_interval_idx]
    l1_sum = 0.0
    l2_sq_sum = 0.0
    linf = 0.0
    outside_entries: list[dict[str, Any]] = []

    for tensor_idx, p_t in enumerate(actor_param_list):
        p_flat = p_t.detach().flatten().cpu()
        l_flat = chosen_l[tensor_idx].detach().flatten().cpu()
        u_flat = chosen_u[tensor_idx].detach().flatten().cpu()
        if p_flat.shape != l_flat.shape or p_flat.shape != u_flat.shape:
            raise ValueError(
                f"{actor_name}: shape mismatch at tensor {tensor_idx}, chosen_interval {best_interval_idx}: "
                f"param={tuple(p_flat.shape)}, lower={tuple(l_flat.shape)}, upper={tuple(u_flat.shape)}",
            )

        below_violation = torch.clamp(l_flat - p_flat, min=0.0)
        above_violation = torch.clamp(p_flat - u_flat, min=0.0)
        violation = below_violation + above_violation
        outside_mask = violation > 0.0
        outside_idx = torch.nonzero(outside_mask, as_tuple=False).flatten().tolist()
        if not outside_idx:
            continue

        violation_values = violation[outside_mask]
        l1_sum += float(violation_values.sum().item())
        l2_sq_sum += float((violation_values * violation_values).sum().item())
        linf = max(linf, float(violation_values.max().item()))

        for flat_idx in outside_idx:
            val = float(p_flat[flat_idx].item())
            lo = float(l_flat[flat_idx].item())
            hi = float(u_flat[flat_idx].item())
            if val < lo:
                nearest_endpoint = "lower"
                distance = lo - val
            else:
                nearest_endpoint = "upper"
                distance = val - hi
            outside_entries.append(
                {
                    "actor": actor_name,
                    "tensor_idx": int(tensor_idx),
                    "flat_idx": int(flat_idx),
                    "interval_idx": int(best_interval_idx),
                    "distance_to_interval": float(distance),
                    "nearest_endpoint": nearest_endpoint,
                    "value": val,
                    "lower": lo,
                    "upper": hi,
                },
            )

    outside_entries.sort(key=lambda x: float(x["distance_to_interval"]), reverse=True)
    n_outside = len(outside_entries)
    mean_outside_distance = (l1_sum / n_outside) if n_outside > 0 else 0.0

    return {
        "actor": actor_name,
        "is_inside": bool(n_outside == 0),
        "n_total_params": int(n_total_params),
        "n_outside": int(n_outside),
        "outside_fraction": float(n_outside / n_total_params) if n_total_params > 0 else 0.0,
        "distance_l1": float(l1_sum),
        "distance_l2": float(l2_sq_sum**0.5),
        "distance_linf": float(linf),
        "mean_outside_distance": float(mean_outside_distance),
        "n_intervals": int(n_intervals),
        "selected_interval_idx": int(best_interval_idx),
        "outside_entries": outside_entries,
        "top_outside_entries": outside_entries[:top_k],
    }


def _sample_param_list_from_bounds(
    *,
    bounds_l: list[torch.Tensor],
    bounds_u: list[torch.Tensor],
    upper_prob: float,
    seed: int,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    if len(bounds_l) != len(bounds_u):
        raise ValueError(
            f"Bounds length mismatch: lower={len(bounds_l)} upper={len(bounds_u)}",
        )
    if upper_prob < 0.0 or upper_prob > 1.0:
        raise ValueError(f"upper_prob must be in [0, 1], got {upper_prob}.")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    sampled: list[torch.Tensor] = []
    total = 0
    num_upper = 0

    for tensor_idx, (l_t, u_t) in enumerate(zip(bounds_l, bounds_u)):
        if l_t.shape != u_t.shape:
            raise ValueError(
                f"Bounds shape mismatch at tensor {tensor_idx}: {tuple(l_t.shape)} vs {tuple(u_t.shape)}",
            )
        l_cpu = l_t.detach().cpu()
        u_cpu = u_t.detach().cpu()
        mask = torch.rand(l_cpu.shape, generator=gen) < upper_prob
        sampled_t = torch.where(mask, u_cpu, l_cpu)
        sampled.append(sampled_t)
        total += int(mask.numel())
        num_upper += int(mask.sum().item())

    sample_stats = {
        "seed": int(seed),
        "upper_prob": float(upper_prob),
        "n_total": int(total),
        "n_upper": int(num_upper),
        "n_lower": int(total - num_upper),
        "upper_fraction": float(num_upper / total) if total > 0 else 0.0,
    }
    return sampled, sample_stats


def _build_reference_actor_from_sampled_params(
    *,
    source_actor_for_template: torch.nn.Module,
    sampled_param_list: list[torch.Tensor],
    device: str,
) -> tuple[torch.nn.Sequential, dict[str, torch.Tensor]]:
    template_state_dict = source_actor_for_template.state_dict()
    state_keys = list(template_state_dict.keys())
    if len(sampled_param_list) != len(state_keys):
        raise ValueError(
            f"Length mismatch: sampled_param_list={len(sampled_param_list)} vs state_keys={len(state_keys)}",
        )

    sampled_state_dict: dict[str, torch.Tensor] = {}
    for key, sampled_t in zip(state_keys, sampled_param_list):
        expected = template_state_dict[key]
        if tuple(sampled_t.shape) != tuple(expected.shape):
            raise ValueError(
                f"Shape mismatch for '{key}': sampled={tuple(sampled_t.shape)} vs expected={tuple(expected.shape)}",
            )
        sampled_state_dict[key] = sampled_t.detach().clone().to(
            device="cpu",
            dtype=expected.dtype,
        )

    actor = _build_actor_from_state_dict(sampled_state_dict).to(device)
    actor.eval()
    return actor, sampled_state_dict


def _summarize_bound_widths(
    *,
    bounds_l: list[torch.Tensor],
    bounds_u: list[torch.Tensor],
) -> dict[str, Any]:
    if len(bounds_l) != len(bounds_u):
        raise ValueError("Lower/upper bounds lengths mismatch.")
    widths = []
    for tensor_idx, (l_t, u_t) in enumerate(zip(bounds_l, bounds_u)):
        if l_t.shape != u_t.shape:
            raise ValueError(
                f"Width summary shape mismatch at tensor {tensor_idx}: {tuple(l_t.shape)} vs {tuple(u_t.shape)}",
            )
        w_t = (u_t.detach().cpu() - l_t.detach().cpu()).flatten()
        if torch.any(w_t < 0):
            bad_idx = int(torch.argmin(w_t).item())
            raise ValueError(
                f"Negative width found at tensor {tensor_idx}, flat_idx {bad_idx}.",
            )
        widths.append(w_t)
    if not widths:
        return {
            "n_tensors": 0,
            "n_params": 0,
            "width_min": 0.0,
            "width_mean": 0.0,
            "width_max": 0.0,
        }
    all_widths = torch.cat(widths)
    return {
        "n_tensors": int(len(bounds_l)),
        "n_params": int(all_widths.numel()),
        "width_min": float(all_widths.min().item()),
        "width_mean": float(all_widths.mean().item()),
        "width_max": float(all_widths.max().item()),
    }


def _outside_analysis_for_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "actor": analysis["actor"],
        "is_inside": bool(analysis["is_inside"]),
        "n_intervals": int(analysis.get("n_intervals", 1)),
        "selected_interval_idx": int(analysis.get("selected_interval_idx", 0)),
        "n_total_params": int(analysis["n_total_params"]),
        "n_outside": int(analysis["n_outside"]),
        "outside_fraction": float(analysis["outside_fraction"]),
        "distance_l1": float(analysis["distance_l1"]),
        "distance_l2": float(analysis["distance_l2"]),
        "distance_linf": float(analysis["distance_linf"]),
        "mean_outside_distance": float(analysis["mean_outside_distance"]),
        "top_outside_entries": list(analysis["top_outside_entries"]),
    }


def _write_pairwise_matrix_csv(
    *,
    matrix: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    actor_names = sorted(matrix.keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actor", *actor_names])
        for row_actor in actor_names:
            writer.writerow([row_actor, *[matrix[row_actor][col_actor] for col_actor in actor_names]])


def _write_pairwise_pairs_csv(*, pairs: list[dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["actor_a", "actor_b", "l2_distance"])
        writer.writeheader()
        for row in pairs:
            writer.writerow(row)


def _write_outside_entries_csv(*, entries: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "actor",
        "tensor_idx",
        "flat_idx",
        "interval_idx",
        "distance_to_interval",
        "nearest_endpoint",
        "value",
        "lower",
        "upper",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in entries:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Expand a Rashomon set for one LunarLander task/seed by sampling a new reference "
            "policy from old bounds, recomputing bounds, unioning interval sets, and adapting with PGD."
        ),
    )
    parser.add_argument("--task-setting", type=str, default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--task-settings-file",
        type=Path,
        default=default_task_settings_file(),
    )
    parser.add_argument(
        "--adapt-settings-file",
        type=Path,
        default=default_adapt_ppo_settings_file(),
    )
    parser.add_argument(
        "--rashomon-settings-file",
        type=Path,
        default=default_adapt_rashomon_settings_file(),
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=default_outputs_root(),
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help="Optional explicit source run directory.",
    )
    parser.add_argument(
        "--unconstrained-run-subdir",
        type=str,
        default="downstream_unconstrained",
    )
    parser.add_argument(
        "--ewc-run-subdir",
        type=str,
        default="downstream_ewc",
    )
    parser.add_argument(
        "--rashomon-run-subdir",
        type=str,
        default="downstream_rashomon",
        help="Preferred Rashomon run subdir. Auto-discovery still searches downstream_rashomon*.",
    )
    parser.add_argument(
        "--rashomon-run-dir",
        type=Path,
        default=None,
        help="Optional explicit Rashomon run directory containing rashomon_bounded_model.pt and rashomon_dataset.pt.",
    )
    parser.add_argument(
        "--run-subdir",
        type=str,
        default="downstream_rashomon_union_expanded",
        help="Output subdir under outputs/<task_setting>/seed_<seed>/.",
    )
    parser.add_argument(
        "--analysis-top-k",
        type=int,
        default=10,
        help="How many top out-of-bounds parameters to surface in summary logs.",
    )
    parser.add_argument(
        "--upper-bound-prob",
        type=float,
        default=0.5,
        help="Per-parameter probability of choosing upper bound when sampling reference actor.",
    )
    parser.add_argument(
        "--bound-sample-seed",
        type=int,
        default=None,
        help="Random seed for lower/upper bound sampling. Defaults to --seed.",
    )
    parser.add_argument(
        "--warm-start-critic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm-start critic from source checkpoint.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Override hidden size. Otherwise inferred from source run summary/checkpoint.",
    )
    parser.add_argument(
        "--task-neutralization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override task-feature neutralization before adaptation.",
    )

    # Optional task/env overrides.
    parser.add_argument("--env-id", type=str, default=None)
    parser.add_argument("--source-gravity", type=float, default=None)
    parser.add_argument("--downstream-gravity", type=float, default=None)
    parser.add_argument("--source-task-id", type=float, default=None)
    parser.add_argument("--downstream-task-id", type=float, default=None)
    parser.add_argument(
        "--append-task-id",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    # Rashomon recompute overrides.
    parser.add_argument("--rashomon-n-iters", type=int, default=None)
    parser.add_argument("--rashomon-min-hard-spec", type=float, default=None)
    parser.add_argument(
        "--rashomon-surrogate-aggregation",
        type=str,
        choices=["mean", "min"],
        default=None,
    )
    parser.add_argument("--inverse-temp-start", type=int, default=None)
    parser.add_argument("--inverse-temp-max", type=int, default=None)
    parser.add_argument("--rashomon-checkpoint", type=int, default=None)

    # PPO overrides.
    parser.add_argument("--total-timesteps-override", type=int, default=None)
    parser.add_argument("--eval-episodes-during-training", type=int, default=None)
    parser.add_argument("--eval-episodes-post-training", type=int, default=None)
    args = parser.parse_args()

    if args.analysis_top_k <= 0:
        raise ValueError("--analysis-top-k must be > 0.")
    if args.upper_bound_prob < 0.0 or args.upper_bound_prob > 1.0:
        raise ValueError("--upper-bound-prob must be in [0, 1].")

    # Load task env configs.
    source_task_cfg = _load_task_settings(args.task_settings_file, args.task_setting, "source")
    downstream_task_cfg = _load_task_settings(args.task_settings_file, args.task_setting, "downstream")

    env_id = str(
        args.env_id
        or source_task_cfg.get("env_id")
        or downstream_task_cfg.get("env_id")
        or "LunarLander-v3",
    )
    source_gravity_raw = args.source_gravity if args.source_gravity is not None else source_task_cfg.get("gravity")
    downstream_gravity_raw = (
        args.downstream_gravity
        if args.downstream_gravity is not None
        else downstream_task_cfg.get("gravity")
    )
    source_gravity = None if source_gravity_raw is None else float(source_gravity_raw)
    downstream_gravity = None if downstream_gravity_raw is None else float(downstream_gravity_raw)

    source_task_id = float(args.source_task_id) if args.source_task_id is not None else float(
        source_task_cfg.get("task_id", 0.0),
    )
    downstream_task_id = float(args.downstream_task_id) if args.downstream_task_id is not None else float(
        downstream_task_cfg.get("task_id", 1.0),
    )
    append_task_id = (
        bool(args.append_task_id)
        if args.append_task_id is not None
        else bool(source_task_cfg.get("append_task_id", True))
    )

    source_dynamics = _resolve_lunarlander_dynamics(
        source_task_cfg,
        cfg_name=f"task_settings[{args.task_setting}:source]",
    )
    downstream_dynamics = _resolve_lunarlander_dynamics(
        downstream_task_cfg,
        cfg_name=f"task_settings[{args.task_setting}:downstream]",
    )

    continuous = bool(source_task_cfg.get("continuous", False) or downstream_task_cfg.get("continuous", False))
    if continuous:
        raise ValueError("This script only supports discrete actions (`continuous=False`).")

    source_env_kwargs = {
        "gravity": source_gravity,
        "task_id": source_task_id,
        "append_task_id": append_task_id,
        **source_dynamics,
    }
    downstream_env_kwargs = {
        "gravity": downstream_gravity,
        "task_id": downstream_task_id,
        "append_task_id": append_task_id,
        **downstream_dynamics,
    }

    # Resolve run directories.
    source_run_dir = (
        args.source_run_dir
        if args.source_run_dir is not None
        else _resolve_default_source_run_dir(args.outputs_root, args.task_setting, args.seed)
    )
    unconstrained_dir = _resolve_policy_dir(
        args.outputs_root,
        args.task_setting,
        args.seed,
        args.unconstrained_run_subdir,
    )
    ewc_dir = _resolve_policy_dir(
        args.outputs_root,
        args.task_setting,
        args.seed,
        args.ewc_run_subdir,
    )
    if args.rashomon_run_dir is not None:
        rashomon_run_dir = args.rashomon_run_dir
    else:
        rashomon_run_dir = _discover_rashomon_run_dir(
            task_setting=args.task_setting,
            seed=args.seed,
            outputs_root=args.outputs_root,
            preferred_subdir=args.rashomon_run_subdir,
        )

    print(f"Using source run dir: {source_run_dir}")
    print(f"Using unconstrained run dir: {unconstrained_dir}")
    print(f"Using EWC run dir: {ewc_dir}")
    print(f"Using Rashomon run dir: {rashomon_run_dir}")

    # Load actors for analysis.
    actor_dirs = {
        "source": source_run_dir,
        "downstream_unconstrained": unconstrained_dir,
        "downstream_ewc": ewc_dir,
        "downstream_rashomon": rashomon_run_dir,
    }
    actors: dict[str, dict[str, Any]] = {}
    for policy_name, policy_dir in actor_dirs.items():
        actors[policy_name] = _load_actor_from_policy_dir(
            policy_name=policy_name,
            policy_dir=policy_dir,
            device=args.device,
        )
    actor_parameters_dct: dict[str, list[torch.Tensor]] = {
        policy_name: list(item["actor"].parameters())
        for policy_name, item in actors.items()
    }

    print("\nLoaded actors:")
    for policy_name in sorted(actors):
        rec = actors[policy_name]
        print(f"- {policy_name:24s} params={rec['num_params']:<10d} path={rec['actor_path']}")

    # Load Rashomon artifacts.
    old_bounded_model_path = rashomon_run_dir / "rashomon_bounded_model.pt"
    rashomon_dataset_path = rashomon_run_dir / "rashomon_dataset.pt"
    if not old_bounded_model_path.exists():
        raise FileNotFoundError(f"Missing Rashomon bounded model: {old_bounded_model_path}")
    if not rashomon_dataset_path.exists():
        raise FileNotFoundError(f"Missing Rashomon dataset: {rashomon_dataset_path}")

    old_bounded_model = _torch_load_any(old_bounded_model_path)
    if not hasattr(old_bounded_model, "param_l") or not hasattr(old_bounded_model, "param_u"):
        raise TypeError(
            f"Unexpected Rashomon bounded model object at {old_bounded_model_path}: "
            f"{type(old_bounded_model)} (missing param_l/param_u).",
        )
    old_bounds_l = [p.detach().cpu().clone() for p in old_bounded_model.param_l]
    old_bounds_u = [p.detach().cpu().clone() for p in old_bounded_model.param_u]

    rashomon_dataset = _torch_load_any(rashomon_dataset_path)
    if not isinstance(rashomon_dataset, TensorDataset):
        raise TypeError(
            f"Expected TensorDataset at {rashomon_dataset_path}, got {type(rashomon_dataset)}.",
        )
    print(
        f"\nLoaded Rashomon artifacts: {len(old_bounds_l)} bound tensors, "
        f"dataset_size={len(rashomon_dataset)}",
    )

    # Step 2: pairwise actor distance analysis.
    pairwise_l2_matrix, pairwise_l2_pairs = _pairwise_actor_l2_distances(actor_parameters_dct)
    print("\nPairwise actor L2 distances (top pairs):")
    for row in pairwise_l2_pairs[: min(6, len(pairwise_l2_pairs))]:
        print(f"- {row['actor_a']} vs {row['actor_b']}: {row['l2_distance']:.6f}")

    # Step 3: outside-of-Rashomon analysis for unconstrained and EWC actors.
    old_outside_unconstrained = _analyze_distance_to_rashomon_interval(
        actor_name="downstream_unconstrained",
        actor_param_list=actor_parameters_dct["downstream_unconstrained"],
        bounds_l_intervals=[old_bounds_l],
        bounds_u_intervals=[old_bounds_u],
        top_k=args.analysis_top_k,
    )
    old_outside_ewc = _analyze_distance_to_rashomon_interval(
        actor_name="downstream_ewc",
        actor_param_list=actor_parameters_dct["downstream_ewc"],
        bounds_l_intervals=[old_bounds_l],
        bounds_u_intervals=[old_bounds_u],
        top_k=args.analysis_top_k,
    )
    print(
        "\nOutside-old-Rashomon summary:\n"
        f"- downstream_unconstrained: outside={old_outside_unconstrained['n_outside']} "
        f"| L2-distance={old_outside_unconstrained['distance_l2']:.6f}\n"
        f"- downstream_ewc: outside={old_outside_ewc['n_outside']} "
        f"| L2-distance={old_outside_ewc['distance_l2']:.6f}",
    )

    old_width_summary = _summarize_bound_widths(bounds_l=old_bounds_l, bounds_u=old_bounds_u)

    old_run_summary_path = rashomon_run_dir / "run_summary.yaml"
    old_summary_run_settings: dict[str, Any] = {}
    if old_run_summary_path.exists():
        old_summary = _load_yaml(old_run_summary_path)
        run_settings = old_summary.get("run_settings", {})
        if isinstance(run_settings, dict):
            old_summary_run_settings = run_settings

    # Load adaptation settings early because we need post-training eval episodes
    # for the pre-expansion success-rate gate.
    adapt_settings = _load_yaml(args.adapt_settings_file)
    adapt_cfg = _resolve_setting_cfg(
        adapt_settings,
        args.task_setting,
        settings_name=str(args.adapt_settings_file),
    )
    adapt_ppo_cfg = adapt_cfg.get("ppo", {})
    if not isinstance(adapt_ppo_cfg, dict):
        raise ValueError(
            f"Expected 'ppo' mapping for setting '{args.task_setting}' in {args.adapt_settings_file}.",
        )
    downstream_eval_cfg = adapt_cfg.get("downstream_eval", {})
    if not isinstance(downstream_eval_cfg, dict):
        downstream_eval_cfg = {}

    eval_episodes_during_training = int(
        args.eval_episodes_during_training
        if args.eval_episodes_during_training is not None
        else adapt_ppo_cfg.get("eval_episodes_during_training", 20),
    )
    eval_episodes_post_training = int(
        args.eval_episodes_post_training
        if args.eval_episodes_post_training is not None
        else downstream_eval_cfg.get("episodes_post_training", 100),
    )
    if eval_episodes_post_training <= 0:
        raise ValueError("--eval-episodes-post-training must be > 0.")

    downstream_eval_env_precheck = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **downstream_env_kwargs,
    )
    (
        existing_downstream_mean_reward,
        existing_downstream_std_reward,
        existing_downstream_failure_rate,
        existing_downstream_success_rate,
    ) = evaluate_with_success(
        downstream_eval_env_precheck,
        actors["downstream_rashomon"]["actor"],
        episodes=eval_episodes_post_training,
        deterministic=True,
        device=args.device,
    )
    downstream_eval_env_precheck.close()
    print(
        "\nExisting Rashomon actor downstream eval before expansion: "
        f"mean={existing_downstream_mean_reward:.3f}, "
        f"std={existing_downstream_std_reward:.3f}, "
        f"failure_rate={existing_downstream_failure_rate:.3f}, "
        f"success_rate={existing_downstream_success_rate:.3f}",
    )

    if existing_downstream_success_rate >= 1.0:
        print(
            "\nSkipping Rashomon expansion and PPO-PGD because the existing Rashomon actor "
            "already reached downstream success_rate=1.0.",
        )

        source_eval_env_precheck = _make_lunarlander_env(
            env_id,
            render_mode=None,
            **source_env_kwargs,
        )
        (
            existing_source_mean_reward,
            existing_source_std_reward,
            existing_source_failure_rate,
            existing_source_success_rate,
        ) = evaluate_with_success(
            source_eval_env_precheck,
            actors["downstream_rashomon"]["actor"],
            episodes=eval_episodes_post_training,
            deterministic=True,
            device=args.device,
        )
        source_eval_env_precheck.close()

        adapted_outside_old = _analyze_distance_to_rashomon_interval(
            actor_name="downstream_rashomon",
            actor_param_list=actor_parameters_dct["downstream_rashomon"],
            bounds_l_intervals=[old_bounds_l],
            bounds_u_intervals=[old_bounds_u],
            top_k=args.analysis_top_k,
        )
        union_interval_width_summary = {
            "n_intervals": 1,
            "per_interval": [
                {
                    "name": "old",
                    **old_width_summary,
                },
            ],
        }

        run_dir = _seed_run_dir(args.outputs_root, args.task_setting, args.seed) / args.run_subdir
        run_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir = run_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        pairwise_matrix_csv = analysis_dir / "actor_pairwise_l2_matrix.csv"
        pairwise_pairs_csv = analysis_dir / "actor_pairwise_l2_pairs.csv"
        outside_old_unconstrained_csv = analysis_dir / "outside_old_rashomon_downstream_unconstrained.csv"
        outside_old_ewc_csv = analysis_dir / "outside_old_rashomon_downstream_ewc.csv"
        outside_union_unconstrained_csv = analysis_dir / "outside_union_rashomon_downstream_unconstrained.csv"
        outside_union_ewc_csv = analysis_dir / "outside_union_rashomon_downstream_ewc.csv"
        outside_union_adapted_csv = analysis_dir / "outside_union_rashomon_adapted_actor.csv"
        summary_path = run_dir / "run_summary.yaml"

        _write_pairwise_matrix_csv(matrix=pairwise_l2_matrix, out_path=pairwise_matrix_csv)
        _write_pairwise_pairs_csv(pairs=pairwise_l2_pairs, out_path=pairwise_pairs_csv)
        _write_outside_entries_csv(
            entries=old_outside_unconstrained["outside_entries"],
            out_path=outside_old_unconstrained_csv,
        )
        _write_outside_entries_csv(
            entries=old_outside_ewc["outside_entries"],
            out_path=outside_old_ewc_csv,
        )
        _write_outside_entries_csv(
            entries=old_outside_unconstrained["outside_entries"],
            out_path=outside_union_unconstrained_csv,
        )
        _write_outside_entries_csv(
            entries=old_outside_ewc["outside_entries"],
            out_path=outside_union_ewc_csv,
        )
        _write_outside_entries_csv(
            entries=adapted_outside_old["outside_entries"],
            out_path=outside_union_adapted_csv,
        )

        copied_artifact_names: list[str] = []
        for src_path in sorted(rashomon_run_dir.iterdir()):
            if not src_path.is_file():
                continue
            if src_path.name == "run_summary.yaml":
                continue
            dst_path = run_dir / src_path.name
            if src_path.resolve() == dst_path.resolve():
                continue
            shutil.copy2(src_path, dst_path)
            copied_artifact_names.append(src_path.name)

        actor_out_path = run_dir / "actor.pt"
        critic_out_path = run_dir / "critic.pt"
        training_data_out_path = run_dir / "training_data.pt"
        copied_rashomon_dataset_out_path = run_dir / "rashomon_dataset.pt"
        copied_bounded_model_out_path = run_dir / "rashomon_bounded_model.pt"
        copied_bounds_out_path = run_dir / "rashomon_param_bounds.pt"
        copied_rollout_stats_out_path = run_dir / "rashomon_rollout_stats.yaml"

        summary = {
            "run_settings": {
                "task_setting": str(args.task_setting),
                "seed": int(args.seed),
                "device": str(args.device),
                "outputs_root": str(args.outputs_root),
                "source_run_dir": str(source_run_dir),
                "unconstrained_run_dir": str(unconstrained_dir),
                "ewc_run_dir": str(ewc_dir),
                "rashomon_run_dir": str(rashomon_run_dir),
                "run_subdir": str(args.run_subdir),
                "env_id": env_id,
                "continuous": bool(continuous),
                "source_gravity": source_gravity,
                "downstream_gravity": downstream_gravity,
                "source_dynamics": source_dynamics,
                "downstream_dynamics": downstream_dynamics,
                "source_task_id": float(source_task_id),
                "downstream_task_id": float(downstream_task_id),
                "append_task_id": bool(append_task_id),
                "analysis_top_k": int(args.analysis_top_k),
                "eval_episodes_during_training": int(eval_episodes_during_training),
                "eval_episodes_post_training": int(eval_episodes_post_training),
                "task_settings_file": str(args.task_settings_file),
                "adapt_settings_file": str(args.adapt_settings_file),
                "rashomon_settings_file": str(args.rashomon_settings_file),
                "expansion_skipped_due_to_perfect_downstream_success": True,
                "old_rashomon_run_summary_path": (
                    str(old_run_summary_path) if old_run_summary_path.exists() else None
                ),
            },
            "analysis": {
                "pairwise_l2_matrix": pairwise_l2_matrix,
                "pairwise_l2_pairs_sorted_desc": pairwise_l2_pairs,
                "outside_old_rashomon": {
                    "downstream_unconstrained": _outside_analysis_for_summary(old_outside_unconstrained),
                    "downstream_ewc": _outside_analysis_for_summary(old_outside_ewc),
                },
                "outside_union_rashomon": {
                    "downstream_unconstrained": _outside_analysis_for_summary(old_outside_unconstrained),
                    "downstream_ewc": _outside_analysis_for_summary(old_outside_ewc),
                },
                "bounds_width_summary": {
                    "old": old_width_summary,
                    "new": None,
                    "union_interval_set": union_interval_width_summary,
                },
            },
            "run_results": {
                "new_rashomon_dataset_size": None,
                "new_rashomon_surrogate_threshold": None,
                "new_rashomon_inverse_temperature": None,
                "new_rashomon_selected_certificate_index": None,
                "new_rashomon_selected_certificate": None,
                "new_rashomon_all_certificates": None,
                "source_mean_reward": float(existing_source_mean_reward),
                "source_std_reward": float(existing_source_std_reward),
                "source_failure_rate": float(existing_source_failure_rate),
                "source_success_rate": float(existing_source_success_rate),
                "downstream_mean_reward": float(existing_downstream_mean_reward),
                "downstream_std_reward": float(existing_downstream_std_reward),
                "downstream_failure_rate": float(existing_downstream_failure_rate),
                "downstream_success_rate": float(existing_downstream_success_rate),
                "adapted_outside_union_rashomon": _outside_analysis_for_summary(adapted_outside_old),
                "adapted_outside_old_rashomon": _outside_analysis_for_summary(adapted_outside_old),
            },
            "artifacts": {
                "actor_path": (
                    str(actor_out_path) if actor_out_path.exists() else str(actors["downstream_rashomon"]["actor_path"])
                ),
                "critic_path": (
                    str(critic_out_path)
                    if critic_out_path.exists()
                    else str(rashomon_run_dir / "critic.pt")
                ),
                "training_data_path": (
                    str(training_data_out_path)
                    if training_data_out_path.exists()
                    else str(rashomon_run_dir / "training_data.pt")
                ),
                "rashomon_dataset_path": (
                    str(copied_rashomon_dataset_out_path)
                    if copied_rashomon_dataset_out_path.exists()
                    else str(rashomon_dataset_path)
                ),
                "rashomon_bounded_model_path": (
                    str(copied_bounded_model_out_path)
                    if copied_bounded_model_out_path.exists()
                    else str(old_bounded_model_path)
                ),
                "rashomon_param_bounds_path": (
                    str(copied_bounds_out_path) if copied_bounds_out_path.exists() else None
                ),
                "rashomon_rollout_stats_path": (
                    str(copied_rollout_stats_out_path) if copied_rollout_stats_out_path.exists() else None
                ),
                "pairwise_l2_matrix_csv": str(pairwise_matrix_csv),
                "pairwise_l2_pairs_csv": str(pairwise_pairs_csv),
                "outside_old_unconstrained_csv": str(outside_old_unconstrained_csv),
                "outside_old_ewc_csv": str(outside_old_ewc_csv),
                "outside_union_unconstrained_csv": str(outside_union_unconstrained_csv),
                "outside_union_ewc_csv": str(outside_union_ewc_csv),
                "outside_union_adapted_csv": str(outside_union_adapted_csv),
                "copied_from_rashomon_artifacts": copied_artifact_names,
                "original_rashomon_bounded_model_path": str(old_bounded_model_path),
                "original_rashomon_dataset_path": str(rashomon_dataset_path),
            },
        }
        summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

        print(
            "\nExisting Rashomon actor evaluation (saved without expansion):\n"
            f"- Source    ({eval_episodes_post_training} ep): mean={existing_source_mean_reward:.3f}, "
            f"std={existing_source_std_reward:.3f}, failure_rate={existing_source_failure_rate:.3f}, "
            f"success_rate={existing_source_success_rate:.3f}\n"
            f"- Downstream({eval_episodes_post_training} ep): mean={existing_downstream_mean_reward:.3f}, "
            f"std={existing_downstream_std_reward:.3f}, failure_rate={existing_downstream_failure_rate:.3f}, "
            f"success_rate={existing_downstream_success_rate:.3f}",
        )
        print(f"\nSaved copied Rashomon artifacts to: {run_dir}")
        print(f"Saved summary: {summary_path}")
        return

    # Step 4: sample lower/upper per-parameter and build a new reference actor.
    sample_seed = int(args.bound_sample_seed) if args.bound_sample_seed is not None else int(args.seed)
    sampled_param_list, sample_stats = _sample_param_list_from_bounds(
        bounds_l=old_bounds_l,
        bounds_u=old_bounds_u,
        upper_prob=float(args.upper_bound_prob),
        seed=sample_seed,
    )
    reference_actor, sampled_reference_state_dict = _build_reference_actor_from_sampled_params(
        source_actor_for_template=actors["source"]["actor"],
        sampled_param_list=sampled_param_list,
        device=args.device,
    )
    print(
        "\nSampled reference actor from old Rashomon bounds: "
        f"upper_fraction={sample_stats['upper_fraction']:.4f}",
    )

    # Step 5: compute a new Rashomon set from sampled reference actor.
    rashomon_settings = _load_yaml(args.rashomon_settings_file)
    rashomon_cfg = _resolve_setting_cfg(
        rashomon_settings,
        args.task_setting,
        settings_name=str(args.rashomon_settings_file),
    )
    rashomon_cfg_raw = rashomon_cfg.get("rashomon", rashomon_cfg)
    if not isinstance(rashomon_cfg_raw, dict):
        raise ValueError(
            f"Expected a mapping for rashomon config in {args.rashomon_settings_file} "
            f"for setting '{args.task_setting}'.",
        )

    rashomon_n_iters = int(
        args.rashomon_n_iters
        if args.rashomon_n_iters is not None
        else old_summary_run_settings.get(
            "rashomon_n_iters",
            rashomon_cfg_raw.get("rashomon_n_iters", 50_000),
        ),
    )
    rashomon_min_hard_spec = float(
        args.rashomon_min_hard_spec
        if args.rashomon_min_hard_spec is not None
        else rashomon_cfg_raw.get("rashomon_min_hard_spec", 1.0),
    )
    rashomon_surrogate_aggregation = str(
        args.rashomon_surrogate_aggregation
        if args.rashomon_surrogate_aggregation is not None
        else old_summary_run_settings.get(
            "surrogate_aggregation",
            rashomon_cfg_raw.get("rashomon_surrogate_aggregation", "min"),
        ),
    )
    inverse_temp_start = int(
        args.inverse_temp_start
        if args.inverse_temp_start is not None
        else old_summary_run_settings.get(
            "inverse_temp_start",
            rashomon_cfg_raw.get("inverse_temp_start", 10),
        ),
    )
    inverse_temp_max = int(
        args.inverse_temp_max
        if args.inverse_temp_max is not None
        else old_summary_run_settings.get(
            "inverse_temp_max",
            rashomon_cfg_raw.get("inverse_temp_max", 1000),
        ),
    )
    rashomon_checkpoint = int(
        args.rashomon_checkpoint
        if args.rashomon_checkpoint is not None
        else old_summary_run_settings.get(
            "rashomon_checkpoint",
            rashomon_cfg_raw.get("rashomon_checkpoint", 100),
        ),
    )

    if inverse_temp_start <= 0 or inverse_temp_max < inverse_temp_start:
        raise ValueError(
            "Invalid inverse-temperature range for Rashomon recompute. "
            "Require 0 < inverse_temp_start <= inverse_temp_max.",
        )

    (
        new_bounds_l,
        new_bounds_u,
        new_bounded_model,
        new_selected_inverse_temp,
        new_surrogate_threshold,
        new_cert_values,
        new_selected_cert_idx,
    ) = compute_rashomon_bounds(
        actor=copy.deepcopy(reference_actor).to("cpu"),
        rashomon_dataset=rashomon_dataset,
        seed=args.seed,
        rashomon_n_iters=rashomon_n_iters,
        min_hard_spec=rashomon_min_hard_spec,
        aggregation=rashomon_surrogate_aggregation,
        inverse_temp_start=inverse_temp_start,
        inverse_temp_max=inverse_temp_max,
        checkpoint=rashomon_checkpoint,
    )
    print(
        "\nRecomputed new Rashomon set from sampled reference actor: "
        f"selected_cert={new_cert_values[new_selected_cert_idx]:.6f}, "
        f"inverse_temp={new_selected_inverse_temp}",
    )

    # Step 6: union interval set (without collapsing to min/max envelopes).
    # We keep both interval families as separate admissible intervals per parameter.
    union_interval_bounds_l = [old_bounds_l, new_bounds_l]
    union_interval_bounds_u = [old_bounds_u, new_bounds_u]

    old_width_summary = _summarize_bound_widths(bounds_l=old_bounds_l, bounds_u=old_bounds_u)
    new_width_summary = _summarize_bound_widths(bounds_l=new_bounds_l, bounds_u=new_bounds_u)
    union_interval_width_summary = {
        "n_intervals": 2,
        "per_interval": [
            {
                "name": "old",
                **old_width_summary,
            },
            {
                "name": "new",
                **new_width_summary,
            },
        ],
    }

    union_outside_unconstrained = _analyze_distance_to_rashomon_interval(
        actor_name="downstream_unconstrained",
        actor_param_list=actor_parameters_dct["downstream_unconstrained"],
        bounds_l_intervals=union_interval_bounds_l,
        bounds_u_intervals=union_interval_bounds_u,
        top_k=args.analysis_top_k,
    )
    union_outside_ewc = _analyze_distance_to_rashomon_interval(
        actor_name="downstream_ewc",
        actor_param_list=actor_parameters_dct["downstream_ewc"],
        bounds_l_intervals=union_interval_bounds_l,
        bounds_u_intervals=union_interval_bounds_u,
        top_k=args.analysis_top_k,
    )

    print(
        "\nOutside-union-Rashomon summary:\n"
        f"- downstream_unconstrained: outside={union_outside_unconstrained['n_outside']} "
        f"| L2-distance={union_outside_unconstrained['distance_l2']:.6f}\n"
        f"- downstream_ewc: outside={union_outside_ewc['n_outside']} "
        f"| L2-distance={union_outside_ewc['distance_l2']:.6f}",
    )

    # Build adaptation PPO config.
    total_timesteps = int(
        args.total_timesteps_override
        if args.total_timesteps_override is not None
        else adapt_ppo_cfg.get("total_timesteps", 200_000),
    )
    if total_timesteps <= 0:
        raise ValueError("--total-timesteps-override (or resolved total_timesteps) must be > 0.")

    early_stop_reward_threshold_cfg = adapt_ppo_cfg.get("early_stop_reward_threshold", None)
    early_stop_reward_threshold = (
        float(early_stop_reward_threshold_cfg)
        if early_stop_reward_threshold_cfg is not None
        else None
    )

    ppo_cfg = PPOConfig(
        seed=int(adapt_ppo_cfg.get("seed", args.seed)),
        total_timesteps=total_timesteps,
        eval_episodes=eval_episodes_during_training,
        rollout_steps=int(adapt_ppo_cfg.get("rollout_steps", 2048)),
        update_epochs=int(adapt_ppo_cfg.get("update_epochs", 10)),
        minibatch_size=int(adapt_ppo_cfg.get("minibatch_size", 256)),
        gamma=float(adapt_ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(adapt_ppo_cfg.get("gae_lambda", 0.95)),
        clip_coef=float(adapt_ppo_cfg.get("clip_coef", 0.2)),
        ent_coef=float(adapt_ppo_cfg.get("ent_coef", 0.01)),
        vf_coef=float(adapt_ppo_cfg.get("vf_coef", 0.5)),
        lr=float(adapt_ppo_cfg.get("lr", 3e-4)),
        max_grad_norm=float(adapt_ppo_cfg.get("max_grad_norm", 0.5)),
        device=str(args.device),
        early_stop_min_steps=int(adapt_ppo_cfg.get("early_stop_min_steps", 0)),
        early_stop_reward_threshold=early_stop_reward_threshold,
        early_stop_failure_rate_threshold=adapt_ppo_cfg.get("early_stop_failure_rate_threshold", None),
        early_stop_success_rate_threshold=adapt_ppo_cfg.get("early_stop_success_rate_threshold", None),
    )

    # Build source actor/critic warm starts for PGD adaptation.
    source_actor_ckpt = source_run_dir / "actor.pt"
    source_critic_ckpt = source_run_dir / "critic.pt"
    if not source_actor_ckpt.exists():
        raise FileNotFoundError(f"Source actor checkpoint not found: {source_actor_ckpt}")
    if args.warm_start_critic and not source_critic_ckpt.exists():
        raise FileNotFoundError(f"Source critic checkpoint not found: {source_critic_ckpt}")

    hidden_size = _load_source_hidden_size(source_run_dir, args.hidden_size)
    source_env_for_dim = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **source_env_kwargs,
    )
    if not isinstance(source_env_for_dim.action_space, gym.spaces.Discrete):
        raise ValueError("Expected discrete action space for LunarLander.")
    obs_dim = int(source_env_for_dim.observation_space.shape[0])  # type: ignore[index]
    n_actions = int(source_env_for_dim.action_space.n)  # type: ignore[union-attr]
    source_env_for_dim.close()

    source_actor_for_adapt, source_critic_for_adapt = build_actor_critic(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_size=hidden_size,
    )
    source_actor_for_adapt.load_state_dict(torch.load(source_actor_ckpt, map_location="cpu"))
    if args.warm_start_critic:
        source_critic_for_adapt.load_state_dict(torch.load(source_critic_ckpt, map_location="cpu"))

    pre_transform_cfg = adapt_cfg.get("pre_adaptation_transform", {})
    if not isinstance(pre_transform_cfg, dict):
        pre_transform_cfg = {}
    default_neutralization = bool(pre_transform_cfg.get("task_feature_neutralization", False))
    do_task_neutralization = (
        bool(args.task_neutralization)
        if args.task_neutralization is not None
        else default_neutralization
    )
    do_task_neutralization = bool(do_task_neutralization and append_task_id)
    task_feature_index = int(pre_transform_cfg.get("task_feature_index", obs_dim - 1))
    if do_task_neutralization:
        neutralize_task_feature(source_actor_for_adapt, task_feature_index, downstream_task_id)
        if args.warm_start_critic:
            neutralize_task_feature(source_critic_for_adapt, task_feature_index, downstream_task_id)

    print(
        "\nStarting PGD adaptation on union Rashomon interval-set bounds: "
        f"total_timesteps={ppo_cfg.total_timesteps}, "
        f"eval_episodes_during_training={ppo_cfg.eval_episodes}, "
        f"eval_episodes_post_training={eval_episodes_post_training}",
    )

    # Step 7: downstream adaptation with PGD on union bounds.
    train_env = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **downstream_env_kwargs,
    )
    early_stop_eval_env = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **downstream_env_kwargs,
    )
    try:
        adapted_actor, adapted_critic, training_data = ppo_train(  # type: ignore[assignment]
            train_env,
            ppo_cfg,
            actor_warm_start=source_actor_for_adapt,
            critic_warm_start=(source_critic_for_adapt if args.warm_start_critic else None),
            actor_param_bounds_l=union_interval_bounds_l,
            actor_param_bounds_u=union_interval_bounds_u,
            early_stop_eval_env=early_stop_eval_env,
            return_training_data=True,
        )
    finally:
        train_env.close()
        early_stop_eval_env.close()

    # Step 8: evaluate adapted actor on source and downstream.
    source_eval_env = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **source_env_kwargs,
    )
    source_mean_reward, source_std_reward, source_failure_rate, source_success_rate = evaluate_with_success(
        source_eval_env,
        adapted_actor,
        episodes=eval_episodes_post_training,
        deterministic=True,
        device=args.device,
    )
    source_eval_env.close()

    downstream_eval_env = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **downstream_env_kwargs,
    )
    (
        downstream_mean_reward,
        downstream_std_reward,
        downstream_failure_rate,
        downstream_success_rate,
    ) = evaluate_with_success(
        downstream_eval_env,
        adapted_actor,
        episodes=eval_episodes_post_training,
        deterministic=True,
        device=args.device,
    )
    downstream_eval_env.close()

    adapted_actor_param_list = list(adapted_actor.parameters())
    adapted_outside_union = _analyze_distance_to_rashomon_interval(
        actor_name="downstream_rashomon_union_expanded",
        actor_param_list=adapted_actor_param_list,
        bounds_l_intervals=union_interval_bounds_l,
        bounds_u_intervals=union_interval_bounds_u,
        top_k=args.analysis_top_k,
    )
    adapted_outside_old = _analyze_distance_to_rashomon_interval(
        actor_name="downstream_rashomon_union_expanded",
        actor_param_list=adapted_actor_param_list,
        bounds_l_intervals=[old_bounds_l],
        bounds_u_intervals=[old_bounds_u],
        top_k=args.analysis_top_k,
    )

    # Persist outputs.
    run_dir = _seed_run_dir(args.outputs_root, args.task_setting, args.seed) / args.run_subdir
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    actor_out_path = run_dir / "actor.pt"
    critic_out_path = run_dir / "critic.pt"
    training_data_out_path = run_dir / "training_data.pt"
    sampled_reference_actor_out_path = run_dir / "sampled_reference_actor.pt"
    copied_rashomon_dataset_out_path = run_dir / "rashomon_dataset.pt"
    new_bounded_model_out_path = run_dir / "new_rashomon_bounded_model.pt"
    union_interval_bounds_out_path = run_dir / "rashomon_union_interval_param_bounds.pt"

    pairwise_matrix_csv = analysis_dir / "actor_pairwise_l2_matrix.csv"
    pairwise_pairs_csv = analysis_dir / "actor_pairwise_l2_pairs.csv"
    outside_old_unconstrained_csv = analysis_dir / "outside_old_rashomon_downstream_unconstrained.csv"
    outside_old_ewc_csv = analysis_dir / "outside_old_rashomon_downstream_ewc.csv"
    outside_union_unconstrained_csv = analysis_dir / "outside_union_rashomon_downstream_unconstrained.csv"
    outside_union_ewc_csv = analysis_dir / "outside_union_rashomon_downstream_ewc.csv"
    outside_union_adapted_csv = analysis_dir / "outside_union_rashomon_adapted_actor.csv"
    summary_path = run_dir / "run_summary.yaml"

    torch.save(adapted_actor.state_dict(), actor_out_path)
    torch.save(adapted_critic.state_dict(), critic_out_path)
    torch.save(training_data, training_data_out_path)
    torch.save(sampled_reference_state_dict, sampled_reference_actor_out_path)
    torch.save(rashomon_dataset, copied_rashomon_dataset_out_path)
    torch.save(new_bounded_model, new_bounded_model_out_path)
    torch.save(
        {
            "old_param_bounds_l": old_bounds_l,
            "old_param_bounds_u": old_bounds_u,
            "new_param_bounds_l": new_bounds_l,
            "new_param_bounds_u": new_bounds_u,
            "union_interval_param_bounds_l": union_interval_bounds_l,
            "union_interval_param_bounds_u": union_interval_bounds_u,
        },
        union_interval_bounds_out_path,
    )

    _write_pairwise_matrix_csv(matrix=pairwise_l2_matrix, out_path=pairwise_matrix_csv)
    _write_pairwise_pairs_csv(pairs=pairwise_l2_pairs, out_path=pairwise_pairs_csv)
    _write_outside_entries_csv(
        entries=old_outside_unconstrained["outside_entries"],
        out_path=outside_old_unconstrained_csv,
    )
    _write_outside_entries_csv(
        entries=old_outside_ewc["outside_entries"],
        out_path=outside_old_ewc_csv,
    )
    _write_outside_entries_csv(
        entries=union_outside_unconstrained["outside_entries"],
        out_path=outside_union_unconstrained_csv,
    )
    _write_outside_entries_csv(
        entries=union_outside_ewc["outside_entries"],
        out_path=outside_union_ewc_csv,
    )
    _write_outside_entries_csv(
        entries=adapted_outside_union["outside_entries"],
        out_path=outside_union_adapted_csv,
    )

    recompute_rashomon_cfg = {
        "rashomon_n_iters": int(rashomon_n_iters),
        "rashomon_min_hard_spec": float(rashomon_min_hard_spec),
        "rashomon_surrogate_aggregation": str(rashomon_surrogate_aggregation),
        "inverse_temp_start": int(inverse_temp_start),
        "inverse_temp_max": int(inverse_temp_max),
        "rashomon_checkpoint": int(rashomon_checkpoint),
    }

    summary = {
        "run_settings": {
            "task_setting": str(args.task_setting),
            "seed": int(args.seed),
            "device": str(args.device),
            "outputs_root": str(args.outputs_root),
            "source_run_dir": str(source_run_dir),
            "unconstrained_run_dir": str(unconstrained_dir),
            "ewc_run_dir": str(ewc_dir),
            "rashomon_run_dir": str(rashomon_run_dir),
            "run_subdir": str(args.run_subdir),
            "env_id": env_id,
            "continuous": bool(continuous),
            "source_gravity": source_gravity,
            "downstream_gravity": downstream_gravity,
            "source_dynamics": source_dynamics,
            "downstream_dynamics": downstream_dynamics,
            "source_task_id": float(source_task_id),
            "downstream_task_id": float(downstream_task_id),
            "append_task_id": bool(append_task_id),
            "task_neutralization": bool(do_task_neutralization),
            "task_feature_index": int(task_feature_index) if do_task_neutralization else None,
            "warm_start_critic": bool(args.warm_start_critic),
            "hidden_size": int(hidden_size),
            "analysis_top_k": int(args.analysis_top_k),
            "bound_sampling": sample_stats,
            "union_interval_count": int(len(union_interval_bounds_l)),
            "eval_episodes_during_training": int(eval_episodes_during_training),
            "eval_episodes_post_training": int(eval_episodes_post_training),
            "ppo_config": asdict(ppo_cfg),
            "task_settings_file": str(args.task_settings_file),
            "adapt_settings_file": str(args.adapt_settings_file),
            "rashomon_settings_file": str(args.rashomon_settings_file),
            "recompute_rashomon_config": recompute_rashomon_cfg,
            "old_rashomon_run_summary_path": (
                str(old_run_summary_path) if old_run_summary_path.exists() else None
            ),
        },
        "analysis": {
            "pairwise_l2_matrix": pairwise_l2_matrix,
            "pairwise_l2_pairs_sorted_desc": pairwise_l2_pairs,
            "outside_old_rashomon": {
                "downstream_unconstrained": _outside_analysis_for_summary(old_outside_unconstrained),
                "downstream_ewc": _outside_analysis_for_summary(old_outside_ewc),
            },
            "outside_union_rashomon": {
                "downstream_unconstrained": _outside_analysis_for_summary(union_outside_unconstrained),
                "downstream_ewc": _outside_analysis_for_summary(union_outside_ewc),
            },
            "bounds_width_summary": {
                "old": old_width_summary,
                "new": new_width_summary,
                "union_interval_set": union_interval_width_summary,
            },
        },
        "run_results": {
            "new_rashomon_dataset_size": int(len(rashomon_dataset)),
            "new_rashomon_surrogate_threshold": float(new_surrogate_threshold),
            "new_rashomon_inverse_temperature": int(new_selected_inverse_temp),
            "new_rashomon_selected_certificate_index": int(new_selected_cert_idx),
            "new_rashomon_selected_certificate": float(new_cert_values[new_selected_cert_idx]),
            "new_rashomon_all_certificates": [float(v) for v in new_cert_values],
            "source_mean_reward": float(source_mean_reward),
            "source_std_reward": float(source_std_reward),
            "source_failure_rate": float(source_failure_rate),
            "source_success_rate": float(source_success_rate),
            "downstream_mean_reward": float(downstream_mean_reward),
            "downstream_std_reward": float(downstream_std_reward),
            "downstream_failure_rate": float(downstream_failure_rate),
            "downstream_success_rate": float(downstream_success_rate),
            "adapted_outside_union_rashomon": _outside_analysis_for_summary(adapted_outside_union),
            "adapted_outside_old_rashomon": _outside_analysis_for_summary(adapted_outside_old),
        },
        "artifacts": {
            "actor_path": str(actor_out_path),
            "critic_path": str(critic_out_path),
            "training_data_path": str(training_data_out_path),
            "sampled_reference_actor_path": str(sampled_reference_actor_out_path),
            "rashomon_dataset_path": str(copied_rashomon_dataset_out_path),
            "new_rashomon_bounded_model_path": str(new_bounded_model_out_path),
            "union_interval_param_bounds_path": str(union_interval_bounds_out_path),
            "pairwise_l2_matrix_csv": str(pairwise_matrix_csv),
            "pairwise_l2_pairs_csv": str(pairwise_pairs_csv),
            "outside_old_unconstrained_csv": str(outside_old_unconstrained_csv),
            "outside_old_ewc_csv": str(outside_old_ewc_csv),
            "outside_union_unconstrained_csv": str(outside_union_unconstrained_csv),
            "outside_union_ewc_csv": str(outside_union_ewc_csv),
            "outside_union_adapted_csv": str(outside_union_adapted_csv),
            "original_rashomon_bounded_model_path": str(old_bounded_model_path),
            "original_rashomon_dataset_path": str(rashomon_dataset_path),
        },
    }
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    print(
        "\nAdapted actor evaluation:\n"
        f"- Source    ({eval_episodes_post_training} ep): mean={source_mean_reward:.3f}, "
        f"std={source_std_reward:.3f}, failure_rate={source_failure_rate:.3f}, "
        f"success_rate={source_success_rate:.3f}\n"
        f"- Downstream({eval_episodes_post_training} ep): mean={downstream_mean_reward:.3f}, "
        f"std={downstream_std_reward:.3f}, failure_rate={downstream_failure_rate:.3f}, "
        f"success_rate={downstream_success_rate:.3f}",
    )
    print(f"\nSaved expanded-Rashomon run to: {run_dir}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
