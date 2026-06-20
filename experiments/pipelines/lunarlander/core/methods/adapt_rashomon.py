"""Adapt source LunarLander policy to downstream task via Rashomon-constrained PPO."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
from typing import Any

os.environ["SDL_AUDIODRIVER"] = "dummy"

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import TensorDataset
import yaml

# Allow running this file directly from experiments/pipelines/lunarlander.
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.lunarlander.core.env.env_factory import (
    _make_lunarlander_env,
)
from experiments.pipelines.lunarlander.core.env.task_loading import (
    _load_task_settings,
    _resolve_lunarlander_dynamics,
)
from experiments.pipelines.lunarlander.core.methods.source_train import (
    _plot_trajectory_grid,
    build_actor_critic,
)
from experiments.pipelines.lunarlander.core.orchestration.run_paths import (
    default_adapt_ppo_settings_file,
    default_adapt_rashomon_settings_file,
    default_outputs_root,
    default_task_settings_file,
    resolve_default_source_run_dir as _resolve_default_source_run_dir,
    seed_run_dir as _seed_run_dir,
)
from experiments.utils.ppo_utils import PPOConfig, evaluate_with_success, ppo_train
from src.rashomon_spec import AccuracyRequirement
from src.trainer.IntervalTrainer import IntervalTrainer


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


def _load_source_hidden_size(source_run_dir: Path, arg_hidden_size: int | None) -> int:
    if arg_hidden_size is not None:
        return int(arg_hidden_size)
    summary_path = source_run_dir / "run_summary.yaml"
    if summary_path.exists():
        summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
        if isinstance(summary, dict):
            if summary.get("hidden_size") is not None:
                return int(summary["hidden_size"])
            run_settings = summary.get("run_settings")
            if isinstance(run_settings, dict) and run_settings.get("hidden_size") is not None:
                return int(run_settings["hidden_size"])
    return 256


def neutralize_task_feature(
    model: torch.nn.Sequential,
    task_feature_index: int,
    target_task_value: float,
) -> None:
    """Neutralize first-layer task feature contribution for target task value."""
    first = model[0]
    if not isinstance(first, torch.nn.Linear):
        raise ValueError("Expected first layer to be torch.nn.Linear for task-feature neutralization.")

    with torch.no_grad():
        w_task = first.weight[:, task_feature_index].clone()
        first.bias[:] = first.bias - w_task * target_task_value
        first.weight[:, task_feature_index] = 0.0


def create_source_rollout_rashomon_dataset(
    actor: torch.nn.Module,
    env,
    *,
    seed: int,
    n_actions: int,
    rashomon_rollouts: int,
) -> tuple[TensorDataset, list[int]]:
    """Roll out the NoAdapt policy `rashomon_rollouts` times and collect state-action pairs."""
    if rashomon_rollouts <= 0:
        raise ValueError(f"rashomon_rollouts must be > 0, got {rashomon_rollouts}.")

    actor_was_training = actor.training
    actor_device = next(actor.parameters()).device
    actor.eval()

    obs_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    rollout_lengths: list[int] = []

    try:
        for rollout_idx in range(rashomon_rollouts):
            obs, _ = env.reset(seed=seed + rollout_idx)
            done = False

            rollout_obs: list[np.ndarray] = []
            rollout_actions: list[int] = []

            while not done:
                obs_np = np.asarray(obs, dtype=np.float32).copy()
                rollout_obs.append(obs_np)

                obs_t = torch.from_numpy(obs_np).to(actor_device).unsqueeze(0)
                with torch.no_grad():
                    logits = actor(obs_t)
                    action = int(torch.argmax(logits, dim=1).item())
                rollout_actions.append(action)

                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            if not rollout_obs:
                continue

            rollout_obs_arr = np.asarray(rollout_obs, dtype=np.float32)
            rollout_labels_arr = np.zeros((len(rollout_actions), n_actions), dtype=np.float32)
            rollout_labels_arr[np.arange(len(rollout_actions)), np.asarray(rollout_actions, dtype=np.int64)] = 1.0

            obs_chunks.append(rollout_obs_arr)
            label_chunks.append(rollout_labels_arr)
            rollout_lengths.append(len(rollout_actions))
    finally:
        if actor_was_training:
            actor.train()

    if not obs_chunks:
        raise RuntimeError("No source rollouts were collected; cannot build Rashomon dataset.")

    obs_tensor = torch.from_numpy(np.concatenate(obs_chunks, axis=0)).float()
    label_tensor = torch.from_numpy(np.concatenate(label_chunks, axis=0)).float()
    return TensorDataset(obs_tensor, label_tensor), rollout_lengths


def _compute_surrogate_threshold_from_dataset(rashomon_dataset: TensorDataset) -> float:
    if len(rashomon_dataset) == 0:
        raise RuntimeError("Rashomon dataset is empty.")

    n_valid_actions = rashomon_dataset.tensors[1].sum(dim=1).tolist()
    max_valid_actions = max(n_valid_actions) if n_valid_actions else 0.0
    if max_valid_actions <= 0:
        raise RuntimeError("Rashomon dataset has no valid-action labels.")
    return float(max_valid_actions / (1.0 + max_valid_actions))


def _assert_actor_satisfies_surrogate_constraint(
    *,
    actor: torch.nn.Module,
    rashomon_dataset: TensorDataset,
    inverse_temp: int,
    surrogate_threshold: float,
    context: str,
    tol: float = 1e-8,
) -> float:
    """Assert actor satisfies the surrogate valid-action-mass constraint."""
    if inverse_temp <= 0:
        raise ValueError(f"inverse_temp must be > 0, got {inverse_temp}.")

    actor.eval()
    with torch.no_grad():
        obs = rashomon_dataset.tensors[0]
        action_mask = rashomon_dataset.tensors[1]
        actor_device = next(actor.parameters()).device
        logits = actor(obs.to(actor_device))
        probs = torch.softmax(logits * int(inverse_temp), dim=1)
        valid_action_mass = (probs * action_mask.to(actor_device)).sum(dim=1)
        min_valid_action_mass = float(valid_action_mass.min().item())

    if min_valid_action_mass < float(surrogate_threshold) - float(tol):
        raise AssertionError(
            f"{context}: surrogate constraint failed for inverse_temp={inverse_temp}. "
            f"min_valid_action_mass={min_valid_action_mass:.6f} < "
            f"surrogate_threshold={float(surrogate_threshold):.6f} (tol={tol:.2e}).",
        )
    return min_valid_action_mass


def compute_rashomon_bounds(
    *,
    actor: torch.nn.Module,
    rashomon_dataset: TensorDataset,
    seed: int,
    rashomon_n_iters: int,
    min_hard_spec: float,
    aggregation: str,
    inverse_temp_start: int,
    inverse_temp_max: int,
    checkpoint: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], object, int, float, list[float], int]:
    """Compute Rashomon parameter bounds for a state-action dataset."""
    surrogate_threshold = _compute_surrogate_threshold_from_dataset(rashomon_dataset)

    actor.eval()
    with torch.no_grad():
        all_obs = rashomon_dataset.tensors[0]
        action_mask = rashomon_dataset.tensors[1]
        logits = actor(all_obs)

        selected_inverse_temp: int | None = None
        min_action_mass = float("-inf")
        for inverse_temp in range(inverse_temp_start, inverse_temp_max + 1):
            probs = torch.softmax(logits * inverse_temp, dim=1)
            valid_action_mass = (probs * action_mask).sum(dim=1)
            min_action_mass = float(valid_action_mass.min().item())
            if min_action_mass >= surrogate_threshold:
                selected_inverse_temp = inverse_temp
                break

        if selected_inverse_temp is None:
            raise ValueError(
                "Could not find inverse temperature satisfying surrogate threshold. "
                f"Best min valid-action mass={min_action_mass:.6f} < threshold={surrogate_threshold:.6f}",
            )

    interval_trainer = IntervalTrainer(
        model=actor,
        accuracy=AccuracyRequirement(
            soft_min=surrogate_threshold,
            hard_min=min_hard_spec,
            soft_temperature=selected_inverse_temp,
            aggregation=aggregation,  # type: ignore[arg-type]
        ),
        seed=seed,
        n_iters=rashomon_n_iters,  # type: ignore[arg-type]
        min_acc_increment=0,
        checkpoint=checkpoint,  # type: ignore[arg-type]
    )
    interval_trainer.compute_rashomon_set(
        dataset=rashomon_dataset,
        multi_label=True,
    )

    cert_values = [
        min((c.min_hard_acc for c in certs), default=float("-inf"))
        for certs in interval_trainer.certificates
    ]
    valid_indices = [i for i, cert in enumerate(cert_values) if cert >= min_hard_spec]
    if not valid_indices:
        best_cert = max(cert_values) if cert_values else float("-inf")
        raise ValueError(
            f"No Rashomon certificate satisfied min_hard_spec={min_hard_spec:.3f}. "
            f"Best certificate={best_cert:.6f}",
        )

    selected_idx = valid_indices[-1]
    bounded_model = interval_trainer.bounds[selected_idx]
    param_bounds_l = [p.detach().cpu() for p in bounded_model.param_l]
    param_bounds_u = [p.detach().cpu() for p in bounded_model.param_u]
    return (
        param_bounds_l,
        param_bounds_u,
        bounded_model,
        selected_inverse_temp,
        surrogate_threshold,
        cert_values,
        selected_idx,
    )


def _sample_reference_network_from_bounds(
    *,
    reference_network: torch.nn.Module,
    bounds_l: list[torch.Tensor],
    bounds_u: list[torch.Tensor],
    seed: int,
    upper_prob: float = 0.5,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Sample a reference network by selecting per-parameter lower/upper bounds."""
    if len(bounds_l) != len(bounds_u):
        raise ValueError(
            f"Bounds length mismatch: lower={len(bounds_l)} upper={len(bounds_u)}.",
        )
    if not (0.0 <= upper_prob <= 1.0):
        raise ValueError(f"upper_prob must be in [0, 1], got {upper_prob}.")

    sampled_network = copy.deepcopy(reference_network)
    sampled_network.eval()
    sampled_params = list(sampled_network.parameters())
    if len(sampled_params) != len(bounds_l):
        raise ValueError(
            "Parameter count mismatch between reference network and Rashomon bounds. "
            f"params={len(sampled_params)} bounds={len(bounds_l)}.",
        )

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    n_total = 0
    n_upper = 0

    with torch.no_grad():
        for p_idx, (param, lower, upper) in enumerate(zip(sampled_params, bounds_l, bounds_u)):
            lower_cpu = lower.detach().cpu()
            upper_cpu = upper.detach().cpu()
            if tuple(lower_cpu.shape) != tuple(param.shape) or tuple(upper_cpu.shape) != tuple(param.shape):
                raise ValueError(
                    f"Bounds shape mismatch at parameter index {p_idx}: "
                    f"param={tuple(param.shape)}, lower={tuple(lower_cpu.shape)}, upper={tuple(upper_cpu.shape)}",
                )

            choose_upper = torch.rand(lower_cpu.shape, generator=gen) < upper_prob
            sampled_cpu = torch.where(choose_upper, upper_cpu, lower_cpu)
            param.copy_(sampled_cpu.to(device=param.device, dtype=param.dtype))

            n_total += int(choose_upper.numel())
            n_upper += int(choose_upper.sum().item())

    sample_stats = {
        "seed": int(seed),
        "upper_prob": float(upper_prob),
        "n_total": int(n_total),
        "n_upper": int(n_upper),
        "n_lower": int(n_total - n_upper),
        "upper_fraction": float(n_upper / n_total) if n_total > 0 else 0.0,
    }
    return sampled_network, sample_stats


def compute_nonconvex_rashomon_bounds(
    *,
    actor: torch.nn.Module,
    rashomon_dataset: TensorDataset,
    seed: int,
    n_iters: int,
    min_hard_spec: float,
    aggregation: str,
    inverse_temp_start: int,
    inverse_temp_max: int,
    checkpoint: int,
    n_convex_sets_budget: int,
    return_checkpoints: bool = False,
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]], list[dict[str, Any]] | None]:
    """Build a non-convex Rashomon set as a budgeted union of convex Rashomon sets.

    `n_iters` is the total iteration budget across all convex-set computations.
    Each convex Rashomon set receives `int(n_iters / n_convex_sets_budget)` iterations.

    Procedure:
      1. Build the first convex Rashomon set around the initial reference network.
      2. Fix inverse temperature to that first set's selected (minimum feasible) value.
      3. Build all subsequent convex sets with the same fixed inverse temperature.
      4. Sample each next reference network from lower/upper bounds (Bernoulli(0.5)).
      5. Assert each sampled reference satisfies the same surrogate constraint.
      6. Repeat until `n_convex_sets_budget` convex sets are built.

    Returns:
      (all_bounds_l, all_bounds_u, checkpoints_or_none)
      where `all_bounds_l[set_idx][param_idx]` and `all_bounds_u[set_idx][param_idx]`
      are the bounds for each convex Rashomon set in the union.
    """
    if n_convex_sets_budget <= 0:
        raise ValueError(f"n_convex_sets_budget must be > 0, got {n_convex_sets_budget}.")
    if n_iters <= 0:
        raise ValueError(f"n_iters must be > 0, got {n_iters}.")

    fixed_n_iters = int(n_iters / n_convex_sets_budget)
    if fixed_n_iters <= 0:
        raise ValueError(
            "n_iters / n_convex_sets_budget must provide at least one iteration per convex set; "
            f"got n_iters={n_iters}, n_convex_sets_budget={n_convex_sets_budget}.",
        )
    fixed_checkpoint = int(checkpoint)
    fixed_min_hard_spec = float(min_hard_spec)
    fixed_aggregation = str(aggregation)
    fixed_dataset = rashomon_dataset

    current_reference = copy.deepcopy(actor).to("cpu")
    current_reference.eval()

    all_bounds_l: list[list[torch.Tensor]] = []
    all_bounds_u: list[list[torch.Tensor]] = []
    checkpoints: list[dict[str, Any]] | None = [] if return_checkpoints else None
    fixed_inverse_temp: int | None = None
    fixed_surrogate_threshold: float | None = None

    for set_idx in range(int(n_convex_sets_budget)):
        if fixed_inverse_temp is None:
            run_inverse_temp_start = int(inverse_temp_start)
            run_inverse_temp_max = int(inverse_temp_max)
        else:
            run_inverse_temp_start = int(fixed_inverse_temp)
            run_inverse_temp_max = int(fixed_inverse_temp)
            # Sanity check: every new reference sampled from previous Rashomon set
            # must satisfy the same surrogate constraint under fixed inverse temperature.
            assert fixed_surrogate_threshold is not None
            _assert_actor_satisfies_surrogate_constraint(
                actor=current_reference,
                rashomon_dataset=fixed_dataset,
                inverse_temp=int(fixed_inverse_temp),
                surrogate_threshold=float(fixed_surrogate_threshold),
                context=f"reference_before_set_{set_idx}",
            )

        (
            bounds_l,
            bounds_u,
            bounded_model,
            selected_inverse_temp,
            surrogate_threshold,
            cert_values,
            selected_cert_idx,
        ) = compute_rashomon_bounds(
            actor=copy.deepcopy(current_reference).to("cpu"),
            rashomon_dataset=fixed_dataset,
            seed=int(seed + set_idx),
            rashomon_n_iters=fixed_n_iters,
            min_hard_spec=fixed_min_hard_spec,
            aggregation=fixed_aggregation,
            inverse_temp_start=run_inverse_temp_start,
            inverse_temp_max=run_inverse_temp_max,
            checkpoint=fixed_checkpoint,
        )

        if fixed_inverse_temp is None:
            # Freeze certification temperature after first set.
            fixed_inverse_temp = int(selected_inverse_temp)
            fixed_surrogate_threshold = float(surrogate_threshold)
        else:
            assert fixed_surrogate_threshold is not None
            if int(selected_inverse_temp) != int(fixed_inverse_temp):
                raise AssertionError(
                    "Inverse temperature changed across convex sets despite fixed-temperature mode. "
                    f"expected={fixed_inverse_temp}, got={selected_inverse_temp}.",
                )
            if abs(float(surrogate_threshold) - float(fixed_surrogate_threshold)) > 1e-12:
                raise AssertionError(
                    "Surrogate min-accuracy limit changed across convex sets. "
                    f"expected={fixed_surrogate_threshold:.12f}, got={float(surrogate_threshold):.12f}.",
                )

        set_bounds_l = [p.detach().cpu().clone() for p in bounds_l]
        set_bounds_u = [p.detach().cpu().clone() for p in bounds_u]
        all_bounds_l.append(set_bounds_l)
        all_bounds_u.append(set_bounds_u)

        if checkpoints is not None:
            checkpoints.append(
                {
                    "set_idx": int(set_idx),
                    "seed": int(seed + set_idx),
                    "n_iters": int(fixed_n_iters),
                    "total_n_iters_budget": int(n_iters),
                    "n_iters_per_convex_set": int(fixed_n_iters),
                    "n_convex_sets_budget": int(n_convex_sets_budget),
                    "selected_inverse_temperature": int(selected_inverse_temp),
                    "surrogate_threshold": float(surrogate_threshold),
                    "certificates": [float(v) for v in cert_values],
                    "selected_certificate_index": int(selected_cert_idx),
                    "selected_certificate": float(cert_values[selected_cert_idx]),
                    "bounded_model": bounded_model,
                    "certification_spec": {
                        "inverse_temperature": int(selected_inverse_temp),
                        "min_acc_limit": float(surrogate_threshold),
                        "min_hard_spec": float(fixed_min_hard_spec),
                        "aggregation": str(fixed_aggregation),
                    },
                },
            )

        if set_idx >= int(n_convex_sets_budget) - 1:
            continue

        sampled_reference, sample_stats = _sample_reference_network_from_bounds(
            reference_network=current_reference,
            bounds_l=set_bounds_l,
            bounds_u=set_bounds_u,
            seed=int(seed + n_convex_sets_budget + set_idx),
            upper_prob=0.5,
        )
        current_reference = sampled_reference.to("cpu")
        current_reference.eval()
        assert fixed_inverse_temp is not None
        assert fixed_surrogate_threshold is not None
        min_valid_action_mass = _assert_actor_satisfies_surrogate_constraint(
            actor=current_reference,
            rashomon_dataset=fixed_dataset,
            inverse_temp=int(fixed_inverse_temp),
            surrogate_threshold=float(fixed_surrogate_threshold),
            context=f"sampled_reference_after_set_{set_idx}",
        )
        if checkpoints is not None:
            checkpoints[-1]["next_reference_sampling"] = sample_stats
            checkpoints[-1]["next_reference_sampling"]["min_valid_action_mass"] = float(min_valid_action_mass)
            checkpoints[-1]["next_reference_sampling"]["surrogate_threshold"] = float(fixed_surrogate_threshold)
            checkpoints[-1]["next_reference_sampling"]["inverse_temperature"] = int(fixed_inverse_temp)

    return all_bounds_l, all_bounds_u, checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run downstream LunarLander adaptation with rollout Rashomon bounds and PPO-PGD.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device passed to PPO training.",
    )
    parser.add_argument(
        "--task-settings-file",
        type=Path,
        default=default_task_settings_file(),
        help="Task pipeline settings YAML (legacy monolithic task settings YAML is also supported).",
    )
    parser.add_argument(
        "--adapt-settings-file",
        type=Path,
        default=default_adapt_ppo_settings_file(),
        help="Shared downstream PPO settings YAML used for training/eval defaults.",
    )
    parser.add_argument(
        "--rashomon-settings-file",
        type=Path,
        default=default_adapt_rashomon_settings_file(),
        help="Rashomon-set construction settings YAML.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        dest="task_setting",
        default="default",
    )
    parser.add_argument("--task-setting", type=str, dest="task_setting", help=argparse.SUPPRESS)
    parser.add_argument("--env-id", type=str, default=None, help="Optional env-id override.")
    parser.add_argument("--source-gravity", type=float, default=None, help="Optional source gravity override.")
    parser.add_argument("--downstream-gravity", type=float, default=None, help="Optional downstream gravity override.")
    parser.add_argument(
        "--append-task-id",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append task-id feature in observations (default inherited from task settings).",
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help="Optional explicit NoAdapt checkpoint directory. Defaults to outputs/<pipeline>/seed_<seed>/noadapt",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=default_outputs_root(),
    )
    parser.add_argument(
        "--run-subdir",
        type=str,
        default="downstream_rashomon",
        help="Subdirectory under outputs/<pipeline>/seed_<seed>/ where outputs are saved.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Actor/critic hidden size. Defaults to source run summary hidden_size if available.",
    )
    parser.add_argument(
        "--warm-start-critic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm-start critic from source checkpoint.",
    )
    parser.add_argument(
        "--enable-task-neutralization",
        action="store_true",
        help="Enable first-layer task-feature neutralization before adaptation.",
    )
    parser.add_argument(
        "--disable-task-neutralization",
        action="store_true",
        help="Disable first-layer task-feature neutralization before adaptation.",
    )

    # PPO adaptation hyperparameters
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Optional override. Defaults to settings/adaptation/ppo.yaml for this pipeline.",
    )
    parser.add_argument(
        "--total-timesteps-override",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--eval-episodes-during-training",
        type=int,
        default=None,
        help="Optional override. Defaults to settings/adaptation/ppo.yaml for this pipeline.",
    )
    parser.add_argument(
        "--eval-episodes-post-training",
        type=int,
        default=None,
        help="Optional override. Defaults to settings/adaptation/ppo.yaml downstream_eval.episodes_post_training.",
    )
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--clip-coef", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--vf-coef", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--early-stop-min-steps", type=int, default=None)
    parser.add_argument("--early-stop-reward-threshold", type=float, default=None)
    parser.add_argument("--early-stop-failure-rate-threshold", type=float, default=None)
    parser.add_argument("--early-stop-success-rate-threshold", type=float, default=None)
    parser.add_argument(
        "--trajectory-episodes",
        type=int,
        default=5,
        help="Number of deterministic episodes visualized per trajectory figure.",
    )
    parser.add_argument(
        "--trajectory-max-frames-per-episode",
        type=int,
        default=5,
        help="Maximum frames shown per episode row (includes first and last frames).",
    )

    # Rashomon arguments
    parser.add_argument(
        "--rashomon-rollouts",
        type=int,
        default=None,
        help="Optional override. Defaults to settings/adaptation/rashomon.yaml for this pipeline.",
    )
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
    args = parser.parse_args()

    if args.enable_task_neutralization and args.disable_task_neutralization:
        raise ValueError("Cannot set both --enable-task-neutralization and --disable-task-neutralization.")
    if args.total_timesteps is not None and args.total_timesteps_override is not None:
        raise ValueError("Use only one of --total-timesteps and --total-timesteps-override.")

    source_task_cfg = _load_task_settings(args.task_settings_file, args.task_setting, "source")
    downstream_task_cfg = _load_task_settings(args.task_settings_file, args.task_setting, "downstream")

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

    rashomon_settings = _load_yaml(args.rashomon_settings_file)
    rashomon_setting_cfg = _resolve_setting_cfg(
        rashomon_settings,
        args.task_setting,
        settings_name=str(args.rashomon_settings_file),
    )
    rashomon_cfg = rashomon_setting_cfg.get("rashomon", {})
    if not isinstance(rashomon_cfg, dict):
        raise ValueError(
            f"Expected 'rashomon' mapping for setting '{args.task_setting}' in {args.rashomon_settings_file}.",
        )

    device = str(args.device)
    total_timesteps_arg = (
        args.total_timesteps
        if args.total_timesteps is not None
        else args.total_timesteps_override
    )
    total_timesteps = int(
        total_timesteps_arg
        if total_timesteps_arg is not None
        else adapt_ppo_cfg.get("total_timesteps", 200_000),
    )
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
    rollout_steps = int(
        args.rollout_steps
        if args.rollout_steps is not None
        else adapt_ppo_cfg.get("rollout_steps", 2048),
    )
    update_epochs = int(
        args.update_epochs
        if args.update_epochs is not None
        else adapt_ppo_cfg.get("update_epochs", 10),
    )
    minibatch_size = int(
        args.minibatch_size
        if args.minibatch_size is not None
        else adapt_ppo_cfg.get("minibatch_size", 256),
    )
    gamma = float(args.gamma if args.gamma is not None else adapt_ppo_cfg.get("gamma", 0.99))
    gae_lambda = float(
        args.gae_lambda
        if args.gae_lambda is not None
        else adapt_ppo_cfg.get("gae_lambda", 0.95),
    )
    clip_coef = float(
        args.clip_coef
        if args.clip_coef is not None
        else adapt_ppo_cfg.get("clip_coef", 0.2),
    )
    ent_coef = float(args.ent_coef if args.ent_coef is not None else adapt_ppo_cfg.get("ent_coef", 0.01))
    vf_coef = float(args.vf_coef if args.vf_coef is not None else adapt_ppo_cfg.get("vf_coef", 0.5))
    lr = float(args.lr if args.lr is not None else adapt_ppo_cfg.get("lr", 3e-4))
    max_grad_norm = float(
        args.max_grad_norm
        if args.max_grad_norm is not None
        else adapt_ppo_cfg.get("max_grad_norm", 0.5),
    )
    early_stop_min_steps = int(
        args.early_stop_min_steps
        if args.early_stop_min_steps is not None
        else adapt_ppo_cfg.get("early_stop_min_steps", 0),
    )
    early_stop_reward_threshold_raw = (
        args.early_stop_reward_threshold
        if args.early_stop_reward_threshold is not None
        else adapt_ppo_cfg.get("early_stop_reward_threshold", None)
    )
    early_stop_failure_rate_threshold_raw = (
        args.early_stop_failure_rate_threshold
        if args.early_stop_failure_rate_threshold is not None
        else adapt_ppo_cfg.get("early_stop_failure_rate_threshold", None)
    )
    early_stop_success_rate_threshold_raw = (
        args.early_stop_success_rate_threshold
        if args.early_stop_success_rate_threshold is not None
        else adapt_ppo_cfg.get("early_stop_success_rate_threshold", None)
    )
    early_stop_reward_threshold = (
        float(early_stop_reward_threshold_raw)
        if early_stop_reward_threshold_raw is not None
        else None
    )
    early_stop_failure_rate_threshold = (
        float(early_stop_failure_rate_threshold_raw)
        if early_stop_failure_rate_threshold_raw is not None
        else None
    )
    early_stop_success_rate_threshold = (
        float(early_stop_success_rate_threshold_raw)
        if early_stop_success_rate_threshold_raw is not None
        else None
    )

    rashomon_rollouts = int(
        args.rashomon_rollouts
        if args.rashomon_rollouts is not None
        else rashomon_cfg.get("rashomon_rollouts", 1),
    )
    rashomon_n_iters = int(
        args.rashomon_n_iters
        if args.rashomon_n_iters is not None
        else rashomon_cfg.get("rashomon_n_iters", 50_000),
    )
    rashomon_min_hard_spec = float(
        args.rashomon_min_hard_spec
        if args.rashomon_min_hard_spec is not None
        else rashomon_cfg.get("rashomon_min_hard_spec", 1.0),
    )
    rashomon_surrogate_aggregation = str(
        args.rashomon_surrogate_aggregation
        if args.rashomon_surrogate_aggregation is not None
        else rashomon_cfg.get("rashomon_surrogate_aggregation", "min"),
    )
    inverse_temp_start = int(
        args.inverse_temp_start
        if args.inverse_temp_start is not None
        else rashomon_cfg.get("inverse_temp_start", 10),
    )
    inverse_temp_max = int(
        args.inverse_temp_max
        if args.inverse_temp_max is not None
        else rashomon_cfg.get("inverse_temp_max", 1000),
    )
    rashomon_checkpoint = int(
        args.rashomon_checkpoint
        if args.rashomon_checkpoint is not None
        else rashomon_cfg.get("rashomon_checkpoint", 100),
    )

    if total_timesteps <= 0:
        raise ValueError("--total-timesteps (or resolved default) must be > 0.")
    if eval_episodes_during_training <= 0:
        raise ValueError("--eval-episodes-during-training (or resolved default) must be > 0.")
    if eval_episodes_post_training <= 0:
        raise ValueError("--eval-episodes-post-training (or resolved default) must be > 0.")
    if rollout_steps <= 0:
        raise ValueError("--rollout-steps (or resolved default) must be > 0.")
    if update_epochs <= 0:
        raise ValueError("--update-epochs (or resolved default) must be > 0.")
    if minibatch_size <= 0:
        raise ValueError("--minibatch-size (or resolved default) must be > 0.")
    if rashomon_rollouts <= 0:
        raise ValueError("--rashomon-rollouts (or resolved default) must be > 0.")
    if rashomon_n_iters <= 0:
        raise ValueError("--rashomon-n-iters (or resolved default) must be > 0.")
    if rashomon_surrogate_aggregation not in {"mean", "min"}:
        raise ValueError(
            "--rashomon-surrogate-aggregation (or resolved default) must be one of: mean, min.",
        )
    if inverse_temp_start <= 0 or inverse_temp_max < inverse_temp_start:
        raise ValueError(
            "Invalid inverse-temperature range. Require 0 < inverse-temp-start <= inverse-temp-max.",
        )

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

    source_task_id = float(source_task_cfg.get("task_id", 0.0))
    downstream_task_id = float(downstream_task_cfg.get("task_id", 1.0))
    append_task_id = (
        bool(args.append_task_id)
        if args.append_task_id is not None
        else bool(source_task_cfg.get("append_task_id", True))
    )
    task_pipelines_file = str(source_task_cfg.get("_task_pipelines_file") or args.task_settings_file)
    task_definitions_file_raw = source_task_cfg.get("_task_definitions_file")
    task_definitions_file = None if task_definitions_file_raw is None else str(task_definitions_file_raw)
    resolved_pipeline_name = source_task_cfg.get("_resolved_pipeline_name")
    resolved_source_definition_name = source_task_cfg.get("_resolved_definition_name")
    resolved_downstream_definition_name = downstream_task_cfg.get("_resolved_definition_name")
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

    source_run_dir = (
        args.source_run_dir
        if args.source_run_dir is not None
        else _resolve_default_source_run_dir(args.outputs_root, args.task_setting, args.seed)
    )
    actor_ckpt = source_run_dir / "actor.pt"
    critic_ckpt = source_run_dir / "critic.pt"

    if not actor_ckpt.exists():
        raise FileNotFoundError(f"NoAdapt actor checkpoint not found: {actor_ckpt}")
    if args.warm_start_critic and not critic_ckpt.exists():
        raise FileNotFoundError(f"NoAdapt critic checkpoint not found: {critic_ckpt}")

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

    source_actor, source_critic = build_actor_critic(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_size=hidden_size,
    )
    source_actor.load_state_dict(torch.load(actor_ckpt, map_location="cpu"))
    if args.warm_start_critic:
        source_critic.load_state_dict(torch.load(critic_ckpt, map_location="cpu"))

    source_rollout_env = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **source_env_kwargs,
    )
    try:
        rashomon_dataset, rollout_lengths = create_source_rollout_rashomon_dataset(
            actor=copy.deepcopy(source_actor),
            env=source_rollout_env,
            seed=args.seed,
            n_actions=n_actions,
            rashomon_rollouts=rashomon_rollouts,
        )
    finally:
        source_rollout_env.close()

    print(
        f"Built source rollout Rashomon dataset: {len(rashomon_dataset)} samples "
        f"from {rashomon_rollouts} rollouts.",
    )

    (
        param_bounds_l,
        param_bounds_u,
        bounded_model,
        selected_inverse_temp,
        surrogate_threshold,
        cert_values,
        selected_cert_idx,
    ) = compute_rashomon_bounds(
        actor=copy.deepcopy(source_actor),
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
        f"Rashomon bounds ready: aggregation={rashomon_surrogate_aggregation} | "
        f"min_hard_spec={rashomon_min_hard_spec:.3f} | selected_cert={cert_values[selected_cert_idx]:.4f} "
        f"(idx={selected_cert_idx}) | inverse_temp={selected_inverse_temp}",
    )

    task_feature_index = obs_dim - 1
    do_task_neutralization = (
        append_task_id
        and bool(args.enable_task_neutralization)
        and not args.disable_task_neutralization
    )
    if do_task_neutralization:
        neutralize_task_feature(source_actor, task_feature_index, downstream_task_id)
        if args.warm_start_critic:
            neutralize_task_feature(source_critic, task_feature_index, downstream_task_id)

    ppo_cfg = PPOConfig(
        seed=int(adapt_ppo_cfg.get("seed", args.seed)),
        total_timesteps=total_timesteps,
        eval_episodes=eval_episodes_during_training,
        rollout_steps=rollout_steps,
        update_epochs=update_epochs,
        minibatch_size=minibatch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        lr=lr,
        max_grad_norm=max_grad_norm,
        device=device,
        early_stop_min_steps=early_stop_min_steps,
        early_stop_reward_threshold=early_stop_reward_threshold,
        early_stop_failure_rate_threshold=early_stop_failure_rate_threshold,
        early_stop_success_rate_threshold=early_stop_success_rate_threshold,
    )

    print(
        f"Adapting LunarLander with Rashomon-PGD | source_task={source_task_id} -> "
        f"downstream_task={downstream_task_id} | warm_critic={args.warm_start_critic} | "
        f"task_neutralization={do_task_neutralization}",
    )

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

    actor, critic, training_data = ppo_train(  # type: ignore[assignment]
        train_env,
        ppo_cfg,
        actor_warm_start=source_actor,
        critic_warm_start=(source_critic if args.warm_start_critic else None),
        actor_param_bounds_l=param_bounds_l,
        actor_param_bounds_u=param_bounds_u,
        early_stop_eval_env=early_stop_eval_env,
        return_training_data=True,
    )

    source_eval_env = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **source_env_kwargs,
    )
    source_mean_reward, source_std_reward, source_failure_rate, source_success_rate = evaluate_with_success(
        source_eval_env,
        actor,
        episodes=eval_episodes_post_training,
        deterministic=True,
        device=device,
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
        actor,
        episodes=eval_episodes_post_training,
        deterministic=True,
        device=device,
    )
    downstream_eval_env.close()

    downstream_run_dir = _seed_run_dir(args.outputs_root, args.task_setting, args.seed) / args.run_subdir
    downstream_run_dir.mkdir(parents=True, exist_ok=True)

    actor_path = downstream_run_dir / "actor.pt"
    critic_path = downstream_run_dir / "critic.pt"
    training_data_path = downstream_run_dir / "training_data.pt"
    rashomon_dataset_path = downstream_run_dir / "rashomon_dataset.pt"
    bounded_model_path = downstream_run_dir / "rashomon_bounded_model.pt"
    bounds_path = downstream_run_dir / "rashomon_param_bounds.pt"
    rollout_stats_path = downstream_run_dir / "rashomon_rollout_stats.yaml"
    source_plot_path = downstream_run_dir / "trajectory_source.png"
    downstream_plot_path = downstream_run_dir / "trajectory_downstream.png"
    summary_path = downstream_run_dir / "run_summary.yaml"

    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    torch.save(training_data, training_data_path)
    torch.save(rashomon_dataset, rashomon_dataset_path)
    torch.save(bounded_model, bounded_model_path)
    torch.save(
        {
            "param_bounds_l": param_bounds_l,
            "param_bounds_u": param_bounds_u,
        },
        bounds_path,
    )

    # Plot with a CPU actor copy to avoid device-mismatch issues in rendering helpers.
    actor_for_plot = copy.deepcopy(actor).to("cpu")
    actor_for_plot.eval()

    rollout_stats: dict[str, Any] = {
        "rashomon_rollouts": int(rashomon_rollouts),
        "total_state_action_pairs": int(len(rashomon_dataset)),
        "rollout_lengths": [int(x) for x in rollout_lengths],
        "rollout_length_min": int(min(rollout_lengths)),
        "rollout_length_max": int(max(rollout_lengths)),
        "rollout_length_mean": float(np.mean(rollout_lengths)),
        "rollout_length_std": float(np.std(rollout_lengths)),
    }
    rollout_stats_path.write_text(
        yaml.safe_dump(rollout_stats, sort_keys=False),
        encoding="utf-8",
    )

    _plot_trajectory_grid(
        env_id=env_id,
        gravity=source_gravity,
        task_id=source_task_id,
        append_task_id=append_task_id,
        dynamics_cfg=source_dynamics,
        actor=actor_for_plot,
        seed=args.seed,
        device="cpu",
        output_path=source_plot_path,
        episodes=int(args.trajectory_episodes),
        max_frames_per_episode=int(args.trajectory_max_frames_per_episode),
    )
    _plot_trajectory_grid(
        env_id=env_id,
        gravity=downstream_gravity,
        task_id=downstream_task_id,
        append_task_id=append_task_id,
        dynamics_cfg=downstream_dynamics,
        actor=actor_for_plot,
        seed=args.seed,
        device="cpu",
        output_path=downstream_plot_path,
        episodes=int(args.trajectory_episodes),
        max_frames_per_episode=int(args.trajectory_max_frames_per_episode),
    )

    run_settings = {
        "seed": int(args.seed),
        "env_id": env_id,
        "continuous": bool(continuous),
        "source_gravity": source_gravity,
        "downstream_gravity": downstream_gravity,
        "source_dynamics": source_dynamics,
        "downstream_dynamics": downstream_dynamics,
        "source_task_id": float(source_task_id),
        "downstream_task_id": float(downstream_task_id),
        "append_task_id": bool(append_task_id),
        "warm_start_critic": bool(args.warm_start_critic),
        "task_feature_neutralization": bool(do_task_neutralization),
        "task_feature_index": int(task_feature_index) if do_task_neutralization else None,
        "hidden_size": int(hidden_size),
        "device": device,
        "total_timesteps": int(total_timesteps),
        "eval_episodes_during_training": int(eval_episodes_during_training),
        "eval_episodes_post_training": int(eval_episodes_post_training),
        "rollout_steps": int(rollout_steps),
        "update_epochs": int(update_epochs),
        "minibatch_size": int(minibatch_size),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "clip_coef": float(clip_coef),
        "ent_coef": float(ent_coef),
        "vf_coef": float(vf_coef),
        "lr": float(lr),
        "max_grad_norm": float(max_grad_norm),
        "early_stop_min_steps": int(early_stop_min_steps),
        "early_stop_reward_threshold": early_stop_reward_threshold,
        "early_stop_failure_rate_threshold": early_stop_failure_rate_threshold,
        "early_stop_success_rate_threshold": early_stop_success_rate_threshold,
        "trajectory_episodes": int(args.trajectory_episodes),
        "trajectory_max_frames_per_episode": int(args.trajectory_max_frames_per_episode),
        "rashomon_rollouts": int(rashomon_rollouts),
        "rashomon_n_iters": int(rashomon_n_iters),
        "surrogate_aggregation": str(rashomon_surrogate_aggregation),
        "inverse_temp_start": int(inverse_temp_start),
        "inverse_temp_max": int(inverse_temp_max),
        "rashomon_checkpoint": int(rashomon_checkpoint),
        "noadapt_checkpoint_dir": str(source_run_dir),
        "source_checkpoint_dir": str(source_run_dir),
        "task_setting": args.task_setting,
        "task_settings_file": str(args.task_settings_file),
        "adapt_settings_file": str(args.adapt_settings_file),
        "rashomon_settings_file": str(args.rashomon_settings_file),
        "task_pipelines_file": task_pipelines_file,
        "task_definitions_file": task_definitions_file,
        "resolved_pipeline_name": resolved_pipeline_name,
        "resolved_source_definition_name": resolved_source_definition_name,
        "resolved_downstream_definition_name": resolved_downstream_definition_name,
    }
    run_results = {
        "rashomon_dataset_size": int(len(rashomon_dataset)),
        "surrogate_threshold": float(surrogate_threshold),
        "inverse_temperature": int(selected_inverse_temp),
        "selected_certificate_index": int(selected_cert_idx),
        "selected_certificate": float(cert_values[selected_cert_idx]),
        "all_certificates": [float(v) for v in cert_values],
        "source_mean_reward": float(source_mean_reward),
        "source_std_reward": float(source_std_reward),
        "source_failure_rate": float(source_failure_rate),
        "source_success_rate": float(source_success_rate),
        "downstream_mean_reward": float(downstream_mean_reward),
        "downstream_std_reward": float(downstream_std_reward),
        "downstream_failure_rate": float(downstream_failure_rate),
        "downstream_success_rate": float(downstream_success_rate),
    }
    artifacts = {
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
        "training_data_path": str(training_data_path),
        "rashomon_dataset_path": str(rashomon_dataset_path),
        "rashomon_bounded_model_path": str(bounded_model_path),
        "rashomon_param_bounds_path": str(bounds_path),
        "rashomon_rollout_stats_path": str(rollout_stats_path),
        "trajectory_source_plot_path": str(source_plot_path),
        "trajectory_downstream_plot_path": str(downstream_plot_path),
    }
    summary = {
        "run_settings": run_settings,
        "run_results": run_results,
        "artifacts": artifacts,
    }
    summary_path.write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )

    print(
        f"Source eval ({eval_episodes_post_training} ep): mean_reward={source_mean_reward:.3f}, "
        f"std={source_std_reward:.3f}, failure_rate={source_failure_rate:.3f}, "
        f"success_rate={source_success_rate:.3f}",
    )
    print(
        f"Downstream eval ({eval_episodes_post_training} ep): mean_reward={downstream_mean_reward:.3f}, "
        f"std={downstream_std_reward:.3f}, failure_rate={downstream_failure_rate:.3f}, "
        f"success_rate={downstream_success_rate:.3f}",
    )
    print(f"Saved actor: {actor_path}")
    print(f"Saved critic: {critic_path}")
    print(f"Saved Rashomon dataset: {rashomon_dataset_path}")
    print(f"Saved Rashomon bounded model: {bounded_model_path}")
    print(f"Saved source trajectory grid: {source_plot_path}")
    print(f"Saved downstream trajectory grid: {downstream_plot_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
