"""Adapt LunarLander with a family of independently computed Rashomon sets."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

os.environ["SDL_AUDIODRIVER"] = "dummy"

import gymnasium as gym
import numpy as np
import torch
import yaml

# Allow running this file directly from experiments/pipelines/lunarlander.
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.lunarlander.core.env.env_factory import _make_lunarlander_env
from experiments.pipelines.lunarlander.core.env.task_loading import (
    _load_task_settings,
    _resolve_lunarlander_dynamics,
)
from experiments.pipelines.lunarlander.core.methods.adapt_rashomon import (
    _load_source_hidden_size,
    _load_yaml,
    _resolve_setting_cfg,
    _sample_reference_network_from_bounds,
    compute_rashomon_bounds,
    create_source_rollout_rashomon_dataset,
    neutralize_task_feature,
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


@dataclass
class FamilyRashomonSet:
    member_idx: int
    reference_kind: str
    reference_actor: torch.nn.Sequential
    bounds_l: list[torch.Tensor]
    bounds_u: list[torch.Tensor]
    bounded_model: object
    selected_inverse_temp: int
    surrogate_threshold: float
    cert_values: list[float]
    selected_cert_idx: int
    sample_stats: dict[str, Any] | None = None


@dataclass
class FamilyAdaptationResult:
    member_idx: int
    actor: torch.nn.Module
    critic: torch.nn.Module
    training_data: dict[str, Any]
    source_mean_reward: float
    source_std_reward: float
    source_failure_rate: float
    source_success_rate: float
    downstream_mean_reward: float
    downstream_std_reward: float
    downstream_failure_rate: float
    downstream_success_rate: float


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None else None


def _compute_family_set(
    *,
    member_idx: int,
    reference_kind: str,
    reference_actor: torch.nn.Sequential,
    rashomon_dataset,
    seed: int,
    rashomon_n_iters_per_run: int,
    min_hard_spec: float,
    aggregation: str,
    inverse_temp_start: int,
    inverse_temp_max: int,
    checkpoint: int,
    sample_stats: dict[str, Any] | None = None,
) -> FamilyRashomonSet:
    (
        bounds_l,
        bounds_u,
        bounded_model,
        selected_inverse_temp,
        surrogate_threshold,
        cert_values,
        selected_cert_idx,
    ) = compute_rashomon_bounds(
        actor=copy.deepcopy(reference_actor).to("cpu"),
        rashomon_dataset=rashomon_dataset,
        seed=int(seed),
        rashomon_n_iters=int(rashomon_n_iters_per_run),
        min_hard_spec=float(min_hard_spec),
        aggregation=str(aggregation),
        inverse_temp_start=int(inverse_temp_start),
        inverse_temp_max=int(inverse_temp_max),
        checkpoint=int(checkpoint),
    )
    return FamilyRashomonSet(
        member_idx=int(member_idx),
        reference_kind=reference_kind,
        reference_actor=copy.deepcopy(reference_actor).to("cpu"),
        bounds_l=[p.detach().cpu().clone() for p in bounds_l],
        bounds_u=[p.detach().cpu().clone() for p in bounds_u],
        bounded_model=bounded_model,
        selected_inverse_temp=int(selected_inverse_temp),
        surrogate_threshold=float(surrogate_threshold),
        cert_values=[float(v) for v in cert_values],
        selected_cert_idx=int(selected_cert_idx),
        sample_stats=sample_stats,
    )


def _member_dir(root: Path, member_idx: int) -> Path:
    return root / f"member_{member_idx:02d}"


def _save_family_set(member_dir: Path, family_set: FamilyRashomonSet) -> dict[str, str]:
    member_dir.mkdir(parents=True, exist_ok=True)
    reference_path = member_dir / "reference_actor.pt"
    bounds_path = member_dir / "rashomon_param_bounds.pt"
    bounded_model_path = member_dir / "rashomon_bounded_model.pt"
    metadata_path = member_dir / "rashomon_metadata.yaml"

    torch.save(family_set.reference_actor.state_dict(), reference_path)
    torch.save(
        {
            "param_bounds_l": family_set.bounds_l,
            "param_bounds_u": family_set.bounds_u,
        },
        bounds_path,
    )
    torch.save(family_set.bounded_model, bounded_model_path)
    metadata = {
        "member_idx": int(family_set.member_idx),
        "reference_kind": str(family_set.reference_kind),
        "selected_inverse_temperature": int(family_set.selected_inverse_temp),
        "surrogate_threshold": float(family_set.surrogate_threshold),
        "selected_certificate_index": int(family_set.selected_cert_idx),
        "selected_certificate": float(family_set.cert_values[family_set.selected_cert_idx]),
        "all_certificates": [float(v) for v in family_set.cert_values],
        "sample_stats": family_set.sample_stats,
    }
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return {
        "reference_actor_path": str(reference_path),
        "rashomon_param_bounds_path": str(bounds_path),
        "rashomon_bounded_model_path": str(bounded_model_path),
        "rashomon_metadata_path": str(metadata_path),
    }


def _adapt_with_family_set(
    *,
    family_set: FamilyRashomonSet,
    source_critic: torch.nn.Sequential,
    warm_start_critic: bool,
    ppo_cfg: PPOConfig,
    env_id: str,
    source_env_kwargs: dict[str, Any],
    downstream_env_kwargs: dict[str, Any],
    append_task_id: bool,
    enable_task_neutralization: bool,
    disable_task_neutralization: bool,
    downstream_task_id: float,
    obs_dim: int,
    eval_episodes_post_training: int,
    device: str,
) -> FamilyAdaptationResult:
    actor_start = copy.deepcopy(family_set.reference_actor).to("cpu")
    critic_start = copy.deepcopy(source_critic).to("cpu")

    task_feature_index = obs_dim - 1
    do_task_neutralization = (
        append_task_id
        and bool(enable_task_neutralization)
        and not bool(disable_task_neutralization)
    )
    if do_task_neutralization:
        neutralize_task_feature(actor_start, task_feature_index, downstream_task_id)
        if warm_start_critic:
            neutralize_task_feature(critic_start, task_feature_index, downstream_task_id)

    train_env = _make_lunarlander_env(env_id, render_mode=None, **downstream_env_kwargs)
    early_stop_eval_env = _make_lunarlander_env(env_id, render_mode=None, **downstream_env_kwargs)
    try:
        actor, critic, training_data = ppo_train(  # type: ignore[assignment]
            train_env,
            ppo_cfg,
            actor_warm_start=actor_start,
            critic_warm_start=(critic_start if warm_start_critic else None),
            actor_param_bounds_l=family_set.bounds_l,
            actor_param_bounds_u=family_set.bounds_u,
            early_stop_eval_env=early_stop_eval_env,
            return_training_data=True,
        )
    finally:
        train_env.close()
        early_stop_eval_env.close()

    source_eval_env = _make_lunarlander_env(env_id, render_mode=None, **source_env_kwargs)
    source_mean_reward, source_std_reward, source_failure_rate, source_success_rate = evaluate_with_success(
        source_eval_env,
        actor,
        episodes=int(eval_episodes_post_training),
        deterministic=True,
        device=device,
    )
    source_eval_env.close()

    downstream_eval_env = _make_lunarlander_env(env_id, render_mode=None, **downstream_env_kwargs)
    (
        downstream_mean_reward,
        downstream_std_reward,
        downstream_failure_rate,
        downstream_success_rate,
    ) = evaluate_with_success(
        downstream_eval_env,
        actor,
        episodes=int(eval_episodes_post_training),
        deterministic=True,
        device=device,
    )
    downstream_eval_env.close()

    return FamilyAdaptationResult(
        member_idx=int(family_set.member_idx),
        actor=actor,
        critic=critic,
        training_data=training_data,
        source_mean_reward=float(source_mean_reward),
        source_std_reward=float(source_std_reward),
        source_failure_rate=float(source_failure_rate),
        source_success_rate=float(source_success_rate),
        downstream_mean_reward=float(downstream_mean_reward),
        downstream_std_reward=float(downstream_std_reward),
        downstream_failure_rate=float(downstream_failure_rate),
        downstream_success_rate=float(downstream_success_rate),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LunarLander adaptation over a sampled family of Rashomon sets.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
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
    parser.add_argument("--pipeline", type=str, dest="task_setting", default="default")
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
    parser.add_argument("--outputs-root", type=Path, default=default_outputs_root())
    parser.add_argument(
        "--run-subdir",
        type=str,
        default="downstream_rashomon_family",
        help="Subdirectory under outputs/<pipeline>/seed_<seed>/ where outputs are saved.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Actor/critic hidden size. Defaults to source run summary hidden_size if available.",
    )
    parser.add_argument(
        "--family-size",
        type=int,
        default=None,
        help="Total number of Rashomon sets/adaptation runs. Defaults to rashomon.yaml family_size or 3.",
    )
    parser.add_argument(
        "--family-workers",
        type=int,
        default=None,
        help="Parallel workers for sampled Rashomon set construction. Defaults to min(family_size - 1, CPU count).",
    )
    parser.add_argument(
        "--rashomon-n-iters",
        type=int,
        default=None,
        help="Total Rashomon iteration budget across all family members. Defaults to rashomon.yaml.",
    )
    parser.add_argument(
        "--warm-start-critic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm-start critic from source checkpoint.",
    )
    parser.add_argument("--enable-task-neutralization", action="store_true")
    parser.add_argument("--disable-task-neutralization", action="store_true")
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
    args = parser.parse_args()

    if args.enable_task_neutralization and args.disable_task_neutralization:
        raise ValueError("Cannot set both --enable-task-neutralization and --disable-task-neutralization.")

    source_task_cfg = _load_task_settings(args.task_settings_file, args.task_setting, "source")
    downstream_task_cfg = _load_task_settings(args.task_settings_file, args.task_setting, "downstream")

    adapt_settings = _load_yaml(args.adapt_settings_file)
    adapt_cfg = _resolve_setting_cfg(adapt_settings, args.task_setting, settings_name=str(args.adapt_settings_file))
    adapt_ppo_cfg = adapt_cfg.get("ppo", {})
    if not isinstance(adapt_ppo_cfg, dict):
        raise ValueError(f"Expected 'ppo' mapping for setting '{args.task_setting}' in {args.adapt_settings_file}.")
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

    family_size = int(args.family_size if args.family_size is not None else rashomon_cfg.get("family_size", 3))
    if family_size <= 0:
        raise ValueError("--family-size must be > 0.")
    family_workers = int(
        args.family_workers
        if args.family_workers is not None
        else max(1, min(max(family_size - 1, 1), os.cpu_count() or 1)),
    )
    if family_workers <= 0:
        raise ValueError("--family-workers must be > 0.")

    device = str(args.device)
    total_timesteps = int(adapt_ppo_cfg.get("total_timesteps", 200_000))
    total_timesteps_per_run = int(total_timesteps / family_size)
    eval_episodes_during_training = int(adapt_ppo_cfg.get("eval_episodes_during_training", 20))
    eval_episodes_post_training = int(downstream_eval_cfg.get("episodes_post_training", 100))
    rollout_steps = int(adapt_ppo_cfg.get("rollout_steps", 2048))
    update_epochs = int(adapt_ppo_cfg.get("update_epochs", 10))
    minibatch_size = int(adapt_ppo_cfg.get("minibatch_size", 256))
    gamma = float(adapt_ppo_cfg.get("gamma", 0.99))
    gae_lambda = float(adapt_ppo_cfg.get("gae_lambda", 0.95))
    clip_coef = float(adapt_ppo_cfg.get("clip_coef", 0.2))
    ent_coef = float(adapt_ppo_cfg.get("ent_coef", 0.01))
    vf_coef = float(adapt_ppo_cfg.get("vf_coef", 0.5))
    lr = float(adapt_ppo_cfg.get("lr", 3e-4))
    max_grad_norm = float(adapt_ppo_cfg.get("max_grad_norm", 0.5))
    early_stop_success_rate_threshold_raw = adapt_ppo_cfg.get("early_stop_success_rate_threshold", 1.0)
    early_stop_success_rate_threshold = (
        float(early_stop_success_rate_threshold_raw)
        if early_stop_success_rate_threshold_raw is not None
        else 1.0
    )
    pgd_projection_distance_norm = str(adapt_ppo_cfg.get("pgd_projection_distance_norm", "l2"))

    rashomon_rollouts = 1
    rashomon_n_iters = int(
        args.rashomon_n_iters
        if args.rashomon_n_iters is not None
        else rashomon_cfg.get("rashomon_n_iters", 50_000),
    )
    rashomon_n_iters_per_run = int(rashomon_n_iters / family_size)
    rashomon_min_hard_spec = float(rashomon_cfg.get("rashomon_min_hard_spec", 1.0))
    rashomon_surrogate_aggregation = str(rashomon_cfg.get("rashomon_surrogate_aggregation", "min"))
    inverse_temp_start = int(rashomon_cfg.get("inverse_temp_start", 10))
    inverse_temp_max = int(rashomon_cfg.get("inverse_temp_max", 1000))
    rashomon_checkpoint = int(rashomon_cfg.get("rashomon_checkpoint", 100))

    if total_timesteps <= 0 or total_timesteps_per_run <= 0:
        raise ValueError("total_timesteps / family_size must provide at least one PPO timestep per member.")
    if rashomon_n_iters <= 0 or rashomon_n_iters_per_run <= 0:
        raise ValueError("rashomon_n_iters / family_size must provide at least one iteration per member.")
    if eval_episodes_during_training <= 0 or eval_episodes_post_training <= 0:
        raise ValueError("Evaluation episode counts must be > 0.")
    if rollout_steps <= 0 or update_epochs <= 0 or minibatch_size <= 0:
        raise ValueError("PPO rollout/update/minibatch settings must be > 0.")
    if not 0.0 <= early_stop_success_rate_threshold <= 1.0:
        raise ValueError("early_stop_success_rate_threshold must be in [0, 1].")
    if pgd_projection_distance_norm not in {"l2", "l1", "linf"}:
        raise ValueError("pgd_projection_distance_norm must be one of: l2, l1, linf.")
    if rashomon_surrogate_aggregation not in {"mean", "min"}:
        raise ValueError("rashomon_surrogate_aggregation must be one of: mean, min.")
    if inverse_temp_start <= 0 or inverse_temp_max < inverse_temp_start:
        raise ValueError("Invalid inverse-temperature range. Require 0 < inverse-temp-start <= inverse-temp-max.")

    env_id = str(args.env_id or source_task_cfg.get("env_id") or downstream_task_cfg.get("env_id") or "LunarLander-v3")
    source_gravity_raw = args.source_gravity if args.source_gravity is not None else source_task_cfg.get("gravity")
    downstream_gravity_raw = args.downstream_gravity if args.downstream_gravity is not None else downstream_task_cfg.get("gravity")
    source_gravity = None if source_gravity_raw is None else float(source_gravity_raw)
    downstream_gravity = None if downstream_gravity_raw is None else float(downstream_gravity_raw)

    source_task_id = float(source_task_cfg.get("task_id", 0.0))
    downstream_task_id = float(downstream_task_cfg.get("task_id", 1.0))
    append_task_id = bool(args.append_task_id) if args.append_task_id is not None else bool(source_task_cfg.get("append_task_id", True))
    task_pipelines_file = str(source_task_cfg.get("_task_pipelines_file") or args.task_settings_file)
    task_definitions_file_raw = source_task_cfg.get("_task_definitions_file")
    task_definitions_file = None if task_definitions_file_raw is None else str(task_definitions_file_raw)
    resolved_pipeline_name = source_task_cfg.get("_resolved_pipeline_name")
    resolved_source_definition_name = source_task_cfg.get("_resolved_definition_name")
    resolved_downstream_definition_name = downstream_task_cfg.get("_resolved_definition_name")
    source_dynamics = _resolve_lunarlander_dynamics(source_task_cfg, cfg_name=f"task_settings[{args.task_setting}:source]")
    downstream_dynamics = _resolve_lunarlander_dynamics(downstream_task_cfg, cfg_name=f"task_settings[{args.task_setting}:downstream]")

    continuous = bool(source_task_cfg.get("continuous", False) or downstream_task_cfg.get("continuous", False))
    if continuous:
        raise ValueError("This script only supports discrete actions (`continuous=False`).")

    source_env_kwargs = {"gravity": source_gravity, "task_id": source_task_id, "append_task_id": append_task_id, **source_dynamics}
    downstream_env_kwargs = {
        "gravity": downstream_gravity,
        "task_id": downstream_task_id,
        "append_task_id": append_task_id,
        **downstream_dynamics,
    }

    source_run_dir = args.source_run_dir if args.source_run_dir is not None else _resolve_default_source_run_dir(args.outputs_root, args.task_setting, args.seed)
    actor_ckpt = source_run_dir / "actor.pt"
    critic_ckpt = source_run_dir / "critic.pt"
    if not actor_ckpt.exists():
        raise FileNotFoundError(f"NoAdapt actor checkpoint not found: {actor_ckpt}")
    if args.warm_start_critic and not critic_ckpt.exists():
        raise FileNotFoundError(f"NoAdapt critic checkpoint not found: {critic_ckpt}")

    hidden_size = _load_source_hidden_size(source_run_dir, args.hidden_size)
    source_env_for_dim = _make_lunarlander_env(env_id, render_mode=None, **source_env_kwargs)
    if not isinstance(source_env_for_dim.action_space, gym.spaces.Discrete):
        raise ValueError("Expected discrete action space for LunarLander.")
    obs_dim = int(source_env_for_dim.observation_space.shape[0])  # type: ignore[index]
    n_actions = int(source_env_for_dim.action_space.n)  # type: ignore[union-attr]
    source_env_for_dim.close()

    source_actor, source_critic = build_actor_critic(obs_dim=obs_dim, n_actions=n_actions, hidden_size=hidden_size)
    source_actor.load_state_dict(torch.load(actor_ckpt, map_location="cpu"))
    if args.warm_start_critic:
        source_critic.load_state_dict(torch.load(critic_ckpt, map_location="cpu"))

    downstream_run_dir = _seed_run_dir(args.outputs_root, args.task_setting, args.seed) / args.run_subdir
    downstream_run_dir.mkdir(parents=True, exist_ok=True)
    family_dir = downstream_run_dir / "family_members"
    family_dir.mkdir(parents=True, exist_ok=True)

    source_rollout_env = _make_lunarlander_env(env_id, render_mode=None, **source_env_kwargs)
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
        f"Built Rashomon family dataset: {len(rashomon_dataset)} samples from one source rollout. "
        f"family_size={family_size} | rashomon_iters_per_run={rashomon_n_iters_per_run} | "
        f"ppo_timesteps_per_run={total_timesteps_per_run}",
    )

    initial_set = _compute_family_set(
        member_idx=0,
        reference_kind="source",
        reference_actor=copy.deepcopy(source_actor).to("cpu"),
        rashomon_dataset=rashomon_dataset,
        seed=int(args.seed),
        rashomon_n_iters_per_run=rashomon_n_iters_per_run,
        min_hard_spec=rashomon_min_hard_spec,
        aggregation=rashomon_surrogate_aggregation,
        inverse_temp_start=inverse_temp_start,
        inverse_temp_max=inverse_temp_max,
        checkpoint=rashomon_checkpoint,
    )

    sampled_references: list[tuple[int, torch.nn.Sequential, dict[str, Any]]] = []
    for member_idx in range(1, family_size):
        sampled_actor, sample_stats = _sample_reference_network_from_bounds(
            reference_network=source_actor,
            bounds_l=initial_set.bounds_l,
            bounds_u=initial_set.bounds_u,
            seed=int(args.seed + 10_000 + member_idx),
            upper_prob=0.5,
        )
        sampled_references.append((member_idx, sampled_actor.to("cpu"), sample_stats))

    family_sets: list[FamilyRashomonSet] = [initial_set]
    if sampled_references:
        with ThreadPoolExecutor(max_workers=min(family_workers, len(sampled_references))) as executor:
            futures = {
                executor.submit(
                    _compute_family_set,
                    member_idx=member_idx,
                    reference_kind="sampled_border",
                    reference_actor=reference_actor,
                    rashomon_dataset=rashomon_dataset,
                    seed=int(args.seed + member_idx),
                    rashomon_n_iters_per_run=rashomon_n_iters_per_run,
                    min_hard_spec=rashomon_min_hard_spec,
                    aggregation=rashomon_surrogate_aggregation,
                    inverse_temp_start=inverse_temp_start,
                    inverse_temp_max=inverse_temp_max,
                    checkpoint=rashomon_checkpoint,
                    sample_stats=sample_stats,
                ): member_idx
                for member_idx, reference_actor, sample_stats in sampled_references
            }
            for future in as_completed(futures):
                family_sets.append(future.result())
    family_sets.sort(key=lambda rec: rec.member_idx)

    family_artifacts: dict[int, dict[str, str]] = {}
    for family_set in family_sets:
        family_artifacts[family_set.member_idx] = _save_family_set(
            _member_dir(family_dir, family_set.member_idx),
            family_set,
        )

    family_results: list[FamilyAdaptationResult] = []
    for family_set in family_sets:
        print(
            f"Adapting Rashomon family member {family_set.member_idx}/{family_size - 1} | "
            f"reference={family_set.reference_kind} | selected_cert="
            f"{family_set.cert_values[family_set.selected_cert_idx]:.4f}",
        )
        member_ppo_cfg = PPOConfig(
            seed=int(args.seed + 20_000 + family_set.member_idx),
            total_timesteps=total_timesteps_per_run,
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
            early_stop_min_steps=0,
            early_stop_reward_threshold=None,
            early_stop_failure_rate_threshold=None,
            early_stop_success_rate_threshold=early_stop_success_rate_threshold,
            pgd_projection_distance_norm=pgd_projection_distance_norm,
        )
        result = _adapt_with_family_set(
            family_set=family_set,
            source_critic=source_critic,
            warm_start_critic=bool(args.warm_start_critic),
            ppo_cfg=member_ppo_cfg,
            env_id=env_id,
            source_env_kwargs=source_env_kwargs,
            downstream_env_kwargs=downstream_env_kwargs,
            append_task_id=append_task_id,
            enable_task_neutralization=bool(args.enable_task_neutralization),
            disable_task_neutralization=bool(args.disable_task_neutralization),
            downstream_task_id=downstream_task_id,
            obs_dim=obs_dim,
            eval_episodes_post_training=eval_episodes_post_training,
            device=device,
        )
        family_results.append(result)

        member_dir = _member_dir(family_dir, family_set.member_idx)
        torch.save(result.actor.state_dict(), member_dir / "adapted_actor.pt")
        torch.save(result.critic.state_dict(), member_dir / "adapted_critic.pt")
        torch.save(result.training_data, member_dir / "training_data.pt")
        result_yaml = {
            "member_idx": int(result.member_idx),
            "source_mean_reward": float(result.source_mean_reward),
            "source_std_reward": float(result.source_std_reward),
            "source_failure_rate": float(result.source_failure_rate),
            "source_success_rate": float(result.source_success_rate),
            "downstream_mean_reward": float(result.downstream_mean_reward),
            "downstream_std_reward": float(result.downstream_std_reward),
            "downstream_failure_rate": float(result.downstream_failure_rate),
            "downstream_success_rate": float(result.downstream_success_rate),
        }
        (member_dir / "adaptation_result.yaml").write_text(
            yaml.safe_dump(result_yaml, sort_keys=False),
            encoding="utf-8",
        )
        family_artifacts[family_set.member_idx].update(
            {
                "adapted_actor_path": str(member_dir / "adapted_actor.pt"),
                "adapted_critic_path": str(member_dir / "adapted_critic.pt"),
                "training_data_path": str(member_dir / "training_data.pt"),
                "adaptation_result_path": str(member_dir / "adaptation_result.yaml"),
            },
        )

    results_by_member = {rec.member_idx: rec for rec in family_results}
    sets_by_member = {rec.member_idx: rec for rec in family_sets}
    best_result = max(family_results, key=lambda rec: (rec.downstream_success_rate, rec.downstream_mean_reward))
    best_set = sets_by_member[best_result.member_idx]

    actor_path = downstream_run_dir / "actor.pt"
    critic_path = downstream_run_dir / "critic.pt"
    training_data_path = downstream_run_dir / "training_data.pt"
    rashomon_dataset_path = downstream_run_dir / "rashomon_dataset.pt"
    family_bounds_path = downstream_run_dir / "rashomon_family_bounds.pt"
    rollout_stats_path = downstream_run_dir / "rashomon_rollout_stats.yaml"
    source_plot_path = downstream_run_dir / "trajectory_source.png"
    downstream_plot_path = downstream_run_dir / "trajectory_downstream.png"
    summary_path = downstream_run_dir / "run_summary.yaml"

    torch.save(best_result.actor.state_dict(), actor_path)
    torch.save(best_result.critic.state_dict(), critic_path)
    torch.save(best_result.training_data, training_data_path)
    torch.save(rashomon_dataset, rashomon_dataset_path)
    torch.save(
        {
            "family_bounds_l": [rec.bounds_l for rec in family_sets],
            "family_bounds_u": [rec.bounds_u for rec in family_sets],
            "best_member_idx": int(best_result.member_idx),
        },
        family_bounds_path,
    )

    actor_for_plot = copy.deepcopy(best_result.actor).to("cpu")
    actor_for_plot.eval()
    rollout_stats = {
        "rashomon_rollouts": int(rashomon_rollouts),
        "total_state_action_pairs": int(len(rashomon_dataset)),
        "rollout_lengths": [int(x) for x in rollout_lengths],
        "rollout_length_min": int(min(rollout_lengths)),
        "rollout_length_max": int(max(rollout_lengths)),
        "rollout_length_mean": float(np.mean(rollout_lengths)),
        "rollout_length_std": float(np.std(rollout_lengths)),
    }
    rollout_stats_path.write_text(yaml.safe_dump(rollout_stats, sort_keys=False), encoding="utf-8")

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

    family_summary = []
    for family_set in family_sets:
        result = results_by_member[family_set.member_idx]
        family_summary.append(
            {
                "member_idx": int(family_set.member_idx),
                "reference_kind": family_set.reference_kind,
                "selected_inverse_temperature": int(family_set.selected_inverse_temp),
                "surrogate_threshold": float(family_set.surrogate_threshold),
                "selected_certificate_index": int(family_set.selected_cert_idx),
                "selected_certificate": float(family_set.cert_values[family_set.selected_cert_idx]),
                "sample_stats": family_set.sample_stats,
                "source_mean_reward": float(result.source_mean_reward),
                "source_std_reward": float(result.source_std_reward),
                "source_failure_rate": float(result.source_failure_rate),
                "source_success_rate": float(result.source_success_rate),
                "downstream_mean_reward": float(result.downstream_mean_reward),
                "downstream_std_reward": float(result.downstream_std_reward),
                "downstream_failure_rate": float(result.downstream_failure_rate),
                "downstream_success_rate": float(result.downstream_success_rate),
                "artifacts": family_artifacts[family_set.member_idx],
            },
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
        "task_feature_neutralization": bool(args.enable_task_neutralization and not args.disable_task_neutralization),
        "hidden_size": int(hidden_size),
        "device": device,
        "family_size": int(family_size),
        "family_workers": int(family_workers),
        "total_timesteps_budget": int(total_timesteps),
        "total_timesteps_per_run": int(total_timesteps_per_run),
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
        "early_stop_min_steps": 0,
        "early_stop_success_rate_threshold": float(early_stop_success_rate_threshold),
        "pgd_projection_distance_norm": pgd_projection_distance_norm,
        "rashomon_rollouts": int(rashomon_rollouts),
        "rashomon_n_iters_budget": int(rashomon_n_iters),
        "rashomon_n_iters_per_run": int(rashomon_n_iters_per_run),
        "surrogate_aggregation": rashomon_surrogate_aggregation,
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
        "best_member_idx": int(best_result.member_idx),
        "best_member_reference_kind": best_set.reference_kind,
        "selection_rule": "max downstream_success_rate, tie max downstream_mean_reward",
        "source_mean_reward": float(best_result.source_mean_reward),
        "source_std_reward": float(best_result.source_std_reward),
        "source_failure_rate": float(best_result.source_failure_rate),
        "source_success_rate": float(best_result.source_success_rate),
        "downstream_mean_reward": float(best_result.downstream_mean_reward),
        "downstream_std_reward": float(best_result.downstream_std_reward),
        "downstream_failure_rate": float(best_result.downstream_failure_rate),
        "downstream_success_rate": float(best_result.downstream_success_rate),
        "family_members": family_summary,
    }
    artifacts = {
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
        "training_data_path": str(training_data_path),
        "rashomon_dataset_path": str(rashomon_dataset_path),
        "rashomon_family_bounds_path": str(family_bounds_path),
        "rashomon_rollout_stats_path": str(rollout_stats_path),
        "trajectory_source_plot_path": str(source_plot_path),
        "trajectory_downstream_plot_path": str(downstream_plot_path),
        "family_members_dir": str(family_dir),
    }
    summary = {
        "run_settings": run_settings,
        "run_results": run_results,
        "artifacts": artifacts,
    }
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    print(
        f"Best family member: {best_result.member_idx} | "
        f"downstream_success_rate={best_result.downstream_success_rate:.3f} | "
        f"downstream_mean_reward={best_result.downstream_mean_reward:.3f}",
    )
    print(f"Saved actor: {actor_path}")
    print(f"Saved critic: {critic_path}")
    print(f"Saved Rashomon family summary: {summary_path}")


if __name__ == "__main__":
    main()
