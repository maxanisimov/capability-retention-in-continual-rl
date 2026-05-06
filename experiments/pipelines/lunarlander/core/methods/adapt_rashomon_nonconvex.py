"""Adapt source LunarLander policy via non-convex Rashomon union-PGD."""

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
    compute_nonconvex_rashomon_bounds,
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


def _strip_checkpoint_for_yaml(checkpoint: dict[str, Any]) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    for key, value in checkpoint.items():
        if key == "bounded_model":
            continue
        rec[key] = value
    return rec


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


def _required_cfg_value(
    cfg: dict[str, Any],
    key: str,
    *,
    cfg_name: str,
) -> Any:
    if key not in cfg:
        raise ValueError(f"Missing required setting '{key}' in {cfg_name}.")
    return cfg[key]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run downstream LunarLander adaptation with non-convex Rashomon union-PGD.",
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
        default="downstream_rashomon_nonconvex",
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
        help=argparse.SUPPRESS,
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
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--eval-episodes-post-training",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--rollout-steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--update-epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--minibatch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gamma", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gae-lambda", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--clip-coef", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ent-coef", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--vf-coef", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lr", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-grad-norm", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--early-stop-min-steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--early-stop-reward-threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--early-stop-failure-rate-threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--early-stop-success-rate-threshold", type=float, default=None, help=argparse.SUPPRESS)
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

    # Non-convex Rashomon arguments.
    parser.add_argument(
        "--rashomon-rollouts",
        type=int,
        default=None,
        help="Optional override. Defaults to settings/adaptation/rashomon.yaml for this pipeline.",
    )
    parser.add_argument(
        "--rashomon-n-iters",
        type=int,
        default=None,
        help=(
            "Optional override for the total Rashomon optimization iteration budget. "
            "Each convex subset receives int(rashomon_n_iters / n_convex_sets)."
        ),
    )
    parser.add_argument(
        "--n-convex-sets",
        type=int,
        default=None,
        help="Number of convex Rashomon subsets to construct for the non-convex union.",
    )
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
    parser.add_argument(
        "--save-rashomon-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist full non-convex Rashomon checkpoint objects to disk.",
    )
    args = parser.parse_args()

    if args.enable_task_neutralization and args.disable_task_neutralization:
        raise ValueError("Cannot set both --enable-task-neutralization and --disable-task-neutralization.")
    ppo_override_args = {
        "--total-timesteps": args.total_timesteps,
        "--total-timesteps-override": args.total_timesteps_override,
        "--eval-episodes-during-training": args.eval_episodes_during_training,
        "--eval-episodes-post-training": args.eval_episodes_post_training,
        "--rollout-steps": args.rollout_steps,
        "--update-epochs": args.update_epochs,
        "--minibatch-size": args.minibatch_size,
        "--gamma": args.gamma,
        "--gae-lambda": args.gae_lambda,
        "--clip-coef": args.clip_coef,
        "--ent-coef": args.ent_coef,
        "--vf-coef": args.vf_coef,
        "--lr": args.lr,
        "--max-grad-norm": args.max_grad_norm,
        "--early-stop-min-steps": args.early_stop_min_steps,
        "--early-stop-reward-threshold": args.early_stop_reward_threshold,
        "--early-stop-failure-rate-threshold": args.early_stop_failure_rate_threshold,
        "--early-stop-success-rate-threshold": args.early_stop_success_rate_threshold,
    }
    provided_ppo_overrides = [name for name, value in ppo_override_args.items() if value is not None]
    if provided_ppo_overrides:
        raise ValueError(
            "PPO configuration for non-convex Rashomon adaptation must come from "
            f"--adapt-settings-file; remove these CLI overrides: {', '.join(provided_ppo_overrides)}.",
        )

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
    ppo_cfg_name = f"{args.adapt_settings_file}:{args.task_setting}.ppo"
    downstream_eval_cfg_name = f"{args.adapt_settings_file}:{args.task_setting}.downstream_eval"
    total_timesteps = int(_required_cfg_value(adapt_ppo_cfg, "total_timesteps", cfg_name=ppo_cfg_name))
    eval_episodes_during_training = int(
        _required_cfg_value(
            adapt_ppo_cfg,
            "eval_episodes_during_training",
            cfg_name=ppo_cfg_name,
        ),
    )
    eval_episodes_post_training = int(
        _required_cfg_value(
            downstream_eval_cfg,
            "episodes_post_training",
            cfg_name=downstream_eval_cfg_name,
        ),
    )
    rollout_steps = int(_required_cfg_value(adapt_ppo_cfg, "rollout_steps", cfg_name=ppo_cfg_name))
    update_epochs = int(_required_cfg_value(adapt_ppo_cfg, "update_epochs", cfg_name=ppo_cfg_name))
    minibatch_size = int(_required_cfg_value(adapt_ppo_cfg, "minibatch_size", cfg_name=ppo_cfg_name))
    gamma = float(_required_cfg_value(adapt_ppo_cfg, "gamma", cfg_name=ppo_cfg_name))
    gae_lambda = float(_required_cfg_value(adapt_ppo_cfg, "gae_lambda", cfg_name=ppo_cfg_name))
    clip_coef = float(_required_cfg_value(adapt_ppo_cfg, "clip_coef", cfg_name=ppo_cfg_name))
    ent_coef = float(_required_cfg_value(adapt_ppo_cfg, "ent_coef", cfg_name=ppo_cfg_name))
    vf_coef = float(_required_cfg_value(adapt_ppo_cfg, "vf_coef", cfg_name=ppo_cfg_name))
    lr = float(_required_cfg_value(adapt_ppo_cfg, "lr", cfg_name=ppo_cfg_name))
    max_grad_norm = float(_required_cfg_value(adapt_ppo_cfg, "max_grad_norm", cfg_name=ppo_cfg_name))
    early_stop_min_steps = int(_required_cfg_value(adapt_ppo_cfg, "early_stop_min_steps", cfg_name=ppo_cfg_name))
    early_stop_reward_threshold_raw = _required_cfg_value(
        adapt_ppo_cfg,
        "early_stop_reward_threshold",
        cfg_name=ppo_cfg_name,
    )
    early_stop_failure_rate_threshold_raw = _required_cfg_value(
        adapt_ppo_cfg,
        "early_stop_failure_rate_threshold",
        cfg_name=ppo_cfg_name,
    )
    early_stop_success_rate_threshold_raw = _required_cfg_value(
        adapt_ppo_cfg,
        "early_stop_success_rate_threshold",
        cfg_name=ppo_cfg_name,
    )
    pgd_projection_distance_norm = str(
        _required_cfg_value(adapt_ppo_cfg, "pgd_projection_distance_norm", cfg_name=ppo_cfg_name),
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
    n_convex_sets = int(
        args.n_convex_sets
        if args.n_convex_sets is not None
        else rashomon_cfg.get("n_convex_sets", 3),
    )
    if n_convex_sets <= 0:
        raise ValueError("--n-convex-sets (or resolved default) must be > 0.")
    rashomon_n_iters_per_convex_set = int(rashomon_n_iters / n_convex_sets)
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
        raise ValueError(f"'total_timesteps' in {ppo_cfg_name} must be > 0.")
    if eval_episodes_during_training <= 0:
        raise ValueError(f"'eval_episodes_during_training' in {ppo_cfg_name} must be > 0.")
    if eval_episodes_post_training <= 0:
        raise ValueError(f"'episodes_post_training' in {downstream_eval_cfg_name} must be > 0.")
    if rollout_steps <= 0:
        raise ValueError(f"'rollout_steps' in {ppo_cfg_name} must be > 0.")
    if update_epochs <= 0:
        raise ValueError(f"'update_epochs' in {ppo_cfg_name} must be > 0.")
    if minibatch_size <= 0:
        raise ValueError(f"'minibatch_size' in {ppo_cfg_name} must be > 0.")
    if early_stop_min_steps != 0:
        raise ValueError(
            "Non-convex Rashomon adaptation requires early_stop_min_steps=0 so PPO runs "
            "the pre-update evaluation check before the first policy update.",
        )
    if early_stop_success_rate_threshold is None:
        raise ValueError(
            "Non-convex Rashomon adaptation requires "
            f"'early_stop_success_rate_threshold' in {ppo_cfg_name} so evaluation can early-stop by success rate.",
        )
    if not 0.0 <= early_stop_success_rate_threshold <= 1.0:
        raise ValueError(
            f"'early_stop_success_rate_threshold' in {ppo_cfg_name} must be in [0, 1].",
        )
    if pgd_projection_distance_norm not in {"l2", "l1", "linf"}:
        raise ValueError(
            f"'pgd_projection_distance_norm' in {ppo_cfg_name} must be one of: l2, l1, linf.",
        )
    if rashomon_rollouts <= 0:
        raise ValueError("--rashomon-rollouts (or resolved default) must be > 0.")
    if rashomon_n_iters <= 0:
        raise ValueError("--rashomon-n-iters (or resolved default) must be > 0.")
    if rashomon_n_iters_per_convex_set <= 0:
        raise ValueError(
            "--rashomon-n-iters / --n-convex-sets must provide at least one iteration per convex set.",
        )
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

    union_interval_bounds_l, union_interval_bounds_u, nonconvex_checkpoints = compute_nonconvex_rashomon_bounds(
        actor=copy.deepcopy(source_actor).to("cpu"),
        rashomon_dataset=rashomon_dataset,
        seed=int(args.seed),
        n_iters=int(rashomon_n_iters),
        min_hard_spec=float(rashomon_min_hard_spec),
        aggregation=str(rashomon_surrogate_aggregation),
        inverse_temp_start=int(inverse_temp_start),
        inverse_temp_max=int(inverse_temp_max),
        checkpoint=int(rashomon_checkpoint),
        n_convex_sets_budget=int(n_convex_sets),
        return_checkpoints=True,
    )

    if nonconvex_checkpoints is None:
        raise RuntimeError("Expected non-convex Rashomon checkpoints, got None.")
    if len(union_interval_bounds_l) != int(n_convex_sets):
        raise RuntimeError(
            "Unexpected number of convex subsets returned by compute_nonconvex_rashomon_bounds: "
            f"expected={n_convex_sets}, got={len(union_interval_bounds_l)}.",
        )

    fixed_inverse_temp = int(nonconvex_checkpoints[0]["selected_inverse_temperature"])
    surrogate_threshold = float(nonconvex_checkpoints[0]["surrogate_threshold"])
    selected_certificates = [float(rec["selected_certificate"]) for rec in nonconvex_checkpoints]

    print(
        "Non-convex Rashomon bounds ready: "
        f"sets={len(union_interval_bounds_l)} | "
        f"total_iters={rashomon_n_iters} | "
        f"iters_per_set={rashomon_n_iters_per_convex_set} | "
        f"fixed_inverse_temp={fixed_inverse_temp} | "
        f"aggregation={rashomon_surrogate_aggregation} | "
        f"min_hard_spec={rashomon_min_hard_spec:.3f}",
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
        seed=int(args.seed),
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
        pgd_projection_distance_norm=pgd_projection_distance_norm,
    )

    print(
        f"Adapting LunarLander with non-convex Rashomon union-PGD | "
        f"source_task={source_task_id} -> downstream_task={downstream_task_id} | "
        f"warm_critic={args.warm_start_critic} | "
        f"task_neutralization={do_task_neutralization} | "
        f"intervals={len(union_interval_bounds_l)}",
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
    try:
        actor, critic, training_data = ppo_train(  # type: ignore[assignment]
            train_env,
            ppo_cfg,
            actor_warm_start=source_actor,
            critic_warm_start=(source_critic if args.warm_start_critic else None),
            actor_param_bounds_l=union_interval_bounds_l,
            actor_param_bounds_u=union_interval_bounds_u,
            early_stop_eval_env=early_stop_eval_env,
            return_training_data=True,
        )
    finally:
        train_env.close()
        early_stop_eval_env.close()

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
    nonconvex_bounds_path = downstream_run_dir / "rashomon_nonconvex_interval_param_bounds.pt"
    nonconvex_checkpoints_path = downstream_run_dir / "nonconvex_rashomon_checkpoints.pt"
    rollout_stats_path = downstream_run_dir / "rashomon_rollout_stats.yaml"
    source_plot_path = downstream_run_dir / "trajectory_source.png"
    downstream_plot_path = downstream_run_dir / "trajectory_downstream.png"
    summary_path = downstream_run_dir / "run_summary.yaml"

    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    torch.save(training_data, training_data_path)
    torch.save(rashomon_dataset, rashomon_dataset_path)
    torch.save(
        {
            "nonconvex_param_bounds_l": union_interval_bounds_l,
            "nonconvex_param_bounds_u": union_interval_bounds_u,
            "n_convex_sets": int(len(union_interval_bounds_l)),
            "rashomon_n_iters": int(rashomon_n_iters),
            "rashomon_n_iters_per_convex_set": int(rashomon_n_iters_per_convex_set),
        },
        nonconvex_bounds_path,
    )
    if args.save_rashomon_checkpoints:
        torch.save(nonconvex_checkpoints, nonconvex_checkpoints_path)

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

    checkpoint_summaries = [_strip_checkpoint_for_yaml(rec) for rec in nonconvex_checkpoints]

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
        "pgd_projection_distance_norm": pgd_projection_distance_norm,
        "trajectory_episodes": int(args.trajectory_episodes),
        "trajectory_max_frames_per_episode": int(args.trajectory_max_frames_per_episode),
        "rashomon_rollouts": int(rashomon_rollouts),
        "rashomon_n_iters": int(rashomon_n_iters),
        "rashomon_n_iters_per_convex_set": int(rashomon_n_iters_per_convex_set),
        "n_convex_sets": int(n_convex_sets),
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
        "inverse_temperature": int(fixed_inverse_temp),
        "n_convex_sets": int(len(union_interval_bounds_l)),
        "selected_certificates": selected_certificates,
        "best_selected_certificate": float(max(selected_certificates)),
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
        "rashomon_nonconvex_interval_param_bounds_path": str(nonconvex_bounds_path),
        "nonconvex_rashomon_checkpoints_path": (
            str(nonconvex_checkpoints_path) if args.save_rashomon_checkpoints else None
        ),
        "rashomon_rollout_stats_path": str(rollout_stats_path),
        "trajectory_source_plot_path": str(source_plot_path),
        "trajectory_downstream_plot_path": str(downstream_plot_path),
    }
    summary = {
        "run_settings": run_settings,
        "run_results": run_results,
        "nonconvex_rashomon": {
            "checkpoints": checkpoint_summaries,
        },
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
    print(f"Saved non-convex Rashomon bounds: {nonconvex_bounds_path}")
    if args.save_rashomon_checkpoints:
        print(f"Saved non-convex Rashomon checkpoints: {nonconvex_checkpoints_path}")
    print(f"Saved source trajectory grid: {source_plot_path}")
    print(f"Saved downstream trajectory grid: {downstream_plot_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
