"""Iterative Rashomon-expanded adaptation for LunarLander."""

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

# Allow running this file directly from experiments/pipelines/behaviour_retention/lunarlander.
_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.behaviour_retention.lunarlander.core.env.env_factory import _make_lunarlander_env
from experiments.pipelines.behaviour_retention.lunarlander.core.env.task_loading import (
    _load_task_settings,
    _resolve_lunarlander_dynamics,
)
from experiments.pipelines.behaviour_retention.lunarlander.core.methods.adapt_rashomon import (
    _load_source_hidden_size,
    compute_rashomon_bounds,
    create_source_rollout_rashomon_dataset,
    neutralize_task_feature,
)
from experiments.pipelines.behaviour_retention.lunarlander.core.methods.source_train import (
    _plot_trajectory_grid,
    build_actor_critic,
)
from experiments.pipelines.behaviour_retention.lunarlander.core.orchestration.run_paths import (
    default_outputs_root,
    default_task_settings_file,
    resolve_default_source_run_dir as _resolve_default_source_run_dir,
    seed_run_dir as _seed_run_dir,
)
from experiments.utils.ppo_utils import PPOConfig, evaluate_with_success, ppo_train


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
    actor_template: torch.nn.Module,
    sampled_param_list: list[torch.Tensor],
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    reference_actor = copy.deepcopy(actor_template).to("cpu")
    ref_params = list(reference_actor.parameters())
    if len(ref_params) != len(sampled_param_list):
        raise ValueError(
            f"Length mismatch: sampled={len(sampled_param_list)} vs actor_params={len(ref_params)}",
        )

    with torch.no_grad():
        for idx, (param, sampled_t) in enumerate(zip(ref_params, sampled_param_list)):
            if tuple(param.shape) != tuple(sampled_t.shape):
                raise ValueError(
                    f"Shape mismatch at param index {idx}: sampled={tuple(sampled_t.shape)} "
                    f"vs expected={tuple(param.shape)}",
                )
            param.copy_(sampled_t.to(device=param.device, dtype=param.dtype))

    reference_actor.eval()
    reference_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in reference_actor.state_dict().items()
    }
    return reference_actor, reference_state_dict


def _empty_training_data(obs_dim: int) -> dict[str, np.ndarray]:
    return {
        "states": np.zeros((0, obs_dim), dtype=np.float32),
        "actions": np.zeros((0,), dtype=np.int64),
        "terminated": np.zeros((0,), dtype=np.float32),
        "truncated": np.zeros((0,), dtype=np.float32),
        "safe": np.zeros((0,), dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run iterative downstream LunarLander adaptation with Rashomon-set expansion "
            "and PPO-PGD on the running union of Rashomon sets."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--task-settings-file",
        type=Path,
        default=default_task_settings_file(),
        help="Task pipeline settings YAML (legacy monolithic task settings YAML is also supported).",
    )
    parser.add_argument("--pipeline", type=str, dest="task_setting", default="default")
    parser.add_argument("--task-setting", type=str, dest="task_setting", help=argparse.SUPPRESS)
    parser.add_argument("--env-id", type=str, default=None, help="Optional env-id override.")
    parser.add_argument("--source-gravity", type=float, default=None, help="Optional source gravity override.")
    parser.add_argument(
        "--downstream-gravity",
        type=float,
        default=None,
        help="Optional downstream gravity override.",
    )
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
        default="downstream_rashomon_expanded",
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
        help="Kept for interface parity. This method always warm-starts critic from current reference.",
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
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument(
        "--eval-episodes-during-training",
        type=int,
        default=20,
        help="Number of episodes per periodic evaluation during PPO training.",
    )
    parser.add_argument(
        "--eval-episodes-post-training",
        type=int,
        default=100,
        help="Number of episodes for final post-training/reference evaluation.",
    )
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--early-stop-min-steps", type=int, default=0)
    parser.add_argument(
        "--early-stop-reward-threshold",
        type=float,
        default=None,
        help="Optional extra early-stop criterion. Defaults to None for success-rate driven stopping.",
    )
    parser.add_argument("--early-stop-failure-rate-threshold", type=float, default=None)
    parser.add_argument(
        "--early-stop-success-rate-threshold",
        type=float,
        default=None,
        help=(
            "Optional legacy override. When provided and different from --success-rate-threshold, "
            "--success-rate-threshold is used."
        ),
    )
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
        default=100,
        help="Number of source-task rollouts used to build the Rashomon dataset.",
    )
    parser.add_argument("--rashomon-n-iters", type=int, default=50_000)
    parser.add_argument("--rashomon-min-hard-spec", type=float, default=1.0)
    parser.add_argument(
        "--rashomon-surrogate-aggregation",
        type=str,
        choices=["mean", "min"],
        default="min",
    )
    parser.add_argument("--inverse-temp-start", type=int, default=10)
    parser.add_argument("--inverse-temp-max", type=int, default=1000)
    parser.add_argument("--rashomon-checkpoint", type=int, default=100)

    # Iterative Rashomon-expanded arguments.
    parser.add_argument(
        "--success-rate-threshold",
        type=float,
        default=1.0,
        help="Downstream success-rate criterion for loop termination and PPO early-stopping.",
    )
    parser.add_argument(
        "--max-loop-iterations",
        type=int,
        default=10,
        help="Maximum number of Rashomon/PPO loop iterations.",
    )
    parser.add_argument(
        "--reference-update-mode",
        type=str,
        choices=["trained", "sample_bounds"],
        default="trained",
        help=(
            "How to define next iteration reference model: "
            "'trained' uses adapted actor+critic; 'sample_bounds' resamples actor from latest bounds "
            "and keeps current reference critic."
        ),
    )
    parser.add_argument(
        "--reference-upper-bound-prob",
        type=float,
        default=0.5,
        help="Per-parameter probability of sampling upper endpoint in sample_bounds mode.",
    )
    parser.add_argument(
        "--reference-bound-sample-seed",
        type=int,
        default=None,
        help="Base random seed for reference actor bound sampling. Defaults to --seed.",
    )

    args = parser.parse_args()

    if args.eval_episodes_during_training < 2:
        raise ValueError("For LunarLander, --eval-episodes-during-training must be >= 2.")
    if args.eval_episodes_post_training < 2:
        raise ValueError("For LunarLander, --eval-episodes-post-training must be >= 2.")
    if args.rashomon_rollouts <= 0:
        raise ValueError("--rashomon-rollouts must be > 0.")
    if args.rashomon_n_iters <= 0:
        raise ValueError("--rashomon-n-iters must be > 0.")
    if args.max_loop_iterations < 0:
        raise ValueError("--max-loop-iterations must be >= 0.")
    if args.success_rate_threshold < 0.0 or args.success_rate_threshold > 1.0:
        raise ValueError("--success-rate-threshold must be in [0, 1].")
    if args.reference_upper_bound_prob < 0.0 or args.reference_upper_bound_prob > 1.0:
        raise ValueError("--reference-upper-bound-prob must be in [0, 1].")
    if args.inverse_temp_start <= 0 or args.inverse_temp_max < args.inverse_temp_start:
        raise ValueError(
            "Invalid inverse-temperature range. Require 0 < inverse-temp-start <= inverse-temp-max.",
        )
    if not bool(args.warm_start_critic):
        raise ValueError(
            "adapt_rashomon_expanded requires reference-critic warm starts at every loop iteration. "
            "Please do not pass --no-warm-start-critic.",
        )

    source_task_cfg = _load_task_settings(args.task_settings_file, args.task_setting, "source")
    downstream_task_cfg = _load_task_settings(args.task_settings_file, args.task_setting, "downstream")

    env_id = str(
        args.env_id
        or source_task_cfg.get("env_id")
        or downstream_task_cfg.get("env_id")
        or "LunarLander-v3"
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
    if not critic_ckpt.exists():
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
            rashomon_rollouts=args.rashomon_rollouts,
        )
    finally:
        source_rollout_env.close()

    print(
        f"Built source rollout Rashomon dataset: {len(rashomon_dataset)} samples "
        f"from {args.rashomon_rollouts} rollouts.",
    )

    task_feature_index = obs_dim - 1
    do_task_neutralization = (
        append_task_id
        and bool(args.enable_task_neutralization)
        and not args.disable_task_neutralization
    )

    initial_reference_actor = copy.deepcopy(source_actor)
    initial_reference_critic = copy.deepcopy(source_critic)
    if do_task_neutralization:
        neutralize_task_feature(initial_reference_actor, task_feature_index, downstream_task_id)
        neutralize_task_feature(initial_reference_critic, task_feature_index, downstream_task_id)

    reference_actor = copy.deepcopy(initial_reference_actor).to("cpu")
    reference_critic = copy.deepcopy(initial_reference_critic).to("cpu")
    reference_actor.eval()
    reference_critic.eval()

    sample_seed_base = (
        int(args.reference_bound_sample_seed)
        if args.reference_bound_sample_seed is not None
        else int(args.seed)
    )

    if (
        args.early_stop_success_rate_threshold is not None
        and float(args.early_stop_success_rate_threshold) != float(args.success_rate_threshold)
    ):
        print(
            "Ignoring --early-stop-success-rate-threshold because iterative stopping is tied to "
            f"--success-rate-threshold={args.success_rate_threshold:.3f}.",
        )

    eval_episodes_post_training = int(args.eval_episodes_post_training)

    # Evaluate initial reference policy before entering iterative loop.
    init_downstream_eval_env = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **downstream_env_kwargs,
    )
    (
        reference_downstream_mean_reward,
        reference_downstream_std_reward,
        reference_downstream_failure_rate,
        reference_downstream_success_rate,
    ) = evaluate_with_success(
        init_downstream_eval_env,
        reference_actor,
        episodes=eval_episodes_post_training,
        deterministic=True,
        device=args.device,
    )
    init_downstream_eval_env.close()

    print(
        "Initial reference downstream eval: "
        f"mean_reward={reference_downstream_mean_reward:.3f}, "
        f"failure_rate={reference_downstream_failure_rate:.3f}, "
        f"success_rate={reference_downstream_success_rate:.3f}",
    )

    interval_bounds_l: list[list[torch.Tensor]] = []
    interval_bounds_u: list[list[torch.Tensor]] = []
    loop_history: list[dict[str, Any]] = []

    latest_bounded_model: Any = None
    latest_param_bounds_l: list[torch.Tensor] = []
    latest_param_bounds_u: list[torch.Tensor] = []

    last_training_data: dict[str, np.ndarray] = _empty_training_data(obs_dim)

    total_ppo_timesteps = 0
    total_rashomon_n_iters = 0
    loop_iterations = 0

    while (
        reference_downstream_success_rate < float(args.success_rate_threshold)
        and loop_iterations < int(args.max_loop_iterations)
    ):
        iteration_idx = loop_iterations + 1
        rashomon_seed = int(args.seed + loop_iterations)

        print(
            f"\n[Loop {iteration_idx}] Computing Rashomon set around current reference actor "
            f"(n_iters={args.rashomon_n_iters}).",
        )

        (
            iter_bounds_l,
            iter_bounds_u,
            iter_bounded_model,
            iter_selected_inverse_temp,
            iter_surrogate_threshold,
            iter_cert_values,
            iter_selected_cert_idx,
        ) = compute_rashomon_bounds(
            actor=copy.deepcopy(reference_actor).to("cpu"),
            rashomon_dataset=rashomon_dataset,
            seed=rashomon_seed,
            rashomon_n_iters=int(args.rashomon_n_iters),
            min_hard_spec=float(args.rashomon_min_hard_spec),
            aggregation=str(args.rashomon_surrogate_aggregation),
            inverse_temp_start=int(args.inverse_temp_start),
            inverse_temp_max=int(args.inverse_temp_max),
            checkpoint=int(args.rashomon_checkpoint),
        )

        iter_bounds_l = [p.detach().cpu().clone() for p in iter_bounds_l]
        iter_bounds_u = [p.detach().cpu().clone() for p in iter_bounds_u]

        interval_bounds_l.append(iter_bounds_l)
        interval_bounds_u.append(iter_bounds_u)
        latest_bounded_model = iter_bounded_model
        latest_param_bounds_l = iter_bounds_l
        latest_param_bounds_u = iter_bounds_u
        total_rashomon_n_iters += int(args.rashomon_n_iters)

        ppo_cfg = PPOConfig(
            seed=int(args.seed + loop_iterations),
            total_timesteps=int(args.total_timesteps),
            eval_episodes=int(args.eval_episodes_during_training),
            rollout_steps=int(args.rollout_steps),
            update_epochs=int(args.update_epochs),
            minibatch_size=int(args.minibatch_size),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            clip_coef=float(args.clip_coef),
            ent_coef=float(args.ent_coef),
            vf_coef=float(args.vf_coef),
            lr=float(args.lr),
            max_grad_norm=float(args.max_grad_norm),
            device=args.device,
            early_stop_min_steps=int(args.early_stop_min_steps),
            early_stop_reward_threshold=(
                float(args.early_stop_reward_threshold)
                if args.early_stop_reward_threshold is not None
                else None
            ),
            early_stop_failure_rate_threshold=(
                float(args.early_stop_failure_rate_threshold)
                if args.early_stop_failure_rate_threshold is not None
                else None
            ),
            early_stop_success_rate_threshold=float(args.success_rate_threshold),
        )

        print(
            f"[Loop {iteration_idx}] Running PPO-PGD on union of {len(interval_bounds_l)} Rashomon set(s)."
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
            adapted_actor, adapted_critic, training_data = ppo_train(  # type: ignore[assignment]
                train_env,
                ppo_cfg,
                actor_warm_start=reference_actor,
                critic_warm_start=reference_critic,
                actor_param_bounds_l=interval_bounds_l,
                actor_param_bounds_u=interval_bounds_u,
                early_stop_eval_env=early_stop_eval_env,
                return_training_data=True,
            )
        finally:
            train_env.close()
            early_stop_eval_env.close()

        last_training_data = training_data
        iter_ppo_timesteps = int(len(training_data.get("states", [])))
        total_ppo_timesteps += iter_ppo_timesteps

        adapted_downstream_eval_env = _make_lunarlander_env(
            env_id,
            render_mode=None,
            **downstream_env_kwargs,
        )
        (
            adapted_downstream_mean_reward,
            adapted_downstream_std_reward,
            adapted_downstream_failure_rate,
            adapted_downstream_success_rate,
        ) = evaluate_with_success(
            adapted_downstream_eval_env,
            adapted_actor,
            episodes=eval_episodes_post_training,
            deterministic=True,
            device=args.device,
        )
        adapted_downstream_eval_env.close()

        reference_sampling: dict[str, Any] | None = None
        sampled_reference_state_dict: dict[str, torch.Tensor] | None = None

        if args.reference_update_mode == "trained":
            reference_actor = copy.deepcopy(adapted_actor).to("cpu")
            reference_critic = copy.deepcopy(adapted_critic).to("cpu")
            reference_downstream_mean_reward = float(adapted_downstream_mean_reward)
            reference_downstream_std_reward = float(adapted_downstream_std_reward)
            reference_downstream_failure_rate = float(adapted_downstream_failure_rate)
            reference_downstream_success_rate = float(adapted_downstream_success_rate)
        else:
            sample_seed = int(sample_seed_base + loop_iterations)
            sampled_param_list, reference_sampling = _sample_param_list_from_bounds(
                bounds_l=iter_bounds_l,
                bounds_u=iter_bounds_u,
                upper_prob=float(args.reference_upper_bound_prob),
                seed=sample_seed,
            )
            reference_actor, sampled_reference_state_dict = _build_reference_actor_from_sampled_params(
                actor_template=reference_actor,
                sampled_param_list=sampled_param_list,
            )
            reference_actor = reference_actor.to("cpu")
            reference_actor.eval()
            reference_critic = copy.deepcopy(reference_critic).to("cpu")
            reference_critic.eval()

            next_reference_eval_env = _make_lunarlander_env(
                env_id,
                render_mode=None,
                **downstream_env_kwargs,
            )
            (
                reference_downstream_mean_reward,
                reference_downstream_std_reward,
                reference_downstream_failure_rate,
                reference_downstream_success_rate,
            ) = evaluate_with_success(
                next_reference_eval_env,
                reference_actor,
                episodes=eval_episodes_post_training,
                deterministic=True,
                device=args.device,
            )
            next_reference_eval_env.close()

        loop_iterations += 1

        loop_record: dict[str, Any] = {
            "iteration": int(iteration_idx),
            "rashomon_seed": int(rashomon_seed),
            "rashomon_n_iters": int(args.rashomon_n_iters),
            "surrogate_threshold": float(iter_surrogate_threshold),
            "inverse_temperature": int(iter_selected_inverse_temp),
            "selected_certificate_index": int(iter_selected_cert_idx),
            "selected_certificate": float(iter_cert_values[iter_selected_cert_idx]),
            "all_certificates": [float(v) for v in iter_cert_values],
            "union_interval_count": int(len(interval_bounds_l)),
            "ppo_seed": int(ppo_cfg.seed),
            "ppo_requested_timesteps": int(ppo_cfg.total_timesteps),
            "ppo_actual_timesteps": int(iter_ppo_timesteps),
            "adapted_downstream_mean_reward": float(adapted_downstream_mean_reward),
            "adapted_downstream_std_reward": float(adapted_downstream_std_reward),
            "adapted_downstream_failure_rate": float(adapted_downstream_failure_rate),
            "adapted_downstream_success_rate": float(adapted_downstream_success_rate),
            "reference_update_mode": str(args.reference_update_mode),
            "reference_downstream_mean_reward": float(reference_downstream_mean_reward),
            "reference_downstream_std_reward": float(reference_downstream_std_reward),
            "reference_downstream_failure_rate": float(reference_downstream_failure_rate),
            "reference_downstream_success_rate": float(reference_downstream_success_rate),
            "reference_sampling": reference_sampling,
        }
        if sampled_reference_state_dict is not None:
            loop_record["sampled_reference_state_dict_keys"] = list(sampled_reference_state_dict.keys())
        loop_history.append(loop_record)

        print(
            f"[Loop {iteration_idx}] Adapted downstream success={adapted_downstream_success_rate:.3f}; "
            f"next reference success={reference_downstream_success_rate:.3f}",
        )

    final_actor = copy.deepcopy(reference_actor).to("cpu")
    final_critic = copy.deepcopy(reference_critic).to("cpu")
    final_actor.eval()
    final_critic.eval()

    source_eval_env = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **source_env_kwargs,
    )
    source_mean_reward, source_std_reward, source_failure_rate, source_success_rate = evaluate_with_success(
        source_eval_env,
        final_actor,
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
        final_actor,
        episodes=eval_episodes_post_training,
        deterministic=True,
        device=args.device,
    )
    downstream_eval_env.close()

    reached_success_threshold = bool(downstream_success_rate >= float(args.success_rate_threshold))
    if reached_success_threshold:
        stop_reason = "success_threshold_met"
    elif loop_iterations >= int(args.max_loop_iterations):
        stop_reason = "max_loop_iterations_reached"
    else:
        stop_reason = "terminated"

    downstream_run_dir = _seed_run_dir(args.outputs_root, args.task_setting, args.seed) / args.run_subdir
    downstream_run_dir.mkdir(parents=True, exist_ok=True)

    actor_path = downstream_run_dir / "actor.pt"
    critic_path = downstream_run_dir / "critic.pt"
    training_data_path = downstream_run_dir / "training_data.pt"
    rashomon_dataset_path = downstream_run_dir / "rashomon_dataset.pt"
    bounded_model_path = downstream_run_dir / "rashomon_bounded_model.pt"
    bounds_path = downstream_run_dir / "rashomon_param_bounds.pt"
    union_bounds_path = downstream_run_dir / "rashomon_union_interval_param_bounds.pt"
    rollout_stats_path = downstream_run_dir / "rashomon_rollout_stats.yaml"
    loop_history_path = downstream_run_dir / "rashomon_loop_history.yaml"
    source_plot_path = downstream_run_dir / "trajectory_source.png"
    downstream_plot_path = downstream_run_dir / "trajectory_downstream.png"
    summary_path = downstream_run_dir / "run_summary.yaml"

    torch.save(final_actor.state_dict(), actor_path)
    torch.save(final_critic.state_dict(), critic_path)
    torch.save(last_training_data, training_data_path)
    torch.save(rashomon_dataset, rashomon_dataset_path)
    torch.save(latest_bounded_model, bounded_model_path)
    torch.save(
        {
            "param_bounds_l": latest_param_bounds_l,
            "param_bounds_u": latest_param_bounds_u,
        },
        bounds_path,
    )
    torch.save(
        {
            "union_interval_param_bounds_l": interval_bounds_l,
            "union_interval_param_bounds_u": interval_bounds_u,
            "latest_param_bounds_l": latest_param_bounds_l,
            "latest_param_bounds_u": latest_param_bounds_u,
        },
        union_bounds_path,
    )

    rollout_stats: dict[str, Any] = {
        "rashomon_rollouts": int(args.rashomon_rollouts),
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

    loop_history_path.write_text(
        yaml.safe_dump(loop_history, sort_keys=False),
        encoding="utf-8",
    )

    # Plot with a CPU actor copy to avoid device-mismatch issues in rendering helpers.
    actor_for_plot = copy.deepcopy(final_actor).to("cpu")
    actor_for_plot.eval()

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

    latest_loop = loop_history[-1] if loop_history else None

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
        "warm_start_critic": True,
        "task_feature_neutralization": bool(do_task_neutralization),
        "task_feature_index": int(task_feature_index) if do_task_neutralization else None,
        "hidden_size": int(hidden_size),
        "eval_episodes_during_training": int(args.eval_episodes_during_training),
        "eval_episodes_post_training": int(eval_episodes_post_training),
        "trajectory_episodes": int(args.trajectory_episodes),
        "trajectory_max_frames_per_episode": int(args.trajectory_max_frames_per_episode),
        "rashomon_rollouts": int(args.rashomon_rollouts),
        "rashomon_n_iters": int(args.rashomon_n_iters),
        "surrogate_aggregation": str(args.rashomon_surrogate_aggregation),
        "inverse_temp_start": int(args.inverse_temp_start),
        "inverse_temp_max": int(args.inverse_temp_max),
        "rashomon_checkpoint": int(args.rashomon_checkpoint),
        "success_rate_threshold": float(args.success_rate_threshold),
        "max_loop_iterations": int(args.max_loop_iterations),
        "reference_update_mode": str(args.reference_update_mode),
        "reference_upper_bound_prob": float(args.reference_upper_bound_prob),
        "reference_bound_sample_seed": int(sample_seed_base),
        "total_timesteps_per_loop": int(args.total_timesteps),
        "noadapt_checkpoint_dir": str(source_run_dir),
        "source_checkpoint_dir": str(source_run_dir),
        "task_setting": args.task_setting,
        "task_settings_file": str(args.task_settings_file),
        "task_pipelines_file": task_pipelines_file,
        "task_definitions_file": task_definitions_file,
        "resolved_pipeline_name": resolved_pipeline_name,
        "resolved_source_definition_name": resolved_source_definition_name,
        "resolved_downstream_definition_name": resolved_downstream_definition_name,
    }

    run_results = {
        "rashomon_dataset_size": int(len(rashomon_dataset)),
        "surrogate_threshold": (
            float(latest_loop["surrogate_threshold"]) if latest_loop is not None else None
        ),
        "inverse_temperature": (
            int(latest_loop["inverse_temperature"]) if latest_loop is not None else None
        ),
        "selected_certificate_index": (
            int(latest_loop["selected_certificate_index"]) if latest_loop is not None else None
        ),
        "selected_certificate": (
            float(latest_loop["selected_certificate"]) if latest_loop is not None else None
        ),
        "all_certificates": (
            list(latest_loop["all_certificates"]) if latest_loop is not None else []
        ),
        "source_mean_reward": float(source_mean_reward),
        "source_std_reward": float(source_std_reward),
        "source_failure_rate": float(source_failure_rate),
        "source_success_rate": float(source_success_rate),
        "downstream_mean_reward": float(downstream_mean_reward),
        "downstream_std_reward": float(downstream_std_reward),
        "downstream_failure_rate": float(downstream_failure_rate),
        "downstream_success_rate": float(downstream_success_rate),
        "total_ppo_timesteps": int(total_ppo_timesteps),
        "total_rashomon_n_iters": int(total_rashomon_n_iters),
        "loop_iterations": int(loop_iterations),
        "reached_success_threshold": bool(reached_success_threshold),
        "stop_reason": str(stop_reason),
        "loop_history": loop_history,
    }

    artifacts = {
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
        "training_data_path": str(training_data_path),
        "rashomon_dataset_path": str(rashomon_dataset_path),
        "rashomon_bounded_model_path": str(bounded_model_path),
        "rashomon_param_bounds_path": str(bounds_path),
        "rashomon_union_interval_param_bounds_path": str(union_bounds_path),
        "rashomon_rollout_stats_path": str(rollout_stats_path),
        "rashomon_loop_history_path": str(loop_history_path),
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
    print(
        f"Loop summary: iterations={loop_iterations}, total_ppo_timesteps={total_ppo_timesteps}, "
        f"total_rashomon_n_iters={total_rashomon_n_iters}, stop_reason={stop_reason}",
    )
    print(f"Saved actor: {actor_path}")
    print(f"Saved critic: {critic_path}")
    print(f"Saved Rashomon dataset: {rashomon_dataset_path}")
    print(f"Saved Rashomon bounded model: {bounded_model_path}")
    print(f"Saved union Rashomon bounds: {union_bounds_path}")
    print(f"Saved source trajectory grid: {source_plot_path}")
    print(f"Saved downstream trajectory grid: {downstream_plot_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
