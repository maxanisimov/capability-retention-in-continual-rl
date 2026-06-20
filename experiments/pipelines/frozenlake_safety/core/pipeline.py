"""Training and adaptation implementation for the FrozenLake safety pipeline."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from experiments.pipelines.frozenlake_safety.core.config import OBS_DIM, PipelineConfig, get_pipeline_config
from experiments.pipelines.frozenlake_safety.core.env import make_env, state_index_to_obs
from experiments.pipelines.frozenlake_safety.core.models import build_actor_critic
from experiments.pipelines.frozenlake_safety.core.paths import (
    NOADAPT_POLICY_SUBDIR,
    default_outputs_root,
    mode_run_dir,
    resolve_source_run_dir,
)
from experiments.pipelines.frozenlake_safety.core.safety import (
    RolloutResult,
    build_noadapt_supervised_payload,
    create_rashomon_dataset,
    finetune_on_allowed_actions,
    greedy_action,
    rollout_greedy_policy,
    safe_action_mask_for_state,
    to_tensor_dataset,
    traversable_nonterminal_states,
    validate_rashomon_payload,
)
from experiments.utils.ewc_ppo import EWCPPOConfig, compute_ewc_state, ewc_ppo_train
from experiments.utils.gymnasium_utils import plot_episode
from experiments.utils.ppo_utils import PPOConfig, evaluate_with_success, ppo_train


class _NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: object) -> bool:
        return True


def _set_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _source_ppo_config(
    cfg: PipelineConfig,
    *,
    seed: int,
    device: str,
    total_timesteps: int,
) -> PPOConfig:
    return PPOConfig(
        seed=seed,
        total_timesteps=total_timesteps,
        eval_episodes=cfg.eval_episodes,
        rollout_steps=cfg.source_rollout_steps,
        update_epochs=cfg.source_update_epochs,
        minibatch_size=cfg.source_minibatch_size,
        gamma=cfg.source_gamma,
        gae_lambda=cfg.source_gae_lambda,
        clip_coef=cfg.source_clip_coef,
        ent_coef=cfg.source_ent_coef,
        vf_coef=cfg.source_vf_coef,
        lr=cfg.source_lr,
        max_grad_norm=cfg.source_max_grad_norm,
        device=device,
        early_stop_min_steps=0,
        early_stop_reward_threshold=cfg.source_early_stop_reward_threshold,
        early_stop_failure_rate_threshold=cfg.source_early_stop_failure_rate_threshold,
        early_stop_success_rate_threshold=None,
    )


def _downstream_ppo_config(
    cfg: PipelineConfig,
    *,
    seed: int,
    device: str,
    total_timesteps: int,
) -> PPOConfig:
    return PPOConfig(
        seed=seed,
        total_timesteps=total_timesteps,
        eval_episodes=cfg.eval_episodes,
        rollout_steps=cfg.downstream_rollout_steps,
        update_epochs=cfg.downstream_update_epochs,
        minibatch_size=cfg.downstream_minibatch_size,
        gamma=cfg.downstream_gamma,
        gae_lambda=cfg.downstream_gae_lambda,
        clip_coef=cfg.downstream_clip_coef,
        ent_coef=cfg.downstream_ent_coef,
        vf_coef=cfg.downstream_vf_coef,
        lr=cfg.downstream_lr,
        max_grad_norm=cfg.downstream_max_grad_norm,
        device=device,
        early_stop_min_steps=0,
        early_stop_reward_threshold=cfg.downstream_early_stop_reward_threshold,
        early_stop_failure_rate_threshold=None,
        early_stop_success_rate_threshold=None,
    )


def _ppo_config(
    cfg: PipelineConfig,
    *,
    seed: int,
    device: str,
    total_timesteps: int,
    source: bool,
) -> PPOConfig:
    if source:
        return _source_ppo_config(cfg, seed=seed, device=device, total_timesteps=total_timesteps)
    return _downstream_ppo_config(cfg, seed=seed, device=device, total_timesteps=total_timesteps)


def _ppo_config_dict(ppo_cfg: PPOConfig) -> dict[str, Any]:
    return {
        "seed": int(ppo_cfg.seed),
        "total_timesteps": int(ppo_cfg.total_timesteps),
        "eval_episodes": int(ppo_cfg.eval_episodes),
        "rollout_steps": int(ppo_cfg.rollout_steps),
        "update_epochs": int(ppo_cfg.update_epochs),
        "minibatch_size": int(ppo_cfg.minibatch_size),
        "gamma": float(ppo_cfg.gamma),
        "gae_lambda": float(ppo_cfg.gae_lambda),
        "clip_coef": float(ppo_cfg.clip_coef),
        "ent_coef": float(ppo_cfg.ent_coef),
        "vf_coef": float(ppo_cfg.vf_coef),
        "lr": float(ppo_cfg.lr),
        "max_grad_norm": float(ppo_cfg.max_grad_norm),
        "device": str(ppo_cfg.device),
        "early_stop_min_steps": int(ppo_cfg.early_stop_min_steps),
        "early_stop_reward_threshold": (
            None if ppo_cfg.early_stop_reward_threshold is None else float(ppo_cfg.early_stop_reward_threshold)
        ),
        "early_stop_failure_rate_threshold": (
            None
            if ppo_cfg.early_stop_failure_rate_threshold is None
            else float(ppo_cfg.early_stop_failure_rate_threshold)
        ),
        "early_stop_success_rate_threshold": (
            None
            if ppo_cfg.early_stop_success_rate_threshold is None
            else float(ppo_cfg.early_stop_success_rate_threshold)
        ),
    }


def _ewc_ppo_config(
    cfg: PipelineConfig,
    *,
    seed: int,
    device: str,
    total_timesteps: int,
    ewc_lambda: float,
) -> EWCPPOConfig:
    base = _downstream_ppo_config(
        cfg,
        seed=seed,
        device=device,
        total_timesteps=total_timesteps,
    )
    return EWCPPOConfig(
        **base.__dict__,
        ewc_lambda=ewc_lambda,
        ewc_apply_to_critic=cfg.ewc_apply_to_critic,
    )


def _rashomon_ppo_config(
    cfg: PipelineConfig,
    *,
    seed: int,
    device: str,
    total_timesteps: int,
) -> PPOConfig:
    return _downstream_ppo_config(cfg, seed=seed, device=device, total_timesteps=total_timesteps)


def _make_source_env(cfg: PipelineConfig, *, shaped: bool, render_mode: str | None = None):
    return make_env(
        cfg.source_map,
        task_num=cfg.source_task_num,
        max_episode_steps=cfg.max_episode_steps,
        shaped=shaped,
        render_mode=render_mode,
    )


def _make_downstream_env(cfg: PipelineConfig, *, shaped: bool, render_mode: str | None = None):
    return make_env(
        cfg.downstream_map,
        task_num=cfg.downstream_task_num,
        max_episode_steps=cfg.max_episode_steps,
        shaped=shaped,
        render_mode=render_mode,
    )


def _load_actor_critic(
    cfg: PipelineConfig,
    *,
    source_dir: Path,
    map_location: str = "cpu",
) -> tuple[torch.nn.Sequential, torch.nn.Sequential]:
    actor, critic = build_actor_critic(
        obs_dim=OBS_DIM,
        hidden=cfg.hidden,
        activation=cfg.activation,
    )
    actor_path = source_dir / "actor.pt"
    critic_path = source_dir / "critic.pt"
    if not actor_path.exists():
        raise FileNotFoundError(f"Source actor checkpoint not found: {actor_path}")
    if not critic_path.exists():
        raise FileNotFoundError(f"Source critic checkpoint not found: {critic_path}")
    actor.load_state_dict(torch.load(actor_path, map_location=map_location))
    critic.load_state_dict(torch.load(critic_path, map_location=map_location))
    return actor, critic


def _evaluate_both_tasks(
    cfg: PipelineConfig,
    *,
    actor: torch.nn.Module,
    device: str,
    seed: int,
) -> dict[str, Any]:
    source_env = _make_source_env(cfg, shaped=False)
    source_mean, source_std, source_failure, source_success = evaluate_with_success(
        source_env,
        actor,
        episodes=cfg.eval_episodes,
        deterministic=True,
        device=device,
    )
    source_env.close()

    downstream_env = _make_downstream_env(cfg, shaped=False)
    downstream_mean, downstream_std, downstream_failure, downstream_success = evaluate_with_success(
        downstream_env,
        actor,
        episodes=cfg.eval_episodes,
        deterministic=True,
        device=device,
    )
    downstream_env.close()

    task_metrics = {
        "source": _compute_task_policy_metrics(
            cfg,
            actor=actor,
            task="source",
            seed=seed,
            device=device,
        ),
        "downstream": _compute_task_policy_metrics(
            cfg,
            actor=actor,
            task="downstream",
            seed=seed,
            device=device,
        ),
    }

    return {
        "source_mean_reward": float(source_mean),
        "source_std_reward": float(source_std),
        "source_failure_rate": float(source_failure),
        "source_success_rate": float(source_success),
        "downstream_mean_reward": float(downstream_mean),
        "downstream_std_reward": float(downstream_std),
        "downstream_failure_rate": float(downstream_failure),
        "downstream_success_rate": float(downstream_success),
        "task_metrics": task_metrics,
        "source_safety_critical_state_safety_rate": task_metrics["source"][
            "safety_critical_state_safety_rate"
        ],
        "source_greedy_trajectory_safety": task_metrics["source"]["greedy_trajectory_safety"],
        "source_total_reward": task_metrics["source"]["total_reward"],
        "downstream_safety_critical_state_safety_rate": task_metrics["downstream"][
            "safety_critical_state_safety_rate"
        ],
        "downstream_greedy_trajectory_safety": task_metrics["downstream"]["greedy_trajectory_safety"],
        "downstream_total_reward": task_metrics["downstream"]["total_reward"],
    }


def _safety_critical_states(env_map: list[str] | tuple[str, ...]) -> list[int]:
    return [
        state_index
        for state_index in traversable_nonterminal_states(env_map)
        if float(safe_action_mask_for_state(env_map, state_index).sum()) < 4.0
    ]


def _safety_critical_state_metrics(
    actor: torch.nn.Module,
    *,
    env_map: list[str] | tuple[str, ...],
    task_num: float,
    device: str | torch.device,
) -> dict[str, Any]:
    critical_states = _safety_critical_states(env_map)
    safe_count = 0
    failures: list[dict[str, int]] = []

    for state_index in critical_states:
        obs = state_index_to_obs(state_index, env_map, task_num)
        action = greedy_action(actor, obs, device=device)
        safe_mask = safe_action_mask_for_state(env_map, state_index)
        if bool(safe_mask[action] > 0):
            safe_count += 1
        else:
            failures.append(
                {
                    "state_index": int(state_index),
                    "action": int(action),
                },
            )

    n_critical = len(critical_states)
    safety_rate = (safe_count / n_critical) if n_critical else None
    return {
        "safety_critical_state_safety_rate": (float(safety_rate) if safety_rate is not None else None),
        "safety_critical_state_safe_count": int(safe_count),
        "safety_critical_state_count": int(n_critical),
        "safety_critical_state_failures": failures,
    }


def _compute_task_policy_metrics(
    cfg: PipelineConfig,
    *,
    actor: torch.nn.Module,
    task: str,
    seed: int,
    device: str | torch.device,
) -> dict[str, Any]:
    if task == "source":
        env_map = cfg.source_map
        task_num = cfg.source_task_num
        env = _make_source_env(cfg, shaped=False)
    elif task == "downstream":
        env_map = cfg.downstream_map
        task_num = cfg.downstream_task_num
        env = _make_downstream_env(cfg, shaped=False)
    else:
        raise ValueError(f"Unsupported task '{task}'.")

    try:
        rollout = rollout_greedy_policy(actor, env, seed=seed, device=device)
    finally:
        env.close()

    critical_metrics = _safety_critical_state_metrics(
        actor,
        env_map=env_map,
        task_num=task_num,
        device=device,
    )
    return {
        **critical_metrics,
        "greedy_trajectory_safety": float(1.0 - rollout.failure_rate),
        "greedy_trajectory_failed": bool(rollout.failed),
        "total_reward": float(rollout.total_reward),
        "greedy_trajectory_steps": int(len(rollout.steps)),
        "greedy_trajectory_terminated": bool(rollout.terminated),
        "greedy_trajectory_truncated": bool(rollout.truncated),
    }


def _save_trajectory_plot(
    *,
    cfg: PipelineConfig,
    actor: torch.nn.Module,
    task: str,
    seed: int,
    path: Path,
) -> None:
    if task == "source":
        env = _make_source_env(cfg, shaped=False, render_mode="rgb_array")
        title = f"NoAdapt policy on source task: {cfg.layout}"
    elif task == "downstream":
        env = _make_downstream_env(cfg, shaped=False, render_mode="rgb_array")
        title = f"Policy on downstream task: {cfg.layout}"
    else:
        raise ValueError(f"Unsupported task '{task}'.")
    try:
        plot_episode(
            env=env,
            actor=actor.cpu(),
            seed=seed,
            deterministic=True,
            save_path=str(path),
            title=title,
        )
    finally:
        env.close()


def _write_summary(
    run_dir: Path,
    *,
    run_settings: dict[str, Any],
    run_results: dict[str, Any],
    artifacts: dict[str, Any],
) -> None:
    summary = {
        "run_settings": run_settings,
        "run_results": run_results,
        "artifacts": artifacts,
        **run_settings,
        **run_results,
        **artifacts,
    }
    (run_dir / "run_summary.yaml").write_text(
        yaml.dump(summary, Dumper=_NoAliasSafeDumper, sort_keys=False),
        encoding="utf-8",
    )


def _assert_successful_source_rollout(rollout: RolloutResult, *, label: str) -> None:
    if rollout.total_reward < 1.0 or rollout.failure_rate > 0.0:
        raise RuntimeError(
            f"{label} source rollout is not optimal and safe: "
            f"reward={rollout.total_reward:.3f}, failure_rate={rollout.failure_rate:.3f}.",
        )


def train_source(args: argparse.Namespace) -> Path:
    cfg = get_pipeline_config(args.layout)
    _set_seeds(args.seed)
    total_timesteps = (
        int(args.total_timesteps_override)
        if args.total_timesteps_override is not None
        else cfg.source_total_timesteps
    )
    run_dir = mode_run_dir(args.outputs_root, args.layout, args.seed, "source")
    run_dir.mkdir(parents=True, exist_ok=True)

    actor, critic = build_actor_critic(
        obs_dim=OBS_DIM,
        hidden=cfg.hidden,
        activation=cfg.activation,
    )
    ppo_cfg = _source_ppo_config(
        cfg,
        seed=args.seed,
        device=args.device,
        total_timesteps=total_timesteps,
    )

    print(f"Training source/noadapt policy for {args.layout} seed={args.seed}.")
    train_env = _make_source_env(cfg, shaped=True)
    early_stop_env = _make_source_env(cfg, shaped=False)
    actor, critic, training_data = ppo_train(  # type: ignore[assignment]
        train_env,
        ppo_cfg,
        actor_warm_start=actor,
        critic_warm_start=critic,
        early_stop_eval_env=early_stop_env,
        return_training_data=True,
    )
    train_env.close()
    early_stop_env.close()
    actor.cpu()
    critic.cpu()

    source_rollout_env = _make_source_env(cfg, shaped=False)
    before_finetune = rollout_greedy_policy(actor, source_rollout_env, seed=args.seed, device="cpu")
    source_rollout_env.close()
    _assert_successful_source_rollout(before_finetune, label="Pre-finetune")

    rashomon_payload = create_rashomon_dataset(cfg.source_map, task_num=cfg.source_task_num)
    supervised_payload = build_noadapt_supervised_payload(
        rashomon_payload,
        env_map=cfg.source_map,
        trajectory_steps=before_finetune.steps,
    )
    finetune_result = finetune_on_allowed_actions(
        actor,
        supervised_payload,
        trajectory_steps=before_finetune.steps,
        env_map=cfg.source_map,
        task_num=cfg.source_task_num,
        lr=float(args.safety_finetune_lr or cfg.safety_finetune_lr),
        max_epochs=int(args.safety_finetune_max_epochs or cfg.safety_finetune_max_epochs),
        seed=args.seed,
        device=args.device,
        verbose=True,
    )
    if not finetune_result["reached_target"]:
        raise RuntimeError(f"Safety fine-tuning failed to reach target: {finetune_result}")

    actor.cpu()
    source_rollout_env = _make_source_env(cfg, shaped=False)
    after_finetune = rollout_greedy_policy(actor, source_rollout_env, seed=args.seed, device="cpu")
    source_rollout_env.close()
    _assert_successful_source_rollout(after_finetune, label="Post-finetune")

    actor_path = run_dir / "actor.pt"
    critic_path = run_dir / "critic.pt"
    training_data_path = run_dir / "training_data.pt"
    rashomon_dataset_path = run_dir / "rashomon_dataset.pt"
    supervised_dataset_path = run_dir / "noadapt_supervised_dataset.pt"
    trajectory_pairs_path = run_dir / "source_policy_state_action_pairs.yaml"
    source_plot_path = run_dir / "trajectory_source.png"
    downstream_plot_path = run_dir / "trajectory_downstream.png"

    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    torch.save(training_data, training_data_path)
    torch.save(rashomon_payload, rashomon_dataset_path)
    torch.save(supervised_payload, supervised_dataset_path)
    trajectory_pairs_path.write_text(
        yaml.safe_dump(before_finetune.state_action_pairs(), sort_keys=False),
        encoding="utf-8",
    )
    _save_trajectory_plot(cfg=cfg, actor=actor, task="source", seed=args.seed, path=source_plot_path)
    _save_trajectory_plot(cfg=cfg, actor=actor, task="downstream", seed=args.seed, path=downstream_plot_path)

    run_results = _evaluate_both_tasks(cfg, actor=actor, device=args.device, seed=args.seed)
    run_results.update(
        {
            "pre_finetune_source_total_reward": float(before_finetune.total_reward),
            "pre_finetune_source_failure_rate": float(before_finetune.failure_rate),
            "post_finetune_source_total_reward": float(after_finetune.total_reward),
            "post_finetune_source_failure_rate": float(after_finetune.failure_rate),
            "safety_finetune_initial_accuracy": float(finetune_result["initial_accuracy"]),
            "safety_finetune_final_accuracy": float(finetune_result["final_accuracy"]),
            "safety_finetune_epochs_run": int(finetune_result["epochs_run"]),
        },
    )
    run_settings = {
        "mode": "source",
        "policy_name": NOADAPT_POLICY_SUBDIR,
        "layout": args.layout,
        "seed": args.seed,
        "activation": cfg.activation,
        "hidden": cfg.hidden,
        "reference_layout": cfg.reference_layout,
        "reference_settings_source": cfg.reference_settings_source,
        "reference_settings_files": cfg.reference_settings_files,
        "source_task_num": cfg.source_task_num,
        "downstream_task_num": cfg.downstream_task_num,
        "total_timesteps": int(total_timesteps),
        "train_shaped": True,
        "early_stop_eval_shaped": False,
        "ppo": _ppo_config_dict(ppo_cfg),
        "rashomon_dataset_size": int(rashomon_payload["state"].shape[0]),
        "outputs_root": str(args.outputs_root),
    }
    artifacts = {
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
        "training_data_path": str(training_data_path),
        "rashomon_dataset_path": str(rashomon_dataset_path),
        "noadapt_supervised_dataset_path": str(supervised_dataset_path),
        "source_policy_state_action_pairs_path": str(trajectory_pairs_path),
        "trajectory_source_plot_path": str(source_plot_path),
        "trajectory_downstream_plot_path": str(downstream_plot_path),
    }
    _write_summary(run_dir, run_settings=run_settings, run_results=run_results, artifacts=artifacts)
    print(f"Saved source/noadapt artifacts to {run_dir}")
    return run_dir


def _compute_rashomon_bounds(
    *,
    actor: torch.nn.Module,
    rashomon_payload: dict[str, torch.Tensor],
    seed: int,
    n_iters: int,
    inverse_temp_start: int,
    inverse_temp_max: int,
    checkpoint: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], object, dict[str, Any]]:
    from src.rashomon_spec import AccuracyRequirement
    from src.trainer.IntervalTrainer import IntervalTrainer

    validate_rashomon_payload(rashomon_payload)
    rashomon_dataset = to_tensor_dataset(rashomon_payload)
    valid_counts = rashomon_payload["actions"].sum(dim=1)
    max_valid = float(valid_counts.max().item())
    if max_valid <= 0:
        raise RuntimeError("Rashomon dataset contains no valid actions.")
    surrogate_threshold = max_valid / (1.0 + max_valid)

    actor.eval()
    with torch.no_grad():
        states = rashomon_payload["state"]
        masks = rashomon_payload["actions"]
        logits = actor(states)
        selected_inverse_temp: int | None = None
        min_valid_mass = float("-inf")
        for inverse_temp in range(inverse_temp_start, inverse_temp_max + 1):
            probs = torch.softmax(logits * inverse_temp, dim=1)
            valid_mass = (probs * masks).sum(dim=1)
            min_valid_mass = float(valid_mass.min().item())
            if min_valid_mass >= surrogate_threshold:
                selected_inverse_temp = inverse_temp
                break
    if selected_inverse_temp is None:
        raise ValueError(
            "Could not find inverse temperature satisfying safety Rashomon threshold: "
            f"min_valid_mass={min_valid_mass:.6f}, threshold={surrogate_threshold:.6f}.",
        )

    interval_trainer = IntervalTrainer(
        model=actor,
        accuracy=AccuracyRequirement(
            soft_min=surrogate_threshold,
            hard_min=1.0,
            soft_temperature=selected_inverse_temp,
            aggregation="min",
        ),
        min_acc_increment=0,
        seed=seed,
        n_iters=n_iters,
        checkpoint=checkpoint,
    )
    interval_trainer.compute_rashomon_set(
        dataset=rashomon_dataset,
        multi_label=True,
    )
    cert_values = [
        min((c.min_hard_acc for c in certs), default=float("-inf"))
        for certs in interval_trainer.certificates
    ]
    valid_indices = [idx for idx, value in enumerate(cert_values) if value >= 1.0]
    if not valid_indices:
        raise ValueError(f"No Rashomon certificate reached 1.0; certificates={cert_values}")
    selected_idx = valid_indices[-1]
    bounded_model = interval_trainer.bounds[selected_idx]
    param_bounds_l = [param.detach().cpu() for param in bounded_model.param_l]
    param_bounds_u = [param.detach().cpu() for param in bounded_model.param_u]
    metadata = {
        "surrogate_threshold": float(surrogate_threshold),
        "surrogate_aggregation": "min",
        "rashomon_min_hard_spec": 1.0,
        "inverse_temperature": int(selected_inverse_temp),
        "selected_certificate_index": int(selected_idx),
        "selected_certificate": float(cert_values[selected_idx]),
        "all_certificates": [float(v) for v in cert_values],
    }
    return param_bounds_l, param_bounds_u, bounded_model, metadata


def adapt_downstream(args: argparse.Namespace, *, mode: str) -> Path:
    cfg = get_pipeline_config(args.layout)
    _set_seeds(args.seed)
    source_dir = args.source_run_dir or resolve_source_run_dir(args.outputs_root, args.layout, args.seed)
    source_actor, source_critic = _load_actor_critic(cfg, source_dir=source_dir)
    total_timesteps = (
        int(args.total_timesteps_override)
        if args.total_timesteps_override is not None
        else cfg.downstream_total_timesteps
    )

    actor_bounds_l = None
    actor_bounds_u = None
    bounded_model = None
    rashomon_metadata: dict[str, Any] = {}
    rashomon_payload = None
    if mode == "downstream_rashomon":
        rashomon_dataset_path = source_dir / "rashomon_dataset.pt"
        if not rashomon_dataset_path.exists():
            raise FileNotFoundError(f"Source Rashomon dataset not found: {rashomon_dataset_path}")
        rashomon_payload = torch.load(rashomon_dataset_path, map_location="cpu", weights_only=False)
        actor_bounds_l, actor_bounds_u, bounded_model, rashomon_metadata = _compute_rashomon_bounds(
            actor=copy.deepcopy(source_actor),
            rashomon_payload=rashomon_payload,
            seed=args.seed,
            n_iters=int(args.rashomon_n_iters or cfg.rashomon_n_iters),
            inverse_temp_start=int(args.inverse_temp_start or cfg.inverse_temp_start),
            inverse_temp_max=int(args.inverse_temp_max or cfg.inverse_temp_max),
            checkpoint=int(args.rashomon_checkpoint or cfg.rashomon_checkpoint),
        )

    train_env = _make_downstream_env(cfg, shaped=True)
    early_stop_env = _make_downstream_env(cfg, shaped=False)
    if mode == "downstream_ewc":
        training_data_path = source_dir / "training_data.pt"
        if not training_data_path.exists():
            raise FileNotFoundError(f"Source training data not found: {training_data_path}")
        source_training_data = torch.load(training_data_path, map_location="cpu", weights_only=False)
        source_states = np.asarray(source_training_data["states"], dtype=np.float32)
        fisher_sample_size = max(1, min(int(args.fisher_sample_size or cfg.fisher_sample_size), len(source_states)))
        ewc_state = compute_ewc_state(
            actor=copy.deepcopy(source_actor),
            observations=source_states,
            device=args.device,
            fisher_sample_size=fisher_sample_size,
            seed=args.seed,
        )
        ppo_cfg = _ewc_ppo_config(
            cfg,
            seed=args.seed,
            device=args.device,
            total_timesteps=total_timesteps,
            ewc_lambda=float(args.ewc_lambda or cfg.ewc_lambda),
        )
        actor, critic, training_data = ewc_ppo_train(  # type: ignore[assignment]
            train_env,
            ppo_cfg,
            ewc_states=[ewc_state],
            actor_warm_start=source_actor,
            critic_warm_start=source_critic,
            early_stop_eval_env=early_stop_env,
            return_training_data=True,
        )
    else:
        if mode == "downstream_rashomon":
            ppo_cfg = _rashomon_ppo_config(
                cfg,
                seed=args.seed,
                device=args.device,
                total_timesteps=total_timesteps,
            )
        else:
            ppo_cfg = _downstream_ppo_config(
                cfg,
                seed=args.seed,
                device=args.device,
                total_timesteps=total_timesteps,
            )
        actor, critic, training_data = ppo_train(  # type: ignore[assignment]
            train_env,
            ppo_cfg,
            actor_warm_start=source_actor,
            critic_warm_start=source_critic,
            actor_param_bounds_l=actor_bounds_l,
            actor_param_bounds_u=actor_bounds_u,
            early_stop_eval_env=early_stop_env,
            return_training_data=True,
        )
    train_env.close()
    early_stop_env.close()
    actor.cpu()
    critic.cpu()

    run_dir = mode_run_dir(args.outputs_root, args.layout, args.seed, mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    actor_path = run_dir / "actor.pt"
    critic_path = run_dir / "critic.pt"
    training_data_path = run_dir / "training_data.pt"
    source_plot_path = run_dir / "trajectory_source.png"
    downstream_plot_path = run_dir / "trajectory_downstream.png"
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    torch.save(training_data, training_data_path)

    artifacts = {
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
        "training_data_path": str(training_data_path),
        "trajectory_source_plot_path": str(source_plot_path),
        "trajectory_downstream_plot_path": str(downstream_plot_path),
        "source_checkpoint_dir": str(source_dir),
    }
    if mode == "downstream_ewc":
        ewc_state_path = run_dir / "ewc_state.pt"
        torch.save(ewc_state, ewc_state_path)
        artifacts["ewc_state_path"] = str(ewc_state_path)
    if mode == "downstream_rashomon":
        assert rashomon_payload is not None
        assert bounded_model is not None
        assert actor_bounds_l is not None and actor_bounds_u is not None
        rashomon_dataset_path = run_dir / "rashomon_dataset.pt"
        bounded_model_path = run_dir / "rashomon_bounded_model.pt"
        bounds_path = run_dir / "rashomon_param_bounds.pt"
        torch.save(rashomon_payload, rashomon_dataset_path)
        torch.save(bounded_model, bounded_model_path)
        torch.save({"param_bounds_l": actor_bounds_l, "param_bounds_u": actor_bounds_u}, bounds_path)
        artifacts.update(
            {
                "rashomon_dataset_path": str(rashomon_dataset_path),
                "rashomon_bounded_model_path": str(bounded_model_path),
                "rashomon_param_bounds_path": str(bounds_path),
            },
        )

    _save_trajectory_plot(cfg=cfg, actor=actor, task="source", seed=args.seed, path=source_plot_path)
    _save_trajectory_plot(cfg=cfg, actor=actor, task="downstream", seed=args.seed, path=downstream_plot_path)

    run_results = _evaluate_both_tasks(cfg, actor=actor, device=args.device, seed=args.seed)
    run_settings = {
        "mode": mode,
        "layout": args.layout,
        "seed": args.seed,
        "activation": cfg.activation,
        "hidden": cfg.hidden,
        "reference_layout": cfg.reference_layout,
        "reference_settings_source": cfg.reference_settings_source,
        "reference_settings_files": cfg.reference_settings_files,
        "source_task_num": cfg.source_task_num,
        "downstream_task_num": cfg.downstream_task_num,
        "total_timesteps": int(total_timesteps),
        "warm_start_actor": True,
        "warm_start_critic": True,
        "train_shaped": True,
        "early_stop_eval_shaped": False,
        "task_feature_neutralization": False,
        "task_feature_index": 2,
        "ppo": _ppo_config_dict(ppo_cfg),
        "outputs_root": str(args.outputs_root),
        "source_checkpoint_dir": str(source_dir),
    }
    if mode == "downstream_ewc":
        run_settings.update(
            {
                "ewc_lambda": float(args.ewc_lambda or cfg.ewc_lambda),
                "ewc_apply_to_critic": bool(cfg.ewc_apply_to_critic),
                "fisher_sample_size": int(fisher_sample_size),
                "requested_fisher_sample_size": int(args.fisher_sample_size or cfg.fisher_sample_size),
            },
        )
    if mode == "downstream_rashomon":
        run_settings.update(
            {
                "rashomon_settings_source": cfg.rashomon_settings_source,
                "rashomon_n_iters": int(args.rashomon_n_iters or cfg.rashomon_n_iters),
                "inverse_temp_start": int(args.inverse_temp_start or cfg.inverse_temp_start),
                "inverse_temp_max": int(args.inverse_temp_max or cfg.inverse_temp_max),
                "rashomon_checkpoint": int(args.rashomon_checkpoint or cfg.rashomon_checkpoint),
                "surrogate_aggregation": cfg.rashomon_surrogate_aggregation,
                "rashomon_min_hard_spec": float(cfg.rashomon_min_hard_spec),
                "rashomon_ppo": _ppo_config_dict(ppo_cfg),
                "rashomon_dataset_size": int(rashomon_payload["state"].shape[0]),  # type: ignore[index]
                **rashomon_metadata,
            },
        )
    _write_summary(run_dir, run_settings=run_settings, run_results=run_results, artifacts=artifacts)
    print(f"Saved {mode} artifacts to {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FrozenLake safety pipeline.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["source", "downstream_unconstrained", "downstream_ewc", "downstream_rashomon"],
    )
    parser.add_argument("--pipeline", "--layout", type=str, dest="layout", default="diagonal_4x4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--outputs-root", type=Path, default=default_outputs_root())
    parser.add_argument("--source-run-dir", type=Path, default=None)
    parser.add_argument("--total-timesteps-override", type=int, default=None)
    parser.add_argument("--safety-finetune-lr", type=float, default=None)
    parser.add_argument("--safety-finetune-max-epochs", type=int, default=None)
    parser.add_argument("--ewc-lambda", type=float, default=None)
    parser.add_argument("--fisher-sample-size", type=int, default=None)
    parser.add_argument("--rashomon-n-iters", type=int, default=None)
    parser.add_argument("--inverse-temp-start", type=int, default=None)
    parser.add_argument("--inverse-temp-max", type=int, default=None)
    parser.add_argument("--rashomon-checkpoint", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        cfg = get_pipeline_config(args.layout)
        run_dir = mode_run_dir(args.outputs_root, args.layout, args.seed, args.mode)
        print(
            f"Dry run: mode={args.mode} layout={cfg.layout} seed={args.seed} "
            f"run_dir={run_dir}",
        )
        return 0
    if args.mode == "source":
        train_source(args)
    else:
        adapt_downstream(args, mode=args.mode)
    return 0
