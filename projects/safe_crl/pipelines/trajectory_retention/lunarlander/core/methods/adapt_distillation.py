"""Adapt a NoAdapt LunarLander policy with source-demo distillation."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

os.environ["SDL_AUDIODRIVER"] = "dummy"

import gymnasium as gym
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[7]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from projects.safe_crl.pipelines._shared.adaptation_utils import neutralize_task_feature
from projects.safe_crl.pipelines.trajectory_retention.lunarlander.core.env.env_factory import (
    _make_lunarlander_env,
)
from projects.safe_crl.pipelines.trajectory_retention.lunarlander.core.env.task_loading import (
    _load_task_settings,
    _resolve_lunarlander_dynamics,
)
from projects.safe_crl.pipelines.trajectory_retention.lunarlander.core.methods.adapt_rashomon import (
    create_source_rollout_rashomon_dataset,
)
from projects.safe_crl.pipelines.trajectory_retention.lunarlander.core.methods.adapt_unconstrained import (
    _load_source_hidden_size,
    _load_yaml,
)
from projects.safe_crl.pipelines.trajectory_retention.lunarlander.core.methods.source_train import (
    _plot_trajectory_grid,
    build_actor_critic,
)
from projects.safe_crl.pipelines.trajectory_retention.lunarlander.core.orchestration.run_paths import (
    default_adapt_distillation_settings_file,
    default_adapt_ppo_settings_file,
    default_outputs_root,
    default_task_settings_file,
    resolve_default_source_run_dir as _resolve_default_source_run_dir,
    seed_run_dir as _seed_run_dir,
)
from projects.safe_crl.utils.distillation_ppo import (
    DistillationPPOConfig,
    demonstration_metrics,
    distillation_ppo_train,
)
from projects.safe_crl.utils.ppo_utils import evaluate_with_success


def _resolve_distillation_cfg(settings: dict, task_setting: str, path: Path) -> dict:
    cfg = settings.get(task_setting, settings.get("default", {}))
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected dict distillation config for '{task_setting}' in {path}.")
    distill_cfg = cfg.get("distillation", cfg)
    if not isinstance(distill_cfg, dict):
        raise ValueError(f"Expected dict distillation section for '{task_setting}' in {path}.")
    return distill_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run downstream adaptation with source-demo distillation.")
    parser.add_argument("--pipeline", type=str, dest="task_setting", default="default")
    parser.add_argument("--task-setting", type=str, dest="task_setting", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--task-settings-file", type=Path, default=default_task_settings_file())
    parser.add_argument("--adapt-settings-file", type=Path, default=default_adapt_ppo_settings_file())
    parser.add_argument(
        "--distillation-settings-file",
        type=Path,
        default=default_adapt_distillation_settings_file(),
    )
    parser.add_argument("--outputs-root", type=Path, default=default_outputs_root())
    parser.add_argument("--source-run-dir", type=Path, default=None)
    parser.add_argument("--run-subdir", type=str, default="downstream_distillation")
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--disable-task-neutralization", action="store_true")
    parser.add_argument("--total-timesteps-override", type=int, default=None)
    parser.add_argument("--eval-episodes-during-training", type=int, default=None)
    parser.add_argument("--eval-episodes-post-training", type=int, default=None)
    parser.add_argument("--distill-lambda-override", type=float, default=None)
    parser.add_argument("--distill-batch-size", type=int, default=None)
    parser.add_argument("--demo-rollouts", type=int, default=None)
    parser.add_argument("--env-id", type=str, default=None)
    parser.add_argument("--source-gravity", type=float, default=None)
    parser.add_argument("--downstream-gravity", type=float, default=None)
    parser.add_argument("--append-task-id", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--trajectory-episodes", type=int, default=5)
    parser.add_argument("--trajectory-max-frames-per-episode", type=int, default=5)
    args = parser.parse_args()

    adapt_settings = _load_yaml(args.adapt_settings_file)
    distillation_settings = _load_yaml(args.distillation_settings_file)
    source_task_cfg = _load_task_settings(args.task_settings_file, args.task_setting, "source")
    downstream_task_cfg = _load_task_settings(args.task_settings_file, args.task_setting, "downstream")
    if args.task_setting not in adapt_settings:
        raise ValueError(f"Setting '{args.task_setting}' not found in {args.adapt_settings_file}")

    source_cfg = source_task_cfg
    downstream_cfg = downstream_task_cfg
    adapt_cfg = adapt_settings[args.task_setting]
    if not isinstance(source_cfg, dict) or not isinstance(downstream_cfg, dict):
        raise ValueError(f"Expected dict task configs for '{args.task_setting}'.")
    if not isinstance(adapt_cfg, dict) or not isinstance(adapt_cfg.get("ppo"), dict):
        raise ValueError(f"Expected ppo section for '{args.task_setting}' in {args.adapt_settings_file}.")
    adapt_ppo_cfg = adapt_cfg["ppo"]
    distill_cfg = _resolve_distillation_cfg(distillation_settings, args.task_setting, args.distillation_settings_file)
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
    if eval_episodes_during_training <= 0 or eval_episodes_post_training <= 0:
        raise ValueError("Evaluation episode counts must be > 0.")

    env_id = str(args.env_id or source_cfg.get("env_id") or downstream_cfg.get("env_id") or "LunarLander-v3")
    source_gravity_raw = args.source_gravity if args.source_gravity is not None else source_cfg.get("gravity")
    downstream_gravity_raw = (
        args.downstream_gravity if args.downstream_gravity is not None else downstream_cfg.get("gravity")
    )
    source_gravity = None if source_gravity_raw is None else float(source_gravity_raw)
    downstream_gravity = None if downstream_gravity_raw is None else float(downstream_gravity_raw)
    source_task_id = float(source_cfg.get("task_id", 0.0))
    downstream_task_id = float(downstream_cfg.get("task_id", 1.0))
    append_task_id = (
        bool(args.append_task_id)
        if args.append_task_id is not None
        else bool(source_cfg.get("append_task_id", True))
    )
    task_pipelines_file = str(source_cfg.get("_task_pipelines_file") or args.task_settings_file)
    task_definitions_file_raw = source_cfg.get("_task_definitions_file")
    task_definitions_file = None if task_definitions_file_raw is None else str(task_definitions_file_raw)
    resolved_pipeline_name = source_cfg.get("_resolved_pipeline_name")
    resolved_source_definition_name = source_cfg.get("_resolved_definition_name")
    resolved_downstream_definition_name = downstream_cfg.get("_resolved_definition_name")
    source_dynamics = _resolve_lunarlander_dynamics(source_cfg, cfg_name=f"{args.task_setting}:source")
    downstream_dynamics = _resolve_lunarlander_dynamics(downstream_cfg, cfg_name=f"{args.task_setting}:downstream")
    if bool(source_cfg.get("continuous", False) or downstream_cfg.get("continuous", False)):
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

    warm_actor = bool(adapt_cfg.get("warm_start", {}).get("actor", True))
    warm_critic = bool(adapt_cfg.get("warm_start", {}).get("critic", True))
    if not warm_actor:
        raise ValueError("This script expects actor warm-start (warm_start.actor=true).")

    source_run_dir = (
        args.source_run_dir
        if args.source_run_dir is not None
        else _resolve_default_source_run_dir(args.outputs_root, args.task_setting, args.seed)
    )
    actor_ckpt = source_run_dir / "actor.pt"
    critic_ckpt = source_run_dir / "critic.pt"
    if not actor_ckpt.exists():
        raise FileNotFoundError(f"NoAdapt actor checkpoint not found: {actor_ckpt}")
    if warm_critic and not critic_ckpt.exists():
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
    if warm_critic:
        source_critic.load_state_dict(torch.load(critic_ckpt, map_location="cpu"))

    demo_rollouts = int(
        args.demo_rollouts
        if args.demo_rollouts is not None
        else distill_cfg.get("demo_rollouts", distill_cfg.get("rashomon_rollouts", 1)),
    )
    source_rollout_env = _make_lunarlander_env(env_id, render_mode=None, **source_env_kwargs)
    try:
        source_demo_dataset, rollout_lengths = create_source_rollout_rashomon_dataset(
            actor=copy.deepcopy(source_actor),
            env=source_rollout_env,
            seed=args.seed,
            n_actions=n_actions,
            rashomon_rollouts=demo_rollouts,
        )
    finally:
        source_rollout_env.close()

    task_transform_cfg = adapt_cfg.get("pre_adaptation_transform", {})
    do_task_neutralization = (
        bool(task_transform_cfg.get("task_feature_neutralization", False))
        and append_task_id
        and not args.disable_task_neutralization
    )
    task_feature_index = int(task_transform_cfg.get("task_feature_index", obs_dim - 1))
    if do_task_neutralization:
        neutralize_task_feature(source_actor, task_feature_index, downstream_task_id)
        if warm_critic:
            neutralize_task_feature(source_critic, task_feature_index, downstream_task_id)

    total_timesteps = (
        int(args.total_timesteps_override)
        if args.total_timesteps_override is not None
        else int(adapt_ppo_cfg["total_timesteps"])
    )
    distill_lambda = (
        float(args.distill_lambda_override)
        if args.distill_lambda_override is not None
        else float(distill_cfg.get("distill_lambda", 1.0))
    )
    distill_batch_size = (
        int(args.distill_batch_size)
        if args.distill_batch_size is not None
        else (None if distill_cfg.get("distill_batch_size") is None else int(distill_cfg["distill_batch_size"]))
    )
    early_stop_reward_threshold_cfg = adapt_ppo_cfg.get("early_stop_reward_threshold", None)
    early_stop_reward_threshold = (
        float(early_stop_reward_threshold_cfg)
        if early_stop_reward_threshold_cfg is not None
        else None
    )
    ppo_cfg = DistillationPPOConfig(
        seed=int(adapt_ppo_cfg.get("seed", args.seed)),
        total_timesteps=total_timesteps,
        eval_episodes=eval_episodes_during_training,
        rollout_steps=int(adapt_ppo_cfg["rollout_steps"]),
        update_epochs=int(adapt_ppo_cfg["update_epochs"]),
        minibatch_size=int(adapt_ppo_cfg["minibatch_size"]),
        gamma=float(adapt_ppo_cfg["gamma"]),
        gae_lambda=float(adapt_ppo_cfg["gae_lambda"]),
        clip_coef=float(adapt_ppo_cfg["clip_coef"]),
        ent_coef=float(adapt_ppo_cfg["ent_coef"]),
        vf_coef=float(adapt_ppo_cfg["vf_coef"]),
        lr=float(adapt_ppo_cfg["lr"]),
        max_grad_norm=float(adapt_ppo_cfg["max_grad_norm"]),
        device=args.device,
        early_stop_min_steps=int(adapt_ppo_cfg.get("early_stop_min_steps", 0)),
        early_stop_reward_threshold=early_stop_reward_threshold,
        early_stop_failure_rate_threshold=adapt_ppo_cfg.get("early_stop_failure_rate_threshold", None),
        early_stop_success_rate_threshold=adapt_ppo_cfg.get("early_stop_success_rate_threshold", None),
        distill_lambda=distill_lambda,
        distill_batch_size=distill_batch_size,
    )

    print(
        f"Adapting LunarLander (distillation) | setting={args.task_setting} | "
        f"demo_samples={len(source_demo_dataset)} | distill_lambda={distill_lambda}",
    )
    train_env = _make_lunarlander_env(env_id, render_mode=None, **downstream_env_kwargs)
    early_stop_eval_env = _make_lunarlander_env(env_id, render_mode=None, **downstream_env_kwargs)
    actor, critic, training_data = distillation_ppo_train(  # type: ignore[assignment]
        train_env,
        ppo_cfg,
        source_demo_dataset=source_demo_dataset,
        actor_warm_start=source_actor,
        critic_warm_start=(source_critic if warm_critic else None),
        early_stop_eval_env=early_stop_eval_env,
        return_training_data=True,
    )

    source_eval_env = _make_lunarlander_env(env_id, render_mode=None, **source_env_kwargs)
    source_mean_reward, source_std_reward, source_failure_rate, source_success_rate = evaluate_with_success(
        source_eval_env,
        actor,
        episodes=eval_episodes_post_training,
        deterministic=True,
        device=args.device,
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
        episodes=eval_episodes_post_training,
        deterministic=True,
        device=args.device,
    )
    downstream_eval_env.close()
    demo_metrics = demonstration_metrics(actor, source_demo_dataset, device=args.device)

    downstream_run_dir = _seed_run_dir(args.outputs_root, args.task_setting, args.seed) / args.run_subdir
    downstream_run_dir.mkdir(parents=True, exist_ok=True)
    actor_path = downstream_run_dir / "actor.pt"
    critic_path = downstream_run_dir / "critic.pt"
    training_data_path = downstream_run_dir / "training_data.pt"
    source_demo_dataset_path = downstream_run_dir / "source_demo_dataset.pt"
    rollout_stats_path = downstream_run_dir / "source_demo_rollout_stats.yaml"
    source_plot_path = downstream_run_dir / "trajectory_source.png"
    downstream_plot_path = downstream_run_dir / "trajectory_downstream.png"
    summary_path = downstream_run_dir / "run_summary.yaml"

    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    torch.save(training_data, training_data_path)
    torch.save(source_demo_dataset, source_demo_dataset_path)
    rollout_stats_path.write_text(
        yaml.safe_dump(
            {
                "demo_rollouts": int(demo_rollouts),
                "rollout_lengths": [int(v) for v in rollout_lengths],
                "total_state_action_pairs": int(len(source_demo_dataset)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    actor_for_plot = copy.deepcopy(actor).to("cpu")
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

    run_settings = {
        "task_setting": args.task_setting,
        "seed": args.seed,
        "env_id": env_id,
        "source_task_id": float(source_task_id),
        "downstream_task_id": float(downstream_task_id),
        "source_gravity": source_gravity,
        "downstream_gravity": downstream_gravity,
        "source_dynamics": source_dynamics,
        "downstream_dynamics": downstream_dynamics,
        "append_task_id": bool(append_task_id),
        "warm_start_actor": warm_actor,
        "warm_start_critic": warm_critic,
        "task_feature_neutralization": do_task_neutralization,
        "task_feature_index": int(task_feature_index) if do_task_neutralization else None,
        "eval_episodes_during_training": int(eval_episodes_during_training),
        "eval_episodes_post_training": int(eval_episodes_post_training),
        "trajectory_episodes": int(args.trajectory_episodes),
        "trajectory_max_frames_per_episode": int(args.trajectory_max_frames_per_episode),
        "distill_lambda": float(distill_lambda),
        "distill_batch_size": int(distill_batch_size or ppo_cfg.minibatch_size),
        "demo_rollouts": int(demo_rollouts),
        "source_demo_dataset_size": int(len(source_demo_dataset)),
        "noadapt_checkpoint_dir": str(source_run_dir),
        "source_checkpoint_dir": str(source_run_dir),
        "task_settings_file": str(args.task_settings_file),
        "task_pipelines_file": task_pipelines_file,
        "task_definitions_file": task_definitions_file,
        "resolved_pipeline_name": resolved_pipeline_name,
        "resolved_source_definition_name": resolved_source_definition_name,
        "resolved_downstream_definition_name": resolved_downstream_definition_name,
        "adapt_settings_file": str(args.adapt_settings_file),
        "distillation_settings_file": str(args.distillation_settings_file),
    }
    run_results = {
        "source_mean_reward": float(source_mean_reward),
        "source_std_reward": float(source_std_reward),
        "source_failure_rate": float(source_failure_rate),
        "source_success_rate": float(source_success_rate),
        "downstream_mean_reward": float(downstream_mean_reward),
        "downstream_std_reward": float(downstream_std_reward),
        "downstream_failure_rate": float(downstream_failure_rate),
        "downstream_success_rate": float(downstream_success_rate),
        **demo_metrics,
    }
    artifacts = {
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
        "training_data_path": str(training_data_path),
        "source_demo_dataset_path": str(source_demo_dataset_path),
        "source_demo_rollout_stats_path": str(rollout_stats_path),
        "trajectory_source_plot_path": str(source_plot_path),
        "trajectory_downstream_plot_path": str(downstream_plot_path),
    }
    summary = {
        "run_settings": run_settings,
        "run_results": run_results,
        "artifacts": artifacts,
    }
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(
        f"Source eval ({eval_episodes_post_training} ep): mean_reward={source_mean_reward:.3f}, "
        f"success_rate={source_success_rate:.3f}, demo_acc={demo_metrics['source_demo_accuracy']:.3f}",
    )
    print(
        f"Downstream eval ({eval_episodes_post_training} ep): mean_reward={downstream_mean_reward:.3f}, "
        f"success_rate={downstream_success_rate:.3f}",
    )
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
