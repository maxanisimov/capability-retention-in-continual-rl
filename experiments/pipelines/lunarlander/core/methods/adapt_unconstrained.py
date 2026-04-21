"""Adapt source LunarLander policy to downstream task via unconstrained PPO."""

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

# Allow running this file directly from experiments/pipelines/lunarlander/core/methods.
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
    default_outputs_root,
    default_task_settings_file,
    resolve_default_source_run_dir as _resolve_default_source_run_dir,
    seed_run_dir as _seed_run_dir,
)
from experiments.utils.ppo_utils import PPOConfig, evaluate, ppo_train


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML dict in {path}, got {type(data)}.")
    return data


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unconstrained downstream adaptation for LunarLander.")
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
        "--outputs-root",
        type=Path,
        default=default_outputs_root(),
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help="Optional explicit source checkpoint directory. Defaults to outputs/<task_setting>/seed_<seed>/source",
    )
    parser.add_argument(
        "--run-subdir",
        type=str,
        default="downstream_unconstrained",
        help="Subdirectory under outputs/<task_setting>/seed_<seed>/ where outputs are saved.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Actor/critic hidden size. Defaults to source run summary hidden_size if available.",
    )
    parser.add_argument(
        "--disable-task-neutralization",
        action="store_true",
        help="Disable first-layer task-feature neutralization before adaptation.",
    )
    parser.add_argument(
        "--total-timesteps-override",
        type=int,
        default=None,
        help="Optional override for PPO total timesteps.",
    )
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
        help="Number of episodes for final post-training evaluation.",
    )
    parser.add_argument("--env-id", type=str, default=None, help="Optional env-id override.")
    parser.add_argument("--source-gravity", type=float, default=None, help="Optional source gravity override.")
    parser.add_argument("--downstream-gravity", type=float, default=None, help="Optional downstream gravity override.")
    parser.add_argument("--source-task-id", type=float, default=None, help="Optional source task-id override.")
    parser.add_argument("--downstream-task-id", type=float, default=None, help="Optional downstream task-id override.")
    parser.add_argument(
        "--append-task-id",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append task-id feature in observations (default inherited from env settings).",
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
    args = parser.parse_args()
    if args.eval_episodes_during_training < 2:
        raise ValueError("For LunarLander, --eval-episodes-during-training must be >= 2.")
    if args.eval_episodes_post_training < 2:
        raise ValueError("For LunarLander, --eval-episodes-post-training must be >= 2.")

    adapt_settings = _load_yaml(args.adapt_settings_file)
    source_task_cfg = _load_task_settings(args.task_settings_file, args.task_setting, "source")
    downstream_task_cfg = _load_task_settings(args.task_settings_file, args.task_setting, "downstream")

    if args.task_setting not in adapt_settings:
        raise ValueError(f"Setting '{args.task_setting}' not found in {args.adapt_settings_file}")

    source_cfg = source_task_cfg
    downstream_cfg = downstream_task_cfg
    adapt_cfg = adapt_settings[args.task_setting]

    if not isinstance(source_cfg, dict):
        raise ValueError(f"Expected dict source config in task settings for '{args.task_setting}'.")
    if not isinstance(downstream_cfg, dict):
        raise ValueError(f"Expected dict downstream config in task settings for '{args.task_setting}'.")
    if not isinstance(adapt_cfg, dict):
        raise ValueError(f"Expected dict adapt settings config for setting '{args.task_setting}'.")
    if "ppo" not in adapt_cfg or not isinstance(adapt_cfg["ppo"], dict):
        raise ValueError(f"Expected 'ppo' section for setting '{args.task_setting}' in {args.adapt_settings_file}.")

    adapt_ppo_cfg = adapt_cfg["ppo"]
    env_id = str(
        args.env_id
        or source_cfg.get("env_id")
        or downstream_cfg.get("env_id")
        or "LunarLander-v3"
    )
    source_gravity_raw = args.source_gravity if args.source_gravity is not None else source_cfg.get("gravity")
    downstream_gravity_raw = (
        args.downstream_gravity if args.downstream_gravity is not None else downstream_cfg.get("gravity")
    )
    source_gravity = None if source_gravity_raw is None else float(source_gravity_raw)
    downstream_gravity = None if downstream_gravity_raw is None else float(downstream_gravity_raw)

    source_task_id = float(args.source_task_id) if args.source_task_id is not None else float(
        adapt_cfg.get("source_task_id", source_cfg.get("task_id", 0.0)),
    )
    downstream_task_id = float(args.downstream_task_id) if args.downstream_task_id is not None else float(
        adapt_cfg.get("downstream_task_id", downstream_cfg.get("task_id", 1.0)),
    )
    append_task_id = (
        bool(args.append_task_id)
        if args.append_task_id is not None
        else bool(source_cfg.get("append_task_id", True))
    )
    source_dynamics = _resolve_lunarlander_dynamics(
        source_cfg,
        cfg_name=f"task_settings[{args.task_setting}:source]",
    )
    downstream_dynamics = _resolve_lunarlander_dynamics(
        downstream_cfg,
        cfg_name=f"task_settings[{args.task_setting}:downstream]",
    )
    continuous = bool(source_cfg.get("continuous", False) or downstream_cfg.get("continuous", False))
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
        raise FileNotFoundError(f"Source actor checkpoint not found: {actor_ckpt}")
    if warm_critic and not critic_ckpt.exists():
        raise FileNotFoundError(f"Source critic checkpoint not found: {critic_ckpt}")

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
    if warm_critic:
        source_critic.load_state_dict(torch.load(critic_ckpt, map_location="cpu"))

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
    early_stop_cfg_value = adapt_ppo_cfg.get("early_stop", True)
    early_stop_enabled = True if early_stop_cfg_value is None else bool(early_stop_cfg_value)
    early_stop_reward_threshold_cfg = adapt_ppo_cfg.get("early_stop_reward_threshold", 200.0)
    early_stop_reward_threshold = (
        float(early_stop_reward_threshold_cfg)
        if early_stop_reward_threshold_cfg is not None
        else 200.0
    )

    ppo_cfg = PPOConfig(
        seed=int(adapt_ppo_cfg.get("seed", args.seed)),
        total_timesteps=total_timesteps,
        eval_episodes=int(args.eval_episodes_during_training),
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
        early_stop=early_stop_enabled,
        early_stop_min_steps=int(adapt_ppo_cfg.get("early_stop_min_steps", 0)),
        early_stop_reward_threshold=early_stop_reward_threshold,
        early_stop_failure_rate_threshold=adapt_ppo_cfg.get("early_stop_failure_rate_threshold", None),
        early_stop_deterministic_total_reward_threshold=adapt_ppo_cfg.get(
            "early_stop_deterministic_total_reward_threshold",
            None,
        ),
        early_stop_deterministic_eval_episodes=int(
            adapt_ppo_cfg.get("early_stop_deterministic_eval_episodes", 20),
        ),
    )

    print(
        f"Adapting LunarLander (unconstrained) | setting={args.task_setting} | "
        f"source_task={source_task_id} -> downstream_task={downstream_task_id} | "
        f"warm_critic={warm_critic} | task_neutralization={do_task_neutralization}",
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
        critic_warm_start=(source_critic if warm_critic else None),
        early_stop_eval_env=early_stop_eval_env,
        return_training_data=True,
    )

    eval_episodes_post_training = int(args.eval_episodes_post_training)

    source_eval_env = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **source_env_kwargs,
    )
    source_mean_reward, source_std_reward, source_failure_rate = evaluate(
        source_eval_env,
        actor,
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
    downstream_mean_reward, downstream_std_reward, downstream_failure_rate = evaluate(
        downstream_eval_env,
        actor,
        episodes=eval_episodes_post_training,
        deterministic=True,
        device=args.device,
    )
    downstream_eval_env.close()

    downstream_run_dir = _seed_run_dir(args.outputs_root, args.task_setting, args.seed) / args.run_subdir
    downstream_run_dir.mkdir(parents=True, exist_ok=True)

    actor_path = downstream_run_dir / "actor.pt"
    critic_path = downstream_run_dir / "critic.pt"
    training_data_path = downstream_run_dir / "training_data.pt"
    source_plot_path = downstream_run_dir / "trajectory_source.png"
    downstream_plot_path = downstream_run_dir / "trajectory_downstream.png"
    summary_path = downstream_run_dir / "run_summary.yaml"

    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    torch.save(training_data, training_data_path)

    # Plot with a CPU actor copy to avoid device-mismatch issues in rendering helpers.
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
        "eval_episodes_during_training": int(args.eval_episodes_during_training),
        "eval_episodes_post_training": int(eval_episodes_post_training),
        "trajectory_episodes": int(args.trajectory_episodes),
        "trajectory_max_frames_per_episode": int(args.trajectory_max_frames_per_episode),
        "source_checkpoint_dir": str(source_run_dir),
        "task_settings_file": str(args.task_settings_file),
        "adapt_settings_file": str(args.adapt_settings_file),
    }
    run_results = {
        "source_mean_reward": float(source_mean_reward),
        "source_std_reward": float(source_std_reward),
        "source_failure_rate": float(source_failure_rate),
        "downstream_mean_reward": float(downstream_mean_reward),
        "downstream_std_reward": float(downstream_std_reward),
        "downstream_failure_rate": float(downstream_failure_rate),
    }
    artifacts = {
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
        "training_data_path": str(training_data_path),
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
        f"std={source_std_reward:.3f}, failure_rate={source_failure_rate:.3f}",
    )
    print(
        f"Downstream eval ({eval_episodes_post_training} ep): mean_reward={downstream_mean_reward:.3f}, "
        f"std={downstream_std_reward:.3f}, failure_rate={downstream_failure_rate:.3f}",
    )
    print(f"Saved actor: {actor_path}")
    print(f"Saved critic: {critic_path}")
    print(f"Saved source trajectory grid: {source_plot_path}")
    print(f"Saved downstream trajectory grid: {downstream_plot_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
