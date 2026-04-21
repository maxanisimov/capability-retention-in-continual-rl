"""Train a source-only PPO policy on LunarLander with discrete actions.

This script mirrors the FrozenLake source-training flow while targeting
Gymnasium LunarLander in its discrete-action setting (`continuous=False`).
Actor and critic hidden layers use ReLU activations.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Allow running this file directly from experiments/pipelines/lunarlander/core/methods.
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.lunarlander.core.env.env_factory import (
    make_lunarlander_env as _make_lunarlander_env,
)
from experiments.pipelines.lunarlander.core.env.task_loading import (
    load_task_settings as _load_task_settings,
)
from experiments.pipelines.lunarlander.core.env.task_loading import (
    resolve_lunarlander_dynamics as _resolve_lunarlander_dynamics,
)
from experiments.pipelines.lunarlander.core.orchestration.run_paths import (
    default_outputs_root,
    default_task_settings_file,
    seed_run_dir,
)
from experiments.utils.gymnasium_utils import plot_multi_episode_frames
from experiments.utils.ppo_utils import PPOConfig, evaluate, ppo_train

def build_actor_critic(obs_dim: int, n_actions: int, hidden_size: int) -> tuple[torch.nn.Sequential, torch.nn.Sequential]:
    """Build MLP actor/critic with ReLU hidden activations."""
    actor = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, n_actions),
    )
    critic = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, 1),
    )
    return actor, critic


def _collect_episode_frames(
    env: gym.Env,
    actor: torch.nn.Module,
    *,
    seed: int,
    device: str,
) -> list[np.ndarray]:
    """Run one deterministic episode and collect rendered RGB frames."""
    frames: list[np.ndarray] = []
    obs, _ = env.reset(seed=seed)

    frame = env.render()
    if frame is None:
        raise RuntimeError("env.render() returned None; ensure render_mode='rgb_array'.")
    frames.append(np.asarray(frame).copy())

    done = False
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = actor(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frame = env.render()
        if frame is None:
            raise RuntimeError("env.render() returned None; ensure render_mode='rgb_array'.")
        frames.append(np.asarray(frame).copy())
    return frames


def _plot_trajectory_grid(
    *,
    env_id: str,
    gravity: float | None,
    task_id: float,
    append_task_id: bool,
    dynamics_cfg: dict[str, Any],
    actor: torch.nn.Module,
    seed: int,
    device: str,
    output_path: Path,
    episodes: int = 10,
    max_frames_per_episode: int = 5,
) -> None:
    if episodes <= 0:
        raise ValueError(f"episodes must be > 0, got {episodes}.")
    if max_frames_per_episode < 2:
        raise ValueError(
            "max_frames_per_episode must be >= 2 so initial and final frames can both be shown.",
        )

    render_env = _make_lunarlander_env(
        env_id,
        gravity=gravity,
        task_id=task_id,
        append_task_id=append_task_id,
        enable_wind=bool(dynamics_cfg["enable_wind"]),
        wind_power=dynamics_cfg["wind_power"],
        turbulence_power=dynamics_cfg["turbulence_power"],
        initial_random_strength=dynamics_cfg["initial_random_strength"],
        dispersion_strength=dynamics_cfg["dispersion_strength"],
        main_engine_power=dynamics_cfg["main_engine_power"],
        side_engine_power=dynamics_cfg["side_engine_power"],
        leg_spring_torque=dynamics_cfg["leg_spring_torque"],
        lander_mass_scale=dynamics_cfg["lander_mass_scale"],
        leg_mass_scale=dynamics_cfg["leg_mass_scale"],
        linear_damping=dynamics_cfg["linear_damping"],
        angular_damping=dynamics_cfg["angular_damping"],
        terrain_heights=dynamics_cfg["terrain_heights"],
        action_repeat=int(dynamics_cfg["action_repeat"]),
        action_delay=int(dynamics_cfg["action_delay"]),
        action_noise_prob=float(dynamics_cfg["action_noise_prob"]),
        action_noise_mode=str(dynamics_cfg["action_noise_mode"]),
        mark_out_of_viewport_as_unsafe=bool(dynamics_cfg["mark_out_of_viewport_as_unsafe"]),
        render_mode="rgb_array",
    )
    actor_was_training = actor.training
    actor.eval()
    try:
        all_episode_frames: list[list[np.ndarray]] = []
        for ep_idx in range(episodes):
            ep_seed = seed + ep_idx
            ep_frames = _collect_episode_frames(
                render_env,
                actor,
                seed=ep_seed,
                device=device,
            )
            all_episode_frames.append(ep_frames)

        plot_multi_episode_frames(
            episodes=all_episode_frames,
            n_cols=max_frames_per_episode,
            episode_labels=[f"Ep {i + 1}" for i in range(episodes)],
            title=(
                f"LunarLander trajectories ({episodes} episodes, up to {max_frames_per_episode} frames each)"
            ),
            save_path=str(output_path),
        )
    finally:
        render_env.close()
        if actor_was_training:
            actor.train()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a source-only LunarLander policy with PPO (discrete actions, ReLU MLP).",
    )
    parser.add_argument(
        "--task-settings-file",
        type=Path,
        default=default_task_settings_file(),
        help="Task settings YAML defining source/downstream env variants.",
    )
    parser.add_argument(
        "--task-setting",
        type=str,
        default="default",
        help="Task-setting key to read from --task-settings-file.",
    )
    parser.add_argument(
        "--task-role",
        type=str,
        choices=["source", "downstream"],
        default="source",
        help="Which task variant to train/evaluate on.",
    )
    parser.add_argument("--env-id", type=str, default=None, help="Override environment id (e.g., LunarLander-v3).")
    parser.add_argument(
        "--gravity",
        type=float,
        default=None,
        help="Override gravity. If omitted, uses task setting gravity; if that is null, uses Gym default.",
    )
    parser.add_argument(
        "--task-id",
        type=float,
        default=None,
        help="Task id appended to observations when --append-task-id is enabled.",
    )
    parser.add_argument(
        "--append-task-id",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append task id to observation vector (enabled by default in env creation).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
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
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--early-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable early stopping on periodic evaluation (default: enabled).",
    )
    parser.add_argument("--early-stop-min-steps", type=int, default=0)
    parser.add_argument("--early-stop-reward-threshold", type=float, default=200.0)
    parser.add_argument("--early-stop-failure-rate-threshold", type=float, default=None)
    parser.add_argument("--early-stop-deterministic-total-reward-threshold", type=float, default=None)
    parser.add_argument("--early-stop-deterministic-eval-episodes", type=int, default=10)
    parser.add_argument(
        "--trajectory-episodes",
        type=int,
        default=10,
        help="Number of deterministic episodes to visualize after training.",
    )
    parser.add_argument(
        "--trajectory-max-frames-per-episode",
        type=int,
        default=5,
        help="Maximum frames shown per episode row in the trajectory figure (includes first and last frames).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_outputs_root(),
    )
    args = parser.parse_args()
    if args.eval_episodes_during_training < 2:
        raise ValueError("For LunarLander, --eval-episodes-during-training must be >= 2.")
    if args.eval_episodes_post_training < 2:
        raise ValueError("For LunarLander, --eval-episodes-post-training must be >= 2.")

    task_settings = _load_task_settings(args.task_settings_file, args.task_setting, args.task_role)
    env_id = str(task_settings.get("env_id") or args.env_id or "LunarLander-v3")
    gravity = args.gravity if args.gravity is not None else task_settings.get("gravity")
    gravity_value = None if gravity is None else float(gravity)
    continuous = bool(task_settings.get("continuous", False))
    if continuous:
        raise ValueError("This script only supports discrete actions (`continuous=False`).")
    default_task_id = 0.0 if args.task_role == "source" else 1.0
    task_id = float(args.task_id) if args.task_id is not None else float(task_settings.get("task_id", default_task_id))
    append_task_id = (
        bool(args.append_task_id)
        if args.append_task_id is not None
        else bool(task_settings.get("append_task_id", True))
    )
    dynamics_cfg = _resolve_lunarlander_dynamics(
        task_settings,
        cfg_name=f"task_settings[{args.task_setting}:{args.task_role}]",
    )
    env_kwargs = {
        "gravity": gravity_value,
        "task_id": task_id,
        "append_task_id": append_task_id,
        "enable_wind": bool(dynamics_cfg["enable_wind"]),
        "wind_power": dynamics_cfg["wind_power"],
        "turbulence_power": dynamics_cfg["turbulence_power"],
        "initial_random_strength": dynamics_cfg["initial_random_strength"],
        "dispersion_strength": dynamics_cfg["dispersion_strength"],
        "main_engine_power": dynamics_cfg["main_engine_power"],
        "side_engine_power": dynamics_cfg["side_engine_power"],
        "leg_spring_torque": dynamics_cfg["leg_spring_torque"],
        "lander_mass_scale": dynamics_cfg["lander_mass_scale"],
        "leg_mass_scale": dynamics_cfg["leg_mass_scale"],
        "linear_damping": dynamics_cfg["linear_damping"],
        "angular_damping": dynamics_cfg["angular_damping"],
        "terrain_heights": dynamics_cfg["terrain_heights"],
        "action_repeat": int(dynamics_cfg["action_repeat"]),
        "action_delay": int(dynamics_cfg["action_delay"]),
        "action_noise_prob": float(dynamics_cfg["action_noise_prob"]),
        "action_noise_mode": str(dynamics_cfg["action_noise_mode"]),
        "mark_out_of_viewport_as_unsafe": bool(dynamics_cfg["mark_out_of_viewport_as_unsafe"]),
    }
    train_env = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **env_kwargs,
    )
    if not isinstance(train_env.action_space, gym.spaces.Discrete):
        raise ValueError(
            "This script expects a discrete-action environment, but got non-discrete action space.",
        )
    if not isinstance(train_env.observation_space, gym.spaces.Box):
        raise ValueError("This script expects a Box observation space.")

    obs_dim = int(train_env.observation_space.shape[0])
    n_actions = int(train_env.action_space.n)
    actor, critic = build_actor_critic(obs_dim=obs_dim, n_actions=n_actions, hidden_size=args.hidden_size)

    ppo_cfg = PPOConfig(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        eval_episodes=args.eval_episodes_during_training,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        early_stop=args.early_stop,
        early_stop_min_steps=args.early_stop_min_steps,
        early_stop_reward_threshold=args.early_stop_reward_threshold,
        early_stop_failure_rate_threshold=args.early_stop_failure_rate_threshold,
        early_stop_deterministic_total_reward_threshold=args.early_stop_deterministic_total_reward_threshold,
        early_stop_deterministic_eval_episodes=args.early_stop_deterministic_eval_episodes,
    )

    print(f"Training {args.task_role} policy on {env_id} (discrete) | seed={args.seed} | device={args.device}")
    print(
        "  "
        f"gravity={gravity_value} | task_id={task_id} | append_task_id={append_task_id} | "
        f"enable_wind={dynamics_cfg['enable_wind']} | wind_power={dynamics_cfg['wind_power']} | "
        f"turbulence_power={dynamics_cfg['turbulence_power']} | "
        f"initial_random_strength={dynamics_cfg['initial_random_strength']} | "
        f"dispersion_strength={dynamics_cfg['dispersion_strength']} | "
        f"main_engine_power={dynamics_cfg['main_engine_power']} | "
        f"side_engine_power={dynamics_cfg['side_engine_power']} | "
        f"leg_spring_torque={dynamics_cfg['leg_spring_torque']} | "
        f"lander_mass_scale={dynamics_cfg['lander_mass_scale']} | "
        f"leg_mass_scale={dynamics_cfg['leg_mass_scale']} | "
        f"linear_damping={dynamics_cfg['linear_damping']} | "
        f"angular_damping={dynamics_cfg['angular_damping']} | "
        f"action_repeat={dynamics_cfg['action_repeat']} | "
        f"action_delay={dynamics_cfg['action_delay']} | action_noise_prob={dynamics_cfg['action_noise_prob']} | "
        f"action_noise_mode={dynamics_cfg['action_noise_mode']} | "
        f"mark_out_of_viewport_as_unsafe={dynamics_cfg['mark_out_of_viewport_as_unsafe']}"
    )
    actor, critic, training_data = ppo_train(  # type: ignore[assignment]
        env=train_env,
        cfg=ppo_cfg,
        actor_warm_start=actor,
        critic_warm_start=critic,
        return_training_data=True,
    )

    eval_env = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **env_kwargs,
    )
    mean_reward, std_reward, failure_rate = evaluate(
        env=eval_env,
        actor=actor,
        episodes=args.eval_episodes_post_training,
        deterministic=True,
        device=args.device,
    )
    eval_env.close()

    downstream_eval_performed = False
    downstream_eval_env_id: str | None = None
    downstream_eval_task_id: float | None = None
    downstream_eval_gravity: float | None = None
    downstream_eval_append_task_id: bool | None = None
    downstream_eval_dynamics: dict[str, Any] | None = None
    downstream_mean_reward: float | None = None
    downstream_std_reward: float | None = None
    downstream_failure_rate: float | None = None

    run_dir = seed_run_dir(args.output_dir, args.task_setting, args.seed)
    task_dir = run_dir / args.task_role
    task_dir.mkdir(parents=True, exist_ok=True)

    actor_path = task_dir / "actor.pt"
    critic_path = task_dir / "critic.pt"
    training_data_path = task_dir / "training_data.pt"
    summary_path = task_dir / "run_summary.yaml"

    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    torch.save(training_data, training_data_path)

    trajectory_source_plot_path = task_dir / "trajectory_source.png"
    _plot_trajectory_grid(
        env_id=env_id,
        gravity=gravity_value,
        task_id=task_id,
        append_task_id=append_task_id,
        dynamics_cfg=dynamics_cfg,
        actor=actor,
        seed=args.seed,
        device=args.device,
        output_path=trajectory_source_plot_path,
        episodes=int(args.trajectory_episodes),
        max_frames_per_episode=int(args.trajectory_max_frames_per_episode),
    )

    trajectory_downstream_plot_path: Path | None = None
    if args.task_role == "source":
        downstream_task_settings = _load_task_settings(args.task_settings_file, args.task_setting, "downstream")
        downstream_eval_env_id = str(args.env_id or downstream_task_settings.get("env_id") or env_id)
        downstream_gravity_raw = downstream_task_settings.get("gravity")
        downstream_eval_gravity = None if downstream_gravity_raw is None else float(downstream_gravity_raw)
        downstream_eval_task_id = float(downstream_task_settings.get("task_id", 1.0))
        downstream_eval_append_task_id = (
            bool(args.append_task_id)
            if args.append_task_id is not None
            else bool(downstream_task_settings.get("append_task_id", append_task_id))
        )
        downstream_eval_dynamics = _resolve_lunarlander_dynamics(
            downstream_task_settings,
            cfg_name=f"task_settings[{args.task_setting}:downstream]",
        )
        downstream_eval_kwargs = {
            "gravity": downstream_eval_gravity,
            "task_id": downstream_eval_task_id,
            "append_task_id": downstream_eval_append_task_id,
            "enable_wind": bool(downstream_eval_dynamics["enable_wind"]),
            "wind_power": downstream_eval_dynamics["wind_power"],
            "turbulence_power": downstream_eval_dynamics["turbulence_power"],
            "initial_random_strength": downstream_eval_dynamics["initial_random_strength"],
            "dispersion_strength": downstream_eval_dynamics["dispersion_strength"],
            "main_engine_power": downstream_eval_dynamics["main_engine_power"],
            "side_engine_power": downstream_eval_dynamics["side_engine_power"],
            "leg_spring_torque": downstream_eval_dynamics["leg_spring_torque"],
            "lander_mass_scale": downstream_eval_dynamics["lander_mass_scale"],
            "leg_mass_scale": downstream_eval_dynamics["leg_mass_scale"],
            "linear_damping": downstream_eval_dynamics["linear_damping"],
            "angular_damping": downstream_eval_dynamics["angular_damping"],
            "terrain_heights": downstream_eval_dynamics["terrain_heights"],
            "action_repeat": int(downstream_eval_dynamics["action_repeat"]),
            "action_delay": int(downstream_eval_dynamics["action_delay"]),
            "action_noise_prob": float(downstream_eval_dynamics["action_noise_prob"]),
            "action_noise_mode": str(downstream_eval_dynamics["action_noise_mode"]),
            "mark_out_of_viewport_as_unsafe": bool(downstream_eval_dynamics["mark_out_of_viewport_as_unsafe"]),
        }
        downstream_eval_env = _make_lunarlander_env(
            downstream_eval_env_id,
            render_mode=None,
            **downstream_eval_kwargs,
        )
        downstream_mean_reward, downstream_std_reward, downstream_failure_rate = evaluate(
            env=downstream_eval_env,
            actor=actor,
            episodes=args.eval_episodes_post_training,
            deterministic=True,
            device=args.device,
        )
        downstream_eval_env.close()

        trajectory_downstream_plot_path = task_dir / "trajectory_downstream.png"
        _plot_trajectory_grid(
            env_id=downstream_eval_env_id,
            gravity=downstream_eval_gravity,
            task_id=downstream_eval_task_id,
            append_task_id=downstream_eval_append_task_id,
            dynamics_cfg=downstream_eval_dynamics,
            actor=actor,
            seed=args.seed,
            device=args.device,
            output_path=trajectory_downstream_plot_path,
            episodes=int(args.trajectory_episodes),
            max_frames_per_episode=int(args.trajectory_max_frames_per_episode),
        )
        downstream_eval_performed = True

    run_settings = {
        "env_id": env_id,
        "continuous": bool(continuous),
        "task_role": args.task_role,
        "task_id": float(task_id),
        "gravity": gravity_value,
        "append_task_id": bool(append_task_id),
        "enable_wind": bool(dynamics_cfg["enable_wind"]),
        "wind_power": dynamics_cfg["wind_power"],
        "turbulence_power": dynamics_cfg["turbulence_power"],
        "initial_random_strength": dynamics_cfg["initial_random_strength"],
        "dispersion_strength": dynamics_cfg["dispersion_strength"],
        "main_engine_power": dynamics_cfg["main_engine_power"],
        "side_engine_power": dynamics_cfg["side_engine_power"],
        "leg_spring_torque": dynamics_cfg["leg_spring_torque"],
        "lander_mass_scale": dynamics_cfg["lander_mass_scale"],
        "leg_mass_scale": dynamics_cfg["leg_mass_scale"],
        "linear_damping": dynamics_cfg["linear_damping"],
        "angular_damping": dynamics_cfg["angular_damping"],
        "terrain_heights": dynamics_cfg["terrain_heights"],
        "action_repeat": int(dynamics_cfg["action_repeat"]),
        "action_delay": int(dynamics_cfg["action_delay"]),
        "action_noise_prob": float(dynamics_cfg["action_noise_prob"]),
        "action_noise_mode": str(dynamics_cfg["action_noise_mode"]),
        "mark_out_of_viewport_as_unsafe": bool(dynamics_cfg["mark_out_of_viewport_as_unsafe"]),
        "task_setting": args.task_setting,
        "task_settings_file": str(args.task_settings_file),
        "seed": int(args.seed),
        "policy_type": f"{args.task_role}_only",
        "algorithm": "ppo",
        "action_space": "discrete",
        "activation": "relu",
        "hidden_size": int(args.hidden_size),
        "device": args.device,
        "total_timesteps": int(args.total_timesteps),
        "eval_episodes_during_training": int(args.eval_episodes_during_training),
        "eval_episodes_post_training": int(args.eval_episodes_post_training),
        "rollout_steps": int(args.rollout_steps),
        "update_epochs": int(args.update_epochs),
        "minibatch_size": int(args.minibatch_size),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "clip_coef": float(args.clip_coef),
        "ent_coef": float(args.ent_coef),
        "vf_coef": float(args.vf_coef),
        "lr": float(args.lr),
        "max_grad_norm": float(args.max_grad_norm),
        "early_stop": bool(args.early_stop),
        "early_stop_min_steps": int(args.early_stop_min_steps),
        "early_stop_reward_threshold": (
            float(args.early_stop_reward_threshold) if args.early_stop_reward_threshold is not None else None
        ),
        "early_stop_failure_rate_threshold": (
            float(args.early_stop_failure_rate_threshold)
            if args.early_stop_failure_rate_threshold is not None
            else None
        ),
        "early_stop_deterministic_total_reward_threshold": (
            float(args.early_stop_deterministic_total_reward_threshold)
            if args.early_stop_deterministic_total_reward_threshold is not None
            else None
        ),
        "early_stop_deterministic_eval_episodes": int(args.early_stop_deterministic_eval_episodes),
        "trajectory_episodes": int(args.trajectory_episodes),
        "trajectory_max_frames_per_episode": int(args.trajectory_max_frames_per_episode),
        "downstream_eval_env_id": downstream_eval_env_id,
        "downstream_eval_task_id": downstream_eval_task_id,
        "downstream_eval_gravity": downstream_eval_gravity,
        "downstream_eval_append_task_id": downstream_eval_append_task_id,
        "downstream_eval_dynamics": downstream_eval_dynamics,
    }
    run_results = {
        f"{args.task_role}_mean_reward": float(mean_reward),
        f"{args.task_role}_std_reward": float(std_reward),
        f"{args.task_role}_failure_rate": float(failure_rate),
        "downstream_eval_performed": bool(downstream_eval_performed),
    }
    if downstream_eval_performed:
        run_results.update(
            {
                "downstream_mean_reward": float(downstream_mean_reward),
                "downstream_std_reward": float(downstream_std_reward),
                "downstream_failure_rate": float(downstream_failure_rate),
            },
        )
    artifacts = {
        "trajectory_plot_path": str(trajectory_source_plot_path),
        "downstream_eval_trajectory_plot_path": (
            str(trajectory_downstream_plot_path)
            if trajectory_downstream_plot_path is not None
            else None
        ),
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
        "training_data_path": str(training_data_path),
    }
    summary = {
        "run_settings": run_settings,
        "run_results": run_results,
        "artifacts": artifacts,
    }
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    print(
        f"{args.task_role.capitalize()} eval over {args.eval_episodes_post_training} episodes: "
        f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}",
    )
    if downstream_eval_performed:
        print(
            f"Downstream eval over {args.eval_episodes_post_training} episodes: "
            f"mean_reward={float(downstream_mean_reward):.2f} +/- {float(downstream_std_reward):.2f}",
        )
    print(f"Saved actor: {actor_path}")
    print(f"Saved critic: {critic_path}")
    print(f"Saved training data: {training_data_path}")
    print(f"Saved trajectory plot: {trajectory_source_plot_path}")
    if downstream_eval_performed and trajectory_downstream_plot_path is not None:
        print(f"Saved downstream evaluation trajectory plot: {trajectory_downstream_plot_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
