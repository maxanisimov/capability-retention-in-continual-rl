"""Train one scaled FrozenLake source policy, then evaluate/plot on source and downstream tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch
import yaml

from rl_project.experiments.frozenlake_scaled.sweep_scaled_ppo import CoordObsWrapper, DenseShapingWrapper, SafetyFlagWrapper
from rl_project.utils.gymnasium_utils import plot_episode
from rl_project.utils.ppo_utils import PPOConfig, evaluate, ppo_train


def make_env_from_layout(
    env_map: list[str],
    max_episode_steps: int,
    *,
    task_num: float = 0.0,
    shaped: bool,
    render_mode: str | None = None,
) -> gym.Env:
    env = gym.make(
        "FrozenLake-v1",
        desc=env_map,
        is_slippery=False,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )
    env = CoordObsWrapper(env, task_num=task_num)
    env = SafetyFlagWrapper(env)
    if shaped:
        env = DenseShapingWrapper(env)
    return env


def _activation_layer(name: str) -> torch.nn.Module:
    if name == "tanh":
        return torch.nn.Tanh()
    if name == "relu":
        return torch.nn.ReLU()
    raise ValueError(f"Unsupported activation '{name}'. Expected one of: tanh, relu")


def build_actor_critic(
    obs_dim: int,
    hidden: int,
    activation: str = "relu",
) -> tuple[torch.nn.Sequential, torch.nn.Sequential]:
    act1 = _activation_layer(activation)
    act2 = _activation_layer(activation)
    act3 = _activation_layer(activation)
    act4 = _activation_layer(activation)
    actor = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden),
        act1,
        torch.nn.Linear(hidden, hidden),
        act2,
        torch.nn.Linear(hidden, 4),
    )
    critic = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden),
        act3,
        torch.nn.Linear(hidden, hidden),
        act4,
        torch.nn.Linear(hidden, 1),
    )
    return actor, critic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one scaled FrozenLake source policy and save source/downstream evaluations.",
    )
    parser.add_argument("--layout", type=str, default="diagonal_30x30")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--settings-file",
        type=Path,
        default=Path(__file__).resolve().parent / "settings" / "train_source_policy_settings.yaml",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(__file__).resolve().parent / "settings" / "source_envs.yaml",
        help="Source environment definitions.",
    )
    parser.add_argument(
        "--downstream-env-file",
        type=Path,
        default=Path(__file__).resolve().parent / "settings" / "downstream_envs.yaml",
        help="Downstream environment definitions.",
    )
    parser.add_argument(
        "--adapt-settings-file",
        type=Path,
        default=Path(__file__).resolve().parent / "settings" / "downstream_adaptation_settings_ppo.yaml",
        help="Optional adaptation settings used to read source/downstream task IDs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["tanh", "relu"],
        default="relu",
        help="Hidden-layer activation for actor/critic (default: relu).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    all_settings = yaml.safe_load(args.settings_file.read_text(encoding="utf-8"))
    all_source_envs = yaml.safe_load(args.env_file.read_text(encoding="utf-8"))
    all_downstream_envs = yaml.safe_load(args.downstream_env_file.read_text(encoding="utf-8"))
    all_adapt_settings = yaml.safe_load(args.adapt_settings_file.read_text(encoding="utf-8"))

    if args.layout not in all_settings:
        raise ValueError(f"Layout '{args.layout}' not found in {args.settings_file}")
    if args.layout not in all_source_envs:
        raise ValueError(f"Layout '{args.layout}' not found in {args.env_file}")
    if args.layout not in all_downstream_envs:
        raise ValueError(f"Layout '{args.layout}' not found in {args.downstream_env_file}")

    source_env_cfg = all_source_envs[args.layout]
    downstream_env_cfg = all_downstream_envs[args.layout]
    run_cfg = all_settings[args.layout]
    adapt_cfg = all_adapt_settings.get(args.layout, {}) if isinstance(all_adapt_settings, dict) else {}
    ppo_cfg_data = run_cfg["ppo"]
    raw_eval_cfg = run_cfg.get("raw_eval", {})

    source_map: list[str] = source_env_cfg["env1_map"]
    downstream_map: list[str] = downstream_env_cfg["env2_map"]
    source_task_num = float(adapt_cfg.get("source_task_num", 0.0))
    downstream_task_num = float(adapt_cfg.get("downstream_task_num", 1.0))
    max_episode_steps = int(source_env_cfg["max_episode_steps"])
    downstream_max_episode_steps = int(downstream_env_cfg.get("max_episode_steps", max_episode_steps))
    eval_episodes = int(raw_eval_cfg.get("episodes", 20))

    early_stop = bool(ppo_cfg_data.get("early_stop", False))
    early_stop_min_steps = int(ppo_cfg_data.get("early_stop_min_steps", 0))
    early_stop_reward_threshold = ppo_cfg_data.get("early_stop_reward_threshold", None)
    if early_stop_reward_threshold is not None:
        early_stop_reward_threshold = float(early_stop_reward_threshold)
    early_stop_failure_rate_threshold = ppo_cfg_data.get("early_stop_failure_rate_threshold", None)
    if early_stop_failure_rate_threshold is not None:
        early_stop_failure_rate_threshold = float(early_stop_failure_rate_threshold)
    early_stop_deterministic_total_reward_threshold = ppo_cfg_data.get(
        "early_stop_deterministic_total_reward_threshold",
        None,
    )
    if early_stop_deterministic_total_reward_threshold is not None:
        early_stop_deterministic_total_reward_threshold = float(early_stop_deterministic_total_reward_threshold)
    early_stop_deterministic_eval_episodes = int(
        ppo_cfg_data.get("early_stop_deterministic_eval_episodes", 1),
    )

    run_dir = args.output_dir / args.layout / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env_from_layout(
        source_map,
        max_episode_steps,
        task_num=source_task_num,
        shaped=True,
    )
    obs_dim = train_env.observation_space.shape[0]
    hidden = int(ppo_cfg_data["hidden"])
    actor, critic = build_actor_critic(
        obs_dim=obs_dim,
        hidden=hidden,
        activation=args.activation,
    )

    ppo_cfg = PPOConfig(
        seed=args.seed,
        total_timesteps=int(ppo_cfg_data["total_timesteps"]),
        eval_episodes=eval_episodes,
        rollout_steps=int(ppo_cfg_data["rollout_steps"]),
        update_epochs=int(ppo_cfg_data["update_epochs"]),
        minibatch_size=int(ppo_cfg_data["minibatch_size"]),
        gamma=float(ppo_cfg_data["gamma"]),
        gae_lambda=float(ppo_cfg_data["gae_lambda"]),
        clip_coef=float(ppo_cfg_data["clip_coef"]),
        ent_coef=float(ppo_cfg_data["ent_coef"]),
        vf_coef=float(ppo_cfg_data["vf_coef"]),
        lr=float(ppo_cfg_data["lr"]),
        max_grad_norm=float(ppo_cfg_data["max_grad_norm"]),
        device=args.device,
        early_stop=early_stop,
        early_stop_min_steps=early_stop_min_steps,
        early_stop_reward_threshold=early_stop_reward_threshold,
        early_stop_failure_rate_threshold=early_stop_failure_rate_threshold,
        early_stop_deterministic_total_reward_threshold=early_stop_deterministic_total_reward_threshold,
        early_stop_deterministic_eval_episodes=early_stop_deterministic_eval_episodes,
    )

    print(f"Training layout={args.layout} seed={args.seed} activation={args.activation} ...")
    if early_stop:
        print(
            "Early stopping enabled; periodic stop checks use sparse source rewards "
            f"(threshold={early_stop_reward_threshold}, "
            f"failure_rate_threshold={early_stop_failure_rate_threshold}, "
            f"deterministic_total_reward_threshold={early_stop_deterministic_total_reward_threshold}).",
        )

    early_stop_eval_env = None
    if early_stop or early_stop_deterministic_total_reward_threshold is not None:
        early_stop_eval_env = make_env_from_layout(
            source_map,
            max_episode_steps,
            task_num=source_task_num,
            shaped=False,
        )
    actor, critic, training_data = ppo_train(  # type: ignore[assignment]
        train_env,
        ppo_cfg,
        actor,
        critic,
        early_stop_eval_env=early_stop_eval_env,
        return_training_data=True,
    )

    source_eval_env = make_env_from_layout(
        source_map,
        max_episode_steps,
        task_num=source_task_num,
        shaped=False,
    )
    source_mean_reward, source_std_reward, source_failure_rate = evaluate(
        source_eval_env,
        actor,
        episodes=eval_episodes,
        deterministic=True,
        device=args.device,
    )
    source_eval_env.close()

    downstream_eval_env = make_env_from_layout(
        downstream_map,
        downstream_max_episode_steps,
        task_num=downstream_task_num,
        shaped=False,
    )
    downstream_mean_reward, downstream_std_reward, downstream_failure_rate = evaluate(
        downstream_eval_env,
        actor,
        episodes=eval_episodes,
        deterministic=True,
        device=args.device,
    )
    downstream_eval_env.close()

    print(
        f"Source eval ({eval_episodes} eps): "
        f"mean_reward={source_mean_reward:.3f}, std_reward={source_std_reward:.3f}, "
        f"failure_rate={source_failure_rate:.3f}",
    )
    print(
        f"Downstream eval ({eval_episodes} eps): "
        f"mean_reward={downstream_mean_reward:.3f}, std_reward={downstream_std_reward:.3f}, "
        f"failure_rate={downstream_failure_rate:.3f}",
    )

    if source_mean_reward < 1.0:
        raise RuntimeError(
            f"Training did not reach successful policy for {args.layout}. "
            f"mean_reward={source_mean_reward:.3f}",
        )

    source_run_dir = run_dir / "source"
    source_run_dir.mkdir(parents=True, exist_ok=True)

    actor_path = source_run_dir / "actor.pt"
    torch.save(actor.state_dict(), actor_path)

    critic_path = source_run_dir / "critic.pt"
    torch.save(critic.state_dict(), critic_path)

    training_data_path = source_run_dir / "training_data.pt"
    torch.save(training_data, training_data_path)

    source_plot_path = source_run_dir / "trajectory_source.png"
    source_render_env = make_env_from_layout(
        source_map,
        max_episode_steps,
        task_num=source_task_num,
        shaped=False,
        render_mode="rgb_array",
    )
    try:
        plot_episode(
            env=source_render_env,
            actor=actor,
            seed=args.seed,
            deterministic=True,
            save_path=str(source_plot_path),
            title=f"Source Task Trajectory: {args.layout}",
        )
    finally:
        source_render_env.close()

    downstream_plot_path = source_run_dir / "trajectory_downstream.png"
    downstream_render_env = make_env_from_layout(
        downstream_map,
        downstream_max_episode_steps,
        task_num=downstream_task_num,
        shaped=False,
        render_mode="rgb_array",
    )
    try:
        plot_episode(
            env=downstream_render_env,
            actor=actor,
            seed=args.seed,
            deterministic=True,
            save_path=str(downstream_plot_path),
            title=f"Downstream Task Trajectory: {args.layout}",
        )
    finally:
        downstream_render_env.close()

    summary = {
        "layout": args.layout,
        "seed": args.seed,
        "activation": args.activation,
        "source_task_num": source_task_num,
        "downstream_task_num": downstream_task_num,
        "eval_episodes": int(eval_episodes),
        "source_eval_episodes": int(eval_episodes),
        "downstream_eval_episodes": int(eval_episodes),
        "train_early_stop": early_stop,
        "train_early_stop_min_steps": int(early_stop_min_steps),
        "train_early_stop_reward_threshold": (
            float(early_stop_reward_threshold) if early_stop_reward_threshold is not None else None
        ),
        "train_early_stop_failure_rate_threshold": (
            float(early_stop_failure_rate_threshold) if early_stop_failure_rate_threshold is not None else None
        ),
        "train_early_stop_deterministic_total_reward_threshold": (
            float(early_stop_deterministic_total_reward_threshold)
            if early_stop_deterministic_total_reward_threshold is not None
            else None
        ),
        "train_early_stop_deterministic_eval_episodes": int(early_stop_deterministic_eval_episodes),
        "source_mean_reward": float(source_mean_reward),
        "source_std_reward": float(source_std_reward),
        "source_failure_rate": float(source_failure_rate),
        "downstream_mean_reward": float(downstream_mean_reward),
        "downstream_std_reward": float(downstream_std_reward),
        "downstream_failure_rate": float(downstream_failure_rate),
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
        "training_data_path": str(training_data_path),
        "trajectory_source_plot_path": str(source_plot_path),
        "trajectory_downstream_plot_path": str(downstream_plot_path),
    }
    (source_run_dir / "run_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    print(f"Saved actor: {actor_path}")
    print(f"Saved critic: {critic_path}")
    print(f"Saved training data: {training_data_path}")
    print(f"Saved source trajectory plot: {source_plot_path}")
    print(f"Saved downstream trajectory plot: {downstream_plot_path}")


if __name__ == "__main__":
    main()
