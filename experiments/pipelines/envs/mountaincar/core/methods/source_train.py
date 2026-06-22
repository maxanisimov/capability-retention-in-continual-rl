"""Train a default Mountain Car PPO source policy.

The default configuration targets Gymnasium's stock Mountain Car dynamics via
``TunableMountainCar-v0`` and appends a constant task-id feature for future
continual-learning experiments.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import gymnasium as gym
import torch
import yaml

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Allow running this file directly from experiments/pipelines/envs/mountaincar/core/methods.
_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.envs.mountaincar.core.env.env_factory import make_mountaincar_env
from experiments.pipelines.envs.mountaincar.core.env.tunable_mountain_car import (
    TUNABLE_MOUNTAIN_CAR_V0_ID,
)
from experiments.pipelines.envs.mountaincar.core.orchestration.run_paths import (
    DEFAULT_TASK_SETTING,
    NOADAPT_POLICY_SUBDIR,
    default_outputs_root,
    seed_run_dir,
)
from experiments.utils.gymnasium_utils import plot_episode
from experiments.utils.ppo_utils import PPOConfig, evaluate_with_success, ppo_train


DEFAULT_SOLVED_REWARD_THRESHOLD = -110.0


def build_actor_critic(
    obs_dim: int,
    n_actions: int,
    hidden_size: int,
) -> tuple[torch.nn.Sequential, torch.nn.Sequential]:
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a default Mountain Car source policy with PPO and ReLU MLPs.",
    )
    parser.add_argument("--task-setting", type=str, default=DEFAULT_TASK_SETTING)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-id", type=str, default=TUNABLE_MOUNTAIN_CAR_V0_ID)
    parser.add_argument(
        "--append-task-id",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append a constant task id to observations.",
    )
    parser.add_argument("--task-id", type=float, default=0.0)
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
        help="Number of episodes for final post-training evaluation.",
    )
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--solved-reward-threshold",
        type=float,
        default=DEFAULT_SOLVED_REWARD_THRESHOLD,
        help="Mean reward threshold used to mark the trained policy as solved.",
    )
    parser.add_argument("--output-dir", type=Path, default=default_outputs_root())
    return parser


def _make_default_env(
    *,
    env_id: str,
    task_id: float,
    append_task_id: bool,
    render_mode: str | None,
) -> gym.Env:
    return make_mountaincar_env(
        env_id=env_id,
        task_id=task_id,
        append_task_id=append_task_id,
        render_mode=render_mode,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    train_env = _make_default_env(
        env_id=args.env_id,
        task_id=args.task_id,
        append_task_id=bool(args.append_task_id),
        render_mode=None,
    )
    if not isinstance(train_env.action_space, gym.spaces.Discrete):
        raise ValueError("Mountain Car trainer expects a discrete action space.")
    if not isinstance(train_env.observation_space, gym.spaces.Box):
        raise ValueError("Mountain Car trainer expects a Box observation space.")

    obs_dim = int(train_env.observation_space.shape[0])
    n_actions = int(train_env.action_space.n)
    actor, critic = build_actor_critic(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_size=int(args.hidden_size),
    )

    ppo_cfg = PPOConfig(
        seed=int(args.seed),
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
        device=str(args.device),
    )

    print(
        "Training MountainCar source policy | "
        f"env_id={args.env_id} | seed={args.seed} | "
        f"append_task_id={args.append_task_id} | task_id={args.task_id} | "
        f"activation=relu | hidden_size={args.hidden_size}"
    )
    actor, critic, training_data = ppo_train(  # type: ignore[assignment]
        env=train_env,
        cfg=ppo_cfg,
        actor_warm_start=actor,
        critic_warm_start=critic,
        return_training_data=True,
    )
    train_env.close()

    eval_env = _make_default_env(
        env_id=args.env_id,
        task_id=args.task_id,
        append_task_id=bool(args.append_task_id),
        render_mode=None,
    )
    mean_reward, std_reward, failure_rate, success_rate = evaluate_with_success(
        env=eval_env,
        actor=actor,
        episodes=int(args.eval_episodes_post_training),
        seed=int(args.seed),
        deterministic=True,
        device=str(args.device),
    )
    eval_env.close()

    solved = bool(float(mean_reward) >= float(args.solved_reward_threshold))

    run_dir = seed_run_dir(args.output_dir, str(args.task_setting), int(args.seed))
    source_run_dir = run_dir / NOADAPT_POLICY_SUBDIR
    source_run_dir.mkdir(parents=True, exist_ok=True)

    actor_path = source_run_dir / "actor.pt"
    critic_path = source_run_dir / "critic.pt"
    training_data_path = source_run_dir / "training_data.pt"
    summary_path = source_run_dir / "run_summary.yaml"
    trajectory_plot_path = source_run_dir / "trajectory_source.png"

    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    torch.save(training_data, training_data_path)

    render_env = _make_default_env(
        env_id=args.env_id,
        task_id=args.task_id,
        append_task_id=bool(args.append_task_id),
        render_mode="rgb_array",
    )
    try:
        plot_episode(
            env=render_env,
            actor=actor,
            seed=int(args.seed),
            deterministic=True,
            save_path=str(trajectory_plot_path),
            n_cols=5,
            one_row=True,
            title="MountainCar source trajectory",
        )
    finally:
        render_env.close()

    run_settings = {
        "env_id": str(args.env_id),
        "task_setting": str(args.task_setting),
        "policy_name": NOADAPT_POLICY_SUBDIR,
        "seed": int(args.seed),
        "algorithm": "ppo",
        "action_space": "discrete",
        "activation": "relu",
        "hidden_size": int(args.hidden_size),
        "obs_dim": int(obs_dim),
        "n_actions": int(n_actions),
        "append_task_id": bool(args.append_task_id),
        "task_id": float(args.task_id),
        "device": str(args.device),
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
        "solved_reward_threshold": float(args.solved_reward_threshold),
    }
    run_results = {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "failure_rate": float(failure_rate),
        "success_rate": float(success_rate),
        "solved": solved,
    }
    artifacts = {
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
        "training_data_path": str(training_data_path),
        "trajectory_source_plot_path": str(trajectory_plot_path),
    }
    summary = {
        "run_settings": run_settings,
        "run_results": run_results,
        "artifacts": artifacts,
        **run_settings,
        **run_results,
        **artifacts,
    }
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    print(
        f"Post-training eval ({args.eval_episodes_post_training} eps): "
        f"mean_reward={mean_reward:.3f}, std_reward={std_reward:.3f}, "
        f"failure_rate={failure_rate:.3f}, success_rate={success_rate:.3f}, "
        f"solved={solved}"
    )
    print(f"Saved actor: {actor_path}")
    print(f"Saved critic: {critic_path}")
    print(f"Saved training data: {training_data_path}")
    print(f"Saved trajectory plot: {trajectory_plot_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

