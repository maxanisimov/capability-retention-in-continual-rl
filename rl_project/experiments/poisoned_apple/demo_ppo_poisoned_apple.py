"""Small PPO demo for PoisonedAppleEnv.

Usage:
    python rl_project/experiments/poisoned_apple/demo_ppo_poisoned_apple.py \
        --timesteps 15000 --render human

Optional trajectory plot (requires matplotlib):
    python rl_project/experiments/poisoned_apple/demo_ppo_poisoned_apple.py \
        --timesteps 15000 --plot-trajectory --save-dir /tmp/poisoned_apple_demo
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch

# Make rl_project importable when running this file directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RL_PROJECT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _RL_PROJECT_DIR not in sys.path:
    sys.path.insert(0, _RL_PROJECT_DIR)

from poisoned_apple_env import PoisonedAppleEnv, visualize_agent_trajectory  # noqa: E402
from utils.ppo_utils import PPOConfig, evaluate, ppo_train  # noqa: E402


def build_env(render_mode: str | None = None) -> PoisonedAppleEnv:
    """Create a deterministic poisoned-apple task used for train/eval/render."""
    return PoisonedAppleEnv(
        grid_size=6,
        agent_start_pos=(5, 0),
        safe_apple_positions=[(0, 5), (2, 2), (5, 5)],
        poisoned_apple_positions=[(1, 3), (4, 2)],
        observation_type="flat",
        max_steps=30,
        reward_safe=1.0,
        reward_poison=-1.0,
        reward_step=-0.01,
        render_mode=render_mode,
    )


def train_policy(total_timesteps: int, seed: int, device: str) -> tuple[torch.nn.Module, Any]:
    """Train PPO on PoisonedAppleEnv using repo PPO utilities."""
    env = build_env(render_mode=None)
    cfg = PPOConfig(
        seed=seed,
        total_timesteps=total_timesteps,
        eval_episodes=50,
        rollout_steps=256,
        update_epochs=6,
        minibatch_size=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        device=device,
    )
    actor, critic = ppo_train(env=env, cfg=cfg)
    return actor, critic


def rollout_and_render(
    actor: torch.nn.Module,
    seed: int,
    render_mode: str,
    max_steps: int,
) -> None:
    """Run one deterministic rollout and render each step."""
    env = build_env(render_mode=render_mode)
    obs, info = env.reset(seed=seed)
    done = False
    step = 0
    total_reward = 0.0
    unsafe_steps = 0

    actor.eval()
    actor_device = next(actor.parameters()).device

    print("\n--- Deterministic rollout ---")
    print(f"start_info={info}")

    if render_mode == "human":
        env.render()

    while not done and step < max_steps:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=actor_device).unsqueeze(0)
        with torch.no_grad():
            logits = actor(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        if not info.get("safe", True):
            unsafe_steps += 1

        step += 1
        print(
            f"step={step:02d} action={action} reward={reward:+.2f} "
            f"safe={info.get('safe')} cost={info.get('cost')} "
            f"safe_remaining={info.get('safe_apples_remaining')} "
            f"poison_remaining={info.get('poisoned_apples_remaining')}"
        )

        if render_mode == "human":
            env.render()

    print(
        f"rollout_done steps={step} total_reward={total_reward:.2f} "
        f"unsafe_steps={unsafe_steps} terminated={terminated} truncated={truncated}"
    )
    env.close()


def maybe_plot_trajectory(
    actor: torch.nn.Module,
    save_dir: str | None,
    cfg_name: str,
) -> None:
    """Plot a trajectory using poisoned_apple_env utility (matplotlib required)."""
    try:
        env_plot = build_env(render_mode=None)
        visualize_agent_trajectory(
            env=env_plot,
            actor=actor,
            num_episodes=1,
            env_name="PoisonedApple",
            cfg_name=cfg_name,
            actor_name="PPO",
            save_dir=save_dir,
        )
        env_plot.close()
    except ModuleNotFoundError as exc:
        print(f"Skipping trajectory plot: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on PoisonedApple and render a rollout.")
    parser.add_argument("--timesteps", type=int, default=15_000, help="PPO training timesteps.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Torch device (cpu/cuda).",
    )
    parser.add_argument(
        "--render",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
        help="Render mode for deterministic rollout.",
    )
    parser.add_argument(
        "--max-rollout-steps",
        type=int,
        default=64,
        help="Max steps for rendered rollout episode.",
    )
    parser.add_argument(
        "--plot-trajectory",
        action="store_true",
        help="Also generate a matplotlib trajectory plot.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save plot when --plot-trajectory is enabled. If omitted, shows plot interactively.",
    )
    args = parser.parse_args()

    print(
        "Training PPO on PoisonedAppleEnv "
        f"(timesteps={args.timesteps}, seed={args.seed}, device={args.device})"
    )
    actor, _ = train_policy(
        total_timesteps=args.timesteps,
        seed=args.seed,
        device=args.device,
    )

    eval_env = build_env(render_mode=None)
    mean_r, std_r, failure_rate = evaluate(
        env=eval_env,
        actor=actor,
        episodes=50,
        seed=args.seed,
        device=args.device,
        deterministic=True,
        render_mode=None,
    )
    eval_env.close()

    print(
        f"Post-train evaluation: mean_reward={mean_r:.2f} +/- {std_r:.2f}, "
        f"failure_rate={failure_rate:.2f}"
    )

    rollout_and_render(
        actor=actor,
        seed=args.seed,
        render_mode=args.render,
        max_steps=args.max_rollout_steps,
    )

    if args.plot_trajectory:
        maybe_plot_trajectory(
            actor=actor,
            save_dir=args.save_dir,
            cfg_name=f"ppo_t{args.timesteps}",
        )


if __name__ == "__main__":
    main()
