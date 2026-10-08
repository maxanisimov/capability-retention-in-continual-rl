"""Run random-policy episodes and report continuous-state shield interventions.

Examples
--------
Run every environment without rendering::

    python core/scripts/demo_continuous_state_shields.py --env all --episodes 3

Render CartPole and print the first twenty activation decisions::

    python core/scripts/demo_continuous_state_shields.py --env cartpole --render --show-activations

LunarLander requires the optional Gymnasium Box2D dependencies.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field

import gymnasium as gym
from continuous_state_shields import (
    CartPoleShield,
    LunarLanderShield,
    MountainCarShield,
    SafetyShield,
)


@dataclass
class RunStatistics:
    steps: int = 0
    active_states: int = 0
    overrides: int = 0
    safe_set_sizes: Counter[int] = field(default_factory=Counter)
    episode_rewards: list[float] = field(default_factory=list)
    outcomes: Counter[str] = field(default_factory=Counter)


ENVIRONMENTS: dict[str, tuple[str, type[SafetyShield]]] = {
    "mountaincar": ("MountainCar-v0", MountainCarShield),
    "cartpole": ("CartPole-v1", CartPoleShield),
    "lunarlander": ("LunarLander-v3", LunarLanderShield),
}


def run_demo(
    name: str,
    *,
    episodes: int,
    seed: int,
    render: bool,
    show_activations: bool,
) -> RunStatistics:
    """Run one named environment and return aggregate shield statistics."""
    env_id, shield_type = ENVIRONMENTS[name]
    env = gym.make(env_id, render_mode="human" if render else None)
    shield = shield_type()
    statistics = RunStatistics()
    shown = 0
    try:
        env.action_space.seed(seed)
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_reward = 0.0
            while True:
                proposed = int(env.action_space.sample())
                safe_actions, info = shield.get_safe_actions(
                    observation, return_info=True
                )
                executed = shield.shield_action(observation, proposed)
                statistics.steps += 1
                statistics.active_states += int(info["shield_active"])
                statistics.overrides += int(executed != proposed)
                statistics.safe_set_sizes[len(safe_actions)] += 1
                if show_activations and info["shield_active"] and shown < 20:
                    print(
                        f"  activation step={statistics.steps}: proposed={proposed}, "
                        f"allowed={safe_actions}, executed={executed}, reason={info['reason']}"
                    )
                    shown += 1
                observation, reward, terminated, truncated, _ = env.step(executed)
                episode_reward += float(reward)
                if terminated or truncated:
                    statistics.outcomes[
                        "terminated" if terminated else "truncated"
                    ] += 1
                    statistics.episode_rewards.append(episode_reward)
                    break
    finally:
        env.close()
    return statistics


def print_statistics(name: str, statistics: RunStatistics) -> None:
    """Print the fields requested by the validation plan."""
    active_percentage = (
        100.0 * statistics.active_states / statistics.steps if statistics.steps else 0.0
    )
    mean_reward = sum(statistics.episode_rewards) / len(statistics.episode_rewards)
    print(f"\n{name}")
    print(f"  environment steps: {statistics.steps}")
    print(
        f"  shield-active states: {statistics.active_states} ({active_percentage:.1f}%)"
    )
    print(f"  proposed actions overridden: {statistics.overrides}")
    print(f"  safe-action-set sizes: {dict(sorted(statistics.safe_set_sizes.items()))}")
    print(f"  episode outcomes: {dict(statistics.outcomes)}")
    print(f"  mean episode reward: {mean_reward:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["all", *ENVIRONMENTS], default="all")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--show-activations", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    return args


def main() -> None:
    args = parse_args()
    names = ENVIRONMENTS if args.env == "all" else [args.env]
    for name in names:
        try:
            statistics = run_demo(
                name,
                episodes=args.episodes,
                seed=args.seed,
                render=args.render,
                show_activations=args.show_activations,
            )
        except gym.error.DependencyNotInstalled as error:
            print(f"\n{name}: skipped ({error})")
            continue
        print_statistics(name, statistics)


if __name__ == "__main__":
    main()
