"""Create and smoke-test a discrete Highway Parking environment from task settings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct execution: `python rl_project/highway/create_env.py`.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rl_project.highway.parking_setup import (
    load_highway_task_setup,
    make_highway_parking_env,
)


def _parse_render_mode(value: str) -> str | None:
    value = value.strip().lower()
    if value in {"none", "null"}:
        return None
    if value not in {"rgb_array", "human"}:
        raise ValueError("render_mode must be one of: none, rgb_array, human")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Highway Parking environment with discrete actions from a "
            "task settings YAML."
        ),
    )
    parser.add_argument(
        "--task-settings-file",
        type=Path,
        default=Path(__file__).resolve().parent / "settings" / "task_settings.yaml",
        help="YAML file containing highway task settings.",
    )
    parser.add_argument(
        "--task-setting",
        type=str,
        default="default",
        help="Task-setting key from --task-settings-file.",
    )
    parser.add_argument(
        "--task-role",
        type=str,
        default="source",
        choices=["source", "downstream"],
        help="Which role to instantiate.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="none",
        help="Render mode: none, rgb_array, or human.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for reset.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=10,
        help="Number of random-action steps for smoke test.",
    )
    args = parser.parse_args()

    render_mode = _parse_render_mode(args.render_mode)
    setup = load_highway_task_setup(
        args.task_settings_file,
        task_setting=args.task_setting,
        task_role=args.task_role,
    )
    env = make_highway_parking_env(
        setup.env_base_config,
        setup.task_config,
        task_id=setup.task_id,
        append_task_id=setup.append_task_id,
        use_goal=setup.use_goal,
        n_bins_accel=setup.n_bins_accel,
        n_bins_steer=setup.n_bins_steer,
        render_mode=render_mode,
    )

    print("Created Highway Parking env")
    print(f"  task_setting: {args.task_setting}")
    print(f"  task_role: {args.task_role}")
    print(f"  action_space: {env.action_space}")
    print(f"  observation_space: {env.observation_space}")
    print(f"  goal_spots: {setup.task_config.get('goal_spots')}")
    print(f"  parked_vehicles_spots: {setup.task_config.get('parked_vehicles_spots')}")
    print(f"  vehicles_count: {setup.task_config.get('vehicles_count')}")

    observation, info = env.reset(seed=args.seed)
    print(f"Reset succeeded. obs_dim={getattr(observation, 'shape', None)}, safe={info.get('safe')}")

    total_reward = 0.0
    terminated = False
    truncated = False
    executed_steps = 0
    for step_idx in range(args.rollout_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        executed_steps += 1
        print(
            f"  step={step_idx:02d} action={action:02d} reward={float(reward):+.3f} "
            f"safe={bool(info.get('safe', True))}",
        )
        if terminated or truncated:
            break

    print(
        "Rollout complete. "
        f"steps={executed_steps}, terminated={terminated}, truncated={truncated}, "
        f"total_reward={total_reward:+.3f}",
    )
    env.close()


if __name__ == "__main__":
    main()
