"""Generate task videos for LunarLander with random or trained policies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from experiments.pipelines.lunarlander.core.env.env_factory import _make_lunarlander_env
from experiments.pipelines.lunarlander.core.env.task_loading import (
    _load_task_settings,
    _resolve_lunarlander_dynamics,
)
from experiments.pipelines.lunarlander.core.eval.rollout_video import (
    _build_actor_from_state_dict,
    _resolve_actor_path,
    _write_video,
)
from experiments.pipelines.lunarlander.core.orchestration.run_paths import (
    default_outputs_root,
    default_task_settings_file,
    resolve_policy_dir,
)


def _sanitize_label(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip())
    if not cleaned:
        return "policy"
    return cleaned


def _resolve_env_config(
    *,
    task_settings_file: Path,
    task_setting: str,
    task_role: str,
) -> tuple[str, dict[str, Any]]:
    task_cfg = _load_task_settings(task_settings_file, task_setting, task_role)
    env_id = str(task_cfg.get("env_id") or "LunarLander-v3")
    continuous = bool(task_cfg.get("continuous", False))
    if continuous:
        raise ValueError("Only discrete-action LunarLander is supported (`continuous=False`).")

    gravity_raw = task_cfg.get("gravity")
    gravity = None if gravity_raw is None else float(gravity_raw)
    default_task_id = 0.0 if task_role == "source" else 1.0
    task_id = float(task_cfg.get("task_id", default_task_id))
    append_task_id = bool(task_cfg.get("append_task_id", True))
    dynamics = _resolve_lunarlander_dynamics(
        task_cfg,
        cfg_name=f"task_settings[{task_setting}:{task_role}]",
    )
    env_kwargs = {
        "gravity": gravity,
        "task_id": task_id,
        "append_task_id": append_task_id,
        "enable_wind": bool(dynamics["enable_wind"]),
        "wind_power": dynamics["wind_power"],
        "turbulence_power": dynamics["turbulence_power"],
        "initial_random_strength": dynamics["initial_random_strength"],
        "dispersion_strength": dynamics["dispersion_strength"],
        "main_engine_power": dynamics["main_engine_power"],
        "side_engine_power": dynamics["side_engine_power"],
        "leg_spring_torque": dynamics["leg_spring_torque"],
        "lander_mass_scale": dynamics["lander_mass_scale"],
        "leg_mass_scale": dynamics["leg_mass_scale"],
        "linear_damping": dynamics["linear_damping"],
        "angular_damping": dynamics["angular_damping"],
        "terrain_heights": dynamics["terrain_heights"],
        "action_repeat": int(dynamics["action_repeat"]),
        "action_delay": int(dynamics["action_delay"]),
        "action_noise_prob": float(dynamics["action_noise_prob"]),
        "action_noise_mode": str(dynamics["action_noise_mode"]),
        "mark_out_of_viewport_as_unsafe": bool(dynamics["mark_out_of_viewport_as_unsafe"]),
    }
    return env_id, env_kwargs


def _resolve_actor_checkpoint(
    *,
    policy: str,
    outputs_root: Path,
    task_setting: str,
    seed: int,
) -> tuple[Path | None, str]:
    policy_label = policy.strip()
    if policy_label.lower() == "random":
        return None, "random"

    policy_path = Path(policy_label).expanduser()
    if policy_path.exists():
        resolved_policy_path = policy_path.resolve()
        if resolved_policy_path.is_dir():
            actor_path = _resolve_actor_path(resolved_policy_path, actor_path=None)
            return actor_path.resolve(), _sanitize_label(resolved_policy_path.name)
        return resolved_policy_path, _sanitize_label(resolved_policy_path.stem)

    policy_dir = resolve_policy_dir(outputs_root, task_setting, seed, policy_label)
    if not policy_dir.exists():
        raise FileNotFoundError(
            "Policy path does not exist and no policy directory was found under outputs: "
            f"policy='{policy_label}', expected directory like {policy_dir}",
        )
    actor_path = _resolve_actor_path(policy_dir, actor_path=None)
    return actor_path.resolve(), _sanitize_label(policy_label)


def _load_actor(actor_path: Path, *, device: str) -> torch.nn.Sequential:
    state_dict = torch.load(actor_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Expected actor state_dict at {actor_path}, got {type(state_dict)}.")
    actor = _build_actor_from_state_dict(state_dict).to(device)
    actor.eval()
    return actor


def _select_action(
    *,
    env,
    actor: torch.nn.Module | None,
    obs: np.ndarray,
    deterministic: bool,
    device: str,
) -> int:
    if actor is None:
        return int(env.action_space.sample())

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = actor(obs_t)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
    return int(action.item())


def _rollout_frames(
    *,
    env,
    actor: torch.nn.Module | None,
    seed: int,
    episodes: int,
    max_steps_per_episode: int,
    deterministic: bool,
    device: str,
) -> tuple[list[np.ndarray], list[float], list[int]]:
    frames: list[np.ndarray] = []
    returns: list[float] = []
    lengths: list[int] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        ep_len = 0

        first_frame = env.render()
        if first_frame is None:
            raise RuntimeError("env.render() returned None; create env with render_mode='rgb_array'.")
        frames.append(np.asarray(first_frame).copy())

        done = False
        while not done and ep_len < max_steps_per_episode:
            action = _select_action(
                env=env,
                actor=actor,
                obs=np.asarray(obs, dtype=np.float32),
                deterministic=deterministic,
                device=device,
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            done = bool(terminated or truncated)

            frame = env.render()
            if frame is None:
                raise RuntimeError("env.render() returned None; create env with render_mode='rgb_array'.")
            frames.append(np.asarray(frame).copy())

        returns.append(ep_return)
        lengths.append(ep_len)

    return frames, returns, lengths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a LunarLander task video for source/downstream environment "
            "using a random policy (default) or a trained policy."
        ),
    )
    parser.add_argument("--task-setting", type=str, required=True, help="Task-setting name in task_settings.yaml.")
    parser.add_argument(
        "--task-role",
        type=str,
        choices=["source", "downstream"],
        required=True,
        help="Task role to render.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="random",
        help=(
            "Policy identifier. Use 'random' (default), a policy subdir name under "
            "outputs/<task-setting>/seed_<seed>/, or an explicit actor/policy path."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Rollout seed.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to include in the video.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=default_outputs_root(),
        help="Outputs root used to resolve policy directories.",
    )
    parser.add_argument(
        "--task-settings-file",
        type=Path,
        default=default_task_settings_file(),
        help="Task settings YAML path.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for policy inference.")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic (argmax) actions for learned policies.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=2000,
        help="Safety cap on episode length.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output video fps.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit output video path (.mp4/.gif).",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError(f"--episodes must be > 0, got {args.episodes}.")
    if args.max_steps_per_episode <= 0:
        raise ValueError(f"--max-steps-per-episode must be > 0, got {args.max_steps_per_episode}.")
    if args.fps <= 0:
        raise ValueError(f"--fps must be > 0, got {args.fps}.")

    env_id, env_kwargs = _resolve_env_config(
        task_settings_file=args.task_settings_file,
        task_setting=args.task_setting,
        task_role=args.task_role,
    )
    actor_path, policy_label = _resolve_actor_checkpoint(
        policy=args.policy,
        outputs_root=args.outputs_root,
        task_setting=args.task_setting,
        seed=int(args.seed),
    )
    actor = _load_actor(actor_path, device=args.device) if actor_path is not None else None

    env = _make_lunarlander_env(
        env_id,
        render_mode="rgb_array",
        **env_kwargs,
    )
    try:
        frames, returns, lengths = _rollout_frames(
            env=env,
            actor=actor,
            seed=int(args.seed),
            episodes=int(args.episodes),
            max_steps_per_episode=int(args.max_steps_per_episode),
            deterministic=bool(args.deterministic),
            device=str(args.device),
        )
    finally:
        env.close()

    if args.output_path is None:
        video_dir = args.outputs_root / args.task_setting / f"seed_{int(args.seed)}" / "videos"
        video_name = (
            f"task-{args.task_role}_policy-{_sanitize_label(policy_label)}"
            f"_seed-{int(args.seed)}_episodes-{int(args.episodes)}.mp4"
        )
        output_path = video_dir / video_name
    else:
        output_path = args.output_path
    _write_video(frames, output_path, fps=int(args.fps))

    metadata = {
        "task_setting": str(args.task_setting),
        "task_role": str(args.task_role),
        "policy": str(args.policy),
        "resolved_policy_label": str(policy_label),
        "resolved_actor_path": (str(actor_path) if actor_path is not None else None),
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "deterministic": bool(args.deterministic),
        "device": str(args.device),
        "fps": int(args.fps),
        "env_id": str(env_id),
        "env_kwargs": env_kwargs,
        "frame_count": int(len(frames)),
        "episode_returns": [float(v) for v in returns],
        "episode_lengths": [int(v) for v in lengths],
        "video_path": str(output_path),
    }
    metadata_path = output_path.parent / f"{output_path.stem}_metadata.yaml"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")

    print(f"Saved video: {output_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Episode returns: {[round(v, 3) for v in returns]}")
    print(f"Episode lengths: {lengths}")


if __name__ == "__main__":
    main()

