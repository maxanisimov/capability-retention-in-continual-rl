"""Roll out a trained LunarLander policy and save an RGB video next to the checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from experiments.pipelines.lunarlander.core.env.env_factory import (
    _make_lunarlander_env,
)
from experiments.pipelines.lunarlander.core.env.task_loading import (
    _load_task_settings,
    _resolve_lunarlander_dynamics,
)
from experiments.pipelines.lunarlander.core.methods.source_train import (
    build_actor_critic,
)
from experiments.pipelines.lunarlander.core.orchestration.run_paths import default_task_settings_file


def _resolve_actor_path(policy_dir: Path, actor_path: Path | None) -> Path:
    if actor_path is not None:
        resolved = actor_path if actor_path.is_absolute() else (policy_dir / actor_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Actor checkpoint not found: {resolved}")
        return resolved

    candidates = [
        policy_dir / "actor.pt",
        policy_dir / "ewc_actor.pt",
        policy_dir / "rashomon_actor.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find actor checkpoint in {policy_dir}. "
        f"Tried: {[str(c) for c in candidates]}",
    )


def _build_actor_from_state_dict(state_dict: dict[str, torch.Tensor]) -> torch.nn.Sequential:
    required = ("0.weight", "2.weight", "4.weight")
    missing = [key for key in required if key not in state_dict]
    if missing:
        raise ValueError(
            f"Unsupported actor checkpoint format; missing keys: {missing}. "
            "Expected a 3-layer Sequential MLP actor.",
        )

    first_w = state_dict["0.weight"]
    second_w = state_dict["2.weight"]
    third_w = state_dict["4.weight"]
    if first_w.ndim != 2 or second_w.ndim != 2 or third_w.ndim != 2:
        raise ValueError("Expected 2D linear weight tensors for actor checkpoint.")

    obs_dim = int(first_w.shape[1])
    hidden_1 = int(first_w.shape[0])
    hidden_2 = int(second_w.shape[0])
    n_actions = int(third_w.shape[0])
    if hidden_1 != hidden_2:
        raise ValueError(
            f"Unsupported actor architecture: hidden sizes differ ({hidden_1} vs {hidden_2}).",
        )

    actor, _ = build_actor_critic(obs_dim=obs_dim, n_actions=n_actions, hidden_size=hidden_1)
    actor.load_state_dict(state_dict, strict=True)
    return actor


def _load_run_summary(policy_dir: Path) -> dict[str, Any]:
    summary_path = policy_dir / "run_summary.yaml"
    if not summary_path.exists():
        return {}
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
    if not isinstance(summary, dict):
        raise ValueError(f"Expected dict in {summary_path}, got {type(summary)}.")
    return summary


def _infer_default_role(policy_dir: Path, run_settings: dict[str, Any]) -> str:
    if isinstance(run_settings.get("task_role"), str):
        return str(run_settings["task_role"])
    folder_name = policy_dir.name
    if folder_name == "source":
        return "source"
    return "downstream"


def _env_from_run_settings(
    *,
    run_settings: dict[str, Any],
    env_role: str,
) -> dict[str, Any]:
    env_id = str(run_settings.get("env_id") or "LunarLander-v3")
    append_task_id = bool(run_settings.get("append_task_id", True))

    if "task_role" in run_settings:
        gravity_raw = run_settings.get("gravity")
        gravity = None if gravity_raw is None else float(gravity_raw)
        task_id = float(run_settings.get("task_id", 0.0 if env_role == "source" else 1.0))
        dynamics = {
            "enable_wind": bool(run_settings.get("enable_wind", False)),
            "wind_power": run_settings.get("wind_power"),
            "turbulence_power": run_settings.get("turbulence_power"),
            "initial_random_strength": run_settings.get("initial_random_strength"),
            "dispersion_strength": run_settings.get("dispersion_strength"),
            "main_engine_power": run_settings.get("main_engine_power"),
            "side_engine_power": run_settings.get("side_engine_power"),
            "leg_spring_torque": run_settings.get("leg_spring_torque"),
            "lander_mass_scale": run_settings.get("lander_mass_scale"),
            "leg_mass_scale": run_settings.get("leg_mass_scale"),
            "linear_damping": run_settings.get("linear_damping"),
            "angular_damping": run_settings.get("angular_damping"),
            "terrain_heights": run_settings.get("terrain_heights"),
            "action_repeat": int(run_settings.get("action_repeat", 1)),
            "action_delay": int(run_settings.get("action_delay", 0)),
            "action_noise_prob": float(run_settings.get("action_noise_prob", 0.0)),
            "action_noise_mode": str(run_settings.get("action_noise_mode", "noop")),
            "mark_out_of_viewport_as_unsafe": bool(
                run_settings.get("mark_out_of_viewport_as_unsafe", False),
            ),
        }
        return {
            "env_id": env_id,
            "gravity": gravity,
            "task_id": task_id,
            "append_task_id": append_task_id,
            "dynamics": dynamics,
        }

    gravity_key = "source_gravity" if env_role == "source" else "downstream_gravity"
    task_id_key = "source_task_id" if env_role == "source" else "downstream_task_id"
    dynamics_key = "source_dynamics" if env_role == "source" else "downstream_dynamics"
    gravity_raw = run_settings.get(gravity_key)
    gravity = None if gravity_raw is None else float(gravity_raw)
    task_id = float(run_settings.get(task_id_key, 0.0 if env_role == "source" else 1.0))
    raw_dynamics = run_settings.get(dynamics_key, {}) or {}
    if not isinstance(raw_dynamics, dict):
        raise ValueError(f"Expected dict for run_settings['{dynamics_key}'], got {type(raw_dynamics)}.")
    dynamics = _resolve_lunarlander_dynamics(raw_dynamics, cfg_name=f"run_settings[{dynamics_key}]")
    return {
        "env_id": env_id,
        "gravity": gravity,
        "task_id": task_id,
        "append_task_id": append_task_id,
        "dynamics": dynamics,
    }


def _resolve_env_config(
    *,
    policy_dir: Path,
    env_setting: str | None,
    env_role: str | None,
    task_settings_file: Path,
) -> tuple[str, dict[str, Any]]:
    summary = _load_run_summary(policy_dir)
    run_settings = summary.get("run_settings", {}) if isinstance(summary.get("run_settings"), dict) else {}

    if env_setting is not None:
        role = env_role or _infer_default_role(policy_dir, run_settings)
        if role not in {"source", "downstream"}:
            raise ValueError(f"env_role must be source/downstream, got {role}.")
        env_cfg = _load_task_settings(task_settings_file, env_setting, role)
        cfg = {
            "env_id": str(env_cfg.get("env_id") or "LunarLander-v3"),
            "gravity": None if env_cfg.get("gravity") is None else float(env_cfg.get("gravity")),
            "task_id": float(env_cfg.get("task_id", 0.0 if role == "source" else 1.0)),
            "append_task_id": bool(env_cfg.get("append_task_id", True)),
            "dynamics": _resolve_lunarlander_dynamics(env_cfg, cfg_name=f"task_settings[{env_setting}:{role}]"),
        }
        return role, cfg

    inferred_role = env_role or _infer_default_role(policy_dir, run_settings)
    if run_settings:
        return inferred_role, _env_from_run_settings(run_settings=run_settings, env_role=inferred_role)

    defaults = {
        "env_id": "LunarLander-v3",
        "gravity": None,
        "task_id": 0.0 if inferred_role == "source" else 1.0,
        "append_task_id": True,
        "dynamics": _resolve_lunarlander_dynamics({}, cfg_name="defaults"),
    }
    return inferred_role, defaults


def _select_action(
    actor: torch.nn.Module,
    obs: np.ndarray,
    *,
    deterministic: bool,
    device: str,
) -> int:
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
    actor: torch.nn.Module,
    env,
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
                actor,
                np.asarray(obs, dtype=np.float32),
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


def _write_video(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("No frames to write.")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}.")

    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
    else:
        imageio.mimwrite(output_path, frames, fps=fps, macro_block_size=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a trained LunarLander policy, roll it out, and save an RGB video.",
    )
    parser.add_argument(
        "--policy-dir",
        type=Path,
        required=True,
        help="Directory containing actor checkpoint (typically actor.pt) and optionally run_summary.yaml.",
    )
    parser.add_argument(
        "--actor-path",
        type=Path,
        default=None,
        help="Optional explicit actor checkpoint path (absolute or relative to --policy-dir).",
    )
    parser.add_argument(
        "--env-setting",
        type=str,
        default=None,
        help="Optional task-setting name from task_settings.yaml to override run-summary env config.",
    )
    parser.add_argument(
        "--env-role",
        type=str,
        choices=["source", "downstream"],
        default=None,
        help="Environment role when using --env-setting or when overriding inferred role.",
    )
    parser.add_argument(
        "--task-settings-file",
        type=Path,
        default=default_task_settings_file(),
        help="Task settings YAML file used when --env-setting is provided.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Initial rollout seed.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record in one video.")
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=2000,
        help="Safety cap on rollout steps per episode.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic (argmax) actions during rollout.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for policy forward pass.")
    parser.add_argument("--fps", type=int, default=30, help="Output video frames per second.")
    parser.add_argument(
        "--video-name",
        type=str,
        default=None,
        help="Output filename. If omitted, auto-generated in policy dir.",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError(f"--episodes must be > 0, got {args.episodes}.")
    if args.max_steps_per_episode <= 0:
        raise ValueError(f"--max-steps-per-episode must be > 0, got {args.max_steps_per_episode}.")

    policy_dir = args.policy_dir.resolve()
    if not policy_dir.exists():
        raise FileNotFoundError(f"Policy directory does not exist: {policy_dir}")

    actor_path = _resolve_actor_path(policy_dir, args.actor_path)
    actor_state = torch.load(actor_path, map_location="cpu")
    if not isinstance(actor_state, dict):
        raise ValueError(f"Expected state_dict dict at {actor_path}, got {type(actor_state)}.")
    actor = _build_actor_from_state_dict(actor_state).to(args.device)
    actor.eval()

    env_role, env_cfg = _resolve_env_config(
        policy_dir=policy_dir,
        env_setting=args.env_setting,
        env_role=args.env_role,
        task_settings_file=args.task_settings_file,
    )
    env = _make_lunarlander_env(
        str(env_cfg["env_id"]),
        gravity=env_cfg["gravity"],
        task_id=float(env_cfg["task_id"]),
        append_task_id=bool(env_cfg["append_task_id"]),
        enable_wind=bool(env_cfg["dynamics"]["enable_wind"]),
        wind_power=env_cfg["dynamics"]["wind_power"],
        turbulence_power=env_cfg["dynamics"]["turbulence_power"],
        initial_random_strength=env_cfg["dynamics"]["initial_random_strength"],
        dispersion_strength=env_cfg["dynamics"]["dispersion_strength"],
        main_engine_power=env_cfg["dynamics"]["main_engine_power"],
        side_engine_power=env_cfg["dynamics"]["side_engine_power"],
        leg_spring_torque=env_cfg["dynamics"]["leg_spring_torque"],
        lander_mass_scale=env_cfg["dynamics"]["lander_mass_scale"],
        leg_mass_scale=env_cfg["dynamics"]["leg_mass_scale"],
        linear_damping=env_cfg["dynamics"]["linear_damping"],
        angular_damping=env_cfg["dynamics"]["angular_damping"],
        terrain_heights=env_cfg["dynamics"]["terrain_heights"],
        action_repeat=int(env_cfg["dynamics"]["action_repeat"]),
        action_delay=int(env_cfg["dynamics"]["action_delay"]),
        action_noise_prob=float(env_cfg["dynamics"]["action_noise_prob"]),
        action_noise_mode=str(env_cfg["dynamics"]["action_noise_mode"]),
        mark_out_of_viewport_as_unsafe=bool(env_cfg["dynamics"]["mark_out_of_viewport_as_unsafe"]),
        render_mode="rgb_array",
    )

    try:
        frames, returns, lengths = _rollout_frames(
            actor=actor,
            env=env,
            seed=int(args.seed),
            episodes=int(args.episodes),
            max_steps_per_episode=int(args.max_steps_per_episode),
            deterministic=bool(args.deterministic),
            device=str(args.device),
        )
    finally:
        env.close()

    if args.video_name is None:
        video_name = (
            f"policy_rollout_role-{env_role}"
            f"_seed-{int(args.seed)}"
            f"_episodes-{int(args.episodes)}.mp4"
        )
    else:
        video_name = args.video_name
    video_path = policy_dir / video_name
    _write_video(frames, video_path, fps=int(args.fps))

    metadata = {
        "policy_dir": str(policy_dir),
        "actor_path": str(actor_path),
        "env_role": str(env_role),
        "env_config": env_cfg,
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "deterministic": bool(args.deterministic),
        "device": str(args.device),
        "fps": int(args.fps),
        "frame_count": int(len(frames)),
        "episode_returns": [float(v) for v in returns],
        "episode_lengths": [int(v) for v in lengths],
        "video_path": str(video_path),
    }
    metadata_path = policy_dir / f"{video_path.stem}_metadata.yaml"
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")

    print(f"Saved video: {video_path}")
    print(f"Saved rollout metadata: {metadata_path}")
    print(f"Episode returns: {[round(v, 3) for v in returns]}")
    print(f"Episode lengths: {lengths}")


if __name__ == "__main__":
    main()
