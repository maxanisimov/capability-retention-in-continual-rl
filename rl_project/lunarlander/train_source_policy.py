"""Train a source-only PPO policy on LunarLander with discrete actions.

This script mirrors the FrozenLake source-training flow while targeting
Gymnasium LunarLander in its discrete-action setting (`continuous=False`).
Actor and critic hidden layers use ReLU activations.
"""

from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from rl_project.utils.gymnasium_utils import plot_multi_episode_frames
from rl_project.utils.ppo_utils import PPOConfig, evaluate, ppo_train


class AppendTaskIDObservationWrapper(gym.ObservationWrapper):
    """Append a constant task-id feature to 1D vector observations."""

    def __init__(self, env: gym.Env, task_id: int | float):
        super().__init__(env)
        self.task_id = float(task_id)

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("AppendTaskIDObservationWrapper requires Box observation space.")
        if len(env.observation_space.shape) != 1:
            raise ValueError("AppendTaskIDObservationWrapper supports only 1D observations.")

        low = np.asarray(env.observation_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(env.observation_space.high, dtype=np.float32).reshape(-1)
        task_arr = np.asarray([self.task_id], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low, task_arr], axis=0),
            high=np.concatenate([high, task_arr], axis=0),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        return np.concatenate([obs, np.asarray([self.task_id], dtype=np.float32)], axis=0)


class ActionRepeatWrapper(gym.Wrapper):
    """Repeat each action for a fixed number of environment steps."""

    def __init__(self, env: gym.Env, repeat: int):
        super().__init__(env)
        self.repeat = int(repeat)
        if self.repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {self.repeat}.")

    def step(self, action):  # type: ignore[override]
        total_reward = 0.0
        terminated = False
        truncated = False
        obs = None
        info: dict[str, Any] = {}
        executed_steps = 0

        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            executed_steps += 1
            if terminated or truncated:
                break

        if obs is None:
            raise RuntimeError("ActionRepeatWrapper.step executed zero internal steps.")
        info = dict(info)
        info["action_repeat_executed_steps"] = int(executed_steps)
        return obs, total_reward, terminated, truncated, info


class ActionDelayWrapper(gym.Wrapper):
    """Apply an action after a fixed delay (in steps)."""

    def __init__(self, env: gym.Env, delay_steps: int, noop_action: int = 0):
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("ActionDelayWrapper requires a Discrete action space.")
        self.delay_steps = int(delay_steps)
        if self.delay_steps < 0:
            raise ValueError(f"delay_steps must be >= 0, got {self.delay_steps}.")

        self.noop_action = int(noop_action)
        n_actions = int(env.action_space.n)
        if self.noop_action < 0 or self.noop_action >= n_actions:
            raise ValueError(
                f"noop_action must be in [0, {n_actions - 1}], got {self.noop_action}.",
            )
        self._queue: deque[int] = deque([], maxlen=self.delay_steps + 1)
        self._reset_queue()

    def _reset_queue(self) -> None:
        self._queue.clear()
        for _ in range(self.delay_steps + 1):
            self._queue.append(self.noop_action)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):  # type: ignore[override]
        self._reset_queue()
        return self.env.reset(seed=seed, options=options)

    def step(self, action):  # type: ignore[override]
        self._queue.append(int(action))
        applied_action = self._queue.popleft()
        obs, reward, terminated, truncated, info = self.env.step(applied_action)
        info = dict(info)
        info["applied_action"] = int(applied_action)
        return obs, reward, terminated, truncated, info


class ActionNoiseWrapper(gym.Wrapper):
    """Inject action noise into a discrete-action policy."""

    def __init__(
        self,
        env: gym.Env,
        noise_prob: float,
        mode: str = "noop",
        noop_action: int = 0,
    ):
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("ActionNoiseWrapper requires a Discrete action space.")
        self.noise_prob = float(noise_prob)
        if self.noise_prob < 0.0 or self.noise_prob > 1.0:
            raise ValueError(f"noise_prob must be in [0, 1], got {self.noise_prob}.")

        valid_modes = {"noop", "previous", "random"}
        self.mode = str(mode)
        if self.mode not in valid_modes:
            raise ValueError(
                f"Unsupported action-noise mode '{self.mode}'. Expected one of {sorted(valid_modes)}.",
            )

        self.noop_action = int(noop_action)
        n_actions = int(env.action_space.n)
        if self.noop_action < 0 or self.noop_action >= n_actions:
            raise ValueError(
                f"noop_action must be in [0, {n_actions - 1}], got {self.noop_action}.",
            )
        self._rng = np.random.default_rng()
        self._prev_action = self.noop_action

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):  # type: ignore[override]
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._prev_action = self.noop_action
        return self.env.reset(seed=seed, options=options)

    def step(self, action):  # type: ignore[override]
        selected_action = int(action)
        if self._rng.random() < self.noise_prob:
            if self.mode == "noop":
                selected_action = self.noop_action
            elif self.mode == "previous":
                selected_action = int(self._prev_action)
            else:  # mode == "random"
                selected_action = int(self._rng.integers(low=0, high=int(self.action_space.n)))

        obs, reward, terminated, truncated, info = self.env.step(selected_action)
        info = dict(info)
        info["action_noise_applied"] = bool(selected_action != int(action))
        info["applied_action"] = int(selected_action)
        self._prev_action = int(selected_action)
        return obs, reward, terminated, truncated, info


class LunarLanderCrashSafetyWrapper(gym.Wrapper):
    """Attach a per-step safety flag for LunarLander termination causes."""

    def __init__(self, env: gym.Env, *, mark_out_of_viewport_as_unsafe: bool = False):
        super().__init__(env)
        self.mark_out_of_viewport_as_unsafe = bool(mark_out_of_viewport_as_unsafe)

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        # In Gymnasium LunarLander, `game_over=True` is set when the lander body
        # (not the legs) contacts the moon surface.
        crashed = bool(getattr(self.unwrapped, "game_over", False))
        out_of_viewport = False
        if isinstance(obs, np.ndarray) and obs.size > 0:
            out_of_viewport = abs(float(obs[0])) >= 1.0

        unsafe = crashed or (self.mark_out_of_viewport_as_unsafe and out_of_viewport)
        info["safe"] = (not unsafe)
        return obs, reward, terminated, truncated, info


def _resolve_lunarlander_dynamics(cfg: dict[str, Any], *, cfg_name: str = "config") -> dict[str, Any]:
    """Normalize and validate dynamics-shift parameters for LunarLander."""
    enable_wind_raw = cfg.get("enable_wind", False)
    enable_wind = False if enable_wind_raw is None else bool(enable_wind_raw)
    wind_power_raw = cfg.get("wind_power", None)
    turbulence_power_raw = cfg.get("turbulence_power", None)
    wind_power = None if wind_power_raw is None else float(wind_power_raw)
    turbulence_power = None if turbulence_power_raw is None else float(turbulence_power_raw)
    initial_random_strength_raw = cfg.get("initial_random_strength", None)
    initial_random_strength = (
        None if initial_random_strength_raw is None else float(initial_random_strength_raw)
    )
    if initial_random_strength is not None and initial_random_strength < 0.0:
        raise ValueError(
            f"{cfg_name}: initial_random_strength must be >= 0, got {initial_random_strength}.",
        )
    dispersion_strength_raw = cfg.get("dispersion_strength", None)
    dispersion_strength = None if dispersion_strength_raw is None else float(dispersion_strength_raw)
    if dispersion_strength is not None and dispersion_strength < 0.0:
        raise ValueError(
            f"{cfg_name}: dispersion_strength must be >= 0, got {dispersion_strength}.",
        )
    main_engine_power_raw = cfg.get("main_engine_power", None)
    main_engine_power = None if main_engine_power_raw is None else float(main_engine_power_raw)
    if main_engine_power is not None and main_engine_power < 0.0:
        raise ValueError(
            f"{cfg_name}: main_engine_power must be >= 0, got {main_engine_power}.",
        )
    side_engine_power_raw = cfg.get("side_engine_power", None)
    side_engine_power = None if side_engine_power_raw is None else float(side_engine_power_raw)
    if side_engine_power is not None and side_engine_power < 0.0:
        raise ValueError(
            f"{cfg_name}: side_engine_power must be >= 0, got {side_engine_power}.",
        )
    leg_spring_torque_raw = cfg.get("leg_spring_torque", None)
    leg_spring_torque = None if leg_spring_torque_raw is None else float(leg_spring_torque_raw)
    if leg_spring_torque is not None and leg_spring_torque < 0.0:
        raise ValueError(
            f"{cfg_name}: leg_spring_torque must be >= 0, got {leg_spring_torque}.",
        )
    lander_mass_scale_raw = cfg.get("lander_mass_scale", None)
    lander_mass_scale = None if lander_mass_scale_raw is None else float(lander_mass_scale_raw)
    if lander_mass_scale is not None and lander_mass_scale <= 0.0:
        raise ValueError(
            f"{cfg_name}: lander_mass_scale must be > 0, got {lander_mass_scale}.",
        )
    leg_mass_scale_raw = cfg.get("leg_mass_scale", None)
    leg_mass_scale = None if leg_mass_scale_raw is None else float(leg_mass_scale_raw)
    if leg_mass_scale is not None and leg_mass_scale <= 0.0:
        raise ValueError(
            f"{cfg_name}: leg_mass_scale must be > 0, got {leg_mass_scale}.",
        )
    linear_damping_raw = cfg.get("linear_damping", None)
    linear_damping = None if linear_damping_raw is None else float(linear_damping_raw)
    if linear_damping is not None and linear_damping < 0.0:
        raise ValueError(
            f"{cfg_name}: linear_damping must be >= 0, got {linear_damping}.",
        )
    angular_damping_raw = cfg.get("angular_damping", None)
    angular_damping = None if angular_damping_raw is None else float(angular_damping_raw)
    if angular_damping is not None and angular_damping < 0.0:
        raise ValueError(
            f"{cfg_name}: angular_damping must be >= 0, got {angular_damping}.",
        )

    action_repeat_raw = cfg.get("action_repeat", 1)
    action_repeat = 1 if action_repeat_raw is None else int(action_repeat_raw)
    if action_repeat < 0:
        raise ValueError(f"{cfg_name}: action_repeat must be >= 0, got {action_repeat}.")

    action_delay_raw = cfg.get("action_delay", 0)
    action_delay = 0 if action_delay_raw is None else int(action_delay_raw)
    if action_delay < 0:
        raise ValueError(f"{cfg_name}: action_delay must be >= 0, got {action_delay}.")

    action_noise_prob_raw = cfg.get("action_noise_prob", 0.0)
    action_noise_prob = 0.0 if action_noise_prob_raw is None else float(action_noise_prob_raw)
    if action_noise_prob < 0.0 or action_noise_prob > 1.0:
        raise ValueError(
            f"{cfg_name}: action_noise_prob must be in [0, 1], got {action_noise_prob}.",
        )

    action_noise_mode_raw = cfg.get("action_noise_mode", "noop")
    action_noise_mode = "noop" if action_noise_mode_raw is None else str(action_noise_mode_raw)
    if action_noise_mode not in {"noop", "previous", "random"}:
        raise ValueError(
            f"{cfg_name}: unsupported action_noise_mode '{action_noise_mode}'. "
            "Expected one of ['noop', 'previous', 'random'].",
        )
    mark_out_of_viewport_as_unsafe_raw = cfg.get("mark_out_of_viewport_as_unsafe", False)
    mark_out_of_viewport_as_unsafe = (
        False
        if mark_out_of_viewport_as_unsafe_raw is None
        else bool(mark_out_of_viewport_as_unsafe_raw)
    )

    return {
        "enable_wind": enable_wind,
        "wind_power": wind_power,
        "turbulence_power": turbulence_power,
        "initial_random_strength": initial_random_strength,
        "dispersion_strength": dispersion_strength,
        "main_engine_power": main_engine_power,
        "side_engine_power": side_engine_power,
        "leg_spring_torque": leg_spring_torque,
        "lander_mass_scale": lander_mass_scale,
        "leg_mass_scale": leg_mass_scale,
        "linear_damping": linear_damping,
        "angular_damping": angular_damping,
        "action_repeat": action_repeat,
        "action_delay": action_delay,
        "action_noise_prob": action_noise_prob,
        "action_noise_mode": action_noise_mode,
        "mark_out_of_viewport_as_unsafe": mark_out_of_viewport_as_unsafe,
    }


def _load_task_settings(
    settings_file: Path,
    setting_name: str,
    task_role: str,
) -> dict[str, Any]:
    if task_role not in ("source", "downstream"):
        raise ValueError(f"task_role must be 'source' or 'downstream', got '{task_role}'.")

    if not settings_file.exists():
        return {}
    all_settings = yaml.safe_load(settings_file.read_text(encoding="utf-8")) or {}
    resolved_setting_name = setting_name
    # Backward-compatible alias for earlier config naming.
    if setting_name == "default_to_low_gravity" and "default" in all_settings:
        resolved_setting_name = "default"

    if resolved_setting_name not in all_settings:
        raise ValueError(f"Task setting '{setting_name}' not found in {settings_file}.")

    setting_cfg = all_settings[resolved_setting_name] or {}
    role_cfg = setting_cfg.get(task_role, {}) or {}
    return {
        "env_id": setting_cfg.get("env_id"),
        "gravity": role_cfg.get("gravity"),
        "task_id": role_cfg.get("task_id"),
        "enable_wind": role_cfg.get("enable_wind"),
        "wind_power": role_cfg.get("wind_power"),
        "turbulence_power": role_cfg.get("turbulence_power"),
        "initial_random_strength": role_cfg.get("initial_random_strength"),
        "dispersion_strength": role_cfg.get("dispersion_strength"),
        "main_engine_power": role_cfg.get("main_engine_power"),
        "side_engine_power": role_cfg.get("side_engine_power"),
        "leg_spring_torque": role_cfg.get("leg_spring_torque"),
        "lander_mass_scale": role_cfg.get("lander_mass_scale"),
        "leg_mass_scale": role_cfg.get("leg_mass_scale"),
        "linear_damping": role_cfg.get("linear_damping"),
        "angular_damping": role_cfg.get("angular_damping"),
        "action_repeat": role_cfg.get("action_repeat"),
        "action_delay": role_cfg.get("action_delay"),
        "action_noise_prob": role_cfg.get("action_noise_prob"),
        "action_noise_mode": role_cfg.get("action_noise_mode"),
        "mark_out_of_viewport_as_unsafe": role_cfg.get("mark_out_of_viewport_as_unsafe"),
        "append_task_id": setting_cfg.get("append_task_id"),
        "continuous": setting_cfg.get("continuous"),
    }


def _make_lunarlander_env(
    env_id: str,
    *,
    gravity: float | None = None,
    task_id: float = 0.0,
    append_task_id: bool = True,
    enable_wind: bool | None = None,
    wind_power: float | None = None,
    turbulence_power: float | None = None,
    initial_random_strength: float | None = None,
    dispersion_strength: float | None = None,
    main_engine_power: float | None = None,
    side_engine_power: float | None = None,
    leg_spring_torque: float | None = None,
    lander_mass_scale: float | None = None,
    leg_mass_scale: float | None = None,
    linear_damping: float | None = None,
    angular_damping: float | None = None,
    action_repeat: int = 1,
    action_delay: int = 0,
    action_noise_prob: float = 0.0,
    action_noise_mode: str = "noop",
    mark_out_of_viewport_as_unsafe: bool = False,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a discrete-action LunarLander env with optional dynamics/task-id shifts."""
    from rl_project.lunarlander.tunable_lunarlander import ensure_lunarlander_v4_registered

    ensure_lunarlander_v4_registered()

    make_kwargs: dict[str, Any] = {"continuous": False, "render_mode": render_mode}
    if gravity is not None:
        make_kwargs["gravity"] = float(gravity)
    if enable_wind is not None:
        make_kwargs["enable_wind"] = bool(enable_wind)
    if wind_power is not None:
        make_kwargs["wind_power"] = float(wind_power)
    if turbulence_power is not None:
        make_kwargs["turbulence_power"] = float(turbulence_power)
    if initial_random_strength is not None:
        make_kwargs["initial_random_strength"] = float(initial_random_strength)
    if dispersion_strength is not None:
        make_kwargs["dispersion_strength"] = float(dispersion_strength)
    if main_engine_power is not None:
        make_kwargs["main_engine_power"] = float(main_engine_power)
    if side_engine_power is not None:
        make_kwargs["side_engine_power"] = float(side_engine_power)
    if leg_spring_torque is not None:
        make_kwargs["leg_spring_torque"] = float(leg_spring_torque)
    if lander_mass_scale is not None:
        make_kwargs["lander_mass_scale"] = float(lander_mass_scale)
    if leg_mass_scale is not None:
        make_kwargs["leg_mass_scale"] = float(leg_mass_scale)
    if linear_damping is not None:
        make_kwargs["linear_damping"] = float(linear_damping)
    if angular_damping is not None:
        make_kwargs["angular_damping"] = float(angular_damping)

    def _retry_without_unsupported_kwargs(
        err: TypeError,
        kwargs: dict[str, Any],
        *,
        env_label: str,
    ) -> dict[str, Any] | None:
        err_text = str(err)
        unsupported_groups = {
            "wind/turbulence parameters": {"enable_wind", "wind_power", "turbulence_power"},
            "stochasticity parameters": {"initial_random_strength", "dispersion_strength"},
            "extended dynamics parameters": {
                "main_engine_power",
                "side_engine_power",
                "leg_spring_torque",
                "lander_mass_scale",
                "leg_mass_scale",
                "linear_damping",
                "angular_damping",
            },
        }
        drop_keys: set[str] = set()
        for group_name, group_keys in unsupported_groups.items():
            if any(key in err_text for key in group_keys):
                drop_keys.update(group_keys)
                print(
                    f"{env_label} does not support {group_name} in this Gymnasium build; "
                    "retrying without them.",
                )
        if not drop_keys:
            return None
        return {k: v for k, v in kwargs.items() if k not in drop_keys}

    try:
        env = gym.make(env_id, **make_kwargs)
    except TypeError as err:
        fallback_kwargs = _retry_without_unsupported_kwargs(err, make_kwargs, env_label=env_id)
        if fallback_kwargs is not None:
            env = gym.make(env_id, **fallback_kwargs)
        else:
            raise
    except gym.error.Error as err:
        if env_id == "LunarLander-v3":
            print("Could not create LunarLander-v3, retrying with LunarLander-v2 ...")
            try:
                env = gym.make("LunarLander-v2", **make_kwargs)
            except TypeError as type_err:
                fallback_kwargs = _retry_without_unsupported_kwargs(
                    type_err,
                    make_kwargs,
                    env_label="LunarLander-v2",
                )
                if fallback_kwargs is not None:
                    env = gym.make("LunarLander-v2", **fallback_kwargs)
                else:
                    raise
        else:
            raise RuntimeError(
                f"Failed to create env '{env_id}'. Ensure Box2D deps are installed "
                "(e.g., `pip install gymnasium[box2d]`)."
            ) from err
    env = LunarLanderCrashSafetyWrapper(
        env,
        mark_out_of_viewport_as_unsafe=mark_out_of_viewport_as_unsafe,
    )
    # Treat action_repeat <= 1 as "no repeat wrapper" for backward compatibility.
    if action_repeat > 1:
        env = ActionRepeatWrapper(env, repeat=action_repeat)
    if action_delay != 0:
        env = ActionDelayWrapper(env, delay_steps=action_delay, noop_action=0)
    if action_noise_prob > 0.0:
        env = ActionNoiseWrapper(
            env,
            noise_prob=action_noise_prob,
            mode=action_noise_mode,
            noop_action=0,
        )
    if append_task_id:
        env = AppendTaskIDObservationWrapper(env, task_id=task_id)
    return env


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
        default=Path(__file__).resolve().parent / "settings" / "task_settings.yaml",
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
        default=Path(__file__).resolve().parent / "outputs",
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

    run_dir = args.output_dir / args.task_setting / f"seed_{args.seed}"
    task_dir = run_dir / args.task_role
    task_dir.mkdir(parents=True, exist_ok=True)

    actor_path = task_dir / "actor.pt"
    critic_path = task_dir / "critic.pt"
    training_data_path = task_dir / "training_data.pt"
    summary_path = task_dir / "run_summary.yaml"

    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    torch.save(training_data, training_data_path)

    trajectory_plot_path = task_dir / "trajectory_episodes_grid.png"
    _plot_trajectory_grid(
        env_id=env_id,
        gravity=gravity_value,
        task_id=task_id,
        append_task_id=append_task_id,
        dynamics_cfg=dynamics_cfg,
        actor=actor,
        seed=args.seed,
        device=args.device,
        output_path=trajectory_plot_path,
        episodes=int(args.trajectory_episodes),
        max_frames_per_episode=int(args.trajectory_max_frames_per_episode),
    )

    downstream_trajectory_plot_path: Path | None = None
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

        downstream_trajectory_plot_path = task_dir / "trajectory_episodes_grid_downstream_eval.png"
        _plot_trajectory_grid(
            env_id=downstream_eval_env_id,
            gravity=downstream_eval_gravity,
            task_id=downstream_eval_task_id,
            append_task_id=downstream_eval_append_task_id,
            dynamics_cfg=downstream_eval_dynamics,
            actor=actor,
            seed=args.seed,
            device=args.device,
            output_path=downstream_trajectory_plot_path,
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
        "trajectory_plot_path": str(trajectory_plot_path),
        "downstream_eval_trajectory_plot_path": (
            str(downstream_trajectory_plot_path)
            if downstream_trajectory_plot_path is not None
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
    print(f"Saved trajectory plot: {trajectory_plot_path}")
    if downstream_eval_performed and downstream_trajectory_plot_path is not None:
        print(f"Saved downstream evaluation trajectory plot: {downstream_trajectory_plot_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
