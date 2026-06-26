"""Environment wrappers for LunarLander projects.safe_crl."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np


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
