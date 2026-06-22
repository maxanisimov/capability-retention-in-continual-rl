"""Environment wrappers for CartPole experiments."""

from __future__ import annotations

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

