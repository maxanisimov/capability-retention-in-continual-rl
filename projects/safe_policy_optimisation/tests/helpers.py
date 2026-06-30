"""Shared mock environments and policies for the stage test suite."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class AlwaysStayPolicy:
    def predict(self, obs, deterministic: bool = True):
        del obs, deterministic
        return np.array(4), None

class AlwaysOnePolicy:
    def predict(self, obs, deterministic: bool = True):
        del obs, deterministic
        return np.array(1), None

class TwoStateEnv(gym.Env):
    observation_space = gym.spaces.Discrete(2)
    action_space = gym.spaces.Discrete(2)

    def __init__(self) -> None:
        super().__init__()
        self.state = 0

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.state = 0
        return self.state, {}

    def step(self, action):
        self.state = 1 - self.state
        return self.state, float(action), False, False, {}

class TwoStepEnv(gym.Env):
    observation_space = gym.spaces.Discrete(2)
    action_space = gym.spaces.Discrete(2)

    def __init__(self) -> None:
        super().__init__()
        self.state = 0
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.state = 0
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        self.step_count += 1
        reward = float(action)
        self.state = 1 - self.state
        return self.state, reward, self.step_count >= 2, False, {}
