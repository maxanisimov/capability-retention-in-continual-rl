"""Small local equivalent of MASA's tabular environment base class."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces


class TabularEnv(gym.Env):
    """Gymnasium environment with optional tabular dynamics accessors."""

    def __init__(self) -> None:
        super().__init__()
        self._transition_matrix = None
        self._successor_states = None
        self._transition_probs = None

    def _validate_tabular_spaces(self) -> None:
        if not isinstance(self.observation_space, spaces.Discrete | spaces.Dict):
            raise TypeError("Tabular env observations must be Discrete or Dict.")
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("Tabular env actions must be Discrete.")

    @property
    def has_transition_matrix(self) -> bool:
        return self._transition_matrix is not None

    @property
    def has_successor_states_dict(self) -> bool:
        return self._successor_states is not None and self._transition_probs is not None

    def get_transition_matrix(self):
        return self._transition_matrix

    def get_successor_states_dict(self):
        if not self.has_successor_states_dict:
            return None
        return self._successor_states, self._transition_probs
