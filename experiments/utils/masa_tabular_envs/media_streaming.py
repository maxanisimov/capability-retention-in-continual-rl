"""Configurable local versions of MASA-Safe-RL MediaStreaming environments."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from experiments.utils.masa_tabular_envs.base import TabularEnv
from experiments.utils.masa_tabular_envs.dynamics import validate_positive_int, validate_probability
from experiments.utils.masa_tabular_envs.renderers.media_streaming import MediaStreamingRenderer

RenderMode = Literal["ansi", "rgb_array", "human"]


class _MediaBase(TabularEnv):
    metadata = {"render_modes": ["ansi", "rgb_array", "human"], "render_fps": 4}

    def _init_rng(self, seed: int | None) -> None:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(np.random.SeedSequence().entropy)

    def render(self):
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()

    @property
    def human_window_closed(self) -> bool:
        return self._renderer.human_window_closed


class CustomMediaStreaming(_MediaBase):
    def __init__(
        self,
        *,
        fast_rate: float = 0.9,
        slow_rate: float = 0.1,
        out_rate: float = 0.7,
        buffer_size: int = 20,
        start_state: int | None = None,
        render_mode: RenderMode | None = None,
        render_window_size: int = 640,
    ) -> None:
        super().__init__()
        self._fast_rate = validate_probability("fast_rate", fast_rate)
        self._slow_rate = validate_probability("slow_rate", slow_rate)
        self._out_rate = validate_probability("out_rate", out_rate)
        self._buffer_size = validate_positive_int("buffer_size", buffer_size)
        self._n_states = self._buffer_size
        self._n_actions = 2
        self._start_state = int(self._buffer_size // 2 if start_state is None else start_state)
        if not 0 <= self._start_state < self._n_states:
            raise ValueError("start_state must be a valid buffer level.")
        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(self._n_actions)
        self._transition_matrix = _media_transition_matrix(
            self._buffer_size,
            self._slow_rate,
            self._fast_rate,
            self._out_rate,
        )
        self.np_random = None
        self._state = None
        self._step_count = 0
        self._last_action = None
        self.render_mode = render_mode
        self.render_window_size = int(render_window_size)
        self._renderer = MediaStreamingRenderer(self)
        self._validate_tabular_spaces()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._init_rng(seed)
        self._state = self._start_state
        self._step_count = 0
        self._last_action = None
        if self.render_mode == "human":
            self.render()
        return self._state, {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise AssertionError(f"Invalid action {action}!")
        self._state = int(self.np_random.choice(self._n_states, p=self._transition_matrix[:, self._state, action]))
        self._last_action = int(action)
        self._step_count += 1
        reward = -1.0 if self._state == 0 else 0.0
        if self.render_mode == "human":
            self.render()
        return self._state, reward, False, False, {}

    def _buffer_level_for_render(self) -> int:
        return int(self._start_state if self._state is None else self._state)

    def label_fn(self, obs: int) -> set[str]:
        labels = set()
        if int(obs) == self._start_state:
            labels.add("start")
        if int(obs) == 0:
            labels.add("empty")
        return labels

    @staticmethod
    def cost_fn(labels: set[str]) -> float:
        return 1.0 if "empty" in labels else 0.0


class CustomMediaStreamingV2(CustomMediaStreaming):
    def __init__(
        self,
        *,
        fast_rate: float = 0.9,
        slow_rate: float = 0.1,
        out_rate: float = 0.7,
        buffer_size: int = 20,
        episode_length: int = 100,
        c_threshold: int | None = None,
        render_mode: RenderMode | None = None,
        render_window_size: int = 640,
    ) -> None:
        _MediaBase.__init__(self)
        self._fast_rate = validate_probability("fast_rate", fast_rate)
        self._slow_rate = validate_probability("slow_rate", slow_rate)
        self._out_rate = validate_probability("out_rate", out_rate)
        self._buffer_size = validate_positive_int("buffer_size", buffer_size)
        self._episode_length = validate_positive_int("episode_length", episode_length)
        self._c_threshold = int(self._episode_length // 2 if c_threshold is None else c_threshold)
        if self._c_threshold < 0:
            raise ValueError("c_threshold must be non-negative.")
        self._fast_count_cap = self._c_threshold + 1
        self._start_buffer = self._buffer_size // 2
        self._n_states = self._buffer_size * (self._fast_count_cap + 1)
        self._n_actions = 2
        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(self._n_actions)
        self._transition_matrix = np.zeros((self._n_states, self._n_states, self._n_actions), dtype=np.float64)
        for state in range(self._n_states):
            buffer_level, fast_count = self._decode_state(state)
            for action in range(self._n_actions):
                in_rate = self._slow_rate if action == 0 else self._fast_rate
                next_fast_count = min(self._fast_count_cap, fast_count + (1 if action == 1 else 0))
                p_up = in_rate * (1.0 - self._out_rate)
                p_down = (1.0 - in_rate) * self._out_rate
                if buffer_level == 0:
                    self._transition_matrix[self._encode_state(buffer_level + 1, next_fast_count), state, action] += p_up
                    self._transition_matrix[self._encode_state(buffer_level, next_fast_count), state, action] += 1.0 - p_up
                elif buffer_level == self._buffer_size - 1:
                    self._transition_matrix[self._encode_state(buffer_level - 1, next_fast_count), state, action] += p_down
                    self._transition_matrix[self._encode_state(buffer_level, next_fast_count), state, action] += 1.0 - p_down
                else:
                    self._transition_matrix[self._encode_state(buffer_level + 1, next_fast_count), state, action] += p_up
                    self._transition_matrix[self._encode_state(buffer_level - 1, next_fast_count), state, action] += p_down
                    self._transition_matrix[self._encode_state(buffer_level, next_fast_count), state, action] += 1.0 - (
                        p_up + p_down
                    )
        self._start_state = self._encode_state(self._start_buffer, 0)
        self.np_random = None
        self._state = None
        self._step_count = 0
        self._last_action = None
        self.render_mode = render_mode
        self.render_window_size = int(render_window_size)
        self._renderer = MediaStreamingRenderer(self)
        self._validate_tabular_spaces()

    def _encode_state(self, buffer_level: int, fast_count: int) -> int:
        return int(fast_count) * self._buffer_size + int(buffer_level)

    def _decode_state(self, state: int) -> tuple[int, int]:
        return int(state) % self._buffer_size, int(state) // self._buffer_size

    def _buffer_level_for_render(self) -> int:
        return self._decode_state(self._start_state if self._state is None else self._state)[0]

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise AssertionError(f"Invalid action {action}!")
        self._state = int(self.np_random.choice(self._n_states, p=self._transition_matrix[:, self._state, action]))
        self._last_action = int(action)
        self._step_count += 1
        buffer_level, _fast_count = self._decode_state(self._state)
        reward = -1.0 if buffer_level == 0 else 0.0
        if self.render_mode == "human":
            self.render()
        return self._state, reward, False, False, {}

    def label_fn(self, obs: int) -> set[str]:
        buffer_level, fast_count = self._decode_state(int(obs))
        labels = set()
        if buffer_level == self._start_buffer and fast_count == 0:
            labels.add("start")
        if buffer_level == 0:
            labels.add("empty")
        if fast_count >= self._fast_count_cap:
            labels.add("unsafe")
        return labels

    @staticmethod
    def cost_fn(labels: set[str]) -> float:
        return 1.0 if "unsafe" in labels else 0.0


class CustomMediaStreamingV3(_MediaBase):
    def __init__(
        self,
        *,
        buffer_size: int = 20,
        episode_length: int = 100,
        danger_threshold: int = 20,
        slow_danger_down_prob: float = 0.5,
        slow_danger_up_prob: float = 0.4,
        fast_danger_up_prob: float = 0.8,
        fast_danger_down_prob: float = 0.1,
        start_buffer: int | None = None,
        start_danger: int = 0,
        start_time: int = 0,
        render_mode: RenderMode | None = None,
        render_window_size: int = 640,
    ) -> None:
        super().__init__()
        self._buffer_size = validate_positive_int("buffer_size", buffer_size)
        self._episode_length = validate_positive_int("episode_length", episode_length)
        self._danger_threshold = int(danger_threshold)
        if self._danger_threshold < 0:
            raise ValueError("danger_threshold must be non-negative.")
        self._danger_cap = self._danger_threshold + 1
        self._danger_states = self._danger_cap + 1
        self._time_states = self._episode_length + 1
        self._slow_danger_down_prob = validate_probability("slow_danger_down_prob", slow_danger_down_prob)
        self._slow_danger_up_prob = validate_probability("slow_danger_up_prob", slow_danger_up_prob)
        self._fast_danger_up_prob = validate_probability("fast_danger_up_prob", fast_danger_up_prob)
        self._fast_danger_down_prob = validate_probability("fast_danger_down_prob", fast_danger_down_prob)
        self._start_buffer = int(self._buffer_size // 2 if start_buffer is None else start_buffer)
        self._start_danger = int(start_danger)
        self._start_time = int(start_time)
        self._n_states = self._danger_states * self._time_states
        self._n_actions = 2
        self.observation_space = spaces.Dict(
            {
                "danger": spaces.Discrete(self._danger_states),
                "buffer": spaces.Discrete(self._buffer_size),
                "time": spaces.Discrete(self._time_states),
            }
        )
        self.action_space = spaces.Discrete(self._n_actions)
        self._successor_states = {}
        self._transition_probs = {}
        for state in range(self._n_states):
            danger_level, time_step = self._decode_safety_state(state)
            probs_by_action = []
            for action in range(self._n_actions):
                if time_step >= self._episode_length:
                    probs_by_action.append({state: 1.0})
                    continue
                probs = {}
                for next_danger, prob in self._danger_transitions(danger_level, action):
                    next_state = self._encode_safety_state(next_danger, time_step + 1)
                    probs[next_state] = probs.get(next_state, 0.0) + prob
                probs_by_action.append(probs)
            successors = sorted(set().union(*[set(probs) for probs in probs_by_action]))
            self._successor_states[state] = successors
            for action, probs in enumerate(probs_by_action):
                self._transition_probs[(state, action)] = np.array(
                    [probs.get(successor, 0.0) for successor in successors],
                    dtype=np.float32,
                )
        self._start_state = self._encode_safety_state(self._start_danger, self._start_time)
        self.np_random = None
        self._state = None
        self._buffer_level = self._start_buffer
        self._step_count = 0
        self._last_action = None
        self.render_mode = render_mode
        self.render_window_size = int(render_window_size)
        self._renderer = MediaStreamingRenderer(self)
        self._validate_tabular_spaces()

    def _encode_safety_state(self, danger_level: int, time_step: int) -> int:
        return int(time_step) * self._danger_states + int(danger_level)

    def _decode_safety_state(self, state: int) -> tuple[int, int]:
        return int(state) % self._danger_states, int(state) // self._danger_states

    def _danger_transitions(self, danger_level: int, action: int) -> list[tuple[int, float]]:
        if action == 0:
            down_prob = self._slow_danger_down_prob
            up_prob = self._slow_danger_up_prob
        else:
            down_prob = self._fast_danger_down_prob
            up_prob = self._fast_danger_up_prob
        stay_prob = max(0.0, 1.0 - down_prob - up_prob)
        down = max(0, danger_level - 1)
        up = min(self._danger_cap, danger_level + 1)
        return [(down, down_prob), (danger_level, stay_prob), (up, up_prob)]

    def safety_abstraction(self, obs: Any) -> int:
        if np.isscalar(obs):
            return int(obs)
        if isinstance(obs, dict):
            return self._encode_safety_state(int(obs["danger"]), int(obs["time"]))
        arr = np.asarray(obs)
        if arr.ndim == 0:
            return int(arr)
        return self._encode_safety_state(int(round(float(arr[0]))), int(round(float(arr[2] if arr.shape[0] > 2 else arr[1]))))

    def _obs(self) -> dict[str, int]:
        danger, time = self._decode_safety_state(self._state)
        return {"danger": danger, "buffer": int(self._buffer_level), "time": time}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._init_rng(seed)
        self._state = self._start_state
        self._buffer_level = self._start_buffer
        self._step_count = 0
        self._last_action = None
        if self.render_mode == "human":
            self.render()
        return self._obs(), {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise AssertionError(f"Invalid action {action}!")
        successors = self._successor_states[self._state]
        probs = self._transition_probs[(self._state, int(action))]
        self._state = int(self.np_random.choice(successors, p=probs))
        self._last_action = int(action)
        self._buffer_level = int(np.clip(self._buffer_level + (1 if action == 1 else -1), 0, self._buffer_size - 1))
        self._step_count += 1
        obs = self._obs()
        reward = -1.0 if obs["buffer"] == 0 else 0.0
        terminated = bool(obs["time"] >= self._episode_length)
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, {}

    def _buffer_level_for_render(self) -> int:
        return int(self._buffer_level)

    def label_fn(self, obs: Any) -> set[str]:
        safety_state = self.safety_abstraction(obs)
        danger, time = self._decode_safety_state(safety_state)
        buffer_level = int(obs["buffer"]) if isinstance(obs, dict) else None
        labels = set()
        if danger == self._start_danger and time == self._start_time:
            labels.add("start")
        if buffer_level == 0:
            labels.add("empty")
        if danger >= self._danger_cap:
            labels.add("unsafe")
        return labels

    @staticmethod
    def cost_fn(labels: set[str]) -> float:
        return 1.0 if "unsafe" in labels else 0.0


def _media_transition_matrix(buffer_size: int, slow_rate: float, fast_rate: float, out_rate: float) -> np.ndarray:
    matrix = np.zeros((buffer_size, buffer_size, 2), dtype=np.float64)
    for state in range(buffer_size):
        for action in range(2):
            in_rate = slow_rate if action == 0 else fast_rate
            if state == 0:
                matrix[state + 1, state, action] = in_rate * (1.0 - out_rate)
                matrix[state, state, action] = 1.0 - in_rate * (1.0 - out_rate)
            elif state == buffer_size - 1:
                edge_in_rate = slow_rate if action == 0 else 1.0
                matrix[state - 1, state, action] = (1.0 - edge_in_rate) * out_rate
                matrix[state, state, action] = 1.0 - (1.0 - edge_in_rate) * out_rate
            else:
                matrix[state + 1, state, action] = in_rate * (1.0 - out_rate)
                matrix[state - 1, state, action] = (1.0 - in_rate) * out_rate
                matrix[state, state, action] = 1.0 - (
                    in_rate * (1.0 - out_rate) + (1.0 - in_rate) * out_rate
                )
    return matrix
