"""Configurable MASA-style tabular FrozenLake environment."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from gymnasium import spaces
from gymnasium.envs.toy_text.frozen_lake import MAPS, generate_random_map

from projects.safe_crl.utils.masa_tabular_envs.base import TabularEnv
from projects.safe_crl.utils.masa_tabular_envs.dynamics import validate_probability
from projects.safe_crl.utils.masa_tabular_envs.renderers.frozen_lake import (
    FrozenLakeRenderer,
    validate_renderer_options,
)

RenderMode = Literal["ansi", "rgb_array", "human"]

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class CustomFrozenLake(TabularEnv):
    """Local FrozenLake with Gymnasium dynamics and MASA tabular accessors."""

    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        *,
        render_mode: RenderMode | None = None,
        desc: list[str] | None = None,
        map_name: str | None = "4x4",
        is_slippery: bool = True,
        success_rate: float = 1.0 / 3.0,
        reward_schedule: tuple[float, float, float] = (1, 0, 0),
        render_window_size: int | None = None,
    ) -> None:
        super().__init__()
        validate_renderer_options(render_mode, render_window_size)
        success_rate = validate_probability("success_rate", success_rate)
        reward_schedule = _validate_reward_schedule(reward_schedule)

        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            if map_name not in MAPS:
                raise ValueError(f"Unknown FrozenLake map_name {map_name!r}; expected one of {sorted(MAPS)}.")
            desc = MAPS[str(map_name)]
        desc_array = _validate_desc(desc)

        self.desc = desc_array
        self.nrow, self.ncol = self.desc.shape
        self.reward_range = (min(reward_schedule), max(reward_schedule))
        self._is_slippery = bool(is_slippery)
        self._success_rate = float(success_rate)
        self._reward_schedule = reward_schedule
        self._n_states = int(self.nrow * self.ncol)
        self._n_actions = 4

        self.initial_state_distrib = np.array(self.desc == b"S").astype("float64").ravel()
        if self.initial_state_distrib.sum() <= 0.0:
            raise ValueError("FrozenLake desc must contain at least one start cell 'S'.")
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self._start_state = int(np.flatnonzero(self.initial_state_distrib)[0])

        self.P = _build_transition_dict(
            self.desc,
            is_slippery=self._is_slippery,
            success_rate=self._success_rate,
            reward_schedule=self._reward_schedule,
        )
        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(self._n_actions)
        self._transition_matrix = _transition_matrix_from_p(self.P, self._n_states, self._n_actions)
        self._successor_states, self._transition_probs = _successor_dict_from_p(
            self.P,
            self._n_states,
            self._n_actions,
        )

        self.render_mode = render_mode
        if render_window_size is None:
            self.window_size = (min(64 * self.ncol, 512), min(64 * self.nrow, 512))
        else:
            self.window_size = (int(render_window_size), int(render_window_size))
        self.render_window_size = int(render_window_size) if render_window_size is not None else None
        self.cell_size = (self.window_size[0] // self.ncol, self.window_size[1] // self.nrow)
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None
        self.s = 0
        self.lastaction = None
        self._renderer = FrozenLakeRenderer(self)
        self._validate_tabular_spaces()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.s = int(_categorical_sample(self.initial_state_distrib, self.np_random))
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise AssertionError(f"Invalid action {action}!")
        transitions = self.P[int(self.s)][int(action)]
        idx = int(_categorical_sample([transition[0] for transition in transitions], self.np_random))
        prob, state, reward, terminated = transitions[idx]
        self.s = int(state)
        self.lastaction = int(action)

        if self.render_mode == "human":
            self.render()
        return int(state), reward, bool(terminated), False, {"prob": prob}

    def render(self):
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()

    @property
    def human_window_closed(self) -> bool:
        return self._renderer.human_window_closed

    def handle_pygame_event(self, event: Any) -> bool:
        return self._renderer.handle_pygame_event(event)

    def label_fn(self, obs: int) -> set[str]:
        row, col = divmod(int(obs), self.ncol)
        cell = self.desc[row, col]
        cell_str = cell.decode("utf-8") if isinstance(cell, bytes | np.bytes_) else str(cell)
        if cell_str == "S":
            return {"start"}
        if cell_str == "G":
            return {"goal"}
        if cell_str == "H":
            return {"hole"}
        return {"frozen"}

    @staticmethod
    def cost_fn(labels: set[str]) -> float:
        return 1.0 if "hole" in labels else 0.0


def _validate_desc(desc: list[str] | tuple[str, ...] | np.ndarray | None) -> np.ndarray:
    if desc is None:
        raise ValueError("desc must not be None after map resolution.")
    if isinstance(desc, np.ndarray):
        if desc.ndim != 2:
            raise ValueError("FrozenLake desc array must be 2D.")
        rows = []
        for row in desc.tolist():
            rows.append(
                "".join(
                    cell.decode("utf-8") if isinstance(cell, bytes | np.bytes_) else str(cell)
                    for cell in row
                ),
            )
    else:
        rows = [row.decode("utf-8") if isinstance(row, bytes) else str(row) for row in list(desc)]
    if not rows:
        raise ValueError("FrozenLake desc must be non-empty.")
    ncol = len(rows[0])
    if ncol == 0 or any(len(row) != ncol for row in rows):
        raise ValueError("FrozenLake desc must be rectangular and non-empty.")
    allowed = {"S", "F", "H", "G"}
    bad = sorted({cell for row in rows for cell in row if cell not in allowed})
    if bad:
        raise ValueError(f"FrozenLake desc contains invalid cells: {bad}.")
    return np.asarray(rows, dtype="c")


def _validate_reward_schedule(reward_schedule: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(reward_schedule) != 3:
        raise ValueError("reward_schedule must contain exactly three values for G, H, and F.")
    return tuple(float(reward) for reward in reward_schedule)


def _build_transition_dict(
    desc: np.ndarray,
    *,
    is_slippery: bool,
    success_rate: float,
    reward_schedule: tuple[float, float, float],
) -> dict[int, dict[int, list[tuple[float, int, float, bool]]]]:
    nrow, ncol = desc.shape
    n_states = nrow * ncol
    n_actions = 4
    transitions: dict[int, dict[int, list[tuple[float, int, float, bool]]]] = {
        state: {action: [] for action in range(n_actions)} for state in range(n_states)
    }
    fail_rate = (1.0 - success_rate) / 2.0

    def to_s(row: int, col: int) -> int:
        return row * ncol + col

    def inc(row: int, col: int, action: int) -> tuple[int, int]:
        if action == LEFT:
            col = max(col - 1, 0)
        elif action == DOWN:
            row = min(row + 1, nrow - 1)
        elif action == RIGHT:
            col = min(col + 1, ncol - 1)
        elif action == UP:
            row = max(row - 1, 0)
        return row, col

    def update_probability_matrix(row: int, col: int, action: int) -> tuple[int, float, bool]:
        new_row, new_col = inc(row, col, action)
        new_state = to_s(new_row, new_col)
        new_letter = desc[new_row, new_col]
        terminated = bytes(new_letter) in b"GH"
        reward = reward_schedule[b"GHF".index(new_letter if new_letter in b"GHF" else b"F")]
        return new_state, reward, terminated

    for row in range(nrow):
        for col in range(ncol):
            state = to_s(row, col)
            for action in range(n_actions):
                state_transitions = transitions[state][action]
                letter = desc[row, col]
                if letter in b"GH":
                    state_transitions.append((1.0, state, 0.0, True))
                elif is_slippery:
                    for actual_action in [(action - 1) % 4, action, (action + 1) % 4]:
                        prob = success_rate if actual_action == action else fail_rate
                        state_transitions.append((prob, *update_probability_matrix(row, col, actual_action)))
                else:
                    state_transitions.append((1.0, *update_probability_matrix(row, col, action)))
    return transitions


def _transition_matrix_from_p(
    transitions: dict[int, dict[int, list[tuple[float, int, float, bool]]]],
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    matrix = np.zeros((n_states, n_states, n_actions), dtype=np.float64)
    for state in range(n_states):
        for action in range(n_actions):
            for prob, next_state, _reward, _terminated in transitions[state][action]:
                matrix[int(next_state), state, action] += float(prob)
    return matrix


def _successor_dict_from_p(
    transitions: dict[int, dict[int, list[tuple[float, int, float, bool]]]],
    n_states: int,
    n_actions: int,
) -> tuple[dict[int, list[int]], dict[tuple[int, int], np.ndarray]]:
    successor_states: dict[int, list[int]] = {}
    transition_probs: dict[tuple[int, int], np.ndarray] = {}
    for state in range(n_states):
        successors = sorted(
            {
                int(next_state)
                for action in range(n_actions)
                for _prob, next_state, _reward, _terminated in transitions[state][action]
            },
        )
        successor_states[state] = successors
        for action in range(n_actions):
            probs = {successor: 0.0 for successor in successors}
            for prob, next_state, _reward, _terminated in transitions[state][action]:
                probs[int(next_state)] += float(prob)
            transition_probs[(state, action)] = np.array([probs[successor] for successor in successors], dtype=np.float32)
    return successor_states, transition_probs


def _categorical_sample(prob_n: list[float] | np.ndarray, np_random: np.random.Generator) -> int:
    prob_n = np.asarray(prob_n)
    cumulative_prob_n = np.cumsum(prob_n)
    return int(np.argmax(cumulative_prob_n > np_random.random()))


__all__ = ["CustomFrozenLake", "LEFT", "DOWN", "RIGHT", "UP"]
