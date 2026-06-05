"""Configurable local versions of MASA-Safe-RL tabular Pacman environments."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from experiments.utils.masa_tabular_envs.base import TabularEnv
from experiments.utils.masa_tabular_envs.dynamics import (
    PACMAN_ACTION_MAP,
    PACMAN_DIRECTION_MAP,
    PACMAN_REVERSE_MAP,
    cached_pacman_dynamics,
    choose_from_successors,
    create_pacman_end_component,
    layout_key,
    validate_probability,
)
from experiments.utils.masa_tabular_envs.renderers.pacman import (
    PacmanHat,
    PacmanRenderer,
    RGBColor,
    validate_renderer_options,
)

RenderMode = Literal["ansi", "rgb_array", "human"]

MINI_MAP = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    ],
    dtype=int,
)

STANDARD_MAP = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=int,
)


class _PacmanBase(TabularEnv):
    metadata = {"render_modes": ["ansi", "rgb_array", "human"], "render_fps": 4}

    def _init_rng(self, seed: int | None) -> None:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(np.random.SeedSequence().entropy)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._init_rng(seed)
        self._state = self._start_state
        self._step_count = 0
        if self.render_mode == "human":
            self.render()
        return self._state, {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise AssertionError(f"Invalid action {action}!")
        if self._transition_matrix is not None:
            self._state = int(self.np_random.choice(self._n_states, p=self._transition_matrix[:, self._state, action]))
        elif self._successor_states is not None:
            self._state = choose_from_successors(
                self.np_random,
                self._successor_states,
                self._transition_probs,
                self._state,
                int(action),
            )
        else:
            successors, probs = self._lazy_successor_distribution(self._state, int(action))
            self._state = int(self.np_random.choice(successors, p=probs))
        self._step_count += 1
        agent_y, agent_x, _agent_direction, ghost_y, ghost_x, _ghost_direction, food = self._reverse_state_map[self._state]
        reward = (
            1.0
            if (agent_y == self._food_y)
            and (agent_x == self._food_x)
            and not ((agent_y, agent_x) == (ghost_y, ghost_x))
            and food
            else 0.0
        )
        terminated = bool((agent_x, agent_y) == (self._agent_term_x, self._agent_term_y))
        if self.render_mode == "human":
            self.render()
        return self._state, reward, terminated, False, {}

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
        agent_y, agent_x, _agent_direction, ghost_y, ghost_x, _ghost_direction, food = self._reverse_state_map[int(obs)]
        if (agent_y == self._food_y) and (agent_x == self._food_x) and (agent_y, agent_x) != (ghost_y, ghost_x) and food:
            return {"food"}
        if (agent_y, agent_x) == (ghost_y, ghost_x):
            return {"ghost"}
        return set()

    @staticmethod
    def cost_fn(labels: set[str]) -> float:
        return 1.0 if "ghost" in labels else 0.0


class CustomMiniPacman(_PacmanBase):
    def __init__(
        self,
        *,
        layout: list[list[int]] | np.ndarray | None = None,
        food: tuple[int, int] = (7, 3),
        ghost_rand_prob: float = 0.6,
        agent_start: tuple[int, int] = (4, 1),
        agent_term: tuple[int, int] = (8, 6),
        agent_direction: int = 1,
        ghost_start: tuple[int, int] = (3, 5),
        ghost_direction: int = 1,
        render_mode: RenderMode | None = None,
        render_window_size: int = 512,
        pacman_hat: PacmanHat = "none",
        ghost_colors: tuple[RGBColor, ...] | None = None,
    ) -> None:
        super().__init__()
        validate_renderer_options(render_mode, render_window_size, pacman_hat)
        self._layout = np.array(MINI_MAP if layout is None else layout, dtype=int)
        self._n_row, self._n_col = self._layout.shape
        self._n_ghosts = 1
        self._n_directions = 4
        self._n_actions = 5
        self._food_x, self._food_y = _validate_xy("food", food, self._layout)
        self._ghost_rand_prob = validate_probability("ghost_rand_prob", ghost_rand_prob)
        dynamics = cached_pacman_dynamics(layout_key(self._layout), False, self._ghost_rand_prob, (self._food_x, self._food_y))
        self._successor_states, self._transition_probs, self._transition_matrix, self._n_states, self._state_map, self._reverse_state_map = dynamics
        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(self._n_actions)
        self._configure_positions(agent_start, agent_term, agent_direction, ghost_start, ghost_direction)
        self.np_random = None
        self._state = None
        self._step_count = 0
        self.render_mode = render_mode
        self.render_window_size = int(render_window_size)
        self.pacman_hat = pacman_hat
        self.ghost_colors = ghost_colors
        self._renderer = PacmanRenderer(self)
        self._validate_tabular_spaces()


class CustomPacman(_PacmanBase):
    def __init__(
        self,
        *,
        layout: list[list[int]] | np.ndarray | None = None,
        food: tuple[int, int] = (13, 1),
        ghost_rand_prob: float = 0.6,
        agent_start: tuple[int, int] = (1, 7),
        agent_term: tuple[int, int] = (9, 7),
        agent_direction: int = 1,
        ghost_start: tuple[int, int] = (12, 7),
        ghost_direction: int = 3,
        render_mode: RenderMode | None = None,
        render_window_size: int = 512,
        pacman_hat: PacmanHat = "none",
        ghost_colors: tuple[RGBColor, ...] | None = None,
    ) -> None:
        super().__init__()
        validate_renderer_options(render_mode, render_window_size, pacman_hat)
        self._layout = np.array(STANDARD_MAP if layout is None else layout, dtype=int)
        self._n_row, self._n_col = self._layout.shape
        self._n_ghosts = 1
        self._n_directions = 4
        self._n_actions = 5
        self._food_x, self._food_y = _validate_xy("food", food, self._layout)
        self._ghost_rand_prob = validate_probability("ghost_rand_prob", ghost_rand_prob)
        self._agent_poses, self._agent_pose_index = _valid_pacman_poses(self._layout)
        self._ghost_poses, self._ghost_pose_index = self._agent_poses, self._agent_pose_index
        self._n_states = len(self._agent_poses) * len(self._ghost_poses) * 2
        self._state_map = _LazyPacmanStateMap(self)
        self._reverse_state_map = _LazyPacmanReverseStateMap(self)
        self._successor_states = None
        self._transition_probs = None
        self._transition_matrix = None
        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(self._n_actions)
        self._configure_positions(agent_start, agent_term, agent_direction, ghost_start, ghost_direction)
        self.np_random = None
        self._state = None
        self._step_count = 0
        self.render_mode = render_mode
        self.render_window_size = int(render_window_size)
        self.pacman_hat = pacman_hat
        self.ghost_colors = ghost_colors
        self._renderer = PacmanRenderer(self)
        self._validate_tabular_spaces()

    def _lazy_successor_distribution(self, state: int, action: int) -> tuple[list[int], np.ndarray]:
        agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, food = self._reverse_state_map[int(state)]
        next_food = 0 if (agent_x, agent_y) == (self._food_x, self._food_y) else food
        ghost_forward_y, ghost_forward_x = _pacman_move(ghost_y, ghost_x, ghost_direction, self._n_row, self._n_col)
        next_loc_free = self._layout[ghost_forward_y, ghost_forward_x] == 0
        ghost_probs = _lazy_ghost_action_probs(
            self._layout,
            agent_y,
            agent_x,
            ghost_y,
            ghost_x,
            ghost_direction,
            bool(next_loc_free),
            self._ghost_rand_prob,
        )
        next_agent_y, next_agent_x, next_agent_direction = _lazy_next_agent(
            self._layout,
            agent_y,
            agent_x,
            agent_direction,
            action,
            bool(next_loc_free),
        )
        accum: dict[int, float] = {}
        for ghost_action, prob in enumerate(ghost_probs):
            if prob <= 0.0:
                continue
            next_ghost_direction = int(ghost_action if ghost_action <= 3 else ghost_direction)
            next_ghost_y, next_ghost_x = _pacman_move(ghost_y, ghost_x, int(ghost_action), self._n_row, self._n_col)
            if (next_agent_y, next_agent_x) == (ghost_y, ghost_x):
                next_state = self._state_map[
                    (
                        next_agent_y,
                        next_agent_x,
                        next_agent_direction,
                        ghost_y,
                        ghost_x,
                        ghost_direction,
                        next_food,
                    )
                ]
            else:
                next_state = self._state_map[
                    (
                        next_agent_y,
                        next_agent_x,
                        next_agent_direction,
                        next_ghost_y,
                        next_ghost_x,
                        next_ghost_direction,
                        next_food,
                    )
                ]
            accum[next_state] = accum.get(next_state, 0.0) + float(prob)
        successors = sorted(accum)
        probs = np.array([accum[successor] for successor in successors], dtype=np.float32)
        probs = probs / probs.sum()
        return successors, probs


def _configure_positions(
    self: _PacmanBase,
    agent_start: tuple[int, int],
    agent_term: tuple[int, int],
    agent_direction: int,
    ghost_start: tuple[int, int],
    ghost_direction: int,
) -> None:
    self._agent_start_x, self._agent_start_y = _validate_xy("agent_start", agent_start, self._layout)
    self._agent_term_x, self._agent_term_y = _validate_xy("agent_term", agent_term, self._layout)
    self._agent_start_direction = int(agent_direction)
    self._ghost_start_x, self._ghost_start_y = _validate_xy("ghost_start", ghost_start, self._layout)
    self._ghost_start_direction = int(ghost_direction)
    start_key = (
        self._agent_start_y,
        self._agent_start_x,
        self._agent_start_direction,
        self._ghost_start_y,
        self._ghost_start_x,
        self._ghost_start_direction,
        1,
    )
    if start_key not in self._state_map:
        raise ValueError(f"Invalid Pacman start configuration: {start_key}.")
    self._start_state = self._state_map[start_key]
    self._end_component = create_pacman_end_component(
        self._layout,
        self._agent_term_x,
        self._agent_term_y,
        self._state_map,
        food=True,
    )


_PacmanBase._configure_positions = _configure_positions  # type: ignore[attr-defined]


def _validate_xy(name: str, xy: tuple[int, int], layout: np.ndarray) -> tuple[int, int]:
    x, y = int(xy[0]), int(xy[1])
    if y < 0 or y >= layout.shape[0] or x < 0 or x >= layout.shape[1] or layout[y, x] != 0:
        raise ValueError(f"{name} must point to a free layout cell, got {(x, y)}.")
    return x, y


class _LazyPacmanStateMap:
    def __init__(self, env: CustomPacman) -> None:
        self.env = env

    def __getitem__(self, key: tuple[int, int, int, int, int, int, int]) -> int:
        agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, food = key
        agent_idx = self.env._agent_pose_index[(agent_y, agent_x, agent_direction)]
        ghost_idx = self.env._ghost_pose_index[(ghost_y, ghost_x, ghost_direction)]
        return (agent_idx * len(self.env._ghost_poses) + ghost_idx) * 2 + int(food)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, tuple) or len(key) != 7:
            return False
        agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, food = key
        return (
            (agent_y, agent_x, agent_direction) in self.env._agent_pose_index
            and (ghost_y, ghost_x, ghost_direction) in self.env._ghost_pose_index
            and int(food) in {0, 1}
        )


class _LazyPacmanReverseStateMap:
    def __init__(self, env: CustomPacman) -> None:
        self.env = env

    def __getitem__(self, state: int) -> tuple[int, int, int, int, int, int, int]:
        pair_idx, food = divmod(int(state), 2)
        agent_idx, ghost_idx = divmod(pair_idx, len(self.env._ghost_poses))
        agent_y, agent_x, agent_direction = self.env._agent_poses[agent_idx]
        ghost_y, ghost_x, ghost_direction = self.env._ghost_poses[ghost_idx]
        return agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, food


def _valid_pacman_poses(layout: np.ndarray) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int, int], int]]:
    poses = []
    for y in range(layout.shape[0]):
        for x in range(layout.shape[1]):
            if layout[y, x] != 0:
                continue
            for direction in range(4):
                front_y, front_x = _pacman_move(y, x, direction, layout.shape[0], layout.shape[1])
                dy, dx = PACMAN_DIRECTION_MAP[direction]
                back_y = int(np.clip(y - dy, 0, layout.shape[0] - 1))
                back_x = int(np.clip(x - dx, 0, layout.shape[1] - 1))
                if layout[front_y, front_x] == 1 and layout[back_y, back_x] == 1:
                    continue
                poses.append((y, x, direction))
    return poses, {pose: idx for idx, pose in enumerate(poses)}


def _pacman_move(y: int, x: int, action: int, nrow: int, ncol: int) -> tuple[int, int]:
    dy, dx = PACMAN_ACTION_MAP.get(int(action), (0, 0))
    return int(np.clip(y + dy, 0, nrow - 1)), int(np.clip(x + dx, 0, ncol - 1))


def _lazy_ghost_action_probs(
    layout: np.ndarray,
    agent_y: int,
    agent_x: int,
    ghost_y: int,
    ghost_x: int,
    ghost_direction: int,
    next_loc_free: bool,
    ghost_rand_prob: float,
) -> np.ndarray:
    prods = np.full(5, -np.inf, dtype=np.float32)
    vector = np.array([agent_y - ghost_y, agent_x - ghost_x], dtype=np.float32)
    for ghost_action in range(5):
        if next_loc_free and ghost_action == PACMAN_REVERSE_MAP[ghost_direction]:
            continue
        next_direction = ghost_action if ghost_action <= 3 else ghost_direction
        next_y, next_x = _pacman_move(ghost_y, ghost_x, next_direction, layout.shape[0], layout.shape[1])
        if layout[next_y, next_x] == 0:
            prods[ghost_action] = float(np.dot(vector, np.array(PACMAN_DIRECTION_MAP[next_direction])))
    available = np.where(prods != -np.inf)[0]
    probs = np.zeros(5, dtype=np.float32)
    if len(available) == 1:
        probs[available[0]] = 1.0
        return probs
    probs[available] = ghost_rand_prob / float(len(available) - 1)
    probs[int(np.argmax(prods))] = 1.0 - ghost_rand_prob
    return probs


def _lazy_next_agent(
    layout: np.ndarray,
    agent_y: int,
    agent_x: int,
    agent_direction: int,
    agent_action: int,
    next_loc_free: bool,
) -> tuple[int, int, int]:
    if next_loc_free and agent_action == PACMAN_REVERSE_MAP[agent_direction]:
        next_direction = agent_direction
    else:
        next_direction = agent_action if agent_action <= 3 else agent_direction
    next_y, next_x = _pacman_move(agent_y, agent_x, next_direction, layout.shape[0], layout.shape[1])
    if layout[next_y, next_x] == 1:
        next_y, next_x = _pacman_move(agent_y, agent_x, agent_direction, layout.shape[0], layout.shape[1])
        if layout[next_y, next_x] == 1:
            next_y, next_x = agent_y, agent_x
        next_direction = agent_direction
    return next_y, next_x, next_direction
