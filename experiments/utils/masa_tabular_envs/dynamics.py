"""Tabular transition builders ported from MASA-Safe-RL with configurable inputs."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Any

import numpy as np


ACTION_MAP = {
    0: (0, -1),
    1: (0, 1),
    2: (1, 0),
    3: (-1, 0),
    4: (0, 0),
    5: (-1, -1),
    6: (-1, 1),
    7: (-1, 1),
    8: (1, 1),
}

PACMAN_ACTION_MAP = {
    0: (0, -1),
    1: (0, 1),
    2: (1, 0),
    3: (-1, 0),
    4: (0, 0),
}
PACMAN_DIRECTION_MAP = {
    0: (0, -1),
    1: (0, 1),
    2: (1, 0),
    3: (-1, 0),
}
PACMAN_REVERSE_MAP = {0: 1, 1: 0, 2: 3, 3: 2}


def validate_probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")
    return value


def validate_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def as_int_list(values: list[int] | tuple[int, ...] | np.ndarray | None) -> list[int]:
    if values is None:
        return []
    return [int(v) for v in values]


def validate_states(name: str, states: list[int], n_states: int) -> list[int]:
    bad = [state for state in states if state < 0 or state >= n_states]
    if bad:
        raise ValueError(f"{name} contains invalid states for n_states={n_states}: {bad[:5]}")
    return states


def create_transition_matrix(
    grid_size: int,
    n_states: int,
    n_actions: int,
    *,
    slip_prob: float = 0.0,
    terminal_states: list[int] | None = None,
    safe_states: list[int] | None = None,
    wall_states: list[int] | None = None,
) -> np.ndarray:
    grid_size = validate_positive_int("grid_size", grid_size)
    n_states = validate_positive_int("n_states", n_states)
    n_actions = validate_positive_int("n_actions", n_actions)
    slip_prob = validate_probability("slip_prob", slip_prob)
    if n_states != grid_size**2:
        raise ValueError("n_states must equal grid_size ** 2.")
    if n_actions >= len(ACTION_MAP):
        raise ValueError(f"n_actions must be less than {len(ACTION_MAP)}.")

    terminal = set(validate_states("terminal_states", as_int_list(terminal_states), n_states))
    safe = set(validate_states("safe_states", as_int_list(safe_states), n_states))
    walls = set(validate_states("wall_states", as_int_list(wall_states), n_states))

    grid = np.arange(grid_size * grid_size).reshape(grid_size, grid_size)
    matrix = np.zeros((n_states, n_states, n_actions), dtype=np.float64)

    for y in range(grid_size):
        for x in range(grid_size):
            state = int(grid[y, x])
            for action in range(n_actions):
                if state in terminal:
                    matrix[state, state, action] = 1.0
                    continue

                next_state = _grid_next_state(grid, grid_size, y, x, action, walls, state)
                intended_prob = 1.0 if state in safe else 1.0 - slip_prob
                matrix[next_state, state, action] += intended_prob
                if intended_prob == 1.0:
                    continue

                random_prob = slip_prob / float(n_actions - 1)
                for random_action in range(n_actions):
                    if random_action == action:
                        continue
                    next_state = _grid_next_state(grid, grid_size, y, x, random_action, walls, state)
                    matrix[next_state, state, action] += random_prob
    return matrix


def create_advanced_transition_matrix(
    grid_size: int,
    n_coloured_zones: int,
    n_states: int,
    n_actions: int,
    coloured_states: list[int],
    *,
    slip_prob: float = 0.0,
    terminal_states: list[int] | None = None,
    safe_states: list[int] | None = None,
    wall_states: list[int] | None = None,
) -> np.ndarray:
    grid_size = validate_positive_int("grid_size", grid_size)
    n_coloured_zones = validate_positive_int("n_coloured_zones", n_coloured_zones)
    n_states = validate_positive_int("n_states", n_states)
    n_actions = validate_positive_int("n_actions", n_actions)
    slip_prob = validate_probability("slip_prob", slip_prob)
    expected_states = (grid_size**2) * n_coloured_zones
    if n_states != expected_states:
        raise ValueError(f"n_states must equal {expected_states}.")
    if n_actions >= len(ACTION_MAP):
        raise ValueError(f"n_actions must be less than {len(ACTION_MAP)}.")

    terminal = set(validate_states("terminal_states", as_int_list(terminal_states), n_states))
    safe = set(validate_states("safe_states", as_int_list(safe_states), n_states))
    walls = set(validate_states("wall_states", as_int_list(wall_states), n_states))
    coloured = set(validate_states("coloured_states", as_int_list(coloured_states), n_states))

    grid = np.arange(grid_size * grid_size).reshape(grid_size, grid_size)
    grid_area = grid_size**2
    matrix = np.zeros((n_states, n_states, n_actions), dtype=np.float64)

    for zone in range(n_coloured_zones):
        offset = zone * grid_area
        for y in range(grid_size):
            for x in range(grid_size):
                state = int(grid[y, x] + offset)
                for action in range(n_actions):
                    if state in terminal:
                        matrix[state, state, action] = 1.0
                        continue
                    intended_prob = 1.0 if state in safe else 1.0 - slip_prob
                    _add_advanced_grid_transition(
                        matrix,
                        grid,
                        grid_size,
                        n_coloured_zones,
                        y,
                        x,
                        action,
                        walls,
                        state,
                        offset,
                        coloured,
                        intended_prob,
                    )
                    if intended_prob == 1.0:
                        continue
                    random_prob = slip_prob / float(n_actions - 1)
                    for random_action in range(n_actions):
                        if random_action == action:
                            continue
                        _add_advanced_grid_transition(
                            matrix,
                            grid,
                            grid_size,
                            n_coloured_zones,
                            y,
                            x,
                            random_action,
                            walls,
                            state,
                            offset,
                            coloured,
                            random_prob,
                        )
    return matrix


def _grid_next_state(
    grid: np.ndarray,
    grid_size: int,
    y: int,
    x: int,
    action: int,
    wall_states: set[int],
    fallback_state: int,
) -> int:
    dy, dx = ACTION_MAP[action]
    next_y = int(np.clip(y + dy, 0, grid_size - 1))
    next_x = int(np.clip(x + dx, 0, grid_size - 1))
    next_state = int(grid[next_y, next_x])
    return fallback_state if next_state in wall_states else next_state


def _add_advanced_grid_transition(
    matrix: np.ndarray,
    grid: np.ndarray,
    grid_size: int,
    n_coloured_zones: int,
    y: int,
    x: int,
    action: int,
    wall_states: set[int],
    state: int,
    offset: int,
    coloured_states: set[int],
    prob: float,
) -> None:
    next_state = _grid_next_state(grid, grid_size, y, x, action, wall_states, state - offset) + offset
    if state in coloured_states:
        zone_prob = prob / float(n_coloured_zones - 1)
        base_next_state = next_state - offset
        grid_area = grid_size**2
        for zone in range(n_coloured_zones):
            zone_offset = zone * grid_area
            if zone_offset == offset:
                continue
            matrix[base_next_state + zone_offset, state, action] += zone_prob
    else:
        matrix[next_state, state, action] += prob


def create_pacman_transition_dict(
    standard_map: np.ndarray,
    *,
    return_matrix: bool = False,
    n_directions: int = 4,
    n_actions: int = 5,
    n_ghosts: int = 1,
    ghost_rand_prob: float = 0.6,
    food_x: int | None = None,
    food_y: int | None = None,
):
    ghost_rand_prob = validate_probability("ghost_rand_prob", ghost_rand_prob)
    standard_map = np.asarray(standard_map, dtype=int)
    if standard_map.ndim != 2:
        raise ValueError("layout must be a 2D array.")
    if n_ghosts != 1:
        raise ValueError(f"function only supports n_ghosts=1, not {n_ghosts}.")
    if n_actions > len(PACMAN_ACTION_MAP):
        raise ValueError(f"n_actions must be <= {len(PACMAN_ACTION_MAP)}.")
    if n_directions > len(PACMAN_DIRECTION_MAP):
        raise ValueError(f"n_directions must be <= {len(PACMAN_DIRECTION_MAP)}.")

    state_map, reverse_state_map = _enumerate_pacman_states(
        standard_map,
        n_directions=n_directions,
        food_x=food_x,
        food_y=food_y,
    )
    n_states = len(state_map)
    successor_states: dict[int, list[int]] = defaultdict(list)
    transition_probs: dict[tuple[int, int], np.ndarray] = {}
    matrix = np.zeros((n_states, n_states, n_actions), dtype=np.float32) if return_matrix else None

    nrow, ncol = standard_map.shape
    for state_tuple, state_idx in state_map.items():
        agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, food = state_tuple
        next_food = 0 if food_x is not None and food_y is not None and (agent_x, agent_y) == (food_x, food_y) else food

        current_ghost_forward = _move(ghost_y, ghost_x, ghost_direction, nrow, ncol)
        next_loc_free = standard_map[current_ghost_forward] == 0
        ghost_action_probs = _ghost_action_probs(
            standard_map,
            agent_y,
            agent_x,
            ghost_y,
            ghost_x,
            ghost_direction,
            next_loc_free=bool(next_loc_free),
            n_actions=n_actions,
            ghost_rand_prob=ghost_rand_prob,
        )
        available_ghost_actions = np.where(ghost_action_probs > 0.0)[0]

        succ_set = set()
        transition_accumulator: dict[tuple[int, int], dict[int, float]] = {
            (state_idx, action): {} for action in range(n_actions)
        }
        for agent_action in range(n_actions):
            next_agent_y, next_agent_x, next_agent_direction = _next_pacman_agent(
                standard_map,
                agent_y,
                agent_x,
                agent_direction,
                agent_action,
                next_loc_free=bool(next_loc_free),
            )
            for ghost_action in available_ghost_actions:
                next_ghost_direction = int(ghost_action if ghost_action <= 3 else ghost_direction)
                next_ghost_y, next_ghost_x = _move(ghost_y, ghost_x, int(ghost_action), nrow, ncol)
                if (next_agent_y, next_agent_x) == (ghost_y, ghost_x):
                    next_tuple = (
                        next_agent_y,
                        next_agent_x,
                        next_agent_direction,
                        ghost_y,
                        ghost_x,
                        ghost_direction,
                        next_food,
                    )
                else:
                    next_tuple = (
                        next_agent_y,
                        next_agent_x,
                        next_agent_direction,
                        next_ghost_y,
                        next_ghost_x,
                        next_ghost_direction,
                        next_food,
                    )
                next_state = state_map[next_tuple]
                prob = float(ghost_action_probs[ghost_action])
                succ_set.add(next_state)
                probs = transition_accumulator[(state_idx, agent_action)]
                probs[next_state] = probs.get(next_state, 0.0) + prob
                if matrix is not None:
                    matrix[next_state, state_idx, agent_action] += prob

        successors = sorted(succ_set)
        successor_states[state_idx] = successors
        for action in range(n_actions):
            probs_by_successor = transition_accumulator[(state_idx, action)]
            transition_probs[(state_idx, action)] = np.array(
                [probs_by_successor.get(successor, 0.0) for successor in successors],
                dtype=np.float32,
            )

    return successor_states, transition_probs, matrix, n_states, state_map, reverse_state_map


@lru_cache(maxsize=16)
def cached_pacman_dynamics(
    layout_key: tuple[tuple[int, ...], ...],
    return_matrix: bool,
    ghost_rand_prob: float,
    food: tuple[int, int] | None,
):
    layout = np.array(layout_key, dtype=int)
    food_x = None if food is None else int(food[0])
    food_y = None if food is None else int(food[1])
    return create_pacman_transition_dict(
        layout,
        return_matrix=return_matrix,
        ghost_rand_prob=ghost_rand_prob,
        food_x=food_x,
        food_y=food_y,
    )


def create_pacman_end_component(
    standard_map: np.ndarray,
    agent_x_term: int,
    agent_y_term: int,
    state_map: dict[tuple[int, int, int, int, int, int, int], int],
    *,
    n_directions: int = 4,
    food: bool = False,
) -> list[int]:
    standard_map = np.asarray(standard_map, dtype=int)
    nrow, ncol = standard_map.shape
    component = []
    for ghost_y in range(nrow):
        for ghost_x in range(ncol):
            for ghost_direction in range(n_directions):
                front = _move(ghost_y, ghost_x, ghost_direction, nrow, ncol)
                back = (
                    int(np.clip(ghost_y - PACMAN_DIRECTION_MAP[ghost_direction][0], 0, nrow - 1)),
                    int(np.clip(ghost_x - PACMAN_DIRECTION_MAP[ghost_direction][1], 0, ncol - 1)),
                )
                if standard_map[front] == 1 and standard_map[back] == 1:
                    continue
                if standard_map[agent_y_term, agent_x_term] != 0 or standard_map[ghost_y, ghost_x] != 0:
                    continue
                if (agent_y_term, agent_x_term) == (ghost_y, ghost_x):
                    continue
                if food:
                    component.append(state_map[(agent_y_term, agent_x_term, 2, ghost_y, ghost_x, ghost_direction, 1)])
                component.append(state_map[(agent_y_term, agent_x_term, 2, ghost_y, ghost_x, ghost_direction, 0)])
    return component


def _enumerate_pacman_states(
    standard_map: np.ndarray,
    *,
    n_directions: int,
    food_x: int | None,
    food_y: int | None,
) -> tuple[dict[tuple[int, int, int, int, int, int, int], int], dict[int, tuple[int, int, int, int, int, int, int]]]:
    nrow, ncol = standard_map.shape
    state_map = {}
    reverse_state_map = {}
    state_idx = 0
    food_values = [0, 1] if food_x is not None and food_y is not None else [0]
    for agent_y in range(nrow):
        for agent_x in range(ncol):
            if standard_map[agent_y, agent_x] != 0:
                continue
            for agent_direction in range(n_directions):
                if _blocked_corridor(standard_map, agent_y, agent_x, agent_direction):
                    continue
                for ghost_y in range(nrow):
                    for ghost_x in range(ncol):
                        if standard_map[ghost_y, ghost_x] != 0:
                            continue
                        for ghost_direction in range(n_directions):
                            if _blocked_corridor(standard_map, ghost_y, ghost_x, ghost_direction):
                                continue
                            for food in food_values:
                                key = (agent_y, agent_x, agent_direction, ghost_y, ghost_x, ghost_direction, food)
                                state_map[key] = state_idx
                                reverse_state_map[state_idx] = key
                                state_idx += 1
    return state_map, reverse_state_map


def _blocked_corridor(layout: np.ndarray, y: int, x: int, direction: int) -> bool:
    nrow, ncol = layout.shape
    front = _move(y, x, direction, nrow, ncol)
    dy, dx = PACMAN_DIRECTION_MAP[direction]
    back = (int(np.clip(y - dy, 0, nrow - 1)), int(np.clip(x - dx, 0, ncol - 1)))
    return bool(layout[front] == 1 and layout[back] == 1)


def _move(y: int, x: int, action: int, nrow: int, ncol: int) -> tuple[int, int]:
    dy, dx = PACMAN_ACTION_MAP.get(action, (0, 0))
    return int(np.clip(y + dy, 0, nrow - 1)), int(np.clip(x + dx, 0, ncol - 1))


def _ghost_action_probs(
    layout: np.ndarray,
    agent_y: int,
    agent_x: int,
    ghost_y: int,
    ghost_x: int,
    ghost_direction: int,
    *,
    next_loc_free: bool,
    n_actions: int,
    ghost_rand_prob: float,
) -> np.ndarray:
    nrow, ncol = layout.shape
    prods = np.full(n_actions, -np.inf, dtype=np.float32)
    vector_to_agent = np.array([agent_y - ghost_y, agent_x - ghost_x], dtype=np.float32)
    for ghost_action in range(n_actions):
        if next_loc_free and ghost_action == PACMAN_REVERSE_MAP[ghost_direction]:
            continue
        next_direction = ghost_action if ghost_action <= 3 else ghost_direction
        next_y, next_x = _move(ghost_y, ghost_x, next_direction, nrow, ncol)
        if layout[next_y, next_x] == 0:
            prods[ghost_action] = float(np.dot(vector_to_agent, np.array(PACMAN_DIRECTION_MAP[next_direction])))
    available = np.where(prods != -np.inf)[0]
    if len(available) == 0:
        raise RuntimeError("Pacman ghost has no available actions.")
    probs = np.zeros(n_actions, dtype=np.float32)
    if len(available) == 1:
        probs[available[0]] = 1.0
        return probs
    probs[available] = ghost_rand_prob / float(len(available) - 1)
    probs[int(np.argmax(prods))] = 1.0 - ghost_rand_prob
    return probs


def _next_pacman_agent(
    layout: np.ndarray,
    agent_y: int,
    agent_x: int,
    agent_direction: int,
    agent_action: int,
    *,
    next_loc_free: bool,
) -> tuple[int, int, int]:
    nrow, ncol = layout.shape
    if next_loc_free and agent_action == PACMAN_REVERSE_MAP[agent_direction]:
        next_direction = agent_direction
    else:
        next_direction = agent_action if agent_action <= 3 else agent_direction
    next_y, next_x = _move(agent_y, agent_x, next_direction, nrow, ncol)
    if layout[next_y, next_x] == 1:
        next_y, next_x = _move(agent_y, agent_x, agent_direction, nrow, ncol)
        if layout[next_y, next_x] == 1:
            next_y, next_x = agent_y, agent_x
        next_direction = agent_direction
    return next_y, next_x, next_direction


def layout_key(layout: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in np.asarray(layout, dtype=int).tolist())


def choose_from_successors(
    rng: np.random.Generator,
    successor_states: dict[int, list[int]],
    transition_probs: dict[tuple[int, int], np.ndarray],
    state: int,
    action: int,
) -> int:
    successors = successor_states[int(state)]
    probs = transition_probs[(int(state), int(action))]
    return int(rng.choice(successors, p=probs))


def obs_from_media_v3_state(state: int, buffer_level: int, danger_states: int) -> dict[str, int]:
    danger = int(state % danger_states)
    time = int(state // danger_states)
    return {"danger": danger, "buffer": int(buffer_level), "time": time}
