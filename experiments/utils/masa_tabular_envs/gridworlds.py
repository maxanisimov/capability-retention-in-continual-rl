"""Configurable local versions of MASA-Safe-RL tabular grid worlds."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from experiments.utils.masa_tabular_envs.base import TabularEnv
from experiments.utils.masa_tabular_envs.dynamics import (
    as_int_list,
    create_advanced_transition_matrix,
    create_transition_matrix,
    validate_positive_int,
    validate_probability,
    validate_states,
)
from experiments.utils.masa_tabular_envs.renderers.bridge_crossing import (
    BridgeCrossingRenderer,
    validate_renderer_options as validate_bridge_renderer_options,
)
from experiments.utils.masa_tabular_envs.renderers.colour_bomb_grid_world import (
    ColourBombGridWorldRenderer,
    validate_renderer_options as validate_colour_bomb_renderer_options,
)
from experiments.utils.masa_tabular_envs.renderers.colour_grid_world import (
    ColourGridWorldRenderer,
    validate_renderer_options as validate_colour_grid_renderer_options,
)

RenderMode = Literal["ansi", "rgb_array", "human"]


def _grid(size: int) -> np.ndarray:
    return np.arange(size**2).reshape(size, size)


def _build_label_dict(**groups: list[int]) -> defaultdict[int, set[str]]:
    labels: defaultdict[int, set[str]] = defaultdict(set)
    for name, states in groups.items():
        for state in states:
            labels[int(state)].add(name)
    return labels


class _MatrixGridWorld(TabularEnv):
    metadata = {"render_modes": ["ansi", "rgb_array", "human"], "render_fps": 4}

    def _init_rng(self, seed: int | None) -> None:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(np.random.SeedSequence().entropy)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._init_rng(seed)
        if getattr(self, "_start_states", None):
            self._state = int(self.np_random.choice(self._start_states))
        else:
            self._state = int(self._start_state)
        self._step_count = 0
        if self.render_mode == "human":
            self.render()
        return self._state, {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise AssertionError(f"Invalid action {action}!")
        self._state = int(self.np_random.choice(self._n_states, p=self._transition_matrix[:, self._state, action]))
        self._step_count += 1
        terminated = bool(self._state in getattr(self, "_terminal_states", []))
        reward = 1.0 if self._state in getattr(self, "_goal_states", []) else 0.0
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
        return set(self._label_dict[int(obs)])

    @staticmethod
    def cost_fn(labels: set[str]) -> float:
        return 1.0 if ("lava" in labels or "bomb" in labels or "blue" in labels) else 0.0


class CustomBridgeCrossing(_MatrixGridWorld):
    def __init__(
        self,
        *,
        slip_prob: float = 0.04,
        grid_size: int = 20,
        start_state: int | None = None,
        goal_states: list[int] | None = None,
        lava_states: list[int] | None = None,
        render_mode: RenderMode | None = None,
        render_window_size: int = 512,
    ) -> None:
        super().__init__()
        self._grid_size = validate_positive_int("grid_size", grid_size)
        grid = _grid(self._grid_size)
        self._ncol = self._grid_size
        self._nrow = self._grid_size
        self._n_states = self._grid_size**2
        self._n_actions = 5
        self._start_state = int(grid[-1, 0] if start_state is None else start_state)
        self._goal_states = as_int_list(list(grid[:7, :].flatten()) if goal_states is None else goal_states)
        default_lava = list(grid[8:12, :8].flatten()) + list(grid[8:12, -9:].flatten())
        self._lava_states = as_int_list(default_lava if lava_states is None else lava_states)
        validate_states("start_state", [self._start_state], self._n_states)
        validate_states("goal_states", self._goal_states, self._n_states)
        validate_states("lava_states", self._lava_states, self._n_states)
        self._terminal_states = self._goal_states + self._lava_states
        self._slip_prob = validate_probability("slip_prob", slip_prob)
        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(self._n_actions)
        self._transition_matrix = create_transition_matrix(
            self._grid_size,
            self._n_states,
            self._n_actions,
            slip_prob=self._slip_prob,
            terminal_states=self._terminal_states,
        )
        self._label_dict = _build_label_dict(start=[self._start_state], goal=self._goal_states, lava=self._lava_states)
        self.np_random = None
        self._state = None
        self._step_count = 0
        self.render_mode = render_mode
        self.render_window_size = int(render_window_size)
        validate_bridge_renderer_options(self.render_mode, self.render_window_size)
        self._renderer = BridgeCrossingRenderer(self)
        self._validate_tabular_spaces()


class CustomBridgeCrossingV2(CustomBridgeCrossing):
    def __init__(
        self,
        *,
        slip_prob: float = 0.04,
        grid_size: int = 20,
        start_state: int | None = None,
        goal_states: list[int] | None = None,
        lava_states: list[int] | None = None,
        render_mode: RenderMode | None = None,
        render_window_size: int = 512,
    ) -> None:
        grid = _grid(grid_size)
        default_lava = list(grid[8:12, 2:16].flatten()) + [int(grid[11, 1])]
        super().__init__(
            slip_prob=slip_prob,
            grid_size=grid_size,
            start_state=start_state,
            goal_states=goal_states,
            lava_states=default_lava if lava_states is None else lava_states,
            render_mode=render_mode,
            render_window_size=render_window_size,
        )


class CustomColourGridWorld(_MatrixGridWorld):
    def __init__(
        self,
        *,
        slip_prob: float = 0.1,
        grid_size: int = 9,
        start_state: int = 0,
        goal_state: int | None = None,
        blue_state: int = 36,
        green_state: int = 40,
        purple_state: int = 4,
        render_mode: RenderMode | None = None,
        render_window_size: int = 512,
    ) -> None:
        super().__init__()
        self._grid_size = validate_positive_int("grid_size", grid_size)
        self._ncol = self._grid_size
        self._nrow = self._grid_size
        self._n_states = self._grid_size**2
        self._n_actions = 5
        self._start_state = int(start_state)
        self._goal_state = int(self._n_states - 1 if goal_state is None else goal_state)
        self._goal_states = [self._goal_state]
        self._blue_state = int(blue_state)
        self._green_state = int(green_state)
        self._purple_state = int(purple_state)
        self._blue_states = [self._blue_state]
        self._green_states = [self._green_state]
        self._purple_states = [self._purple_state]
        validate_states(
            "colour_grid_states",
            [self._start_state, self._goal_state, self._blue_state, self._green_state, self._purple_state],
            self._n_states,
        )
        self._terminal_states = [self._goal_state]
        self._slip_prob = validate_probability("slip_prob", slip_prob)
        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(self._n_actions)
        self._transition_matrix = create_transition_matrix(
            self._grid_size,
            self._n_states,
            self._n_actions,
            slip_prob=self._slip_prob,
            terminal_states=self._terminal_states,
        )
        self._label_dict = _build_label_dict(
            start=[self._start_state],
            goal=[self._goal_state],
            blue=[self._blue_state],
            green=[self._green_state],
            purple=[self._purple_state],
        )
        self.np_random = None
        self._state = None
        self._step_count = 0
        self.render_mode = render_mode
        self.render_window_size = int(render_window_size)
        validate_colour_grid_renderer_options(self.render_mode, self.render_window_size)
        self._renderer = ColourGridWorldRenderer(self)
        self._validate_tabular_spaces()


class CustomColourBombGridWorld(_MatrixGridWorld):
    def __init__(
        self,
        *,
        slip_prob: float = 0.1,
        grid_size: int = 9,
        start_state: int = 74,
        start_states: list[int] | None = None,
        green_states: list[int] | None = None,
        yellow_states: list[int] | None = None,
        red_states: list[int] | None = None,
        blue_states: list[int] | None = None,
        pink_states: list[int] | None = None,
        wall_states: list[int] | None = None,
        bomb_states: list[int] | None = None,
        medic_states: list[int] | None = None,
        render_mode: RenderMode | None = None,
        render_window_size: int = 512,
    ) -> None:
        super().__init__()
        self._grid_size = validate_positive_int("grid_size", grid_size)
        self._ncol = self._grid_size
        self._nrow = self._grid_size
        self._n_states = self._grid_size**2
        self._n_actions = 5
        self._start_state = int(start_state)
        self._start_states = as_int_list([self._start_state] if start_states is None else start_states)
        self._green_states = as_int_list([65] if green_states is None else green_states)
        self._yellow_states = as_int_list([70, 79] if yellow_states is None else yellow_states)
        self._red_states = as_int_list([] if red_states is None else red_states)
        self._blue_states = as_int_list([9, 10, 18, 19] if blue_states is None else blue_states)
        self._pink_states = as_int_list([7, 8, 16, 17] if pink_states is None else pink_states)
        self._wall_states = as_int_list(
            [11, 12, 13, 14, 15, 29, 30, 50, 52, 53, 55, 56, 57, 59, 64, 66, 69]
            if wall_states is None
            else wall_states
        )
        self._bomb_states = as_int_list([27, 43, 78] if bomb_states is None else bomb_states)
        self._medic_states = as_int_list([] if medic_states is None else medic_states)
        self._goal_states = (
            self._green_states + self._yellow_states + self._red_states + self._blue_states + self._pink_states
        )
        for name in (
            "start_states",
            "green_states",
            "yellow_states",
            "red_states",
            "blue_states",
            "pink_states",
            "wall_states",
            "bomb_states",
            "medic_states",
            "goal_states",
        ):
            validate_states(name, getattr(self, f"_{name}"), self._n_states)
        self._terminal_states = self._goal_states
        self._safe_states = []
        self._active_colour_dict = {}
        self._slip_prob = validate_probability("slip_prob", slip_prob)
        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(self._n_actions)
        self._transition_matrix = create_transition_matrix(
            self._grid_size,
            self._n_states,
            self._n_actions,
            slip_prob=self._slip_prob,
            terminal_states=self._terminal_states,
            wall_states=self._wall_states,
        )
        self._label_dict = _build_label_dict(
            start=self._start_states,
            green=self._green_states,
            yellow=self._yellow_states,
            red=self._red_states,
            blue=self._blue_states,
            pink=self._pink_states,
            bomb=self._bomb_states,
            medic=self._medic_states,
        )
        self.np_random = None
        self._state = None
        self._step_count = 0
        self.render_mode = render_mode
        self.render_window_size = int(render_window_size)
        validate_colour_bomb_renderer_options(self.render_mode, self.render_window_size)
        self._renderer = ColourBombGridWorldRenderer(self)
        self._validate_tabular_spaces()


class CustomColourBombGridWorldV2(CustomColourBombGridWorld):
    def __init__(
        self,
        *,
        slip_prob: float = 0.1,
        render_mode: RenderMode | None = None,
        render_window_size: int = 512,
        **overrides: Any,
    ) -> None:
        defaults = _colour_bomb_v2_defaults()
        defaults.update(overrides)
        super().__init__(
            slip_prob=slip_prob,
            grid_size=15,
            render_mode=render_mode,
            render_window_size=render_window_size,
            **defaults,
        )
        self._medic_states = as_int_list(defaults["medic_states"])
        self._safe_states = self._medic_states
        self._terminal_states = []
        self._transition_matrix = create_transition_matrix(
            self._grid_size,
            self._n_states,
            self._n_actions,
            slip_prob=self._slip_prob,
            terminal_states=self._goal_states,
            safe_states=self._safe_states,
            wall_states=self._wall_states,
        )
        self._add_goal_restart_transitions()
        for state in self._medic_states:
            self._label_dict[state].add("medic")

    def _add_goal_restart_transitions(self) -> None:
        for state in self._goal_states:
            for action in range(self._n_actions):
                probs = np.zeros_like(self._transition_matrix[:, state, action])
                probs[self._start_states] = 1.0 / len(self._start_states)
                self._transition_matrix[:, state, action] = probs

    def step(self, action: int):
        obs, reward, _terminated, truncated, info = super().step(action)
        return obs, reward, False, truncated, info


class CustomColourBombGridWorldV3(CustomColourBombGridWorldV2):
    def __init__(
        self,
        *,
        slip_prob: float = 0.1,
        n_coloured_zones: int = 5,
        render_mode: RenderMode | None = None,
        render_window_size: int = 512,
        **overrides: Any,
    ) -> None:
        base = _colour_bomb_v2_defaults()
        base.update(overrides)
        self._grid_size = 15
        self._n_coloured_zones = validate_positive_int("n_coloured_zones", n_coloured_zones)
        expanded = _expand_colour_bomb_defaults(base, self._grid_size, self._n_coloured_zones)
        _MatrixGridWorld.__init__(self)
        self._n_coloured_zones = self._n_coloured_zones
        self._ncol = self._grid_size
        self._nrow = self._grid_size
        self._n_states = (self._grid_size**2) * self._n_coloured_zones
        self._n_actions = 5
        self._start_state = int(expanded["start_state"])
        self._start_states = expanded["start_states"]
        self._green_states = expanded["green_states"]
        self._yellow_states = expanded["yellow_states"]
        self._red_states = expanded["red_states"]
        self._blue_states = expanded["blue_states"]
        self._pink_states = expanded["pink_states"]
        self._wall_states = expanded["wall_states"]
        self._bomb_states = expanded["bomb_states"]
        self._medic_states = expanded["medic_states"]
        self._goal_states = (
            self._green_states + self._yellow_states + self._red_states + self._blue_states + self._pink_states
        )
        for name in (
            "start_states",
            "green_states",
            "yellow_states",
            "red_states",
            "blue_states",
            "pink_states",
            "wall_states",
            "bomb_states",
            "medic_states",
            "goal_states",
        ):
            validate_states(name, getattr(self, f"_{name}"), self._n_states)
        self._slip_prob = validate_probability("slip_prob", slip_prob)
        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(self._n_actions)
        self._medic_states = expanded["medic_states"]
        self._safe_states = self._medic_states
        self._active_colour_dict = {0: "green", 1: "yellow", 2: "red", 3: "blue", 4: "pink"}
        self._transition_matrix = create_advanced_transition_matrix(
            self._grid_size,
            self._n_coloured_zones,
            self._n_states,
            self._n_actions,
            self._goal_states,
            slip_prob=self._slip_prob,
            safe_states=self._safe_states,
            wall_states=self._wall_states,
        )
        self._add_goal_restart_transitions()
        self._label_dict = _build_label_dict(
            start=self._start_states,
            green=self._green_states,
            yellow=self._yellow_states,
            red=self._red_states,
            blue=self._blue_states,
            pink=self._pink_states,
            bomb=self._bomb_states,
            medic=self._medic_states,
        )
        self._terminal_states = []
        self.np_random = None
        self._state = None
        self._step_count = 0
        self.render_mode = render_mode
        self.render_window_size = int(render_window_size)
        validate_colour_bomb_renderer_options(self.render_mode, self.render_window_size)
        self._renderer = ColourBombGridWorldRenderer(self)
        self._validate_tabular_spaces()


def _colour_bomb_v2_defaults() -> dict[str, list[int] | int]:
    return {
        "start_state": 16,
        "start_states": [16, 199, 178, 112, 26],
        "green_states": [170],
        "yellow_states": [176, 191],
        "red_states": [88, 89, 103, 104],
        "blue_states": [121, 122, 136, 137],
        "pink_states": [53, 54, 68, 69],
        "wall_states": [
            45,
            60,
            75,
            210,
            195,
            180,
            165,
            150,
            142,
            211,
            212,
            213,
            214,
            215,
            216,
            220,
            221,
            222,
            223,
            224,
            209,
            183,
            184,
            169,
            185,
            186,
            187,
            192,
            177,
            162,
            161,
            160,
            143,
            144,
            129,
            138,
            139,
            140,
            141,
            125,
            3,
            18,
            47,
            62,
            63,
            64,
            50,
            35,
            20,
            80,
            81,
            95,
            83,
            84,
            99,
            100,
            116,
            131,
            133,
            134,
            87,
            72,
            57,
            70,
            55,
            39,
            9,
            13,
            14,
            29,
            44,
            59,
        ],
        "bomb_states": [76, 181, 123, 82, 207, 8, 58],
        "medic_states": [154, 93, 38, 205, 74],
    }


def _expand_colour_bomb_defaults(
    base: dict[str, Any],
    grid_size: int,
    n_coloured_zones: int,
) -> dict[str, Any]:
    grid_area = grid_size**2
    expanded: dict[str, Any] = {"start_state": int(base["start_states"][0])}
    list_keys = [
        "start_states",
        "green_states",
        "yellow_states",
        "red_states",
        "blue_states",
        "pink_states",
        "wall_states",
        "bomb_states",
        "medic_states",
    ]
    for key in list_keys:
        values = as_int_list(base.get(key, []))
        expanded[key] = [value + zone * grid_area for zone in range(n_coloured_zones) for value in values]
    return expanded
