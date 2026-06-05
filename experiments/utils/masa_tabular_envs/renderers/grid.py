"""Lightweight MASA-style renderers for custom grid worlds."""

from __future__ import annotations

from typing import Any

import numpy as np

RGBColor = tuple[int, int, int]

FLOOR_COLOR = (235, 232, 221)
FLOOR_ALT_COLOR = (226, 222, 211)
WALL_COLOR = (68, 78, 92)
START_COLOR = (245, 188, 82)
AGENT_COLOR = (55, 115, 206)
GOAL_COLOR = (82, 162, 107)
LAVA_COLOR = (207, 72, 54)
BOMB_COLOR = (34, 39, 48)
MEDIC_COLOR = (78, 166, 129)
ZONE_COLORS: dict[str, RGBColor] = {
    "green": (109, 178, 103),
    "yellow": (230, 200, 81),
    "red": (216, 93, 83),
    "blue": (89, 143, 219),
    "pink": (213, 116, 176),
    "purple": (172, 111, 205),
}
ZONE_MARKERS = {
    "green": "G",
    "yellow": "Y",
    "red": "R",
    "blue": "B",
    "pink": "P",
    "purple": "U",
}


class GridWorldRenderer:
    """Shared renderer compatible with the MASA tabular gridworld attributes."""

    def __init__(self, env: Any, title: str) -> None:
        self.env = env
        self.title = title
        self._human_window = None
        self._human_clock = None
        self._human_window_closed = False

    @property
    def human_window_closed(self) -> bool:
        return self._human_window_closed

    def render(self) -> str | np.ndarray | None:
        if self.env.render_mode is None:
            return None
        if self.env.render_mode == "ansi":
            return self._render_ansi()
        if self.env.render_mode == "human":
            self._render_human()
            return None
        return self._render_rgb_array()

    def close(self) -> None:
        if self._human_window is not None:
            import pygame

            pygame.display.quit()
            self._human_window = None
            self._human_clock = None
            self._human_window_closed = True

    def handle_pygame_event(self, event: Any) -> bool:
        import pygame

        if event.type == pygame.QUIT:
            self.close()
            return False
        return True

    def _state_mod(self, state: int) -> int:
        return int(state) % int(self.env._grid_size**2)

    def _position(self, state: int) -> tuple[int, int]:
        state = self._state_mod(state)
        return state // int(self.env._grid_size), state % int(self.env._grid_size)

    def _render_ansi(self) -> str:
        size = int(self.env._grid_size)
        chars = np.full((size, size), " ", dtype="<U1")
        for state in getattr(self.env, "_wall_states", []):
            chars[self._position(state)] = "#"
        for state in getattr(self.env, "_goal_states", []):
            chars[self._position(state)] = "G"
        for state in getattr(self.env, "_lava_states", []):
            chars[self._position(state)] = "L"
        for name, marker in ZONE_MARKERS.items():
            for state in getattr(self.env, f"_{name}_states", []):
                chars[self._position(state)] = marker
        for state in getattr(self.env, "_medic_states", []):
            chars[self._position(state)] = "M"
        for state in getattr(self.env, "_bomb_states", []):
            chars[self._position(state)] = "X"
        for state in getattr(self.env, "_start_states", [getattr(self.env, "_start_state", 0)]):
            chars[self._position(state)] = "S"
        state = getattr(self.env, "_state", None)
        chars[self._position(getattr(self.env, "_start_state", 0) if state is None else state)] = "A"
        return "\n".join("".join(row) for row in chars)

    def _render_rgb_array(self) -> np.ndarray:
        size = int(self.env._grid_size)
        cell = max(12, int(self.env.render_window_size) // size)
        frame = np.zeros((size * cell, size * cell, 3), dtype=np.uint8)
        for row in range(size):
            for col in range(size):
                color = FLOOR_COLOR if (row + col) % 2 == 0 else FLOOR_ALT_COLOR
                self._fill_cell(frame, row, col, cell, color)
        for state in getattr(self.env, "_wall_states", []):
            self._fill_state(frame, state, cell, WALL_COLOR)
        for state in getattr(self.env, "_goal_states", []):
            self._fill_state(frame, state, cell, GOAL_COLOR)
        for state in getattr(self.env, "_lava_states", []):
            self._fill_state(frame, state, cell, LAVA_COLOR)
        for name, color in ZONE_COLORS.items():
            for state in getattr(self.env, f"_{name}_states", []):
                self._fill_state(frame, state, cell, color)
        for state in getattr(self.env, "_medic_states", []):
            self._fill_state(frame, state, cell, MEDIC_COLOR)
        for state in getattr(self.env, "_bomb_states", []):
            self._fill_state(frame, state, cell, BOMB_COLOR)
        for state in getattr(self.env, "_start_states", [getattr(self.env, "_start_state", 0)]):
            self._fill_state(frame, state, cell, START_COLOR)
        state = getattr(self.env, "_state", None)
        row, col = self._position(getattr(self.env, "_start_state", 0) if state is None else state)
        rr = slice(row * cell + cell // 4, row * cell + (3 * cell) // 4)
        cc = slice(col * cell + cell // 4, col * cell + (3 * cell) // 4)
        frame[rr, cc] = AGENT_COLOR
        return frame

    def _render_human(self) -> None:
        import pygame

        if self._human_window_closed:
            return
        frame = self._render_rgb_array()
        if self._human_window is None:
            pygame.init()
            pygame.display.set_caption(self.title)
            self._human_window = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
            self._human_clock = pygame.time.Clock()
        for event in pygame.event.get():
            self.handle_pygame_event(event)
        if self._human_window is None:
            return
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._human_window.blit(surface, (0, 0))
        pygame.display.flip()
        if self._human_clock is not None:
            self._human_clock.tick(self.env.metadata["render_fps"])

    def _fill_state(self, frame: np.ndarray, state: int, cell: int, color: RGBColor) -> None:
        row, col = self._position(state)
        self._fill_cell(frame, row, col, cell, color)

    @staticmethod
    def _fill_cell(frame: np.ndarray, row: int, col: int, cell: int, color: RGBColor) -> None:
        frame[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell] = color
