"""FrozenLake renderer matching Gymnasium's toy-text visuals."""

from __future__ import annotations

from contextlib import closing
from io import StringIO
import os
from os import path
from typing import Any, Protocol

import numpy as np
from gymnasium import logger, utils

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


class FrozenLakeEnv(Protocol):
    metadata: dict[str, Any]
    render_mode: str | None
    desc: np.ndarray
    nrow: int
    ncol: int
    s: int
    lastaction: int | None
    window_size: tuple[int, int]
    cell_size: tuple[int, int]
    window_surface: Any
    clock: Any
    hole_img: Any
    cracked_hole_img: Any
    ice_img: Any
    elf_images: list[Any] | None
    goal_img: Any
    start_img: Any
    spec: Any


class FrozenLakeRenderer:
    """Renderer ported from Gymnasium's ``FrozenLakeEnv`` implementation."""

    def __init__(self, env: FrozenLakeEnv) -> None:
        self.env = env
        self._human_window_closed = False

    @property
    def human_window_closed(self) -> bool:
        return self._human_window_closed

    def render(self) -> str | np.ndarray | None:
        if self.env.render_mode is None:
            env_id = self.env.spec.id if self.env.spec is not None else "CustomFrozenLake-v0"
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{env_id}", render_mode="rgb_array")',
            )
            return None

        if self.env.render_mode == "ansi":
            return self._render_text()
        return self._render_gui(self.env.render_mode)

    def close(self) -> None:
        if self.env.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.env.window_surface = None
            self.env.clock = None
            self._human_window_closed = True

    def handle_pygame_event(self, event: Any) -> bool:
        import pygame

        if event.type == pygame.QUIT:
            self.close()
            return False
        return True

    def _render_gui(self, mode: str) -> np.ndarray | None:
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            from gymnasium.error import DependencyNotInstalled

            raise DependencyNotInstalled('pygame is not installed, run `pip install "gymnasium[toy-text]"`') from exc

        if self._human_window_closed and mode == "human":
            return None

        if self.env.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Frozen Lake")
                self.env.window_surface = pygame.display.set_mode(self.env.window_size)
            elif mode == "rgb_array":
                self.env.window_surface = pygame.Surface(self.env.window_size)

        assert self.env.window_surface is not None, "Something went wrong with pygame. This should never happen."

        if self.env.clock is None:
            self.env.clock = pygame.time.Clock()
        if self.env.hole_img is None:
            file_name = path.join(_toy_text_dir(), "img/hole.png")
            self.env.hole_img = pygame.transform.scale(pygame.image.load(file_name), self.env.cell_size)
        if self.env.cracked_hole_img is None:
            file_name = path.join(_toy_text_dir(), "img/cracked_hole.png")
            self.env.cracked_hole_img = pygame.transform.scale(pygame.image.load(file_name), self.env.cell_size)
        if self.env.ice_img is None:
            file_name = path.join(_toy_text_dir(), "img/ice.png")
            self.env.ice_img = pygame.transform.scale(pygame.image.load(file_name), self.env.cell_size)
        if self.env.goal_img is None:
            file_name = path.join(_toy_text_dir(), "img/goal.png")
            self.env.goal_img = pygame.transform.scale(pygame.image.load(file_name), self.env.cell_size)
        if self.env.start_img is None:
            file_name = path.join(_toy_text_dir(), "img/stool.png")
            self.env.start_img = pygame.transform.scale(pygame.image.load(file_name), self.env.cell_size)
        if self.env.elf_images is None:
            elfs = [
                path.join(_toy_text_dir(), "img/elf_left.png"),
                path.join(_toy_text_dir(), "img/elf_down.png"),
                path.join(_toy_text_dir(), "img/elf_right.png"),
                path.join(_toy_text_dir(), "img/elf_up.png"),
            ]
            self.env.elf_images = [
                pygame.transform.scale(pygame.image.load(file_name), self.env.cell_size) for file_name in elfs
            ]

        desc = self.env.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.env.nrow):
            for x in range(self.env.ncol):
                pos = (x * self.env.cell_size[0], y * self.env.cell_size[1])
                rect = (*pos, *self.env.cell_size)

                self.env.window_surface.blit(self.env.ice_img, pos)
                if desc[y][x] == b"H":
                    self.env.window_surface.blit(self.env.hole_img, pos)
                elif desc[y][x] == b"G":
                    self.env.window_surface.blit(self.env.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.env.window_surface.blit(self.env.start_img, pos)

                pygame.draw.rect(self.env.window_surface, (180, 200, 230), rect, 1)

        bot_row, bot_col = self.env.s // self.env.ncol, self.env.s % self.env.ncol
        cell_rect = (bot_col * self.env.cell_size[0], bot_row * self.env.cell_size[1])
        last_action = self.env.lastaction if self.env.lastaction is not None else 1
        elf_img = self.env.elf_images[last_action]

        if desc[bot_row][bot_col] == b"H":
            self.env.window_surface.blit(self.env.cracked_hole_img, cell_rect)
        else:
            self.env.window_surface.blit(elf_img, cell_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.env.clock.tick(self.env.metadata["render_fps"])
            return None
        if mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.env.window_surface)), axes=(1, 0, 2))
        return None

    def _render_text(self) -> str:
        desc = self.env.desc.tolist()
        outfile = StringIO()

        row, col = self.env.s // self.env.ncol, self.env.s % self.env.ncol
        desc = [[cell.decode("utf-8") for cell in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.env.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.env.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()


def validate_renderer_options(render_mode: str | None, render_window_size: int | None) -> None:
    if render_mode not in (None, "ansi", "rgb_array", "human"):
        raise ValueError("render_mode must be None, 'ansi', 'rgb_array', or 'human'.")
    if render_window_size is not None and int(render_window_size) <= 0:
        raise ValueError("render_window_size must be positive when provided.")


def _toy_text_dir() -> str:
    import gymnasium.envs.toy_text.frozen_lake as frozen_lake

    return path.dirname(frozen_lake.__file__)


__all__ = ["FrozenLakeRenderer", "validate_renderer_options"]
