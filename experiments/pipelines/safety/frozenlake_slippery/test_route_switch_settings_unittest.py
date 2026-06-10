"""Unit tests for FrozenLake slippery shield-safety route-switch layouts."""

from __future__ import annotations

from collections import deque
import unittest

from experiments.pipelines.safety.frozenlake_slippery.core.config import (
    get_pipeline_config,
)


def _is_solvable(env_map: tuple[str, ...]) -> bool:
    nrow = len(env_map)
    ncol = len(env_map[0])
    queue: deque[tuple[int, int]] = deque([(0, 0)])
    seen = {(0, 0)}
    while queue:
        row, col = queue.popleft()
        if env_map[row][col] == "G":
            return True
        for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = row + drow
            nc = col + dcol
            if not (0 <= nr < nrow and 0 <= nc < ncol):
                continue
            if (nr, nc) in seen or env_map[nr][nc] == "H":
                continue
            seen.add((nr, nc))
            queue.append((nr, nc))
    return False


class FrozenLakeShieldSafetyRouteSwitchSettingsTests(unittest.TestCase):
    def test_route_switch_6x6_loads_with_opposite_initial_routes(self) -> None:
        cfg = get_pipeline_config("route_switch_6x6")

        self.assertEqual(cfg.max_episode_steps, 36)
        self.assertEqual(len(cfg.source_map), 6)
        self.assertTrue(all(len(row) == 6 for row in cfg.source_map))
        self.assertTrue(all(len(row) == 6 for row in cfg.downstream_map))

        self.assertEqual(cfg.source_map[0][1], "F")
        self.assertEqual(cfg.source_map[1][0], "H")
        self.assertEqual(cfg.downstream_map[0][1], "H")
        self.assertEqual(cfg.downstream_map[1][0], "F")

        self.assertTrue(_is_solvable(cfg.source_map))
        self.assertTrue(_is_solvable(cfg.downstream_map))

    def test_old_route_blocked_6x6_blocks_source_path_and_opens_detour(self) -> None:
        cfg = get_pipeline_config("old_route_blocked_6x6")

        self.assertEqual(cfg.max_episode_steps, 36)
        self.assertEqual(cfg.source_map[0], "SFFHHH")
        self.assertEqual(cfg.downstream_map[0], "SFHHHH")

        blocked_old_route_cells = [
            (0, 2),
            (1, 2),
            (2, 2),
            (2, 3),
            (2, 4),
            (4, 4),
        ]
        for row, col in blocked_old_route_cells:
            self.assertEqual(cfg.source_map[row][col], "F")
            self.assertEqual(cfg.downstream_map[row][col], "H")

        opened_detour_cells = [
            (1, 0),
            (1, 1),
            (2, 0),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 5),
            (4, 5),
        ]
        for row, col in opened_detour_cells:
            self.assertEqual(cfg.source_map[row][col], "H")
            self.assertEqual(cfg.downstream_map[row][col], "F")

        self.assertTrue(_is_solvable(cfg.source_map))
        self.assertTrue(_is_solvable(cfg.downstream_map))


if __name__ == "__main__":
    unittest.main()
