"""Generate downstream FrozenLake tasks by swapping safe/hole cells from source maps.

Rules:
- Swap F <-> H everywhere.
- Keep S and G unchanged.
- Never place holes directly adjacent (4-neighborhood) to S or G.
- Validate each downstream map remains solvable from S to G.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import yaml


def neighbors4(r: int, c: int, n: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            yield nr, nc


def swap_frozenlake_cells(env_map: list[str]) -> list[list[str]]:
    grid = [list(row) for row in env_map]
    n = len(grid)
    for r in range(n):
        for c in range(n):
            if grid[r][c] == "F":
                grid[r][c] = "H"
            elif grid[r][c] == "H":
                grid[r][c] = "F"
    return grid


def protect_start_and_goal_neighbors(grid: list[list[str]]) -> None:
    n = len(grid)
    protected_centers = ((0, 0), (n - 1, n - 1))
    for sr, sc in protected_centers:
        for nr, nc in neighbors4(sr, sc, n):
            if grid[nr][nc] not in ("S", "G"):
                grid[nr][nc] = "F"
    grid[0][0] = "S"
    grid[n - 1][n - 1] = "G"


def map_is_solvable(env_map: list[str]) -> bool:
    n = len(env_map)
    start = (0, 0)
    goal = (n - 1, n - 1)
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for nr, nc in neighbors4(r, c, n):
            if (nr, nc) in seen:
                continue
            if env_map[nr][nc] == "H":
                continue
            seen.add((nr, nc))
            q.append((nr, nc))

    return False


def no_hole_next_to_start_or_goal(env_map: list[str]) -> bool:
    n = len(env_map)
    for sr, sc in ((0, 0), (n - 1, n - 1)):
        for nr, nc in neighbors4(sr, sc, n):
            if env_map[nr][nc] == "H":
                return False
    return True


def create_downstream_map(source_map: list[str]) -> list[str]:
    grid = swap_frozenlake_cells(source_map)
    protect_start_and_goal_neighbors(grid)
    downstream_map = ["".join(row) for row in grid]

    if not no_hole_next_to_start_or_goal(downstream_map):
        raise ValueError("Generated map has hole next to start/goal.")
    if not map_is_solvable(downstream_map):
        raise ValueError("Generated map is not solvable.")

    return downstream_map


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    settings_dir = base_dir / "settings"
    source_path = settings_dir / "source_envs.yaml"
    out_path = settings_dir / "downstream_envs.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source_cfgs = yaml.safe_load(source_path.read_text(encoding="utf-8"))

    out_cfgs: dict[str, dict] = {}
    for layout_name, cfg in source_cfgs.items():
        source_map = cfg["env1_map"]
        downstream_map = create_downstream_map(source_map)

        out_cfgs[layout_name] = {
            "grid_size": int(cfg["grid_size"]),
            "is_slippery": bool(cfg.get("is_slippery", False)),
            "max_episode_steps": int(cfg["max_episode_steps"]),
            "source_layout": layout_name,
            "env2_map": downstream_map,
            "notes": "F/H swapped from source map; holes are disallowed next to S and G.",
        }

    out_path.write_text(yaml.safe_dump(out_cfgs, sort_keys=False), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
