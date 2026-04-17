"""Generate large diagonal FrozenLake source environments (10x10 .. 100x100)."""

from __future__ import annotations

from pathlib import Path

import yaml

SIZES = list(range(10, 101, 10))
CORRIDOR_HALF_WIDTH = 1


def make_diagonal_source_map(size: int, corridor_half_width: int = CORRIDOR_HALF_WIDTH) -> list[str]:
    """Return a diagonal-corridor source map with holes outside the corridor."""
    rows: list[str] = []
    for r in range(size):
        row_chars: list[str] = []
        for c in range(size):
            if r == 0 and c == 0:
                row_chars.append("S")
            elif r == size - 1 and c == size - 1:
                row_chars.append("G")
            elif abs(r - c) <= corridor_half_width:
                row_chars.append("F")
            else:
                row_chars.append("H")
        rows.append("".join(row_chars))
    return rows


def build_env_payload() -> dict[str, dict]:
    payload: dict[str, dict] = {}
    for size in SIZES:
        key = f"diagonal_{size}x{size}"
        payload[key] = {
            "grid_size": size,
            "is_slippery": False,
            "corridor_half_width": CORRIDOR_HALF_WIDTH,
            "max_episode_steps": 4 * size,
            "env1_map": make_diagonal_source_map(size),
        }
    return payload


def main() -> None:
    out_path = Path(__file__).resolve().parent / "settings" / "source_envs.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_env_payload()
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
