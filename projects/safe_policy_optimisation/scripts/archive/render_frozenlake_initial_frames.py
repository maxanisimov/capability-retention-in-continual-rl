#!/usr/bin/env python3
"""Save actual rgb_array reset renders of all five FrozenLake experiment maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from projects.safe_policy_optimisation.utils import (  # noqa: E402
    frozen_lake_experiment as frozen,
)

RUNS = REPO / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
DEFAULT_OUTPUT = (
    REPO / "projects/safe_policy_optimisation/figures/frozenlake_initial_frames"
)
COHORTS = {
    16: "frozenlake16_shaping_pspo_segment_verify_first_t204800_20261008T123000Z",
    32: "frozenlake32_shaping_pspo_segment_verify_first_t204800_20261008T123000Z",
    64: "frozenlake64_shaping_pspo_segment_verify_first_t409600_20261008T124400Z",
    128: "frozenlake128_shaping_pspo_segment_verify_first_t1024000_20261008T124400Z",
    256: "frozenlake256_shaping_pspo_segment_verify_first_t4096000_20261008T124400Z",
}


def load_experiment(size: int, runs: Path = RUNS) -> tuple[dict, Path, str]:
    cohort = runs / COHORTS[size]
    config_path = cohort / "pspo/seed0/config.json"
    config = json.loads(config_path.read_text())
    if config["env_kwargs"]["size"] != size:
        raise ValueError("Experiment configuration has the wrong layout size")
    source = Path(frozen.__file__).resolve()
    expected_source = config["source_sha256"][str(source.relative_to(REPO))]
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_source:
        raise ValueError("FrozenLake implementation changed since the experiment")
    layout = cohort / "pspo/seed0/_inputs/layout.txt"
    data = layout.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    lines = data.decode().splitlines()
    if len(lines) != size or any(len(line) != size for line in lines):
        raise ValueError("Saved experiment layout has the wrong dimensions")
    for seed in range(10):
        other = cohort / f"pspo/seed{seed}/_inputs/layout.txt"
        if hashlib.sha256(other.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Layout differs for training seed {seed}")
    return config, layout, digest


def render_initial_frame(config: dict, layout: Path, tile_pixels: int = 16):
    if tile_pixels <= 0:
        raise ValueError("Tile resolution must be positive")
    size = config["env_kwargs"]["size"]
    env = gym.make(
        config["env_id"],
        render_mode="rgb_array",
        max_episode_steps=config["max_episode_steps"],
        layout_path=str(layout),
        **config["env_kwargs"],
    )
    try:
        state, _ = env.reset(seed=0)
        if int(state) != 0 or env.unwrapped.lastaction is not None:
            raise AssertionError("Expected reset state before any action")
        # Set resolution before first render, when surfaces/sprites are created.
        # The renderer and map/dynamics remain the experiment's own environment.
        env.unwrapped.window_size = (size * tile_pixels, size * tile_pixels)
        env.unwrapped.cell_size = (tile_pixels, tile_pixels)
        frame = env.render()
        if (
            frame.shape != (size * tile_pixels, size * tile_pixels, 3)
            or frame.dtype != np.uint8
        ):
            raise AssertionError("Unexpected rgb_array output")
        if env.unwrapped.s != 0 or env.unwrapped.lastaction is not None:
            raise AssertionError("Rendering changed the initial environment state")
        return frame
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sizes", type=int, nargs="+", choices=COHORTS, default=list(COHORTS)
    )
    parser.add_argument("--tile-pixels", type=int, default=16)
    args = parser.parse_args()
    if args.tile_pixels <= 0 or len(args.sizes) != len(set(args.sizes)):
        parser.error("Require positive --tile-pixels and distinct sizes")
    inputs = {size: load_experiment(size, args.runs_dir) for size in args.sizes}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {}
    for size in args.sizes:
        config, layout, digest = inputs[size]
        frame = render_initial_frame(config, layout, args.tile_pixels)
        filename = f"frozenlake{size}x{size}_initial_frame.png"
        destination = output / filename
        Image.fromarray(frame).save(destination)
        with Image.open(destination) as saved:
            if not np.array_equal(np.asarray(saved), frame):
                raise AssertionError("Saved PNG differs from env.render() array")
        metadata[str(size)] = {
            "filename": filename,
            "frame_shape": list(frame.shape),
            "render_mode": "rgb_array",
            "render_call": "env.render()",
            "renderer": "Gymnasium native FrozenLake renderer",
            "reset_seed": 0,
            "initial_state_id": 0,
            "actions_executed": 0,
            "tile_pixels": args.tile_pixels,
            "layout_path": str(layout.resolve()),
            "layout_sha256": digest,
            "all_ten_seed_layouts_identical": True,
            "png_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "env_kwargs": config["env_kwargs"],
            "max_episode_steps": config["max_episode_steps"],
        }
        print(
            f"{size}x{size}: {frame.shape[1]}x{frame.shape[0]} pixels -> {destination}",
            flush=True,
        )
    (output / "frame_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
