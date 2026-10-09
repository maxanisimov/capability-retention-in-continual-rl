"""Write the complete interval certificate for the LunarLander descent shield.

The descent shield is active on the open set ``y < critical_height`` and
``v_y < safe_min_vertical_speed``.  Its closed extension is one axis-aligned box
in the eight-dimensional observation: the two coordinates the shield reads are
bounded by the band, and the remaining six span the environment's declared
observation space.  Certifying that single box therefore certifies the whole
safety-critical set, exactly as the MountainCar critical box does in two
dimensions.

``--band-splits`` optionally subdivides the box along the free coordinates.
Splitting costs nothing in coverage (the pieces tile the same set) and tightens
the interval bounds when a single box proves too loose for IBP.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import TensorDataset

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())

from continuous_state_shields import (  # noqa: E402
    LunarLanderDescentShield,
    LunarLanderDescentShieldConfig,
)

from projects.safe_policy_optimisation.utils.io import write_json  # noqa: E402
from projects.safe_policy_optimisation.utils.log import log_info  # noqa: E402

ENV_ID = "LunarLander-v3"
ALTITUDE_INDEX = 1
VERTICAL_SPEED_INDEX = 3
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "continuous_state_shields" / "synthesised" / "lunarlander_descent"
)


def observation_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return the environment's declared finite observation bounds."""

    env = gym.make(ENV_ID, continuous=False)
    try:
        low = np.asarray(env.observation_space.low, dtype=np.float64).copy()
        high = np.asarray(env.observation_space.high, dtype=np.float64).copy()
    finally:
        env.close()
    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        raise ValueError(
            "The certificate requires finite observation bounds; "
            f"got low={low.tolist()} high={high.tolist()}."
        )
    return low, high


def descent_band_box(
    config: LunarLanderDescentShieldConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Close the open descent band into one conservative interval box."""

    low, high = observation_bounds()
    box_low, box_high = low.copy(), high.copy()
    # The open conditions y < h and v_y < v are covered by their closed
    # extensions, so the certificate is a superset of the runtime shield.
    box_high[ALTITUDE_INDEX] = float(config.critical_height)
    box_high[VERTICAL_SPEED_INDEX] = float(config.safe_min_vertical_speed)
    if np.any(box_low > box_high):
        raise ValueError(
            "The descent band is empty under the declared observation bounds: "
            f"low={box_low.tolist()} high={box_high.tolist()}."
        )
    return box_low, box_high


def split_box(
    box_low: np.ndarray,
    box_high: np.ndarray,
    splits: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Tile one box into a grid by subdividing the requested coordinates."""

    edges: list[list[tuple[float, float]]] = []
    for index in range(box_low.shape[0]):
        count = max(1, int(splits.get(index, 1)))
        cuts = np.linspace(box_low[index], box_high[index], count + 1)
        edges.append([(float(cuts[i]), float(cuts[i + 1])) for i in range(count)])
    lows, highs = [], []
    for corner in itertools.product(*edges):
        lows.append([pair[0] for pair in corner])
        highs.append([pair[1] for pair in corner])
    return np.asarray(lows, dtype=np.float64), np.asarray(highs, dtype=np.float64)


def parse_splits(values: Sequence[str] | None) -> dict[int, int]:
    """Parse ``INDEX=COUNT`` grid subdivisions of the free coordinates."""

    splits: dict[int, int] = {}
    for item in values or ():
        if "=" not in item:
            raise ValueError(f"--band-splits entries must be INDEX=COUNT; got {item!r}.")
        raw_index, raw_count = item.split("=", 1)
        index, count = int(raw_index), int(raw_count)
        if not 0 <= index < 8:
            raise ValueError(f"Observation index must lie in [0, 8); got {index}.")
        if count < 1:
            raise ValueError(f"Split count must be positive; got {count}.")
        splits[index] = count
    return splits


def verify_box_matches_shield(
    box_low: np.ndarray,
    box_high: np.ndarray,
    shield: LunarLanderDescentShield,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    """Sample the box and confirm the runtime shield agrees with its mask.

    Every sampled point that the runtime shield calls critical must admit only
    the main engine.  Points the shield leaves unconstrained are the closure's
    conservative surplus, which the certificate is allowed to over-cover.
    """

    rng = np.random.default_rng(seed)
    box_index = rng.integers(0, box_low.shape[0], size=samples)
    unit = rng.random((samples, box_low.shape[1]))
    points = box_low[box_index] + unit * (box_high[box_index] - box_low[box_index])
    critical = 0
    for point in points:
        safe_actions = shield.get_safe_actions(point)
        if shield.in_critical_band(point):
            critical += 1
            if safe_actions != [int(shield.config.main_engine_action)]:
                raise RuntimeError(
                    f"Shield admits {safe_actions} inside the certified box at "
                    f"{point.tolist()}; the certificate mask expects only the main engine."
                )
    return {
        "samples": int(samples),
        "critical_samples": int(critical),
        "critical_fraction": float(critical / samples) if samples else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--critical-height", type=float, default=0.6)
    parser.add_argument("--safe-min-vertical-speed", type=float, default=-0.35)
    parser.add_argument("--main-engine-action", type=int, default=2)
    parser.add_argument(
        "--band-splits",
        nargs="*",
        default=None,
        metavar="INDEX=COUNT",
        help="Subdivide observation coordinates, e.g. '0=2 4=2' for x and theta.",
    )
    parser.add_argument("--check-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    config = LunarLanderDescentShieldConfig(
        critical_height=float(args.critical_height),
        safe_min_vertical_speed=float(args.safe_min_vertical_speed),
        main_engine_action=int(args.main_engine_action),
    )
    shield = LunarLanderDescentShield(config)
    box_low, box_high = descent_band_box(config)
    splits = parse_splits(args.band_splits)
    lows, highs = split_box(box_low, box_high, splits)

    safe_mask = np.zeros((lows.shape[0], config.n_actions), dtype=bool)
    safe_mask[:, int(config.main_engine_action)] = True
    check = verify_box_matches_shield(
        lows, highs, shield, samples=int(args.check_samples), seed=int(args.seed)
    )

    run_id = args.run_id or "descent_band"
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset = TensorDataset(
        torch.as_tensor(lows, dtype=torch.float32),
        torch.as_tensor(highs, dtype=torch.float32),
        torch.as_tensor(safe_mask, dtype=torch.float32),
    )
    dataset_path = run_dir / "critical_interval_dataset.pt"
    torch.save(dataset, dataset_path)

    summary = {
        "env_id": ENV_ID,
        "shield": "lunarlander-descent",
        "shield_config": {
            "critical_height": float(config.critical_height),
            "safe_min_vertical_speed": float(config.safe_min_vertical_speed),
            "main_engine_action": int(config.main_engine_action),
            "n_actions": int(config.n_actions),
            "tolerance": float(config.tolerance),
        },
        "critical_interval_dataset_path": str(dataset_path.resolve()),
        "boxes": int(lows.shape[0]),
        "band_splits": {str(k): int(v) for k, v in splits.items()},
        "box_low": box_low.tolist(),
        "box_high": box_high.tolist(),
        "runtime_shield_agreement": check,
    }
    write_json(run_dir / "summary.json", summary)
    log_info(
        f"Wrote {lows.shape[0]} certified descent box(es) to {dataset_path} "
        f"({check['critical_samples']}/{check['samples']} sampled points inside the "
        "open runtime band)."
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
