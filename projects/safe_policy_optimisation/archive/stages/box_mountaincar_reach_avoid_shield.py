"""Convert a synthesized MountainCar reach-avoid shield to PSPO boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import TensorDataset

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())

from continuous_state_shields import (  # noqa: E402
    MountainCarIntervalBoxShield,
    box_reach_avoid_grid,
    save_interval_box_shield,
)

from projects.safe_policy_optimisation.utils.io import write_json  # noqa: E402

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "continuous_state_shields"
    / "synthesised"
    / "mountaincar_reach_avoid_boxes"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_boxed_shield(
    shield: MountainCarIntervalBoxShield,
    *,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate the boxed shield with the standard momentum proposal policy."""

    env = gym.make("MountainCar-v0")
    rewards: list[float] = []
    goals = boundary_hits = interventions = steps = 0
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + 10_000 + episode)
            done = hit_boundary = False
            reward_sum = 0.0
            while not done:
                proposed = 2 if float(observation[1]) >= 0.0 else 0
                action = shield.shield_action(observation, proposed)
                interventions += int(action != proposed)
                observation, reward, terminated, truncated, _ = env.step(action)
                hit_boundary |= float(observation[0]) <= shield.config.min_position
                reward_sum += float(reward)
                steps += 1
                done = bool(terminated or truncated)
            goals += int(terminated)
            boundary_hits += int(hit_boundary)
            rewards.append(reward_sum)
    finally:
        env.close()
    return {
        "controller": "momentum_with_boxed_shield",
        "episodes": int(episodes),
        "goal_count": int(goals),
        "goal_rate": float(goals / episodes) if episodes else 0.0,
        "left_boundary_hit_count": int(boundary_hits),
        "safety_rate": float((episodes - boundary_hits) / episodes)
        if episodes
        else 0.0,
        "mean_total_reward": float(np.mean(rewards)) if rewards else 0.0,
        "interventions": int(interventions),
        "intervention_rate": float(interventions / steps) if steps else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-shield", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--velocity-bands", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source_shield).resolve()
    with np.load(source_path, allow_pickle=False) as source:
        source_metadata = json.loads(str(source["metadata_json"].item()))
        payload = box_reach_avoid_grid(
            positions=source["positions"],
            velocities=source["velocities"],
            action_mask=source["action_mask"],
            velocity_bands=int(args.velocity_bands),
            source_metadata=source_metadata,
        )
    payload["metadata"].update(
        {
            "source_shield": str(source_path),
            "source_shield_sha256": _sha256(source_path),
        }
    )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shield_path = run_dir / "mountaincar_reach_avoid_box_shield.npz"
    dataset_path = run_dir / "critical_interval_dataset.pt"
    save_interval_box_shield(shield_path, payload)
    torch.save(
        TensorDataset(
            torch.as_tensor(payload["box_lows"], dtype=torch.float32),
            torch.as_tensor(payload["box_highs"], dtype=torch.float32),
            torch.as_tensor(payload["safe_masks"], dtype=torch.float32),
        ),
        dataset_path,
    )
    shield = MountainCarIntervalBoxShield.load(shield_path)
    validation = evaluate_boxed_shield(
        shield, episodes=int(args.eval_episodes), seed=int(args.seed)
    )
    config = {
        "source_shield": str(source_path),
        "velocity_bands": int(args.velocity_bands),
        "eval_episodes": int(args.eval_episodes),
        "seed": int(args.seed),
    }
    summary = {
        "run_dir": str(run_dir.resolve()),
        "boxed_shield_artifact": str(shield_path.resolve()),
        "boxed_shield_sha256": _sha256(shield_path),
        "certificate_dataset": str(dataset_path.resolve()),
        "certificate_dataset_sha256": _sha256(dataset_path),
        "conversion": payload["metadata"],
        "validation": validation,
        "pspo_arguments": {
            "continuous_shield": "mountaincar-boxes",
            "continuous_shield_artifact": str(shield_path.resolve()),
            "certificate_dataset": str(dataset_path.resolve()),
        },
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(build_parser().parse_args(argv))
    print(f"Stored boxed shield at {summary['boxed_shield_artifact']}")
    print(f"Stored PSPO certificate at {summary['certificate_dataset']}")
    print(
        f"boxes={summary['conversion']['box_count']}, "
        f"safety={summary['validation']['safety_rate']:.1%}, "
        f"goal={summary['validation']['goal_rate']:.1%}, "
        f"reward={summary['validation']['mean_total_reward']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
