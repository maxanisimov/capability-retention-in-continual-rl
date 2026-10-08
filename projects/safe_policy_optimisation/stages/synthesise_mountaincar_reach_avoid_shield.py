"""Synthesize and validate a dynamics-aware MountainCar reach-avoid shield."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

from continuous_state_shields import (  # noqa: E402
    MountainCarReachAvoidShield,
    save_reach_avoid_grid,
    synthesise_reach_avoid_grid,
)

from projects.safe_policy_optimisation.utils.io import write_json  # noqa: E402

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "continuous_state_shields"
    / "synthesised"
    / "mountaincar_reach_avoid"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_controller(
    shield: MountainCarReachAvoidShield,
    *,
    controller: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate safety and goal reachability under a simple proposal policy."""

    if controller not in {
        "rank",
        "momentum",
        "momentum_unshielded",
        "random",
        "push_left",
    }:
        raise ValueError(f"Unknown controller: {controller}.")
    env = gym.make("MountainCar-v0")
    rng = np.random.default_rng(seed)
    goals = 0
    boundary_hits = 0
    interventions = 0
    total_steps = 0
    rewards: list[float] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + 10_000 + episode)
            done = False
            hit_boundary = False
            total_reward = 0.0
            while not done:
                if controller == "rank":
                    proposed = shield.preferred_action(observation)
                elif controller in {"momentum", "momentum_unshielded"}:
                    proposed = 2 if float(observation[1]) >= 0.0 else 0
                elif controller == "push_left":
                    proposed = 0
                else:
                    proposed = int(rng.integers(3))
                action = (
                    proposed
                    if controller == "momentum_unshielded"
                    else shield.shield_action(observation, proposed)
                )
                interventions += int(action != proposed)
                observation, reward, terminated, truncated, _ = env.step(action)
                hit_boundary |= float(observation[0]) <= shield.config.min_position
                total_reward += float(reward)
                total_steps += 1
                done = bool(terminated or truncated)
            goals += int(terminated)
            boundary_hits += int(hit_boundary)
            rewards.append(total_reward)
    finally:
        env.close()
    return {
        "controller": controller,
        "shielded_execution": controller != "momentum_unshielded",
        "episodes": int(episodes),
        "goal_count": int(goals),
        "goal_rate": float(goals / episodes) if episodes else 0.0,
        "left_boundary_hit_count": int(boundary_hits),
        "safety_rate": float((episodes - boundary_hits) / episodes)
        if episodes
        else 0.0,
        "mean_total_reward": float(np.mean(rewards)) if rewards else 0.0,
        "total_steps": int(total_steps),
        "interventions": int(interventions),
        "intervention_rate": float(interventions / total_steps) if total_steps else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-positions", type=int, default=721)
    parser.add_argument("--n-velocities", type=int, default=281)
    parser.add_argument("--goal-horizon", type=int, default=200)
    parser.add_argument("--boundary-margin", type=float, default=0.0)
    parser.add_argument("--eval-episodes", type=int, default=100)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_seed_1")
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = synthesise_reach_avoid_grid(
        n_positions=int(args.n_positions),
        n_velocities=int(args.n_velocities),
        goal_horizon=int(args.goal_horizon),
        boundary_margin=float(args.boundary_margin),
    )
    artifact_path = run_dir / "mountaincar_reach_avoid_shield.npz"
    save_reach_avoid_grid(artifact_path, payload)
    shield = MountainCarReachAvoidShield.load(artifact_path)
    validations = {
        controller: evaluate_controller(
            shield,
            controller=controller,
            episodes=int(args.eval_episodes),
            seed=int(args.seed),
        )
        for controller in (
            "rank",
            "momentum",
            "momentum_unshielded",
            "push_left",
            "random",
        )
    }
    winning_mask = np.asarray(payload["winning"], dtype=bool)
    safe_action_counts = np.asarray(payload["action_mask"], dtype=bool).sum(axis=-1)
    permissiveness = {
        str(count): int(((safe_action_counts == count) & winning_mask).sum())
        for count in range(4)
    }
    config = {
        "environment": "MountainCar-v0",
        "seed": int(args.seed),
        "n_positions": int(args.n_positions),
        "n_velocities": int(args.n_velocities),
        "goal_horizon": int(args.goal_horizon),
        "boundary_margin": float(args.boundary_margin),
        "eval_episodes": int(args.eval_episodes),
    }
    summary = {
        "run_dir": str(run_dir.resolve()),
        "shield_artifact": str(artifact_path.resolve()),
        "shield_artifact_sha256": _sha256(artifact_path),
        "synthesis": payload["metadata"],
        "winning_state_action_permissiveness": permissiveness,
        "validation": validations,
        "guarantee_scope": (
            "Finite-grid reach-avoid synthesis with conservative runtime cell "
            "membership; this is not yet a neural interval certificate."
        ),
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(build_parser().parse_args(argv))
    print(f"Stored shield at {summary['shield_artifact']}")
    for name, result in summary["validation"].items():
        print(
            f"{name}: safety={result['safety_rate']:.1%}, "
            f"goal={result['goal_rate']:.1%}, reward={result['mean_total_reward']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
