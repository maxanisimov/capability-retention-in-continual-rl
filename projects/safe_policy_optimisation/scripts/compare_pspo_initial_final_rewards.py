"""Re-evaluate region-first PSPO initial and final policies.

The comparison unit is a training seed.  For every completed seed, the shared
behaviour-cloned base policy and that seed's final region-first checkpoint are
evaluated unshielded with identical reset seeds.  Per-seed files are written as
jobs finish, making a stopped comparison safely resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = (
    REPO
    / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
    / "adaptive_v2_two_hidden_directional_replace_all_margin2_200iters/two_hidden"
)


@dataclass(frozen=True)
class SeedRun:
    environment: str
    seed: int
    run_dir: Path


class BasePolicyPredictor:
    """Give a saved BC network the small ``predict`` interface evaluators use."""

    def __init__(self, policy: torch.nn.Module, *, input_dim: int) -> None:
        self.policy = policy.eval()
        self.input_dim = int(input_dim)

    @torch.no_grad()
    def predict(self, observation: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        del deterministic
        array = np.asarray(observation)
        if array.ndim == 0 or (array.ndim == 1 and array.size == 1):
            state = int(array.reshape(-1)[0])
            inputs = torch.nn.functional.one_hot(
                torch.tensor(state), num_classes=self.input_dim
            ).to(torch.float32).unsqueeze(0)
        else:
            inputs = torch.as_tensor(array, dtype=torch.float32)
            if inputs.ndim == 1:
                inputs = inputs.unsqueeze(0)
        actions = self.policy(inputs).argmax(dim=-1).cpu().numpy()
        if actions.size == 1:
            return np.asarray(int(actions.item())), None
        return actions, None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_seed_runs(run_root: Path) -> list[SeedRun]:
    runs: list[SeedRun] = []
    for environment_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        for seed_dir in sorted(environment_dir.glob("seed*")):
            required = ("config.json", "metrics.json", "model.zip")
            if not all((seed_dir / name).is_file() for name in required):
                continue
            try:
                seed = int(seed_dir.name.removeprefix("seed"))
            except ValueError:
                continue
            config = _read_json(seed_dir / "config.json")
            algorithm = config.get("algorithm")
            adaptive = config.get("adaptive") or {}
            if algorithm != "adaptive_safe_ppo_v2" and not (
                algorithm in {"pspo", "pspo_adaptive"}
                and not adaptive.get("verify_first", False)
            ):
                continue
            runs.append(SeedRun(environment_dir.name, seed, seed_dir))
    return runs


def _load_base_policy(path: Path) -> BasePolicyPredictor:
    from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (
        build_base_policy,
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    architecture = dict(payload["architecture"])
    policy = build_base_policy(
        int(architecture["input_dim"]),
        int(architecture["n_actions"]),
        hidden_dim=int(architecture["hidden_dim"]),
        n_hidden=int(architecture["n_hidden"]),
    )
    policy.load_state_dict(payload["state_dict"], strict=True)
    return BasePolicyPredictor(policy, input_dim=int(architecture["input_dim"]))


def exhaustive_shield_alignment(
    model: Any,
    shield_mask: np.ndarray,
    *,
    input_dim: int,
    batch_size: int = 512,
) -> dict[str, int | float]:
    """Audit the greedy policy on every shield state that has a safe action."""

    valid_states = np.flatnonzero(np.asarray(shield_mask).any(axis=1))
    safe = 0
    for start in range(0, len(valid_states), batch_size):
        states = valid_states[start : start + batch_size]
        observation_space = getattr(model, "observation_space", None)
        if observation_space is not None and hasattr(observation_space, "n"):
            observations = states
        else:
            observations = np.zeros((len(states), input_dim), dtype=np.float32)
            observations[np.arange(len(states)), states] = 1.0
        actions, _ = model.predict(observations, deterministic=True)
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        safe += int(np.asarray(shield_mask)[states, actions].sum())
    checked = int(len(valid_states))
    return {
        "checked_states": checked,
        "unsafe_states": checked - safe,
        "alignment_rate": float(safe / checked) if checked else 0.0,
    }


def _evaluate(
    model: Any,
    config: dict[str, Any],
    shield_mask: np.ndarray,
    *,
    episodes: int,
    eval_seed: int,
) -> dict[str, Any]:
    from provably_safe_policy_optimisation import Shield

    from projects.safe_policy_optimisation.stages.train_ppo_shield import (
        evaluate_unshielded_policy,
        make_unshielded_env,
    )

    env = make_unshielded_env(
        config["env_id"],
        env_kwargs=dict(config.get("env_kwargs") or {}),
        max_episode_steps=int(config["max_episode_steps"]),
        cost_limit=float(config["cost_limit"]),
        record_episodes=True,
    )
    try:
        # The audit goes through the Shield interface, not the bare mask; the
        # policy still runs unshielded, nothing is overridden.
        records, action_safety = evaluate_unshielded_policy(
            model,
            env,
            Shield(
                shield_mask,
                obs_to_state=env.unwrapped.make_obs_to_state(),
                seed=eval_seed,
            ),
            episodes=episodes,
            seed=eval_seed,
        )
    finally:
        env.close()
    rewards = [float(record["reward"]) for record in records]
    safe_trajectories = [bool(record["safe_trajectory"]) for record in records]
    return {
        "rewards": rewards,
        "safe_trajectory_rate": float(mean(safe_trajectories)),
        **action_safety,
    }


def compare_seed(
    run: SeedRun,
    episodes_override: int | None,
    eval_seed_offset: int = 10_000,
) -> dict[str, Any]:
    from provably_safe_policy_optimisation import AdaptiveSafePPOV2

    from projects.safe_policy_optimisation.utils.shield import load_shield_mask

    config = _read_json(run.run_dir / "config.json")
    episodes = int(episodes_override or config["eval_episodes"])
    eval_seed = int(config["seed"]) + int(eval_seed_offset)
    base_path = Path(config["base_policy_path"])
    if not base_path.is_absolute():
        base_path = REPO / base_path
    shield_path = Path(config["shield_path"])
    if not shield_path.is_absolute():
        shield_path = REPO / shield_path
    shield_mask = load_shield_mask(shield_path)
    input_dim = int(config["base_policy_architecture"]["input_dim"])

    initial = _load_base_policy(base_path)
    final = AdaptiveSafePPOV2.load(run.run_dir / "model.zip", device="cpu")
    initial_evaluation = _evaluate(
        initial, config, shield_mask, episodes=episodes, eval_seed=eval_seed
    )
    final_evaluation = _evaluate(
        final, config, shield_mask, episodes=episodes, eval_seed=eval_seed
    )
    initial_exact = exhaustive_shield_alignment(
        initial, shield_mask, input_dim=input_dim
    )
    final_exact = exhaustive_shield_alignment(final, shield_mask, input_dim=input_dim)
    del final
    initial_rewards = initial_evaluation.pop("rewards")
    final_rewards = final_evaluation.pop("rewards")

    episode_rows = [
        {
            "episode": episode,
            "reset_seed": eval_seed + episode,
            "initial_total_reward": initial_reward,
            "final_total_reward": final_reward,
            "reward_delta": final_reward - initial_reward,
        }
        for episode, (initial_reward, final_reward) in enumerate(
            zip(initial_rewards, final_rewards, strict=True)
        )
    ]
    existing_final = _read_json(run.run_dir / "metrics.json")["reward"]["mean_total_reward"]
    return {
        "environment": run.environment,
        "seed": run.seed,
        "episodes": episodes,
        "eval_seed_offset": int(eval_seed_offset),
        "eval_seed_start": eval_seed,
        "initial_policy_path": str(base_path),
        "final_policy_path": str(run.run_dir / "model.zip"),
        "initial_mean_total_reward": mean(initial_rewards),
        "final_mean_total_reward": mean(final_rewards),
        "mean_reward_delta": mean(final_rewards) - mean(initial_rewards),
        "existing_final_mean_total_reward": float(existing_final),
        "final_reproduction_difference": mean(final_rewards) - float(existing_final),
        "initial_empirical_safety": initial_evaluation,
        "final_empirical_safety": final_evaluation,
        "initial_exhaustive_safety": initial_exact,
        "final_exhaustive_safety": final_exact,
        "episode_results": episode_rows,
    }


def _sem(values: list[float]) -> float:
    return stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    environments = sorted({str(result["environment"]) for result in results})
    for environment in environments:
        group = [result for result in results if result["environment"] == environment]
        initial = [float(result["initial_mean_total_reward"]) for result in group]
        final = [float(result["final_mean_total_reward"]) for result in group]
        deltas = [float(result["mean_reward_delta"]) for result in group]
        rows.append(
            {
                "environment": environment,
                "n_seeds": len(group),
                "eval_episodes_per_seed": int(group[0]["episodes"]),
                "initial_mean_total_reward": mean(initial),
                "initial_sem_across_seeds": _sem(initial),
                "final_mean_total_reward": mean(final),
                "final_sem_across_seeds": _sem(final),
                "paired_mean_reward_delta": mean(deltas),
                "paired_delta_sem_across_seeds": _sem(deltas),
                "final_empirical_safe_trajectory_rate": mean(
                    [float(result["final_empirical_safety"]["safe_trajectory_rate"]) for result in group]
                ),
                "final_empirical_unsafe_action_count": sum(
                    int(result["final_empirical_safety"]["unsafe_proposed_action_count"])
                    for result in group
                ),
                "final_exhaustive_alignment_rate": min(
                    float(result["final_exhaustive_safety"]["alignment_rate"])
                    for result in group
                ),
                "final_exhaustive_unsafe_states": sum(
                    int(result["final_exhaustive_safety"]["unsafe_states"])
                    for result in group
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, results: list[dict[str, Any]]) -> None:
    summary_rows = aggregate(results)
    seed_rows = [
        {key: value for key, value in result.items() if key != "episode_results"}
        for result in sorted(results, key=lambda item: (item["environment"], item["seed"]))
    ]
    episode_rows = [
        {"environment": result["environment"], "seed": result["seed"], **row}
        for result in sorted(results, key=lambda item: (item["environment"], item["seed"]))
        for row in result["episode_results"]
    ]
    (output_dir / "aggregate.json").write_text(
        json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "aggregate.csv", summary_rows)
    _write_csv(output_dir / "per_seed.csv", seed_rows)
    _write_csv(output_dir / "per_episode.csv", episode_rows)

    markdown = [
        "# Adaptive-v2 initial versus final total reward",
        "",
        "Unshielded deterministic evaluation on matched reset seeds. Values are mean ± SEM across training seeds.",
        "",
        "| Environment | Seeds | Initial reward | Final reward | Final − initial | Empirical safety | Exact shield alignment |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        markdown.append(
            "| {environment} | {n_seeds} | {initial_mean_total_reward:.4f} ± "
            "{initial_sem_across_seeds:.4f} | {final_mean_total_reward:.4f} ± "
            "{final_sem_across_seeds:.4f} | {paired_mean_reward_delta:+.4f} ± "
            "{paired_delta_sem_across_seeds:.4f} | "
            "{final_empirical_safe_trajectory_rate:.4f} | "
            "{final_exhaustive_alignment_rate:.4f} |".format(**row)
        )
    (output_dir / "README.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-seed-offset", type=int, default=10_000)
    parser.add_argument("--environments", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true", help="Recompute completed per-seed files.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.eval_episodes is not None and args.eval_episodes <= 0:
        raise SystemExit("--eval-episodes must be positive")
    if args.eval_seed_offset < 0:
        raise SystemExit("--eval-seed-offset must be non-negative")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    run_root = args.run_root.resolve()
    output_dir = (args.output_dir or run_root / "_initial_final_reward_comparison").resolve()
    per_seed_dir = output_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_seed_runs(run_root)
    if args.environments is not None:
        requested_environments = set(args.environments)
        runs = [run for run in runs if run.environment in requested_environments]
    if args.seeds is not None:
        requested_seeds = set(args.seeds)
        runs = [run for run in runs if run.seed in requested_seeds]
    if not runs:
        raise SystemExit(f"No completed region-first PSPO runs found under {run_root}")
    print(f"Discovered {len(runs)} completed seed runs under {run_root}", flush=True)

    results: list[dict[str, Any]] = []
    pending: list[SeedRun] = []
    for run in runs:
        result_path = per_seed_dir / run.environment / f"seed{run.seed}.json"
        if result_path.is_file() and not args.force:
            cached = _read_json(result_path)
            expected_episodes = args.eval_episodes or _read_json(run.run_dir / "config.json")["eval_episodes"]
            if (
                int(cached.get("episodes", -1)) == int(expected_episodes)
                and int(cached.get("eval_seed_offset", 10_000)) == args.eval_seed_offset
            ):
                results.append(cached)
                print(f"cached {run.environment}/seed{run.seed}", flush=True)
                continue
        pending.append(run)

    with ProcessPoolExecutor(max_workers=min(args.workers, len(pending) or 1)) as executor:
        futures = {
            executor.submit(
                compare_seed, run, args.eval_episodes, args.eval_seed_offset
            ): run
            for run in pending
        }
        for future in as_completed(futures):
            run = futures[future]
            result = future.result()
            result_path = per_seed_dir / run.environment / f"seed{run.seed}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            results.append(result)
            write_outputs(output_dir, results)
            print(
                f"finished {run.environment}/seed{run.seed}: "
                f"initial={result['initial_mean_total_reward']:.4f}, "
                f"final={result['final_mean_total_reward']:.4f}, "
                f"delta={result['mean_reward_delta']:+.4f}",
                flush=True,
            )

    write_outputs(output_dir, results)
    print(f"Comparison complete: {output_dir / 'README.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
