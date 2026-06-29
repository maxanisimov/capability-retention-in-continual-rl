"""Roll out a trained safe-policy model and save episode GIFs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
for import_path in (REPO_ROOT, REPO_ROOT / "core"):
    path_str = str(import_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from projects.safe_policy_optimisation.utils.minipacman_safe_rl import (  # noqa: E402
    ALGORITHM_NAMES,
    EpisodeMetrics,
    aggregate_violations,
    episode_rows,
    load_checkpoint_model,
    make_safe_rl_env,
    minipacman_state_cost,
    rollout_policy_frames,
    save_gif,
    write_episode_csv,
    write_json,
)
from projects.safe_policy_optimisation.stages.train_discrete_shielded_policy import (  # noqa: E402
    ALGORITHM_NAME as SHIELDED_ALGORITHM_NAME,
    load_shield_mask,
    make_unshielded_env,
    validate_shield_for_env,
)
from projects.safe_policy_optimisation.stages.train_masa_shielded_policy import (  # noqa: E402
    ALGORITHM_NAME as MASA_SHIELDED_ALGORITHM_NAME,
    make_masa_shielded_env,
)
from projects.safe_policy_optimisation.stages.train_rashomon_shielded_policy import (  # noqa: E402
    ALGORITHM_NAME as RASHOMON_SHIELDED_ALGORITHM_NAME,
)
from provably_safe_policy_optimisation import ProvablySafePPO, Shield  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402


PROVABLY_SAFE_MODEL_ZIP_ALGORITHMS = {
    SHIELDED_ALGORITHM_NAME,
    RASHOMON_SHIELDED_ALGORITHM_NAME,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Roll out a trained safe-policy model and save animated GIFs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Path to a saved <algorithm>.pt checkpoint.")
    source.add_argument(
        "--run-dir",
        type=Path,
        help=(
            "Training artifact run directory. For safe-RL baselines this contains "
            "<algorithm>.pt; for SB3 runs this contains model.zip and config.json. "
            "May also point directly to model.zip."
        ),
    )
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHM_NAMES,
        default=None,
        help=(
            "Safe-RL baseline checkpoint to load when --run-dir contains <algorithm>.pt. "
            "If omitted for a baseline run directory, all available baseline checkpoints are rolled out."
        ),
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of rollout episodes/GIFs to save.")
    parser.add_argument("--seed", type=int, default=0, help="Base rollout seed.")
    parser.add_argument("--fps", type=float, default=4.0, help="GIF frames per second.")
    parser.add_argument("--device", default="cpu", help="Torch device used to reconstruct the policy.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="GIF output directory. Defaults to <checkpoint-parent>/rollouts.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy instead of using deterministic actions.",
    )
    return parser


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.algorithm is None:
        raise ValueError("--algorithm is required when using --run-dir.")
    return args.run_dir / f"{args.algorithm}.pt"


def _resolve_baseline_checkpoints(args: argparse.Namespace) -> list[Path]:
    """Resolve safe-RL baseline .pt checkpoints from CLI arguments."""

    if args.checkpoint is not None:
        return [args.checkpoint]
    if args.run_dir is None:
        raise ValueError("Either --checkpoint or --run-dir is required.")
    if args.run_dir.suffix == ".pt":
        return [args.run_dir]
    if args.algorithm is not None:
        return [args.run_dir / f"{args.algorithm}.pt"]

    checkpoint_paths = [args.run_dir / f"{algorithm}.pt" for algorithm in ALGORITHM_NAMES]
    existing = [path for path in checkpoint_paths if path.exists()]
    if not existing:
        expected = ", ".join(path.name for path in checkpoint_paths)
        raise FileNotFoundError(f"No baseline checkpoints found in {args.run_dir}; expected one of: {expected}")
    return existing


def _normalise_run_dir(run_dir_or_model: Path) -> Path:
    return run_dir_or_model.parent if run_dir_or_model.name == "model.zip" else run_dir_or_model


def _read_run_config(run_dir_or_model: Path) -> dict[str, Any] | None:
    run_dir = _normalise_run_dir(run_dir_or_model)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text(encoding="utf-8"))


def _is_shielded_run_dir(run_dir: Path) -> bool:
    config = _read_run_config(run_dir)
    if config is None:
        return False
    normalised = _normalise_run_dir(run_dir)
    return (
        config.get("algorithm") in PROVABLY_SAFE_MODEL_ZIP_ALGORITHMS
        and (normalised / "model.zip").exists()
    )


def _is_masa_shielded_run_dir(run_dir: Path) -> bool:
    config = _read_run_config(run_dir)
    if config is None:
        return False
    normalised = _normalise_run_dir(run_dir)
    return config.get("algorithm") == MASA_SHIELDED_ALGORITHM_NAME and (normalised / "model.zip").exists()


def _resolve_artifact_path(path: str | Path, *, run_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    repo_relative = REPO_ROOT / candidate
    if repo_relative.exists():
        return repo_relative
    run_relative = run_dir / candidate
    if run_relative.exists():
        return run_relative
    return candidate


def _state_cost(env: Any, obs: Any, info: dict[str, Any]) -> float:
    if "cost" in info:
        return float(info["cost"])
    try:
        return minipacman_state_cost(env, obs)
    except Exception:
        return 0.0


def _rollout_shielded_policy_frames(
    model: ProvablySafePPO,
    shield: Shield,
    env: Any,
    *,
    seed: int,
    deterministic: bool,
) -> tuple[list[np.ndarray], EpisodeMetrics]:
    obs, _ = env.reset(seed=seed)
    frames = [np.asarray(env.render())]
    done = False
    total_reward = 0.0
    total_cost = 0.0
    length = 0

    while not done:
        proposed, _ = model.predict(obs, deterministic=deterministic)
        state = shield.obs_to_state(np.asarray([obs]))
        executed = shield.override(state, np.asarray([int(np.asarray(proposed).item())]))
        obs, reward, terminated, truncated, info = env.step(int(executed[0]))
        frames.append(np.asarray(env.render()))
        total_reward += float(reward)
        total_cost += _state_cost(env, obs, dict(info))
        length += 1
        done = bool(terminated or truncated)

    return frames, EpisodeMetrics(
        episode=0,
        reward=total_reward,
        cost=total_cost,
        length=length,
        violated=False,
    )


def _run_shielded_policy(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_dir is None:
        raise ValueError("Shielded PPO GIF rollout currently expects --run-dir.")
    run_dir = _normalise_run_dir(args.run_dir)
    config_path = run_dir / "config.json"
    model_path = run_dir / "model.zip"
    if not config_path.exists():
        raise FileNotFoundError(f"Shielded run config not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Shielded model not found: {model_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    algorithm = str(config.get("algorithm", SHIELDED_ALGORITHM_NAME))
    shield_path = _resolve_artifact_path(config["shield_path"], run_dir=run_dir)
    mask = load_shield_mask(
        shield_path,
        shield_key=str(config.get("shield_key", "shield")),
        source=str(config.get("shield_source", "shield")),
        risk_threshold=config.get("risk_threshold"),
    )
    env_id = str(config["env_id"])
    env_kwargs = dict(config.get("env_kwargs", {}))
    max_episode_steps = config.get("max_episode_steps")
    cost_limit = float(config.get("cost_limit", 0.0))
    output_dir = args.output_dir or (run_dir / "rollouts")

    bootstrap_env = make_unshielded_env(
        env_id,
        env_kwargs=env_kwargs,
        max_episode_steps=max_episode_steps,
        cost_limit=cost_limit,
        record_episodes=False,
    )
    validate_shield_for_env(mask, bootstrap_env)
    try:
        model = ProvablySafePPO.load(model_path, env=bootstrap_env, device=args.device)
        model.set_shield(mask, seed=int(args.seed))
    finally:
        bootstrap_env.close()

    episode_metrics = []
    csv_rows = []
    shield = Shield(mask, seed=int(args.seed))
    for episode in range(args.episodes):
        env = make_unshielded_env(
            env_id,
            env_kwargs=env_kwargs,
            max_episode_steps=max_episode_steps,
            cost_limit=cost_limit,
            record_episodes=False,
            render_mode="rgb_array",
        )
        try:
            frames, metrics = _rollout_shielded_policy_frames(
                model,
                shield,
                env,
                seed=int(args.seed) + episode,
                deterministic=not args.stochastic,
            )
        finally:
            env.close()

        metrics = EpisodeMetrics(
            episode=episode,
            reward=metrics.reward,
            cost=metrics.cost,
            length=metrics.length,
            violated=metrics.cost > cost_limit,
        )
        gif_path = output_dir / f"{algorithm}_episode_{episode:03d}.gif"
        save_gif(gif_path, frames, fps=args.fps)
        episode_metrics.append(metrics)
        csv_rows.extend(episode_rows(algorithm, [metrics]))
        print(
            "[{algorithm}] episode {episode}: reward={reward:.2f} cost={cost:.2f} "
            "violated={violated} gif={gif}".format(
                algorithm=algorithm,
                episode=episode,
                reward=metrics.reward,
                cost=metrics.cost,
                violated=metrics.violated,
                gif=gif_path,
            )
        )

    summary = {
        "algorithm": algorithm,
        "model": str(model_path),
        "shield_path": str(shield_path),
        "output_dir": str(output_dir),
        "cost_limit": cost_limit,
        "deterministic": not args.stochastic,
        "shield_diagnostics": shield.diagnostics(),
        **aggregate_violations(episode_metrics),
    }
    write_json(output_dir / f"{algorithm}_rollout_summary.json", summary)
    write_episode_csv(output_dir / f"{algorithm}_rollout_episodes.csv", csv_rows)
    return summary


def _rollout_masa_shielded_policy_frames(
    model: PPO,
    env: Any,
    *,
    seed: int,
    deterministic: bool,
) -> tuple[list[np.ndarray], EpisodeMetrics]:
    obs, _ = env.reset(seed=seed)
    frames = [np.asarray(env.render())]
    done = False
    total_reward = 0.0
    total_cost = 0.0
    length = 0

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(np.asarray(env.render()))
        total_reward += float(reward)
        total_cost += _state_cost(env, obs, dict(info))
        length += 1
        done = bool(terminated or truncated)

    return frames, EpisodeMetrics(
        episode=0,
        reward=total_reward,
        cost=total_cost,
        length=length,
        violated=False,
    )


def _run_masa_shielded_policy(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_dir is None:
        raise ValueError("MASA-shielded PPO GIF rollout expects --run-dir.")
    run_dir = _normalise_run_dir(args.run_dir)
    config_path = run_dir / "config.json"
    model_path = run_dir / "model.zip"
    if not config_path.exists():
        raise FileNotFoundError(f"MASA-shielded run config not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"MASA-shielded model not found: {model_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir or (run_dir / "rollouts")
    cost_limit = float(config.get("cost_limit", 0.0))
    env_id = str(config.get("env_id", config.get("env", "CustomMiniPacman-v0")))
    env_kwargs = dict(config.get("env_kwargs", {}))
    if not env_kwargs and env_id == "CustomMiniPacman-v0":
        env_kwargs = {"ghost_rand_prob": float(config.get("ghost_rand_prob", 0.0))}

    bootstrap_env = make_masa_shielded_env(
        env_id,
        max_episode_steps=int(config.get("max_episode_steps", 100)),
        env_kwargs=env_kwargs,
        safety_tolerance=float(config.get("safety_tolerance", 0.0)),
        theta=float(config.get("theta", 1e-10)),
        max_vi_steps=int(config.get("max_vi_steps", 1000)),
        granularity=int(config.get("granularity", 20)),
        cost_limit=cost_limit,
        record_episodes=False,
    )
    try:
        model = PPO.load(model_path, env=bootstrap_env, device=args.device)
    finally:
        bootstrap_env.close()

    episode_metrics = []
    csv_rows = []
    for episode in range(args.episodes):
        env = make_masa_shielded_env(
            env_id,
            max_episode_steps=int(config.get("max_episode_steps", 100)),
            env_kwargs=env_kwargs,
            safety_tolerance=float(config.get("safety_tolerance", 0.0)),
            theta=float(config.get("theta", 1e-10)),
            max_vi_steps=int(config.get("max_vi_steps", 1000)),
            granularity=int(config.get("granularity", 20)),
            cost_limit=cost_limit,
            record_episodes=False,
            render_mode="rgb_array",
        )
        try:
            frames, metrics = _rollout_masa_shielded_policy_frames(
                model,
                env,
                seed=int(args.seed) + episode,
                deterministic=not args.stochastic,
            )
        finally:
            env.close()

        metrics = EpisodeMetrics(
            episode=episode,
            reward=metrics.reward,
            cost=metrics.cost,
            length=metrics.length,
            violated=metrics.cost > cost_limit,
        )
        gif_path = output_dir / f"{MASA_SHIELDED_ALGORITHM_NAME}_episode_{episode:03d}.gif"
        save_gif(gif_path, frames, fps=args.fps)
        episode_metrics.append(metrics)
        csv_rows.extend(episode_rows(MASA_SHIELDED_ALGORITHM_NAME, [metrics]))
        print(
            "[{algorithm}] episode {episode}: reward={reward:.2f} cost={cost:.2f} "
            "violated={violated} gif={gif}".format(
                algorithm=MASA_SHIELDED_ALGORITHM_NAME,
                episode=episode,
                reward=metrics.reward,
                cost=metrics.cost,
                violated=metrics.violated,
                gif=gif_path,
            )
        )

    summary = {
        "algorithm": MASA_SHIELDED_ALGORITHM_NAME,
        "model": str(model_path),
        "env_id": env_id,
        "env_kwargs": env_kwargs,
        "output_dir": str(output_dir),
        "cost_limit": cost_limit,
        "deterministic": not args.stochastic,
        **aggregate_violations(episode_metrics),
    }
    write_json(output_dir / f"{MASA_SHIELDED_ALGORITHM_NAME}_rollout_summary.json", summary)
    write_episode_csv(output_dir / f"{MASA_SHIELDED_ALGORITHM_NAME}_rollout_episodes.csv", csv_rows)
    return summary


def _run_baseline_checkpoint(
    args: argparse.Namespace,
    checkpoint_path: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_dir = output_dir or args.output_dir or (checkpoint_path.parent / "rollouts")

    checkpoint_payload = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    metadata = dict(checkpoint_payload.get("metadata", {}))
    env_id = str(metadata.get("env_id", "CustomMiniPacman-v0"))
    env_kwargs = dict(metadata.get("env_kwargs", {}))
    if not env_kwargs and env_id == "CustomMiniPacman-v0":
        env_kwargs = {"ghost_rand_prob": float(metadata.get("ghost_rand_prob", 0.0))}
    max_episode_steps = metadata.get("max_episode_steps", 100)
    max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)

    bootstrap_env = make_safe_rl_env(
        env_id,
        max_episode_steps=max_episode_steps,
        env_kwargs=env_kwargs,
        render_mode="rgb_array",
    )
    try:
        model, checkpoint = load_checkpoint_model(checkpoint_path, env=bootstrap_env, device=args.device)
    finally:
        bootstrap_env.close()

    algorithm = str(checkpoint["algorithm"])
    cost_limit = float(metadata.get("cost_limit", 0.0))

    episode_metrics = []
    csv_rows = []
    for episode in range(args.episodes):
        env = make_safe_rl_env(
            env_id,
            max_episode_steps=max_episode_steps,
            env_kwargs=env_kwargs,
            render_mode="rgb_array",
        )
        try:
            frames, metrics = rollout_policy_frames(
                model,
                env,
                seed=int(args.seed) + episode,
                deterministic=not args.stochastic,
            )
        finally:
            env.close()

        metrics = type(metrics)(
            episode=episode,
            reward=metrics.reward,
            cost=metrics.cost,
            length=metrics.length,
            violated=metrics.cost > cost_limit,
        )
        gif_path = output_dir / f"{algorithm}_episode_{episode:03d}.gif"
        save_gif(gif_path, frames, fps=args.fps)
        episode_metrics.append(metrics)
        csv_rows.extend(episode_rows(algorithm, [metrics]))
        print(
            "[{algorithm}] episode {episode}: reward={reward:.2f} cost={cost:.2f} "
            "violated={violated} gif={gif}".format(
                algorithm=algorithm,
                episode=episode,
                reward=metrics.reward,
                cost=metrics.cost,
                violated=metrics.violated,
                gif=gif_path,
            )
        )

    summary = {
        "algorithm": algorithm,
        "checkpoint": str(checkpoint_path),
        "env_id": env_id,
        "env_kwargs": env_kwargs,
        "output_dir": str(output_dir),
        "cost_limit": cost_limit,
        "deterministic": not args.stochastic,
        **aggregate_violations(episode_metrics),
    }
    write_json(output_dir / f"{algorithm}_rollout_summary.json", summary)
    write_episode_csv(output_dir / f"{algorithm}_rollout_episodes.csv", csv_rows)
    return summary


def _run_baseline_policies(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_paths = _resolve_baseline_checkpoints(args)
    shared_output_dir = args.output_dir
    if shared_output_dir is None and args.run_dir is not None and args.run_dir.suffix != ".pt":
        shared_output_dir = args.run_dir / "rollouts"

    summaries = [
        _run_baseline_checkpoint(args, checkpoint_path, output_dir=shared_output_dir)
        for checkpoint_path in checkpoint_paths
    ]
    if len(summaries) == 1:
        return summaries[0]

    output_dir = shared_output_dir or (checkpoint_paths[0].parent / "rollouts")
    combined = {
        "algorithms": [summary["algorithm"] for summary in summaries],
        "checkpoints": [summary["checkpoint"] for summary in summaries],
        "output_dir": str(output_dir),
        "deterministic": not args.stochastic,
        "policies": summaries,
    }
    write_json(output_dir / "baseline_rollout_summary.json", combined)
    return combined


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_dir is not None and _is_shielded_run_dir(args.run_dir):
        return _run_shielded_policy(args)
    if args.run_dir is not None and _is_masa_shielded_run_dir(args.run_dir):
        return _run_masa_shielded_policy(args)
    return _run_baseline_policies(args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
