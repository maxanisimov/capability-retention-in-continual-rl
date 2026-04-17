"""Evaluate a trained LunarLander policy and save metrics + trajectory frames."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.pipelines.lunarlander.train_source_policy import (
    _load_task_settings,
    _make_lunarlander_env,
    _plot_trajectory_grid,
    _resolve_lunarlander_dynamics,
    build_actor_critic,
)
from experiments.utils.ppo_utils import evaluate


POLICY_TO_SUBDIR = {
    "source": "source",
    "downstream_unconstrained": "downstream_unconstrained",
    "downstream_ewc": "downstream_ewc",
    "downstream_rashomon": "downstream_rashomon",
}


def _resolve_actor_path(policy_dir: Path, policy_name: str) -> Path:
    if policy_name == "downstream_ewc":
        candidates = [policy_dir / "ewc_actor.pt", policy_dir / "actor.pt"]
    elif policy_name == "downstream_rashomon":
        candidates = [policy_dir / "rashomon_actor.pt", policy_dir / "actor.pt"]
    else:
        candidates = [policy_dir / "actor.pt"]

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find actor checkpoint for policy '{policy_name}' in {policy_dir}. "
        f"Tried: {[str(c) for c in candidates]}",
    )


def _build_actor_from_state_dict(state_dict: dict[str, torch.Tensor]) -> torch.nn.Sequential:
    required = ("0.weight", "2.weight", "4.weight")
    missing = [k for k in required if k not in state_dict]
    if missing:
        raise ValueError(
            f"Unsupported actor checkpoint format. Missing keys: {missing}. "
            "Expected a 3-layer MLP Sequential actor.",
        )

    first_w = state_dict["0.weight"]
    second_w = state_dict["2.weight"]
    third_w = state_dict["4.weight"]
    if first_w.ndim != 2 or second_w.ndim != 2 or third_w.ndim != 2:
        raise ValueError("Expected 2D linear weight tensors for actor checkpoint.")

    obs_dim = int(first_w.shape[1])
    hidden_1 = int(first_w.shape[0])
    hidden_2 = int(second_w.shape[0])
    if hidden_1 != hidden_2:
        raise ValueError(
            f"Unsupported actor architecture with different hidden sizes ({hidden_1} vs {hidden_2}).",
        )
    n_actions = int(third_w.shape[0])

    actor, _ = build_actor_critic(obs_dim=obs_dim, n_actions=n_actions, hidden_size=hidden_1)
    actor.load_state_dict(state_dict, strict=True)
    return actor


def _sanitize(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "-" for ch in s)


def _resolve_policy_dir(
    outputs_root: Path,
    train_task_setting: str,
    train_seed: int,
    policy_subdir: str,
) -> Path:
    """Prefer new layout; fall back to legacy layout if needed."""
    preferred = outputs_root / train_task_setting / f"seed_{train_seed}" / policy_subdir
    legacy = outputs_root / f"seed_{train_seed}" / policy_subdir
    if preferred.exists() or not legacy.exists():
        return preferred
    return legacy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained LunarLander policy.")
    parser.add_argument("--train-seed", type=int, required=True, help="Seed used during policy training.")
    parser.add_argument(
        "--policy-name",
        type=str,
        required=True,
        choices=sorted(POLICY_TO_SUBDIR.keys()),
        help="Policy checkpoint group to evaluate.",
    )
    parser.add_argument(
        "--env-setting",
        type=str,
        required=True,
        help="Environment configuration key from LunarLander task settings.",
    )
    parser.add_argument(
        "--train-task-setting",
        type=str,
        default=None,
        help="Task setting key used when training the policy. Defaults to --env-setting.",
    )
    parser.add_argument(
        "--eval-episodes-post-training",
        type=int,
        default=100,
        help="Number of episodes for post-training policy evaluation.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Deprecated alias for --eval-episodes-post-training.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=0,
        help="Evaluation seed used for trajectory rendering episodes.",
    )
    parser.add_argument(
        "--env-role",
        type=str,
        choices=["source", "downstream"],
        default=None,
        help="Task role to evaluate on. Defaults to source for source policy, downstream otherwise.",
    )
    parser.add_argument(
        "--trajectory-max-frames-per-episode",
        type=int,
        default=5,
        help="Maximum frames shown per episode row (includes first and last frames).",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Evaluation device.")
    parser.add_argument(
        "--task-settings-file",
        type=Path,
        default=Path(__file__).resolve().parent / "settings" / "task_settings.yaml",
        help="Task settings YAML with source/downstream environment configs.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Root outputs directory containing <task_setting>/seed_<N> policy folders.",
    )
    args = parser.parse_args()

    eval_episodes_post_training = (
        int(args.eval_episodes)
        if args.eval_episodes is not None
        else int(args.eval_episodes_post_training)
    )
    if eval_episodes_post_training <= 0:
        raise ValueError(
            f"--eval-episodes-post-training must be > 0, got {eval_episodes_post_training}.",
        )

    env_role = args.env_role
    if env_role is None:
        env_role = "source" if args.policy_name == "source" else "downstream"
    train_task_setting = args.train_task_setting or args.env_setting

    policy_subdir = POLICY_TO_SUBDIR[args.policy_name]
    policy_dir = _resolve_policy_dir(
        args.outputs_root,
        train_task_setting,
        int(args.train_seed),
        policy_subdir,
    )
    if not policy_dir.exists():
        raise FileNotFoundError(f"Policy directory does not exist: {policy_dir}")

    actor_path = _resolve_actor_path(policy_dir, args.policy_name)
    actor_state = torch.load(actor_path, map_location="cpu")
    if not isinstance(actor_state, dict):
        raise ValueError(f"Expected actor checkpoint state_dict dict at {actor_path}.")

    actor = _build_actor_from_state_dict(actor_state).to(args.device)
    actor.eval()

    env_cfg = _load_task_settings(args.task_settings_file, args.env_setting, env_role)
    env_id = str(env_cfg.get("env_id") or "LunarLander-v3")
    gravity_raw = env_cfg.get("gravity")
    gravity = None if gravity_raw is None else float(gravity_raw)
    task_id_default = 0.0 if env_role == "source" else 1.0
    task_id = float(env_cfg.get("task_id", task_id_default))
    append_task_id = bool(env_cfg.get("append_task_id", True))
    continuous = bool(env_cfg.get("continuous", False))
    if continuous:
        raise ValueError("Only discrete-action LunarLander is supported in this evaluator.")
    dynamics_cfg = _resolve_lunarlander_dynamics(
        env_cfg,
        cfg_name=f"task_settings[{args.env_setting}:{env_role}]",
    )
    env_kwargs = {
        "gravity": gravity,
        "task_id": task_id,
        "append_task_id": append_task_id,
        **dynamics_cfg,
    }

    eval_env = _make_lunarlander_env(env_id, render_mode=None, **env_kwargs)
    mean_reward, std_reward, _ = evaluate(
        env=eval_env,
        actor=actor,
        episodes=eval_episodes_post_training,
        deterministic=True,
        device=args.device,
    )
    eval_env.close()

    actor_for_plot = copy.deepcopy(actor).to("cpu")
    actor_for_plot.eval()

    eval_dir = policy_dir / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)
    file_stem = (
        f"eval_env-{_sanitize(args.env_setting)}"
        f"_role-{env_role}"
        f"_seed-{int(args.eval_seed)}"
        f"_episodes-{eval_episodes_post_training}"
    )
    metrics_path = eval_dir / f"{file_stem}.yaml"
    frames_path = eval_dir / f"{file_stem}.png"

    _plot_trajectory_grid(
        env_id=env_id,
        gravity=gravity,
        task_id=task_id,
        append_task_id=append_task_id,
        dynamics_cfg=dynamics_cfg,
        actor=actor_for_plot,
        seed=int(args.eval_seed),
        device="cpu",
        output_path=frames_path,
        episodes=eval_episodes_post_training,
        max_frames_per_episode=int(args.trajectory_max_frames_per_episode),
    )

    summary: dict[str, Any] = {
        "train_seed": int(args.train_seed),
        "policy_name": str(args.policy_name),
        "policy_dir": str(policy_dir),
        "actor_path": str(actor_path),
        "task_settings_file": str(args.task_settings_file),
        "train_task_setting": str(train_task_setting),
        "env_setting": str(args.env_setting),
        "env_role": str(env_role),
        "env_id": env_id,
        "task_id": float(task_id),
        "gravity": gravity,
        "append_task_id": bool(append_task_id),
        "dynamics": dynamics_cfg,
        "eval_seed": int(args.eval_seed),
        "eval_episodes_post_training": int(eval_episodes_post_training),
        "trajectory_max_frames_per_episode": int(args.trajectory_max_frames_per_episode),
        "total_reward_mean": float(mean_reward),
        "total_reward_std": float(std_reward),
        "frames_figure_path": str(frames_path),
    }
    metrics_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    print(f"Evaluation done: mean={mean_reward:.3f}, std={std_reward:.3f}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved frames figure: {frames_path}")


if __name__ == "__main__":
    main()
