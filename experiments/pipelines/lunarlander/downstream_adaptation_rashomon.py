"""Adapt source LunarLander policy to downstream task via Rashomon-constrained PPO."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
from typing import Any

os.environ["SDL_AUDIODRIVER"] = "dummy"

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import TensorDataset
import yaml

# Allow running this file directly from experiments/pipelines/lunarlander.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.lunarlander.train_source_policy import (
    _load_task_settings,
    _make_lunarlander_env,
    _plot_trajectory_grid,
    _resolve_lunarlander_dynamics,
    build_actor_critic,
)
from experiments.utils.ppo_utils import PPOConfig, evaluate, ppo_train
from src.trainer.IntervalTrainer import IntervalTrainer


def _certificate_to_float(certificate: object) -> float:
    if certificate is None:
        return float("-inf")
    if isinstance(certificate, list):
        vals = [float(v) for v in certificate if v is not None]
        return min(vals) if vals else float("-inf")
    return float(certificate)  # type: ignore[arg-type]


def _load_source_hidden_size(source_run_dir: Path, arg_hidden_size: int | None) -> int:
    if arg_hidden_size is not None:
        return int(arg_hidden_size)
    summary_path = source_run_dir / "run_summary.yaml"
    if summary_path.exists():
        summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
        if isinstance(summary, dict):
            if summary.get("hidden_size") is not None:
                return int(summary["hidden_size"])
            run_settings = summary.get("run_settings")
            if isinstance(run_settings, dict) and run_settings.get("hidden_size") is not None:
                return int(run_settings["hidden_size"])
    return 256


def _seed_run_dir(outputs_root: Path, task_setting: str, seed: int) -> Path:
    return outputs_root / task_setting / f"seed_{seed}"


def _resolve_default_source_run_dir(outputs_root: Path, task_setting: str, seed: int) -> Path:
    """Prefer new layout; fall back to legacy layout if needed."""
    preferred = _seed_run_dir(outputs_root, task_setting, seed) / "source"
    legacy = outputs_root / f"seed_{seed}" / "source"
    if preferred.exists() or not legacy.exists():
        return preferred
    return legacy


def neutralize_task_feature(
    model: torch.nn.Sequential,
    task_feature_index: int,
    target_task_value: float,
) -> None:
    """Neutralize first-layer task feature contribution for target task value."""
    first = model[0]
    if not isinstance(first, torch.nn.Linear):
        raise ValueError("Expected first layer to be torch.nn.Linear for task-feature neutralization.")

    with torch.no_grad():
        w_task = first.weight[:, task_feature_index].clone()
        first.bias[:] = first.bias - w_task * target_task_value
        first.weight[:, task_feature_index] = 0.0


def create_source_rollout_rashomon_dataset(
    actor: torch.nn.Module,
    env,
    *,
    seed: int,
    n_actions: int,
    rashomon_rollouts: int,
) -> tuple[TensorDataset, list[int]]:
    """Roll out source policy `rashomon_rollouts` times and collect state-action pairs."""
    if rashomon_rollouts <= 0:
        raise ValueError(f"rashomon_rollouts must be > 0, got {rashomon_rollouts}.")

    actor_was_training = actor.training
    actor_device = next(actor.parameters()).device
    actor.eval()

    obs_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    rollout_lengths: list[int] = []

    try:
        for rollout_idx in range(rashomon_rollouts):
            obs, _ = env.reset(seed=seed + rollout_idx)
            done = False

            rollout_obs: list[np.ndarray] = []
            rollout_actions: list[int] = []

            while not done:
                obs_np = np.asarray(obs, dtype=np.float32).copy()
                rollout_obs.append(obs_np)

                obs_t = torch.from_numpy(obs_np).to(actor_device).unsqueeze(0)
                with torch.no_grad():
                    logits = actor(obs_t)
                    action = int(torch.argmax(logits, dim=1).item())
                rollout_actions.append(action)

                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            if not rollout_obs:
                continue

            rollout_obs_arr = np.asarray(rollout_obs, dtype=np.float32)
            rollout_labels_arr = np.zeros((len(rollout_actions), n_actions), dtype=np.float32)
            rollout_labels_arr[np.arange(len(rollout_actions)), np.asarray(rollout_actions, dtype=np.int64)] = 1.0

            obs_chunks.append(rollout_obs_arr)
            label_chunks.append(rollout_labels_arr)
            rollout_lengths.append(len(rollout_actions))
    finally:
        if actor_was_training:
            actor.train()

    if not obs_chunks:
        raise RuntimeError("No source rollouts were collected; cannot build Rashomon dataset.")

    obs_tensor = torch.from_numpy(np.concatenate(obs_chunks, axis=0)).float()
    label_tensor = torch.from_numpy(np.concatenate(label_chunks, axis=0)).float()
    return TensorDataset(obs_tensor, label_tensor), rollout_lengths


def compute_rashomon_bounds(
    *,
    actor: torch.nn.Module,
    rashomon_dataset: TensorDataset,
    seed: int,
    rashomon_n_iters: int,
    min_hard_spec: float,
    aggregation: str,
    inverse_temp_start: int,
    inverse_temp_max: int,
    checkpoint: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], object, int, float, list[float], int]:
    """Compute Rashomon parameter bounds for a state-action dataset."""
    if len(rashomon_dataset) == 0:
        raise RuntimeError("Rashomon dataset is empty.")

    n_valid_actions = rashomon_dataset.tensors[1].sum(dim=1).tolist()
    max_valid_actions = max(n_valid_actions) if n_valid_actions else 0.0
    if max_valid_actions <= 0:
        raise RuntimeError("Rashomon dataset has no valid-action labels.")

    surrogate_threshold = float(max_valid_actions / (1.0 + max_valid_actions))

    actor.eval()
    with torch.no_grad():
        all_obs = rashomon_dataset.tensors[0]
        action_mask = rashomon_dataset.tensors[1]
        logits = actor(all_obs)

        selected_inverse_temp: int | None = None
        min_action_mass = float("-inf")
        for inverse_temp in range(inverse_temp_start, inverse_temp_max + 1):
            probs = torch.softmax(logits * inverse_temp, dim=1)
            valid_action_mass = (probs * action_mask).sum(dim=1)
            min_action_mass = float(valid_action_mass.min().item())
            if min_action_mass >= surrogate_threshold:
                selected_inverse_temp = inverse_temp
                break

        if selected_inverse_temp is None:
            raise ValueError(
                "Could not find inverse temperature satisfying surrogate threshold. "
                f"Best min valid-action mass={min_action_mass:.6f} < threshold={surrogate_threshold:.6f}",
            )

    interval_trainer = IntervalTrainer(
        model=actor,
        min_acc_limit=surrogate_threshold,
        seed=seed,
        n_iters=rashomon_n_iters,  # type: ignore[arg-type]
        min_acc_increment=0,
        T=selected_inverse_temp,
        checkpoint=checkpoint,  # type: ignore[arg-type]
    )
    interval_trainer.compute_rashomon_set(
        dataset=rashomon_dataset,
        multi_label=True,
        aggregation=aggregation,  # type: ignore[arg-type]
    )

    cert_values = [_certificate_to_float(cert) for cert in interval_trainer.certificates]
    valid_indices = [i for i, cert in enumerate(cert_values) if cert >= min_hard_spec]
    if not valid_indices:
        best_cert = max(cert_values) if cert_values else float("-inf")
        raise ValueError(
            f"No Rashomon certificate satisfied min_hard_spec={min_hard_spec:.3f}. "
            f"Best certificate={best_cert:.6f}",
        )

    selected_idx = valid_indices[-1]
    bounded_model = interval_trainer.bounds[selected_idx]
    param_bounds_l = [p.detach().cpu() for p in bounded_model.param_l]
    param_bounds_u = [p.detach().cpu() for p in bounded_model.param_u]
    return (
        param_bounds_l,
        param_bounds_u,
        bounded_model,
        selected_inverse_temp,
        surrogate_threshold,
        cert_values,
        selected_idx,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run downstream LunarLander adaptation with rollout Rashomon bounds and PPO-PGD.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--task-settings-file",
        type=Path,
        default=Path(__file__).resolve().parent / "settings" / "task_settings.yaml",
    )
    parser.add_argument(
        "--task-setting",
        type=str,
        default="default",
    )
    parser.add_argument("--env-id", type=str, default=None, help="Optional env-id override.")
    parser.add_argument("--source-gravity", type=float, default=None, help="Optional source gravity override.")
    parser.add_argument("--downstream-gravity", type=float, default=None, help="Optional downstream gravity override.")
    parser.add_argument("--source-task-id", type=float, default=None, help="Optional source task-id override.")
    parser.add_argument("--downstream-task-id", type=float, default=None, help="Optional downstream task-id override.")
    parser.add_argument(
        "--append-task-id",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append task-id feature in observations (default inherited from task settings).",
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help="Optional explicit source checkpoint directory. Defaults to outputs/<task_setting>/seed_<seed>/source",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument(
        "--run-subdir",
        type=str,
        default="downstream_rashomon",
        help="Subdirectory under outputs/<task_setting>/seed_<seed>/ where outputs are saved.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Actor/critic hidden size. Defaults to source run summary hidden_size if available.",
    )
    parser.add_argument(
        "--warm-start-critic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm-start critic from source checkpoint.",
    )
    parser.add_argument(
        "--enable-task-neutralization",
        action="store_true",
        help="Enable first-layer task-feature neutralization before adaptation.",
    )
    parser.add_argument(
        "--disable-task-neutralization",
        action="store_true",
        help="Disable first-layer task-feature neutralization before adaptation.",
    )

    # PPO adaptation hyperparameters
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument(
        "--eval-episodes-during-training",
        type=int,
        default=20,
        help="Number of episodes per periodic evaluation during PPO training.",
    )
    parser.add_argument(
        "--eval-episodes-post-training",
        type=int,
        default=100,
        help="Number of episodes for final post-training evaluation.",
    )
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--early-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable early stopping on periodic evaluation (default: enabled).",
    )
    parser.add_argument("--early-stop-min-steps", type=int, default=0)
    parser.add_argument("--early-stop-reward-threshold", type=float, default=200.0)
    parser.add_argument("--early-stop-failure-rate-threshold", type=float, default=None)
    parser.add_argument("--early-stop-deterministic-total-reward-threshold", type=float, default=None)
    parser.add_argument("--early-stop-deterministic-eval-episodes", type=int, default=20)
    parser.add_argument(
        "--trajectory-episodes",
        type=int,
        default=5,
        help="Number of deterministic episodes visualized per trajectory figure.",
    )
    parser.add_argument(
        "--trajectory-max-frames-per-episode",
        type=int,
        default=5,
        help="Maximum frames shown per episode row (includes first and last frames).",
    )

    # Rashomon arguments
    parser.add_argument(
        "--rashomon-rollouts",
        type=int,
        default=100,
        help="Number of source-task rollouts used to build the Rashomon dataset.",
    )
    parser.add_argument("--rashomon-n-iters", type=int, default=50_000)
    parser.add_argument("--rashomon-min-hard-spec", type=float, default=1.0)
    parser.add_argument(
        "--rashomon-surrogate-aggregation",
        type=str,
        choices=["mean", "min"],
        default="min",
    )
    parser.add_argument("--inverse-temp-start", type=int, default=10)
    parser.add_argument("--inverse-temp-max", type=int, default=1000)
    parser.add_argument("--rashomon-checkpoint", type=int, default=100)
    args = parser.parse_args()

    if args.eval_episodes_during_training < 2:
        raise ValueError("For LunarLander, --eval-episodes-during-training must be >= 2.")
    if args.eval_episodes_post_training < 2:
        raise ValueError("For LunarLander, --eval-episodes-post-training must be >= 2.")
    if args.rashomon_rollouts <= 0:
        raise ValueError("--rashomon-rollouts must be > 0.")
    if args.inverse_temp_start <= 0 or args.inverse_temp_max < args.inverse_temp_start:
        raise ValueError(
            "Invalid inverse-temperature range. Require 0 < inverse-temp-start <= inverse-temp-max.",
        )

    source_task_cfg = _load_task_settings(args.task_settings_file, args.task_setting, "source")
    downstream_task_cfg = _load_task_settings(args.task_settings_file, args.task_setting, "downstream")

    env_id = str(
        args.env_id
        or source_task_cfg.get("env_id")
        or downstream_task_cfg.get("env_id")
        or "LunarLander-v3",
    )
    source_gravity_raw = args.source_gravity if args.source_gravity is not None else source_task_cfg.get("gravity")
    downstream_gravity_raw = (
        args.downstream_gravity
        if args.downstream_gravity is not None
        else downstream_task_cfg.get("gravity")
    )
    source_gravity = None if source_gravity_raw is None else float(source_gravity_raw)
    downstream_gravity = None if downstream_gravity_raw is None else float(downstream_gravity_raw)

    source_task_id = float(args.source_task_id) if args.source_task_id is not None else float(
        source_task_cfg.get("task_id", 0.0),
    )
    downstream_task_id = float(args.downstream_task_id) if args.downstream_task_id is not None else float(
        downstream_task_cfg.get("task_id", 1.0),
    )
    append_task_id = (
        bool(args.append_task_id)
        if args.append_task_id is not None
        else bool(source_task_cfg.get("append_task_id", True))
    )
    source_dynamics = _resolve_lunarlander_dynamics(
        source_task_cfg,
        cfg_name=f"task_settings[{args.task_setting}:source]",
    )
    downstream_dynamics = _resolve_lunarlander_dynamics(
        downstream_task_cfg,
        cfg_name=f"task_settings[{args.task_setting}:downstream]",
    )

    continuous = bool(source_task_cfg.get("continuous", False) or downstream_task_cfg.get("continuous", False))
    if continuous:
        raise ValueError("This script only supports discrete actions (`continuous=False`).")

    source_env_kwargs = {
        "gravity": source_gravity,
        "task_id": source_task_id,
        "append_task_id": append_task_id,
        **source_dynamics,
    }
    downstream_env_kwargs = {
        "gravity": downstream_gravity,
        "task_id": downstream_task_id,
        "append_task_id": append_task_id,
        **downstream_dynamics,
    }

    source_run_dir = (
        args.source_run_dir
        if args.source_run_dir is not None
        else _resolve_default_source_run_dir(args.outputs_root, args.task_setting, args.seed)
    )
    actor_ckpt = source_run_dir / "actor.pt"
    critic_ckpt = source_run_dir / "critic.pt"

    if not actor_ckpt.exists():
        raise FileNotFoundError(f"Source actor checkpoint not found: {actor_ckpt}")
    if args.warm_start_critic and not critic_ckpt.exists():
        raise FileNotFoundError(f"Source critic checkpoint not found: {critic_ckpt}")

    hidden_size = _load_source_hidden_size(source_run_dir, args.hidden_size)

    source_env_for_dim = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **source_env_kwargs,
    )
    if not isinstance(source_env_for_dim.action_space, gym.spaces.Discrete):
        raise ValueError("Expected discrete action space for LunarLander.")
    obs_dim = int(source_env_for_dim.observation_space.shape[0])  # type: ignore[index]
    n_actions = int(source_env_for_dim.action_space.n)  # type: ignore[union-attr]
    source_env_for_dim.close()

    source_actor, source_critic = build_actor_critic(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_size=hidden_size,
    )
    source_actor.load_state_dict(torch.load(actor_ckpt, map_location="cpu"))
    if args.warm_start_critic:
        source_critic.load_state_dict(torch.load(critic_ckpt, map_location="cpu"))

    source_rollout_env = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **source_env_kwargs,
    )
    try:
        rashomon_dataset, rollout_lengths = create_source_rollout_rashomon_dataset(
            actor=copy.deepcopy(source_actor),
            env=source_rollout_env,
            seed=args.seed,
            n_actions=n_actions,
            rashomon_rollouts=args.rashomon_rollouts,
        )
    finally:
        source_rollout_env.close()

    print(
        f"Built source rollout Rashomon dataset: {len(rashomon_dataset)} samples "
        f"from {args.rashomon_rollouts} rollouts.",
    )

    (
        param_bounds_l,
        param_bounds_u,
        bounded_model,
        selected_inverse_temp,
        surrogate_threshold,
        cert_values,
        selected_cert_idx,
    ) = compute_rashomon_bounds(
        actor=copy.deepcopy(source_actor),
        rashomon_dataset=rashomon_dataset,
        seed=args.seed,
        rashomon_n_iters=int(args.rashomon_n_iters),
        min_hard_spec=float(args.rashomon_min_hard_spec),
        aggregation=str(args.rashomon_surrogate_aggregation),
        inverse_temp_start=int(args.inverse_temp_start),
        inverse_temp_max=int(args.inverse_temp_max),
        checkpoint=int(args.rashomon_checkpoint),
    )

    print(
        f"Rashomon bounds ready: aggregation={args.rashomon_surrogate_aggregation} | "
        f"min_hard_spec={args.rashomon_min_hard_spec:.3f} | selected_cert={cert_values[selected_cert_idx]:.4f} "
        f"(idx={selected_cert_idx}) | inverse_temp={selected_inverse_temp}",
    )

    task_feature_index = obs_dim - 1
    do_task_neutralization = (
        append_task_id
        and bool(args.enable_task_neutralization)
        and not args.disable_task_neutralization
    )
    if do_task_neutralization:
        neutralize_task_feature(source_actor, task_feature_index, downstream_task_id)
        if args.warm_start_critic:
            neutralize_task_feature(source_critic, task_feature_index, downstream_task_id)

    ppo_cfg = PPOConfig(
        seed=int(args.seed),
        total_timesteps=int(args.total_timesteps),
        eval_episodes=int(args.eval_episodes_during_training),
        rollout_steps=int(args.rollout_steps),
        update_epochs=int(args.update_epochs),
        minibatch_size=int(args.minibatch_size),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_coef=float(args.clip_coef),
        ent_coef=float(args.ent_coef),
        vf_coef=float(args.vf_coef),
        lr=float(args.lr),
        max_grad_norm=float(args.max_grad_norm),
        device=args.device,
        early_stop=bool(args.early_stop),
        early_stop_min_steps=int(args.early_stop_min_steps),
        early_stop_reward_threshold=(
            float(args.early_stop_reward_threshold)
            if args.early_stop_reward_threshold is not None
            else None
        ),
        early_stop_failure_rate_threshold=(
            float(args.early_stop_failure_rate_threshold)
            if args.early_stop_failure_rate_threshold is not None
            else None
        ),
        early_stop_deterministic_total_reward_threshold=(
            float(args.early_stop_deterministic_total_reward_threshold)
            if args.early_stop_deterministic_total_reward_threshold is not None
            else None
        ),
        early_stop_deterministic_eval_episodes=int(args.early_stop_deterministic_eval_episodes),
    )

    print(
        f"Adapting LunarLander with Rashomon-PGD | source_task={source_task_id} -> "
        f"downstream_task={downstream_task_id} | warm_critic={args.warm_start_critic} | "
        f"task_neutralization={do_task_neutralization}",
    )

    train_env = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **downstream_env_kwargs,
    )
    early_stop_eval_env = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **downstream_env_kwargs,
    )

    actor, critic, training_data = ppo_train(  # type: ignore[assignment]
        train_env,
        ppo_cfg,
        actor_warm_start=source_actor,
        critic_warm_start=(source_critic if args.warm_start_critic else None),
        actor_param_bounds_l=param_bounds_l,
        actor_param_bounds_u=param_bounds_u,
        early_stop_eval_env=early_stop_eval_env,
        return_training_data=True,
    )

    eval_episodes_post_training = int(args.eval_episodes_post_training)
    source_eval_env = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **source_env_kwargs,
    )
    source_mean_reward, source_std_reward, source_failure_rate = evaluate(
        source_eval_env,
        actor,
        episodes=eval_episodes_post_training,
        deterministic=True,
        device=args.device,
    )
    source_eval_env.close()

    downstream_eval_env = _make_lunarlander_env(
        env_id,
        render_mode=None,
        **downstream_env_kwargs,
    )
    downstream_mean_reward, downstream_std_reward, downstream_failure_rate = evaluate(
        downstream_eval_env,
        actor,
        episodes=eval_episodes_post_training,
        deterministic=True,
        device=args.device,
    )
    downstream_eval_env.close()

    downstream_run_dir = _seed_run_dir(args.outputs_root, args.task_setting, args.seed) / args.run_subdir
    downstream_run_dir.mkdir(parents=True, exist_ok=True)

    actor_path = downstream_run_dir / "actor.pt"
    critic_path = downstream_run_dir / "critic.pt"
    training_data_path = downstream_run_dir / "training_data.pt"
    rashomon_dataset_path = downstream_run_dir / "rashomon_dataset.pt"
    bounded_model_path = downstream_run_dir / "rashomon_bounded_model.pt"
    bounds_path = downstream_run_dir / "rashomon_param_bounds.pt"
    rollout_stats_path = downstream_run_dir / "rashomon_rollout_stats.yaml"
    source_plot_path = downstream_run_dir / "trajectory_source.png"
    downstream_plot_path = downstream_run_dir / "trajectory_downstream.png"
    summary_path = downstream_run_dir / "run_summary.yaml"

    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    torch.save(training_data, training_data_path)
    torch.save(rashomon_dataset, rashomon_dataset_path)
    torch.save(bounded_model, bounded_model_path)
    torch.save(
        {
            "param_bounds_l": param_bounds_l,
            "param_bounds_u": param_bounds_u,
        },
        bounds_path,
    )

    # Plot with a CPU actor copy to avoid device-mismatch issues in rendering helpers.
    actor_for_plot = copy.deepcopy(actor).to("cpu")
    actor_for_plot.eval()

    rollout_stats: dict[str, Any] = {
        "rashomon_rollouts": int(args.rashomon_rollouts),
        "total_state_action_pairs": int(len(rashomon_dataset)),
        "rollout_lengths": [int(x) for x in rollout_lengths],
        "rollout_length_min": int(min(rollout_lengths)),
        "rollout_length_max": int(max(rollout_lengths)),
        "rollout_length_mean": float(np.mean(rollout_lengths)),
        "rollout_length_std": float(np.std(rollout_lengths)),
    }
    rollout_stats_path.write_text(
        yaml.safe_dump(rollout_stats, sort_keys=False),
        encoding="utf-8",
    )

    _plot_trajectory_grid(
        env_id=env_id,
        gravity=source_gravity,
        task_id=source_task_id,
        append_task_id=append_task_id,
        dynamics_cfg=source_dynamics,
        actor=actor_for_plot,
        seed=args.seed,
        device="cpu",
        output_path=source_plot_path,
        episodes=int(args.trajectory_episodes),
        max_frames_per_episode=int(args.trajectory_max_frames_per_episode),
    )
    _plot_trajectory_grid(
        env_id=env_id,
        gravity=downstream_gravity,
        task_id=downstream_task_id,
        append_task_id=append_task_id,
        dynamics_cfg=downstream_dynamics,
        actor=actor_for_plot,
        seed=args.seed,
        device="cpu",
        output_path=downstream_plot_path,
        episodes=int(args.trajectory_episodes),
        max_frames_per_episode=int(args.trajectory_max_frames_per_episode),
    )

    run_settings = {
        "seed": int(args.seed),
        "env_id": env_id,
        "continuous": bool(continuous),
        "source_gravity": source_gravity,
        "downstream_gravity": downstream_gravity,
        "source_dynamics": source_dynamics,
        "downstream_dynamics": downstream_dynamics,
        "source_task_id": float(source_task_id),
        "downstream_task_id": float(downstream_task_id),
        "append_task_id": bool(append_task_id),
        "warm_start_critic": bool(args.warm_start_critic),
        "task_feature_neutralization": bool(do_task_neutralization),
        "task_feature_index": int(task_feature_index) if do_task_neutralization else None,
        "hidden_size": int(hidden_size),
        "eval_episodes_during_training": int(args.eval_episodes_during_training),
        "eval_episodes_post_training": int(eval_episodes_post_training),
        "trajectory_episodes": int(args.trajectory_episodes),
        "trajectory_max_frames_per_episode": int(args.trajectory_max_frames_per_episode),
        "rashomon_rollouts": int(args.rashomon_rollouts),
        "rashomon_n_iters": int(args.rashomon_n_iters),
        "surrogate_aggregation": str(args.rashomon_surrogate_aggregation),
        "inverse_temp_start": int(args.inverse_temp_start),
        "inverse_temp_max": int(args.inverse_temp_max),
        "rashomon_checkpoint": int(args.rashomon_checkpoint),
        "source_checkpoint_dir": str(source_run_dir),
        "task_setting": args.task_setting,
        "task_settings_file": str(args.task_settings_file),
    }
    run_results = {
        "rashomon_dataset_size": int(len(rashomon_dataset)),
        "surrogate_threshold": float(surrogate_threshold),
        "inverse_temperature": int(selected_inverse_temp),
        "selected_certificate_index": int(selected_cert_idx),
        "selected_certificate": float(cert_values[selected_cert_idx]),
        "all_certificates": [float(v) for v in cert_values],
        "source_mean_reward": float(source_mean_reward),
        "source_std_reward": float(source_std_reward),
        "source_failure_rate": float(source_failure_rate),
        "downstream_mean_reward": float(downstream_mean_reward),
        "downstream_std_reward": float(downstream_std_reward),
        "downstream_failure_rate": float(downstream_failure_rate),
    }
    artifacts = {
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
        "training_data_path": str(training_data_path),
        "rashomon_dataset_path": str(rashomon_dataset_path),
        "rashomon_bounded_model_path": str(bounded_model_path),
        "rashomon_param_bounds_path": str(bounds_path),
        "rashomon_rollout_stats_path": str(rollout_stats_path),
        "trajectory_source_plot_path": str(source_plot_path),
        "trajectory_downstream_plot_path": str(downstream_plot_path),
    }
    summary = {
        "run_settings": run_settings,
        "run_results": run_results,
        "artifacts": artifacts,
    }
    summary_path.write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )

    print(
        f"Source eval ({eval_episodes_post_training} ep): mean_reward={source_mean_reward:.3f}, "
        f"std={source_std_reward:.3f}, failure_rate={source_failure_rate:.3f}",
    )
    print(
        f"Downstream eval ({eval_episodes_post_training} ep): mean_reward={downstream_mean_reward:.3f}, "
        f"std={downstream_std_reward:.3f}, failure_rate={downstream_failure_rate:.3f}",
    )
    print(f"Saved actor: {actor_path}")
    print(f"Saved critic: {critic_path}")
    print(f"Saved Rashomon dataset: {rashomon_dataset_path}")
    print(f"Saved Rashomon bounded model: {bounded_model_path}")
    print(f"Saved source trajectory grid: {source_plot_path}")
    print(f"Saved downstream trajectory grid: {downstream_plot_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
