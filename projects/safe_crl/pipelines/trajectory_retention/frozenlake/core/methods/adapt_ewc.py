"""Adapt source policy to downstream scaled FrozenLake using EWC-PPO."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import os
os.environ["SDL_AUDIODRIVER"] = "dummy" # to disable ALSA warnings when running on headless servers without audio devices
import numpy as np
import torch
import yaml

from projects.safe_crl.pipelines._shared.adaptation_utils import load_yaml as _load_yaml
from projects.safe_crl.pipelines._shared.adaptation_utils import neutralize_task_feature
from projects.safe_crl.pipelines.trajectory_retention.frozenlake.core.methods.source_train import build_actor_critic, make_env_from_layout
from projects.safe_crl.pipelines.trajectory_retention.frozenlake.core.orchestration.run_paths import (
    default_adapt_ewc_settings_file,
    default_adapt_ppo_settings_file,
    default_downstream_envs_file,
    default_outputs_root,
    default_source_envs_file,
    default_train_source_settings_file,
    resolve_default_source_run_dir,
    seed_run_dir,
)
from projects.safe_crl.utils.ewc_ppo import EWCPPOConfig, compute_ewc_state, ewc_ppo_train
from projects.safe_crl.utils.gymnasium_utils import plot_episode
from projects.safe_crl.utils.ppo_utils import evaluate


def _find_layout_with_ppo(layout: str, current_file: Path) -> Path | None:
    """Return an alternative settings file that contains a PPO block for this layout."""
    candidates = [
        default_adapt_ppo_settings_file(),
    ]
    current_resolved = current_file.resolve()
    for candidate in candidates:
        if not candidate.exists() or candidate.resolve() == current_resolved:
            continue
        try:
            data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        layout_cfg = data.get(layout, None)
        if isinstance(layout_cfg, dict) and isinstance(layout_cfg.get("ppo", None), dict):
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run downstream adaptation with EWC-PPO for one layout.")
    parser.add_argument("--pipeline", "--layout", type=str, dest="layout", default="diagonal_30x30")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--activation",
        type=str,
        choices=["tanh", "relu"],
        default="relu",
    )
    parser.add_argument(
        "--source-env-file",
        type=Path,
        default=default_source_envs_file(),
    )
    parser.add_argument(
        "--downstream-env-file",
        type=Path,
        default=default_downstream_envs_file(),
    )
    parser.add_argument(
        "--source-settings-file",
        type=Path,
        default=default_train_source_settings_file(),
    )
    parser.add_argument(
        "--adapt-settings-file",
        type=Path,
        default=default_adapt_ppo_settings_file(),
        help="Shared downstream settings file with PPO/common per-layout config.",
    )
    parser.add_argument(
        "--ewc-settings-file",
        type=Path,
        default=default_adapt_ewc_settings_file(),
        help="EWC-specific downstream settings file.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=default_outputs_root(),
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help="Optional explicit source checkpoint directory. Defaults to outputs/<layout>/seed_<seed>/source",
    )
    parser.add_argument(
        "--run-subdir",
        type=str,
        default="downstream_ewc",
        help="Subdirectory under outputs/<layout>/seed_<seed>/ where outputs are saved.",
    )
    parser.add_argument(
        "--disable-task-neutralization",
        action="store_true",
        help="Disable first-layer task-feature neutralization before adaptation.",
    )
    parser.add_argument(
        "--total-timesteps-override",
        type=int,
        default=None,
        help="Optional override for PPO total timesteps.",
    )
    parser.add_argument(
        "--ewc-lambda-override",
        type=float,
        default=None,
        help="Optional override for EWC lambda.",
    )
    parser.add_argument(
        "--fisher-sample-size",
        type=int,
        default=10_000,
        help="Maximum number of source states used to estimate Fisher diagonal.",
    )
    parser.add_argument(
        "--ewc-apply-to-critic",
        action="store_true",
        help="Also apply EWC regularization to critic parameters.",
    )
    args = parser.parse_args()

    source_envs = _load_yaml(args.source_env_file)
    downstream_envs = _load_yaml(args.downstream_env_file)
    source_settings = _load_yaml(args.source_settings_file)
    adapt_settings = _load_yaml(args.adapt_settings_file)
    ewc_settings = _load_yaml(args.ewc_settings_file)

    if args.layout not in source_envs:
        raise ValueError(f"Layout '{args.layout}' not found in {args.source_env_file}")
    if args.layout not in downstream_envs:
        raise ValueError(f"Layout '{args.layout}' not found in {args.downstream_env_file}")
    if args.layout not in source_settings:
        raise ValueError(f"Layout '{args.layout}' not found in {args.source_settings_file}")
    if args.layout not in adapt_settings:
        raise ValueError(f"Layout '{args.layout}' not found in {args.adapt_settings_file}")
    if args.layout not in ewc_settings:
        raise ValueError(f"Layout '{args.layout}' not found in {args.ewc_settings_file}")

    source_cfg = source_envs[args.layout]
    downstream_cfg = downstream_envs[args.layout]

    source_settings_cfg = source_settings[args.layout]
    if not isinstance(source_settings_cfg, dict) or "ppo" not in source_settings_cfg:
        raise ValueError(
            f"Layout '{args.layout}' in {args.source_settings_file} does not define a 'ppo' section.",
        )
    source_ppo_cfg = source_settings_cfg["ppo"]

    adapt_cfg = adapt_settings[args.layout]
    if not isinstance(adapt_cfg, dict):
        raise ValueError(
            f"Expected dict config for layout '{args.layout}' in {args.adapt_settings_file}, got {type(adapt_cfg)}.",
        )
    if "ppo" not in adapt_cfg:
        success = adapt_cfg.get("success", None)
        notes = adapt_cfg.get("notes", None)
        available_keys = ", ".join(sorted(str(k) for k in adapt_cfg.keys()))
        fallback_file = _find_layout_with_ppo(args.layout, args.adapt_settings_file)
        fallback_hint = (
            f" A matching fallback exists: --adapt-settings-file {fallback_file}."
            if fallback_file is not None
            else ""
        )
        raise ValueError(
            f"Layout '{args.layout}' in {args.adapt_settings_file} has no 'ppo' section "
            f"(success={success}, keys=[{available_keys}]). "
            f"notes={notes!r}.{fallback_hint}",
        )
    adapt_ppo_cfg = adapt_cfg["ppo"]
    if not isinstance(adapt_ppo_cfg, dict):
        raise ValueError(
            f"Expected 'ppo' to be a dict for layout '{args.layout}' in {args.adapt_settings_file}.",
        )
    ewc_layout_cfg = ewc_settings[args.layout]
    if not isinstance(ewc_layout_cfg, dict):
        raise ValueError(
            f"Expected dict config for layout '{args.layout}' in {args.ewc_settings_file}, "
            f"got {type(ewc_layout_cfg)}.",
        )
    if "ewc" in ewc_layout_cfg:
        adapt_ewc_cfg = ewc_layout_cfg["ewc"]
    elif any(k in ewc_layout_cfg for k in ("ewc_lambda", "lambda", "ewc_apply_to_critic")):
        # Backward-compatible support for flat EWC config blocks.
        adapt_ewc_cfg = ewc_layout_cfg
    else:
        adapt_ewc_cfg = {}
    if not isinstance(adapt_ewc_cfg, dict):
        raise ValueError(
            f"Expected EWC config dict for layout '{args.layout}' in {args.ewc_settings_file}.",
        )

    source_task_num = float(adapt_cfg.get("source_task_num", 0.0))
    downstream_task_num = float(adapt_cfg.get("downstream_task_num", 1.0))
    warm_actor = bool(adapt_cfg.get("warm_start", {}).get("actor", True))
    warm_critic = bool(adapt_cfg.get("warm_start", {}).get("critic", False))
    train_shaped = bool(adapt_cfg.get("train_shaped", False))

    if not warm_actor:
        raise ValueError("This script expects actor warm-start (warm_start.actor=true).")

    source_map: list[str] = source_cfg["env1_map"]
    downstream_map: list[str] = downstream_cfg["env2_map"]
    max_episode_steps = int(source_cfg["max_episode_steps"])
    downstream_max_episode_steps = int(downstream_cfg.get("max_episode_steps", max_episode_steps))
    hidden = int(source_ppo_cfg["hidden"])

    source_run_dir = (
        args.source_run_dir
        if args.source_run_dir is not None
        else resolve_default_source_run_dir(args.outputs_root, args.layout, args.seed)
    )
    actor_ckpt = source_run_dir / "actor.pt"
    critic_ckpt = source_run_dir / "critic.pt"
    training_data_ckpt = source_run_dir / "training_data.pt"
    ewc_apply_to_critic = bool(adapt_ewc_cfg.get("ewc_apply_to_critic", False) or args.ewc_apply_to_critic)
    need_critic_checkpoint = warm_critic or ewc_apply_to_critic

    if not actor_ckpt.exists():
        raise FileNotFoundError(f"Source actor checkpoint not found: {actor_ckpt}")
    if need_critic_checkpoint and not critic_ckpt.exists():
        raise FileNotFoundError(f"Source critic checkpoint not found: {critic_ckpt}")
    if not training_data_ckpt.exists():
        raise FileNotFoundError(f"Source training data not found: {training_data_ckpt}")

    source_env_for_dim = make_env_from_layout(
        source_map,
        max_episode_steps,
        task_num=source_task_num,
        shaped=False,
    )
    obs_dim = int(source_env_for_dim.observation_space.shape[0])
    source_env_for_dim.close()

    source_actor, source_critic = build_actor_critic(
        obs_dim=obs_dim,
        hidden=hidden,
        activation=args.activation,
    )
    source_actor.load_state_dict(torch.load(actor_ckpt, map_location="cpu"))
    if need_critic_checkpoint:
        source_critic.load_state_dict(torch.load(critic_ckpt, map_location="cpu"))
    source_actor_for_ewc = copy.deepcopy(source_actor)
    source_critic_for_ewc = copy.deepcopy(source_critic) if ewc_apply_to_critic else None

    task_transform_cfg = adapt_cfg.get("pre_adaptation_transform", {})
    do_task_neutralization = (
        bool(task_transform_cfg.get("task_feature_neutralization", False))
        and not args.disable_task_neutralization
    )
    task_feature_index = int(task_transform_cfg.get("task_feature_index", 2))

    if do_task_neutralization:
        neutralize_task_feature(source_actor, task_feature_index, downstream_task_num)
        if warm_critic:
            neutralize_task_feature(source_critic, task_feature_index, downstream_task_num)

    source_training_data = torch.load(training_data_ckpt, map_location="cpu", weights_only=False)
    if not isinstance(source_training_data, dict) or "states" not in source_training_data:
        raise ValueError(f"Expected dict with key 'states' in source training data: {training_data_ckpt}")
    source_states = np.asarray(source_training_data["states"], dtype=np.float32)
    if source_states.ndim != 2 or source_states.shape[0] == 0:
        raise ValueError(
            f"Expected source_training_data['states'] to be shape (N, obs_dim), got {source_states.shape}",
        )

    fisher_sample_size = max(1, min(int(args.fisher_sample_size), int(source_states.shape[0])))
    ewc_lambda = float(
        args.ewc_lambda_override
        if args.ewc_lambda_override is not None
        else adapt_ewc_cfg.get("ewc_lambda", adapt_ewc_cfg.get("lambda", 5_000.0)),
    )

    ewc_state = compute_ewc_state(
        actor=source_actor_for_ewc,
        observations=source_states,
        compute_critic=ewc_apply_to_critic,
        critic=source_critic_for_ewc,
        device=args.device,
        fisher_sample_size=fisher_sample_size,
        seed=args.seed,
    )

    total_timesteps = (
        int(args.total_timesteps_override)
        if args.total_timesteps_override is not None
        else int(adapt_ppo_cfg["total_timesteps"])
    )

    ppo_cfg = EWCPPOConfig(
        seed=int(adapt_ppo_cfg.get("seed", args.seed)),
        total_timesteps=total_timesteps,
        eval_episodes=int(adapt_ppo_cfg.get("eval_episodes", 1)),
        rollout_steps=int(adapt_ppo_cfg["rollout_steps"]),
        update_epochs=int(adapt_ppo_cfg["update_epochs"]),
        minibatch_size=int(adapt_ppo_cfg["minibatch_size"]),
        gamma=float(adapt_ppo_cfg["gamma"]),
        gae_lambda=float(adapt_ppo_cfg["gae_lambda"]),
        clip_coef=float(adapt_ppo_cfg["clip_coef"]),
        ent_coef=float(adapt_ppo_cfg["ent_coef"]),
        vf_coef=float(adapt_ppo_cfg["vf_coef"]),
        lr=float(adapt_ppo_cfg["lr"]),
        max_grad_norm=float(adapt_ppo_cfg["max_grad_norm"]),
        device=args.device,
        early_stop=bool(adapt_ppo_cfg.get("early_stop", False)),
        early_stop_deterministic_total_reward_threshold=adapt_ppo_cfg.get(
            "early_stop_deterministic_total_reward_threshold",
            None,
        ),
        early_stop_deterministic_eval_episodes=int(
            adapt_ppo_cfg.get("early_stop_deterministic_eval_episodes", 1),
        ),
        ewc_lambda=ewc_lambda,
        ewc_apply_to_critic=ewc_apply_to_critic,
    )

    print(
        f"Adapting {args.layout} with EWC-PPO | source_task={source_task_num} -> downstream_task={downstream_task_num} | "
        f"warm_critic={warm_critic} | task_neutralization={do_task_neutralization} | "
        f"train_shaped={train_shaped} | ewc_lambda={ewc_lambda} | fisher_sample_size={fisher_sample_size}",
    )
    if train_shaped:
        print(
            "Using sparse-reward downstream environment for periodic eval and early-stop checks "
            "(training still uses shaped rewards).",
        )

    train_env = make_env_from_layout(
        downstream_map,
        max_episode_steps,
        task_num=downstream_task_num,
        shaped=train_shaped,
    )
    early_stop_eval_env = make_env_from_layout(
        downstream_map,
        max_episode_steps,
        task_num=downstream_task_num,
        shaped=False,
    )
    actor, critic, training_data = ewc_ppo_train(  # type: ignore[assignment]
        train_env,
        ppo_cfg,
        ewc_states=[ewc_state],
        actor_warm_start=source_actor,
        critic_warm_start=(source_critic if warm_critic else None),
        early_stop_eval_env=early_stop_eval_env,
        return_training_data=True,
    )

    eval_episodes = int(adapt_cfg.get("downstream_eval", {}).get("episodes", 1))
    source_eval_env = make_env_from_layout(
        source_map,
        max_episode_steps,
        task_num=source_task_num,
        shaped=False,
    )
    source_mean_reward, source_std_reward, source_failure_rate = evaluate(
        source_eval_env,
        actor,
        episodes=eval_episodes,
        deterministic=True,
        device=args.device,
    )
    source_eval_env.close()

    downstream_eval_env = make_env_from_layout(
        downstream_map,
        downstream_max_episode_steps,
        task_num=downstream_task_num,
        shaped=False,
    )
    downstream_mean_reward, downstream_std_reward, downstream_failure_rate = evaluate(
        downstream_eval_env,
        actor,
        episodes=eval_episodes,
        deterministic=True,
        device=args.device,
    )
    downstream_eval_env.close()

    downstream_run_dir = seed_run_dir(args.outputs_root, args.layout, args.seed) / args.run_subdir
    downstream_run_dir.mkdir(parents=True, exist_ok=True)

    actor_path = downstream_run_dir / "actor.pt"
    critic_path = downstream_run_dir / "critic.pt"
    training_data_path = downstream_run_dir / "training_data.pt"
    ewc_state_path = downstream_run_dir / "ewc_state.pt"
    source_plot_path = downstream_run_dir / "trajectory_source.png"
    downstream_plot_path = downstream_run_dir / "trajectory_downstream.png"

    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    torch.save(training_data, training_data_path)
    torch.save(ewc_state, ewc_state_path)

    source_render_env = make_env_from_layout(
        source_map,
        max_episode_steps,
        task_num=source_task_num,
        shaped=False,
        render_mode="rgb_array",
    )
    try:
        plot_episode(
            env=source_render_env,
            actor=actor,
            seed=args.seed,
            deterministic=True,
            save_path=str(source_plot_path),
            title=f"EWC Adapted Policy on Source Task: {args.layout}",
        )
    finally:
        source_render_env.close()

    downstream_render_env = make_env_from_layout(
        downstream_map,
        downstream_max_episode_steps,
        task_num=downstream_task_num,
        shaped=False,
        render_mode="rgb_array",
    )
    try:
        plot_episode(
            env=downstream_render_env,
            actor=actor,
            seed=args.seed,
            deterministic=True,
            save_path=str(downstream_plot_path),
            title=f"EWC Adapted Policy on Downstream Task: {args.layout}",
        )
    finally:
        downstream_render_env.close()

    run_settings = {
        "layout": args.layout,
        "seed": args.seed,
        "source_task_num": source_task_num,
        "downstream_task_num": downstream_task_num,
        "eval_episodes": int(eval_episodes),
        "source_eval_episodes": int(eval_episodes),
        "downstream_eval_episodes": int(eval_episodes),
        "warm_start_actor": warm_actor,
        "warm_start_critic": warm_critic,
        "train_shaped": train_shaped,
        "task_feature_neutralization": do_task_neutralization,
        "task_feature_index": task_feature_index,
        "activation": args.activation,
        "ewc_lambda": float(ewc_lambda),
        "ewc_apply_to_critic": ewc_apply_to_critic,
        "fisher_sample_size": int(fisher_sample_size),
        "source_env_file": str(args.source_env_file),
        "downstream_env_file": str(args.downstream_env_file),
        "source_settings_file": str(args.source_settings_file),
        "adapt_settings_file": str(args.adapt_settings_file),
        "ewc_settings_file": str(args.ewc_settings_file),
        "outputs_root": str(args.outputs_root),
        "run_subdir": str(args.run_subdir),
        "source_checkpoint_dir": str(source_run_dir),
    }
    run_results = {
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
        "ewc_state_path": str(ewc_state_path),
        "trajectory_source_plot_path": str(source_plot_path),
        "trajectory_downstream_plot_path": str(downstream_plot_path),
        "source_checkpoint_dir": str(source_run_dir),
    }
    summary = {
        "run_settings": run_settings,
        "run_results": run_results,
        "artifacts": artifacts,
        **run_settings,
        **run_results,
        **artifacts,
    }
    (downstream_run_dir / "run_summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )

    print(
        f"Source eval ({eval_episodes} ep): mean_reward={source_mean_reward:.3f}, "
        f"std={source_std_reward:.3f}, failure_rate={source_failure_rate:.3f}",
    )
    print(
        f"Downstream eval ({eval_episodes} ep): mean_reward={downstream_mean_reward:.3f}, "
        f"std={downstream_std_reward:.3f}, failure_rate={downstream_failure_rate:.3f}",
    )
    print(f"Saved actor: {actor_path}")
    print(f"Saved critic: {critic_path}")
    print(f"Saved EWC state: {ewc_state_path}")
    print(f"Saved source trajectory plot: {source_plot_path}")
    print(f"Saved downstream trajectory plot: {downstream_plot_path}")


if __name__ == "__main__":
    main()
