"""Helpers shared by every FrozenLake safety training/adaptation method script."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import torch
import yaml

from projects.safe_crl.pipelines.safety_retention.frozenlake.core.config import PipelineConfig
from projects.safe_crl.pipelines.safety_retention.frozenlake.core.env import make_env, state_index_to_obs
from projects.safe_crl.pipelines.safety_retention.frozenlake.core.paths import default_outputs_root
from projects.safe_crl.pipelines.safety_retention.frozenlake.core.safety import (
    RolloutResult,
    greedy_action,
    rollout_greedy_policy,
    safe_action_mask_for_state,
    traversable_nonterminal_states,
)
from projects.safe_crl.utils.gymnasium_utils import plot_episode
from projects.safe_crl.utils.ppo_utils import PPOConfig, evaluate_with_success


RL_CHOICES = ("ppo",)


class NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: object) -> bool:
        return True


def validate_rl(rl: str) -> None:
    if rl not in RL_CHOICES:
        raise NotImplementedError(
            f"--rl '{rl}' is not implemented. Only {RL_CHOICES} exist today, under "
            f"core/methods/<rl>/. Add a new core/methods/<rl>/ package to support another algorithm.",
        )


def import_method_module(rl: str, name: str):
    """Import core.methods.<rl>.<name>, raising a clear error for an unsupported --rl."""
    import importlib

    try:
        return importlib.import_module(
            f"projects.safe_crl.pipelines.safety_retention.frozenlake.core.methods.{rl}.{name}",
        )
    except ModuleNotFoundError as exc:
        raise NotImplementedError(
            f"--rl '{rl}' is not implemented: no core/methods/{rl}/{name}.py module exists. "
            f"Only {RL_CHOICES} exist today; add a new core/methods/<rl>/ package to support another algorithm.",
        ) from exc


def validate_deterministic(deterministic: bool) -> None:
    if not deterministic:
        warnings.warn(
            "Running with stochastic (slippery) FrozenLake dynamics: core/safety.py's "
            "safety-critical-state checks and the Rashomon certificate assume deterministic "
            "transitions (see core/safety.py's module docstring), so any safety guarantee / "
            "Rashomon certificate produced for this run is NOT sound. Source training and the "
            "unconstrained/EWC baselines remain meaningful.",
            stacklevel=2,
        )


def resolve_deterministic(layout: str, cli_value: bool | None) -> bool:
    """Resolve the effective deterministic flag for a pipeline.

    Dynamics (deterministic vs. slippery) is a property of the pipeline's tasks, so when the
    CLI flag is omitted (``None``) we take it from the pipeline definition. An explicitly passed
    flag that contradicts the pipeline is rejected to avoid mislabeling artifacts (the env always
    follows the pipeline's task definition, so a contradicting tag would be wrong).
    """
    from projects.safe_crl.pipelines.safety_retention.frozenlake.core.config import pipeline_is_deterministic

    pipeline_deterministic = pipeline_is_deterministic(layout)
    if cli_value is None:
        return pipeline_deterministic
    if cli_value != pipeline_deterministic:
        flag = "--deterministic" if cli_value else "--no-deterministic"
        expected = "--deterministic" if pipeline_deterministic else "--no-deterministic"
        raise ValueError(
            f"Pipeline '{layout}' is defined as deterministic={pipeline_deterministic} in "
            f"tasks.yaml, but {flag} was passed. Omit the flag to use the pipeline's setting, "
            f"or pass {expected}.",
        )
    return cli_value


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--pipeline", "--layout", type=str, dest="layout", default="diagonal_4x4")
    parser.add_argument("--rl", type=str, default="ppo", choices=RL_CHOICES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--outputs-root", type=Path, default=default_outputs_root())
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override dynamics regime. Defaults to the pipeline's task definition when omitted.",
    )
    parser.add_argument("--total-timesteps-override", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def add_adaptation_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_common_args(parser)
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help="Override the source/noadapt checkpoint directory used for warm-starting adaptation.",
    )
    return parser


def set_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def source_ppo_config(
    cfg: PipelineConfig,
    *,
    seed: int,
    device: str,
    total_timesteps: int,
) -> PPOConfig:
    return PPOConfig(
        seed=seed,
        total_timesteps=total_timesteps,
        eval_episodes=cfg.eval_episodes,
        rollout_steps=cfg.source_rollout_steps,
        update_epochs=cfg.source_update_epochs,
        minibatch_size=cfg.source_minibatch_size,
        gamma=cfg.source_gamma,
        gae_lambda=cfg.source_gae_lambda,
        clip_coef=cfg.source_clip_coef,
        ent_coef=cfg.source_ent_coef,
        vf_coef=cfg.source_vf_coef,
        lr=cfg.source_lr,
        max_grad_norm=cfg.source_max_grad_norm,
        device=device,
        early_stop_min_steps=0,
        early_stop_reward_threshold=cfg.source_early_stop_reward_threshold,
        early_stop_failure_rate_threshold=cfg.source_early_stop_failure_rate_threshold,
        early_stop_success_rate_threshold=None,
    )


def downstream_ppo_config(
    cfg: PipelineConfig,
    *,
    seed: int,
    device: str,
    total_timesteps: int,
) -> PPOConfig:
    return PPOConfig(
        seed=seed,
        total_timesteps=total_timesteps,
        eval_episodes=cfg.eval_episodes,
        rollout_steps=cfg.downstream_rollout_steps,
        update_epochs=cfg.downstream_update_epochs,
        minibatch_size=cfg.downstream_minibatch_size,
        gamma=cfg.downstream_gamma,
        gae_lambda=cfg.downstream_gae_lambda,
        clip_coef=cfg.downstream_clip_coef,
        ent_coef=cfg.downstream_ent_coef,
        vf_coef=cfg.downstream_vf_coef,
        lr=cfg.downstream_lr,
        max_grad_norm=cfg.downstream_max_grad_norm,
        device=device,
        early_stop_min_steps=0,
        early_stop_reward_threshold=cfg.downstream_early_stop_reward_threshold,
        early_stop_failure_rate_threshold=None,
        early_stop_success_rate_threshold=None,
    )


def ewc_ppo_config(
    cfg: PipelineConfig,
    *,
    seed: int,
    device: str,
    total_timesteps: int,
    ewc_lambda: float,
):
    from projects.safe_crl.utils.ewc_ppo import EWCPPOConfig

    base = downstream_ppo_config(cfg, seed=seed, device=device, total_timesteps=total_timesteps)
    return EWCPPOConfig(
        **base.__dict__,
        ewc_lambda=ewc_lambda,
        ewc_apply_to_critic=cfg.ewc_apply_to_critic,
    )


def ppo_config_dict(ppo_cfg: PPOConfig) -> dict[str, Any]:
    return {
        "seed": int(ppo_cfg.seed),
        "total_timesteps": int(ppo_cfg.total_timesteps),
        "eval_episodes": int(ppo_cfg.eval_episodes),
        "rollout_steps": int(ppo_cfg.rollout_steps),
        "update_epochs": int(ppo_cfg.update_epochs),
        "minibatch_size": int(ppo_cfg.minibatch_size),
        "gamma": float(ppo_cfg.gamma),
        "gae_lambda": float(ppo_cfg.gae_lambda),
        "clip_coef": float(ppo_cfg.clip_coef),
        "ent_coef": float(ppo_cfg.ent_coef),
        "vf_coef": float(ppo_cfg.vf_coef),
        "lr": float(ppo_cfg.lr),
        "max_grad_norm": float(ppo_cfg.max_grad_norm),
        "device": str(ppo_cfg.device),
        "early_stop_min_steps": int(ppo_cfg.early_stop_min_steps),
        "early_stop_reward_threshold": (
            None if ppo_cfg.early_stop_reward_threshold is None else float(ppo_cfg.early_stop_reward_threshold)
        ),
        "early_stop_failure_rate_threshold": (
            None
            if ppo_cfg.early_stop_failure_rate_threshold is None
            else float(ppo_cfg.early_stop_failure_rate_threshold)
        ),
        "early_stop_success_rate_threshold": (
            None
            if ppo_cfg.early_stop_success_rate_threshold is None
            else float(ppo_cfg.early_stop_success_rate_threshold)
        ),
    }


def make_source_env(cfg: PipelineConfig, *, shaped: bool, render_mode: str | None = None):
    return make_env(
        cfg.source_map,
        task_num=cfg.source_task_num,
        max_episode_steps=cfg.max_episode_steps,
        shaped=shaped,
        is_slippery=not cfg.deterministic,
        success_rate=1.0 - cfg.slip_probability,
        render_mode=render_mode,
    )


def make_downstream_env(cfg: PipelineConfig, *, shaped: bool, render_mode: str | None = None):
    return make_env(
        cfg.downstream_map,
        task_num=cfg.downstream_task_num,
        max_episode_steps=cfg.max_episode_steps,
        shaped=shaped,
        is_slippery=not cfg.deterministic,
        success_rate=1.0 - cfg.slip_probability,
        render_mode=render_mode,
    )


def load_actor_critic(
    cfg: PipelineConfig,
    *,
    source_dir: Path,
    map_location: str = "cpu",
) -> tuple[torch.nn.Sequential, torch.nn.Sequential]:
    from projects.safe_crl.pipelines.safety_retention.frozenlake.core.config import OBS_DIM
    from projects.safe_crl.pipelines.safety_retention.frozenlake.core.models import build_actor_critic

    actor, critic = build_actor_critic(
        obs_dim=OBS_DIM,
        hidden=cfg.hidden,
        activation=cfg.activation,
    )
    actor_path = source_dir / "actor.pt"
    critic_path = source_dir / "critic.pt"
    if not actor_path.exists():
        raise FileNotFoundError(f"Source actor checkpoint not found: {actor_path}")
    if not critic_path.exists():
        raise FileNotFoundError(f"Source critic checkpoint not found: {critic_path}")
    actor.load_state_dict(torch.load(actor_path, map_location=map_location))
    critic.load_state_dict(torch.load(critic_path, map_location=map_location))
    return actor, critic


def safety_critical_states(env_map: list[str] | tuple[str, ...]) -> list[int]:
    return [
        state_index
        for state_index in traversable_nonterminal_states(env_map)
        if float(safe_action_mask_for_state(env_map, state_index).sum()) < 4.0
    ]


def safety_critical_state_metrics(
    actor: torch.nn.Module,
    *,
    env_map: list[str] | tuple[str, ...],
    task_num: float,
    device: str | torch.device,
) -> dict[str, Any]:
    critical_states = safety_critical_states(env_map)
    safe_count = 0
    failures: list[dict[str, int]] = []

    for state_index in critical_states:
        obs = state_index_to_obs(state_index, env_map, task_num)
        action = greedy_action(actor, obs, device=device)
        safe_mask = safe_action_mask_for_state(env_map, state_index)
        if bool(safe_mask[action] > 0):
            safe_count += 1
        else:
            failures.append(
                {
                    "state_index": int(state_index),
                    "action": int(action),
                },
            )

    n_critical = len(critical_states)
    safety_rate = (safe_count / n_critical) if n_critical else None
    return {
        "safety_critical_state_safety_rate": (float(safety_rate) if safety_rate is not None else None),
        "safety_critical_state_safe_count": int(safe_count),
        "safety_critical_state_count": int(n_critical),
        "safety_critical_state_failures": failures,
    }


def compute_task_policy_metrics(
    cfg: PipelineConfig,
    *,
    actor: torch.nn.Module,
    task: str,
    seed: int,
    device: str | torch.device,
) -> dict[str, Any]:
    if task == "source":
        env_map = cfg.source_map
        task_num = cfg.source_task_num
        env = make_source_env(cfg, shaped=False)
    elif task == "downstream":
        env_map = cfg.downstream_map
        task_num = cfg.downstream_task_num
        env = make_downstream_env(cfg, shaped=False)
    else:
        raise ValueError(f"Unsupported task '{task}'.")

    try:
        rollout = rollout_greedy_policy(actor, env, seed=seed, device=device)
    finally:
        env.close()

    critical_metrics = safety_critical_state_metrics(
        actor,
        env_map=env_map,
        task_num=task_num,
        device=device,
    )
    return {
        **critical_metrics,
        "greedy_trajectory_safety": float(1.0 - rollout.failure_rate),
        "greedy_trajectory_failed": bool(rollout.failed),
        "total_reward": float(rollout.total_reward),
        "greedy_trajectory_steps": int(len(rollout.steps)),
        "greedy_trajectory_terminated": bool(rollout.terminated),
        "greedy_trajectory_truncated": bool(rollout.truncated),
    }


def evaluate_both_tasks(
    cfg: PipelineConfig,
    *,
    actor: torch.nn.Module,
    device: str,
    seed: int,
) -> dict[str, Any]:
    source_env = make_source_env(cfg, shaped=False)
    source_mean, source_std, source_failure, source_success = evaluate_with_success(
        source_env,
        actor,
        episodes=cfg.eval_episodes,
        deterministic=True,
        device=device,
    )
    source_env.close()

    downstream_env = make_downstream_env(cfg, shaped=False)
    downstream_mean, downstream_std, downstream_failure, downstream_success = evaluate_with_success(
        downstream_env,
        actor,
        episodes=cfg.eval_episodes,
        deterministic=True,
        device=device,
    )
    downstream_env.close()

    task_metrics = {
        "source": compute_task_policy_metrics(
            cfg,
            actor=actor,
            task="source",
            seed=seed,
            device=device,
        ),
        "downstream": compute_task_policy_metrics(
            cfg,
            actor=actor,
            task="downstream",
            seed=seed,
            device=device,
        ),
    }

    return {
        "source_mean_reward": float(source_mean),
        "source_std_reward": float(source_std),
        "source_failure_rate": float(source_failure),
        "source_success_rate": float(source_success),
        "downstream_mean_reward": float(downstream_mean),
        "downstream_std_reward": float(downstream_std),
        "downstream_failure_rate": float(downstream_failure),
        "downstream_success_rate": float(downstream_success),
        "task_metrics": task_metrics,
        "source_safety_critical_state_safety_rate": task_metrics["source"][
            "safety_critical_state_safety_rate"
        ],
        "source_greedy_trajectory_safety": task_metrics["source"]["greedy_trajectory_safety"],
        "source_total_reward": task_metrics["source"]["total_reward"],
        "downstream_safety_critical_state_safety_rate": task_metrics["downstream"][
            "safety_critical_state_safety_rate"
        ],
        "downstream_greedy_trajectory_safety": task_metrics["downstream"]["greedy_trajectory_safety"],
        "downstream_total_reward": task_metrics["downstream"]["total_reward"],
    }


def save_trajectory_plot(
    *,
    cfg: PipelineConfig,
    actor: torch.nn.Module,
    task: str,
    seed: int,
    path: Path,
) -> None:
    if task == "source":
        env = make_source_env(cfg, shaped=False, render_mode="rgb_array")
        title = f"NoAdapt policy on source task: {cfg.layout}"
    elif task == "downstream":
        env = make_downstream_env(cfg, shaped=False, render_mode="rgb_array")
        title = f"Policy on downstream task: {cfg.layout}"
    else:
        raise ValueError(f"Unsupported task '{task}'.")
    try:
        plot_episode(
            env=env,
            actor=actor.cpu(),
            seed=seed,
            deterministic=True,
            save_path=str(path),
            title=title,
        )
    finally:
        env.close()


def write_summary(
    run_dir: Path,
    *,
    run_settings: dict[str, Any],
    run_results: dict[str, Any],
    artifacts: dict[str, Any],
) -> None:
    summary = {
        "run_settings": run_settings,
        "run_results": run_results,
        "artifacts": artifacts,
        **run_settings,
        **run_results,
        **artifacts,
    }
    (run_dir / "run_summary.yaml").write_text(
        yaml.dump(summary, Dumper=NoAliasSafeDumper, sort_keys=False),
        encoding="utf-8",
    )


def assert_successful_source_rollout(rollout: RolloutResult, *, label: str) -> None:
    if rollout.total_reward < 1.0 or rollout.failure_rate > 0.0:
        raise RuntimeError(
            f"{label} source rollout is not optimal and safe: "
            f"reward={rollout.total_reward:.3f}, failure_rate={rollout.failure_rate:.3f}.",
        )


def assert_source_policy_success(
    cfg: PipelineConfig,
    actor: torch.nn.Module,
    *,
    seed: int,
    device: str | torch.device,
    label: str,
) -> None:
    """Gate source-policy quality, branching on dynamics.

    Deterministic pipelines keep the strict single-rollout check (optimal + zero failures).
    Stochastic pipelines instead require an episode-averaged greedy success_rate at or above
    ``cfg.source_success_rate_threshold``, since a single greedy rollout can slip into a hole
    even for an optimal policy.
    """
    if cfg.deterministic:
        env = make_source_env(cfg, shaped=False)
        try:
            rollout = rollout_greedy_policy(actor, env, seed=seed, device=device)
        finally:
            env.close()
        assert_successful_source_rollout(rollout, label=label)
        return

    env = make_source_env(cfg, shaped=False)
    try:
        mean_reward, _std_reward, failure_rate, success_rate = evaluate_with_success(
            env,
            actor,
            episodes=cfg.source_success_eval_episodes,
            seed=seed,
            device=str(device),
            deterministic=True,
        )
    finally:
        env.close()
    if success_rate < cfg.source_success_rate_threshold:
        raise RuntimeError(
            f"{label} stochastic source policy success_rate={success_rate:.3f} over "
            f"{cfg.source_success_eval_episodes} episodes is below threshold "
            f"{cfg.source_success_rate_threshold:.3f} "
            f"(mean_reward={mean_reward:.3f}, failure_rate={failure_rate:.3f}).",
        )


def load_source_for_adaptation(
    cfg: PipelineConfig,
    args: argparse.Namespace,
) -> tuple[torch.nn.Sequential, torch.nn.Sequential, Path]:
    from projects.safe_crl.pipelines.safety_retention.frozenlake.core.paths import resolve_source_run_dir

    source_dir = args.source_run_dir or resolve_source_run_dir(
        args.outputs_root,
        args.layout,
        args.rl,
        args.deterministic,
        args.seed,
    )
    actor, critic = load_actor_critic(cfg, source_dir=source_dir)
    return actor, critic, source_dir


def finalize_downstream_run(
    cfg: PipelineConfig,
    args: argparse.Namespace,
    *,
    mode: str,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    training_data: Any,
    source_dir: Path,
    ppo_cfg: PPOConfig,
    total_timesteps: int,
    extra_artifacts: dict[str, Any] | None = None,
    extra_settings: dict[str, Any] | None = None,
) -> Path:
    from projects.safe_crl.pipelines.safety_retention.frozenlake.core.paths import mode_run_dir

    actor.cpu()
    critic.cpu()

    run_dir = mode_run_dir(args.outputs_root, args.layout, args.rl, args.deterministic, args.seed, mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    actor_path = run_dir / "actor.pt"
    critic_path = run_dir / "critic.pt"
    training_data_path = run_dir / "training_data.pt"
    source_plot_path = run_dir / "trajectory_source.png"
    downstream_plot_path = run_dir / "trajectory_downstream.png"
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    torch.save(training_data, training_data_path)

    artifacts = {
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
        "training_data_path": str(training_data_path),
        "trajectory_source_plot_path": str(source_plot_path),
        "trajectory_downstream_plot_path": str(downstream_plot_path),
        "source_checkpoint_dir": str(source_dir),
        **(extra_artifacts or {}),
    }

    save_trajectory_plot(cfg=cfg, actor=actor, task="source", seed=args.seed, path=source_plot_path)
    save_trajectory_plot(cfg=cfg, actor=actor, task="downstream", seed=args.seed, path=downstream_plot_path)

    run_results = evaluate_both_tasks(cfg, actor=actor, device=args.device, seed=args.seed)
    run_settings = {
        "mode": mode,
        "layout": args.layout,
        "rl": args.rl,
        "deterministic": bool(args.deterministic),
        "seed": args.seed,
        "activation": cfg.activation,
        "hidden": cfg.hidden,
        "reference_layout": cfg.reference_layout,
        "reference_settings_source": cfg.reference_settings_source,
        "reference_settings_files": cfg.reference_settings_files,
        "source_task_num": cfg.source_task_num,
        "downstream_task_num": cfg.downstream_task_num,
        "total_timesteps": int(total_timesteps),
        "warm_start_actor": True,
        "warm_start_critic": True,
        "train_shaped": True,
        "early_stop_eval_shaped": False,
        "task_feature_neutralization": False,
        "task_feature_index": 2,
        "ppo": ppo_config_dict(ppo_cfg),
        "outputs_root": str(args.outputs_root),
        "source_checkpoint_dir": str(source_dir),
        **(extra_settings or {}),
    }
    write_summary(run_dir, run_settings=run_settings, run_results=run_results, artifacts=artifacts)
    print(f"Saved {mode} artifacts to {run_dir}")
    return run_dir
