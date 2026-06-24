"""Source/NoAdapt PPO training plus supervised safety fine-tuning."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pathlib import Path

import torch
import yaml

from experiments.pipelines.safety_retention.frozenlake.core.config import OBS_DIM, get_pipeline_config
from experiments.pipelines.safety_retention.frozenlake.core.models import build_actor_critic
from experiments.pipelines.safety_retention.frozenlake.core.paths import NOADAPT_POLICY_SUBDIR, mode_run_dir
from experiments.pipelines.safety_retention.frozenlake.core.safety import (
    build_noadapt_supervised_payload,
    create_rashomon_dataset,
    finetune_on_allowed_actions,
    rollout_greedy_policy,
)
from experiments.pipelines.safety_retention.frozenlake.core.training_common import (
    add_common_args,
    assert_successful_source_rollout,
    evaluate_both_tasks,
    make_source_env,
    ppo_config_dict,
    save_trajectory_plot,
    set_seeds,
    source_ppo_config,
    validate_deterministic,
    validate_rl,
    write_summary,
)
from experiments.utils.ppo_utils import ppo_train


def train_source(args: argparse.Namespace) -> Path:
    validate_rl(args.rl)
    validate_deterministic(args.deterministic)
    cfg = get_pipeline_config(args.layout)
    set_seeds(args.seed)
    total_timesteps = (
        int(args.total_timesteps_override)
        if args.total_timesteps_override is not None
        else cfg.source_total_timesteps
    )
    run_dir = mode_run_dir(args.outputs_root, args.layout, args.rl, args.deterministic, args.seed, "source")
    run_dir.mkdir(parents=True, exist_ok=True)

    actor, critic = build_actor_critic(
        obs_dim=OBS_DIM,
        hidden=cfg.hidden,
        activation=cfg.activation,
    )
    ppo_cfg = source_ppo_config(
        cfg,
        seed=args.seed,
        device=args.device,
        total_timesteps=total_timesteps,
    )

    print(f"Training source/noadapt policy for {args.layout} seed={args.seed}.")
    train_env = make_source_env(cfg, shaped=True)
    early_stop_env = make_source_env(cfg, shaped=False)
    actor, critic, training_data = ppo_train(  # type: ignore[assignment]
        train_env,
        ppo_cfg,
        actor_warm_start=actor,
        critic_warm_start=critic,
        early_stop_eval_env=early_stop_env,
        return_training_data=True,
    )
    train_env.close()
    early_stop_env.close()
    actor.cpu()
    critic.cpu()

    source_rollout_env = make_source_env(cfg, shaped=False)
    before_finetune = rollout_greedy_policy(actor, source_rollout_env, seed=args.seed, device="cpu")
    source_rollout_env.close()
    assert_successful_source_rollout(before_finetune, label="Pre-finetune")

    rashomon_payload = create_rashomon_dataset(cfg.source_map, task_num=cfg.source_task_num)
    supervised_payload = build_noadapt_supervised_payload(
        rashomon_payload,
        env_map=cfg.source_map,
        trajectory_steps=before_finetune.steps,
    )
    finetune_result = finetune_on_allowed_actions(
        actor,
        supervised_payload,
        trajectory_steps=before_finetune.steps,
        env_map=cfg.source_map,
        task_num=cfg.source_task_num,
        lr=float(args.safety_finetune_lr or cfg.safety_finetune_lr),
        max_epochs=int(args.safety_finetune_max_epochs or cfg.safety_finetune_max_epochs),
        seed=args.seed,
        device=args.device,
        verbose=True,
    )
    if not finetune_result["reached_target"]:
        raise RuntimeError(f"Safety fine-tuning failed to reach target: {finetune_result}")

    actor.cpu()
    source_rollout_env = make_source_env(cfg, shaped=False)
    after_finetune = rollout_greedy_policy(actor, source_rollout_env, seed=args.seed, device="cpu")
    source_rollout_env.close()
    assert_successful_source_rollout(after_finetune, label="Post-finetune")

    actor_path = run_dir / "actor.pt"
    critic_path = run_dir / "critic.pt"
    training_data_path = run_dir / "training_data.pt"
    rashomon_dataset_path = run_dir / "rashomon_dataset.pt"
    supervised_dataset_path = run_dir / "noadapt_supervised_dataset.pt"
    trajectory_pairs_path = run_dir / "source_policy_state_action_pairs.yaml"
    source_plot_path = run_dir / "trajectory_source.png"
    downstream_plot_path = run_dir / "trajectory_downstream.png"

    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    torch.save(training_data, training_data_path)
    torch.save(rashomon_payload, rashomon_dataset_path)
    torch.save(supervised_payload, supervised_dataset_path)
    trajectory_pairs_path.write_text(
        yaml.safe_dump(before_finetune.state_action_pairs(), sort_keys=False),
        encoding="utf-8",
    )
    save_trajectory_plot(cfg=cfg, actor=actor, task="source", seed=args.seed, path=source_plot_path)
    save_trajectory_plot(cfg=cfg, actor=actor, task="downstream", seed=args.seed, path=downstream_plot_path)

    run_results = evaluate_both_tasks(cfg, actor=actor, device=args.device, seed=args.seed)
    run_results.update(
        {
            "pre_finetune_source_total_reward": float(before_finetune.total_reward),
            "pre_finetune_source_failure_rate": float(before_finetune.failure_rate),
            "post_finetune_source_total_reward": float(after_finetune.total_reward),
            "post_finetune_source_failure_rate": float(after_finetune.failure_rate),
            "safety_finetune_initial_accuracy": float(finetune_result["initial_accuracy"]),
            "safety_finetune_final_accuracy": float(finetune_result["final_accuracy"]),
            "safety_finetune_epochs_run": int(finetune_result["epochs_run"]),
        },
    )
    run_settings = {
        "mode": "source",
        "policy_name": NOADAPT_POLICY_SUBDIR,
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
        "train_shaped": True,
        "early_stop_eval_shaped": False,
        "ppo": ppo_config_dict(ppo_cfg),
        "rashomon_dataset_size": int(rashomon_payload["state"].shape[0]),
        "outputs_root": str(args.outputs_root),
    }
    artifacts = {
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
        "training_data_path": str(training_data_path),
        "rashomon_dataset_path": str(rashomon_dataset_path),
        "noadapt_supervised_dataset_path": str(supervised_dataset_path),
        "source_policy_state_action_pairs_path": str(trajectory_pairs_path),
        "trajectory_source_plot_path": str(source_plot_path),
        "trajectory_downstream_plot_path": str(downstream_plot_path),
    }
    write_summary(run_dir, run_settings=run_settings, run_results=run_results, artifacts=artifacts)
    print(f"Saved source/noadapt artifacts to {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the FrozenLake safety source/NoAdapt policy.")
    add_common_args(parser)
    parser.add_argument("--safety-finetune-lr", type=float, default=None)
    parser.add_argument("--safety-finetune-max-epochs", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        validate_rl(args.rl)
        validate_deterministic(args.deterministic)
        cfg = get_pipeline_config(args.layout)
        run_dir = mode_run_dir(args.outputs_root, args.layout, args.rl, args.deterministic, args.seed, "source")
        print(f"Dry run: mode=source layout={cfg.layout} seed={args.seed} run_dir={run_dir}")
        return 0
    train_source(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
