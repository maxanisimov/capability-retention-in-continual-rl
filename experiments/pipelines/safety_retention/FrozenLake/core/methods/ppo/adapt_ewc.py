"""EWC-regularized downstream PPO adaptation."""

from __future__ import annotations

import argparse
import copy
import os

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pathlib import Path

import numpy as np
import torch

from experiments.pipelines.safety_retention.FrozenLake.core.config import get_pipeline_config
from experiments.pipelines.safety_retention.FrozenLake.core.paths import mode_run_dir
from experiments.pipelines.safety_retention.FrozenLake.core.training_common import (
    add_adaptation_args,
    ewc_ppo_config,
    finalize_downstream_run,
    load_source_for_adaptation,
    make_downstream_env,
    resolve_deterministic,
    set_seeds,
    validate_deterministic,
    validate_rl,
)
from experiments.utils.ewc_ppo import compute_ewc_state, ewc_ppo_train


MODE = "downstream_ewc"


def adapt_ewc(args: argparse.Namespace) -> Path:
    validate_rl(args.rl)
    validate_deterministic(args.deterministic)
    cfg = get_pipeline_config(args.layout)
    set_seeds(args.seed)
    source_actor, source_critic, source_dir = load_source_for_adaptation(cfg, args)
    total_timesteps = (
        int(args.total_timesteps_override)
        if args.total_timesteps_override is not None
        else cfg.downstream_total_timesteps
    )

    training_data_path = source_dir / "training_data.pt"
    if not training_data_path.exists():
        raise FileNotFoundError(f"Source training data not found: {training_data_path}")
    source_training_data = torch.load(training_data_path, map_location="cpu", weights_only=False)
    source_states = np.asarray(source_training_data["states"], dtype=np.float32)
    fisher_sample_size = max(1, min(int(args.fisher_sample_size or cfg.fisher_sample_size), len(source_states)))
    ewc_state = compute_ewc_state(
        actor=copy.deepcopy(source_actor),
        observations=source_states,
        device=args.device,
        fisher_sample_size=fisher_sample_size,
        seed=args.seed,
    )

    ewc_lambda = float(args.ewc_lambda or cfg.ewc_lambda)
    ppo_cfg = ewc_ppo_config(
        cfg,
        seed=args.seed,
        device=args.device,
        total_timesteps=total_timesteps,
        ewc_lambda=ewc_lambda,
    )

    train_env = make_downstream_env(cfg, shaped=True)
    early_stop_env = make_downstream_env(cfg, shaped=False)
    actor, critic, training_data = ewc_ppo_train(  # type: ignore[assignment]
        train_env,
        ppo_cfg,
        ewc_states=[ewc_state],
        actor_warm_start=source_actor,
        critic_warm_start=source_critic,
        early_stop_eval_env=early_stop_env,
        return_training_data=True,
    )
    train_env.close()
    early_stop_env.close()

    run_dir = mode_run_dir(args.outputs_root, args.layout, args.rl, args.deterministic, args.seed, MODE)
    run_dir.mkdir(parents=True, exist_ok=True)
    ewc_state_path = run_dir / "ewc_state.pt"
    torch.save(ewc_state, ewc_state_path)

    return finalize_downstream_run(
        cfg,
        args,
        mode=MODE,
        actor=actor,
        critic=critic,
        training_data=training_data,
        source_dir=source_dir,
        ppo_cfg=ppo_cfg,
        total_timesteps=total_timesteps,
        extra_artifacts={"ewc_state_path": str(ewc_state_path)},
        extra_settings={
            "ewc_lambda": ewc_lambda,
            "ewc_apply_to_critic": bool(cfg.ewc_apply_to_critic),
            "fisher_sample_size": int(fisher_sample_size),
            "requested_fisher_sample_size": int(args.fisher_sample_size or cfg.fisher_sample_size),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EWC-regularized downstream PPO adaptation.")
    add_adaptation_args(parser)
    parser.add_argument("--ewc-lambda", type=float, default=None)
    parser.add_argument("--fisher-sample-size", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.deterministic = resolve_deterministic(args.layout, args.deterministic)
    if args.dry_run:
        validate_rl(args.rl)
        validate_deterministic(args.deterministic)
        cfg = get_pipeline_config(args.layout)
        run_dir = mode_run_dir(args.outputs_root, args.layout, args.rl, args.deterministic, args.seed, MODE)
        print(f"Dry run: mode={MODE} layout={cfg.layout} seed={args.seed} run_dir={run_dir}")
        return 0
    adapt_ewc(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
