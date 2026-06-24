"""Unconstrained downstream PPO adaptation (no EWC, no Rashomon bounds)."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pathlib import Path

from experiments.pipelines.safety_retention.frozenlake.core.config import get_pipeline_config
from experiments.pipelines.safety_retention.frozenlake.core.paths import mode_run_dir
from experiments.pipelines.safety_retention.frozenlake.core.training_common import (
    add_adaptation_args,
    downstream_ppo_config,
    finalize_downstream_run,
    load_source_for_adaptation,
    make_downstream_env,
    set_seeds,
    validate_deterministic,
    validate_rl,
)
from experiments.utils.ppo_utils import ppo_train


MODE = "downstream_unconstrained"


def adapt_unconstrained(args: argparse.Namespace) -> Path:
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

    train_env = make_downstream_env(cfg, shaped=True)
    early_stop_env = make_downstream_env(cfg, shaped=False)
    ppo_cfg = downstream_ppo_config(cfg, seed=args.seed, device=args.device, total_timesteps=total_timesteps)
    actor, critic, training_data = ppo_train(  # type: ignore[assignment]
        train_env,
        ppo_cfg,
        actor_warm_start=source_actor,
        critic_warm_start=source_critic,
        early_stop_eval_env=early_stop_env,
        return_training_data=True,
    )
    train_env.close()
    early_stop_env.close()

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
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run unconstrained downstream PPO adaptation.")
    add_adaptation_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        validate_rl(args.rl)
        validate_deterministic(args.deterministic)
        cfg = get_pipeline_config(args.layout)
        run_dir = mode_run_dir(args.outputs_root, args.layout, args.rl, args.deterministic, args.seed, MODE)
        print(f"Dry run: mode={MODE} layout={cfg.layout} seed={args.seed} run_dir={run_dir}")
        return 0
    adapt_unconstrained(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
