"""Shared argparse building blocks for stage scripts."""

from __future__ import annotations

import argparse

# Default PPO / PPO-Lagrangian optimisation hyperparameters, shared by every
# stage that builds an on-policy learner. Kept here so the three stages cannot
# drift apart.
PPO_HYPERPARAMETER_DEFAULTS: dict[str, float | int] = {
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


def add_ppo_hyperparameter_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the standard PPO optimisation hyperparameter flags to ``parser``."""

    parser.add_argument("--learning-rate", type=float, default=PPO_HYPERPARAMETER_DEFAULTS["learning_rate"])
    parser.add_argument("--n-steps", type=int, default=PPO_HYPERPARAMETER_DEFAULTS["n_steps"])
    parser.add_argument("--batch-size", type=int, default=PPO_HYPERPARAMETER_DEFAULTS["batch_size"])
    parser.add_argument("--n-epochs", type=int, default=PPO_HYPERPARAMETER_DEFAULTS["n_epochs"])
    parser.add_argument("--gamma", type=float, default=PPO_HYPERPARAMETER_DEFAULTS["gamma"])
    parser.add_argument("--gae-lambda", type=float, default=PPO_HYPERPARAMETER_DEFAULTS["gae_lambda"])
    parser.add_argument("--clip-range", type=float, default=PPO_HYPERPARAMETER_DEFAULTS["clip_range"])
    parser.add_argument("--ent-coef", type=float, default=PPO_HYPERPARAMETER_DEFAULTS["ent_coef"])
    parser.add_argument("--vf-coef", type=float, default=PPO_HYPERPARAMETER_DEFAULTS["vf_coef"])
    parser.add_argument("--max-grad-norm", type=float, default=PPO_HYPERPARAMETER_DEFAULTS["max_grad_norm"])
    return parser
