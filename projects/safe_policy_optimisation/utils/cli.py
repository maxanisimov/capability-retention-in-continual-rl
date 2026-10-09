"""Shared argparse building blocks for stage scripts."""

from __future__ import annotations

import argparse
import os
import sys
import warnings

# Default PPO / PPO-Lagrangian optimisation hyperparameters, shared by every
# stage that builds an on-policy learner. Kept here so the three stages cannot
# drift apart.
PPO_HYPERPARAMETER_DEFAULTS: dict[str, float | int] = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
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


# Default actor-critic architecture, shared by every stage that builds a policy
# network (RL baselines and PSPO alike), so they compare like-for-like unless
# told otherwise.
ARCHITECTURE_DEFAULTS: dict[str, int] = {
    "n_hidden": 2,
    "hidden_dim": 64,
}


def add_architecture_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the standard actor-critic architecture flags to ``parser``."""

    parser.add_argument(
        "--n-hidden",
        type=int,
        default=ARCHITECTURE_DEFAULTS["n_hidden"],
        help="Hidden layers in the actor/critic MLP.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=ARCHITECTURE_DEFAULTS["hidden_dim"],
        help="Width of each hidden layer.",
    )
    return parser


def net_arch_from_args(args: argparse.Namespace) -> list[int]:
    """Build an SB3 ``net_arch`` list from ``--n-hidden``/``--hidden-dim``."""

    return [int(args.hidden_dim)] * int(args.n_hidden)


# Renaming an environment variable or a command-line flag would silently change
# the meaning of launch commands people already have in shell history and in
# running screen sessions. Every rename therefore keeps the old spelling working
# for one release, warns, and prefers the new spelling when both are supplied.


def env_with_legacy_alias(
    name: str,
    legacy_name: str,
    default: str | None = None,
) -> str | None:
    """Read ``name``, falling back to a deprecated ``legacy_name``.

    ``name`` wins whenever it is set at all, so exporting the canonical variable
    is always enough to override a stale ``legacy_name`` left in the environment.
    """

    if name in os.environ:
        return os.environ[name]
    if legacy_name in os.environ:
        warnings.warn(
            f"{legacy_name} is deprecated; use {name}.",
            FutureWarning,
            stacklevel=2,
        )
        return os.environ[legacy_name]
    return default


def option_was_supplied(argv: list[str], option: str) -> bool:
    """True if ``option`` appears in ``argv``, either bare or as ``option=value``."""

    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in argv
    )


def add_legacy_option(
    parser: argparse.ArgumentParser,
    legacy_option: str,
    canonical_dest: str,
    **kwargs: object,
) -> argparse.ArgumentParser:
    """Register a hidden, deprecated spelling of an existing flag.

    The legacy flag stores into ``legacy_<canonical_dest>`` and defaults to
    ``None`` so :func:`resolve_legacy_options` can tell "not supplied" from a
    real value.
    """

    parser.add_argument(
        legacy_option,
        dest=f"legacy_{canonical_dest}",
        default=None,
        help=argparse.SUPPRESS,
        **kwargs,
    )
    return parser


def resolve_legacy_options(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    raw_argv: list[str],
    aliases: tuple[tuple[str, str, str], ...],
) -> argparse.Namespace:
    """Fold deprecated flags into their canonical destinations on ``args``.

    ``aliases`` holds ``(legacy_option, replacement, canonical_dest)`` triples.
    Supplying both spellings is an error; supplying only the legacy one warns.
    The ``legacy_*`` attributes are removed so downstream code cannot read them.
    """

    for legacy_option, replacement, canonical_dest in aliases:
        legacy_dest = f"legacy_{canonical_dest}"
        legacy_value = getattr(args, legacy_dest)
        if legacy_value is not None:
            if option_was_supplied(raw_argv, replacement):
                parser.error(f"{legacy_option} cannot be combined with {replacement}")
            print(
                f"warning: {legacy_option} is deprecated; use {replacement} instead. "
                "It will be removed in the next CLI-breaking cleanup.",
                file=sys.stderr,
            )
            setattr(args, canonical_dest, legacy_value)
        delattr(args, legacy_dest)
    return args
