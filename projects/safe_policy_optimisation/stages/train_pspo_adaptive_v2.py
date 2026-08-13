"""Train AdaptiveSafePPOV2: projected updates + projection-triggered regions."""

from __future__ import annotations

import argparse
import math
import sys
from typing import Sequence

from projects.safe_policy_optimisation.stages import train_pspo_adaptive

ALGORITHM_NAME = "adaptive_safe_ppo_v2"


def estimate_max_optimizer_steps(args: argparse.Namespace) -> int:
    """Worst-case PPO optimizer steps for this project's single-env setup."""

    train_phases = int(math.ceil(float(args.total_timesteps) / float(args.n_steps)))
    minibatches_per_phase = int(math.ceil(float(args.n_steps) / float(args.batch_size)))
    return int(train_phases * int(args.n_epochs) * minibatches_per_phase)


def build_parser() -> argparse.ArgumentParser:
    parser = train_pspo_adaptive.build_parser()
    parser.description = (
        "Train AdaptiveSafePPOV2 from a saved shield and safe base policy."
    )
    for action in parser._actions:
        if action.dest == "rashomon_n_iters":
            action.default = None
            action.help = (
                "Optimization budget per safe-region computation. Defaults to 100 "
                "unless --rashomon-total-iters is used."
            )
            break
    parser.add_argument(
        "--region-update-mode",
        choices=("union", "replace"),
        default="union",
        help="Whether a newly certified region replaces the old one or is unioned with it.",
    )
    parser.add_argument(
        "--rashomon-total-iters",
        type=int,
        default=None,
        help=(
            "Hard total Rashomon iteration budget across the initial region and all "
            "possible projection-triggered recomputations. Mutually exclusive with "
            "--rashomon-n-iters."
        ),
    )
    parser.add_argument(
        "--rashomon-initial-n-iters",
        type=int,
        default=None,
        help=(
            "Rashomon iterations for the initial safe-region computation. Defaults "
            "to --rashomon-n-iters in per-computation mode, or --rashomon-total-iters "
            "in total-budget mode."
        ),
    )
    parser.add_argument(
        "--rashomon-recompute-n-iters",
        type=int,
        default=None,
        help=(
            "Rashomon iterations for each projection-triggered safe-region "
            "recomputation. Defaults to the initial-region iteration count."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    explicit_per_computation = any(
        arg == "--rashomon-n-iters" or arg.startswith("--rashomon-n-iters=")
        for arg in raw_argv
    )
    explicit_initial_iters = any(
        arg == "--rashomon-initial-n-iters" or arg.startswith("--rashomon-initial-n-iters=")
        for arg in raw_argv
    )
    explicit_recompute_iters = any(
        arg == "--rashomon-recompute-n-iters"
        or arg.startswith("--rashomon-recompute-n-iters=")
        for arg in raw_argv
    )
    if args.rashomon_total_iters is not None and explicit_per_computation:
        parser.error("--rashomon-total-iters and --rashomon-n-iters are mutually exclusive.")
    if explicit_per_computation and (explicit_initial_iters or explicit_recompute_iters):
        parser.error(
            "--rashomon-n-iters cannot be combined with "
            "--rashomon-initial-n-iters or --rashomon-recompute-n-iters."
        )
    for name in ("rashomon_total_iters", "rashomon_initial_n_iters", "rashomon_recompute_n_iters"):
        value = getattr(args, name)
        if value is not None and int(value) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")

    max_region_computations = 1 + estimate_max_optimizer_steps(args)
    args.adaptive_version = "v2"
    args.algorithm_name = ALGORITHM_NAME
    args.adaptive_granularity = "gradient_step"
    args.unsafe_update_strategy = "rashomon_project"
    args.rashomon_max_region_computations = max_region_computations
    if args.rashomon_total_iters is None:
        args.rashomon_budget_mode = "per_computation"
        args.rashomon_n_iters = 100 if args.rashomon_n_iters is None else int(args.rashomon_n_iters)
        args.rashomon_initial_n_iters = (
            args.rashomon_n_iters
            if args.rashomon_initial_n_iters is None
            else int(args.rashomon_initial_n_iters)
        )
        args.rashomon_recompute_n_iters = (
            args.rashomon_initial_n_iters
            if args.rashomon_recompute_n_iters is None
            else int(args.rashomon_recompute_n_iters)
        )
        args.rashomon_n_iters = max(
            int(args.rashomon_initial_n_iters),
            int(args.rashomon_recompute_n_iters),
        )
    else:
        args.rashomon_budget_mode = "total"
        args.rashomon_initial_n_iters = (
            int(args.rashomon_total_iters)
            if args.rashomon_initial_n_iters is None
            else int(args.rashomon_initial_n_iters)
        )
        args.rashomon_recompute_n_iters = (
            int(args.rashomon_initial_n_iters)
            if args.rashomon_recompute_n_iters is None
            else int(args.rashomon_recompute_n_iters)
        )
        if args.rashomon_initial_n_iters > args.rashomon_total_iters:
            parser.error("--rashomon-initial-n-iters cannot exceed --rashomon-total-iters.")
        if args.rashomon_recompute_n_iters > args.rashomon_total_iters:
            parser.error("--rashomon-recompute-n-iters cannot exceed --rashomon-total-iters.")
        args.rashomon_n_iters = max(
            int(args.rashomon_initial_n_iters),
            int(args.rashomon_recompute_n_iters),
        )
        if args.rashomon_checkpoint is None:
            args.rashomon_checkpoint = max(1, int(args.rashomon_n_iters) // 10)
    return args


def run(args: argparse.Namespace) -> dict:
    return train_pspo_adaptive.run(args)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
