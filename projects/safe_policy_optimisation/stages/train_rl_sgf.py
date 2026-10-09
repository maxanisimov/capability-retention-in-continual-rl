"""Train RL-SGF (anytime safe gradient flow) on a local MASA-style tabular environment."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

from projects.safe_policy_optimisation.stages import train_ppo_lagrangian  # noqa: E402
from projects.safe_policy_optimisation.utils.safe_rl import RL_SGF_ALGORITHM_NAMES  # noqa: E402

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "projects" / "safe_policy_optimisation" / "artifacts" / "rl_sgf"
)


def build_parser() -> argparse.ArgumentParser:
    parser = train_ppo_lagrangian.build_parser(
        algorithm_names=RL_SGF_ALGORITHM_NAMES,
        default_algorithms=list(RL_SGF_ALGORITHM_NAMES),
        description=(
            "Train RL-SGF (Mestres et al., L4DC 2025) on a MASA-style Gymnasium env and "
            "report cost violations. Use a positive --cost-limit: with a zero budget and "
            "no sampled cost the RL-SGF step is exactly zero."
        ),
        output_dir=DEFAULT_OUTPUT_DIR,
        algorithm_help="RL-SGF algorithm selection; only 'rl_sgf' is supported.",
    )
    group = parser.add_argument_group("RL-SGF")
    group.add_argument(
        "--rl-sgf-step-size",
        type=float,
        default=0.1,
        help="Proximal step size h; requires h * alpha < 1 (and h < 1/L for the formal guarantee).",
    )
    group.add_argument(
        "--rl-sgf-alpha",
        type=float,
        default=1.0,
        help="Safe-gradient-flow gain alpha (how fast the cost may approach the budget).",
    )
    group.add_argument(
        "--rl-sgf-episodes-per-iter",
        type=int,
        default=100,
        help="Complete on-policy episodes N used for each estimate (paper: 100-200).",
    )
    group.add_argument(
        "--rl-sgf-baseline",
        choices=("none", "critic"),
        default="none",
        help="REINFORCE baseline: 'none' as in the paper, or learned reward/cost critics.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    args.algorithms = list(RL_SGF_ALGORITHM_NAMES)
    return train_ppo_lagrangian.run(args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
