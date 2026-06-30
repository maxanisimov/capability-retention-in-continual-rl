"""Train CPO on a local MASA-style tabular environment."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

from projects.safe_policy_optimisation.stages import train_ppo_lagrangian  # noqa: E402
from projects.safe_policy_optimisation.utils.safe_rl import CPO_ALGORITHM_NAMES  # noqa: E402

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "projects" / "safe_policy_optimisation" / "artifacts" / "cpo"
)


def build_parser() -> argparse.ArgumentParser:
    return train_ppo_lagrangian.build_parser(
        algorithm_names=CPO_ALGORITHM_NAMES,
        default_algorithms=list(CPO_ALGORITHM_NAMES),
        description="Train CPO on a MASA-style Gymnasium env and report cost violations.",
        output_dir=DEFAULT_OUTPUT_DIR,
        algorithm_help="CPO algorithm selection; only 'cpo' is supported.",
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    args.algorithms = list(CPO_ALGORITHM_NAMES)
    return train_ppo_lagrangian.run(args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
