"""Entrypoint: run Rashomon-constrained downstream PPO adaptation for one seed."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from projects.safe_crl.pipelines.safety_retention.frozenlake.core.training_common import import_method_module


def main(argv: list[str] | None = None) -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--rl", default="ppo")
    pre_args, _ = pre_parser.parse_known_args(argv)
    module = import_method_module(pre_args.rl, "adapt_rashomon")
    return module.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
