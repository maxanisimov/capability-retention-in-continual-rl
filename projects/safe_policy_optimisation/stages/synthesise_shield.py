"""Synthesise a tabular safety shield for safe-policy-optimisation runs.

This is a project-local entry point that reuses the implementation in
``projects.safe_crl.pipelines.safety_retention.synthesise_shield``. It keeps
shield synthesis separate from shielded policy optimisation while storing the
result under this project's artifacts directory by default.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


REPO_ROOT = Path(__file__).resolve().parents[3]

from projects.safe_crl.pipelines.safety_retention.synthesise_shield import (  # noqa: E402
    FROZEN_LAKE_ENV,
    run_frozenlake_task,
    run_masa_env,
)
from projects.safe_crl.pipelines.safety_retention.task_library import (  # noqa: E402
    environment_subdir,
    load_masa_task,
)
from projects.safe_policy_optimisation.utils.log import log_info  # noqa: E402


PROJECT_ROOT = REPO_ROOT / "projects" / "safe_policy_optimisation"


def default_output_dir(env_id: str, task: str) -> Path:
    """Default shield directory inside the safe_policy_optimisation project."""

    return PROJECT_ROOT / "artifacts" / "shields" / environment_subdir(env_id) / task


def _parse_json_dict(value: str | None, *, flag: str) -> dict:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag} must be a JSON object, got {type(parsed).__name__}.")
    return parsed


def _resolve_risk_threshold(constraint_kwargs: dict) -> float:
    if "alpha" in constraint_kwargs:
        return float(constraint_kwargs["alpha"])
    return 0.0


def _resolve_max_episode_steps(env_id: str, task: str, cli_value: int | None) -> int | None:
    if cli_value is not None:
        return int(cli_value)
    if env_id == FROZEN_LAKE_ENV:
        return None
    task_block = load_masa_task(env_id, task)
    if "max_episode_steps" in task_block and task_block["max_episode_steps"] is not None:
        return int(task_block["max_episode_steps"])
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesise a probabilistic safety shield and save shield_q.pt under "
            "projects/safe_policy_optimisation/artifacts/shields by default."
        ),
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help=(
            "Environment id. Custom...-v0 ids use the local MASA-style tabular envs; "
            "FrozenLake-v1 uses the safety-retention FrozenLake task library."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task-library key defining the environment instance.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed passed to env.reset during synthesis.")
    parser.add_argument(
        "--constraint",
        type=str,
        default="PCTL",
        help="Constraint family recorded in the saved shield metadata. Default: PCTL.",
    )
    parser.add_argument(
        "--constraint-kwargs",
        type=str,
        default=None,
        help='JSON constraint kwargs recorded in metadata, e.g. \'{"alpha": 0.01}\'.',
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=1e-10,
        help="Value-iteration convergence tolerance for probabilistic shield synthesis.",
    )
    parser.add_argument(
        "--max-vi-steps",
        type=int,
        default=1000,
        help="Maximum value-iteration steps for probabilistic shield synthesis.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help=(
            "Maximum episode length used when constructing the MASA environment. "
            "If omitted, the task's max_episode_steps is used when present."
        ),
    )
    parser.add_argument(
        "--init-safety-bound",
        type=float,
        default=0.5,
        help="MASA ProbShieldWrapperDisc initial safety bound recorded in the shield metadata.",
    )
    parser.add_argument(
        "--granularity",
        type=int,
        default=20,
        help="MASA ProbShieldWrapperDisc safety-bound granularity recorded in the shield metadata.",
    )
    parser.add_argument(
        "--unsafe-cost-threshold",
        type=float,
        default=0.5,
        help="Cost threshold used to classify a state as unsafe.",
    )
    parser.add_argument(
        "--env-kwargs",
        type=str,
        default=None,
        help="Optional JSON dict overriding task env_kwargs/stochasticity for this synthesis run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory. The script writes shield_q.pt inside this directory.",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    if args.env is None:
        raise ValueError("--env is required for shield synthesis.")
    if args.task is None:
        raise ValueError("--task is required for shield synthesis.")
    constraint_kwargs = _parse_json_dict(args.constraint_kwargs, flag="--constraint-kwargs")
    args.constraint_kwargs = constraint_kwargs
    args.risk_threshold = _resolve_risk_threshold(constraint_kwargs)
    args.max_episode_steps = _resolve_max_episode_steps(args.env, args.task, args.max_episode_steps)

    output_dir = args.output_dir or default_output_dir(args.env, args.task)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "shield_q.pt"

    if args.env == FROZEN_LAKE_ENV:
        start_safety = run_frozenlake_task(args, output_path)
        plot_hint = (
            "python projects/safe_crl/pipelines/safety_retention/plot_shield.py "
            f"--task {args.task} --shield-path {output_path}"
        )
    else:
        start_safety = run_masa_env(args, output_path)
        plot_hint = (
            "python projects/safe_crl/pipelines/safety_retention/plot_shield.py "
            f"--env {args.env} --task {args.task} --shield-path {output_path}"
        )

    log_info(f"Saved shield Q-function to {output_path}")
    log_info(f"Start-state eventual-safety value: {start_safety:.3f}")
    log_info(f"Plot it with: {plot_hint}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
