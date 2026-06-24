"""Run source training plus the requested adaptation methods for one seed, in-process."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.safety_retention.frozenlake.cli import (
    adapt_ewc,
    adapt_rashomon,
    adapt_unconstrained,
    train_source,
)
from experiments.pipelines.safety_retention.frozenlake.core.paths import (
    default_outputs_root,
    is_mode_complete,
    resolve_source_run_dir,
)
from experiments.pipelines.safety_retention.frozenlake.core.training_common import (
    RL_CHOICES,
    validate_deterministic,
    validate_rl,
)


METHOD_TO_MODE = {
    "unconstrained": "downstream_unconstrained",
    "ewc": "downstream_ewc",
    "rashomon": "downstream_rashomon",
}
METHOD_ORDER = ("unconstrained", "ewc", "rashomon")
METHOD_TO_CLI = {
    "unconstrained": adapt_unconstrained,
    "ewc": adapt_ewc,
    "rashomon": adapt_rashomon,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source training plus selected adaptation methods for one seed.",
    )
    parser.add_argument("--pipeline", "--layout", type=str, dest="layout", default="diagonal_4x4")
    parser.add_argument("--rl", type=str, default="ppo", choices=RL_CHOICES)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=[])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--outputs-root", type=Path, default=default_outputs_root())
    parser.add_argument(
        "--source-run-root",
        type=Path,
        default=None,
        help="Optional root containing pre-trained source checkpoints, if different from --outputs-root.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume-policy", choices=["skip_completed", "rerun_all"], default="skip_completed")
    parser.add_argument("--total-timesteps-override", type=int, default=None)
    parser.add_argument("--safety-finetune-lr", type=float, default=None)
    parser.add_argument("--safety-finetune-max-epochs", type=int, default=None)
    parser.add_argument("--ewc-lambda", type=float, default=None)
    parser.add_argument("--fisher-sample-size", type=int, default=None)
    parser.add_argument("--rashomon-n-iters", type=int, default=None)
    parser.add_argument("--inverse-temp-start", type=int, default=None)
    parser.add_argument("--inverse-temp-max", type=int, default=None)
    parser.add_argument("--rashomon-checkpoint", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _common_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "--pipeline", args.layout,
        "--rl", args.rl,
        "--seed", str(args.seed),
        "--device", args.device,
        "--outputs-root", str(args.outputs_root),
        "--deterministic" if args.deterministic else "--no-deterministic",
    ]
    if args.total_timesteps_override is not None:
        argv.extend(["--total-timesteps-override", str(args.total_timesteps_override)])
    return argv


def _run_source(args: argparse.Namespace) -> None:
    if args.source_run_root is not None:
        print(f"Using pre-trained source checkpoints from {args.source_run_root}; skipping source training.")
        return
    if args.resume_policy == "skip_completed" and is_mode_complete(
        args.outputs_root, args.layout, args.rl, args.deterministic, args.seed, "source",
    ):
        print(f"[skip] source already complete for seed={args.seed}")
        return
    argv = _common_argv(args)
    if args.safety_finetune_lr is not None:
        argv.extend(["--safety-finetune-lr", str(args.safety_finetune_lr)])
    if args.safety_finetune_max_epochs is not None:
        argv.extend(["--safety-finetune-max-epochs", str(args.safety_finetune_max_epochs)])
    rc = train_source.main(argv)
    if rc:
        raise RuntimeError(f"Source training failed for seed={args.seed} (rc={rc}).")


def _source_run_dir_argv(args: argparse.Namespace) -> list[str]:
    if args.source_run_root is None:
        return []
    source_dir = resolve_source_run_dir(args.source_run_root, args.layout, args.rl, args.deterministic, args.seed)
    return ["--source-run-dir", str(source_dir)]


def _run_method(args: argparse.Namespace, method: str) -> None:
    mode = METHOD_TO_MODE[method]
    if args.resume_policy == "skip_completed" and is_mode_complete(
        args.outputs_root, args.layout, args.rl, args.deterministic, args.seed, mode,
    ):
        print(f"[skip] {mode} already complete for seed={args.seed}")
        return

    argv = _common_argv(args) + _source_run_dir_argv(args)
    if method == "ewc":
        if args.ewc_lambda is not None:
            argv.extend(["--ewc-lambda", str(args.ewc_lambda)])
        if args.fisher_sample_size is not None:
            argv.extend(["--fisher-sample-size", str(args.fisher_sample_size)])
    if method == "rashomon":
        if args.rashomon_n_iters is not None:
            argv.extend(["--rashomon-n-iters", str(args.rashomon_n_iters)])
        if args.inverse_temp_start is not None:
            argv.extend(["--inverse-temp-start", str(args.inverse_temp_start)])
        if args.inverse_temp_max is not None:
            argv.extend(["--inverse-temp-max", str(args.inverse_temp_max)])
        if args.rashomon_checkpoint is not None:
            argv.extend(["--rashomon-checkpoint", str(args.rashomon_checkpoint)])

    rc = METHOD_TO_CLI[method].main(argv)
    if rc:
        raise RuntimeError(f"{mode} failed for seed={args.seed} (rc={rc}).")


def run_seed_pipeline(args: argparse.Namespace) -> None:
    validate_rl(args.rl)
    validate_deterministic(args.deterministic)
    requested = [method for method in METHOD_ORDER if method in args.methods]

    _run_source(args)
    for method in requested:
        _run_method(args, method)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
        validate_rl(args.rl)
        validate_deterministic(args.deterministic)
        requested = [method for method in METHOD_ORDER if method in args.methods]
        print(f"Dry run: seed={args.seed} layout={args.layout} methods=[source, {', '.join(requested)}]")
        return 0
    run_seed_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
