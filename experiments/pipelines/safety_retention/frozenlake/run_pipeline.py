"""Single entrypoint: launch a multi-seed FrozenLake safety run and aggregate results.

    python run_pipeline.py --pipeline diagonal_4x4 --rl ppo --deterministic \\
        --methods unconstrained ewc rashomon --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.safety_retention.frozenlake.cli import aggregate_metrics, launch_multi_seed
from experiments.pipelines.safety_retention.frozenlake.core.paths import default_outputs_root
from experiments.pipelines.safety_retention.frozenlake.core.training_common import (
    RL_CHOICES,
    validate_deterministic,
    validate_rl,
)


METHOD_CHOICES = ("unconstrained", "ewc", "rashomon")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a multi-seed FrozenLake safety pipeline run, then aggregate results.",
    )
    parser.add_argument("--pipeline", "--layout", type=str, dest="layout", default="diagonal_4x4")
    parser.add_argument("--rl", type=str, default="ppo", choices=RL_CHOICES)
    parser.add_argument("--methods", nargs="+", choices=METHOD_CHOICES, default=list(METHOD_CHOICES))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--cores", type=int, nargs="+", default=None)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--outputs-root", type=Path, default=default_outputs_root())
    parser.add_argument("--source-run-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume-policy", choices=["skip_completed", "rerun_all"], default="skip_completed")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--total-timesteps-override", type=int, default=None)
    parser.add_argument("--safety-finetune-lr", type=float, default=None)
    parser.add_argument("--safety-finetune-max-epochs", type=int, default=None)
    parser.add_argument("--ewc-lambda", type=float, default=None)
    parser.add_argument("--fisher-sample-size", type=int, default=None)
    parser.add_argument("--rashomon-n-iters", type=int, default=None)
    parser.add_argument("--inverse-temp-start", type=int, default=None)
    parser.add_argument("--inverse-temp-max", type=int, default=None)
    parser.add_argument("--rashomon-checkpoint", type=int, default=None)
    parser.add_argument("--metrics", nargs="+", default=None, help="Metrics to aggregate (default: all 6).")
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _launch_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "--pipeline", args.layout,
        "--rl", args.rl,
        "--methods", *args.methods,
        "--seeds", *[str(seed) for seed in args.seeds],
        "--device", args.device,
        "--outputs-root", str(args.outputs_root),
        "--resume-policy", args.resume_policy,
        "--poll-seconds", str(args.poll_seconds),
        "--deterministic" if args.deterministic else "--no-deterministic",
    ]
    if args.cores is not None:
        argv.extend(["--cores", *[str(core) for core in args.cores]])
    if args.max_parallel is not None:
        argv.extend(["--max-parallel", str(args.max_parallel)])
    if args.source_run_root is not None:
        argv.extend(["--source-run-root", str(args.source_run_root)])
    if args.total_timesteps_override is not None:
        argv.extend(["--total-timesteps-override", str(args.total_timesteps_override)])
    if args.safety_finetune_lr is not None:
        argv.extend(["--safety-finetune-lr", str(args.safety_finetune_lr)])
    if args.safety_finetune_max_epochs is not None:
        argv.extend(["--safety-finetune-max-epochs", str(args.safety_finetune_max_epochs)])
    if args.ewc_lambda is not None:
        argv.extend(["--ewc-lambda", str(args.ewc_lambda)])
    if args.fisher_sample_size is not None:
        argv.extend(["--fisher-sample-size", str(args.fisher_sample_size)])
    if args.rashomon_n_iters is not None:
        argv.extend(["--rashomon-n-iters", str(args.rashomon_n_iters)])
    if args.inverse_temp_start is not None:
        argv.extend(["--inverse-temp-start", str(args.inverse_temp_start)])
    if args.inverse_temp_max is not None:
        argv.extend(["--inverse-temp-max", str(args.inverse_temp_max)])
    if args.rashomon_checkpoint is not None:
        argv.extend(["--rashomon-checkpoint", str(args.rashomon_checkpoint)])
    if args.dry_run:
        argv.append("--dry-run")
    return argv


def _aggregate_argv(args: argparse.Namespace) -> list[str]:
    method_dir_names = {"unconstrained": "downstream_unconstrained", "ewc": "downstream_ewc", "rashomon": "downstream_rashomon"}
    methods = ["noadapt"] + [method_dir_names[method] for method in args.methods]
    argv = [
        "--pipeline", args.layout,
        "--rl", args.rl,
        "--outputs-root", str(args.outputs_root),
        "--methods", *methods,
        "--seeds", *[str(seed) for seed in args.seeds],
        "--precision", str(args.precision),
        "--deterministic" if args.deterministic else "--no-deterministic",
    ]
    if args.metrics is not None:
        argv.extend(["--metrics", *args.metrics])
    return argv


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    validate_rl(args.rl)
    validate_deterministic(args.deterministic)

    rc = launch_multi_seed.main(_launch_argv(args))
    if rc:
        print("Multi-seed launch reported failures; skipping aggregation.")
        return rc
    if args.dry_run:
        return 0

    return aggregate_metrics.main(_aggregate_argv(args))


if __name__ == "__main__":
    raise SystemExit(main())
