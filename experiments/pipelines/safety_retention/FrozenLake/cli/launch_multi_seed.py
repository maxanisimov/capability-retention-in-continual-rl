"""Run the FrozenLake safety pipeline (source + selected methods) across seeds, CPU-pinned.

Each seed runs as one `run_seed_pipeline.py` subprocess pinned to one CPU core via the
shared `experiments.pipelines._shared.multi_seed_launcher` scheduler (the same helper
`trajectory_retention/frozenlake` uses), instead of reimplementing job scheduling here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines._shared.multi_seed_launcher import dedupe_preserve_order, resolve_core_pool, run_seed_pool
from experiments.pipelines.safety_retention.frozenlake.core.paths import default_outputs_root, layout_run_root, pipeline_root
from experiments.pipelines.safety_retention.frozenlake.core.training_common import (
    RL_CHOICES,
    resolve_deterministic,
    validate_deterministic,
    validate_rl,
)


METHOD_CHOICES = ("unconstrained", "ewc", "rashomon")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source training plus selected adaptation methods across seeds, CPU-pinned.",
    )
    parser.add_argument("--pipeline", "--layout", type=str, dest="layout", default="diagonal_4x4")
    parser.add_argument("--rl", type=str, default="ppo", choices=RL_CHOICES)
    parser.add_argument("--methods", nargs="+", choices=METHOD_CHOICES, default=list(METHOD_CHOICES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--cores", type=int, nargs="+", default=None)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--outputs-root", type=Path, default=default_outputs_root())
    parser.add_argument("--source-run-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override dynamics regime. Defaults to the pipeline's task definition when omitted.",
    )
    parser.add_argument("--resume-policy", choices=["skip_completed", "rerun_all"], default="skip_completed")
    parser.add_argument("--log-dir", type=Path, default=None)
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _apply_max_parallel(core_pool: list[int], max_parallel: int | None) -> list[int]:
    if max_parallel is None:
        return core_pool
    if max_parallel <= 0:
        raise ValueError(f"--max-parallel must be > 0, got {max_parallel}.")
    if max_parallel > len(core_pool):
        raise ValueError(f"--max-parallel={max_parallel} exceeds selected unique core count {len(core_pool)}.")
    return core_pool[:max_parallel]


def build_cmd(args: argparse.Namespace, seed: int) -> list[str]:
    cmd = [
        sys.executable,
        str(pipeline_root() / "cli" / "run_seed_pipeline.py"),
        "--pipeline", args.layout,
        "--rl", args.rl,
        "--seed", str(seed),
        "--methods", *args.methods,
        "--device", args.device,
        "--outputs-root", str(args.outputs_root),
        "--resume-policy", args.resume_policy,
        "--deterministic" if args.deterministic else "--no-deterministic",
    ]
    if args.source_run_root is not None:
        cmd.extend(["--source-run-root", str(args.source_run_root)])
    if args.total_timesteps_override is not None:
        cmd.extend(["--total-timesteps-override", str(args.total_timesteps_override)])
    if args.safety_finetune_lr is not None:
        cmd.extend(["--safety-finetune-lr", str(args.safety_finetune_lr)])
    if args.safety_finetune_max_epochs is not None:
        cmd.extend(["--safety-finetune-max-epochs", str(args.safety_finetune_max_epochs)])
    if args.ewc_lambda is not None:
        cmd.extend(["--ewc-lambda", str(args.ewc_lambda)])
    if args.fisher_sample_size is not None:
        cmd.extend(["--fisher-sample-size", str(args.fisher_sample_size)])
    if args.rashomon_n_iters is not None:
        cmd.extend(["--rashomon-n-iters", str(args.rashomon_n_iters)])
    if args.inverse_temp_start is not None:
        cmd.extend(["--inverse-temp-start", str(args.inverse_temp_start)])
    if args.inverse_temp_max is not None:
        cmd.extend(["--inverse-temp-max", str(args.inverse_temp_max)])
    if args.rashomon_checkpoint is not None:
        cmd.extend(["--rashomon-checkpoint", str(args.rashomon_checkpoint)])
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    validate_rl(args.rl)
    args.deterministic = resolve_deterministic(args.layout, args.deterministic)
    validate_deterministic(args.deterministic)

    seeds = dedupe_preserve_order(list(args.seeds))
    if not seeds:
        raise ValueError("No seeds provided. Pass at least one seed via --seeds.")

    core_pool = _apply_max_parallel(resolve_core_pool(args.cores), args.max_parallel)
    log_dir = args.log_dir or (
        layout_run_root(args.outputs_root, args.layout, args.rl, args.deterministic) / "multi_seed_logs"
    )

    print(
        f"Launching methods={list(args.methods)} for {len(seeds)} seed(s) on "
        f"{len(core_pool)} unique CPU core(s): {core_pool}",
    )

    if args.dry_run:
        for seed in seeds:
            cmd = build_cmd(args, seed)
            print(f"[dry-run] seed={seed} log={log_dir / f'seed_{seed}.log'}")
            print("  " + " ".join(cmd))
        return 0

    return run_seed_pool(
        seeds=seeds,
        cores=core_pool,
        build_cmd=lambda seed: build_cmd(args, seed),
        log_dir=log_dir,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
