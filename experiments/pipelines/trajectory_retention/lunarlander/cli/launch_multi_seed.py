"""Run the LunarLander pipeline (source + selected methods) across seeds, CPU-pinned.

Each seed runs as one `run_seed_pipeline.py` subprocess pinned to one CPU core via the
shared `experiments.pipelines._shared.multi_seed_launcher` scheduler, instead of
reimplementing job scheduling here. Replaces the old core/orchestration/launch_multi_seed.py
and core/orchestration/launch_full_pipeline_multi_seed.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines._shared.multi_seed_launcher import dedupe_preserve_order, resolve_core_pool, run_seed_pool
from experiments.pipelines.trajectory_retention.lunarlander.core.orchestration.run_paths import (
    RL_CHOICES,
    default_adapt_ewc_settings_file,
    default_adapt_ppo_settings_file,
    default_adapt_rashomon_settings_file,
    default_outputs_root,
    default_task_settings_file,
    pipeline_root,
    seed_run_dir,
    validate_rl,
)


METHOD_CHOICES = ("unconstrained", "ewc", "rashomon", "rashomon_expanded", "rashomon_plus")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source training plus selected adaptation methods across seeds, CPU-pinned.",
    )
    parser.add_argument("--pipeline", "--task-setting", type=str, dest="task_setting", default="default")
    parser.add_argument("--rl", type=str, default="ppo", choices=RL_CHOICES)
    parser.add_argument("--methods", nargs="+", choices=METHOD_CHOICES, default=list(METHOD_CHOICES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--cores", type=int, nargs="+", default=None)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--task-settings-file", type=Path, default=default_task_settings_file())
    parser.add_argument("--adapt-settings-file", type=Path, default=default_adapt_ppo_settings_file())
    parser.add_argument("--ewc-settings-file", type=Path, default=default_adapt_ewc_settings_file())
    parser.add_argument("--rashomon-settings-file", type=Path, default=default_adapt_rashomon_settings_file())
    parser.add_argument("--outputs-root", type=Path, default=default_outputs_root())
    parser.add_argument("--source-run-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume-policy", choices=["skip_completed", "rerun_all"], default="skip_completed")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--disable-task-neutralization", action="store_true")
    parser.add_argument("--total-timesteps-override", type=int, default=None)
    parser.add_argument("--ewc-lambda-override", type=float, default=None)
    parser.add_argument("--fisher-sample-size", type=int, default=10_000)
    parser.add_argument("--ewc-apply-to-critic", action="store_true")
    parser.add_argument("--rashomon-n-iters", type=int, default=None)
    parser.add_argument("--second-rashomon-n-iters", type=int, default=None)
    parser.add_argument("--rashomon-min-hard-spec", type=float, default=None)
    parser.add_argument("--rashomon-surrogate-aggregation", choices=["mean", "min"], default=None)
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
        "--pipeline", args.task_setting,
        "--rl", args.rl,
        "--seed", str(seed),
        "--methods", *args.methods,
        "--device", args.device,
        "--task-settings-file", str(args.task_settings_file),
        "--adapt-settings-file", str(args.adapt_settings_file),
        "--ewc-settings-file", str(args.ewc_settings_file),
        "--rashomon-settings-file", str(args.rashomon_settings_file),
        "--outputs-root", str(args.outputs_root),
        "--resume-policy", args.resume_policy,
    ]
    if args.source_run_root is not None:
        cmd.extend(["--source-run-root", str(args.source_run_root)])
    if args.disable_task_neutralization:
        cmd.append("--disable-task-neutralization")
    if args.total_timesteps_override is not None:
        cmd.extend(["--total-timesteps-override", str(args.total_timesteps_override)])
    if args.ewc_lambda_override is not None:
        cmd.extend(["--ewc-lambda-override", str(args.ewc_lambda_override)])
    cmd.extend(["--fisher-sample-size", str(args.fisher_sample_size)])
    if args.ewc_apply_to_critic:
        cmd.append("--ewc-apply-to-critic")
    if args.rashomon_n_iters is not None:
        cmd.extend(["--rashomon-n-iters", str(args.rashomon_n_iters)])
    if args.second_rashomon_n_iters is not None:
        cmd.extend(["--second-rashomon-n-iters", str(args.second_rashomon_n_iters)])
    if args.rashomon_min_hard_spec is not None:
        cmd.extend(["--rashomon-min-hard-spec", str(args.rashomon_min_hard_spec)])
    if args.rashomon_surrogate_aggregation is not None:
        cmd.extend(["--rashomon-surrogate-aggregation", args.rashomon_surrogate_aggregation])
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

    seeds = dedupe_preserve_order(list(args.seeds))
    if not seeds:
        raise ValueError("No seeds provided. Pass at least one seed via --seeds.")

    core_pool = _apply_max_parallel(resolve_core_pool(args.cores), args.max_parallel)
    log_dir = args.log_dir or (
        seed_run_dir(args.outputs_root, args.task_setting, 0, rl=args.rl).parent / "multi_seed_logs"
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
