"""Single entrypoint: launch a multi-seed LunarLander pipeline run and aggregate results.

    python run_pipeline.py --pipeline deterministic__default_to_underpowered_vehicle --rl ppo \\
        --methods unconstrained ewc rashomon rashomon_expanded rashomon_plus --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from projects.safe_crl.pipelines.trajectory_retention.lunarlander.cli import launch_multi_seed
from projects.safe_crl.pipelines.trajectory_retention.lunarlander.core.orchestration.run_paths import (
    RL_CHOICES,
    default_outputs_root,
    pipeline_root,
    validate_rl,
)


METHOD_CHOICES = ("unconstrained", "ewc", "rashomon", "rashomon_expanded", "rashomon_plus")
METHOD_TO_POLICY = {
    "unconstrained": "downstream_unconstrained",
    "ewc": "downstream_ewc",
    "rashomon": "downstream_rashomon",
    "rashomon_expanded": "downstream_rashomon_expanded",
    "rashomon_plus": "downstream_rashomon_plus",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a multi-seed LunarLander pipeline run, then aggregate results.",
    )
    parser.add_argument("--pipeline", "--task-setting", type=str, dest="task_setting", default="default")
    parser.add_argument("--rl", type=str, default="ppo", choices=RL_CHOICES)
    parser.add_argument("--methods", nargs="+", choices=METHOD_CHOICES, default=list(METHOD_CHOICES))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--cores", type=int, nargs="+", default=None)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--outputs-root", type=Path, default=default_outputs_root())
    parser.add_argument("--source-run-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume-policy", choices=["skip_completed", "rerun_all"], default="skip_completed")
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
    parser.add_argument("--metric-groups", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _launch_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "--pipeline", args.task_setting,
        "--rl", args.rl,
        "--methods", *args.methods,
        "--seeds", *[str(seed) for seed in args.seeds],
        "--device", args.device,
        "--outputs-root", str(args.outputs_root),
        "--resume-policy", args.resume_policy,
        "--poll-seconds", str(args.poll_seconds),
        "--fisher-sample-size", str(args.fisher_sample_size),
    ]
    if args.cores is not None:
        argv.extend(["--cores", *[str(core) for core in args.cores]])
    if args.max_parallel is not None:
        argv.extend(["--max-parallel", str(args.max_parallel)])
    if args.source_run_root is not None:
        argv.extend(["--source-run-root", str(args.source_run_root)])
    if args.disable_task_neutralization:
        argv.append("--disable-task-neutralization")
    if args.total_timesteps_override is not None:
        argv.extend(["--total-timesteps-override", str(args.total_timesteps_override)])
    if args.ewc_lambda_override is not None:
        argv.extend(["--ewc-lambda-override", str(args.ewc_lambda_override)])
    if args.ewc_apply_to_critic:
        argv.append("--ewc-apply-to-critic")
    if args.rashomon_n_iters is not None:
        argv.extend(["--rashomon-n-iters", str(args.rashomon_n_iters)])
    if args.second_rashomon_n_iters is not None:
        argv.extend(["--second-rashomon-n-iters", str(args.second_rashomon_n_iters)])
    if args.rashomon_min_hard_spec is not None:
        argv.extend(["--rashomon-min-hard-spec", str(args.rashomon_min_hard_spec)])
    if args.rashomon_surrogate_aggregation is not None:
        argv.extend(["--rashomon-surrogate-aggregation", args.rashomon_surrogate_aggregation])
    if args.inverse_temp_start is not None:
        argv.extend(["--inverse-temp-start", str(args.inverse_temp_start)])
    if args.inverse_temp_max is not None:
        argv.extend(["--inverse-temp-max", str(args.inverse_temp_max)])
    if args.rashomon_checkpoint is not None:
        argv.extend(["--rashomon-checkpoint", str(args.rashomon_checkpoint)])
    if args.dry_run:
        argv.append("--dry-run")
    return argv


def _run_aggregate(args: argparse.Namespace) -> int:
    policies = ["noadapt"] + [METHOD_TO_POLICY[method] for method in args.methods]
    cmd = [
        sys.executable,
        str(pipeline_root() / "cli" / "aggregate_layout_metrics.py"),
        "--pipeline", args.task_setting,
        "--rl", args.rl,
        "--outputs-root", str(args.outputs_root),
        "--policies", *policies,
        "--compute-relative-to-source",
    ]
    if args.metric_groups is not None:
        cmd.extend(["--metric-groups", *args.metric_groups])
    print("\nRunning aggregate metrics export:")
    print("  " + " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    validate_rl(args.rl)

    rc = launch_multi_seed.main(_launch_argv(args))
    if rc:
        print("Multi-seed launch reported failures; skipping aggregation.")
        return rc
    if args.dry_run:
        return 0

    return _run_aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
