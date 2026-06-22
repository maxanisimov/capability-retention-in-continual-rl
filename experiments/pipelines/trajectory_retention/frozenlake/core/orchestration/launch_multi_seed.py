"""Run one FrozenLake mode across multiple seeds with CPU pinning."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from experiments.pipelines._shared.multi_seed_launcher import (
    dedupe_preserve_order,
    resolve_core_pool,
    run_seed_pool,
)
from experiments.pipelines.trajectory_retention.frozenlake.core.orchestration.run_paths import (
    NOADAPT_POLICY_SUBDIR,
    default_adapt_ewc_settings_file,
    default_adapt_ppo_settings_file,
    default_adapt_rashomon_settings_file,
    default_downstream_envs_file,
    default_outputs_root,
    default_source_envs_file,
    default_train_source_settings_file,
    pipeline_root,
    resolve_default_source_run_dir,
)


MODE_TO_CLI = {
    "source": "train_source.py",
    "downstream_unconstrained": "adapt_unconstrained.py",
    "downstream_ewc": "adapt_ewc.py",
    "downstream_rashomon": "adapt_rashomon.py",
}

MODE_TO_DEFAULT_RUN_SUBDIR = {
    "downstream_unconstrained": "downstream_unconstrained",
    "downstream_ewc": "downstream_ewc",
    "downstream_rashomon": "downstream_rashomon",
}


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run one FrozenLake mode across multiple seeds while pinning each run to a CPU core.",
    )
    parser.add_argument("--mode", required=True, choices=sorted(MODE_TO_CLI))
    parser.add_argument("--pipeline", "--layout", type=str, dest="layout", default="diagonal_30x30")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--cores", type=int, nargs="+", default=None)
    parser.add_argument("--source-env-file", type=Path, default=default_source_envs_file())
    parser.add_argument("--downstream-env-file", type=Path, default=default_downstream_envs_file())
    parser.add_argument("--source-settings-file", type=Path, default=default_train_source_settings_file())
    parser.add_argument("--adapt-settings-file", type=Path, default=default_adapt_ppo_settings_file())
    parser.add_argument("--ewc-settings-file", type=Path, default=default_adapt_ewc_settings_file())
    parser.add_argument("--rashomon-settings-file", type=Path, default=default_adapt_rashomon_settings_file())
    parser.add_argument("--outputs-root", type=Path, default=default_outputs_root())
    parser.add_argument("--output-dir", type=Path, default=default_outputs_root())
    parser.add_argument("--source-run-root", type=Path, default=None)
    parser.add_argument("--run-subdir", type=str, default=None)
    parser.add_argument("--activation", choices=["tanh", "relu"], default="relu")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--disable-task-neutralization", action="store_true")
    parser.add_argument("--total-timesteps-override", type=int, default=None)
    parser.add_argument("--ewc-lambda-override", type=float, default=None)
    parser.add_argument("--fisher-sample-size", type=int, default=10_000)
    parser.add_argument("--ewc-apply-to-critic", action="store_true")
    parser.add_argument("--rashomon-n-iters", type=int, default=None)
    parser.add_argument("--rashomon-min-hard-spec", type=float, default=None)
    parser.add_argument("--rashomon-surrogate-aggregation", choices=["mean", "min"], default=None)
    parser.add_argument("--inverse-temp-start", type=int, default=None)
    parser.add_argument("--inverse-temp-max", type=int, default=None)
    parser.add_argument("--rashomon-checkpoint", type=int, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser.parse_known_args(argv)


def _worker_script(mode: str) -> Path:
    return pipeline_root() / "cli" / MODE_TO_CLI[mode]


def _build_worker_cmd(args: argparse.Namespace, *, seed: int, passthrough: list[str]) -> list[str]:
    cmd = [sys.executable, str(_worker_script(args.mode))]

    common = [
        "--pipeline",
        str(args.layout),
        "--seed",
        str(seed),
        "--device",
        str(args.device),
        "--activation",
        str(args.activation),
        "--source-env-file",
        str(args.source_env_file),
        "--downstream-env-file",
        str(args.downstream_env_file),
    ]
    cmd.extend(common)

    if args.mode == "source":
        cmd.extend(
            [
                "--settings-file",
                str(args.source_settings_file),
                "--adapt-settings-file",
                str(args.adapt_settings_file),
                "--output-dir",
                str(args.output_dir),
            ],
        )
    else:
        cmd.extend(
            [
                "--source-settings-file",
                str(args.source_settings_file),
                "--adapt-settings-file",
                str(args.adapt_settings_file),
                "--outputs-root",
                str(args.outputs_root),
            ],
        )
        run_subdir = args.run_subdir or MODE_TO_DEFAULT_RUN_SUBDIR[args.mode]
        cmd.extend(["--run-subdir", run_subdir])
        if args.source_run_root is not None:
            cmd.extend(
                [
                    "--source-run-dir",
                    str(resolve_default_source_run_dir(args.source_run_root, args.layout, seed)),
                ],
            )
        if args.disable_task_neutralization:
            cmd.append("--disable-task-neutralization")
        if args.total_timesteps_override is not None:
            cmd.extend(["--total-timesteps-override", str(args.total_timesteps_override)])
        if args.mode == "downstream_ewc":
            cmd.extend(["--ewc-settings-file", str(args.ewc_settings_file)])
            cmd.extend(["--fisher-sample-size", str(args.fisher_sample_size)])
            if args.ewc_lambda_override is not None:
                cmd.extend(["--ewc-lambda-override", str(args.ewc_lambda_override)])
            if args.ewc_apply_to_critic:
                cmd.append("--ewc-apply-to-critic")
        if args.mode == "downstream_rashomon":
            cmd.extend(["--rashomon-settings-file", str(args.rashomon_settings_file)])
            if args.rashomon_n_iters is not None:
                cmd.extend(["--rashomon-n-iters", str(args.rashomon_n_iters)])
            if args.rashomon_min_hard_spec is not None:
                cmd.extend(["--rashomon-min-hard-spec", str(args.rashomon_min_hard_spec)])
            if args.rashomon_surrogate_aggregation is not None:
                cmd.extend(["--rashomon-surrogate-aggregation", str(args.rashomon_surrogate_aggregation)])
            if args.inverse_temp_start is not None:
                cmd.extend(["--inverse-temp-start", str(args.inverse_temp_start)])
            if args.inverse_temp_max is not None:
                cmd.extend(["--inverse-temp-max", str(args.inverse_temp_max)])
            if args.rashomon_checkpoint is not None:
                cmd.extend(["--rashomon-checkpoint", str(args.rashomon_checkpoint)])

    cmd.extend(passthrough)
    return cmd


def _default_log_dir(args: argparse.Namespace) -> Path:
    mode_suffix = NOADAPT_POLICY_SUBDIR if args.mode == "source" else args.mode
    base = args.output_dir if args.mode == "source" else args.outputs_root
    return base / args.layout / "multi_seed_logs" / mode_suffix


def main(argv: list[str] | None = None) -> int:
    args, passthrough = _parse_args(argv)
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("CPU affinity pinning requires os.sched_setaffinity/os.sched_getaffinity support.")

    seeds = dedupe_preserve_order(list(args.seeds))
    if not seeds:
        raise ValueError("No seeds provided. Pass at least one seed via --seeds.")
    core_pool = resolve_core_pool(args.cores)
    if not core_pool:
        raise RuntimeError("No CPU cores available for scheduling.")

    log_dir = args.log_dir or _default_log_dir(args)

    print(f"Launching {len(seeds)} runs for mode={args.mode} layout={args.layout} on cores: {core_pool}")
    return run_seed_pool(
        seeds=seeds,
        cores=core_pool,
        build_cmd=lambda seed: _build_worker_cmd(args, seed=seed, passthrough=passthrough),
        log_dir=log_dir,
        poll_seconds=max(args.poll_seconds, 0.1),
    )


def main_with_mode(mode: str) -> int:
    return main(["--mode", mode, *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())

