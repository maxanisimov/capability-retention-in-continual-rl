"""Run one FrozenLake mode across multiple seeds with CPU pinning."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import TextIO

from experiments.pipelines.frozenlake.core.orchestration.run_paths import (
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


@dataclass
class SeedRun:
    seed: int
    core: int
    process: subprocess.Popen[bytes]
    log_path: Path
    log_handle: TextIO


def _dedupe_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


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


def _resolve_core_pool(requested_cores: list[int] | None) -> list[int]:
    available_cores = sorted(os.sched_getaffinity(0))
    if requested_cores is None:
        return available_cores
    invalid = sorted(set(requested_cores) - set(available_cores))
    if invalid:
        raise ValueError(f"Requested --cores {invalid} are not available in affinity mask {available_cores}.")
    return _dedupe_preserve_order(requested_cores)


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


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["BLIS_NUM_THREADS"] = "1"
    return env


def _start_seed_run(*, seed: int, cmd: list[str], core: int, log_path: Path) -> SeedRun:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")

    def _pin_to_core() -> None:
        os.sched_setaffinity(0, {core})

    process = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=_worker_env(),
        preexec_fn=_pin_to_core,
    )
    return SeedRun(seed=seed, core=core, process=process, log_path=log_path, log_handle=log_handle)


def _default_log_dir(args: argparse.Namespace) -> Path:
    mode_suffix = NOADAPT_POLICY_SUBDIR if args.mode == "source" else args.mode
    base = args.output_dir if args.mode == "source" else args.outputs_root
    return base / args.layout / "multi_seed_logs" / mode_suffix


def main(argv: list[str] | None = None) -> int:
    args, passthrough = _parse_args(argv)
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("CPU affinity pinning requires os.sched_setaffinity/os.sched_getaffinity support.")

    seeds = _dedupe_preserve_order(list(args.seeds))
    if not seeds:
        raise ValueError("No seeds provided. Pass at least one seed via --seeds.")
    core_pool = _resolve_core_pool(args.cores)
    if not core_pool:
        raise RuntimeError("No CPU cores available for scheduling.")

    log_dir = args.log_dir or _default_log_dir(args)
    pending: deque[int] = deque(seeds)
    free_cores: deque[int] = deque(core_pool)
    active: list[SeedRun] = []
    failures: list[tuple[int, int, int, Path]] = []

    print(f"Launching {len(seeds)} runs for mode={args.mode} layout={args.layout} on cores: {core_pool}")
    while pending or active:
        while pending and free_cores:
            seed = pending.popleft()
            core = free_cores.popleft()
            cmd = _build_worker_cmd(args, seed=seed, passthrough=passthrough)
            run = _start_seed_run(seed=seed, cmd=cmd, core=core, log_path=log_dir / f"seed_{seed}.log")
            active.append(run)
            print(f"[start] seed={seed} core={core} pid={run.process.pid} log={run.log_path}")

        time.sleep(max(args.poll_seconds, 0.1))
        still_active: list[SeedRun] = []
        for run in active:
            return_code = run.process.poll()
            if return_code is None:
                still_active.append(run)
                continue
            run.log_handle.close()
            free_cores.append(run.core)
            if return_code == 0:
                print(f"[done] seed={run.seed} core={run.core} rc=0")
            else:
                print(f"[fail] seed={run.seed} core={run.core} rc={return_code} log={run.log_path}")
                failures.append((run.seed, run.core, int(return_code), run.log_path))
        active = still_active

    if failures:
        print("\nOne or more runs failed:")
        for seed, core, rc, log_path in failures:
            print(f"  seed={seed} core={core} rc={rc} log={log_path}")
        return 1
    print("\nAll runs completed successfully.")
    return 0


def main_with_mode(mode: str) -> int:
    return main(["--mode", mode, *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())

