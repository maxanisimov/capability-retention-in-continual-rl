"""Unified launcher for FrozenLake source training and downstream adaptation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from experiments.pipelines.trajectory_retention.frozenlake.core.orchestration.run_paths import (
    default_adapt_ewc_settings_file,
    default_adapt_ppo_settings_file,
    default_adapt_rashomon_settings_file,
    default_downstream_envs_file,
    default_outputs_root,
    default_source_envs_file,
    default_train_source_settings_file,
    pipeline_root,
)


MODE_TO_SCRIPT = {
    "source": "train_source.py",
    "downstream_unconstrained": "adapt_unconstrained.py",
    "downstream_ewc": "adapt_ewc.py",
    "downstream_rashomon": "adapt_rashomon.py",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a FrozenLake experiment by mode with one interface for layout and seed.",
    )
    parser.add_argument("--mode", required=True, choices=sorted(MODE_TO_SCRIPT))
    parser.add_argument("--pipeline", "--layout", type=str, dest="layout", default="diagonal_30x30")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--activation", choices=["tanh", "relu"], default="relu")
    parser.add_argument("--source-env-file", type=Path, default=default_source_envs_file())
    parser.add_argument("--downstream-env-file", type=Path, default=default_downstream_envs_file())
    parser.add_argument("--source-settings-file", type=Path, default=default_train_source_settings_file())
    parser.add_argument("--adapt-settings-file", type=Path, default=default_adapt_ppo_settings_file())
    parser.add_argument("--ewc-settings-file", type=Path, default=default_adapt_ewc_settings_file())
    parser.add_argument("--rashomon-settings-file", type=Path, default=default_adapt_rashomon_settings_file())
    parser.add_argument("--outputs-root", type=Path, default=default_outputs_root())
    parser.add_argument("--output-dir", type=Path, default=default_outputs_root())
    parser.add_argument("--source-run-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args, passthrough = parser.parse_known_args(argv)

    cmd = [
        sys.executable,
        str(pipeline_root() / "cli" / MODE_TO_SCRIPT[args.mode]),
        "--pipeline",
        str(args.layout),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--activation",
        str(args.activation),
        "--source-env-file",
        str(args.source_env_file),
        "--downstream-env-file",
        str(args.downstream_env_file),
    ]

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
        if args.source_run_dir is not None:
            cmd.extend(["--source-run-dir", str(args.source_run_dir)])
        if args.mode == "downstream_ewc":
            cmd.extend(["--ewc-settings-file", str(args.ewc_settings_file)])
        if args.mode == "downstream_rashomon":
            cmd.extend(["--rashomon-settings-file", str(args.rashomon_settings_file)])

    cmd.extend(passthrough)
    print("Launching:")
    print("  " + " ".join(cmd))
    if args.dry_run:
        return 0
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

