"""Unified launcher for LunarLander source training and downstream adaptation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


MODE_TO_SCRIPT = {
    "source": "train_source_policy.py",
    "downstream_unconstrained": "downstream_adaptation_unconstrained.py",
    "downstream_ewc": "downstream_adaptation_ewc.py",
    "downstream_rashomon": "downstream_adaptation_rashomon.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a LunarLander experiment by mode with a single interface for "
            "task-setting and seed."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=sorted(MODE_TO_SCRIPT.keys()),
        help="Experiment mode to run.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--task-setting",
        type=str,
        default="default",
        help="Environment configuration name from task_settings.yaml.",
    )
    parser.add_argument(
        "--task-settings-file",
        type=Path,
        default=Path(__file__).resolve().parent / "settings" / "task_settings.yaml",
        help="Task settings YAML containing source/downstream environment configs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device passed to the selected script.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Outputs root directory used by downstream scripts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Output directory used by source-task training.",
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help="Optional explicit source run directory for downstream adaptation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )

    args, passthrough = parser.parse_known_args()

    script_dir = Path(__file__).resolve().parent
    script_path = script_dir / MODE_TO_SCRIPT[args.mode]

    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "--seed",
        str(args.seed),
        "--task-setting",
        str(args.task_setting),
        "--task-settings-file",
        str(args.task_settings_file),
        "--device",
        str(args.device),
    ]

    if args.mode == "source":
        cmd.extend(
            [
                "--task-role",
                "source",
                "--output-dir",
                str(args.output_dir),
            ],
        )
    else:
        cmd.extend(
            [
                "--outputs-root",
                str(args.outputs_root),
            ],
        )
        if args.source_run_dir is not None:
            cmd.extend(["--source-run-dir", str(args.source_run_dir)])

    cmd.extend(passthrough)

    print("Launching:")
    print("  " + " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
