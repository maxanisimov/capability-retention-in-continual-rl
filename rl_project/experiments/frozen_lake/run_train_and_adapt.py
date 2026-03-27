#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Suppress ALSA warnings from pygame/gymnasium on headless servers
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def run_and_log(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> Running: {' '.join(cmd)}")
    print(f">>> Log file: {log_path}")

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run train_source_policy.py then downstream_adaptation.py with logging."
    )
    parser.add_argument("--cfg", type=str, default="standard_4x4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=64)

    parser.add_argument("--source-total-steps", type=int, default=500_000)
    parser.add_argument("--downstream-total-timesteps", type=int, default=50_000)

    parser.add_argument("--ent-coef", type=float, default=0.1)
    parser.add_argument("--ewc-lambda", type=float, default=5_000.0)
    parser.add_argument("--rashomon-n-iters", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--source-mode", type=str, default="safe", choices=["original", "safe"],
                        help="'original': use source policy as-is; "
                             "'safe': finetune for safety before adaptation")

    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable to use for both scripts.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root output dir (default: <this_folder>/outputs).",
    )
    parser.add_argument(
        "--logs-root",
        type=str,
        default=None,
        help="Root logs dir (default: <this_folder>/logs).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train_source_policy.py"
    downstream_script = script_dir / "downstream_adaptation.py"

    output_root = Path(args.output_root) if args.output_root else script_dir / "outputs"
    logs_root = Path(args.logs_root) if args.logs_root else script_dir / "logs"

    run_dir = output_root / args.cfg / str(args.seed)
    source_output_dir = run_dir / "source"
    downstream_output_dir = run_dir / "downstream"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_logs_dir = logs_root / args.cfg / str(args.seed) / run_id

    train_log = run_logs_dir / "01_train_source_policy.log"
    downstream_log = run_logs_dir / "02_downstream_adaptation.log"

    # else:
    train_cmd = [
        args.python_bin,
        str(train_script),
        "--cfg",
        args.cfg,
        "--seed",
        str(args.seed),
        "--total-steps",
        str(args.source_total_steps),
        "--hidden",
        str(args.hidden),
        "--output-dir",
        str(source_output_dir),
    ]

    downstream_cmd = [
        args.python_bin,
        str(downstream_script),
        "--cfg",
        args.cfg,
        "--seed",
        str(args.seed),
        "--source-dir",
        str(run_dir),
        "--total-timesteps",
        str(args.downstream_total_timesteps),
        "--ent-coef",
        str(args.ent_coef),
        "--ewc-lambda",
        str(args.ewc_lambda),
        "--rashomon-n-iters",
        str(args.rashomon_n_iters),
        "--eval-episodes",
        str(args.eval_episodes),
        "--hidden",
        str(args.hidden),
        "--device",
        args.device,
    ]

    print("=" * 80)
    print("FrozenLake pipeline: train_source_policy -> downstream_adaptation")
    print(f"Config: {args.cfg}, Seed: {args.seed}")
    print(f"Run dir: {run_dir}")
    print(f"  Source:     {source_output_dir}")
    print(f"  Downstream: {downstream_output_dir}")
    print(f"Logs dir: {run_logs_dir}")
    print("=" * 80)

    run_and_log(train_cmd, train_log)
    run_and_log(downstream_cmd, downstream_log)

    print("\nPipeline complete.")
    print(f"Logs saved in: {run_logs_dir}")


if __name__ == "__main__":
    main()
