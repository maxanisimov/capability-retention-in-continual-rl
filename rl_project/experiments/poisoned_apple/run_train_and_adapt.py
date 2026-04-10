#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Suppress ALSA warnings from pygame/gymnasium on headless servers
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("ALSA_CONFIG_PATH", "/dev/null")


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
        description=(
            "Run PoisonedApple train_source_policy.py then "
            "downstream_adaptation.py with logging."
        )
    )
    parser.add_argument("--cfg", type=str, default="simple_5x5")
    parser.add_argument("--seed", type=int, default=0)

    # Source-stage controls
    parser.add_argument(
        "--source-total-timesteps",
        type=int,
        default=None,
        help="Override source PPO total timesteps (default: from config).",
    )
    parser.add_argument(
        "--source-total-steps",
        dest="source_total_steps_legacy",
        type=int,
        default=None,
        help="Deprecated alias for --source-total-timesteps.",
    )
    parser.add_argument(
        "--source-eval-episodes",
        type=int,
        default=None,
        help="Override source evaluation episodes (default: from config).",
    )
    parser.add_argument(
        "--source-safety-finetuning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run source safety behavior-cloning finetuning.",
    )
    parser.add_argument("--bc-epochs", type=int, default=None)
    parser.add_argument("--bc-lr", type=float, default=None)
    parser.add_argument("--bc-batch-size", type=int, default=None)

    # Downstream-stage controls
    parser.add_argument(
        "--downstream-total-timesteps",
        type=int,
        default=None,
        help="Override downstream adaptation timesteps (default: from config).",
    )
    parser.add_argument(
        "--downstream-eval-episodes",
        type=int,
        default=None,
        help="Override downstream evaluation episodes (default: from config).",
    )
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--ewc-lambda", type=float, default=None)
    parser.add_argument("--ewc-fisher-sample-size", type=int, default=None)
    parser.add_argument("--rashomon-n-iters", type=int, default=None)
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether downstream adaptation should save trajectory plots.",
    )

    # Shared overrides
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--min-safety-accuracy", type=float, default=None)

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


def _maybe_add(cmd: list[str], key: str, value: object | None) -> None:
    if value is None:
        return
    cmd.extend([key, str(value)])


def _load_cfg(script_dir: Path, cfg_name: str) -> dict[str, Any]:
    config_path = script_dir / "configs.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        all_cfgs = yaml.safe_load(f)

    if not isinstance(all_cfgs, dict):
        raise ValueError(f"Expected mapping in {config_path}, got {type(all_cfgs).__name__}.")
    if cfg_name not in all_cfgs:
        available = sorted(all_cfgs.keys())
        raise KeyError(f"Config '{cfg_name}' not found in {config_path}. Available: {available}")

    cfg = all_cfgs[cfg_name]
    if not isinstance(cfg, dict):
        raise ValueError(f"Config '{cfg_name}' must be a mapping.")
    return cfg


def _resolve(cli_value: Any, cfg: dict[str, Any], key: str, fallback: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return cfg.get(key, fallback)


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train_source_policy.py"
    downstream_script = script_dir / "downstream_adaptation.py"

    cfg = _load_cfg(script_dir=script_dir, cfg_name=args.cfg)

    if (
        args.source_total_timesteps is not None
        and args.source_total_steps_legacy is not None
        and args.source_total_timesteps != args.source_total_steps_legacy
    ):
        raise ValueError(
            "Both --source-total-timesteps and deprecated --source-total-steps were provided "
            "with different values. Please provide only one value."
        )

    source_cli_value = args.source_total_timesteps
    if source_cli_value is None and args.source_total_steps_legacy is not None:
        source_cli_value = args.source_total_steps_legacy
        print(
            "WARNING: --source-total-steps is deprecated; use --source-total-timesteps instead.",
            file=sys.stderr,
        )

    source_total_timesteps = int(_resolve(source_cli_value, cfg, "source_total_timesteps", 50_000))
    downstream_total_timesteps = int(
        _resolve(
            args.downstream_total_timesteps,
            cfg,
            "downstream_total_timesteps",
            source_total_timesteps,
        )
    )
    source_eval_episodes = int(_resolve(args.source_eval_episodes, cfg, "eval_episodes", 100))
    downstream_eval_episodes = int(_resolve(args.downstream_eval_episodes, cfg, "eval_episodes", 100))
    ent_coef = float(_resolve(args.ent_coef, cfg, "ppo_ent_coef", 0.01))
    ewc_lambda = float(_resolve(args.ewc_lambda, cfg, "ewc_lambda", 5_000.0))
    ewc_fisher_sample_size = int(_resolve(args.ewc_fisher_sample_size, cfg, "ewc_fisher_sample_size", 1_000))
    rashomon_n_iters = int(_resolve(args.rashomon_n_iters, cfg, "rashomon_n_iters", 20_000))
    device = str(_resolve(args.device, cfg, "device", "cpu"))

    min_safety_accuracy = args.min_safety_accuracy
    if min_safety_accuracy is None and "min_safety_accuracy" in cfg:
        min_safety_accuracy = float(cfg["min_safety_accuracy"])

    output_root = Path(args.output_root) if args.output_root else script_dir / "outputs"
    logs_root = Path(args.logs_root) if args.logs_root else script_dir / "logs"

    run_dir = output_root / args.cfg / str(args.seed)
    source_output_dir = run_dir / "source"
    downstream_output_dir = run_dir / "downstream"
    plots_dir = run_dir / "plots"

    source_output_dir.mkdir(parents=True, exist_ok=True)
    downstream_output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_logs_dir = logs_root / args.cfg / str(args.seed) / run_id
    train_log = run_logs_dir / "01_train_source_policy.log"
    downstream_log = run_logs_dir / "02_downstream_adaptation.log"

    train_cmd = [
        args.python_bin,
        str(train_script),
        "--cfg",
        args.cfg,
        "--seed",
        str(args.seed),
        "--device",
        device,
        "--output-dir",
        str(source_output_dir),
        "--plots-dir",
        str(plots_dir),
        "--total-steps",
        str(source_total_timesteps),
        "--eval-episodes",
        str(source_eval_episodes),
    ]
    _maybe_add(train_cmd, "--bc-epochs", args.bc_epochs)
    _maybe_add(train_cmd, "--bc-lr", args.bc_lr)
    _maybe_add(train_cmd, "--bc-batch-size", args.bc_batch_size)
    _maybe_add(train_cmd, "--min-safety-accuracy", min_safety_accuracy)
    train_cmd.append("--safety-finetuning" if args.source_safety_finetuning else "--no-safety-finetuning")

    downstream_cmd = [
        args.python_bin,
        str(downstream_script),
        "--cfg",
        args.cfg,
        "--seed",
        str(args.seed),
        "--source-dir",
        str(source_output_dir),
        "--output-dir",
        str(downstream_output_dir),
        "--plots-dir",
        str(plots_dir),
        "--ent-coef",
        str(ent_coef),
        "--ewc-lambda",
        str(ewc_lambda),
        "--ewc-fisher-sample-size",
        str(ewc_fisher_sample_size),
        "--rashomon-n-iters",
        str(rashomon_n_iters),
        "--device",
        device,
        "--total-timesteps",
        str(downstream_total_timesteps),
        "--eval-episodes",
        str(downstream_eval_episodes),
    ]
    _maybe_add(downstream_cmd, "--min-safety-accuracy", min_safety_accuracy)
    downstream_cmd.append("--save-plots" if args.save_plots else "--no-save-plots")

    print("=" * 80)
    print("PoisonedApple pipeline: train_source_policy -> downstream_adaptation")
    print(f"Config: {args.cfg}, Seed: {args.seed}")
    print(f"Run dir: {run_dir}")
    print(f"  Source:     {source_output_dir}")
    print(f"  Downstream: {downstream_output_dir}")
    print(f"  Plots:      {plots_dir}")
    print(f"Logs dir: {run_logs_dir}")
    print("-" * 80)
    print(f"source_total_timesteps     : {source_total_timesteps}")
    print(f"downstream_total_timesteps : {downstream_total_timesteps}")
    print(f"source_eval_episodes       : {source_eval_episodes}")
    print(f"downstream_eval_episodes   : {downstream_eval_episodes}")
    print(f"ent_coef                   : {ent_coef}")
    print(f"ewc_lambda                 : {ewc_lambda}")
    print(f"ewc_fisher_sample_size     : {ewc_fisher_sample_size}")
    print(f"rashomon_n_iters           : {rashomon_n_iters}")
    print(f"device                     : {device}")
    if min_safety_accuracy is not None:
        print(f"min_safety_accuracy        : {min_safety_accuracy}")
    print("=" * 80)

    run_and_log(train_cmd, train_log)
    run_and_log(downstream_cmd, downstream_log)

    print("\nPipeline complete.")
    print(f"Logs saved in: {run_logs_dir}")


if __name__ == "__main__":
    main()
