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
os.environ.setdefault("ALSA_CONFIG_PATH", os.devnull)


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


def _load_cfg(script_dir: Path, cfg_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = script_dir / "configs.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        all_cfgs = yaml.safe_load(handle)

    if not isinstance(all_cfgs, dict):
        raise ValueError(f"Expected mapping in {config_path}, got {type(all_cfgs).__name__}.")

    if cfg_name not in all_cfgs:
        available = sorted(all_cfgs.keys())
        raise KeyError(f"Config '{cfg_name}' not found in {config_path}. Available: {available}")

    cfg = all_cfgs[cfg_name]
    if not isinstance(cfg, dict):
        raise ValueError(f"Config '{cfg_name}' must be a mapping.")

    train_cfg = cfg.get("train", {})
    if train_cfg is None:
        train_cfg = {}
    if not isinstance(train_cfg, dict):
        raise ValueError(f"Config '{cfg_name}'.train must be a mapping.")

    return cfg, train_cfg


def _resolve(cli_value: Any, train_cfg: dict[str, Any], key: str, fallback: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return train_cfg.get(key, fallback)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run train_source_policy.py then downstream_adaptation.py with logging."
    )
    parser.add_argument("--cfg", type=str, default="standard_4x4")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--source-total-timesteps",
        type=int,
        default=None,
        help="Max source-policy PPO timesteps (defaults to configs.yaml train.source_total_timesteps).",
    )
    parser.add_argument(
        "--source-total-steps",
        dest="source_total_steps_legacy",
        type=int,
        default=None,
        help="Deprecated alias for --source-total-timesteps.",
    )
    parser.add_argument(
        "--downstream-total-timesteps",
        type=int,
        default=None,
        help="Max downstream adaptation timesteps (defaults to configs.yaml train.downstream_total_timesteps).",
    )

    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--ewc-lambda", type=float, default=None)
    parser.add_argument("--rashomon-n-iters", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

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

    _, train_cfg = _load_cfg(script_dir=script_dir, cfg_name=args.cfg)

    if (
        args.source_total_timesteps is not None
        and args.source_total_steps_legacy is not None
        and args.source_total_timesteps != args.source_total_steps_legacy
    ):
        raise ValueError(
            "Both --source-total-timesteps and deprecated --source-total-steps were provided "
            "with different values. Please provide only one value."
        )

    legacy_value = args.source_total_steps_legacy
    source_cli_value = args.source_total_timesteps
    if source_cli_value is None and legacy_value is not None:
        source_cli_value = legacy_value
        print(
            "WARNING: --source-total-steps is deprecated; use --source-total-timesteps instead.",
            file=sys.stderr,
        )

    source_total_timesteps = int(
        _resolve(source_cli_value, train_cfg, "source_total_timesteps", 500_000)
    )
    downstream_total_timesteps = int(
        _resolve(args.downstream_total_timesteps, train_cfg, "downstream_total_timesteps", 50_000)
    )
    hidden = int(_resolve(args.hidden, train_cfg, "hidden", 64))
    ent_coef = float(_resolve(args.ent_coef, train_cfg, "ent_coef", 0.1))
    ewc_lambda = float(_resolve(args.ewc_lambda, train_cfg, "ewc_lambda", 5_000.0))
    rashomon_n_iters = int(_resolve(args.rashomon_n_iters, train_cfg, "rashomon_n_iters", 5_000))
    eval_episodes = int(_resolve(args.eval_episodes, train_cfg, "eval_episodes", 1))
    device = str(_resolve(args.device, train_cfg, "device", "cpu"))

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
        "--total-steps",
        str(source_total_timesteps),
        "--hidden",
        str(hidden),
        "--output-dir",
        str(source_output_dir),
        "--plots-dir",
        str(plots_dir),
    ]

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
        "--total-timesteps",
        str(downstream_total_timesteps),
        "--ent-coef",
        str(ent_coef),
        "--ewc-lambda",
        str(ewc_lambda),
        "--rashomon-n-iters",
        str(rashomon_n_iters),
        "--eval-episodes",
        str(eval_episodes),
        "--hidden",
        str(hidden),
        "--device",
        device,
    ]

    print("=" * 80)
    print("FrozenLake pipeline: train_source_policy -> downstream_adaptation")
    print(f"Config: {args.cfg}, Seed: {args.seed}")
    print(f"Run dir: {run_dir}")
    print(f"  Source:     {source_output_dir}")
    print(f"  Downstream: {downstream_output_dir}")
    print(f"  Plots:      {plots_dir}")
    print(f"Logs dir: {run_logs_dir}")
    print("-" * 80)
    print(f"source_total_timesteps     : {source_total_timesteps}")
    print(f"downstream_total_timesteps : {downstream_total_timesteps}")
    print(f"hidden                     : {hidden}")
    print(f"ent_coef                   : {ent_coef}")
    print(f"ewc_lambda                 : {ewc_lambda}")
    print(f"rashomon_n_iters           : {rashomon_n_iters}")
    print(f"eval_episodes              : {eval_episodes}")
    print(f"device                     : {device}")
    print("=" * 80)

    run_and_log(train_cmd, train_log)
    run_and_log(downstream_cmd, downstream_log)

    print("\nPipeline complete.")
    print(f"Logs saved in: {run_logs_dir}")


if __name__ == "__main__":
    main()
