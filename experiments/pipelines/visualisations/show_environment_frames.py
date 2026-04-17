#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import yaml


# Keep rendering quiet on headless machines.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("ALSA_CONFIG_PATH", "/dev/null")

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one frame per selected environment/task and save one "
            "combined figure."
        )
    )
    parser.add_argument(
        "--env",
        choices=["frozen_lake", "poisoned_apple", "both"],
        default="both",
        help="Which environment(s) to render.",
    )
    parser.add_argument(
        "--frozen-cfg",
        type=str,
        default="standard_4x4",
        help="FrozenLake config key in experiments/frozen_lake/configs.yaml.",
    )
    parser.add_argument(
        "--poisoned-cfg",
        type=str,
        default="simple_5x5",
        help="PoisonedApple config key in experiments/poisoned_apple/configs.yaml.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="1,2",
        help="Comma-separated task IDs to render (subset of 1,2).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Deprecated. Kept for compatibility; this script always plots one frame per env/task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic reset/action sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for frames (default: experiments/outputs/env_frames).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the combined figure interactively in addition to saving.",
    )
    return parser.parse_args()


def _parse_tasks(tasks_arg: str) -> list[int]:
    out: list[int] = []
    for token in tasks_arg.split(","):
        token = token.strip()
        if not token:
            continue
        task_id = int(token)
        if task_id not in (1, 2):
            raise ValueError(f"Invalid task id '{task_id}'. Allowed values are 1 and 2.")
        if task_id not in out:
            out.append(task_id)
    if not out:
        raise ValueError("No tasks selected. Use --tasks with values from {1,2}.")
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def _ensure_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 frame, got shape {arr.shape}.")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _figure_to_rgb_array(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Expected HxWx4 RGBA canvas buffer, got shape {rgba.shape}.")
    return _ensure_uint8_rgb(np.ascontiguousarray(rgba[..., :3]))


def _crop_near_white_border(
    frame: np.ndarray,
    *,
    white_threshold: int = 245,
    margin_px: int = 2,
) -> np.ndarray:
    arr = _ensure_uint8_rgb(frame)
    non_white = np.any(arr < white_threshold, axis=2)
    if not np.any(non_white):
        return arr

    ys, xs = np.where(non_white)
    y0 = max(0, int(ys.min()) - margin_px)
    y1 = min(arr.shape[0], int(ys.max()) + 1 + margin_px)
    x0 = max(0, int(xs.min()) - margin_px)
    x1 = min(arr.shape[1], int(xs.max()) + 1 + margin_px)
    return np.ascontiguousarray(arr[y0:y1, x0:x1])


def _collect_frozenlake_frames(
    *,
    cfg: dict[str, Any],
    task_id: int,
    seed: int,
) -> np.ndarray:
    env_map_key = "env1_map" if task_id == 1 else "env2_map"
    env_map = cfg[env_map_key]
    is_slippery = bool(cfg.get("is_slippery", False))

    env = gym.make(
        "FrozenLake-v1",
        desc=env_map,
        is_slippery=is_slippery,
        render_mode="rgb_array",
    )
    try:
        _, _ = env.reset(seed=seed)
        frame = env.render()
        return _ensure_uint8_rgb(np.asarray(frame))
    finally:
        env.close()


def _collect_poisoned_apple_frames(
    *,
    cfg: dict[str, Any],
    task_id: int,
    seed: int,
) -> np.ndarray:
    from experiments.pipelines.poisoned_apple.poisoned_apple_env import (
        PoisonedAppleEnv,
        plot_trajectory,
    )

    layout = {
        "agent_start_pos": tuple(cfg[f"task{task_id}_agent_start_pos"]),
        "safe_apples": [tuple(p) for p in cfg[f"task{task_id}_safe_apples"]],
        "poisoned_apples": [tuple(p) for p in cfg[f"task{task_id}_poisoned_apples"]],
    }

    env = PoisonedAppleEnv(
        grid_size=int(cfg["grid_size"]),
        observation_type=str(cfg.get("observation_type", "flat")),
        max_steps=int(cfg.get("max_steps", 30)),
        reward_safe=float(cfg.get("reward_safe", 1.0)),
        reward_poison=float(cfg.get("reward_poison", -1.0)),
        reward_step=float(cfg.get("reward_step", -0.01)),
        agent_start_pos=layout["agent_start_pos"],
        safe_apple_positions=layout["safe_apples"],
        poisoned_apple_positions=layout["poisoned_apples"],
        render_mode=None,
    )

    try:
        _, _ = env.reset(seed=seed)
        if env.agent_pos is None or env.safe_apples is None or env.poisoned_apples is None:
            raise RuntimeError("PoisonedAppleEnv reset did not initialize state for plotting.")

        # Match downstream_adaptation plotting path:
        # visualize_agent_trajectory(...) -> plot_trajectory(...)
        existing_figs = set(plt.get_fignums())
        plot_trajectory(
            env=env,
            trajectory=[
                {
                    "agent_pos": tuple(int(v) for v in env.agent_pos.tolist()),
                    "safe_apples": set(env.safe_apples),
                    "poisoned_apples": set(env.poisoned_apples),
                }
            ],
            rewards_list=[],
            actions_list=[],
            env_name=f"Task_{task_id}",
            save_dir=None,
        )
        new_figs = [num for num in plt.get_fignums() if num not in existing_figs]
        if not new_figs:
            raise RuntimeError("PoisonedApple plot_trajectory did not create a figure.")
        plot_fig = plt.figure(new_figs[-1])
        for fig_ax in plot_fig.axes:
            fig_ax.set_title("")
            fig_ax.set_xlabel("")
            fig_ax.set_ylabel("")
        if plot_fig._suptitle is not None:
            plot_fig._suptitle.set_text("")
        frame = _figure_to_rgb_array(plot_fig)
        frame = _crop_near_white_border(frame)
        plt.close(plot_fig)
        return frame
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    tasks = _parse_tasks(args.tasks)
    if args.num_frames != 1:
        print("Note: --num-frames is deprecated; this script always plots one frame per env/task.")

    script_dir = Path(__file__).resolve().parent
    out_root = args.output_dir if args.output_dir else (script_dir / "outputs" / "env_frames")
    out_root.mkdir(parents=True, exist_ok=True)

    do_frozen = args.env in {"frozen_lake", "both"}
    do_poisoned = args.env in {"poisoned_apple", "both"}
    env_rows: list[tuple[str, str, dict[str, Any]]] = []

    if do_frozen:
        frozen_cfg_path = script_dir / "frozen_lake" / "configs.yaml"
        frozen_cfgs = _load_yaml(frozen_cfg_path)
        if args.frozen_cfg not in frozen_cfgs:
            raise KeyError(
                f"FrozenLake config '{args.frozen_cfg}' not found in {frozen_cfg_path}. "
                f"Available: {list(frozen_cfgs)}"
            )
        env_rows.append(("FrozenLake", args.frozen_cfg, frozen_cfgs[args.frozen_cfg]))

    if do_poisoned:
        poisoned_cfg_path = script_dir / "poisoned_apple" / "configs.yaml"
        poisoned_cfgs = _load_yaml(poisoned_cfg_path)
        if args.poisoned_cfg not in poisoned_cfgs:
            raise KeyError(
                f"PoisonedApple config '{args.poisoned_cfg}' not found in {poisoned_cfg_path}. "
                f"Available: {list(poisoned_cfgs)}"
            )
        env_rows.append(("PoisonedApple", args.poisoned_cfg, poisoned_cfgs[args.poisoned_cfg]))

    if not env_rows:
        raise ValueError("No environments selected. Use --env frozen_lake, poisoned_apple, or both.")

    fig, axes = plt.subplots(
        nrows=len(env_rows),
        ncols=len(tasks),
        figsize=(4.4 * len(tasks), 4.2 * len(env_rows)),
        squeeze=False,
    )

    for row_idx, (env_name, cfg_name, cfg) in enumerate(env_rows):
        for col_idx, task_id in enumerate(tasks):
            ax = axes[row_idx, col_idx]

            if env_name == "FrozenLake":
                frame = _collect_frozenlake_frames(
                    cfg=cfg,
                    task_id=task_id,
                    seed=args.seed,
                )
            else:
                frame = _collect_poisoned_apple_frames(
                    cfg=cfg,
                    task_id=task_id,
                    seed=args.seed,
                )
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col_idx == 0:
                if env_name == "FrozenLake":
                    ax.set_ylabel("Frozen Lake", fontsize=11, labelpad=12)
                elif env_name == "PoisonedApple":
                    ax.set_ylabel("Poisoned Apple", fontsize=11, labelpad=12)
                else:
                    ax.set_ylabel(env_name, fontsize=11, labelpad=12)
            title_pad = 2 if env_name == "PoisonedApple" else 6
            ax.set_title(f"Task {task_id}", fontsize=11, pad=title_pad)

    fig.tight_layout()

    env_token = args.env
    tasks_token = "-".join(str(t) for t in tasks)
    combined_path = out_root / f"combined_frames_{env_token}_tasks_{tasks_token}.png"
    fig.savefig(combined_path, dpi=250, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)

    print("=" * 80)
    print("Environment frame export complete (single combined figure)")
    print(f"Output root: {out_root}")
    print(f"Figure path: {combined_path}")
    print(f"Env mode   : {args.env}")
    print(f"Tasks      : {tasks}")
    print("Num frames : 1 per environment-task")
    print("=" * 80)


if __name__ == "__main__":
    main()
