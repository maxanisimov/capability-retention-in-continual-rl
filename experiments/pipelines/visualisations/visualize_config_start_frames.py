#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
            "Validate FrozenLake/PoisonedApple config files and export start-frame "
            "visualizations for source (Task 1) and downstream (Task 2)."
        )
    )
    parser.add_argument(
        "--env",
        choices=["frozen_lake", "poisoned_apple", "both"],
        default="both",
        help="Which experiment family to process.",
    )
    parser.add_argument(
        "--frozen-configs",
        type=str,
        default=None,
        help=(
            "Optional comma-separated FrozenLake config names to render. "
            "Default: all configs in frozen_lake/configs.yaml."
        ),
    )
    parser.add_argument(
        "--poisoned-configs",
        type=str,
        default=None,
        help=(
            "Optional comma-separated PoisonedApple config names to render. "
            "Default: all configs in poisoned_apple/configs.yaml."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for deterministic reset before taking the starting frame.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_SCRIPT_DIR / "outputs" / "config_start_frames",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="Output figure DPI.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show each figure interactively while exporting.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on first invalid config/rendering failure.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return data


def _parse_config_filter(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    names = {tok.strip() for tok in raw.split(",") if tok.strip()}
    return names if names else None


def _ensure_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 frame, got shape {arr.shape}.")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _validate_frozen_map(raw_map: Any, *, map_name: str, cfg_name: str) -> list[str]:
    if not isinstance(raw_map, list) or not raw_map:
        raise ValueError(f"{cfg_name}: {map_name} must be a non-empty list of strings.")

    parsed = [str(row) for row in raw_map]
    widths = {len(row) for row in parsed}
    if len(widths) != 1:
        raise ValueError(f"{cfg_name}: {map_name} rows must all have the same length.")

    valid_symbols = {"S", "F", "H", "G"}
    for r, row in enumerate(parsed):
        for c, ch in enumerate(row):
            if ch not in valid_symbols:
                raise ValueError(
                    f"{cfg_name}: {map_name}[{r}][{c}] has invalid symbol '{ch}'. "
                    f"Allowed: {sorted(valid_symbols)}"
                )

    flat = "".join(parsed)
    if flat.count("S") != 1:
        raise ValueError(f"{cfg_name}: {map_name} must contain exactly one 'S' start tile.")
    if flat.count("G") != 1:
        raise ValueError(f"{cfg_name}: {map_name} must contain exactly one 'G' goal tile.")
    return parsed


def _validate_frozen_cfg(cfg_name: str, cfg: Any) -> tuple[list[str], list[str], bool]:
    if not isinstance(cfg, dict):
        raise ValueError(f"{cfg_name}: config must be a mapping.")
    if "env1_map" not in cfg or "env2_map" not in cfg:
        raise ValueError(f"{cfg_name}: config must define 'env1_map' and 'env2_map'.")

    env1_map = _validate_frozen_map(cfg["env1_map"], map_name="env1_map", cfg_name=cfg_name)
    env2_map = _validate_frozen_map(cfg["env2_map"], map_name="env2_map", cfg_name=cfg_name)
    if len(env1_map) != len(env2_map) or len(env1_map[0]) != len(env2_map[0]):
        raise ValueError(
            f"{cfg_name}: env1_map and env2_map must have identical dimensions."
        )

    is_slippery = bool(cfg.get("is_slippery", False))
    return env1_map, env2_map, is_slippery


def _as_pos(raw: Any, *, field_name: str, cfg_name: str) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{cfg_name}: {field_name} must be [row, col].")
    return int(raw[0]), int(raw[1])


def _as_positions(raw: Any, *, field_name: str, cfg_name: str) -> list[tuple[int, int]]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{cfg_name}: {field_name} must be a list of [row, col] positions.")
    return [_as_pos(p, field_name=f"{field_name}[{i}]", cfg_name=cfg_name) for i, p in enumerate(raw)]


def _validate_poisoned_layout(
    *,
    cfg_name: str,
    grid_size: int,
    task_id: int,
    agent_start_pos: tuple[int, int],
    safe_apples: list[tuple[int, int]],
    poisoned_apples: list[tuple[int, int]],
) -> None:
    if grid_size <= 0:
        raise ValueError(f"{cfg_name}: grid_size must be positive, got {grid_size}.")

    def _in_bounds(pos: tuple[int, int]) -> bool:
        return 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size

    if not _in_bounds(agent_start_pos):
        raise ValueError(f"{cfg_name}: task{task_id}_agent_start_pos out of bounds: {agent_start_pos}.")

    safe_set = set(safe_apples)
    poisoned_set = set(poisoned_apples)
    if len(safe_set) != len(safe_apples):
        raise ValueError(f"{cfg_name}: task{task_id}_safe_apples has duplicate positions.")
    if len(poisoned_set) != len(poisoned_apples):
        raise ValueError(f"{cfg_name}: task{task_id}_poisoned_apples has duplicate positions.")
    if safe_set & poisoned_set:
        raise ValueError(
            f"{cfg_name}: task{task_id} safe/poisoned apples overlap at {sorted(safe_set & poisoned_set)}."
        )
    if agent_start_pos in (safe_set | poisoned_set):
        raise ValueError(
            f"{cfg_name}: task{task_id}_agent_start_pos overlaps an apple at {agent_start_pos}."
        )

    for pos in safe_set:
        if not _in_bounds(pos):
            raise ValueError(f"{cfg_name}: task{task_id}_safe_apples has out-of-bounds position {pos}.")
    for pos in poisoned_set:
        if not _in_bounds(pos):
            raise ValueError(
                f"{cfg_name}: task{task_id}_poisoned_apples has out-of-bounds position {pos}."
            )


def _validate_poisoned_cfg(
    cfg_name: str,
    cfg: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(cfg, dict):
        raise ValueError(f"{cfg_name}: config must be a mapping.")
    required_keys = [
        "grid_size",
        "task1_agent_start_pos",
        "task1_safe_apples",
        "task1_poisoned_apples",
        "task2_agent_start_pos",
        "task2_safe_apples",
        "task2_poisoned_apples",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"{cfg_name}: missing required keys: {missing}")

    grid_size = int(cfg["grid_size"])

    task1 = {
        "agent_start_pos": _as_pos(cfg["task1_agent_start_pos"], field_name="task1_agent_start_pos", cfg_name=cfg_name),
        "safe_apples": _as_positions(cfg["task1_safe_apples"], field_name="task1_safe_apples", cfg_name=cfg_name),
        "poisoned_apples": _as_positions(
            cfg["task1_poisoned_apples"], field_name="task1_poisoned_apples", cfg_name=cfg_name
        ),
    }
    task2 = {
        "agent_start_pos": _as_pos(cfg["task2_agent_start_pos"], field_name="task2_agent_start_pos", cfg_name=cfg_name),
        "safe_apples": _as_positions(cfg["task2_safe_apples"], field_name="task2_safe_apples", cfg_name=cfg_name),
        "poisoned_apples": _as_positions(
            cfg["task2_poisoned_apples"], field_name="task2_poisoned_apples", cfg_name=cfg_name
        ),
    }

    _validate_poisoned_layout(cfg_name=cfg_name, grid_size=grid_size, task_id=1, **task1)
    _validate_poisoned_layout(cfg_name=cfg_name, grid_size=grid_size, task_id=2, **task2)
    return task1, task2


def _render_frozenlake_start_frame(env_map: list[str], *, is_slippery: bool, seed: int) -> np.ndarray:
    env = gym.make(
        "FrozenLake-v1",
        desc=env_map,
        is_slippery=is_slippery,
        render_mode="rgb_array",
    )
    try:
        env.reset(seed=seed)
        frame = env.render()
        return _ensure_uint8_rgb(np.asarray(frame))
    finally:
        env.close()


def _render_poisoned_start_frame(
    cfg: dict[str, Any],
    *,
    task_layout: dict[str, Any],
    seed: int,
) -> np.ndarray:
    from experiments.pipelines.poisoned_apple.poisoned_apple_env import PoisonedAppleEnv

    env = PoisonedAppleEnv(
        grid_size=int(cfg["grid_size"]),
        observation_type=str(cfg.get("observation_type", "flat")),
        max_steps=int(cfg.get("max_steps", 30)),
        reward_safe=float(cfg.get("reward_safe", 1.0)),
        reward_poison=float(cfg.get("reward_poison", -1.0)),
        reward_step=float(cfg.get("reward_step", -0.01)),
        agent_start_pos=task_layout["agent_start_pos"],
        safe_apple_positions=task_layout["safe_apples"],
        poisoned_apple_positions=task_layout["poisoned_apples"],
        render_mode="rgb_array",
    )
    try:
        env.reset(seed=seed)
        frame = env.render()
        return _ensure_uint8_rgb(np.asarray(frame))
    finally:
        env.close()


def _save_pair_figure(
    *,
    source_frame: np.ndarray,
    downstream_frame: np.ndarray,
    title: str,
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8.6, 4.3))
    axes[0].imshow(source_frame)
    axes[0].set_title("Source (Task 1)", fontsize=11, pad=6)
    axes[1].imshow(downstream_frame)
    axes[1].set_title("Downstream (Task 2)", fontsize=11, pad=6)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _iter_selected_configs(
    all_cfgs: dict[str, Any],
    selected: set[str] | None,
) -> list[tuple[str, Any]]:
    if selected is None:
        return [(k, all_cfgs[k]) for k in sorted(all_cfgs.keys())]
    return [(k, all_cfgs[k]) for k in sorted(all_cfgs.keys()) if k in selected]


def main() -> None:
    args = parse_args()

    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    frozen_path = _SCRIPT_DIR / "frozen_lake" / "configs.yaml"
    poisoned_path = _SCRIPT_DIR / "poisoned_apple" / "configs.yaml"

    frozen_filter = _parse_config_filter(args.frozen_configs)
    poisoned_filter = _parse_config_filter(args.poisoned_configs)

    total = 0
    succeeded = 0
    failed: list[tuple[str, str, str]] = []

    do_frozen = args.env in {"frozen_lake", "both"}
    do_poisoned = args.env in {"poisoned_apple", "both"}

    if do_frozen:
        frozen_cfgs = _load_yaml(frozen_path)
        if frozen_filter is not None:
            missing = sorted(frozen_filter - set(frozen_cfgs.keys()))
            if missing:
                raise KeyError(
                    f"FrozenLake config(s) not found in {frozen_path}: {missing}. "
                    f"Available: {sorted(frozen_cfgs.keys())}"
                )

        for cfg_name, cfg in _iter_selected_configs(frozen_cfgs, frozen_filter):
            total += 1
            try:
                env1_map, env2_map, is_slippery = _validate_frozen_cfg(cfg_name, cfg)
                src_frame = _render_frozenlake_start_frame(
                    env1_map,
                    is_slippery=is_slippery,
                    seed=args.seed,
                )
                dst_frame = _render_frozenlake_start_frame(
                    env2_map,
                    is_slippery=is_slippery,
                    seed=args.seed,
                )
                output_path = out_root / "frozen_lake" / f"{cfg_name}_source_downstream_start.png"
                _save_pair_figure(
                    source_frame=src_frame,
                    downstream_frame=dst_frame,
                    title=f"FrozenLake | {cfg_name}",
                    output_path=output_path,
                    dpi=args.dpi,
                    show=args.show,
                )
                succeeded += 1
                print(f"[OK] FrozenLake/{cfg_name} -> {output_path}")
            except Exception as exc:
                failed.append(("FrozenLake", cfg_name, str(exc)))
                print(f"[ERROR] FrozenLake/{cfg_name}: {exc}")
                if args.fail_fast:
                    raise

    if do_poisoned:
        poisoned_cfgs = _load_yaml(poisoned_path)
        if poisoned_filter is not None:
            missing = sorted(poisoned_filter - set(poisoned_cfgs.keys()))
            if missing:
                raise KeyError(
                    f"PoisonedApple config(s) not found in {poisoned_path}: {missing}. "
                    f"Available: {sorted(poisoned_cfgs.keys())}"
                )

        for cfg_name, cfg in _iter_selected_configs(poisoned_cfgs, poisoned_filter):
            total += 1
            try:
                task1, task2 = _validate_poisoned_cfg(cfg_name, cfg)
                src_frame = _render_poisoned_start_frame(cfg, task_layout=task1, seed=args.seed)
                dst_frame = _render_poisoned_start_frame(cfg, task_layout=task2, seed=args.seed)
                output_path = out_root / "poisoned_apple" / f"{cfg_name}_source_downstream_start.png"
                _save_pair_figure(
                    source_frame=src_frame,
                    downstream_frame=dst_frame,
                    title=f"PoisonedApple | {cfg_name}",
                    output_path=output_path,
                    dpi=args.dpi,
                    show=args.show,
                )
                succeeded += 1
                print(f"[OK] PoisonedApple/{cfg_name} -> {output_path}")
            except Exception as exc:
                failed.append(("PoisonedApple", cfg_name, str(exc)))
                print(f"[ERROR] PoisonedApple/{cfg_name}: {exc}")
                if args.fail_fast:
                    raise

    print("=" * 88)
    print("Config visualization summary")
    print(f"Processed: {total}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed   : {len(failed)}")
    print(f"Output   : {out_root}")
    if failed:
        print("- Failed configurations -")
        for env_name, cfg_name, error_text in failed:
            print(f"  {env_name}/{cfg_name}: {error_text}")
    print("=" * 88)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
