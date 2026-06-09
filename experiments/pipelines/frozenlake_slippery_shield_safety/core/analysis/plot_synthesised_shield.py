"""Synthesize and plot a FrozenLake slippery shield without training."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from experiments.pipelines.frozenlake_slippery_shield_safety.core.analysis.plot_shield import (
    plot_shield_safety_probabilities,
)
from experiments.pipelines.frozenlake_slippery_shield_safety.core.config import (
    PipelineConfig,
    get_pipeline_config,
)
from experiments.pipelines.frozenlake_slippery_shield_safety.core.env import make_env
from experiments.pipelines.frozenlake_slippery_shield_safety.core.paths import artifacts_root
from experiments.pipelines.frozenlake_slippery_shield_safety.core.safety import (
    shield_allowed_action_stats,
    synthesise_frozenlake_shield,
)


def default_shield_figure_dir() -> Path:
    return artifacts_root() / "figures" / "shields"


def _clean_suffix(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _map_for_task(cfg: PipelineConfig, task: str) -> tuple[str, ...]:
    if task == "source":
        return cfg.source_map
    if task == "downstream":
        return cfg.downstream_map
    raise ValueError(f"Unsupported task '{task}'.")


def _array_summary(prefix: str, values: np.ndarray | None) -> dict[str, float]:
    if values is None:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
        }
    return {
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_mean": float(arr.mean()),
    }


def _shield_info_summary(info: object) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "shield_winning_state_count": int(np.asarray(getattr(info, "winning_states", [])).size),
        "shield_safe_state_count": int(np.asarray(getattr(info, "safe_states", [])).size),
    }
    vi_steps = getattr(info, "vi_steps", None)
    vi_residual = getattr(info, "vi_residual", None)
    if vi_steps is not None:
        summary["shield_vi_steps"] = int(vi_steps)
    if vi_residual is not None:
        summary["shield_vi_residual"] = float(vi_residual)
    summary.update(_array_summary("shield_state_risk", getattr(info, "state_risk", None)))
    summary.update(_array_summary("shield_action_risk", getattr(info, "action_risk", None)))
    return summary


def save_synthesised_shield_plot(
    *,
    layout: str,
    task: str,
    output_dir: Path,
    basename: str | None = None,
    formats: list[str] | None = None,
    seed: int = 0,
    dpi: int = 300,
    success_rate: float | None = None,
    shield_type: str | None = None,
    shield_risk_threshold: float | None = None,
    shield_theta: float | None = None,
    shield_max_vi_steps: int | None = None,
    unsafe_cost_threshold: float | None = None,
    show_all_actions: bool = True,
    save_tensors: bool = True,
) -> dict[str, Any]:
    if formats is None:
        formats = ["png", "pdf"]

    cfg = get_pipeline_config(layout)
    if success_rate is not None:
        cfg = replace(cfg, success_rate=float(success_rate))
    if not 0.0 <= cfg.success_rate <= 1.0:
        raise ValueError(f"success_rate must be in [0, 1], got {cfg.success_rate}.")

    task_map = _map_for_task(cfg, task)
    resolved_shield_type = str(shield_type or cfg.shield_type)
    resolved_risk_threshold = float(
        cfg.shield_risk_threshold if shield_risk_threshold is None else shield_risk_threshold,
    )
    resolved_theta = float(cfg.shield_theta if shield_theta is None else shield_theta)
    resolved_max_vi_steps = int(
        cfg.shield_max_vi_steps if shield_max_vi_steps is None else shield_max_vi_steps,
    )
    resolved_unsafe_cost_threshold = float(
        cfg.unsafe_cost_threshold if unsafe_cost_threshold is None else unsafe_cost_threshold,
    )

    shield, info = synthesise_frozenlake_shield(
        task_map,
        shield_type=resolved_shield_type,  # type: ignore[arg-type]
        risk_threshold=resolved_risk_threshold,
        theta=resolved_theta,
        max_vi_steps=resolved_max_vi_steps,
        unsafe_cost_threshold=resolved_unsafe_cost_threshold,
        is_slippery=cfg.is_slippery,
        success_rate=cfg.success_rate,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = basename or (
        f"frozenlake_slippery_{layout}_{task}_{resolved_shield_type}_shield_"
        f"risk_{_clean_suffix(resolved_risk_threshold)}_success_{_clean_suffix(cfg.success_rate)}"
    )

    env = make_env(
        task_map,
        task_num=cfg.source_task_num if task == "source" else cfg.downstream_task_num,
        max_episode_steps=cfg.max_episode_steps,
        shaped=False,
        is_slippery=cfg.is_slippery,
        success_rate=cfg.success_rate,
        render_mode="rgb_array",
    )
    fig = None
    saved_paths: list[Path] = []
    try:
        fig = plot_shield_safety_probabilities(
            env,
            shield,
            action_risk=getattr(info, "action_risk", None),
            title=(
                f"{layout} {task} {resolved_shield_type} shield "
                f"(risk <= {resolved_risk_threshold:g}, success_rate={cfg.success_rate:g})"
            ),
            seed=seed,
            show_probability_text=getattr(info, "action_risk", None) is not None,
            show_disallowed_actions=show_all_actions and getattr(info, "action_risk", None) is not None,
        )
        for fmt in formats:
            fmt_clean = fmt.lower().lstrip(".")
            out_path = output_dir / f"{stem}.{fmt_clean}"
            fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
            saved_paths.append(out_path)
            print(f"Saved {out_path}")
    finally:
        env.close()
        if fig is not None:
            plt.close(fig)

    shield_path = output_dir / f"{stem}_shield.pt"
    shield_info_path = output_dir / f"{stem}_shield_info.pt"
    if save_tensors:
        torch.save(torch.as_tensor(shield, dtype=torch.int64), shield_path)
        torch.save(info, shield_info_path)

    allowed_rows = shield[shield.sum(axis=1) > 0]
    stats_payload = {
        "state": torch.empty((int(allowed_rows.shape[0]), 3), dtype=torch.float32),
        "actions": torch.as_tensor(allowed_rows, dtype=torch.float32),
    }

    metadata = {
        "layout": layout,
        "task": task,
        "source_or_downstream_map": list(task_map),
        "is_slippery": bool(cfg.is_slippery),
        "success_rate": float(cfg.success_rate),
        "shield_type": resolved_shield_type,
        "shield_risk_threshold": resolved_risk_threshold,
        "shield_theta": resolved_theta,
        "shield_max_vi_steps": resolved_max_vi_steps,
        "unsafe_cost_threshold": resolved_unsafe_cost_threshold,
        "show_all_actions": bool(show_all_actions),
        "probability_label": "1 - action_risk = probability of eventually staying safe",
        "figure_paths": [str(path) for path in saved_paths],
        "shield_path": str(shield_path) if save_tensors else None,
        "shield_info_path": str(shield_info_path) if save_tensors else None,
        **shield_allowed_action_stats(task_map, shield, stats_payload),
        **_shield_info_summary(info),
    }
    metadata_path = output_dir / f"{stem}_metadata.yaml"
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    print(f"Saved {metadata_path}")

    return {
        "shield": shield,
        "shield_info": info,
        "figure_paths": saved_paths,
        "metadata_path": metadata_path,
        "shield_path": shield_path if save_tensors else None,
        "shield_info_path": shield_info_path if save_tensors else None,
        "metadata": metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthesize a FrozenLake slippery shield and plot it on the environment frame.",
    )
    parser.add_argument("--pipeline", "--layout", dest="layout", default="diagonal_4x4")
    parser.add_argument("--task", choices=["source", "downstream"], default="source")
    parser.add_argument("--output-dir", type=Path, default=default_shield_figure_dir())
    parser.add_argument("--basename", default=None)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--success-rate", type=float, default=None)
    parser.add_argument("--shield-type", choices=["deterministic", "probabilistic"], default=None)
    parser.add_argument("--shield-risk-threshold", type=float, default=None)
    parser.add_argument("--shield-theta", type=float, default=None)
    parser.add_argument("--shield-max-vi-steps", type=int, default=None)
    parser.add_argument("--unsafe-cost-threshold", type=float, default=None)
    parser.add_argument(
        "--allowed-actions-only",
        action="store_true",
        help="For probabilistic shields, hide rejected state-action probabilities.",
    )
    parser.add_argument(
        "--no-save-tensors",
        action="store_true",
        help="Only save figures and metadata, not shield.pt/shield_info.pt.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    return save_synthesised_shield_plot(
        layout=args.layout,
        task=args.task,
        output_dir=args.output_dir,
        basename=args.basename,
        formats=args.formats,
        seed=args.seed,
        dpi=args.dpi,
        success_rate=args.success_rate,
        shield_type=args.shield_type,
        shield_risk_threshold=args.shield_risk_threshold,
        shield_theta=args.shield_theta,
        shield_max_vi_steps=args.shield_max_vi_steps,
        unsafe_cost_threshold=args.unsafe_cost_threshold,
        show_all_actions=not args.allowed_actions_only,
        save_tensors=not args.no_save_tensors,
    )


if __name__ == "__main__":
    main()
