#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import yaml


ACTION_LABELS = ["Left", "Down", "Right", "Up"]
ACTION_DELTAS = {
    0: (0, -1),  # Left
    1: (1, 0),   # Down
    2: (0, 1),   # Right
    3: (-1, 0),  # Up
}
FROZEN_LAKE_DIR = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Rashomon logit bounds and worst-case action probabilities."
    )
    parser.add_argument("--cfg", type=str, default="standard_4x4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=None,
        help="Path to FrozenLake outputs root (default: <frozen_lake>/outputs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots are written (default: outputs/<cfg>/<seed>/rashomon_logit_bounds).",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def stable_softmax(logits: np.ndarray, tau: float = 1.0) -> np.ndarray:
    shifted = (logits - logits.max()) / tau
    exp = np.exp(shifted)
    return exp / exp.sum()


def build_state_observations(n_states: int, in_dim: int) -> np.ndarray:
    has_task_flag = (in_dim == n_states + 1)
    obs = np.zeros((n_states, in_dim), dtype=np.float32)
    for s in range(n_states):
        obs[s, s] = 1.0
        if has_task_flag:
            obs[s, -1] = 0.0
    return obs


def load_env_map(cfg_path: Path, cfg_name: str) -> list[str]:
    with cfg_path.open("r", encoding="utf-8") as f:
        cfgs = yaml.safe_load(f)
    if cfg_name not in cfgs:
        raise KeyError(f"Config '{cfg_name}' not found in {cfg_path}")
    return cfgs[cfg_name]["env1_map"]


def compute_safe_actions(env_map: list[str]) -> dict[int, set[int] | None]:
    grid = [list(row) for row in env_map]
    nrows, ncols = len(grid), len(grid[0])
    hole_states = {
        r * ncols + c
        for r in range(nrows)
        for c in range(ncols)
        if grid[r][c] == "H"
    }
    safe_actions: dict[int, set[int] | None] = {}
    for r in range(nrows):
        for c in range(ncols):
            s = r * ncols + c
            cell = grid[r][c]
            if cell in ("H", "G"):
                safe_actions[s] = None
                continue
            allowed: set[int] = set()
            for a, (dr, dc) in ACTION_DELTAS.items():
                nr, nc = r + dr, c + dc
                hits_wall = (nr < 0 or nr >= nrows or nc < 0 or nc >= ncols)
                if hits_wall:
                    allowed.add(a)
                else:
                    ns = nr * ncols + nc
                    if ns not in hole_states:
                        allowed.add(a)
            safe_actions[s] = allowed
    return safe_actions


def format_axes_clean(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7)


def save_figure(fig: plt.Figure, out_base: Path, dpi: int) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")


def plot_logit_bounds(
    env_map: list[str],
    lb: np.ndarray,
    ub: np.ndarray,
    safe_actions: dict[int, set[int] | None],
    cfg: str,
    seed: int,
    out_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    grid = [list(row) for row in env_map]
    nrows, ncols = len(grid), len(grid[0])

    y_min = float(np.min(lb))
    y_max = float(np.max(ub))
    y_pad = max(0.1, 0.08 * (y_max - y_min))
    y_min -= y_pad
    y_max += y_pad

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.0 * ncols, 2.6 * nrows),
        squeeze=False,
    )

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            s = r * ncols + c
            safe_set = safe_actions.get(s, None)
            if safe_set is None:
                colors = ["gray"] * 4
            else:
                colors = ["tab:green" if a in safe_set else "tab:red" for a in range(4)]

            heights = ub[s] - lb[s]
            ax.bar(
                np.arange(4),
                heights,
                bottom=lb[s],
                color=colors,
                alpha=0.35,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.scatter(
                np.arange(4),
                0.5 * (lb[s] + ub[s]),
                c=colors,
                s=18,
                zorder=3,
            )
            # ax.set_ylim(y_min, y_max)
            ax.set_xticks(np.arange(4))
            ax.set_xticklabels(["L", "D", "R", "U"], fontsize=7)
            ax.set_title(f"{grid[r][c]} ({r},{c})", fontsize=8)
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.3)
            ax.grid(axis="y", alpha=0.2)
            format_axes_clean(ax)

    legend_handles = [
        Patch(facecolor="tab:green", edgecolor="tab:green", label="Safe", alpha=0.5),
        Patch(facecolor="tab:red", edgecolor="tab:red", label="Unsafe", alpha=0.5),
        Patch(facecolor="gray", edgecolor="gray", label="Terminal", alpha=0.5),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        f"Logit bounds per state (Task 1) — {cfg}, seed {seed}",
        y=0.99,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_base = out_dir / f"logit_bounds_task1"
    save_figure(fig, out_base, dpi)
    if show:
        plt.show()
    plt.close(fig)


def plot_worst_case_probs(
    env_map: list[str],
    obs: np.ndarray,
    safe_mask: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    cfg: str,
    seed: int,
    temperature: float,
    out_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    n_states = len(env_map) * len(env_map[0])
    if obs.shape[1] == n_states + 1:
        state_ids = np.argmax(obs[:, :-1], axis=1)
    else:
        state_ids = np.argmax(obs, axis=1)

    worst_case_probs: list[tuple[int, np.ndarray, np.ndarray]] = []
    for i, state_idx in enumerate(state_ids):
        safe_flags = safe_mask[i].astype(bool)
        worst_logits = np.where(safe_flags, lb[i], ub[i])
        probs = stable_softmax(worst_logits, tau=temperature)
        worst_case_probs.append((state_idx, probs, safe_flags))

    n_plots = len(worst_case_probs)
    ncols = 3
    nrows = int(np.ceil(n_plots / ncols)) if n_plots > 0 else 1

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 3.2 * nrows),
        squeeze=False,
    )

    for idx, (state_idx, probs, safe_flags) in enumerate(worst_case_probs):
        ax = axes[idx // ncols, idx % ncols]
        colors = ["tab:green" if flag else "tab:red" for flag in safe_flags]
        ax.bar(
            ACTION_LABELS,
            probs,
            color=colors,
            edgecolor=colors,
            linewidth=1.2,
            alpha=0.9,
            width=0.55,
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Probability", fontsize=8)
        ax.set_title(f"State {state_idx}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        format_axes_clean(ax)
        for i, prob in enumerate(probs):
            ax.text(i, prob + 0.02, f"{prob:.2f}", ha="center", va="bottom", fontsize=7)

    for idx in range(n_plots, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    legend_handles = [
        Patch(facecolor="tab:green", edgecolor="tab:green", label="Safe"),
        Patch(facecolor="tab:red", edgecolor="tab:red", label="Unsafe"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        f"Worst-case action probabilities (T={temperature}) — {cfg}, seed {seed}",
        y=0.99,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_base = out_dir / f"worst_case_action_probs_task1"
    save_figure(fig, out_base, dpi)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root or (FROZEN_LAKE_DIR / "outputs")
    run_dir = outputs_root / args.cfg / str(args.seed)

    plots_dir = args.output_dir if args.output_dir else (run_dir / "rashomon_logit_bounds")
    plots_dir.mkdir(parents=True, exist_ok=True)

    bounded_model_candidates = [
        run_dir / "bounded_model.pt",
        run_dir / "downstream" / "bounded_model.pt",
    ]
    rashomon_dataset_candidates = [
        run_dir / "rashomon_dataset.pt",
        run_dir / "downstream" / "rashomon_dataset.pt",
    ]
    configs_path = FROZEN_LAKE_DIR / "configs.yaml"

    bounded_model_path = next((p for p in bounded_model_candidates if p.exists()), None)
    rashomon_dataset_path = next((p for p in rashomon_dataset_candidates if p.exists()), None)

    if bounded_model_path is None:
        raise FileNotFoundError(f"Missing bounded_model. Checked: {bounded_model_candidates}")
    if rashomon_dataset_path is None:
        raise FileNotFoundError(f"Missing rashomon_dataset. Checked: {rashomon_dataset_candidates}")
    if not configs_path.exists():
        raise FileNotFoundError(f"Missing configs.yaml: {configs_path}")

    bounded_model = torch.load(bounded_model_path, map_location="cpu", weights_only=False)
    rashomon_dataset = torch.load(rashomon_dataset_path, map_location="cpu", weights_only=False)

    env_map = load_env_map(configs_path, args.cfg)
    n_states = len(env_map) * len(env_map[0])

    in_dim = bounded_model.param_n[0].shape[1]
    all_obs = build_state_observations(n_states, in_dim)
    X_all = torch.tensor(all_obs, dtype=torch.float32)

    with torch.no_grad():
        lb_all, ub_all = bounded_model.bound_forward(X_all, X_all)
    lb_all = lb_all.cpu().numpy()
    ub_all = ub_all.cpu().numpy()

    safe_actions = compute_safe_actions(env_map)

    # NeurIPS-friendly styling (compact serif, light grid/lines)
    sns.set_style("whitegrid", {"grid.linestyle": ":", "grid.linewidth": 0.5})
    sns.set_context("paper", font_scale=1.0)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    plot_logit_bounds(
        env_map=env_map,
        lb=lb_all,
        ub=ub_all,
        safe_actions=safe_actions,
        cfg=args.cfg,
        seed=args.seed,
        out_dir=plots_dir,
        dpi=args.dpi,
        show=args.show,
    )

    obs = rashomon_dataset.tensors[0].cpu().numpy()
    safe_mask = rashomon_dataset.tensors[1].cpu().numpy()
    X_crit = torch.tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        lb_crit, ub_crit = bounded_model.bound_forward(X_crit, X_crit)
    lb_crit = lb_crit.cpu().numpy()
    ub_crit = ub_crit.cpu().numpy()

    plot_worst_case_probs(
        env_map=env_map,
        obs=obs,
        safe_mask=safe_mask,
        lb=lb_crit,
        ub=ub_crit,
        cfg=args.cfg,
        seed=args.seed,
        temperature=args.temperature,
        out_dir=plots_dir,
        dpi=args.dpi,
        show=args.show,
    )

    print(f"Loaded bounded_model from: {bounded_model_path}")
    print(f"Loaded rashomon_dataset from: {rashomon_dataset_path}")
    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
