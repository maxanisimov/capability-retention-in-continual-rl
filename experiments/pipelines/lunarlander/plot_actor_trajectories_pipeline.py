"""Plot source/downstream LunarLander rollouts for one pipeline actor."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

_MPLCONFIGDIR = Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.lunarlander.core.env.env_factory import (  # noqa: E402
    _make_lunarlander_env,
)
from experiments.pipelines.lunarlander.core.eval.generate_task_video import (  # noqa: E402
    _load_actor,
    _resolve_actor_checkpoint,
    _resolve_env_config,
)
from experiments.pipelines.lunarlander.core.orchestration.run_paths import (  # noqa: E402
    NOADAPT_POLICY_SUBDIR,
    artifacts_root,
    default_outputs_root,
    default_task_settings_file,
)

ICML_TWO_COLUMN_WIDTH_IN = 6.75
ICML_TRAJECTORY_HEIGHT_IN = 3.1
N_DISPLAY_COLUMNS = 5


@dataclass
class RolloutResult:
    frames: list[np.ndarray]
    steps: list[int]
    actions: list[int]
    rewards: list[float]
    episode_return: float
    episode_length: int


def _apply_icml_style(dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        },
    )


def _sanitize_label(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return cleaned.strip("_") or "trajectory"


def _make_env(env_id: str, env_kwargs: dict[str, Any]):
    return _make_lunarlander_env(env_id, render_mode="rgb_array", **env_kwargs)


def _select_action(
    actor: torch.nn.Module,
    obs: np.ndarray,
    *,
    deterministic: bool,
    device: str,
) -> int:
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = actor(obs_t)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
    return int(action.item())


def _rollout_one_episode(
    *,
    actor: torch.nn.Module,
    env,
    seed: int,
    max_steps: int,
    deterministic: bool,
    device: str,
) -> RolloutResult:
    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {max_steps}.")

    frames: list[np.ndarray] = []
    steps: list[int] = []
    actions: list[int] = []
    rewards: list[float] = []

    obs, _ = env.reset(seed=seed)
    frame = env.render()
    if frame is None:
        raise RuntimeError("env.render() returned None; create env with render_mode='rgb_array'.")
    frames.append(np.asarray(frame).copy())
    steps.append(0)

    episode_return = 0.0
    episode_length = 0
    done = False
    while not done and episode_length < max_steps:
        action = _select_action(
            actor,
            np.asarray(obs, dtype=np.float32),
            deterministic=deterministic,
            device=device,
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)
        episode_length += 1
        done = bool(terminated or truncated)

        frame = env.render()
        if frame is None:
            raise RuntimeError("env.render() returned None; create env with render_mode='rgb_array'.")
        frames.append(np.asarray(frame).copy())
        steps.append(episode_length)
        actions.append(action)
        rewards.append(float(reward))

    return RolloutResult(
        frames=frames,
        steps=steps,
        actions=actions,
        rewards=rewards,
        episode_return=episode_return,
        episode_length=episode_length,
    )


def _select_display_indices(n_frames: int, n_cols: int = N_DISPLAY_COLUMNS) -> np.ndarray:
    if n_frames <= 0:
        raise ValueError("Cannot select display frames from an empty rollout.")
    if n_cols != N_DISPLAY_COLUMNS:
        raise ValueError(f"This figure expects exactly {N_DISPLAY_COLUMNS} columns.")
    if n_frames == 1:
        return np.zeros(n_cols, dtype=int)

    interior = np.linspace(0, n_frames - 1, n_cols)[1:-1]
    indices = np.concatenate(
        [
            np.array([0]),
            np.rint(interior).astype(int),
            np.array([n_frames - 1]),
        ],
    )
    indices[0] = 0
    indices[-1] = n_frames - 1
    return np.clip(indices, 0, n_frames - 1)


def _plot_rollout_grid(
    *,
    source: RolloutResult,
    downstream: RolloutResult,
    output_paths: list[Path],
    dpi: int,
    figure_width: float,
    figure_height: float,
) -> tuple[list[int], list[int]]:
    _apply_icml_style(dpi)

    source_indices = _select_display_indices(len(source.frames))
    downstream_indices = _select_display_indices(len(downstream.frames))

    fig, axes = plt.subplots(
        2,
        N_DISPLAY_COLUMNS,
        figsize=(figure_width, figure_height),
        constrained_layout=False,
    )

    rows = [
        ("Source", source, source_indices),
        ("Downstream", downstream, downstream_indices),
    ]
    for row_idx, (row_label, rollout, indices) in enumerate(rows):
        for col_idx, frame_idx in enumerate(indices):
            ax = axes[row_idx, col_idx]
            ax.imshow(rollout.frames[int(frame_idx)], interpolation="nearest")
            ax.set_title(f"step {rollout.steps[int(frame_idx)]}", pad=2.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col_idx == 0:
                ax.text(
                    0.02,
                    0.94,
                    row_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    color="white",
                    fontsize=7,
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "black",
                        "edgecolor": "none",
                        "alpha": 0.65,
                    },
                )

    fig.subplots_adjust(
        left=0.01,
        right=0.995,
        bottom=0.025,
        top=0.955,
        wspace=0.025,
        hspace=0.22,
    )
    try:
        for output_path in output_paths:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=dpi)
    finally:
        plt.close(fig)

    return source_indices.astype(int).tolist(), downstream_indices.astype(int).tolist()


def _save_rollout_frames_npz(
    *,
    output_path: Path,
    source: RolloutResult,
    downstream: RolloutResult,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        source_frames=np.stack(source.frames, axis=0),
        source_steps=np.asarray(source.steps, dtype=np.int32),
        source_actions=np.asarray(source.actions, dtype=np.int32),
        source_rewards=np.asarray(source.rewards, dtype=np.float32),
        downstream_frames=np.stack(downstream.frames, axis=0),
        downstream_steps=np.asarray(downstream.steps, dtype=np.int32),
        downstream_actions=np.asarray(downstream.actions, dtype=np.int32),
        downstream_rewards=np.asarray(downstream.rewards, dtype=np.float32),
    )


def _write_metadata(
    *,
    output_path: Path,
    metadata: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load one LunarLander actor for a pipeline/seed, roll it out once in "
            "the source and downstream tasks, and save a 2x5 trajectory figure."
        ),
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        dest="task_setting",
        required=True,
        help="Task pipeline key, e.g. deterministic__default_to_sluggish_vehicle.",
    )
    parser.add_argument("--task-setting", type=str, dest="task_setting", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, required=True, help="Training seed and default rollout seed.")
    parser.add_argument(
        "--rollout-seed",
        type=int,
        default=None,
        help="Optional rollout seed. Defaults to --seed.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=NOADAPT_POLICY_SUBDIR,
        help=(
            "Policy subdirectory under outputs/<pipeline>/seed_<seed>/, or an "
            "explicit actor/policy path. Defaults to noadapt."
        ),
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=default_outputs_root(),
        help="Runs root used to resolve policy directories.",
    )
    parser.add_argument(
        "--task-settings-file",
        type=Path,
        default=default_task_settings_file(),
        help="Task pipeline settings YAML.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for actor inference.")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic argmax actions during rollout.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Safety cap for each source/downstream rollout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=artifacts_root() / "figures",
        help="Directory for the figure, frame archive, and metadata.",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default=None,
        help="Output filename stem without extension.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Figure formats to save, e.g. pdf png svg.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Raster output DPI.")
    parser.add_argument(
        "--figure-width",
        type=float,
        default=ICML_TWO_COLUMN_WIDTH_IN,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=ICML_TRAJECTORY_HEIGHT_IN,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--save-rollout-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save complete source/downstream rollout frames as a compressed NPZ archive.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_steps <= 0:
        raise ValueError(f"--max-steps must be > 0, got {args.max_steps}.")
    if args.dpi <= 0:
        raise ValueError(f"--dpi must be > 0, got {args.dpi}.")
    if not args.formats:
        raise ValueError("--formats must contain at least one output format.")

    rollout_seed = int(args.seed if args.rollout_seed is None else args.rollout_seed)
    policy_label = _sanitize_label(str(args.policy))
    stem = args.basename or (
        f"lunarlander_{_sanitize_label(args.task_setting)}"
        f"_seed_{int(args.seed)}_policy_{policy_label}_trajectories"
    )

    actor_path, resolved_policy_label = _resolve_actor_checkpoint(
        policy=str(args.policy),
        outputs_root=args.outputs_root,
        task_setting=str(args.task_setting),
        seed=int(args.seed),
    )
    if actor_path is None:
        raise ValueError("This script requires a trained actor; use a policy other than 'random'.")
    actor = _load_actor(actor_path, device=str(args.device))
    actor.eval()

    source_env_id, source_env_kwargs = _resolve_env_config(
        task_settings_file=args.task_settings_file,
        task_setting=str(args.task_setting),
        task_role="source",
    )
    downstream_env_id, downstream_env_kwargs = _resolve_env_config(
        task_settings_file=args.task_settings_file,
        task_setting=str(args.task_setting),
        task_role="downstream",
    )

    source_env = _make_env(source_env_id, source_env_kwargs)
    downstream_env = _make_env(downstream_env_id, downstream_env_kwargs)
    try:
        source_rollout = _rollout_one_episode(
            actor=actor,
            env=source_env,
            seed=rollout_seed,
            max_steps=int(args.max_steps),
            deterministic=bool(args.deterministic),
            device=str(args.device),
        )
        downstream_rollout = _rollout_one_episode(
            actor=actor,
            env=downstream_env,
            seed=rollout_seed,
            max_steps=int(args.max_steps),
            deterministic=bool(args.deterministic),
            device=str(args.device),
        )
    finally:
        source_env.close()
        downstream_env.close()

    figure_paths = [
        args.output_dir / f"{stem}.{fmt.lower().lstrip('.')}"
        for fmt in args.formats
    ]
    source_indices, downstream_indices = _plot_rollout_grid(
        source=source_rollout,
        downstream=downstream_rollout,
        output_paths=figure_paths,
        dpi=int(args.dpi),
        figure_width=float(args.figure_width),
        figure_height=float(args.figure_height),
    )

    frames_path = args.output_dir / f"{stem}_frames.npz"
    if args.save_rollout_frames:
        _save_rollout_frames_npz(
            output_path=frames_path,
            source=source_rollout,
            downstream=downstream_rollout,
        )

    metadata_path = args.output_dir / f"{stem}_metadata.yaml"
    metadata = {
        "task_setting": str(args.task_setting),
        "seed": int(args.seed),
        "rollout_seed": rollout_seed,
        "policy": str(args.policy),
        "resolved_policy_label": str(resolved_policy_label),
        "actor_path": str(actor_path),
        "outputs_root": str(args.outputs_root),
        "task_settings_file": str(args.task_settings_file),
        "device": str(args.device),
        "deterministic": bool(args.deterministic),
        "max_steps": int(args.max_steps),
        "source": {
            "env_id": str(source_env_id),
            "env_kwargs": source_env_kwargs,
            "episode_return": float(source_rollout.episode_return),
            "episode_length": int(source_rollout.episode_length),
            "n_frames": int(len(source_rollout.frames)),
            "selected_frame_indices": source_indices,
            "selected_steps": [int(source_rollout.steps[i]) for i in source_indices],
        },
        "downstream": {
            "env_id": str(downstream_env_id),
            "env_kwargs": downstream_env_kwargs,
            "episode_return": float(downstream_rollout.episode_return),
            "episode_length": int(downstream_rollout.episode_length),
            "n_frames": int(len(downstream_rollout.frames)),
            "selected_frame_indices": downstream_indices,
            "selected_steps": [int(downstream_rollout.steps[i]) for i in downstream_indices],
        },
        "figure_paths": [str(path) for path in figure_paths],
        "frames_path": str(frames_path) if args.save_rollout_frames else None,
        "metadata_path": str(metadata_path),
    }
    _write_metadata(output_path=metadata_path, metadata=metadata)

    for path in figure_paths:
        print(f"Saved figure: {path}")
    if args.save_rollout_frames:
        print(f"Saved rollout frames: {frames_path}")
    print(f"Saved metadata: {metadata_path}")
    print(
        "Source return/length: "
        f"{source_rollout.episode_return:.3f}/{source_rollout.episode_length}; "
        "Downstream return/length: "
        f"{downstream_rollout.episode_return:.3f}/{downstream_rollout.episode_length}",
    )


if __name__ == "__main__":
    main()
