#!/usr/bin/env python
"""Plot unshielded-policy learning curves (safety rate, success rate, reward).

Aggregates the per-seed ``unshielded_reward_evaluations`` curves written by
``utils/learning_curves.LearningCurveLogger`` into one figure per environment:
mean +/- N standard errors across seeds, one line per method.

Expects the ``<root>/<env>/seed<N>/<method_dir>/summary.json`` layout produced
by the ``paper_2503_07671_experiments*`` pipelines, where each method's
``summary.json`` holds (possibly wrapped in a named sub-key, for method
directories that bundle more than one algorithm) a
``learning_curves.unshielded_reward_evaluations`` list of per-eval dicts with
``timestep``, ``safety_rate``, ``success_rate``, ``mean_total_reward``.

Example
-------
Media Streaming, no-earlystop batch, all default methods::

    python projects/safe_policy_optimisation/scripts/plot_unshielded_learning_curves.py \
        --root artifacts/paper_2503_07671/runs/no_earlystop \
        --env media_streaming

Only two methods, custom output path::

    python .../plot_unshielded_learning_curves.py --root ... --env colour_bomb \
        --method shielded_policy --method rashomon_policy --out /tmp/colour_bomb.png
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

METRICS = [
    ("safety_rate", "Safety rate"),
    ("success_rate", "Success rate"),
    ("mean_total_reward", "Total reward"),
]

# (display name, method subdir, json key to unwrap [None = top-level summary],
#  color, linestyle). Colors/order follow the project's validated categorical
# palette (blue, aqua, yellow, green, violet, red - CVD-safe adjacent-pair
# ordering, slots 1-6). Linestyles differ so perfectly-overlapping curves (e.g.
# safety_rate == 1.0 for several baselines) stay visually distinguishable
# instead of hiding each other, and so the figure survives grayscale printing
# (6-way shape redundancy, not just color) - required for the AAAI
# camera-ready grid figure.
#
# The last element is the summary.json curve key. ``None`` means "the default
# curve for this run" (unshielded, falling back to plain PPO's unprefixed key).
# ``shielded_reward_evaluations`` is the deployed system: the shield overrides
# unsafe proposed actions. So "PPO-Shield" is what you would ship, while
# "PPO-Shield-Nominal" is the same policy evaluated *without* its shield, which
# is what reveals whether safety was actually internalised.
DEFAULT_METHODS: list[tuple[str, str, str | None, str, str | tuple, str | None]] = [
    ("PPO-Lagrangian", "ppo_lagrangian", "ppo_lagrangian", "#2a78d6", (0, (1, 1)), None),
    ("PPO-PID-Lagrangian", "ppo_lagrangian", "ppo_pid_lagrangian", "#1baf7a", (0, (5, 2)), None),
    ("CPO", "cpo", "cpo", "#eda100", (0, (3, 1, 1, 1)), None),
    ("PSPO (Rashomon)", "rashomon_policy", None, "#008300", (0, (3, 1, 1, 1, 1, 1)), None),
    ("PPO-Shield", "shielded_policy", None, "#4a3aa7", "solid", "shielded_reward_evaluations"),
    ("PPO-Shield-Nominal", "shielded_policy", None, "#8f83d4", (0, (6, 2)), None),
    ("PPO", "ppo_policy", None, "#e34948", (0, (4, 1)), None),
    # Rashomon-set-size ablation (not part of the default comparison; select
    # explicitly with --method for a focused 2K-vs-10K figure).
    ("PSPO (Rashomon, 2K)", "rashomon_policy", None, "#008300", "solid", None),
    ("PSPO (Rashomon, 10K)", "rashomon_policy_10k", None, "#e87ba4", (0, (4, 2)), None),
]

# Environment directory name -> display label for figure titles.
ENV_DISPLAY_NAMES: dict[str, str] = {
    "bridge_crossing": "Bridge Crossing",
    "bridge_crossing_v2": "Bridge Crossing v2",
    "colour_bomb": "Colour Bomb",
    "colour_bomb_v2": "Colour Bomb v2",
    "media_streaming": "Media Streaming",
    "mini_pacman": "MiniPacman",
}


def display_env_name(env: str) -> str:
    return ENV_DISPLAY_NAMES.get(env, env.replace("_", " ").title())

# Chart chrome, matching the project's dataviz-skill reference palette (light mode).
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

_SEED_DIR_RE = re.compile(r"^seed(\d+)$")


@dataclass
class MethodSpec:
    name: str
    subdir: str
    json_key: str | None
    color: str
    linestyle: str | tuple
    curve_key: str | None = None


def discover_seeds(env_root: Path) -> list[int]:
    seeds = []
    for child in env_root.iterdir():
        m = _SEED_DIR_RE.match(child.name)
        if m and child.is_dir():
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def load_curve(env_root: Path, seed: int, spec: MethodSpec) -> list[dict] | None:
    path = env_root / f"seed{seed}" / spec.subdir / "summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    node = data[spec.json_key] if spec.json_key else data
    curves = node.get("learning_curves") or {}
    if spec.curve_key:
        # Absent for runs predating that curve being logged; aggregate_method
        # reports the method as skipped rather than failing the whole figure.
        return curves.get(spec.curve_key)
    # Plain PPO trains directly in the unshielded env, so its summary has no
    # shielded/unshielded distinction and logs under the unprefixed key.
    return curves.get("unshielded_reward_evaluations") or curves.get("reward_evaluations")


def aggregate_method(
    env_root: Path, seeds: list[int], spec: MethodSpec
) -> dict[str, np.ndarray] | None:
    per_seed = []
    for seed in seeds:
        evals = load_curve(env_root, seed, spec)
        if evals:
            per_seed.append(evals)
    if not per_seed:
        print(f"  [skip] {spec.name}: no data found under any seed")
        return None
    min_len = min(len(c) for c in per_seed)
    if any(len(c) != min_len for c in per_seed):
        print(f"  [warn] {spec.name}: uneven eval counts across seeds; truncating to {min_len}")
    timesteps = np.array([e["timestep"] for e in per_seed[0][:min_len]])
    out = {"timesteps": timesteps}
    for metric, _ in METRICS:
        vals = np.array([[e[metric] for e in c[:min_len]] for c in per_seed])
        mean = vals.mean(axis=0)
        se = vals.std(axis=0, ddof=1) / np.sqrt(vals.shape[0]) if vals.shape[0] > 1 else np.zeros_like(mean)
        out[metric] = (mean, se)
    out["n_seeds"] = len(per_seed)
    return out


def _abbreviate_thousands(value: float, _pos: int) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def plot(
    env: str, env_root: Path, seeds: list[int], methods: list[MethodSpec],
    *, ci_multiplier: float, out_path: Path,
) -> None:
    data: dict[str, dict] = {}
    for spec in methods:
        agg = aggregate_method(env_root, seeds, spec)
        if agg is not None:
            data[spec.name] = agg

    if not data:
        raise SystemExit(f"No learning-curve data found for env={env!r} under {env_root}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_PRIMARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)

    for ax, (metric, title) in zip(axes, METRICS):
        ax.set_facecolor(SURFACE)
        for spec in methods:
            if spec.name not in data:
                continue
            agg = data[spec.name]
            t = agg["timesteps"]
            mean, se = agg[metric]
            ax.plot(t, mean, color=spec.color, linewidth=2, linestyle=spec.linestyle, label=spec.name)
            band = ci_multiplier * se
            ax.fill_between(t, mean - band, mean + band, color=spec.color, alpha=0.15, linewidth=0)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Training timestep")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(_abbreviate_thousands))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_color(AXIS)

    axes[0].set_ylabel("Rate")
    axes[2].set_ylabel("Mean episodic reward")

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False,
               bbox_to_anchor=(0.5, -0.05), fontsize=9.5)

    n_seeds_seen = sorted({agg["n_seeds"] for agg in data.values()})
    seeds_label = str(n_seeds_seen[0]) if len(n_seeds_seen) == 1 else "/".join(map(str, n_seeds_seen))
    # Curves are unshielded-policy unless a shielded series is on the figure, in
    # which case the title must not claim otherwise.
    plotted = [spec for spec in methods if spec.name in data]
    kind = "learning curves" if any(s.curve_key for s in plotted) else "unshielded-policy learning curves"
    fig.suptitle(
        f"{env} — {kind} (mean ± {ci_multiplier:g} SE over {seeds_label} seeds)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_grid(
    envs: list[str], root: Path, methods: list[MethodSpec],
    *, ci_multiplier: float, out_path: Path, seeds: list[int] | None,
) -> None:
    """Paper-ready grid: one column per environment, one row per metric.

    Sized and font-scaled for a full-textwidth LaTeX figure (``\\figure*`` at
    ~7in wide in a two-column AAAI layout) - vector PDF output so the LaTeX
    scale factor never blurs it, and line *shape* (not just color) separates
    all 5 methods so the figure still reads correctly printed in grayscale.
    No caption/suptitle is baked in: that belongs in the LaTeX ``\\caption``.
    """

    per_env: dict[str, dict[str, dict]] = {}
    for env in envs:
        env_root = root / env
        if not env_root.is_dir():
            print(f"  [skip] {env}: no such directory under {root}")
            continue
        env_seeds = seeds if seeds is not None else discover_seeds(env_root)
        if not env_seeds:
            print(f"  [skip] {env}: no seed<N> directories found")
            continue
        data: dict[str, dict] = {}
        for spec in methods:
            agg = aggregate_method(env_root, env_seeds, spec)
            if agg is not None:
                data[spec.name] = agg
        if data:
            per_env[env] = data

    envs = [e for e in envs if e in per_env]
    if not envs:
        raise SystemExit(f"No learning-curve data found for any of the requested environments under {root}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 7,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_PRIMARY,
        "axes.labelsize": 7.5,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.linewidth": 0.6,
    })

    ncols = len(envs)
    fig, axes = plt.subplots(
        3, ncols, figsize=(min(7.2, 1.25 * ncols), 4.6), sharex="col",
        facecolor=SURFACE, squeeze=False,
    )

    for col, env in enumerate(envs):
        data = per_env[env]
        for row, (metric, title) in enumerate(METRICS):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE)
            for spec in methods:
                if spec.name not in data:
                    continue
                agg = data[spec.name]
                t = agg["timesteps"]
                mean, se = agg[metric]
                ax.plot(t, mean, color=spec.color, linewidth=1.2, linestyle=spec.linestyle, label=spec.name)
                band = ci_multiplier * se
                ax.fill_between(t, mean - band, mean + band, color=spec.color, alpha=0.12, linewidth=0)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color(AXIS)
            ax.spines["bottom"].set_color(AXIS)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
            ax.xaxis.set_major_formatter(FuncFormatter(_abbreviate_thousands))

            if row == 0:
                ax.set_title(display_env_name(env), fontsize=7.5, fontweight="bold", pad=4)
            if row in (0, 1):
                ax.set_ylim(-0.03, 1.05)
                if col != 0:
                    ax.tick_params(labelleft=False)
            if row == 2 and col != 0:
                ax.tick_params(labelleft=True)
            if col == 0:
                ax.set_ylabel(title, fontsize=7.5)
            if row == 2:
                ax.set_xlabel("Timestep", fontsize=7)
            else:
                ax.tick_params(labelbottom=False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if not handles:
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=7)

    fig.tight_layout(rect=[0, 0.035, 1, 1], w_pad=0.4, h_pad=0.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    if out_path.suffix.lower() != ".png":
        preview_path = out_path.with_suffix(".png")
        fig.savefig(preview_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved {preview_path} (preview)")
    plt.close(fig)
    print(f"Saved {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean +/- SE unshielded-policy learning curves across seeds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root", type=Path, required=True,
        help="Experiment batch root, e.g. artifacts/paper_2503_07671/runs/no_earlystop",
    )
    env_group = parser.add_mutually_exclusive_group(required=True)
    env_group.add_argument("--env", help="Single environment subdirectory name, e.g. media_streaming")
    env_group.add_argument(
        "--envs", nargs="+", metavar="NAME",
        help="Two or more environments -> one paper-ready grid figure (rows=metrics, cols=envs), "
             "vector PDF by default.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Explicit seed list. Default: auto-discover all seed<N> dirs under <root>/<env>.",
    )
    parser.add_argument(
        "--method", dest="methods", action="append", default=None, metavar="NAME",
        help="Restrict to one method by display name (repeatable). Default: all of "
             + ", ".join(m[0] for m in DEFAULT_METHODS),
    )
    parser.add_argument("--ci-multiplier", type=float, default=2.0, help="Std-error band width (default 2.0).")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output path. Default (single env): <root>/_learning_curves/<env>_unshielded_all_methods.png. "
             "Default (--envs grid): <root>/_learning_curves/all_envs_unshielded_all_methods.pdf",
    )
    return parser.parse_args(argv)


def _resolve_methods(args: argparse.Namespace) -> list[MethodSpec]:
    all_specs = [MethodSpec(*m) for m in DEFAULT_METHODS]
    if not args.methods:
        return all_specs
    wanted = set(args.methods)
    specs = [s for s in all_specs if s.name in wanted]
    missing = wanted - {s.name for s in specs}
    if missing:
        raise SystemExit(f"Unknown method name(s): {sorted(missing)}. "
                          f"Choices: {[s.name for s in all_specs]}")
    return specs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specs = _resolve_methods(args)

    if args.envs:
        out_path = args.out or (args.root / "_learning_curves" / "all_envs_unshielded_all_methods.pdf")
        print(f"Root: {args.root}")
        print(f"Environments ({len(args.envs)}): {args.envs}")
        print(f"Methods: {[s.name for s in specs]}")
        plot_grid(args.envs, args.root, specs, ci_multiplier=args.ci_multiplier, out_path=out_path, seeds=args.seeds)
        return 0

    env_root = args.root / args.env
    if not env_root.is_dir():
        raise SystemExit(f"No such environment directory: {env_root}")

    seeds = args.seeds if args.seeds is not None else discover_seeds(env_root)
    if not seeds:
        raise SystemExit(f"No seed<N> directories found under {env_root}")

    out_path = args.out or (args.root / "_learning_curves" / f"{args.env}_unshielded_all_methods.png")

    print(f"Env root: {env_root}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Methods: {[s.name for s in specs]}")

    plot(args.env, env_root, seeds, specs, ci_multiplier=args.ci_multiplier, out_path=out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
