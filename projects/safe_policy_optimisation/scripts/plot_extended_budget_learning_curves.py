#!/usr/bin/env python3
"""Plot evaluation- and exploration-time learning curves for paper environments.

Each curve is the across-seed mean. Shaded bands show a configurable multiple
of the standard error (two by default). Evaluation curves use deterministic
periodic evaluations; exploration curves use completed training episodes,
aggregated within timestep bins per seed before statistics are computed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from projects.safe_policy_optimisation.utils.cli import (  # noqa: E402
    add_legacy_option,
    resolve_legacy_options,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paper_style import (  # noqa: E402
    METHOD_COLORS,
    METHOD_LINESTYLES,
    TEXT_WIDTH_IN,
    apply_paper_style,
    save_paper_figure,
)

RUNS = REPO / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
DEFAULT_BASELINE_ROOT = RUNS / "rl_baselines_extended_budgets_1600k/bridge_crossing_v2"
DEFAULT_ADAPTIVE_ROOT = (
    RUNS
    / "pspo_adaptive_bridge_v2_safe_entropy_w1_min0p95_freq1_t1600k_reuse_base"
    / "two_hidden/bridge_crossing_v2"
)
DEFAULT_OUTPUT_DIR = (
    REPO
    / "projects/safe_policy_optimisation/figures"
    / "pspo_vs_rl_baselines_learning_curves_all_envs"
)

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
# Panel layouts. "metric-rows" is 2 rows (reward, safety) x N environment
# columns; "environment-rows" is its transpose, N environment rows x 2
# metric columns. Both render at ICLR \textwidth.
METRIC_ROWS = "metric-rows"
ENVIRONMENT_ROWS = "environment-rows"
LAYOUTS = (METRIC_ROWS, ENVIRONMENT_ROWS)

SEED_RE = re.compile(r"^seed(\d+)$")
EXPECTED_SEEDS = tuple(range(10))
ROLLOUT_SIZE = 2048


@dataclass(frozen=True)
class EnvironmentSpec:
    key: str
    label: str
    nominal_budget: int
    baseline_root: Path
    adaptive_root: Path

    @property
    def horizon(self) -> int:
        """First rollout boundary at or beyond the nominal training budget."""
        return math.ceil(self.nominal_budget / ROLLOUT_SIZE) * ROLLOUT_SIZE


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    color: str
    linestyle: str | tuple
    evaluation_path: str
    exploration_path: str
    algorithm_filter: str
    adaptive: bool = False


ENVIRONMENTS = (
    EnvironmentSpec(
        "media_streaming",
        "Media Streaming",
        25_000,
        REPO / "outputs/_sweeps_2hidden_media_streaming_baselines_only/media_streaming",
        RUNS
        / "pspo_adaptive_cb_media_safe_entropy_w1_min0p95_freq1_default"
        / "two_hidden/media_streaming",
    ),
    EnvironmentSpec(
        "colour_bomb",
        "Colour Bomb v1",
        25_000,
        REPO / "outputs/_sweeps_2hidden_colour_bomb_baselines_only/colour_bomb",
        RUNS
        / "pspo_adaptive_cb_media_safe_entropy_w1_min0p95_freq1_default"
        / "two_hidden/colour_bomb",
    ),
    EnvironmentSpec(
        "colour_bomb_v2",
        "Colour Bomb v2",
        100_000,
        REPO / "outputs/_sweeps_2hidden_colour_bomb_v2_baselines_only/colour_bomb_v2",
        RUNS
        / "pspo_adaptive_cb_media_safe_entropy_w1_min0p95_freq1_default"
        / "two_hidden/colour_bomb_v2",
    ),
    EnvironmentSpec(
        "bridge_crossing",
        "Bridge Crossing v1",
        200_000,
        REPO / "outputs/_sweeps_2hidden_bridge_crossing_baselines_only/bridge_crossing",
        RUNS
        / "pspo_adaptive_bridge_v1_safe_entropy_w1_min0p95_freq1"
        / "two_hidden/bridge_crossing",
    ),
    EnvironmentSpec(
        "bridge_crossing_v2",
        "Bridge Crossing v2",
        1_600_000,
        DEFAULT_BASELINE_ROOT,
        DEFAULT_ADAPTIVE_ROOT,
    ),
    # Extended budget, like Bridge Crossing v2. The baselines come from
    # rl_baselines_extended_budgets/ (2M, budget-matched to the PSPO run), not
    # from rl_baselines_extended_budgets_1600k/, whose mini_pacman sibling does
    # not exist -- that root holds only the 1.6M bridge_crossing_v2 rerun.
    EnvironmentSpec(
        "mini_pacman",
        "MiniPacman",
        2_000_000,
        RUNS / "rl_baselines_extended_budgets/mini_pacman",
        RUNS
        / "pspo_adaptive_mini_pacman_safe_entropy_w1_min0p95_freq100_t2000k_reuse_base"
        / "two_hidden/mini_pacman",
    ),
)
ENVIRONMENT_BY_KEY = {environment.key: environment for environment in ENVIRONMENTS}

METHODS = (
    MethodSpec(
        "ppo_policy",
        "PPO",
        METHOD_COLORS["ppo_policy"],
        METHOD_LINESTYLES["ppo_policy"],
        "ppo_policy/learning_curves/evaluation_unshielded_summary.csv",
        "ppo_policy/training_episodes.csv",
        "plain_ppo",
    ),
    MethodSpec(
        "ppo_lagrangian",
        "PPO-Lagrangian",
        METHOD_COLORS["ppo_lagrangian"],
        METHOD_LINESTYLES["ppo_lagrangian"],
        "ppo_lagrangian/learning_curves/ppo_lagrangian/evaluation_unshielded_summary.csv",
        "ppo_lagrangian/training_episodes.csv",
        "ppo_lagrangian",
    ),
    MethodSpec(
        "ppo_pid_lagrangian",
        "PPO-PID-Lagrangian",
        METHOD_COLORS["ppo_pid_lagrangian"],
        METHOD_LINESTYLES["ppo_pid_lagrangian"],
        "ppo_lagrangian/learning_curves/ppo_pid_lagrangian/evaluation_unshielded_summary.csv",
        "ppo_lagrangian/training_episodes.csv",
        "ppo_pid_lagrangian",
    ),
    MethodSpec(
        "cpo",
        "CPO",
        METHOD_COLORS["cpo"],
        METHOD_LINESTYLES["cpo"],
        "cpo/learning_curves/cpo/evaluation_unshielded_summary.csv",
        "cpo/training_episodes.csv",
        "cpo",
    ),
    MethodSpec(
        "pspo",
        "PSPO",
        METHOD_COLORS["pspo"],
        METHOD_LINESTYLES["pspo"],
        "learning_curves/evaluation_unshielded_summary.csv",
        "training_episodes.csv",
        "shielded_ppo",
        adaptive=True,
    ),
    MethodSpec(
        "ppo_shield",
        "PPO-Shield",
        METHOD_COLORS["ppo_shield"],
        METHOD_LINESTYLES["ppo_shield"],
        "ppo_shield/learning_curves/evaluation_shielded_summary.csv",
        "ppo_shield/training_episodes.csv",
        "shielded_ppo",
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        action="append",
        choices=tuple(ENVIRONMENT_BY_KEY),
        help="Environment to include; repeat as needed (default: all six).",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help="Override the baseline root when plotting exactly one environment.",
    )
    parser.add_argument(
        "--pspo-root",
        dest="adaptive_root",
        type=Path,
        help="Override the PSPO root when plotting exactly one environment.",
    )
    add_legacy_option(parser, "--adaptive-root", "adaptive_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--layout",
        action="append",
        choices=LAYOUTS,
        help=(
            "Panel layout; repeat to write both. "
            "'metric-rows' (default) is reward/safety rows x environment "
            "columns; 'environment-rows' is the transpose, one row per "
            "environment. The transposed files carry a "
            "'_by_environment' suffix, so the two never collide."
        ),
    )
    parser.add_argument(
        "--budget",
        type=int,
        help="Override nominal budget when plotting exactly one environment.",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=ROLLOUT_SIZE,
        help=f"Exploration timestep-bin width (default: {ROLLOUT_SIZE}).",
    )
    parser.add_argument(
        "--ci-multiplier",
        type=float,
        default=2.0,
        help="Multiplier applied to standard-error bands (default: 2).",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit missing seeds/checkpoints; intended only for diagnostics.",
    )
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    return resolve_legacy_options(
        parser,
        args,
        raw_argv,
        (("--adaptive-root", "--pspo-root", "adaptive_root"),),
    )


def selected_environments(args: argparse.Namespace) -> tuple[EnvironmentSpec, ...]:
    keys = args.environment or [environment.key for environment in ENVIRONMENTS]
    environments = tuple(ENVIRONMENT_BY_KEY[key] for key in dict.fromkeys(keys))
    overrides = (args.baseline_root, args.adaptive_root, args.budget)
    if any(value is not None for value in overrides):
        if len(environments) != 1:
            raise SystemExit(
                "--baseline-root, --pspo-root, and --budget require exactly one "
                "--environment"
            )
        environment = environments[0]
        environments = (
            replace(
                environment,
                baseline_root=args.baseline_root or environment.baseline_root,
                adaptive_root=args.adaptive_root or environment.adaptive_root,
                nominal_budget=args.budget or environment.nominal_budget,
            ),
        )
    return environments


def discover_seeds(root: Path) -> list[tuple[int, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {root}")
    seeds: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = SEED_RE.fullmatch(child.name)
        if match and child.is_dir():
            seeds.append((int(match.group(1)), child))
    return sorted(seeds)


def source_root(spec: MethodSpec, environment: EnvironmentSpec) -> Path:
    return environment.adaptive_root if spec.adaptive else environment.baseline_root


def validate_sources(environment: EnvironmentSpec) -> None:
    """Require every method to contain seeds 0--9 through the final checkpoint."""
    problems: list[str] = []
    for spec in METHODS:
        root = source_root(spec, environment)
        try:
            seed_dirs = dict(discover_seeds(root))
        except FileNotFoundError as error:
            problems.append(str(error))
            continue
        missing_seeds = sorted(set(EXPECTED_SEEDS) - set(seed_dirs))
        if missing_seeds:
            problems.append(
                f"{environment.key}/{spec.key}: missing seeds {missing_seeds}"
            )
        for seed in EXPECTED_SEEDS:
            seed_dir = seed_dirs.get(seed)
            if seed_dir is None:
                continue
            evaluation_path = seed_dir / spec.evaluation_path
            exploration_path = seed_dir / spec.exploration_path
            if not evaluation_path.is_file():
                problems.append(f"missing {evaluation_path}")
            else:
                timesteps = pd.read_csv(evaluation_path, usecols=["timestep"])[
                    "timestep"
                ]
                if timesteps.empty or int(timesteps.max()) < environment.horizon:
                    last = "none" if timesteps.empty else str(int(timesteps.max()))
                    problems.append(
                        f"{environment.key}/{spec.key}/seed{seed}: final evaluation "
                        f"is {last}, expected at least {environment.horizon}"
                    )
            if not exploration_path.is_file():
                problems.append(f"missing {exploration_path}")
    if problems:
        preview = "\n  - ".join(problems[:30])
        suffix = "" if len(problems) <= 30 else f"\n  ... and {len(problems) - 30} more"
        raise RuntimeError(f"Incomplete learning-curve inputs:\n  - {preview}{suffix}")


def _summary_stats(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = (
        frame.groupby(group_columns, sort=True, observed=True)
        .agg(
            seed_count=("seed", "nunique"),
            reward_mean=("reward", "mean"),
            reward_sd=("reward", "std"),
            safety_mean=("safety", "mean"),
            safety_sd=("safety", "std"),
        )
        .reset_index()
    )
    grouped[["reward_sd", "safety_sd"]] = grouped[["reward_sd", "safety_sd"]].fillna(
        0.0
    )
    grouped["reward_sem"] = grouped["reward_sd"] / np.sqrt(grouped["seed_count"])
    grouped["safety_sem"] = grouped["safety_sd"] / np.sqrt(grouped["seed_count"])
    return grouped


def _add_environment_columns(
    frame: pd.DataFrame, environment: EnvironmentSpec
) -> pd.DataFrame:
    frame.insert(0, "environment", environment.label)
    frame.insert(0, "environment_key", environment.key)
    frame["nominal_budget_timesteps"] = environment.nominal_budget
    frame["plot_horizon_timesteps"] = environment.horizon
    return frame


def aggregate_evaluation(
    spec: MethodSpec,
    *,
    environment: EnvironmentSpec | None = None,
    baseline_root: Path | None = None,
    adaptive_root: Path | None = None,
    budget: int | None = None,
) -> pd.DataFrame:
    """Aggregate evaluation summaries; legacy root arguments aid focused reuse."""
    if environment is None:
        if baseline_root is None or adaptive_root is None or budget is None:
            raise TypeError("provide environment or baseline_root/adaptive_root/budget")
        environment = EnvironmentSpec(
            "custom", "Custom", budget, baseline_root, adaptive_root
        )
    root = source_root(spec, environment)
    frames: list[pd.DataFrame] = []
    for seed, seed_dir in discover_seeds(root):
        path = seed_dir / spec.evaluation_path
        if not path.is_file():
            continue
        frame = pd.read_csv(
            path,
            usecols=["timestep", "mean_total_reward", "safety_rate"],
        )
        frame = frame.loc[frame["timestep"] <= environment.horizon].copy()
        if frame.empty:
            continue
        frame["seed"] = seed
        frame["reward"] = frame.pop("mean_total_reward").astype(float)
        frame["safety"] = frame.pop("safety_rate").astype(float)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No evaluation curves found for {spec.label} under {root}"
        )
    aggregate = _summary_stats(pd.concat(frames, ignore_index=True), ["timestep"])
    aggregate.insert(0, "method", spec.label)
    aggregate.insert(0, "method_key", spec.key)
    aggregate["source_root"] = str(root.resolve())
    aggregate["source_rel_path"] = spec.evaluation_path
    return _add_environment_columns(aggregate, environment)


def _parse_optional_bool(series: pd.Series) -> pd.Series:
    missing = series.isna() | series.astype(str).str.strip().eq("")
    values = series.astype(str).str.strip().str.lower()
    invalid = ~missing & ~values.isin({"true", "false", "1", "0"})
    if invalid.any():
        bad = sorted(values.loc[invalid].unique())
        raise ValueError(f"Unrecognised boolean values: {bad}")
    parsed = values.isin({"true", "1"}).astype("boolean")
    parsed.loc[missing] = pd.NA
    return parsed


def aggregate_exploration(
    spec: MethodSpec,
    *,
    environment: EnvironmentSpec | None = None,
    bin_size: int,
    baseline_root: Path | None = None,
    adaptive_root: Path | None = None,
    budget: int | None = None,
) -> pd.DataFrame:
    if environment is None:
        if baseline_root is None or adaptive_root is None or budget is None:
            raise TypeError("provide environment or baseline_root/adaptive_root/budget")
        environment = EnvironmentSpec(
            "custom", "Custom", budget, baseline_root, adaptive_root
        )
    root = source_root(spec, environment)
    per_seed_bins: list[pd.DataFrame] = []
    safety_sources: set[str] = set()
    for seed, seed_dir in discover_seeds(root):
        path = seed_dir / spec.exploration_path
        if not path.is_file():
            continue
        columns = set(pd.read_csv(path, nrows=0).columns)
        safety_columns = [
            column for column in ("safe_trajectory", "violated") if column in columns
        ]
        if not safety_columns:
            raise ValueError(f"No trajectory-safety column in {path}")
        frame = pd.read_csv(
            path,
            usecols=["algorithm", "end_timestep", "reward", *safety_columns],
        )
        frame = frame.loc[
            (frame["algorithm"] == spec.algorithm_filter)
            & (frame["end_timestep"] > 0)
            & (frame["end_timestep"] <= environment.horizon)
        ].copy()
        if frame.empty:
            continue
        if "safe_trajectory" in safety_columns:
            safety = _parse_optional_bool(frame["safe_trajectory"])
            safety_sources.add("safe_trajectory")
            if safety.isna().any():
                if "violated" not in safety_columns:
                    raise ValueError(
                        f"Missing safe_trajectory values without violated fallback in {path}"
                    )
                fallback = ~_parse_optional_bool(frame["violated"])
                safety = safety.fillna(fallback)
                safety_sources.add("not_violated_fallback")
        else:
            safety = ~_parse_optional_bool(frame["violated"])
            safety_sources.add("not_violated_fallback")
        if safety.isna().any():
            raise ValueError(f"Missing trajectory safety after fallback in {path}")
        frame["safety"] = safety.astype(float)
        frame["bin_index"] = (
            (frame["end_timestep"].astype(int) - 1) // bin_size
        ).astype(int)
        seed_bins = (
            frame.groupby("bin_index", sort=True, observed=True)
            .agg(
                reward=("reward", "mean"),
                safety=("safety", "mean"),
                episodes=("reward", "size"),
            )
            .reset_index()
        )
        seed_bins["timestep"] = np.minimum(
            (seed_bins["bin_index"] + 1) * bin_size,
            environment.horizon,
        )
        seed_bins["seed"] = seed
        per_seed_bins.append(seed_bins)
    if not per_seed_bins:
        raise FileNotFoundError(
            f"No exploration episodes found for {spec.label} under {root}"
        )
    all_bins = pd.concat(per_seed_bins, ignore_index=True)
    aggregate = _summary_stats(all_bins, ["bin_index", "timestep"])
    episode_counts = (
        all_bins.groupby(["bin_index", "timestep"], sort=True)
        .agg(mean_episodes_per_seed_in_bin=("episodes", "mean"))
        .reset_index()
    )
    aggregate = aggregate.merge(
        episode_counts, on=["bin_index", "timestep"], validate="one_to_one"
    )
    aggregate.insert(0, "method", spec.label)
    aggregate.insert(0, "method_key", spec.key)
    aggregate["bin_size_timesteps"] = bin_size
    aggregate["source_root"] = str(root.resolve())
    aggregate["source_rel_path"] = spec.exploration_path
    aggregate["algorithm_filter"] = spec.algorithm_filter
    aggregate["safety_source"] = "+".join(sorted(safety_sources))
    return _add_environment_columns(aggregate, environment)


def _common_seed_support(
    aggregate: pd.DataFrame, *, expected: int, allow_incomplete: bool
) -> pd.DataFrame:
    if allow_incomplete:
        return aggregate
    complete = aggregate.loc[aggregate["seed_count"] == expected].copy()
    if complete.empty:
        method = aggregate["method_key"].iloc[0]
        environment = aggregate["environment_key"].iloc[0]
        raise RuntimeError(f"No {expected}-seed support for {environment}/{method}")
    return complete


def _format_timesteps(value: float, _position: int) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def _apply_style() -> None:
    """Paper typography, shared with the bar charts via ``_paper_style``."""
    apply_paper_style()


def _plot_metric(
    axis,
    aggregates: pd.DataFrame,
    *,
    metric: str,
    ci_multiplier: float,
) -> None:
    """Draw every method's across-seed mean and standard-error band."""
    for spec in METHODS:
        curve = aggregates.loc[aggregates["method_key"] == spec.key].sort_values(
            "timestep"
        )
        if curve.empty:
            continue
        x = curve["timestep"].to_numpy(dtype=float)
        mean = curve[f"{metric}_mean"].to_numpy(dtype=float)
        sem = curve[f"{metric}_sem"].to_numpy(dtype=float)
        axis.plot(
            x,
            mean,
            color=spec.color,
            linewidth=1.0,
            linestyle=spec.linestyle,
            label=spec.label,
        )
        band = ci_multiplier * sem
        lower, upper = mean - band, mean + band
        if metric == "safety":
            lower = np.clip(lower, 0.0, 1.0)
            upper = np.clip(upper, 0.0, 1.0)
        axis.fill_between(x, lower, upper, color=spec.color, alpha=0.13, linewidth=0)


def _style_metric_axis(
    axis, *, environment: EnvironmentSpec, metric: str, x_nbins: int
) -> None:
    """Shared axis furniture: horizon, timestep ticks, spines, safety range."""
    axis.set_xlim(0, environment.horizon)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=x_nbins, integer=True))
    axis.xaxis.set_major_formatter(FuncFormatter(_format_timesteps))
    axis.spines[["top", "right"]].set_visible(False)
    if metric == "safety":
        # A rate, so the scale is identical in every environment.
        axis.set_ylim(-0.025, 1.025)


def _draw_environment_column(
    axes: np.ndarray,
    aggregates: pd.DataFrame,
    *,
    environment: EnvironmentSpec,
    ci_multiplier: float,
    is_leftmost: bool,
) -> None:
    """Draw one environment as a column: reward on top, safety below.

    Environments run along the columns rather than the rows so the figure keeps
    a fixed ICML text-width and a constant height as environments are added --
    the previous rows-per-environment layout grew to 16 in tall at five
    environments and would not fit on a page at six.
    """
    # Short y-labels: at ~1.1 in of panel height the full phrasings
    # ("Mean episodic reward" / "Safe-trajectory rate") overrun their axis and
    # collide with each other. The caption carries the precise definition.
    for axis, metric, ylabel in (
        (axes[0], "reward", "Reward"),
        (axes[1], "safety", "Safety rate"),
    ):
        _plot_metric(axis, aggregates, metric=metric, ci_multiplier=ci_multiplier)
        # Reward is on a different scale in every environment (Colour Bomb v2
        # reaches ~35, Media Streaming is negative, MiniPacman is within [0,1]),
        # so each reward panel keeps its own axis and its own tick labels.
        # Safety is a rate in [0,1] everywhere, so those panels share one scale
        # and only the leftmost prints the ticks.
        if is_leftmost:
            axis.set_ylabel(ylabel)
        if metric == "reward":
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        elif not is_leftmost:
            axis.tick_params(labelleft=False)
        _style_metric_axis(axis, environment=environment, metric=metric, x_nbins=3)
    # Wrapped so long names ("Bridge Crossing v2") fit a ~1 in wide column
    # instead of running into the neighbouring panel's title.
    axes[0].set_title(
        textwrap.fill(environment.label, width=11),
        fontsize=7,
        fontweight="bold",
        pad=3,
        linespacing=0.95,
    )


def _draw_environment_row(
    axes: np.ndarray,
    aggregates: pd.DataFrame,
    *,
    environment: EnvironmentSpec,
    ci_multiplier: float,
    is_top: bool,
) -> None:
    """Draw one environment as a row: reward left, safety right.

    The transpose of ``_draw_environment_column``. Each panel now gets half the
    text width (~2.2 in) rather than a sixth of it, which is what makes the
    long horizons legible -- MiniPacman's 2M steps are no longer squeezed into
    a ~0.8 in column.
    """
    for axis, metric in ((axes[0], "reward"), (axes[1], "safety")):
        _plot_metric(axis, aggregates, metric=metric, ci_multiplier=ci_multiplier)
        _style_metric_axis(axis, environment=environment, metric=metric, x_nbins=4)
        if metric == "reward":
            # Per-environment reward scale, so every panel prints its own ticks.
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
        else:
            # Endpoints plus the midpoint are enough to read a rate and leave
            # the ~1 in panel uncluttered.
            axis.set_yticks((0.0, 0.5, 1.0))
    # Every row has its own horizon, so x tick labels cannot be deferred to the
    # bottom row the way a shared-x column layout would.
    #
    # The environment name is the row header, carried by the left panel's
    # y-label; "Reward"/"Safety rate" become column titles on the top row.
    axes[0].set_ylabel(
        textwrap.fill(environment.label, width=13),
        fontsize=6.8,
        fontweight="bold",
        linespacing=0.95,
    )
    if is_top:
        for axis, title in ((axes[0], "Reward"), (axes[1], "Safety rate")):
            axis.set_title(title, fontsize=7.5, fontweight="bold", pad=4)


def _add_shared_furniture(fig, axes: np.ndarray) -> None:
    """One x-label for the whole figure plus a single deduplicated legend.

    The legend goes above the panels: constrained_layout pins both supxlabel
    and an "outside lower center" legend to the figure bottom and does not
    deconflict them, so they overlap at any figure height.
    """
    fig.supxlabel("Training timestep", fontsize=7.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(labels),
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
    )


def save_figure(
    aggregates: pd.DataFrame,
    *,
    kind: str,
    environments: tuple[EnvironmentSpec, ...],
    output_dir: Path,
    ci_multiplier: float,
    layout: str = METRIC_ROWS,
) -> None:
    if layout == METRIC_ROWS:
        _save_metric_rows(
            aggregates,
            kind=kind,
            environments=environments,
            output_dir=output_dir,
            ci_multiplier=ci_multiplier,
        )
    elif layout == ENVIRONMENT_ROWS:
        _save_environment_rows(
            aggregates,
            kind=kind,
            environments=environments,
            output_dir=output_dir,
            ci_multiplier=ci_multiplier,
        )
    else:  # pragma: no cover - argparse constrains the choices
        raise ValueError(f"unknown layout: {layout}")


def _save_metric_rows(
    aggregates: pd.DataFrame,
    *,
    kind: str,
    environments: tuple[EnvironmentSpec, ...],
    output_dir: Path,
    ci_multiplier: float,
) -> None:
    # 2 rows (reward, safety) x N columns (environments), at exactly ICLR
    # \textwidth so the PDF can be included at native size.
    #
    # sharey is off: reward has a different scale per environment, so sharing
    # would flatten every panel onto Colour Bomb v2's ~35-unit range. The
    # safety row is equalised explicitly instead (fixed [0,1] limits, ticks
    # printed only on the leftmost panel).
    fig, axes = plt.subplots(
        2,
        len(environments),
        figsize=(TEXT_WIDTH_IN, 2.75),
        facecolor=SURFACE,
        squeeze=False,
        sharex="col",
        sharey=False,
        layout="constrained",
    )
    for column, environment in enumerate(environments):
        subset = aggregates.loc[aggregates["environment_key"] == environment.key]
        _draw_environment_column(
            axes[:, column],
            subset,
            environment=environment,
            ci_multiplier=ci_multiplier,
            is_leftmost=column == 0,
        )
    _add_shared_furniture(fig, axes)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.04)
    stem = output_dir / f"{kind}_time_reward_safety_learning_curves"
    save_paper_figure(fig, stem)
    plt.close(fig)


def _save_environment_rows(
    aggregates: pd.DataFrame,
    *,
    kind: str,
    environments: tuple[EnvironmentSpec, ...],
    output_dir: Path,
    ci_multiplier: float,
) -> None:
    # N rows (environments) x 2 columns (reward, safety) -- the transpose of
    # _save_metric_rows, at the same ICLR \textwidth so it still drops in with
    # \includegraphics at native scale.
    #
    # Height grows with the environment count instead of width, at ~1.05 in per
    # row. Six environments come to 7.2 in, inside ICLR's 9.0 in \textheight
    # with room left for a caption; beyond about seven environments this layout
    # stops fitting on a page and the metric-rows one should be preferred.
    #
    # sharex is off even though the two panels in a row share a horizon: with
    # sharex="row" matplotlib hides the x tick labels of every row but the
    # last, and here each row has its own horizon, so those labels have to
    # stay. _style_metric_axis sets identical limits on both panels instead.
    fig, axes = plt.subplots(
        len(environments),
        2,
        figsize=(TEXT_WIDTH_IN, 1.05 * len(environments) + 0.9),
        facecolor=SURFACE,
        squeeze=False,
        sharex=False,
        sharey=False,
        layout="constrained",
    )
    for row, environment in enumerate(environments):
        subset = aggregates.loc[aggregates["environment_key"] == environment.key]
        _draw_environment_row(
            axes[row, :],
            subset,
            environment=environment,
            ci_multiplier=ci_multiplier,
            is_top=row == 0,
        )
    _add_shared_furniture(fig, axes)
    # Reward tick labels differ in width per row ("-20" vs "30" vs "1.0"), which
    # would otherwise leave the environment names in a ragged left edge.
    fig.align_ylabels(axes[:, 0])
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06, hspace=0.12)
    stem = output_dir / f"{kind}_time_reward_safety_learning_curves_by_environment"
    save_paper_figure(fig, stem)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    environments = selected_environments(args)
    # dict.fromkeys so "--layout X --layout X" writes each file once.
    layouts = tuple(dict.fromkeys(args.layout or (METRIC_ROWS,)))
    if args.bin_size <= 0:
        raise SystemExit("--bin-size must be positive")
    if args.ci_multiplier < 0:
        raise SystemExit("--ci-multiplier must be non-negative")
    if any(environment.nominal_budget <= 0 for environment in environments):
        raise SystemExit("--budget must be positive")

    if not args.allow_incomplete:
        for environment in environments:
            validate_sources(environment)

    evaluation_parts: list[pd.DataFrame] = []
    exploration_parts: list[pd.DataFrame] = []
    expected = len(EXPECTED_SEEDS)
    for environment in environments:
        for spec in METHODS:
            evaluation_parts.append(
                _common_seed_support(
                    aggregate_evaluation(spec, environment=environment),
                    expected=expected,
                    allow_incomplete=args.allow_incomplete,
                )
            )
            exploration_parts.append(
                _common_seed_support(
                    aggregate_exploration(
                        spec, environment=environment, bin_size=args.bin_size
                    ),
                    expected=expected,
                    allow_incomplete=args.allow_incomplete,
                )
            )
    evaluation = pd.concat(evaluation_parts, ignore_index=True)
    exploration = pd.concat(exploration_parts, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(args.output_dir / "evaluation_curve_aggregates.csv", index=False)
    exploration.to_csv(args.output_dir / "exploration_curve_aggregates.csv", index=False)

    metadata = {
        "environments": {
            environment.key: {
                "label": environment.label,
                "nominal_budget_timesteps": environment.nominal_budget,
                "plot_horizon_timesteps": environment.horizon,
                "baseline_root": str(environment.baseline_root.resolve()),
                "pspo_root": str(environment.adaptive_root.resolve()),
            }
            for environment in environments
        },
        "expected_seeds": list(EXPECTED_SEEDS),
        "exploration_bin_size_timesteps": args.bin_size,
        "standard_error_multiplier": args.ci_multiplier,
        "evaluation_definition": "Periodic deterministic policy evaluation.",
        "exploration_definition": (
            "Per-seed means of completed rollout episodes binned by end timestep."
        ),
        "exploration_safety_definition": (
            "safe_trajectory when recorded; otherwise logical negation of violated."
        ),
        "common_support_note": (
            "Unless --allow-incomplete is used, plotted points contain all ten seeds."
        ),
        "methods": {
            spec.key: {
                "label": spec.label,
                "evaluation_variant": (
                    "shielded/deployed"
                    if spec.key == "ppo_shield"
                    else "unshielded/nominal"
                ),
                "algorithm_filter": spec.algorithm_filter,
            }
            for spec in METHODS
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    _apply_style()
    for layout in layouts:
        for frame, kind in ((evaluation, "evaluation"), (exploration, "exploration")):
            save_figure(
                frame,
                kind=kind,
                environments=environments,
                output_dir=args.output_dir,
                ci_multiplier=args.ci_multiplier,
                layout=layout,
            )

    print(f"Saved learning curves to {args.output_dir} (layouts: {', '.join(layouts)})")
    for environment in environments:
        print(f"  {environment.label}: seeds 0-9, horizon {environment.horizon:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
