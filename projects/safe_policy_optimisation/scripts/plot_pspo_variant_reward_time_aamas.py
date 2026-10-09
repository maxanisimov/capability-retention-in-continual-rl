#!/usr/bin/env python
"""AAMAS-size comparison of canonical PSPO, verify-first, and segment search.

Use seed-level reward means and launcher-measured RL-stage elapsed seconds for
all three variants. The latter excludes base-policy pretraining but includes
process setup, in-training evaluations, final evaluation, and artifact saving.
Do not mix this quantity with summary.json's narrower training_wall_time_s.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

REPO = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = (
    REPO / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
)
DEFAULT_OUTPUT = (
    REPO / "projects/safe_policy_optimisation/results/pspo/variant_reward_time_aamas"
)
SEEDS = tuple(range(10))
ENVIRONMENTS = (
    ("media_streaming", "Media\nStreaming"),
    ("colour_bomb", "Colour Bomb\nv1"),
    ("colour_bomb_v2", "Colour Bomb\nv2"),
    ("bridge_crossing", "Bridge Crossing\nv1"),
    ("bridge_crossing_v2", "Bridge Crossing\nv2"),
    ("mini_pacman", "MiniPacman"),
)
# The same canonical controls designated by run_pspo_four_ablations.py.
CONTROLS = {
    "bridge_crossing": "pspo_adaptive_bridge_v1_safe_entropy_w1_min0p95_freq1",
    "bridge_crossing_v2": "pspo_adaptive_bridge_v2_safe_entropy_w1_min0p95_freq1_t1600k_reuse_base",
    "colour_bomb": "pspo_adaptive_cb_media_safe_entropy_w1_min0p95_freq1_default",
    "colour_bomb_v2": "pspo_adaptive_cb_media_safe_entropy_w1_min0p95_freq1_default",
    "media_streaming": "pspo_adaptive_cb_media_safe_entropy_w1_min0p95_freq1_default",
    "mini_pacman": "pspo_adaptive_mini_pacman_safe_entropy_w1_min0p95_freq100_t2000k_reuse_base",
}
VARIANTS = (
    ("default", "PSPO orthotope (default)", "#009E73", ""),
    ("verify_first", "PSPO orthotope verify-first", "#E69F00", ""),
    ("segment", "PSPO line segment", "#0072B2", ""),
)
# Sky blue is the light partner of the line-segment blue; it keeps every
# pair CVD-separable (worst protan/deutan dE 11.4 against the other three).
SEGMENT_VERIFY_FIRST = ("segment_verify_first", "PSPO line segment verify-first", "#56B4E9", "")
# --all-variants: region-first orthotope and segment, then their verify-first
# versions, in the order of generate_pspo_variant_tables.py's columns.
ALL_VARIANTS = (VARIANTS[0], VARIANTS[2], VARIANTS[1], SEGMENT_VERIFY_FIRST)
VERIFY_FIRST_VARIANTS = frozenset({"verify_first", "segment_verify_first"})
SEGMENT_VARIANTS = frozenset({"segment", "segment_verify_first"})
ELAPSED_PATTERN = re.compile(r"^ok seed(\d+) core=\d+ (\d+)s\s*$", re.MULTILINE)
WIDTH_IN, HEIGHT_IN = 7.0, 2.10
COLUMN_WIDTH_IN, COLUMN_HEIGHT_IN = 3.33, 3.95
METRICS = (
    ("reward_mean", "reward_two_se"),
    ("rl_stage_minutes_mean", "rl_stage_minutes_two_se"),
)


def mean_two_se(values: list[float]) -> tuple[float, float]:
    """Unbiased sample SD across seeds, not across evaluation episodes."""
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("Need at least two finite seed-level observations")
    return statistics.mean(values), 2 * statistics.stdev(values) / math.sqrt(
        len(values)
    )


def read_elapsed_logs(paths: list[Path]) -> dict[int, float]:
    """Explicit log selection avoids confusing v1 with v2 via substring matches."""
    elapsed: dict[int, float] = {}
    for path in paths:
        for seed, seconds in ELAPSED_PATTERN.findall(path.read_text()):
            key = int(seed)
            if key in elapsed:
                raise ValueError(
                    f"Ambiguous repeated completion for seed{seed}: {path}"
                )
            elapsed[key] = float(seconds)
    if set(elapsed) != set(SEEDS):
        raise ValueError(f"Expected seeds 0--9 in {paths}, found {sorted(elapsed)}")
    return elapsed


def cohort_name(variant: str, environment: str) -> str:
    return (
        CONTROLS[environment]
        if variant == "default"
        else {
            "verify_first": "pspo_verifyfirst_masa",
            "segment": "segment_lid",
            "segment_verify_first": "pspo_verifyfirst_segment_masa_20261007T131200Z",
        }[variant]
    )


def load_data(runs: Path, variants=VARIANTS) -> tuple[list[dict], list[dict]]:
    rows, seed_rows = [], []
    for environment, _ in ENVIRONMENTS:
        reference: dict | None = None
        for variant, label, _, _ in variants:
            cohort = runs / cohort_name(variant, environment)
            log_root = cohort / "_orchestrator"
            if variant == "default" and environment == "bridge_crossing_v2":
                logs = [
                    log_root / "extend_remaining_20260826.log",
                    log_root / "screen.log",
                ]
            elif variant == "default" and environment == "mini_pacman":
                logs = [log_root / "screen.log"]
            else:
                logs = [log_root / f"{environment}.log"]
            elapsed = read_elapsed_logs(logs)
            rewards, times, safety = [], [], []
            for seed in SEEDS:
                directory = cohort / "two_hidden" / environment / f"seed{seed}"
                metrics = json.loads((directory / "metrics.json").read_text())
                config = json.loads((directory / "config.json").read_text())
                summary = json.loads((directory / "summary.json").read_text())
                if (
                    config["evaluation_policy"] != "unshielded"
                    or metrics["eval_episodes"] != 100
                ):
                    raise ValueError(f"Unexpected evaluation protocol: {directory}")
                if summary["early_stop_triggered"]:
                    raise ValueError(f"Unexpected early stopping: {directory}")
                adaptive = config["adaptive"]
                if adaptive["verify_first"] != (variant in VERIFY_FIRST_VARIANTS):
                    raise ValueError(f"Unexpected verify_first setting: {directory}")
                if adaptive["safe_region_shape"] != (
                    "segment" if variant in SEGMENT_VARIANTS else "orthotope"
                ):
                    raise ValueError(f"Unexpected region shape: {directory}")
                signature = {
                    key: config[key]
                    for key in (
                        "env_id",
                        "env_kwargs",
                        "max_episode_steps",
                        "total_timesteps",
                        "training_hyperparameters",
                        "evaluation_policy",
                        "eval_episodes",
                        "base_policy_architecture",
                        "curve_eval_freq",
                        "curve_eval_episodes",
                        "early_stop_eval_freq",
                        "success_reward_threshold",
                    )
                }
                if reference is None:
                    reference = signature
                elif reference != signature:
                    raise ValueError(
                        f"Unmatched training/evaluation settings: {directory}"
                    )
                reward = float(metrics["reward"]["mean_total_reward"])
                minutes = elapsed[seed] / 60
                rewards.append(reward)
                times.append(minutes)
                safety.append(float(metrics["safety"]["safety_rate"]))
                seed_rows.append(
                    {
                        "environment": environment,
                        "variant": variant,
                        "seed": seed,
                        "reward": reward,
                        "rl_stage_seconds": elapsed[seed],
                        "rl_stage_minutes": minutes,
                        "safety_rate": safety[-1],
                        "training_seconds": summary.get("timing", {}).get(
                            "training_wall_time_s"
                        ),
                        "source_directory": str(directory.relative_to(runs)),
                        "source_logs": ";".join(str(p.relative_to(runs)) for p in logs),
                    }
                )
            reward_mean, reward_error = mean_two_se(rewards)
            time_mean, time_error = mean_two_se(times)
            rows.append(
                {
                    "environment": environment,
                    "variant": variant,
                    "label": label,
                    "n_seeds": len(SEEDS),
                    "reward_mean": reward_mean,
                    "reward_two_se": reward_error,
                    "rl_stage_minutes_mean": time_mean,
                    "rl_stage_minutes_two_se": time_error,
                    "safety_rate_mean": statistics.mean(safety),
                }
            )
    return rows, seed_rows


def validated_lookup(rows: list[dict], variants=VARIANTS) -> dict[tuple[str, str], dict]:
    expected = {
        (environment, variant)
        for environment, _ in ENVIRONMENTS
        for variant, _, _, _ in variants
    }
    lookup = {}
    for row in rows:
        key = (row["environment"], row["variant"])
        if key in lookup or key not in expected:
            raise ValueError(f"Unexpected or duplicate summary row: {key}")
        if int(row["n_seeds"]) != len(SEEDS):
            raise ValueError(f"Expected ten seeds for {key}")
        for mean_key, error_key in METRICS:
            mean, error = float(row[mean_key]), float(row[error_key])
            if not math.isfinite(mean) or not math.isfinite(error) or error < 0:
                raise ValueError(f"Invalid metric or error bar for {key}")
        lookup[key] = row
    if set(lookup) != expected:
        raise ValueError(f"Missing summary rows: {sorted(expected - set(lookup))}")
    return lookup


def read_summary_csv(path: Path, variants=VARIANTS) -> list[dict]:
    """Restyle saved statistics without rereading or rewriting experiment data."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["n_seeds"] = int(row["n_seeds"])
        for key in (
            "reward_mean",
            "reward_two_se",
            "rl_stage_minutes_mean",
            "rl_stage_minutes_two_se",
            "safety_rate_mean",
        ):
            row[key] = float(row[key])
    validated_lookup(rows, variants)
    return rows


def single_column_height(variants=VARIANTS) -> float:
    """The single-column legend has one row per variant: grow the canvas by the
    extra rows so every panel keeps its three-variant size."""
    return COLUMN_HEIGHT_IN + 0.14 * max(len(variants) - 3, 0)


def full_width_legend_columns(variants=VARIANTS) -> int:
    """One legend row up to three variants; four do not fit across 7 in, so they
    form a 2x2 grid (columns fill first: region-first left, verify-first right)."""
    return len(variants) if len(variants) <= 3 else 2


def full_width_height(variants=VARIANTS) -> float:
    rows = math.ceil(len(variants) / full_width_legend_columns(variants))
    return HEIGHT_IN + 0.17 * (rows - 1)


def build_figure(rows: list[dict], *, single_column: bool = False, variants=VARIANTS):
    """Use native AAMAS widths, with metrics in rows or transposed columns."""
    lookup = validated_lookup(rows, variants)
    column_height = single_column_height(variants)
    width_height = full_width_height(variants)

    def width_fraction(fraction: float) -> float:
        return fraction * HEIGHT_IN / width_height

    def column_fraction(fraction: float) -> float:
        return fraction * COLUMN_HEIGHT_IN / column_height

    # Deliberately do not use _paper_style: it targets ICLR, not AAMAS.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "ytick.major.size": 2,
            "ytick.major.pad": 2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
        }
    )
    if single_column:
        fig, axes = plt.subplots(6, 2, figsize=(COLUMN_WIDTH_IN, column_height))
        fig.subplots_adjust(
            left=0.15, right=0.985, bottom=column_fraction(0.045),
            top=column_fraction(0.76), wspace=0.22, hspace=0.80,
        )
    else:
        fig, axes = plt.subplots(2, 6, figsize=(WIDTH_IN, width_height))
        fig.subplots_adjust(
            left=0.072, right=0.993, bottom=width_fraction(0.10),
            top=width_fraction(0.725), wspace=0.48, hspace=0.34,
        )

    for environment_index, (environment, title) in enumerate(ENVIRONMENTS):
        if not single_column:
            axes[0, environment_index].set_title(title, pad=3)
        for metric_index, (mean_key, error_key) in enumerate(METRICS):
            ax = (
                axes[environment_index, metric_index]
                if single_column
                else axes[metric_index, environment_index]
            )
            bounds = [0.0]
            for position, (variant, _, colour, _) in enumerate(variants):
                data = lookup[environment, variant]
                mean, error = data[mean_key], data[error_key]
                ax.bar(
                    position,
                    mean,
                    yerr=error,
                    width=0.69,
                    color=colour,
                    edgecolor="#252525",
                    linewidth=0.45,
                    capsize=1.6,
                    error_kw={"elinewidth": 0.65, "capthick": 0.65},
                    zorder=3,
                )
                bounds.extend((mean - error, mean + error))
            low, high = min(bounds), max(bounds)
            span = max(high - low, 0.01)
            ax.set_ylim(
                low - 0.09 * span if low < 0 else 0,
                high + 0.12 * span if high > 0 else 0,
            )
            ax.set_xlim(-0.60, len(variants) - 0.40)
            ax.set_xticks([])
            ax.yaxis.set_major_locator(
                MaxNLocator(
                    nbins=2 if single_column else 3,
                    min_n_ticks=2,
                    steps=[1, 2, 2.5, 5, 10],
                )
            )
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", linewidth=0.35, color="#D8D8D8", zorder=0)
            ax.set_axisbelow(True)

    handles = [
        Patch(facecolor=colour, edgecolor="#252525", linewidth=0.45, label=label)
        for _, label, colour, _ in variants
    ]
    if single_column:
        for index, (_, title) in enumerate(ENVIRONMENTS):
            left, right = axes[index]
            heading_y = left.get_position().y1 + 3 / (72 * column_height)
            fig.text(
                0.5,
                heading_y,
                title.replace("\n", " "),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        for axis, label in zip(axes[0], ("Total reward", "Time (min)")):
            position = axis.get_position()
            fig.text(
                (position.x0 + position.x1) / 2,
                column_fraction(0.835),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=1,
            frameon=False,
            fontsize=7.5,
            handlelength=1.3,
            handletextpad=0.5,
            labelspacing=0.15,
            borderpad=0.1,
            borderaxespad=0,
        )
    else:
        for axis, label in zip(axes[:, 0], ("Total reward", "Time (min)")):
            position = axis.get_position()
            fig.text(
                0.012,
                (position.y0 + position.y1) / 2,
                label,
                rotation=90,
                va="center",
                fontsize=8,
            )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.53, 0.995),
            ncol=full_width_legend_columns(variants),
            frameon=False,
            handlelength=1.4,
            handletextpad=0.5,
            columnspacing=1.1,
            borderpad=0.1,
            borderaxespad=0,
        )
    return fig


def plot(rows: list[dict], stem: Path, *, single_column: bool = False, variants=VARIANTS) -> None:
    fig = build_figure(rows, single_column=single_column, variants=variants)
    # No tight cropping: preserve native printed widths and readable font sizes.
    fig.savefig(
        stem.with_suffix(".pdf"),
        metadata={"Title": "PSPO variant reward and RL-stage time"},
    )
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    plt.close(fig)


def write_figure_snippet(output: Path, *, single_column: bool = False) -> None:
    environment = "figure" if single_column else "figure*"
    width = r"\columnwidth" if single_column else r"\textwidth"
    suffix = "_column" if single_column else ""
    positions = "left" if single_column else "top"
    time_positions = "right" if single_column else "bottom"
    order = "top to bottom" if single_column else "left to right"
    text = rf"""% Use either this snippet or its alternative, not both (same label).
% Official AAMAS class; adjust only the relative graphics path.
\begin{{{environment}}}[t]
  \centering
  \includegraphics[width={width}]{{pspo_variant_reward_time_aamas{suffix}.pdf}}
  \caption{{PSPO variants on six tasks: total episode reward ({positions}) and
  RL-stage wall time in minutes ({time_positions}). Bars show mean $\pm 2$
  standard errors over ten seeds; reward uses 100 unshielded greedy evaluation
  episodes per seed. Runtime includes setup, certification, training,
  evaluation and saving, but excludes initial-policy fitting. All variants
  attain $100\%$ empirical evaluation safety. Axes are task-specific, linear
  and zero-based. Initial policies and execution cohorts differ.}}
  \label{{fig:pspo-variants-reward-time}}
  \Description{{Grouped solid-colour bars compare PSPO orthotope (default),
  PSPO orthotope verify-first and PSPO line segment. Environments are Media
  Streaming, Colour Bomb v1, Colour Bomb v2, Bridge Crossing v1, Bridge Crossing
  v2 and MiniPacman, from {order}. Reward appears {positions} and runtime
  {time_positions}; error bars are two standard errors over ten seeds.
  Runtime is lower for the alternatives on the four non-bridge tasks, but
  not on the bridges. All variants have 100 percent empirical safety.
  The cohorts have different initial policies.}}
\end{{{environment}}}
"""
    (output / f"figure_aamas{suffix}.tex").write_text(text)


def write_all_variants_snippet(output: Path, *, single_column: bool = False) -> None:
    environment = "figure" if single_column else "figure*"
    width = r"\columnwidth" if single_column else r"\textwidth"
    suffix = "_column" if single_column else ""
    positions = "left" if single_column else "top"
    time_positions = "right" if single_column else "bottom"
    order = "top to bottom" if single_column else "left to right"
    text = rf"""% Use either this snippet or its alternative, not both (same label).
% Official AAMAS class; adjust only the relative graphics path.
\begin{{{environment}}}[t]
  \centering
  \includegraphics[width={width}]{{pspo_all_variants_reward_time_aamas{suffix}.pdf}}
  \caption{{Four PSPO variants on six tasks: total episode reward ({positions}) and
  RL-stage wall time in minutes ({time_positions}). Region-first variants certify a
  LID for every enforced update; verify-first variants check the proposed
  parameters exactly and certify a LID only when that check fails. Bars show
  mean $\pm 2$ standard errors over ten seeds; reward uses 100 unshielded greedy
  evaluation episodes per seed. Runtime includes setup, certification, training,
  evaluation and saving, but excludes initial-policy fitting. All variants
  attain $100\%$ empirical evaluation safety. Axes are task-specific, linear
  and zero-based. Initial policies and execution cohorts differ.}}
  \label{{fig:pspo-all-variants-reward-time}}
  \Description{{Grouped solid-colour bars compare PSPO orthotope (default), PSPO
  line segment, PSPO orthotope verify-first and PSPO line segment verify-first.
  Environments are Media Streaming, Colour Bomb v1, Colour Bomb v2, Bridge
  Crossing v1, Bridge Crossing v2 and MiniPacman, from {order}. Reward appears
  {positions} and runtime {time_positions}; error bars are two standard errors
  over ten seeds. All variants have 100 percent empirical safety. The cohorts
  have different initial policies.}}
\end{{{environment}}}
"""
    (output / f"figure_all_variants_aamas{suffix}.tex").write_text(text)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Restyle saved statistics; leave summary and seed CSVs untouched",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Add PSPO line segment verify-first; write separate all_variants files",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = ALL_VARIANTS if args.all_variants else VARIANTS
    prefix = "all_variants_" if args.all_variants else ""
    if args.summary_csv is None:
        rows, seed_rows = load_data(args.runs_root, variants)
        write_csv(args.output_dir / f"{prefix}summary.csv", rows)
        write_csv(args.output_dir / f"{prefix}seed_metrics.csv", seed_rows)
    else:
        rows = read_summary_csv(args.summary_csv, variants)
    stem = args.output_dir / (
        "pspo_all_variants_reward_time_aamas"
        if args.all_variants
        else "pspo_variant_reward_time_aamas"
    )
    snippet = write_all_variants_snippet if args.all_variants else write_figure_snippet
    for single_column in (False, True):
        output_stem = stem.with_name(stem.name + "_column") if single_column else stem
        plot(rows, output_stem, single_column=single_column, variants=variants)
        snippet(args.output_dir, single_column=single_column)
        width, height = (
            (COLUMN_WIDTH_IN, single_column_height(variants))
            if single_column
            else (WIDTH_IN, full_width_height(variants))
        )
        print(f"Saved {output_stem.with_suffix('.pdf')} ({width} x {height} in)")
        print(f"Saved {output_stem.with_suffix('.png')}; error bars = 2 SE")


if __name__ == "__main__":
    main()
