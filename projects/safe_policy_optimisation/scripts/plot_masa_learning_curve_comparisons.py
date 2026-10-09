#!/usr/bin/env python3
"""MASA evaluation/exploration curves for baselines, final projection and PSPO variants.

PSPO denotes region-first PSPO-LS in the first two comparisons. Projected
baselines have saved pre-projection training curves and final deployed-policy
evaluations, so only their terminal evaluation markers report projection.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_extended_budget_learning_curves as canon
from _aamas_reward_safety import apply_aamas_style, legend_indices
from _paper_style import METHOD_COLORS, METHOD_LINESTYLES
from generate_final_policy_table import METHODS as TERMINAL_METHODS
from plot_pspo_variant_reward_time_aamas import ALL_VARIANTS, cohort_name

RUNS = canon.RUNS
FIGURES = REPO / "projects/safe_policy_optimisation/figures"
DEFAULT_OUTPUT = FIGURES / "masa_learning_curve_comparisons"
RL_SGF_RUNS = REPO / ".claude/worktrees/rl-sgf/projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
COMPARISONS = {
    "a_pspo_vs_baselines": "(a) PSPO versus baselines",
    "b_pspo_vs_projected_baselines": "(b) PSPO versus baselines with final projection",
    "c_four_pspo_variants": "(c) Four PSPO variants",
}
PANELS = (("evaluation", "reward"), ("evaluation", "safety"),
          ("exploration", "reward"), ("exploration", "safety"))


@dataclasses.dataclass(frozen=True)
class Series:
    spec: canon.MethodSpec
    root: Path
    terminal_path: str = "metrics.json"
    terminal_section: tuple[str, ...] = ()
    exploration: bool = True
    projected: bool = False
    held_evaluation: bool = False


def make_spec(key: str, label: str, evaluation: str, exploration: str,
              algorithm: str, color: str | None = None, linestyle=None) -> canon.MethodSpec:
    return canon.MethodSpec(key, label, color or METHOD_COLORS.get(key, "brown"),
                            METHOD_LINESTYLES.get(key, (0, (7, 1, 1, 1))) if linestyle is None else linestyle,
                            evaluation, exploration, algorithm)


def pspo_series(environment, key="pspo", label="PSPO", variant="segment",
                color=None, linestyle="solid") -> Series:
    root = RUNS / cohort_name(variant, environment.key) / "two_hidden" / environment.key
    return Series(make_spec(key, label, "learning_curves/evaluation_unshielded_summary.csv",
                            "training_episodes.csv", "shielded_ppo", color, linestyle), root)


def series_for(comparison: str, environment, rl_sgf_runs: Path) -> list[Series]:
    if comparison == "c_four_pspo_variants":
        labels = {"default": "Orthotope, region-first", "segment": "Line segment, region-first",
                  "verify_first": "Orthotope, verify-first", "segment_verify_first": "Line segment, verify-first"}
        styles = {"default": (0, (4, 1.5)), "segment": "solid",
                  "verify_first": (0, (1, 1.2)), "segment_verify_first": (0, (5, 1.5, 1, 1.5))}
        return [pspo_series(environment, key, labels[key], key, color, styles[key])
                for key, _, color, _ in ALL_VARIANTS]
    if comparison == "a_pspo_vs_baselines":
        terminals = {spec.key: spec for spec in TERMINAL_METHODS}
        series = [Series(spec, environment.baseline_root,
                         terminals[spec.key].relative_path, terminals[spec.key].section)
                  for spec in canon.METHODS if spec.key not in {"pspo", "ppo_shield"}]
        sgf = make_spec("rl_sgf", "RL-SGF", "learning_curves/rl_sgf/evaluation_unshielded_summary.csv",
                        "training_episodes.csv", "rl_sgf")
        series.append(Series(sgf, rl_sgf_runs / "rl_sgf_safe_initialised" / environment.key / "rl_sgf",
                             terminal_section=("rl_sgf",), held_evaluation=True))
        shield = next(spec for spec in canon.METHODS if spec.key == "ppo_shield")
        series.append(Series(dataclasses.replace(shield, label="PPO-Shield (shield on)"),
                             environment.baseline_root, "ppo_shield/metrics.json", ("shielded",)))
        nominal = make_spec("ppo_shield_nominal", "PPO-Shield (shield off)",
                            "ppo_shield/learning_curves/evaluation_unshielded_summary.csv",
                            "ppo_shield/training_episodes.csv", "shielded_ppo", "#C9A0DC")
        series.append(Series(nominal, environment.baseline_root, "ppo_shield/metrics.json",
                             ("nominal",), exploration=False))
        return series + [pspo_series(environment)]
    specs = (
        ("ppo_policy", "PPO", "ppo", "plain_ppo", "learning_curves/evaluation_unshielded_summary.csv"),
        ("ppo_lagrangian", "PPO-Lagrangian", "ppo_lagrangian", "ppo_lagrangian", "learning_curves/ppo_lagrangian/evaluation_unshielded_summary.csv"),
        ("ppo_pid_lagrangian", "PPO-PID-Lagrangian", "ppo_pid_lagrangian", "ppo_pid_lagrangian", "learning_curves/ppo_pid_lagrangian/evaluation_unshielded_summary.csv"),
        ("cpo", "CPO", "cpo", "cpo", "learning_curves/cpo/evaluation_unshielded_summary.csv"),
        ("rl_sgf", "RL-SGF", "rl_sgf", "rl_sgf", "learning_curves/rl_sgf/evaluation_unshielded_summary.csv"),
        ("ppo_shield_nominal", "PPO-Shield (shield off)", "ppo_shield", "shielded_ppo", "learning_curves/evaluation_unshielded_summary.csv"),
    )
    series = []
    for key, label, directory, algorithm, evaluation in specs:
        root = (rl_sgf_runs / "rl_sgf_safe_initialised_projection" if key == "rl_sgf"
                else RUNS / "safe_initialised_baseline_projection") / environment.key / directory
        spec = make_spec(key, label + " + final projection", evaluation, "training_episodes.csv", algorithm,
                         "#C9A0DC" if key == "ppo_shield_nominal" else None)
        series.append(Series(spec, root, "postprocess_metrics.json", projected=True,
                             held_evaluation=key == "rl_sgf"))
    return series + [pspo_series(environment)]


def aggregate_held_evaluation(spec, environment) -> pd.DataFrame:
    """Align RL-SGF's irregular update boundaries by past observations only."""
    grid = np.arange(canon.ROLLOUT_SIZE, environment.horizon + 1, canon.ROLLOUT_SIZE)
    parts = []
    for seed, directory in canon.discover_seeds(environment.baseline_root):
        raw = pd.read_csv(directory / spec.evaluation_path,
                          usecols=["timestep", "mean_total_reward", "safety_rate"]).sort_values("timestep")
        if raw.empty:
            raise ValueError(f"Empty evaluation log: {directory}")
        times = raw.timestep.to_numpy(dtype=np.int64)
        indices = np.searchsorted(times, grid, side="right") - 1
        valid = indices >= 0
        if times[-1] < environment.nominal_budget:
            valid &= grid <= times[-1]
        held = raw.iloc[indices[valid]]
        parts.append(pd.DataFrame({"timestep": grid[valid], "seed": seed,
                                   "reward": held.mean_total_reward.to_numpy(dtype=float),
                                   "safety": held.safety_rate.to_numpy(dtype=float),
                                   "lag": grid[valid] - held.timestep.to_numpy(dtype=np.int64)}))
    per_seed = pd.concat(parts, ignore_index=True)
    frame = canon._summary_stats(per_seed, ["timestep"])
    frame = frame.merge(per_seed.groupby("timestep").lag.max().rename("max_hold_lag_timesteps"),
                        on="timestep", validate="one_to_one")
    frame.insert(0, "method", spec.label)
    frame.insert(0, "method_key", spec.key)
    frame["source_root"] = str(environment.baseline_root.resolve())
    frame["source_rel_path"] = spec.evaluation_path
    return canon._add_environment_columns(frame, environment)


class Collector:
    def __init__(self):
        self.cache = {}
        self.sources = {}
        self.bin_sizes = {}

    def record(self, path: Path) -> None:
        path = path.resolve()
        if str(path) not in self.sources:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            self.sources[str(path)] = {"path": str(path), "sha256": digest.hexdigest(),
                                       "bytes": path.stat().st_size}

    def collect(self, series: Series, environment, kind: str) -> pd.DataFrame:
        spec = series.spec
        # At most ~80 exploration bins; bin edges remain on rollout boundaries.
        bin_size = max(canon.ROLLOUT_SIZE, math.ceil(environment.horizon / 80 / canon.ROLLOUT_SIZE) * canon.ROLLOUT_SIZE)
        self.bin_sizes[environment.key] = bin_size
        key = (str(series.root), spec.evaluation_path, spec.exploration_path,
               spec.algorithm_filter, series.held_evaluation, kind, bin_size)
        if key not in self.cache:
            seeds = canon.discover_seeds(series.root)
            if [seed for seed, _ in seeds] != list(canon.EXPECTED_SEEDS):
                raise ValueError(f"Expected exactly seeds 0--9: {series.root}")
            for seed, directory in seeds:
                path = directory / (spec.evaluation_path if kind == "evaluation" else spec.exploration_path)
                self.record(path)
                if kind == "evaluation":
                    times = pd.read_csv(path, usecols=["timestep"]).timestep
                    required = environment.nominal_budget if series.held_evaluation else environment.horizon
                    if times.empty or times.max() < required:
                        raise ValueError(f"Incomplete evaluation horizon: {path}")
                if spec.algorithm_filter == "shielded_ppo" and "two_hidden" in str(series.root):
                    config_path = directory / "config.json"
                    config = json.loads(config_path.read_text())
                    self.record(config_path)
                    if config["total_timesteps"] != environment.nominal_budget:
                        raise ValueError(f"Unmatched training budget: {config_path}")
            source = dataclasses.replace(environment, baseline_root=series.root, adaptive_root=series.root)
            if kind == "evaluation":
                frame = (aggregate_held_evaluation(spec, source) if series.held_evaluation
                         else canon.aggregate_evaluation(spec, environment=source))
            else:
                frame = canon.aggregate_exploration(spec, environment=source, bin_size=bin_size)
            frame["plotted"] = frame.seed_count.eq(10)
            if not frame.plotted.any():
                raise ValueError(f"No ten-seed support: {series.root}/{kind}")
            if not np.isfinite(frame[["reward_mean", "reward_sem", "safety_mean", "safety_sem"]]).all().all():
                raise ValueError(f"Nonfinite curve statistics: {series.root}")
            self.cache[key] = frame
        frame = self.cache[key].copy()
        frame["method_key"], frame["method"] = spec.key, spec.label
        return frame

    def terminal(self, series: Series, environment) -> dict:
        rewards, safety = [], []
        for seed in canon.EXPECTED_SEEDS:
            path = series.root / f"seed{seed}" / series.terminal_path
            self.record(path)
            node = json.loads(path.read_text())
            if series.projected:
                if node["status"] != "complete" or node["smoke"] or node["episodes"] != 100:
                    raise ValueError(f"Incomplete projected final evaluation: {path}")
                if node["evaluation_policy"] != "nominal_unshielded_deterministic":
                    raise ValueError(f"Unexpected projected evaluation policy: {path}")
                rewards.append(float(node["deployed"]["mean_total_reward"]))
                safety.append(float(node["deployed"]["safe_trajectory_rate"]))
            else:
                for section in series.terminal_section:
                    node = node[section]
                if node["eval_episodes"] != 100:
                    raise ValueError(f"Unexpected final evaluation budget: {path}")
                rewards.append(float(node["reward"]["mean_total_reward"]))
                safety.append(float(node["safety"]["safety_rate"]))
        row = {"environment_key": environment.key, "environment": environment.label,
               "method_key": series.spec.key, "method": series.spec.label,
               "timestep": environment.horizon, "seed_count": 10,
               "projected": series.projected, "episodes_per_seed": 100}
        for metric, values in (("reward", rewards), ("safety", safety)):
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_sem"] = float(np.std(values, ddof=1) / math.sqrt(10))
        return row


def draw_figure(comparison, frames, terminals, by_environment, *, kinds=("evaluation", "exploration")):
    apply_aamas_style()
    plt.rcParams.update({"xtick.labelsize": 6, "ytick.labelsize": 6})
    panels = [panel for panel in PANELS if panel[0] in kinds]
    fig, axes = plt.subplots(6, len(panels), figsize=(7.0, 8.0 if len(panels) == 4 else 6.3), squeeze=False)
    height = fig.get_size_inches()[1]
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.14 if comparison != "c_four_pspo_variants" else 0.11,
                        top=0.86, wspace=0.34, hspace=0.74)
    fig.suptitle(COMPARISONS[comparison], x=0.53, y=0.985, fontsize=10, fontweight="bold")
    for column, (kind, metric) in enumerate(panels):
        bounds = axes[0, column].get_position()
        fig.text((bounds.x0 + bounds.x1) / 2, 0.935,
                 ("Evaluation" if kind == "evaluation" else "Exploration") + " time\n" +
                 ("Total reward" if metric == "reward" else "Safety rate"),
                 ha="center", va="center", fontsize=8, linespacing=1.3)
    for index, environment in enumerate(canon.ENVIRONMENTS):
        specs = by_environment[environment.key]
        bounds = axes[index, 0].get_position()
        fig.text(0.53, bounds.y1 + 4 / (72 * height), environment.label,
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
        # Solid highlighted curves go down first, so coincident dashed curves remain visible.
        order = sorted(specs, key=lambda series: series.spec.linestyle != "solid")
        for column, (kind, metric) in enumerate(panels):
            axis = axes[index, column]
            frame = frames[kind]
            for series in order:
                spec = series.spec
                if kind == "exploration" and not series.exploration:
                    continue
                curve = frame.loc[(frame.environment_key == environment.key) & (frame.method_key == spec.key)].sort_values("timestep").copy()
                # Retain gaps when a bin/checkpoint does not contain all ten seeds.
                curve.loc[~curve.plotted, [metric + "_mean", metric + "_sem"]] = np.nan
                x = curve.timestep.to_numpy(dtype=float)
                y = curve[metric + "_mean"].to_numpy(dtype=float)
                band = 2 * curve[metric + "_sem"].to_numpy(dtype=float)
                axis.plot(x, y, color=spec.color, linestyle=spec.linestyle, linewidth=0.9)
                lower, upper = y - band, y + band
                if metric == "safety":
                    lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
                axis.fill_between(x, lower, upper, color=spec.color, alpha=0.10, linewidth=0)
                if kind == "evaluation":
                    end = terminals.loc[(terminals.environment_key == environment.key) & (terminals.method_key == spec.key)].iloc[0]
                    axis.errorbar([end.timestep], [end[metric + "_mean"]],
                                  yerr=[2 * end[metric + "_sem"]], fmt="D" if series.projected else "o",
                                  color=spec.color, markeredgecolor="#252525", markeredgewidth=0.3,
                                  markersize=2.5, capsize=1, elinewidth=0.5, zorder=5)
            axis.set_xlim(0, environment.horizon * 1.045)
            axis.set_xticks([0, environment.nominal_budget / 2, environment.nominal_budget])
            axis.xaxis.set_major_formatter(FuncFormatter(canon._format_timesteps))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))
            axis.grid(axis="both", color="#DADADA", linewidth=0.35)
            if metric == "safety":
                axis.set_ylim(-0.03, 1.05)
                axis.set_yticks([0, 1])
            if index == 5:
                axis.set_xlabel("Environment steps", fontsize=7)
    specs = by_environment[canon.ENVIRONMENTS[0].key]
    handles = [Line2D([], [], color=s.spec.color, linestyle=s.spec.linestyle, linewidth=1.0) for s in specs]
    labels = [s.spec.label for s in specs]
    columns = 3 if comparison == "a_pspo_vs_baselines" else 2
    indices = legend_indices(len(handles), columns)
    fig.legend([handles[i] for i in indices], [labels[i] for i in indices],
               loc="lower center", bbox_to_anchor=(0.52, 0.008), ncol=columns,
               fontsize=6.5, frameon=False, handlelength=2.1, columnspacing=1.0, labelspacing=0.4)
    return fig


def write_snippet(path, comparison, kinds):
    projected_note = (
        " Baseline curves precede projection." +
        (" Evaluation diamonds show the final deployed-policy evaluations after conditional projection."
         if "evaluation" in kinds else "") +
        " Projection was not used during exploration. The PSPO-LS initial actor "
        "differs from the projected baselines' fixed-LID reference actor."
        if comparison == "b_pspo_vs_projected_baselines" else ""
    )
    variant_note = (" PSPO denotes region-first PSPO-LS." if comparison != "c_four_pspo_variants"
                    else " The variants are orthotope and line-segment LIDs, each region-first or verify-first.")
    measurement_note = ""
    if "evaluation" in kinds:
        measurement_note += ("Periodic evaluation uses 20 greedy episodes per checkpoint; isolated terminal "
                             "markers use 100 episodes per seed. ")
    if "exploration" in kinds:
        measurement_note += ("Exploration is measured from completed training episodes, averaged within "
                             "timestep bins per seed before averaging seeds. ")
    caption = (COMPARISONS[comparison] + ": " + " and ".join(kinds) +
               " total reward and safe-trajectory rate for the six MASA environments. "
               "Lines show ten-seed means; bands show two standard errors. "
               + measurement_note + "Only points with all ten seeds are drawn." + variant_note + projected_note)
    if "evaluation" in kinds:
        caption += (" Periodic PSPO evaluations can occur before safety enforcement, notably with "
                    "MiniPacman's 100-rollout enforcement interval; the certificate applies to "
                    "enforced checkpoints.")
    if comparison != "c_four_pspo_variants" and "evaluation" in kinds:
        caption += " RL-SGF evaluations are aligned using the most recent past checkpoint."
    if comparison == "a_pspo_vs_baselines":
        caption += " PPO-Shield is shown with and without its shield at evaluation; only its actual shielded exploration trace exists."
    description_kinds = " and ".join("policy evaluation" if kind == "evaluation" else
                                     "training exploration" for kind in kinds)
    path.write_text("\\begin{figure*}[t]\n  \\centering\n"
                    f"  \\includegraphics[width=7in]{{{path.stem}.pdf}}\n"
                    f"  \\caption{{{caption}}}\n"
                    f"  \\label{{fig:{path.stem.replace('_', '-')}}}\n"
                    "  \\Description{Six environment rows compare total reward and safety "
                    f"during {description_kinds}. Colours and line styles "
                    "identify methods; shaded bands show two standard errors.}\n\\end{figure*}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rl-sgf-runs", type=Path, default=RL_SGF_RUNS)
    parser.add_argument("--figure-only", action="store_true",
                        help="Render from the saved aggregates and terminal evaluations.")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    collector = Collector()
    metadata = {"pspo_primary_variant": "PSPO-LS (region-first)", "expected_seeds": list(range(10)),
                "error_band": "two sample standard errors across seeds", "comparisons": {},
                "rl_sgf_alignment": "Last observation carried forward; no future observation is used. Before the first recorded evaluation the curve is absent.",
                "exploration_safety": "Recorded safe_trajectory, falling back to the negation of violated when absent.",
                "safety_band_display": "Clipped to [0,1]; unmodified SE values are retained in exported CSVs."}
    if args.figure_only:
        metadata = json.loads((args.output_dir / "provenance.json").read_text())
    metadata["projected_baseline_semantics"] = "Saved curves are pre-projection; final evaluation diamonds show deployed policies after conditional projection, retaining actors already safe. No post-projection exploration trace is claimed."
    metadata["periodic_pspo_safety_scope"] = "Periodic evaluations may precede enforcement; MiniPacman enforces every 100 rollouts. The certificate applies to enforced checkpoints, not every saved periodic evaluation."
    metadata["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with PdfPages(args.output_dir / "masa_learning_curve_comparisons.pdf") as bundle:
        for comparison in COMPARISONS:
            by_environment = {}
            parts = {"evaluation": [], "exploration": []}
            endpoints = []
            for environment in canon.ENVIRONMENTS:
                series = series_for(comparison, environment, args.rl_sgf_runs.resolve())
                by_environment[environment.key] = series
                for item in ([] if args.figure_only else series):
                    for kind in parts:
                        if kind == "exploration" and not item.exploration:
                            continue
                        parts[kind].append(collector.collect(item, environment, kind))
                    endpoints.append(collector.terminal(item, environment))
            if args.figure_only:
                frames = {kind: pd.read_csv(args.output_dir / f"{comparison}_{kind}_aggregates.csv") for kind in parts}
                terminals = pd.read_csv(args.output_dir / f"{comparison}_terminal_evaluations.csv")
            else:
                frames = {kind: pd.concat(values, ignore_index=True) for kind, values in parts.items()}
                terminals = pd.DataFrame(endpoints)
                for kind, frame in frames.items():
                    frame.to_csv(args.output_dir / f"{comparison}_{kind}_aggregates.csv", index=False)
                terminals.to_csv(args.output_dir / f"{comparison}_terminal_evaluations.csv", index=False)
            for kinds, suffix in ((("evaluation", "exploration"), ""), (("evaluation",), "_evaluation"), (("exploration",), "_exploration")):
                fig = draw_figure(comparison, frames, terminals, by_environment, kinds=kinds)
                stem = args.output_dir / (comparison + suffix)
                fig.savefig(stem.with_suffix(".pdf"))
                fig.savefig(stem.with_suffix(".png"), dpi=300)
                write_snippet(stem.with_suffix(".tex"), comparison, kinds)
                if not suffix:
                    bundle.savefig(fig)
                plt.close(fig)
            metadata["comparisons"][comparison] = {
                "sources": {env: [{"method_key": item.spec.key, "method": item.spec.label,
                                   "root": str(item.root.resolve()), "evaluation_path": item.spec.evaluation_path,
                                   "exploration_path": item.spec.exploration_path,
                                   "exploration_available": item.exploration, "final_projection": item.projected}
                                  for item in series] for env, series in by_environment.items()},
                "evaluation_rows": len(frames["evaluation"]), "exploration_rows": len(frames["exploration"]),
            }
            print(f"Saved {comparison}: {len(frames['evaluation'])} evaluation and {len(frames['exploration'])} exploration aggregate rows", flush=True)
    if not args.figure_only:
        metadata["exploration_bin_size_timesteps"] = collector.bin_sizes
        metadata["source_files"] = list(collector.sources.values())
    (args.output_dir / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# MASA learning-curve comparisons\n\n"
        "The three-page `masa_learning_curve_comparisons.pdf` contains (a) PSPO versus "
        "baselines, (b) PSPO versus safe-initialised baselines with final projection, "
        "and (c) the four PSPO variants. Each page has six environment rows and four "
        "columns: evaluation reward/safety and exploration reward/safety. Individual "
        "comparison PDFs/PNGs/LaTeX snippets and separate evaluation/exploration figures "
        "are also supplied. PSPO means region-first PSPO-LS in (a) and (b).\n\n"
        "Lines average ten seeds; bands show two between-seed standard errors. "
        "Exploration episodes are first averaged within timestep bins per seed. "
        "Bin widths follow rollout boundaries and are chosen to give approximately "
        "80 bins for long runs. Missing ten-seed support creates gaps. RL-SGF's "
        "off-grid evaluations use only the latest past observation; holding lags "
        "are exported. Periodic evaluations use 20 episodes; final evaluation "
        "markers use 100 episodes per seed. Safety rates use a common [0,1] scale.\n\n"
        "Periodic PSPO evaluations can occur before safety enforcement. In particular, "
        "MiniPacman enforces every 100 rollouts, so intermediate evaluation safety can "
        "be below the certified final checkpoint's safety. The certificate applies "
        "to enforced checkpoints.\n\n"
        "Projection in (b) happens only after training. Baseline lines are therefore "
        "pre-projection learning curves; isolated diamonds in evaluation panels show "
        "the final deployed actor after conditional projection, retaining actors already "
        "safe. No post-projection exploration curve is claimed. "
        "PSPO-LS has a different initial actor from the baselines' fixed-LID reference. "
        "In (a), PPO-Shield has two evaluation traces but only one actual exploration "
        "trace, with its shield attached.\n\n"
        "Unrounded aggregate and terminal-evaluation CSVs are supplied. "
        "`provenance.json` records all source paths/hashes, run roots, bin widths and "
        "measurement semantics. Original runs and figures are retained.\n\n"
        "Regenerate from the repository root:\n\n```bash\n"
        "MPLCONFIGDIR=/tmp/masa-learning-curves-mpl .venv/bin/python "
        "projects/safe_policy_optimisation/scripts/plot_masa_learning_curve_comparisons.py\n```\n")
    print(f"Saved all learning-curve comparisons in {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
