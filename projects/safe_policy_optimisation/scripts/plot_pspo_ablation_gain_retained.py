#!/usr/bin/env python3
"""Fraction of PSPO's reward gain over its safe initial policy kept by each ablation.

For environment e and seed i, a final reward R is scored as

    s = (R - R_init) / (R_PSPO - R_init),

with R_PSPO the mean PSPO reward over seeds and R_init the reward of the safe
initial policy that PSPO starts from, taken from the "Init only" column of
docs/pspo_ablation_reward_table.tex. 0 is no better than not training, 1 is
full PSPO, and scores are not clipped. Environments where PSPO gains nothing
over its initial policy have no defined score and are excluded.

The aggregate is the mean over all environment-seed scores with a 95%
stratified-bootstrap interval that resamples seeds within each environment
(Agarwal et al., 2021); seeds are resampled jointly across the paired variants.
The interquartile mean is deliberately not reported: with five environments it
trims about one environment's worth of seeds, hiding a loss confined to one
environment (fixed LID on MiniPacman scores IQM 1.00).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _aamas_reward_safety import COLUMN_WIDTH_IN, apply_aamas_style  # noqa: E402
from _paper_style import METHOD_COLORS  # noqa: E402
from generate_final_policy_table import ENV_LABELS  # noqa: E402
from run_pspo_four_ablations import (  # noqa: E402
    CONTROL_COHORTS,
    ENVIRONMENTS,
    OUTPUT_ROOT as ABLATION_ROOT,
    RUNS,
)

ARCHITECTURE = "two_hidden"
SEEDS = tuple(range(10))
FIXED_LID_ROOT = RUNS / "static_lid_ablation_masa_matched"
DEFAULT_OUTPUT_DIR = REPO / "projects/safe_policy_optimisation/figures/aamas"
DEFAULT_NAME = "pspo_ablation_gain_retained"
# Below this |R_PSPO - R_init| the score is undefined (0/0) or noise-amplified.
MIN_HEADROOM = 1e-6
ABLATION_COLOR = "#9a9a9a"

# Safe initial policy reward before any RL: the "Init only" column of
# docs/pspo_ablation_reward_table.tex (100 unshielded greedy episodes).
INIT_REWARDS = {
    "media_streaming": -22.82,
    "colour_bomb": 1.00,
    "colour_bomb_v2": 0.00,
    "bridge_crossing": 0.00,
    "bridge_crossing_v2": 0.00,
    "mini_pacman": 0.23,
}


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    group: str  # empty for PSPO itself


VARIANTS = (
    Variant("pspo", "PSPO", ""),
    Variant("ce_only", "CE-only", "(a) Initialisation"),
    Variant("no_entropy", "w/o entropy", "(a) Initialisation"),
    Variant("fixed_lid", "fixed LID", "(b) Safe update"),
    Variant("no_gradient", "w/o dir. growth", "(b) Safe update"),
)


def seed_dir(variant: str, environment: str, seed: int) -> Path:
    if variant == "pspo":
        return RUNS / CONTROL_COHORTS[environment] / ARCHITECTURE / environment / f"seed{seed}"
    if variant == "fixed_lid":
        return FIXED_LID_ROOT / environment / "runs" / f"seed{seed}"
    return ABLATION_ROOT / variant / ARCHITECTURE / environment / f"seed{seed}"


# PSPO-LS (line-segment LIDs) and its initialisation ablations, from
# scripts/run_pspo_segment_init_ablations.py. The control key stays "pspo".
SEGMENT_CONTROL = RUNS / "segment_lid"
SEGMENT_ABLATION_ROOT = REPO / "artifacts/ablation_studies/pspo_segment_init_ablations"
SEGMENT_VARIANTS = (
    Variant("pspo", "PSPO-LS", ""),
    Variant("no_entropy", "w/o entropy", "Initialisation"),
    Variant("no_margin", "w/o margin", "Initialisation"),
)


def segment_seed_dir(variant: str, environment: str, seed: int) -> Path:
    if variant == "pspo":
        return SEGMENT_CONTROL / ARCHITECTURE / environment / f"seed{seed}"
    return SEGMENT_ABLATION_ROOT / variant / ARCHITECTURE / environment / f"seed{seed}"


@dataclass(frozen=True)
class Family:
    method: str  # name of the full method in captions and labels
    symbol: str  # subscript of its mean reward, \bar R_{symbol}
    variants: tuple[Variant, ...]
    seed_dir: Callable[[str, str, int], Path]
    name: str  # default output stem
    label: str  # LaTeX label stem
    chart_method: str = "PSPO"  # name of the full method on the bar chart


FAMILIES = {
    "orthotope": Family("PSPO", "\\mathrm{PSPO}", VARIANTS, seed_dir, DEFAULT_NAME,
                        "pspo-ablation-gain-retained"),
    "segment": Family("PSPO-LS", "\\mathrm{LS}", SEGMENT_VARIANTS, segment_seed_dir,
                      "pspo_ls_ablation_gain_retained", "pspo-ls-ablation-gain-retained"),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def final_reward(path: Path) -> float:
    metrics = read_json(path / "metrics.json")
    if int(metrics["eval_episodes"]) != 100:
        raise ValueError(f"Expected 100 evaluation episodes: {path}")
    return float(metrics["reward"]["mean_total_reward"])


def gain_retained(family: Family = FAMILIES["orthotope"]):
    """Return per-seed scores {variant: {env: array}}, anchors and excluded envs."""
    rewards = {variant.key: {environment: np.array([final_reward(family.seed_dir(variant.key, environment, seed))
                                                    for seed in SEEDS])
                             for environment in ENVIRONMENTS}
               for variant in family.variants}
    anchors, excluded = {}, []
    for environment in ENVIRONMENTS:
        r_init = INIT_REWARDS[environment]
        r_pspo = float(rewards["pspo"][environment].mean())
        anchors[environment] = (r_init, r_pspo)
        if abs(r_pspo - r_init) <= MIN_HEADROOM:
            excluded.append(environment)
    included = [environment for environment in ENVIRONMENTS if environment not in excluded]
    scores = {variant.key: {environment: (rewards[variant.key][environment] - anchors[environment][0])
                            / (anchors[environment][1] - anchors[environment][0])
                            for environment in included}
              for variant in family.variants}
    return scores, rewards, anchors, included, excluded


def aggregate(scores, included, *, reps: int, rng_seed: int, variants=VARIANTS):
    """Mean with a 95% stratified-bootstrap interval and a stratified standard error.

    With equal seeds per environment the pooled mean is the average of the
    environment means, so its standard error is sqrt(sum_e s_e^2 / n) / E;
    variation between environments does not count as sampling noise.
    """
    rng = np.random.default_rng(rng_seed)
    draws = {environment: rng.integers(0, len(SEEDS), size=(reps, len(SEEDS))) for environment in included}
    summary = {}
    for variant in variants:
        per_env = scores[variant.key]
        pooled = np.concatenate([per_env[environment] for environment in included])
        boot = np.concatenate([per_env[environment][draws[environment]] for environment in included], axis=1)
        within = [per_env[environment].var(ddof=1) / len(per_env[environment]) for environment in included]
        summary[variant.key] = {
            "mean": float(pooled.mean()),
            "mean_ci": tuple(np.percentile(boot.mean(axis=1), [2.5, 97.5])),
            "se": float(np.sqrt(np.sum(within)) / len(included)),
        }
    return summary


def env_header(environment: str) -> str:
    name, _, last = ENV_LABELS[environment].rpartition(" ")
    return f"\\shortstack{{{name}\\\\{last}}}" if name else ENV_LABELS[environment]


def excluded_note(excluded, method: str = "PSPO") -> str:
    if not excluded:
        return ""
    names = ", ".join(ENV_LABELS[environment] for environment in excluded)
    return (f" {names} {'is' if len(excluded) == 1 else 'are'} excluded because {method} "
            "gains no reward over its initial policy there, so the fraction is undefined.")


def latex_table(scores, anchors, included, excluded, summary, *, reps: int,
                family: Family = FAMILIES["orthotope"]) -> str:
    method, symbol = family.method, family.symbol

    def number(value: float) -> str:
        return f"${value:.2f}$" if round(value, 2) != 0 else "$0.00$"

    def with_ci(value: float, ci: tuple[float, float]) -> str:
        return f"${value:.2f}$ {{\\scriptsize$[{ci[0]:.2f}, {ci[1]:.2f}]$}}"

    width = len(included) + 2
    lines = [
        f"% {method} ablations as the fraction of {method}'s reward gain over its safe initial policy retained.",
        "% Generated by scripts/plot_pspo_ablation_gain_retained.py.",
        "% Requires \\usepackage{booktabs}.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        f"\\caption{{Fraction of {method}'s reward gain retained by each ablation, "
        f"$s=(R-R_{{\\mathrm{{init}}}})/(\\bar R_{{{symbol}}}-R_{{\\mathrm{{init}}}})$, where "
        f"$R_{{\\mathrm{{init}}}}$ is the reward of the safe initial policy {method} starts from. "
        f"$0$ is no better than the initial policy, $1$ is full {method}, and negative values are worse "
        "than the initial policy. Rewards are means over 100 unshielded greedy evaluation episodes "
        f"per seed; per-environment scores are means over $n={len(SEEDS)}$ seeds. The aggregate "
        "column is the mean over all environment--seed scores with a 95\\% stratified-bootstrap "
        f"interval ({reps:,} resamples of seeds within each environment).{excluded_note(excluded, method)}}}",
        f"\\label{{tab:{family.label}}}",
        f"\\begin{{tabular}}{{l{'r' * len(included)}c}}",
        "\\toprule",
        f"& \\multicolumn{{{len(included)}}}{{c}}{{Per environment}} & Aggregate \\\\",
        f"\\cmidrule(lr){{2-{len(included) + 1}}}\\cmidrule(lr){{{width}-{width}}}",
        "& " + " & ".join(env_header(environment) for environment in included) + " & Mean \\\\",
        "\\midrule",
        f"\\multicolumn{{{width}}}{{l}}{{\\emph{{Anchors (total reward)}}}} \\\\",
        "Initial policy $R_{\\mathrm{init}}$ & "
        + " & ".join(number(anchors[environment][0]) for environment in included) + " & \\\\",
        f"{method} $\\bar R_{{{symbol}}}$ & "
        + " & ".join(number(anchors[environment][1]) for environment in included) + " & \\\\",
        "\\midrule",
        f"\\multicolumn{{{width}}}{{l}}{{\\emph{{Fraction of {method}'s gain retained}}}} \\\\",
    ]
    group = None
    for variant in family.variants:
        if variant.group != group and variant.group:
            lines.append(f"\\multicolumn{{{width}}}{{l}}{{\\quad {variant.group}}} \\\\")
        group = variant.group
        indent = "\\quad\\quad " if variant.group else ""
        cells = [number(float(scores[variant.key][environment].mean())) for environment in included]
        stats = summary[variant.key]
        lines.append(f"{indent}{variant.label} & " + " & ".join(cells) + " & "
                     + with_ci(stats["mean"], stats["mean_ci"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def aggregate_latex_table(included, excluded, summary, *, reps: int,
                          family: Family = FAMILIES["orthotope"]) -> str:
    """One row per variant: the scaled reward averaged over all environments and seeds."""
    method, symbol = family.method, family.symbol
    lines = [
        f"% {method} ablations: scaled total reward aggregated across environments.",
        "% Generated by scripts/plot_pspo_ablation_gain_retained.py.",
        "% Requires \\usepackage{booktabs}.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{method} ablations aggregated across environments. Each seed's total reward $R$ is "
        f"scaled per environment as the fraction of {method}'s reward gain retained, "
        f"$s=(R-R_{{\\mathrm{{init}}}})/(\\bar R_{{{symbol}}}-R_{{\\mathrm{{init}}}})$, where "
        f"$R_{{\\mathrm{{init}}}}$ is the reward of the safe initial policy {method} starts from: $0$ is no "
        f"better than the initial policy and $1$ is full {method}. Entries are the mean of $s$ over all "
        f"${len(included)}$ environments $\\times$ ${len(SEEDS)}$ seeds with a 95\\% "
        f"stratified-bootstrap interval ({reps:,} resamples of seeds within each "
        f"environment).{excluded_note(excluded, method)}}}",
        f"\\label{{tab:{family.label}-aggregate}}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Variant & Gain retained & 95\\% CI \\\\",
        "\\midrule",
    ]
    group = None
    for variant in family.variants:
        if variant.group != group and variant.group:
            lines.append(f"\\multicolumn{{3}}{{l}}{{\\emph{{{variant.group}}}}} \\\\")
        group = variant.group
        indent = "\\quad " if variant.group else ""
        stats = summary[variant.key]
        low, high = stats["mean_ci"]
        lines.append(f"{indent}{variant.label} & ${stats['mean']:.2f}$ & $[{low:.2f}, {high:.2f}]$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def bar_chart(summary, family: Family = FAMILIES["orthotope"]):
    """Horizontal bars of the mean fraction retained, sorted descending, with +/- 2 SE."""
    apply_aamas_style()
    ordered = sorted(family.variants, key=lambda variant: summary[variant.key]["mean"], reverse=True)
    positions = np.arange(len(ordered))
    means = np.array([summary[variant.key]["mean"] for variant in ordered])
    errors = np.array([2.0 * summary[variant.key]["se"] for variant in ordered])
    colors = [METHOD_COLORS["pspo"] if variant.key == "pspo" else ABLATION_COLOR for variant in ordered]
    labels = [family.chart_method if variant.key == "pspo" else variant.label for variant in ordered]

    # 1.7 in for the five orthotope bars; keep the same bar pitch for fewer.
    height = 1.7 if len(ordered) >= 5 else 0.75 + 0.19 * len(ordered)
    fig, axis = plt.subplots(figsize=(COLUMN_WIDTH_IN, height), layout="constrained")
    axis.barh(positions, means, height=0.68, color=colors, edgecolor="#252525", linewidth=0.35,
              xerr=errors, error_kw={"elinewidth": 0.65, "capthick": 0.65, "capsize": 1.5}, zorder=3)
    for position, mean, error in zip(positions, means, errors):
        axis.text(mean + error + 0.02, position, f"{mean:.2f} $\\pm$ {error:.2f}", va="center",
                  ha="left", fontsize=7, zorder=4,
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})
    axis.axvline(1.0, color="#777777", linestyle="--", linewidth=0.5, zorder=1)
    axis.axvline(0.0, color="#252525", linewidth=0.55, zorder=2)
    axis.set_yticks(positions, labels)
    axis.tick_params(axis="y", length=0)
    axis.set_ylim(len(ordered) - 0.4, -0.6)
    axis.set_xlim(min(0.0, float((means - errors).min())) - 0.03, float((means + errors).max()) + 0.42)
    axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.set_xlabel(f"Fraction of {family.chart_method}'s reward gain retained")
    axis.grid(axis="x", color="#DADADA", linewidth=0.4, zorder=0)
    axis.set_axisbelow(True)
    axis.spines["left"].set_visible(False)
    return fig


def write_csv(path: Path, scores, rewards, anchors, included, variants=VARIANTS) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["environment", "variant", "seed", "total_reward", "r_init", "r_pspo_mean",
                         "fraction_retained"])
        for environment in included:
            for variant in variants:
                for seed, (reward, score) in enumerate(zip(rewards[variant.key][environment],
                                                           scores[variant.key][environment])):
                    writer.writerow([ENV_LABELS[environment], variant.label, seed, float(reward),
                                     anchors[environment][0], anchors[environment][1], float(score)])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--family", choices=tuple(FAMILIES), default="orthotope",
                        help="orthotope: PSPO and its four ablations; segment: PSPO-LS and its "
                             "initialisation ablations.")
    parser.add_argument("--name", default=None, help="Output stem (default depends on --family).")
    parser.add_argument("--bootstrap-reps", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    family = FAMILIES[args.family]
    scores, rewards, anchors, included, excluded = gain_retained(family)
    summary = aggregate(scores, included, reps=args.bootstrap_reps, rng_seed=args.bootstrap_seed,
                        variants=family.variants)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / (args.name or family.name)
    table = latex_table(scores, anchors, included, excluded, summary, reps=args.bootstrap_reps,
                        family=family)
    stem.with_suffix(".tex").write_text(table, encoding="utf-8")
    aggregate_path = stem.with_name(f"{stem.name}_aggregate.tex")
    aggregate_path.write_text(
        aggregate_latex_table(included, excluded, summary, reps=args.bootstrap_reps, family=family),
        encoding="utf-8",
    )
    write_csv(stem.with_suffix(".csv"), scores, rewards, anchors, included, family.variants)
    fig = bar_chart(summary, family)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    plt.close(fig)
    for environment in ENVIRONMENTS:
        r_init, r_pspo = anchors[environment]
        state = "excluded" if environment in excluded else "included"
        print(f"{ENV_LABELS[environment]:<20} R_init={r_init:9.3f}  R_{family.method}={r_pspo:9.3f}  {state}")
    for variant in family.variants:
        stats = summary[variant.key]
        print(f"{variant.label:<16} mean={stats['mean']:.3f} +/- {2 * stats['se']:.3f} (2 SE)"
              f"  95% bootstrap [{stats['mean_ci'][0]:.3f}, {stats['mean_ci'][1]:.3f}]")
    print(f"Saved {stem}.tex, {aggregate_path.name}, .csv, .pdf and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
