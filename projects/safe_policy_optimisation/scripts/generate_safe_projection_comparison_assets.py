#!/usr/bin/env python3
"""Generate the adaptive-PSPO versus safe-initialised baseline comparison.

Read completed per-seed evaluations, never rounded report values. Produce a
LaTeX table, vector PDF/PNG reward figure, inclusion snippets, aggregate CSV,
and source provenance. Error bars are twice the between-seed standard error.
The figure is rendered at native AAMAS single-column width (3.33 inches).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import METHOD_COLORS  # noqa: E402
from _aamas_reward_safety import (  # noqa: E402
    COLUMN_WIDTH_IN, apply_aamas_style, legend_indices, panel_title,
)
RUNS = REPO / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
DEFAULT_OUTPUT = REPO / "projects/safe_policy_optimisation/figures/pspo_vs_safe_projection"
ENVIRONMENTS = {
    "media_streaming": "Media Streaming",
    "colour_bomb": "Colour Bomb v1",
    "colour_bomb_v2": "Colour Bomb v2",
    "bridge_crossing": "Bridge Crossing v1",
    "bridge_crossing_v2": "Bridge Crossing v2",
    "mini_pacman": "MiniPacman",
}
# (long name, figure legend label, colour key)
METHODS = {
    "pspo": ("PSPO (adaptive LID)", "PSPO", "pspo"),
    "ppo": ("PPO + projection", "PPO projected", "ppo_policy"),
    "ppo_lagrangian": ("PPO-Lagrangian + projection", "PPO-Lagrangian projected", "ppo_lagrangian"),
    "ppo_pid_lagrangian": (
        "PPO-PID-Lagrangian + projection", "PPO-PID-Lagrangian projected", "ppo_pid_lagrangian"
    ),
    "cpo": ("CPO + projection", "CPO projected", "cpo"),
    "ppo_shield": ("PPO-Shield + projection", "PPO-Shield projected", "ppo_shield"),
}
SEEDS = tuple(range(10))
STEM = "pspo_vs_safe_projection_reward"

# Keep PSPO in the final bar position, matching the other reward/safety figures.
PLOT_METHODS = tuple(method for method in METHODS if method != "pspo") + ("pspo",)
# Projected baselines are hatched; PSPO stays solid. Hatch lines take the patch
# edge colour, so the white hatch and the dark outline are drawn as two layers
# (a dark hatch would vanish on the black PPO bars).
PROJECTED_HATCH = "//////"
HATCH_COLOR = "white"
OUTLINE_COLOR = "#252525"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def summarise(values: list[float]) -> tuple[float, float]:
    if len(values) != len(SEEDS) or not all(math.isfinite(v) for v in values):
        raise ValueError("Expected ten finite per-seed measurements.")
    return statistics.fmean(values), 2 * statistics.stdev(values) / math.sqrt(len(values))


def collect(baseline_root: Path, lid_root: Path) -> tuple[list[dict], list[dict]]:
    rows, sources = [], []
    for environment in ENVIRONMENTS:
        manifest_path = lid_root / environment / "comparison_manifest.json"
        manifest = read_json(manifest_path)
        adaptive_root = Path(manifest["adaptive_run_dir"])
        sources.append({"environment": environment, "lid_manifest": str(manifest_path)})
        for method in METHODS:
            rewards, safeties = [], []
            for seed in SEEDS:
                if method == "pspo":
                    path = adaptive_root / f"seed{seed}" / "metrics.json"
                    data = read_json(path)
                    config = read_json(path.with_name("config.json"))
                    if data["algorithm"] != "pspo_adaptive" or not config[
                        "adaptive"
                    ]["directional_rashomon_growth"]:
                        raise ValueError(f"Not adaptive-LID-growth PSPO: {path}")
                    if int(data["eval_episodes"]) != 100:
                        raise ValueError(f"Unexpected evaluation budget: {path}")
                    reward = float(data["reward"]["mean_total_reward"])
                    safety = float(data["safety"]["safety_rate"])
                else:
                    path = baseline_root / environment / method / f"seed{seed}" / "postprocess_metrics.json"
                    data = read_json(path)
                    if data["status"] != "complete" or data["smoke"]:
                        raise ValueError(f"Not a completed production run: {path}")
                    if data["evaluation_policy"] != "nominal_unshielded_deterministic":
                        raise ValueError(f"Unexpected deployed evaluation policy: {path}")
                    if int(data["episodes"]) != 100 or int(data["seed"]) != seed:
                        raise ValueError(f"Unexpected seed or evaluation budget: {path}")
                    reward = float(data["deployed"]["mean_total_reward"])
                    safety = float(data["deployed"]["safe_trajectory_rate"])
                rewards.append(reward)
                safeties.append(safety)
                sources.append({
                    "environment": environment, "method": method, "seed": seed,
                    "metrics_path": str(path), "mean_total_reward": reward,
                    "safety_rate": safety,
                })
            mean, error = summarise(rewards)
            safety_mean, safety_error = summarise(safeties)
            rows.append({
                "environment": environment, "method": method, "n_seeds": len(rewards),
                "mean_total_reward": mean, "reward_2se": error,
                "mean_safety_rate": safety_mean, "safety_2se": safety_error,
            })
    return rows, sources


def write_table(output: Path, rows: list[dict]) -> None:
    lookup = {(r["environment"], r["method"]): r for r in rows}
    lines = [
        "% Generated from unrounded per-seed evaluations; requires booktabs and graphicx.",
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}", r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Environment & \shortstack{PSPO\\(adaptive LID)} & \shortstack{PPO\\+ proj.} & \shortstack{PPO-Lag.\\+ proj.} & \shortstack{PPO-PID-Lag.\\+ proj.} & \shortstack{CPO\\+ proj.} & \shortstack{PPO-Shield\\+ proj.} \\",
        r"\midrule",
    ]
    for environment, label in ENVIRONMENTS.items():
        best = max(lookup[environment, m]["mean_total_reward"] for m in METHODS)
        cells = []
        for method in METHODS:
            row = lookup[environment, method]
            value = f'{row["mean_total_reward"]:.3f} \\pm {row["reward_2se"]:.3f}'
            if math.isclose(row["mean_total_reward"], best, abs_tol=1e-12):
                value = r"\mathbf{" + value + "}"
            cells.append("$" + value + "$")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule", r"\end{tabular}%", "}",
        r"\caption{Final total reward of PSPO with adaptive LID growth and safe-initialised baselines with conditional post-training projection into a fixed LID. Values are means $\pm$ two standard errors over ten seeds, with 100 evaluation episodes per seed. Baselines use the same safe initial actor as the fixed LID's reference policy; unsafe final actors are projected, while safe actors are retained. Evaluation uses nominal policies without runtime shielding, including PPO-Shield. All methods attain 100\% measured trajectory safety in every environment. Bold denotes the highest mean in each row, including ties, not statistical significance.}",
        r"\label{tab:pspo-safe-projection-reward}", r"\end{table}",
    ])
    (output / "reward_table.tex").write_text("\n".join(lines) + "\n")


FIGURE_HEIGHT_IN = 2.85


def validate_rows(rows: list[dict]) -> None:
    expected = {(environment, method) for environment in ENVIRONMENTS for method in METHODS}
    actual = [(row["environment"], row["method"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Expected one summary per environment-method pair")
    for row in rows:
        if row["n_seeds"] != len(SEEDS):
            raise ValueError("Expected ten seeds for every bar")
        for field in ("mean_total_reward", "reward_2se", "mean_safety_rate", "safety_2se"):
            if not math.isfinite(row[field]):
                raise ValueError(f"Non-finite summary: {field}")
        if row["reward_2se"] < 0 or row["safety_2se"] < 0:
            raise ValueError("Negative uncertainty")
        if row["mean_safety_rate"] != 1 or row["safety_2se"] != 0:
            raise ValueError("Caption assumes every final policy has 100% measured safety")


def read_summary_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["n_seeds"] = int(row["n_seeds"])
        for field in ("mean_total_reward", "reward_2se", "mean_safety_rate", "safety_2se"):
            row[field] = float(row[field])
    validate_rows(rows)
    return rows


def build_reward_figure(rows: list[dict]):
    validate_rows(rows)
    apply_aamas_style()
    lookup = {(r["environment"], r["method"]): r for r in rows}
    fig, axes = plt.subplots(3, 2, figsize=(COLUMN_WIDTH_IN, FIGURE_HEIGHT_IN), squeeze=False)
    fig.subplots_adjust(left=0.155, right=0.99, bottom=0.22, top=0.88,
                        wspace=0.35, hspace=0.75)
    positions = list(range(len(PLOT_METHODS)))
    colors = [METHOD_COLORS[METHODS[method][2]] for method in PLOT_METHODS]
    hatches = [None if method == "pspo" else PROJECTED_HATCH for method in PLOT_METHODS]
    for ax, (environment, title) in zip(axes.flat, ENVIRONMENTS.items()):
        rewards = [lookup[environment, method]["mean_total_reward"] for method in PLOT_METHODS]
        errors = [lookup[environment, method]["reward_2se"] for method in PLOT_METHODS]
        # Preserve the original baseline convention: zero for nonnegative
        # panels, panel minimum minus slack if any uncertainty extends below zero.
        lowest_error_extent = min(reward - error for reward, error in zip(rewards, errors))
        bottom = (lowest_error_extent - 0.05 * max(abs(lowest_error_extent), 1.0)
                  if lowest_error_extent < 0 else 0.0)
        upper = max(reward + error for reward, error in zip(rewards, errors))
        heights = [reward - bottom for reward in rewards]
        filled = ax.bar(positions, heights, bottom=bottom, width=0.76, color=colors,
                        edgecolor=HATCH_COLOR, linewidth=0, zorder=3)
        for bar, hatch in zip(filled.patches, hatches):
            bar.set_hatch(hatch)
        ax.bar(positions, heights, bottom=bottom, yerr=errors, width=0.76, fill=False,
               edgecolor=OUTLINE_COLOR, linewidth=0.35, capsize=1.3,
               error_kw={"elinewidth": 0.65, "capthick": 0.65}, zorder=3)
        ax.set_ylim(bottom, upper + 0.10 * max(upper - bottom, 0.01))
        ax.set_title(panel_title(title), fontsize=8, fontweight="bold", pad=4,
                     linespacing=0.95)
        ax.set_xlim(-0.60, len(PLOT_METHODS) - 0.40)
        ax.set_xticks([])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(axis="y", color="#DADADA", linewidth=0.4)
        ax.set_axisbelow(True)

    fig.text(0.015, 0.57, "Total reward", rotation=90, va="center", fontsize=8)
    # Tuple handles overlay the hatched fill and the outline, as in the bars.
    handles = [(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=HATCH_COLOR,
                              linewidth=0, hatch=hatch),
                plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=OUTLINE_COLOR,
                              linewidth=0.35))
               for color, hatch in zip(colors, hatches)]
    labels = [METHODS[method][1] for method in PLOT_METHODS]
    indices = legend_indices(len(labels), 2)
    fig.legend([handles[i] for i in indices], [labels[i] for i in indices],
               loc="lower center", bbox_to_anchor=(0.52, 0.015), ncol=2,
               frameon=False, fontsize=7, columnspacing=0.75, handletextpad=0.4,
               handlelength=1.1, labelspacing=0.3)
    return fig


def plot(output: Path, rows: list[dict]) -> None:
    fig = build_reward_figure(rows)
    # Do not crop: the PDF canvas must retain its native AAMAS column width.
    fig.savefig(output / f"{STEM}.pdf")
    fig.savefig(output / f"{STEM}.png", dpi=400)
    plt.close(fig)


def write_figure_snippet(output: Path) -> None:
    (output / "reward_figure.tex").write_text(
        "\\begin{figure}[t]\n\\centering\n"
        f"\\includegraphics[width={COLUMN_WIDTH_IN:g}in]{{{STEM}.pdf}}\n"
        "\\caption{Final total reward: adaptive-LID PSPO versus safe-initialised baselines "
        "with conditional projection into a fixed safe LID. Bars show means and error bars "
        "show two standard errors over ten seeds (100 evaluation episodes per seed). "
        "Each panel has an independent reward scale; panels with negative rewards use "
        "a lower anchor rather than zero. All final nominal policies attain "
        "100\\% measured trajectory safety; no runtime shield is used at evaluation. "
        "Hatched bars are the projected baselines.}\n"
        "\\label{fig:pspo-safe-projection-reward}\n"
        "\\Description{Six environment panels compare total reward of adaptive-LID PSPO "
        "with PPO, PPO-Lagrangian, PPO-PID-Lagrangian, CPO and PPO-Shield baselines "
        "using safe initialisation and conditional final projection into a fixed LID. "
        "Panels run left to right and top to bottom: Media Streaming, Colour Bomb v1, "
        "Colour Bomb v2, Bridge Crossing v1, Bridge Crossing v2, MiniPacman. "
        "Colours and the shared legend identify the methods; the projected baselines' bars "
        "are diagonally hatched and PSPO's bars are solid. "
        "Error bars show two standard errors across ten seeds.}\n"
        "\\end{figure}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=RUNS / "safe_initialised_baseline_projection")
    parser.add_argument("--lid-root", type=Path, default=RUNS / "static_lid_ablation_masa_matched")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-only", action="store_true",
                        help="Update only the figure and snippet from the existing unrounded CSV.")
    args = parser.parse_args()
    if args.figure_only:
        output = args.output_dir.resolve()
        rows = read_summary_csv(output / "reward_comparison.csv")
        plot(output, rows)
        write_figure_snippet(output)
        print(f"Updated AAMAS single-column figure from unchanged CSV in {output}")
        return 0
    rows, sources = collect(args.baseline_root.resolve(), args.lid_root.resolve())
    if any(r["mean_safety_rate"] != 1 or r["safety_2se"] != 0 for r in rows):
        raise ValueError("Caption assumes every final policy has 100% measured safety.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "reward_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "provenance.json").write_text(json.dumps({
        "pspo_variant": "adaptive LID growth", "seeds": SEEDS,
        "evaluation_episodes_per_seed": 100, "error": "2 * sample SD / sqrt(10)",
        "sources": sources,
    }, indent=2) + "\n")
    write_table(output, rows)
    plot(output, rows)
    write_figure_snippet(output)
    (output / "comparison.tex").write_text(
        "\\documentclass{article}\n"
        "\\usepackage[textwidth=5.5in,textheight=9in]{geometry}\n"
        "\\usepackage{booktabs,graphicx}\n"
        "\\providecommand{\\Description}[1]{}\n\\begin{document}\n"
        "\\input{reward_table.tex}\n\\input{reward_figure.tex}\n\\end{document}\n"
    )
    print(f"Generated {len(rows)} method-environment summaries from 360 seed evaluations in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
