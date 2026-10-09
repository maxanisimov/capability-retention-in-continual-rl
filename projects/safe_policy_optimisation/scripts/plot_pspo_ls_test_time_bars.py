#!/usr/bin/env python3
"""Create new test-time charts with region-first PSPO-LS labelled as PSPO.

Keep the existing unrounded baseline summaries; recompute only PSPO from the
ten seed-level final evaluations in segment_lid. Export both reward/safety
comparisons and the projected-baseline reward-only layout used in the paper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _aamas_reward_safety import BarMethod, BarPanel, compact_figure, save_compact_figure
from generate_final_policy_table import mean_and_error
import generate_safe_projection_comparison_assets as projection
from plot_extended_budget_learning_curves import ENVIRONMENTS, EXPECTED_SEEDS, RUNS
from plot_rl_sgf_final_policy_reward_safety_bars import method_order, zoom_axes

FIGURES = REPO / "projects/safe_policy_optimisation/figures"
DEFAULT_OUTPUT = FIGURES / "pspo_ls_test_time"
MAIN_STEM = "main_test_time_reward_safety_pspo_ls"
PROJECTED_STEM = "projected_baselines_test_time_reward_safety_pspo_ls"
REWARD_STEM = "projected_baselines_test_time_reward_pspo_ls"
VARIANT_CAPTION = (
    "The PSPO bars report PSPO-LS with region-first line-segment certification "
    "($K=4$, $K_{\\max}=8$, $\\varepsilon=10^{-3}$). "
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_segment_summaries(root: Path) -> tuple[dict[str, dict], list[dict]]:
    summaries, sources = {}, []
    for environment in ENVIRONMENTS:
        rewards, safeties = [], []
        for seed in EXPECTED_SEEDS:
            directory = root / environment.key / f"seed{seed}"
            metrics_path = directory / "metrics.json"
            config_path = directory / "config.json"
            metrics, config = read_json(metrics_path), read_json(config_path)
            adaptive = config["adaptive"]
            if (
                adaptive["safe_region_shape"] != "segment"
                or adaptive["verify_first"]
                or config["evaluation_policy"] != "unshielded"
                or config["seed"] != seed
                or config["total_timesteps"] != environment.nominal_budget
                or metrics["algorithm"] != "pspo"
                or metrics["eval_episodes"] != 100
                or config["eval_episodes"] != 100
                or adaptive["segment_splits"] != 4
                or adaptive["segment_max_splits"] != 8
                or adaptive["segment_tolerance"] != 1e-3
            ):
                raise ValueError(f"Unexpected PSPO-LS run/protocol: {directory}")
            reward = float(metrics["reward"]["mean_total_reward"])
            safety = float(metrics["safety"]["safety_rate"])
            if not math.isfinite(reward) or not 0 <= safety <= 1:
                raise ValueError(f"Invalid final evaluation: {metrics_path}")
            rewards.append(reward)
            safeties.append(safety)
            sources.append({
                "environment": environment.key, "seed": seed,
                "metrics_path": str(metrics_path), "metrics_sha256": sha256(metrics_path),
                "config_path": str(config_path), "config_sha256": sha256(config_path),
                "base_policy_path": config["base_policy_path"],
                "reward": reward, "safety_rate": safety,
            })
        reward_mean, reward_error = mean_and_error(rewards, 2.0)
        safety_mean, safety_error = mean_and_error(safeties, 2.0)
        summaries[environment.key] = {
            "reward_mean": reward_mean, "reward_error": reward_error,
            "safety_mean": safety_mean, "safety_error": safety_error,
        }
    return summaries, sources


def load_main_panels(path: Path, summaries: dict[str, dict]) -> tuple[list[BarPanel], list[BarMethod]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(row["environment"], row["method"]): row for row in rows}
    methods = [method for _, method in method_order()]
    expected = {(environment.label, method.label)
                for environment in ENVIRONMENTS for method in methods}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise ValueError("Expected the complete six-environment, eight-method main chart")
    panels = []
    for environment in ENVIRONMENTS:
        values = []
        for method in methods:
            row = lookup[environment.label, method.label]
            if float(row["se_multiplier"]) != 2:
                raise ValueError("Main chart must use two standard errors")
            values.append(summaries[environment.key] if method.key == "pspo" else {
                field: float(row[field])
                for field in ("reward_mean", "reward_error", "safety_mean", "safety_error")
            })
        panels.append(BarPanel(environment.label, *[
            [value[field] for value in values]
            for field in ("reward_mean", "reward_error", "safety_mean", "safety_error")
        ]))
    return panels, methods


def load_projected_rows(path: Path, summaries: dict[str, dict]) -> list[dict]:
    rows = projection.read_summary_csv(path)
    for row in rows:
        if row["method"] == "pspo":
            summary = summaries[row["environment"]]
            row.update(mean_total_reward=summary["reward_mean"],
                       reward_2se=summary["reward_error"],
                       mean_safety_rate=summary["safety_mean"],
                       safety_2se=summary["safety_error"])
    projection.validate_rows(rows)
    return rows


def projected_panels(rows: list[dict]) -> tuple[list[BarPanel], list[BarMethod]]:
    lookup = {(row["environment"], row["method"]): row for row in rows}
    methods = [BarMethod(key, projection.METHODS[key][1],
                         projection.COLORS[projection.METHODS[key][2]],
                         None if key == "pspo" else projection.PROJECTED_HATCH)
               for key in projection.PLOT_METHODS]
    panels = []
    for key, label in projection.ENVIRONMENTS.items():
        values = [lookup[key, method.key] for method in methods]
        panels.append(BarPanel(label, *[
            [row[field] for row in values]
            for field in ("mean_total_reward", "reward_2se", "mean_safety_rate", "safety_2se")
        ]))
    return panels, methods


def projected_hatches(fig, methods: list[BarMethod]) -> None:
    """Use the existing projection chart's white hatching and dark outlines."""
    for axis in fig.axes:
        for bar, method in zip(list(axis.patches), methods):
            if method.key != "pspo":
                bar.set_edgecolor(projection.HATCH_COLOR)
                bar.set_linewidth(0)
                axis.add_patch(Rectangle(
                    (bar.get_x(), bar.get_y()), bar.get_width(), bar.get_height(),
                    fill=False, edgecolor=projection.OUTLINE_COLOR, linewidth=0.35,
                    zorder=bar.get_zorder() + 0.1,
                ))
    # Legend entries retain the same hatch colour as the plotted bars.
    handles = fig.legends[0].legend_handles
    for handle in handles:
        if handle.get_hatch():
            handle.set_edgecolor(projection.HATCH_COLOR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-root", type=Path, default=RUNS / "segment_lid/two_hidden")
    parser.add_argument("--main-csv", type=Path, default=FIGURES / "aamas/final_policy_reward_safety_bars_rl_sgf_transposed.csv")
    parser.add_argument("--projected-csv", type=Path, default=FIGURES / "pspo_vs_safe_projection/reward_comparison.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    summaries, sources = load_segment_summaries(args.segment_root.resolve())
    main_panels, main_methods = load_main_panels(args.main_csv, summaries)
    rows = load_projected_rows(args.projected_csv, summaries)
    panels, methods = projected_panels(rows)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    caption = (
        "Final greedy-policy total reward and safety rate (mean $\\pm$ two standard "
        "errors over ten seeds, 100 evaluation episodes per seed). " + VARIANT_CAPTION +
        "RL-SGF and PSPO use safe initialisation; the other methods use random initialisation. "
        "PPO-Shield (shield on) uses its runtime shield; the shield-off bar reports "
        "the same trained policy without it. Both axes are truncated: each panel starts "
        "25\\% of its lowest mean below that mean."
    )
    fig = compact_figure(main_panels, main_methods, transpose=True)
    zoom_axes(fig, main_panels)
    save_compact_figure(fig, output / MAIN_STEM, panels=main_panels,
                        methods=main_methods, se_multiplier=2.0, caption=caption)

    caption = (
        "Final nominal-policy total reward and safety rate: PSPO versus safe-initialised "
        "baselines with conditional final projection into a fixed safe LID. " + VARIANT_CAPTION +
        "All policies are evaluated without runtime shielding, including PPO-Shield. "
        "Bars show means $\\pm$ two standard errors over ten seeds, with 100 evaluation "
        "episodes per seed. The PSPO-LS initial actor differs from the baselines' fixed-LID "
        "reference actor. Cross-hatched bars are the projected baselines. "
        "All final policies attain 100\\% measured trajectory safety."
    )
    fig = compact_figure(panels, methods, transpose=True)
    # Long projected-baseline names use the existing reward-only legend typography.
    legend = fig.legends[0]
    legend.set_bbox_to_anchor((0.50, 0.015))
    for text in legend.get_texts():
        text.set_fontsize(6.0)
    projected_hatches(fig, methods)
    save_compact_figure(fig, output / PROJECTED_STEM, panels=panels,
                        methods=methods, se_multiplier=2.0, caption=caption)

    fig = projection.build_reward_figure(rows)
    fig.savefig(output / f"{REWARD_STEM}.pdf")
    fig.savefig(output / f"{REWARD_STEM}.png", dpi=400)
    plt.close(fig)
    with (output / f"{REWARD_STEM}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    reward_caption = caption.replace("total reward and safety rate:", "total reward:")
    (output / f"{REWARD_STEM}.tex").write_text(
        "\\begin{figure}[t]\n  \\centering\n"
        f"  \\includegraphics[width=3.33in]{{{REWARD_STEM}.pdf}}\n"
        f"  \\caption{{{reward_caption} Reward axes are truncated at 25\\% of the lowest "
        "mean below that mean; compare labelled values rather than bar lengths.}\n"
        "  \\label{fig:projected-baselines-test-time-reward-pspo-ls}\n"
        "  \\Description{Six panels compare final total reward. PSPO bars use PSPO-LS "
        "results; projected baselines are cross-hatched. Error bars show two standard "
        "errors across ten seeds.}\n\\end{figure}\n"
    )
    provenance = {
        "pspo_bar_label": "PSPO", "pspo_variant": "PSPO-LS (region-first)",
        "segment_root": str(args.segment_root.resolve()),
        "seeds": list(EXPECTED_SEEDS), "evaluation_episodes_per_seed": 100,
        "error": "2 * sample SD / sqrt(10)",
        "baseline_summary_sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in (args.main_csv, args.projected_csv)
        ],
        "initialisation_note": "PSPO-LS initial weights differ from the projected baselines' fixed-LID reference policy; no identical-initialisation claim is made.",
        "pspo_seed_sources": sources,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Test-time charts with PSPO-LS as PSPO\n\n"
        "The PSPO bars use region-first line-segment results from `segment_lid`, "
        "not the verify-first variant. All baseline values and uncertainties are "
        "retained from the existing unrounded chart CSVs. Each PSPO value is "
        "recomputed from ten final unshielded evaluations of 100 episodes per seed. "
        "Error bars are two between-seed standard errors.\n\n"
        "The PSPO-LS initial weights differ from the projected baselines' fixed-LID "
        "reference actor. This is a comparison of existing completed runs, not a "
        "new experiment with identical initial weights.\n\n"
        "Each chart has a vector PDF, 400-dpi PNG, unrounded CSV and LaTeX snippet. "
        "The projected comparison is supplied both with reward/safety panels and "
        "in the paper's reward-only layout. `provenance.json` records input paths "
        "and hashes for the existing baseline summaries and all 60 PSPO-LS evaluations.\n\n"
        "Regenerate from the repository root:\n\n```bash\n"
        "MPLCONFIGDIR=/tmp/pspo-ls-charts-mpl .venv/bin/python "
        "projects/safe_policy_optimisation/scripts/plot_pspo_ls_test_time_bars.py\n```\n"
    )
    print(f"Created three PSPO-LS test-time charts in {output}")
    for key, summary in summaries.items():
        print(f"{key}: reward {summary['reward_mean']:.4f} +/- "
              f"{summary['reward_error']:.4f}; safety {summary['safety_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
