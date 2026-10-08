#!/usr/bin/env python3
"""Reward and safety under temperature-scaled stochastic deployment.

Consumes the per-cell JSONs written by ``run_temperature_sweep.py`` and draws
reward and safety rate against the action-sampling temperature, one line per
method. ``T = 0`` is greedy execution -- the regime every other figure in this
project reports, and the only one PSPO's safe-region verification certifies,
since that certificate is an argmax property.

Two x-axes are written because they answer different questions:

* **by temperature** -- the swept control variable. Plotted on evenly spaced
  categorical positions rather than a log axis, because ``T = 0`` has no place
  on a log scale and the grid is a set of chosen points, not a continuum.
* **by measured entropy** -- the same cells against the achieved policy entropy
  normalised by ``log |A|``. A fixed ``T`` is not equally stochastic across
  methods, since logit scales differ, so this is the method-fair comparison.

Tolerates partial sweeps: cells that do not exist yet are skipped and the
coverage is reported, so a running sweep can be previewed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paper_style import (  # noqa: E402
    METHOD_COLORS,
    METHOD_LINESTYLES,
    TEXT_WIDTH_IN,
    apply_paper_style,
    save_paper_figure,
)
from generate_final_policy_table import ENV_LABELS, mean_and_error  # noqa: E402
from plot_extended_budget_learning_curves import (  # noqa: E402
    ENVIRONMENTS,
    EXPECTED_SEEDS,
)
from run_temperature_sweep import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    VARIANTS,
)

DEFAULT_FIGURE_DIR = (
    REPO
    / "projects/safe_policy_optimisation/figures"
    / "pspo_vs_rl_baselines_learning_curves_all_envs"
)

# Temperatures that get a printed x tick label; the rest keep an unlabelled
# tick so the grid still shows where every swept point sits.
LABELLED_TEMPERATURES = (0.0, 0.5, 2.0, 50.0)

METRICS = (
    ("reward", "Reward"),
    ("safety", "Safety rate"),
    ("unsafe_rate", "Unsafe prop. (%)"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--ci-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--name", default="temperature_sweep", help="Output file stem prefix."
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=[key for key, _label in METRICS],
        help="Metric rows to draw; repeat (default: reward and safety).",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help=(
            "Skip the CSV and LaTeX writes. Use when re-rendering a figure "
            "variant under a different --name, so the tabular outputs are not "
            "duplicated under that name too."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Plot whatever cells exist instead of requiring a complete sweep.",
    )
    return parser.parse_args(argv)


def load_cells(sweep_dir: Path) -> pd.DataFrame:
    """One row per (environment, seed, variant, temperature)."""
    rows: list[dict] = []
    for path in sorted(sweep_dir.glob("*/seed*/*.json")):
        payload = json.loads(path.read_text())
        for temperature, summary in payload["results"].items():
            proposed = summary.get("proposed_action_safety", {})
            rows.append(
                {
                    "environment_key": payload["environment_key"],
                    "seed": payload["seed"],
                    "variant": payload["variant"],
                    "label": payload["variant_label"],
                    "temperature": float(temperature),
                    "reward": summary["reward"]["mean_total_reward"],
                    "safety": summary["safety"]["safety_rate"],
                    "success": summary["success"]["success_rate"],
                    "entropy": summary["entropy"]["mean_normalised"],
                    "unsafe_rate": 100.0
                    * proposed.get("unsafe_proposed_action_rate", float("nan")),
                }
            )
    if not rows:
        raise SystemExit(f"no sweep cells found under {sweep_dir}")
    return pd.DataFrame(rows)


def summarise(cells: pd.DataFrame, ci_multiplier: float) -> pd.DataFrame:
    """Across-seed mean and error per (environment, variant, temperature)."""
    records: list[dict] = []
    group = cells.groupby(
        ["environment_key", "variant", "label", "temperature"], sort=True
    )
    for (env_key, variant, label, temperature), frame in group:
        record = {
            "environment_key": env_key,
            "variant": variant,
            "label": label,
            "temperature": temperature,
            "seeds": len(frame),
        }
        for column in ("reward", "safety", "success", "entropy", "unsafe_rate"):
            mean, error = mean_and_error(frame[column].tolist(), ci_multiplier)
            record[f"{column}_mean"] = mean
            record[f"{column}_err"] = error
        records.append(record)
    return pd.DataFrame(records)


def _variant_style(variant: str) -> tuple[str, object]:
    return (
        METHOD_COLORS.get(variant, "grey"),
        METHOD_LINESTYLES.get(variant, "solid"),
    )


def _apply_style() -> None:
    apply_paper_style()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def save_figure(
    summary: pd.DataFrame,
    *,
    x_axis: str,
    metrics: tuple[tuple[str, str], ...],
    environments: list,
    output_dir: Path,
    ci_multiplier: float,
    name: str,
) -> Path:
    """Metric rows x environment columns, one line per method."""
    by_temperature = x_axis == "temperature"
    temperatures = sorted(summary["temperature"].unique())
    positions = {value: index for index, value in enumerate(temperatures)}

    _apply_style()
    fig, axes = plt.subplots(
        len(metrics), len(environments),
        figsize=(TEXT_WIDTH_IN, 1.15 * len(metrics) + (0.95 if len(metrics) > 1 else 1.05)),
        squeeze=False, sharex=by_temperature, layout="constrained",
    )
    for row, (metric, ylabel) in enumerate(metrics):
        for column, environment in enumerate(environments):
            axis = axes[row][column]
            block = summary[summary["environment_key"] == environment.key]
            for spec in VARIANTS:
                curve = block[block["variant"] == spec.key].sort_values("temperature")
                if curve.empty:
                    continue
                color, linestyle = _variant_style(spec.key)
                if by_temperature:
                    x = [positions[value] for value in curve["temperature"]]
                else:
                    x = curve["entropy_mean"].to_numpy(dtype=float)
                mean = curve[f"{metric}_mean"].to_numpy(dtype=float)
                band = ci_multiplier * curve[f"{metric}_err"].to_numpy(dtype=float)
                axis.plot(
                    x, mean, color=color, linestyle=linestyle, linewidth=1.0,
                    marker="o", markersize=1.6, label=spec.label,
                )
                lower, upper = mean - band, mean + band
                if metric == "safety":
                    lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
                axis.fill_between(x, lower, upper, color=color, alpha=0.12, linewidth=0)

            if metric == "safety":
                axis.set_ylim(-0.03, 1.05)
            elif metric == "unsafe_rate":
                axis.set_ylim(-3, 103)
            if by_temperature:
                # Every point gets a tick, but only a readable subset gets a
                # label: at six columns all eight labels collide into an
                # unreadable run ("00.250.5 1 2 5 1050").
                axis.set_xticks(list(positions.values()))
                axis.set_xticklabels(
                    [
                        f"{value:g}" if value in LABELLED_TEMPERATURES else ""
                        for value in temperatures
                    ],
                    fontsize=5.4,
                )
            else:
                axis.set_xlim(-0.03, 1.03)
            if column == 0:
                axis.set_ylabel(ylabel)
            elif metric != "reward":
                # Reward scale differs per environment, so those keep ticks.
                axis.tick_params(labelleft=False)
            if row == 0:
                axis.set_title(
                    ENV_LABELS[environment.key].replace(" ", "\n", 1),
                    fontsize=7, fontweight="bold", pad=3, linespacing=0.95,
                )

    fig.supxlabel(
        r"Sampling temperature $\tau$  ($\tau=0$ is greedy)" if by_temperature
        else r"Measured policy entropy $H/\log|A|$",
        fontsize=7.5,
    )
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center",
        ncol=min(4, len(labels)), frameon=False, handlelength=1.8,
        columnspacing=1.2,
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.08)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "by_temperature" if by_temperature else "by_entropy"
    stem = output_dir / f"{name}_{suffix}"
    save_paper_figure(fig, stem)
    plt.close(fig)
    return stem


# Row order for the table: the comparable, unshielded-at-deployment policies
# first, then PPO-Shield below a rule. PPO-Shield is not a point on the same
# axis as the others -- a runtime wrapper vetoes its unsafe actions, so its row
# is flat in tau by construction and would win every column trivially.
TABLE_BLOCK = (
    "ppo_policy",
    "ppo_lagrangian",
    "ppo_pid_lagrangian",
    "cpo",
    "ppo_shield_nominal",
    "pspo",
)
TABLE_CORRECTED = ("ppo_shield",)


def build_latex(summary: pd.DataFrame, *, decimals: int = 1) -> str:
    r"""Compact safety-rate table: methods down the rows, temperatures across.

    Safety rate is a rate in [0, 1] with the same meaning in every environment,
    so the across-environment mean is a legitimate summary -- unlike reward,
    whose scale differs per environment and which therefore stays figure-only.
    Cells are the mean over environments of the across-seed mean, printed as a
    percentage.

    Bold marks the best value per temperature *within the comparable block*
    (ties at the printed precision all marked). PPO-Shield sits below a rule
    and is excluded from that comparison: it corrects unsafe actions at run
    time, so it is 100% in every column by construction.

    Standard errors are omitted: at 8 temperature columns they do not fit an
    ICLR column, and the per-seed CSV carries them.
    """
    temperatures = sorted(summary["temperature"].unique())
    present = {key for key in summary["variant"].unique()}

    def percentages(key: str) -> dict[float, float]:
        block = summary[summary["variant"] == key]
        cells: dict[float, float] = {}
        for value in temperatures:
            column = block[block["temperature"] == value]["safety_mean"]
            if not column.empty:
                cells[value] = 100.0 * column.mean()
        return cells

    values = {
        spec.key: percentages(spec.key) for spec in VARIANTS if spec.key in present
    }
    # Best per column, over the comparable block only.
    best = {
        value: max(
            (
                round(values[key][value], decimals)
                for key in TABLE_BLOCK
                if key in values and value in values[key]
            ),
            default=None,
        )
        for value in temperatures
    }

    def row(spec, *, contest: bool, marker: str = "") -> str:
        cells = []
        for value in temperatures:
            if value not in values.get(spec.key, {}):
                cells.append("---")
                continue
            body = rf"{values[spec.key][value]:.{decimals}f}\%"
            winner = contest and round(values[spec.key][value], decimals) == best[value]
            cells.append(rf"\textbf{{{body}}}" if winner else body)
        return f"{spec.label}{marker} & " + " & ".join(cells) + r" \\"

    header = " & ".join(
        rf"\multicolumn{{1}}{{c}}{{{value:g}}}" for value in temperatures
    )
    columns = 1 + len(temperatures)
    lines = [
        r"% Safety rate (\%) under temperature-scaled stochastic deployment.",
        rf"% Mean over the six environments of the across-seed mean; "
        rf"{len(EXPECTED_SEEDS)} seeds, 100 episodes per cell.",
        r"% $\tau=0$ is greedy execution -- the only regime PSPO's safe-region",
        r"% certificate covers, since that certificate is an argmax property.",
        r"% Bold marks the best value per column among the methods above the",
        r"% rule; PPO-Shield is excluded because its runtime correction makes it",
        r"% 100\% by construction.",
        r"% Requires \usepackage{booktabs}. Measured 385.7pt inside \small,",
        r"% so it fits the 397.5pt (5.5in) ICLR \textwidth; the tightened",
        r"% \tabcolsep is scoped by the surrounding braces and does not leak.",
        r"{\setlength{\tabcolsep}{3pt}%",
        rf"\begin{{tabular}}{{@{{}}l{'r' * len(temperatures)}@{{}}}}",
        r"\toprule",
        rf" & \multicolumn{{{len(temperatures)}}}{{c}}{{Sampling temperature $\tau$}} \\",
        rf"\cmidrule(lr){{2-{columns}}}",
        rf"Method & {header} \\",
        r"\midrule",
    ]
    lines += [
        row(spec, contest=True)
        for spec in VARIANTS
        if spec.key in TABLE_BLOCK and spec.key in values
    ]
    corrected = [spec for spec in VARIANTS if spec.key in TABLE_CORRECTED]
    if corrected:
        lines.append(r"\midrule")
        lines += [
            row(spec, contest=False, marker=r"$^{\dagger}$")
            for spec in corrected
            if spec.key in values
        ]
    lines += [
        r"\bottomrule",
        r"\addlinespace[2pt]",
        rf"\multicolumn{{{columns}}}{{@{{}}l@{{}}}}{{\footnotesize $^{{\dagger}}$"
        r"Runtime shield corrects unsafe actions, so this row is $100\%$ by "
        r"construction.}",
        r"\end{tabular}}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cells = load_cells(args.sweep_dir)
    metrics = tuple(
        (key, label)
        for key, label in METRICS
        if key in (args.metric or ("reward", "safety"))
    )

    expected = len(ENVIRONMENTS) * 10 * len(VARIANTS)
    found = cells.groupby(["environment_key", "seed", "variant"]).ngroups
    print(f"coverage: {found}/{expected} cells")
    if found < expected and not args.allow_partial:
        raise SystemExit(
            f"sweep incomplete ({found}/{expected}); pass --allow-partial to "
            "plot anyway"
        )

    summary = summarise(cells, args.ci_multiplier)
    environments = [
        environment
        for environment in ENVIRONMENTS
        if environment.key in set(summary["environment_key"])
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.figures_only:
        cells.to_csv(args.output_dir / f"{args.name}_per_seed.csv", index=False)
        summary.to_csv(args.output_dir / f"{args.name}_summary.csv", index=False)
        table_path = args.output_dir / f"{args.name}_safety_table.tex"
        table_path.write_text(build_latex(summary), encoding="utf-8")
        print(f"Wrote {table_path}")

    for x_axis in ("temperature", "entropy"):
        stem = save_figure(
            summary, x_axis=x_axis, metrics=metrics, environments=environments,
            output_dir=args.output_dir, ci_multiplier=args.ci_multiplier,
            name=args.name,
        )
        print(f"Saved {stem}.pdf and {stem}.png")

    print("\nSafety rate by temperature (mean over seeds):")
    for environment in environments:
        print(f"  {ENV_LABELS[environment.key]}")
        block = summary[summary["environment_key"] == environment.key]
        for spec in VARIANTS:
            curve = block[block["variant"] == spec.key].sort_values("temperature")
            if curve.empty:
                continue
            cells_text = "  ".join(
                f"T={row.temperature:g}:{row.safety_mean:.2f}"
                for row in curve.itertuples()
            )
            print(f"    {spec.label:26s} {cells_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
