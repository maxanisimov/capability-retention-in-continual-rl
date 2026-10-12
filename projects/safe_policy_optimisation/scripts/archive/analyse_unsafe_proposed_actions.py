#!/usr/bin/env python3
"""How often does each method *propose* an unsafe action, and in which phase?

The safety rates in the other figures say whether an unsafe action was ever
*executed*. That is a property of the deployed system, shield included. This
one asks the upstream question -- what did the policy want to do -- which is
what separates a policy that is safe from a policy that is being kept safe.

Three phases, each with its own denominator:

* **exploration** -- every environment step taken during training. Note *both*
  methods explore with a shield attached (``ProvablySafePPO`` shields inside
  ``collect_rollouts``), so the number here is the shield's intervention rate:
  how often it had to correct the policy. Read from ``summary.json``
  (``unsafe_proposed_actions_during_exploration`` over
  ``training_shield_diagnostics.checked``).

* **training-time evaluation** -- the periodic 20-episode checkpoints, summed
  over every checkpoint of the run, from the ``learning_curves`` summaries.

* **deployment** -- the terminal 100-episode greedy evaluation, from
  ``summary.json``'s ``evaluation_proposed_action_safety*`` blocks.

PPO-Shield is reported twice in the two evaluation phases, because the shield
changes the *state distribution* the policy is asked about, not just the action
that gets executed: a shielded rollout keeps the agent in states it handles,
while a shield-free rollout lets it drift into states where it proposes far
worse actions. Exploration has no shield-free counterpart -- the agent explored
once, with the shield on -- so that series is absent there rather than zero.

Counts are *not* comparable across series: episodes that end early check fewer
actions, so a collapsing policy can post a low count purely by dying sooner.
The rate is the comparable quantity, and the table prints the denominator next
to every count so the difference is visible.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paper_style import (  # noqa: E402
    METHOD_COLORS,
    TEXT_WIDTH_IN,
    apply_paper_style,
    save_paper_figure,
)
from generate_final_policy_table import (  # noqa: E402
    ENV_LABELS,
    mean_and_error,
)
from plot_extended_budget_learning_curves import (  # noqa: E402
    ENVIRONMENT_BY_KEY,
    ENVIRONMENTS,
    EXPECTED_SEEDS,
    EnvironmentSpec,
)

DEFAULT_OUTPUT_DIR = (
    REPO
    / "projects/safe_policy_optimisation/figures"
    / "pspo_vs_rl_baselines_learning_curves_all_envs"
)

_CURVES = "ppo_shield/learning_curves"


@dataclass(frozen=True)
class SeriesSpec:
    key: str
    label: str
    color: str
    hatch: str | None = None


SERIES = (
    SeriesSpec("ppo_shield", "PPO-Shield", METHOD_COLORS["ppo_shield"]),
    SeriesSpec(
        "ppo_shield_nominal",
        "PPO-Shield (shield-free rollout)",
        METHOD_COLORS["ppo_shield_nominal"],
        "//",
    ),
    SeriesSpec("pspo", "PSPO", METHOD_COLORS["pspo"]),
)
SERIES_BY_KEY = {spec.key: spec for spec in SERIES}
# Shorter than the figure labels: the table's method column is narrow at ICLR
# single-column width.
LATEX_LABELS = {
    "ppo_shield": "PPO-Shield",
    "ppo_shield_nominal": "PPO-Shield (shield-free)",
    "pspo": "PSPO",
}
SLOT = {spec.key: index for index, spec in enumerate(SERIES)}

PHASES = ("exploration", "training_evaluation", "deployment")
PHASE_LABELS = {
    "exploration": "Exploration",
    "training_evaluation": "Training-time eval.",
    "deployment": "Deployment eval.",
}
# Exploration happened once, with the shield attached, so there is no
# shield-free counterpart to report.
PHASE_SERIES = {
    "exploration": ("ppo_shield", "pspo"),
    "training_evaluation": ("ppo_shield", "ppo_shield_nominal", "pspo"),
    "deployment": ("ppo_shield", "ppo_shield_nominal", "pspo"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        action="append",
        choices=tuple(ENVIRONMENT_BY_KEY),
        help="Environment to include; repeat as needed (default: all six).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ci-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--reuse-summary",
        action="store_true",
        help=(
            "Rebuild the table and figures from the previously written "
            "summary CSV instead of re-reading every per-seed curve. Only "
            "valid when nothing upstream has changed."
        ),
    )
    parser.add_argument(
        "--decimals", type=int, default=2,
        help="Printed decimal places in the LaTeX table (default: 2).",
    )
    parser.add_argument(
        "--name",
        default="unsafe_proposed_actions",
        help="Output file stem (default: unsafe_proposed_actions).",
    )
    return parser.parse_args(argv)


def _curve_totals(path: Path) -> tuple[float, float]:
    """Sum proposals and unsafe proposals over every periodic checkpoint."""
    frame = pd.read_csv(
        path, usecols=["proposed_action_checks", "unsafe_proposed_action_count"]
    )
    return (
        float(frame["unsafe_proposed_action_count"].sum()),
        float(frame["proposed_action_checks"].sum()),
    )


def read_seed_rows(environment: EnvironmentSpec) -> list[dict]:
    """One row per (phase, series, seed): unsafe count and proposals checked."""
    rows: list[dict] = []
    for seed in EXPECTED_SEEDS:
        shield_dir = environment.baseline_root / f"seed{seed}" / "ppo_shield"
        pspo_dir = environment.adaptive_root / f"seed{seed}"
        shield = json.loads((shield_dir / "summary.json").read_text())
        pspo = json.loads((pspo_dir / "summary.json").read_text())

        def add(phase: str, key: str, unsafe: float, checks: float) -> None:
            rows.append(
                {
                    "environment_key": environment.key,
                    "environment": ENV_LABELS[environment.key],
                    "phase": phase,
                    "method_key": key,
                    "method": SERIES_BY_KEY[key].label,
                    "seed": seed,
                    "unsafe_proposed": unsafe,
                    "proposals_checked": checks,
                    "unsafe_rate": unsafe / checks if checks else float("nan"),
                }
            )

        add(
            "exploration", "ppo_shield",
            shield["unsafe_proposed_actions_during_exploration"],
            shield["training_shield_diagnostics"]["checked"],
        )
        add(
            "exploration", "pspo",
            pspo["unsafe_proposed_actions_during_exploration"],
            pspo["training_shield_diagnostics"]["checked"],
        )

        for key, filename in (
            ("ppo_shield", "evaluation_shielded_summary.csv"),
            ("ppo_shield_nominal", "evaluation_unshielded_summary.csv"),
        ):
            add("training_evaluation", key, *_curve_totals(
                shield_dir / "learning_curves" / filename
            ))
        add("training_evaluation", "pspo", *_curve_totals(
            pspo_dir / "learning_curves/evaluation_unshielded_summary.csv"
        ))

        for key, block in (
            ("ppo_shield", "evaluation_proposed_action_safety_shielded"),
            ("ppo_shield_nominal", "evaluation_proposed_action_safety_nominal"),
        ):
            node = shield[block]
            add("deployment", key, node["unsafe_proposed_action_count"],
                node["proposed_action_checks"])
        node = pspo["evaluation_proposed_action_safety"]
        add("deployment", "pspo", node["unsafe_proposed_action_count"],
            node["proposed_action_checks"])
    return rows


def summarise(per_seed: pd.DataFrame, ci_multiplier: float) -> pd.DataFrame:
    """Across-seed mean +/- ci_multiplier standard errors, per cell."""
    records: list[dict] = []
    group = per_seed.groupby(
        ["environment_key", "environment", "phase", "method_key", "method"],
        sort=False,
    )
    for (env_key, env_label, phase, key, label), frame in group:
        count_mean, count_error = mean_and_error(
            frame["unsafe_proposed"].tolist(), ci_multiplier
        )
        rate_mean, rate_error = mean_and_error(
            frame["unsafe_rate"].tolist(), ci_multiplier
        )
        checks_mean, _ = mean_and_error(
            frame["proposals_checked"].tolist(), ci_multiplier
        )
        records.append(
            {
                "environment_key": env_key, "environment": env_label,
                "phase": phase, "method_key": key, "method": label,
                "seeds": len(frame),
                "unsafe_proposed_mean": count_mean,
                "unsafe_proposed_err": count_error,
                "proposals_checked_mean": checks_mean,
                "unsafe_rate_mean": rate_mean,
                "unsafe_rate_err": rate_error,
            }
        )
    return pd.DataFrame(records)


def _apply_style() -> None:
    apply_paper_style()
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def save_figure(
    summary: pd.DataFrame,
    *,
    metric: str,
    environments: tuple[EnvironmentSpec, ...],
    output_dir: Path,
    name: str,
) -> Path:
    """Phase rows x environment columns; one bar per series.

    Rate is a percentage on a shared 0--100 axis, so every panel is directly
    comparable. Count is symlog: the values span zero to several million, and a
    linear axis would flatten every small-but-nonzero bar onto the baseline.
    """
    is_rate = metric == "rate"
    # Ticks are printed only on the leftmost panel, so the y-axis must be
    # shared across each row or equal-height bars would encode different
    # values. The rate axis is pinned to 0--100 everywhere and needs no
    # sharing; counts span orders of magnitude and do.
    fig, axes = plt.subplots(
        len(PHASES), len(environments), figsize=(TEXT_WIDTH_IN, 3.5),
        squeeze=False, layout="constrained",
        sharey=False if is_rate else "row",
    )
    for row, phase in enumerate(PHASES):
        keys = PHASE_SERIES[phase]
        # Tracked across the row because the count axis is shared: the top has
        # to clear the tallest bar in the whole row, not just the first panel's.
        row_ceiling = 0.0
        for column, environment in enumerate(environments):
            axis = axes[row][column]
            cell = summary[
                (summary["phase"] == phase)
                & (summary["environment_key"] == environment.key)
            ].set_index("method_key")
            positions, values, errors, colors, hatches = [], [], [], [], []
            for key in keys:
                if key not in cell.index:
                    continue
                spec = SERIES_BY_KEY[key]
                # Slot is fixed by SERIES order, not by this phase's member
                # list, so a method occupies the same x position in every row.
                # Exploration therefore leaves the middle slot empty rather
                # than sliding PSPO left into the shield-free column.
                index = SLOT[key]
                scale = 100.0 if is_rate else 1.0
                positions.append(index)
                values.append(cell.loc[key, f"unsafe_{'rate' if is_rate else 'proposed'}_mean"] * scale)
                errors.append(cell.loc[key, f"unsafe_{'rate' if is_rate else 'proposed'}_err"] * scale)
                colors.append(spec.color)
                hatches.append(spec.hatch)
            bars = axis.bar(
                positions, values, yerr=errors, capsize=2, color=colors,
                edgecolor="black", linewidth=0.5,
                error_kw={"linewidth": 0.8, "ecolor": "black"},
            )
            for bar, hatch in zip(bars, hatches):
                if hatch:
                    bar.set_hatch(hatch)
            axis.set_xticks(range(len(SERIES)))
            axis.set_xticklabels([])
            axis.set_xlim(-0.7, len(SERIES) - 0.3)
            if is_rate:
                axis.set_ylim(0, 100)
                axis.set_yticks((0, 50, 100))
            else:
                axis.set_yscale("symlog", linthresh=1.0)
                row_ceiling = max(row_ceiling, *(values or [0.0]))
            if column == 0:
                axis.set_ylabel(
                    PHASE_LABELS[phase], fontsize=6.5, fontweight="bold",
                )
            else:
                axis.tick_params(labelleft=False)
            if row == 0:
                axis.set_title(
                    ENV_LABELS[environment.key].replace(" ", "\n", 1),
                    fontsize=7, fontweight="bold", pad=3, linespacing=0.95,
                )
            # A zero bar is the headline result, so say so rather than leaving
            # an empty slot the reader has to interpret.
            for index, value in zip(positions, values):
                if value == 0:
                    axis.annotate(
                        "0", (index, 0), xytext=(0, 2),
                        textcoords="offset points", ha="center",
                        fontsize=5.5, fontweight="bold", color="#1b7a3d",
                    )
        if not is_rate:
            # Headroom for the error bars, applied once the row is complete.
            for axis in axes[row]:
                axis.set_ylim(0, max(row_ceiling * 2.0, 1.0))

    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=spec.color, edgecolor="black",
            linewidth=0.5, hatch=spec.hatch,
        )
        for spec in SERIES
    ]
    fig.legend(
        handles, [spec.label for spec in SERIES], loc="outside lower center",
        ncol=len(SERIES), frameon=False, columnspacing=1.2,
        handletextpad=0.5, handlelength=1.2,
    )
    fig.supylabel(
        "Unsafe proposed actions (%)" if is_rate
        else "Unsafe proposed actions (count, symlog)",
        fontsize=7.5,
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.08)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{name}_{metric}"
    save_paper_figure(fig, stem)
    plt.close(fig)
    return stem


# The paper table compares the deployed system against PSPO. The shield-free
# rollout stays in the figures and the markdown, where there is room to explain
# that it is the same weights on a different state distribution.
TABLE_SERIES = ("ppo_shield", "pspo")
PHASE_TABLE_HEADERS = {
    "exploration": "Exploration",
    "training_evaluation": "Training-time eval.",
    "deployment": "Deployment eval.",
}


def _compact_cell(mean: float, decimals: int, best: bool) -> str:
    r"""``12.87\%``, bolded when it is the better of the pair.

    Plain text rather than math mode: with the standard errors gone there is no
    ``\pm`` left to typeset, so ``\textbf`` suffices and the table needs no
    amsmath.
    """
    body = rf"{mean:.{decimals}f}\%"
    return rf"\textbf{{{body}}}" if best else body


def build_latex(summary: pd.DataFrame, *, decimals: int, ci_multiplier: float) -> str:
    r"""Environments down the rows, phase x method across the columns.

    One row per environment rather than one per (environment, method): with
    only two methods the whole table is six body rows instead of eighteen.
    Bold marks the lower rate of the pair -- fewer unsafe proposals is better,
    the opposite of the reward table.

    Numeric columns are right-aligned so the decimal points line up, with the
    method names centred over them via ``\multicolumn``. Setting the column
    type to ``c`` instead would centre the numbers too and break that
    alignment, since the integer parts differ in width (``1.05\%`` vs
    ``81.20\%``).
    """
    columns = "l" + "r" * (len(PHASES) * len(TABLE_SERIES))
    group_header = " & ".join(
        rf"\multicolumn{{{len(TABLE_SERIES)}}}{{c}}{{{PHASE_TABLE_HEADERS[phase]}}}"
        for phase in PHASES
    )
    rules = " ".join(
        rf"\cmidrule(lr){{{2 + index * len(TABLE_SERIES)}-"
        rf"{1 + (index + 1) * len(TABLE_SERIES)}}}"
        for index in range(len(PHASES))
    )
    method_header = " & ".join(
        rf"\multicolumn{{1}}{{c}}{{{LATEX_LABELS[key]}}}"
        for _phase in PHASES
        for key in TABLE_SERIES
    )
    row_end = r" \\"
    lines = [
        r"% Unsafe proposed-action rate, by training/evaluation phase.",
        rf"% Cells are the mean over {len(EXPECTED_SEEDS)} seeds; bold marks "
        r"the lower rate of each pair.",
        rf"% Standard errors are omitted for space; see "
        rf"unsafe_proposed_actions.md (mean +/- {ci_multiplier:g} s.e.).",
        r"% Both methods explore with a shield attached, so the exploration",
        r"% column is the shield's intervention rate rather than a shield-free",
        r"% measurement. The evaluation columns are shielded for PPO-Shield (as",
        r"% deployed) and unshielded for PSPO (which uses no shield at all).",
        r"% Requires \usepackage{booktabs}. Measured natural widths against",
        r"% ICLR's 5.5in (397.5pt) column: \normalsize is 411.3pt (overruns),",
        r"% \small is 386.4pt, \small with \tabcolsep=4pt is 360.4pt. Wrap the",
        r"% size in a group so it stays local to the table.",
        rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}",
        r"\toprule",
        f" & {group_header}{row_end}",
        rules,
        f"Environment & {method_header}{row_end}",
        r"\midrule",
    ]
    for environment_key in dict.fromkeys(summary["environment_key"]):
        block = summary[summary["environment_key"] == environment_key]
        cells = []
        for phase in PHASES:
            rates = {}
            for key in TABLE_SERIES:
                row = block[(block["phase"] == phase) & (block["method_key"] == key)]
                rates[key] = None if row.empty else row.iloc[0]
            printed = [
                round(100 * row["unsafe_rate_mean"], decimals)
                for row in rates.values()
                if row is not None
            ]
            best = min(printed) if printed else None
            for key in TABLE_SERIES:
                row = rates[key]
                if row is None:
                    cells.append("---")
                    continue
                mean = 100 * row["unsafe_rate_mean"]
                cells.append(
                    _compact_cell(mean, decimals, round(mean, decimals) == best)
                )
        lines.append(
            f"{ENV_LABELS[environment_key]} & " + " & ".join(cells) + row_end
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def build_markdown(summary: pd.DataFrame, ci_multiplier: float) -> str:
    lines = [
        "# Unsafe proposed actions by phase",
        "",
        f"Mean ± {ci_multiplier:g} s.e. over {len(EXPECTED_SEEDS)} seeds. "
        "`checked` is the mean number of proposed actions audited per seed --",
        "counts are only comparable against a like denominator, so read the "
        "rate column when the denominators differ.",
        "",
    ]
    for phase in PHASES:
        lines += [
            f"## {PHASE_LABELS[phase]}",
            "",
            "| Environment | Method | Unsafe proposed | Checked | Unsafe rate (%) |",
            "|---|---|---:|---:|---:|",
        ]
        subset = summary[summary["phase"] == phase]
        for env_key in dict.fromkeys(subset["environment_key"]):
            for key in PHASE_SERIES[phase]:
                row = subset[
                    (subset["environment_key"] == env_key)
                    & (subset["method_key"] == key)
                ]
                if row.empty:
                    continue
                row = row.iloc[0]
                lines.append(
                    f"| {row['environment']} | {row['method']} | "
                    f"{row['unsafe_proposed_mean']:,.0f} ± "
                    f"{row['unsafe_proposed_err']:,.0f} | "
                    f"{row['proposals_checked_mean']:,.0f} | "
                    f"{100 * row['unsafe_rate_mean']:.3f} ± "
                    f"{100 * row['unsafe_rate_err']:.3f} |"
                )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    keys = args.environment or [environment.key for environment in ENVIRONMENTS]
    environments = tuple(ENVIRONMENT_BY_KEY[key] for key in dict.fromkeys(keys))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = args.output_dir / f"{args.name}_per_seed.csv"
    summary_path = args.output_dir / f"{args.name}_summary.csv"
    if args.reuse_summary:
        per_seed = pd.read_csv(per_seed_path)
        summary = pd.read_csv(summary_path)
    else:
        rows: list[dict] = []
        for environment in environments:
            rows.extend(read_seed_rows(environment))
        per_seed = pd.DataFrame(rows)
        summary = summarise(per_seed, args.ci_multiplier)
    markdown_path = args.output_dir / f"{args.name}.md"
    if not args.reuse_summary:
        per_seed.to_csv(per_seed_path, index=False)
        summary.to_csv(summary_path, index=False)
    markdown_path.write_text(
        build_markdown(summary, args.ci_multiplier), encoding="utf-8"
    )
    latex_path = args.output_dir / f"{args.name}_rate_table.tex"
    latex_path.write_text(
        build_latex(
            summary, decimals=args.decimals, ci_multiplier=args.ci_multiplier
        ),
        encoding="utf-8",
    )

    _apply_style()
    for metric in ("rate", "count"):
        stem = save_figure(
            summary, metric=metric, environments=environments,
            output_dir=args.output_dir, name=args.name,
        )
        print(f"Saved {stem}.pdf and {stem}.png")
    print(f"Wrote {per_seed_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {latex_path}")

    print("\nUnsafe proposed-action rate (%), mean over seeds:")
    for phase in PHASES:
        print(f"  {PHASE_LABELS[phase]}")
        subset = summary[summary["phase"] == phase]
        for env_key in dict.fromkeys(subset["environment_key"]):
            cells = []
            for key in PHASE_SERIES[phase]:
                row = subset[
                    (subset["environment_key"] == env_key)
                    & (subset["method_key"] == key)
                ]
                if not row.empty:
                    cells.append(
                        f"{SERIES_BY_KEY[key].label}={100 * row.iloc[0]['unsafe_rate_mean']:.3f}"
                    )
            print(f"    {ENV_LABELS[env_key]:20s} " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
