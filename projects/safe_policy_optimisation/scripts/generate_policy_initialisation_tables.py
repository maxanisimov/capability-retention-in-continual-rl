#!/usr/bin/env python3
"""Emit the compact LaTeX tables for the policy-initialisation ablation.

Three tables, all sized for the ICLR single-column ``\\textwidth`` of 5.5 in
and all read from one file -- the ``results.json`` written by
``analyse_pspo_four_ablations.py`` -- so the tables, the figure and the
analysis report can never disagree:

``main``
    Final unshielded reward and normalised reward AUC per environment and
    initialiser (mean $\\pm$ s.e. over seeds).
``paired``
    The two single-factor paired contrasts -- entropy (``no_entropy`` minus
    control) and margin (``ce_only`` minus ``no_entropy``) -- plus the
    composite ``ce_only`` minus control, with Holm-corrected exact p-values.
``base``
    Initialiser diagnostics for the twelve shared base policies, which is the
    evidence that the ``ce_only`` arm starts from degenerate bases.

Safety never appears: ``exact_shield_alignment`` and
``empirical_evaluation_safety`` are exactly 1.0 in all 18 cells, so it is
stated once in the caption instead of spending six identical columns on it.

Each emitted snippet is a self-contained ``{\\footnotesize ...}`` group with
its own ``\\tabcolsep``, so dropping it into a ``table`` environment needs no
extra sizing commands. Requires ``booktabs`` and ``multirow``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

DEFAULT_RESULTS = (
    REPO
    / "artifacts/ablation_studies/pspo_four_ablations/analysis"
    / "policy_initialisation/results.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO / "projects/safe_policy_optimisation/figures/pspo_policy_initialisation"
)

# Row order is the canonical environment order used by every other figure.
ENVIRONMENTS = (
    "media_streaming",
    "colour_bomb",
    "colour_bomb_v2",
    "bridge_crossing",
    "bridge_crossing_v2",
    "mini_pacman",
)

# Long names are what the other tables use; the short forms exist because six
# environments plus six numeric columns do not fit 5.5 in otherwise.
ENV_LABELS = {
    "media_streaming": "Media Streaming",
    "colour_bomb": "Colour Bomb v1",
    "colour_bomb_v2": "Colour Bomb v2",
    "bridge_crossing": "Bridge Crossing v1",
    "bridge_crossing_v2": "Bridge Crossing v2",
    "mini_pacman": "MiniPacman",
}
ENV_LABELS_SHORT = {
    "media_streaming": "Media Stream.",
    "colour_bomb": "Colour Bomb v1",
    "colour_bomb_v2": "Colour Bomb v2",
    "bridge_crossing": "Bridge Cross.\\ v1",
    "bridge_crossing_v2": "Bridge Cross.\\ v2",
    "mini_pacman": "MiniPacman",
}

# control is the canonical PSPO initialiser: margin loss weight 1, safe-action
# entropy weight 1, all-safe target margin 2.
VARIANTS = ("control", "no_entropy", "ce_only")
VARIANT_LABELS = {
    "control": "PSPO",
    "no_entropy": "PSPO w/o entropy",
    "ce_only": "PSPO w/ CE-only init.",
}
VARIANT_LABELS_SHORT = {
    "control": "PSPO",
    "no_entropy": "\\shortstack{PSPO\\\\$-$Ent.}",
    "ce_only": "\\shortstack{PSPO\\\\$-$Ent.\\\\$-$Margin}",
}

COMPARISONS = (
    ("entropy", "\\shortstack{PSPO$-$Ent.\\\\vs. PSPO}"),
    (
        "margin_conditional_on_no_entropy",
        "\\shortstack{PSPO$-$Ent.$-$Margin\\\\vs. PSPO$-$Ent.}",
    ),
    ("ce_only_vs_control", "\\shortstack{PSPO$-$Ent.$-$Margin\\\\vs. PSPO}"),
)

ENDPOINTS = (
    ("final_unshielded_reward", "Final reward"),
    ("normalized_reward_auc", "Reward AUC"),
)

HOLM_ALPHA = 0.05


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--decimals", type=int, default=2, help="Printed decimal places (default: 2)."
    )
    return parser.parse_args(argv)


def load_results(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing analysis results: {path}")
    return json.loads(path.read_text())


def summary_index(results: dict) -> dict[tuple[str, str, str], dict]:
    """(variant, environment, endpoint) -> summary record."""
    return {
        (row["variant"], row["environment"], row["endpoint"]): row
        for row in results["summary"]
    }


def paired_index(results: dict) -> dict[tuple[str, str, str], dict]:
    """(comparison, environment, endpoint) -> paired record."""
    return {
        (row["comparison"], row["environment"], row["endpoint"]): row
        for row in results["paired"]
    }


def seed_count(summary: dict[tuple[str, str, str], dict]) -> int:
    counts = {
        int(row["complete_seeds"])
        for key, row in summary.items()
        if key[2] == "final_unshielded_reward"
    }
    if len(counts) != 1:
        raise ValueError(f"ragged seed counts across cells: {sorted(counts)}")
    return counts.pop()


def base_policy_diagnostics(results: dict) -> dict[tuple[str, str], dict]:
    """(variant, environment) -> the shared base policy's initialiser metrics.

    One base policy is trained per (variant, environment) and reused by every
    seed, so the per-seed rows repeat it; disagreement means the seeds did not
    actually share a base and the caller should not aggregate them.
    """
    fields = (
        "initialization_epochs",
        "initialization_final_min_all_margin",
        "initialization_final_safe_entropy",
    )
    collected: dict[tuple[str, str], dict] = {}
    for row in results["per_seed"]:
        key = (row["variant"], row["environment"])
        values = {field: row.get(field) for field in fields}
        previous = collected.setdefault(key, values)
        if previous != values:
            raise ValueError(f"seeds of {key} disagree on the shared base policy")
    return collected


def format_value(value: float, decimals: int) -> str:
    """Render a math-mode number token, normalising negative zero."""
    text = f"{value:.{decimals}f}"
    if float(text) == 0.0:
        text = f"{0.0:.{decimals}f}"
    return text


def format_mean_error(mean: float, error: float, decimals: int, *, bold: bool) -> str:
    body = (
        f"${format_value(mean, decimals)}"
        f"\\,{{\\scriptstyle\\pm\\,{error:.{decimals}f}}}$"
    )
    # ``\\textbf`` alone does not affect mathematics. ``\\boldmath`` is
    # available in the LaTeX kernel, keeping the snippet free of an amsmath
    # dependency while making the selected cell genuinely bold.
    return f"{{\\boldmath {body}}}" if bold else body


def format_p_value(p_value: float) -> str:
    """Holm-corrected p, printed at the resolution the test can support."""
    if not math.isfinite(p_value):
        return "--"
    if p_value < 0.001:
        return "$<$.001"
    return f"{p_value:.3f}".lstrip("0")


def emit(lines: list[str], tabcolsep: str) -> str:
    body = "\n".join(lines)
    return f"{{\\footnotesize\n\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n{body}\n}}\n"


def build_main_table(
    summary: dict[tuple[str, str, str], dict], *, decimals: int, seeds: int
) -> str:
    header = [
        "% Policy-initialisation ablation: outcome per environment and initialiser.",
        f"% Cells are the across-seed mean $\\pm$ 1 standard error, n={seeds} seeds.",
        "% Bold marks the best initialiser per environment and endpoint (ties included).",
        "% exact_shield_alignment and empirical evaluation safety are 1.00 in all",
        "% 18 cells, so safety is stated in the caption rather than tabulated.",
        "% Requires \\usepackage{booktabs}.",
    ]
    lines = [
        "\\begin{tabular}{l rrr rrr}",
        "\\toprule",
        " & \\multicolumn{3}{c}{Final unshielded reward}"
        " & \\multicolumn{3}{c}{Normalised reward AUC} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "Environment"
        + "".join(f" & {VARIANT_LABELS_SHORT[variant]}" for variant in VARIANTS) * 2
        + " \\\\",
        "\\midrule",
    ]
    for environment in ENVIRONMENTS:
        cells: list[str] = []
        for endpoint, _ in ENDPOINTS:
            records = [
                summary[(variant, environment, endpoint)] for variant in VARIANTS
            ]
            best = max(round(record["mean"], decimals) for record in records)
            cells.extend(
                format_mean_error(
                    record["mean"],
                    record["standard_error"],
                    decimals,
                    bold=round(record["mean"], decimals) == best,
                )
                for record in records
            )
        lines.append(
            f"{ENV_LABELS_SHORT[environment]} & " + " & ".join(cells) + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    # This is the widest table.  A 2.5 pt column gap keeps its natural width
    # below ICLR's 5.5 in text block without shrinking the type.
    return "\n".join(header) + "\n" + emit(lines, "2.5pt")


def build_paired_table(
    paired: dict[tuple[str, str, str], dict], *, decimals: int, seeds: int
) -> str:
    header = [
        "% Policy-initialisation ablation: paired per-seed differences.",
        "% Each cell is the mean paired difference (treatment minus reference)",
        "% with the Holm-corrected exact sign-randomisation p-value in brackets;",
        f"% n={seeds} paired seeds, Holm correction is across the six environments",
        "% within a comparison and endpoint. Bold marks Holm p < 0.05.",
        "% The first two columns are sequential contrasts; the final column",
        "% is their composite and not a single-factor ablation (the CE-init arm",
        "% also switches the Rashomon multi-label mode from all-safe to any-safe).",
        "% Requires \\usepackage{booktabs} and \\usepackage{multirow}.",
    ]
    lines = [
        "\\begin{tabular}{ll rrr}",
        "\\toprule",
        "Endpoint & Environment"
        + "".join(f" & {label}" for _, label in COMPARISONS)
        + " \\\\",
        "\\midrule",
    ]
    for position, (endpoint, endpoint_label) in enumerate(ENDPOINTS):
        for index, environment in enumerate(ENVIRONMENTS):
            first = (
                f"\\multirow{{{len(ENVIRONMENTS)}}}{{*}}{{{endpoint_label}}}"
                if index == 0
                else ""
            )
            cells = []
            for comparison, _ in COMPARISONS:
                record = paired[(comparison, environment, endpoint)]
                difference = f"${format_value(record['mean_difference'], decimals)}$"
                holm = record["holm_p_value"]
                cell = f"{difference} [{format_p_value(holm)}]"
                if math.isfinite(holm) and holm < HOLM_ALPHA:
                    cell = f"{{\\boldmath \\textbf{{{cell}}}}}"
                cells.append(cell)
            lines.append(
                f"{first} & {ENV_LABELS_SHORT[environment]} & "
                + " & ".join(cells)
                + " \\\\"
            )
        lines.append("\\midrule" if position < len(ENDPOINTS) - 1 else "\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(header) + "\n" + emit(lines, "4pt")


def build_base_table(diagnostics: dict[tuple[str, str], dict]) -> str:
    header = [
        "% Initialiser diagnostics for the twelve shared base policies (one per",
        "% variant and environment, reused by all seeds of that cell).",
        "% Epochs is the number of initialiser epochs run before the stopping",
        "% criterion fired; margin is the minimum safe-action logit margin under",
        "% that variant's criterion (all-safe for Full and $-$Entropy, any-safe",
        "% for CE only); entropy is the minimum normalised safe-action entropy.",
        "% CE only stops at any-safe margin > 0, which is equivalent to 100%",
        "% allowed-action accuracy and therefore carries no slack: three of its",
        "% six bases stop within four epochs and media_streaming stops at epoch 0,",
        "% i.e. at random initialisation. Read its reward deficit accordingly.",
        "% Requires \\usepackage{booktabs}.",
    ]
    lines = [
        "\\begin{tabular}{l rrr rrr rrr}",
        "\\toprule",
        " & \\multicolumn{3}{c}{Epochs}"
        " & \\multicolumn{3}{c}{Min.\\ margin}"
        " & \\multicolumn{3}{c}{Min.\\ safe entropy} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
        "Environment"
        + "".join(f" & {VARIANT_LABELS_SHORT[variant]}" for variant in VARIANTS) * 3
        + " \\\\",
        "\\midrule",
    ]
    for environment in ENVIRONMENTS:
        records = [diagnostics[(variant, environment)] for variant in VARIANTS]
        cells = [f"{int(record['initialization_epochs'])}" for record in records]
        cells += [
            f"${format_value(float(record['initialization_final_min_all_margin']), 2)}$"
            for record in records
        ]
        cells += [
            f"${format_value(float(record['initialization_final_safe_entropy']), 2)}$"
            for record in records
        ]
        lines.append(
            f"{ENV_LABELS_SHORT[environment]} & " + " & ".join(cells) + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(header) + "\n" + emit(lines, "3.5pt")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = load_results(args.results)
    summary = summary_index(results)
    paired = paired_index(results)
    seeds = seed_count(summary)

    tables = {
        "policy_initialisation_main_table.tex": build_main_table(
            summary, decimals=args.decimals, seeds=seeds
        ),
        "policy_initialisation_paired_table.tex": build_paired_table(
            paired, decimals=args.decimals, seeds=seeds
        ),
        "policy_initialisation_base_table.tex": build_base_table(
            base_policy_diagnostics(results)
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = args.output_dir / name
        path.write_text(table, encoding="utf-8")
        print(table)
        print(f"Wrote {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
