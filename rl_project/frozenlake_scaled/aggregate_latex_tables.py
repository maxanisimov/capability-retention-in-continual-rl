"""Aggregate multiple layout metrics CSVs into one LaTeX table.

Supported modes:
- row-wise (default): layouts appear in the first column via \multirow
- column-wise: layouts appear in header groups via \multicolumn
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


POLICY_ORDER = [
    "source",
    "downstream_unconstrained",
    "downstream_ewc",
    "downstream_rashomon",
]

POLICY_RENAME = {
    "source": "Source",
    "downstream_unconstrained": "Unconstrained",
    "downstream_ewc": "EWC",
    "downstream_rashomon": "Rashomon",
}


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def _format_reward(mean: float, std: float, *, bold_mean: bool) -> str:
    mean_text = f"{mean:.2f}"
    std_text = f"{std:.2f}"
    if bold_mean:
        return rf"$\mathbf{{{mean_text}}} \pm {std_text}$"
    return rf"${mean_text} \pm {std_text}$"


def _ordered_policies_for(policy_names: set[str]) -> list[str]:
    ordered = [policy for policy in POLICY_ORDER if policy in policy_names]
    ordered.extend(sorted(policy for policy in policy_names if policy not in POLICY_ORDER))
    return ordered


def _resolve_layout_csv_paths(
    *,
    outputs_root: Path,
    layouts: list[str] | None,
    csv_name: str,
) -> list[Path]:
    if layouts is not None and len(layouts) > 0:
        paths = [outputs_root / layout / csv_name for layout in layouts]
    else:
        paths = sorted(p for p in outputs_root.glob(f"*/{csv_name}") if p.is_file())

    if not paths:
        raise FileNotFoundError(
            f"No CSVs found. Looked for '{csv_name}' under {outputs_root}",
        )

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected CSV file(s): {missing}")
    return paths


def _read_layout_csv(csv_path: Path) -> tuple[str, int, dict[str, dict[str, float]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV has no rows: {csv_path}")

    layout_names = {str(row["layout"]) for row in rows}
    if len(layout_names) != 1:
        raise ValueError(
            f"Expected one layout value in {csv_path}, found: {sorted(layout_names)}",
        )
    layout = next(iter(layout_names))

    seeds = {int(row["num_seeds"]) for row in rows}
    if len(seeds) != 1:
        raise ValueError(
            f"Expected one num_seeds value in {csv_path}, found: {sorted(seeds)}",
        )
    num_seeds = next(iter(seeds))

    policy_map: dict[str, dict[str, float]] = {}
    for row in rows:
        policy = str(row["policy"])
        if policy not in POLICY_RENAME:
            continue
        policy_map[policy] = {
            "source_mean": float(row["source_total_reward_mean"]),
            "source_std": float(row["source_total_reward_std"]),
            "downstream_mean": float(row["downstream_total_reward_mean"]),
            "downstream_std": float(row["downstream_total_reward_std"]),
        }

    if not policy_map:
        raise ValueError(
            "No recognized policies found in "
            f"{csv_path}. Expected one of {sorted(POLICY_RENAME)}",
        )

    return layout, num_seeds, policy_map


def _build_latex_row_wise(
    *,
    layout_order: list[str],
    layout_to_policy: dict[str, dict[str, dict[str, float]]],
    caption: str,
) -> str:
    lines: list[str] = [
        r"% Requires \usepackage{multirow}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Layout & Policy & Source & Downstream \\",
        r"\midrule",
    ]

    for layout_idx, layout in enumerate(layout_order):
        policy_map = layout_to_policy[layout]
        ordered_policies = _ordered_policies_for(set(policy_map.keys()))
        if not ordered_policies:
            continue

        max_source = max(policy_map[p]["source_mean"] for p in ordered_policies)
        max_downstream = max(policy_map[p]["downstream_mean"] for p in ordered_policies)
        span = len(ordered_policies)

        first_row = True
        for policy in ordered_policies:
            metrics = policy_map[policy]
            layout_cell = rf"\multirow{{{span}}}{{*}}{{{_escape_latex(layout)}}}" if first_row else ""
            policy_cell = POLICY_RENAME.get(policy, _escape_latex(policy))
            source_cell = _format_reward(
                metrics["source_mean"],
                metrics["source_std"],
                bold_mean=abs(metrics["source_mean"] - max_source) <= 1e-12,
            )
            downstream_cell = _format_reward(
                metrics["downstream_mean"],
                metrics["downstream_std"],
                bold_mean=abs(metrics["downstream_mean"] - max_downstream) <= 1e-12,
            )
            lines.append(f"{layout_cell} & {policy_cell} & {source_cell} & {downstream_cell} \\\\")
            first_row = False

        if layout_idx != len(layout_order) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            r"\label{tab:source_downstream_multi_layout}",
            r"\end{table}",
        ],
    )
    return "\n".join(lines)


def _build_latex_column_wise(
    *,
    layout_order: list[str],
    layout_to_policy: dict[str, dict[str, dict[str, float]]],
    caption: str,
) -> str:
    discovered_policies = {policy for layout in layout_order for policy in layout_to_policy[layout]}
    ordered_policies = _ordered_policies_for(discovered_policies)

    if not ordered_policies:
        raise ValueError("No policies available to render.")

    col_spec = "l" + ("cc" * len(layout_order))
    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    header_group_cells = ["Policy"]
    for layout in layout_order:
        header_group_cells.append(rf"\multicolumn{{2}}{{c}}{{{_escape_latex(layout)}}}")
    lines.append(" & ".join(header_group_cells) + r" \\")

    header_metric_cells = [""]
    for _ in layout_order:
        header_metric_cells.extend(["Source", "Downstream"])
    lines.append(" & ".join(header_metric_cells) + r" \\")

    cmidrules: list[str] = []
    col_start = 2
    for _ in layout_order:
        cmidrules.append(rf"\cmidrule(lr){{{col_start}-{col_start + 1}}}")
        col_start += 2
    lines.append(" ".join(cmidrules))
    lines.append(r"\midrule")

    max_source_by_layout: dict[str, float] = {}
    max_downstream_by_layout: dict[str, float] = {}
    for layout in layout_order:
        policy_map = layout_to_policy[layout]
        source_vals = [metrics["source_mean"] for metrics in policy_map.values()]
        downstream_vals = [metrics["downstream_mean"] for metrics in policy_map.values()]
        max_source_by_layout[layout] = max(source_vals)
        max_downstream_by_layout[layout] = max(downstream_vals)

    for policy in ordered_policies:
        row_cells = [POLICY_RENAME.get(policy, _escape_latex(policy))]
        for layout in layout_order:
            metrics = layout_to_policy[layout].get(policy)
            if metrics is None:
                row_cells.extend([r"--", r"--"])
                continue

            source_cell = _format_reward(
                metrics["source_mean"],
                metrics["source_std"],
                bold_mean=abs(metrics["source_mean"] - max_source_by_layout[layout]) <= 1e-12,
            )
            downstream_cell = _format_reward(
                metrics["downstream_mean"],
                metrics["downstream_std"],
                bold_mean=abs(metrics["downstream_mean"] - max_downstream_by_layout[layout]) <= 1e-12,
            )
            row_cells.extend([source_cell, downstream_cell])
        lines.append(" & ".join(row_cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            r"\label{tab:source_downstream_multi_layout}",
            r"\end{table}",
        ],
    )
    return "\n".join(lines)


def build_latex_table(csv_paths: list[Path], *, mode: str = "row-wise") -> str:
    layout_order: list[str] = []
    layout_to_policy: dict[str, dict[str, dict[str, float]]] = {}
    layout_to_num_seeds: dict[str, int] = {}

    for csv_path in csv_paths:
        layout, num_seeds, policy_map = _read_layout_csv(csv_path)
        if layout in layout_to_policy:
            raise ValueError(
                f"Duplicate layout '{layout}' encountered. "
                f"Each layout should appear once. Offending file: {csv_path}",
            )
        layout_order.append(layout)
        layout_to_policy[layout] = policy_map
        layout_to_num_seeds[layout] = num_seeds

    unique_seeds = sorted(set(layout_to_num_seeds.values()))
    if len(unique_seeds) == 1:
        caption = (
            "Source and downstream performance across layouts "
            rf"(mean $\pm$ std over {unique_seeds[0]} seeds)."
        )
    else:
        caption = (
            "Source and downstream performance across layouts "
            r"(mean $\pm$ std; number of seeds may vary by layout)."
        )

    if mode == "row-wise":
        return _build_latex_row_wise(
            layout_order=layout_order,
            layout_to_policy=layout_to_policy,
            caption=caption,
        )
    if mode == "column-wise":
        return _build_latex_column_wise(
            layout_order=layout_order,
            layout_to_policy=layout_to_policy,
            caption=caption,
        )
    raise ValueError(f"Unsupported mode: {mode}. Expected 'row-wise' or 'column-wise'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-layout aggregate_layout_metrics.csv files into one "
            "LaTeX table (row-wise with multirow or column-wise with multicolumn)."
        ),
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Root directory containing layout folders.",
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        default=None,
        help=(
            "Optional explicit layout order to include. If omitted, all layouts "
            "with the target CSV under --outputs-root are included."
        ),
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="aggregate_layout_metrics.csv",
        help="Per-layout CSV filename to read inside each layout directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output .tex path. If omitted, writes to "
            "outputs/aggregate_layout_metrics_all_layouts.tex"
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["row-wise", "column-wise"],
        default="row-wise",
        help=(
            "Aggregation layout mode: "
            "'row-wise' (default) puts layouts in first column with \\multirow; "
            "'column-wise' puts layouts in header with \\multicolumn."
        ),
    )
    args = parser.parse_args()

    csv_paths = _resolve_layout_csv_paths(
        outputs_root=args.outputs_root,
        layouts=args.layouts,
        csv_name=args.csv_name,
    )
    latex = build_latex_table(csv_paths, mode=args.mode)

    output_path = args.output
    if output_path is None:
        output_path = args.outputs_root / "aggregate_layout_metrics_all_layouts.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex + "\n", encoding="utf-8")

    print(latex)
    print(f"\nWrote aggregated LaTeX table: {output_path}")


if __name__ == "__main__":
    main()
