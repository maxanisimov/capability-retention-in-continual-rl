"""Convert aggregate layout metrics CSV files to NeurIPS-style LaTeX tables."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
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


def _read_csv_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for raw in reader:
            rows.append(
                {
                    "layout": str(raw["layout"]),
                    "policy": str(raw["policy"]),
                    "num_seeds": int(raw["num_seeds"]),
                    "source_mean": float(raw["source_total_reward_mean"]),
                    "source_std": float(raw["source_total_reward_std"]),
                    "downstream_mean": float(raw["downstream_total_reward_mean"]),
                    "downstream_std": float(raw["downstream_total_reward_std"]),
                },
            )
    return rows


def build_latex_table(csv_path: Path) -> str:
    rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"CSV file has no data rows: {csv_path}")

    seeds = sorted({row["num_seeds"] for row in rows})
    if len(seeds) != 1:
        raise ValueError(
            f"Expected a single num_seeds value across rows, found {seeds} in {csv_path}",
        )
    num_seeds = seeds[0]

    # environment -> policy -> row
    env_to_policy: dict[str, OrderedDict[str, dict]] = OrderedDict()
    for row in rows:
        policy = row["policy"]
        if policy not in POLICY_RENAME:
            continue
        env = row["layout"]
        if env not in env_to_policy:
            env_to_policy[env] = OrderedDict()
        env_to_policy[env][policy] = row

    if not env_to_policy:
        raise ValueError(
            "No rows matched known policies "
            f"{sorted(POLICY_RENAME)} in {csv_path}",
        )

    env_names = sorted(env_to_policy.keys())
    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Environment & Policy & Source Total Reward & Downstream Total Reward \\",
        r"\midrule",
    ]

    for env_idx, env in enumerate(env_names):
        policy_map = env_to_policy[env]
        ordered_rows = [policy_map[p] for p in POLICY_ORDER if p in policy_map]
        if not ordered_rows:
            continue

        max_source_mean = max(row["source_mean"] for row in ordered_rows)
        max_downstream_mean = max(row["downstream_mean"] for row in ordered_rows)

        first_row = True
        for row in ordered_rows:
            policy = row["policy"]
            env_cell = _escape_latex(env) if first_row else ""
            policy_cell = POLICY_RENAME[policy]
            source_cell = _format_reward(
                row["source_mean"],
                row["source_std"],
                bold_mean=abs(row["source_mean"] - max_source_mean) <= 1e-12,
            )
            downstream_cell = _format_reward(
                row["downstream_mean"],
                row["downstream_std"],
                bold_mean=abs(row["downstream_mean"] - max_downstream_mean) <= 1e-12,
            )
            lines.append(f"{env_cell} & {policy_cell} & {source_cell} & {downstream_cell} \\\\")
            first_row = False

        if env_idx != len(env_names) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{Source and downstream performance across environments (mean $\pm$ std over {num_seeds} seeds).}}",
            r"\label{tab:source_downstream_performance}",
            r"\end{table}",
        ],
    )
    return "\n".join(lines)


def _resolve_csv_path(*, csv: Path | None, layout: str | None, outputs_root: Path) -> Path:
    if csv is not None:
        return csv
    if layout is None:
        raise ValueError("Either --layout or --csv must be provided.")
    return outputs_root / layout / "aggregate_layout_metrics.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert aggregate metrics CSV to a LaTeX table.")
    parser.add_argument(
        "--layout",
        type=str,
        default=None,
        help=(
            "Layout name used to auto-resolve CSV as "
            "outputs/<layout>/aggregate_layout_metrics.csv."
        ),
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Root directory that contains layout result folders.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional explicit path to aggregate_layout_metrics.csv (overrides --layout auto-discovery).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save generated LaTeX code. Defaults to stdout.",
    )
    args = parser.parse_args()

    csv_path = _resolve_csv_path(csv=args.csv, layout=args.layout, outputs_root=args.outputs_root)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    latex = build_latex_table(csv_path)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(latex + "\n", encoding="utf-8")
    print(latex)


if __name__ == "__main__":
    main()
