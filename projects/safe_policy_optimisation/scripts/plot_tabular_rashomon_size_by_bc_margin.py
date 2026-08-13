#!/usr/bin/env python
"""Plot certified tabular Rashomon-box size against requested BC margin.

The plotted size matches ``core.src.interval_utils._bounded_model_width``:
the sum of ``upper - lower`` over every actor parameter.  Settings are only
compared within a sweep and at a fixed number of Rashomon optimisation
iterations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = (
    REPO
    / "projects/safe_policy_optimisation/results/pspo/precomputed"
    / "rashomon_size_by_bc_margin"
)
DEFAULT_SWEEPS = (
    (
        "Bridge Crossing v2",
        REPO / "outputs/_pspo_hparam/bridge_crossing_v2_tabular",
    ),
    (
        "MiniPacman",
        REPO / "outputs/_pspo_hparam_bc_all/tabular_mini_pacman",
    ),
)
SETTING_RE = re.compile(r"^iters_(?P<iters>\d+)__margin_(?P<margin>[^_]+)(?:__.*)?$")


@dataclass(frozen=True)
class Result:
    environment: str
    sweep_root: str
    setting_dir: str
    rashomon_iterations: int
    requested_bc_margin: float
    bc_margin_mode: str
    selected_certificate: float
    base_policy_reached_target: bool
    parameter_count: int
    nonzero_width_count: int
    total_parameter_width: float
    mean_parameter_width: float
    relative_total_width: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def decode_margin(token: str) -> float:
    return float(token.replace("p", "."))


def load_result(environment: str, sweep_root: Path, summary_path: Path) -> Result:
    setting_dir = summary_path.parent.parent
    match = SETTING_RE.fullmatch(setting_dir.name)
    if match is None:
        raise ValueError(f"Cannot parse setting name: {setting_dir.name}")

    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    bounds = torch.load(
        setting_dir / "set/rashomon_param_bounds.pt",
        map_location="cpu",
        weights_only=False,
    )
    widths = torch.cat(
        [
            (upper.detach().cpu() - lower.detach().cpu()).reshape(-1)
            for lower, upper in zip(bounds["param_bounds_l"], bounds["param_bounds_u"])
        ]
    )
    if bool((widths < -1e-7).any().item()):
        raise ValueError(f"Negative parameter width in {setting_dir}")
    widths = widths.clamp_min(0.0)

    rashomon = summary.get("rashomon", {})
    base_policy = summary.get("base_policy", {})
    return Result(
        environment=environment,
        sweep_root=str(sweep_root.relative_to(REPO)),
        setting_dir=str(setting_dir.relative_to(REPO)),
        rashomon_iterations=int(match.group("iters")),
        requested_bc_margin=decode_margin(match.group("margin")),
        bc_margin_mode=str(base_policy.get("bc_margin_mode", "any")),
        selected_certificate=float(rashomon.get("selected_certificate", float("nan"))),
        base_policy_reached_target=bool(base_policy.get("reached_target", False)),
        parameter_count=int(widths.numel()),
        nonzero_width_count=int((widths > 0).sum().item()),
        total_parameter_width=float(widths.sum().item()),
        mean_parameter_width=float(widths.mean().item()),
    )


def collect_results() -> list[Result]:
    results: list[Result] = []
    for environment, sweep_root in DEFAULT_SWEEPS:
        for summary_path in sorted(sweep_root.glob("precomputed/iters_*/set/summary.json")):
            results.append(load_result(environment, sweep_root, summary_path))

    grouped: dict[tuple[str, int], list[Result]] = {}
    for result in results:
        grouped.setdefault((result.environment, result.rashomon_iterations), []).append(result)

    comparable: list[Result] = []
    for group in grouped.values():
        certified = [row for row in group if row.selected_certificate >= 1.0]
        if len({row.requested_bc_margin for row in certified}) < 2:
            continue
        baseline = min(certified, key=lambda row: row.requested_bc_margin).total_parameter_width
        for row in certified:
            comparable.append(
                Result(
                    **{
                        **asdict(row),
                        "relative_total_width": row.total_parameter_width / baseline,
                    }
                )
            )
    return sorted(
        comparable,
        key=lambda row: (row.environment, row.rashomon_iterations, row.requested_bc_margin),
    )


def write_csv(results: list[Result], output_dir: Path) -> Path:
    path = output_dir / "tabular_rashomon_size_by_bc_margin.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in results)
    return path


def grouped_results(results: list[Result]) -> list[tuple[tuple[str, int], list[Result]]]:
    grouped: dict[tuple[str, int], list[Result]] = {}
    for result in results:
        grouped.setdefault((result.environment, result.rashomon_iterations), []).append(result)
    return sorted(grouped.items())


def save_absolute_plot(results: list[Result], output_dir: Path) -> tuple[Path, Path]:
    groups = grouped_results(results)
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 4.1), squeeze=False)
    for axis, ((environment, iterations), rows) in zip(axes[0], groups):
        margins = [row.requested_bc_margin for row in rows]
        widths = [row.total_parameter_width for row in rows]
        axis.plot(margins, widths, marker="o", linewidth=2.0, color="#276FBF")
        axis.set_xscale("log")
        axis.set_title(f"{environment}\n{iterations:,} Rashomon iterations")
        axis.set_xlabel("Requested BC margin")
        axis.set_ylabel(r"Rashomon box size  $\sum_i (u_i-l_i)$")
        axis.grid(True, alpha=0.25)
        for margin, width in zip(margins, widths):
            axis.annotate(
                f"{width:,.1f}",
                (margin, width),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    fig.suptitle("Certified tabular precomputed Rashomon-set size vs BC margin", fontsize=13)
    fig.tight_layout()
    png = output_dir / "tabular_rashomon_width_by_bc_margin.png"
    pdf = output_dir / "tabular_rashomon_width_by_bc_margin.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def save_relative_plot(results: list[Result], output_dir: Path) -> tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(6.8, 4.4))
    colors = ("#276FBF", "#D1495B", "#2A9D8F", "#7A5195")
    for color, ((environment, iterations), rows) in zip(colors, grouped_results(results)):
        axis.plot(
            [row.requested_bc_margin for row in rows],
            [row.relative_total_width for row in rows],
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"{environment} ({iterations:,} iterations)",
        )
    axis.set_xscale("log")
    axis.set_xlabel("Requested BC margin")
    axis.set_ylabel("Box size relative to smallest tested margin")
    axis.set_title("Relative growth of certified tabular Rashomon boxes")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    png = output_dir / "tabular_rashomon_width_relative_by_bc_margin.png"
    pdf = output_dir / "tabular_rashomon_width_relative_by_bc_margin.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = collect_results()
    if not results:
        raise SystemExit("No certified fixed-iteration tabular margin sweeps were found.")

    paths = [
        write_csv(results, output_dir),
        *save_absolute_plot(results, output_dir),
        *save_relative_plot(results, output_dir),
    ]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
