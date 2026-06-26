"""Entrypoint: aggregate FrozenLake safety run_summary metrics across seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from projects.safe_crl.pipelines.safety_retention.frozenlake.core.analysis.aggregate_layout_metrics import (
    DEFAULT_CSV_NAME,
    DEFAULT_METRICS,
    DEFAULT_TEX_NAME,
    METHOD_ORDER,
    METRIC_SPECS,
    aggregate_metrics,
    write_csv,
    write_latex_table,
)
from projects.safe_crl.pipelines.safety_retention.frozenlake.core.config import LAYOUT_NAME
from projects.safe_crl.pipelines.safety_retention.frozenlake.core.paths import default_outputs_root, layout_run_root
from projects.safe_crl.pipelines.safety_retention.frozenlake.core.training_common import (
    RL_CHOICES,
    resolve_deterministic,
    validate_deterministic,
    validate_rl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate FrozenLake safety run_summary metrics across seeds and "
            "export a CSV plus a LaTeX table."
        ),
    )
    parser.add_argument(
        "--pipeline",
        "--layout",
        dest="layout",
        default=LAYOUT_NAME,
        help=f"Pipeline/layout name under outputs-root. Default: {LAYOUT_NAME}.",
    )
    parser.add_argument("--rl", type=str, default="ppo", choices=RL_CHOICES)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override dynamics regime. Defaults to the pipeline's task definition when omitted.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=default_outputs_root(),
        help="Root containing <pipeline>/<rl>_<dynamics>/seed_*/<method>/run_summary.yaml.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=f"Optional method directory names to aggregate. Default: {' '.join(METHOD_ORDER)}.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(METRIC_SPECS.keys()),
        default=list(DEFAULT_METRICS),
        help="Metrics to aggregate (safety and performance, source and downstream).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional seed filter.")
    parser.add_argument("--precision", type=int, default=3, help="Decimal places for mean/std cells. Default: 3.")
    parser.add_argument("--output-csv", type=Path, default=None, help=f"CSV path. Default: <run_root>/{DEFAULT_CSV_NAME}.")
    parser.add_argument("--output-tex", type=Path, default=None, help=f"LaTeX path. Default: <run_root>/{DEFAULT_TEX_NAME}.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_rl(args.rl)
    args.deterministic = resolve_deterministic(args.layout, args.deterministic)
    validate_deterministic(args.deterministic)
    metric_keys = list(args.metrics)
    layout = str(args.layout)
    outputs_root = Path(args.outputs_root)
    run_root = layout_run_root(outputs_root, layout, args.rl, args.deterministic)
    output_csv = args.output_csv or (run_root / DEFAULT_CSV_NAME)
    output_tex = args.output_tex or (run_root / DEFAULT_TEX_NAME)

    rows = aggregate_metrics(
        outputs_root=outputs_root,
        layout=layout,
        rl=args.rl,
        deterministic=args.deterministic,
        metric_keys=metric_keys,
        methods=args.methods,
        seeds=set(args.seeds) if args.seeds is not None else None,
    )
    write_csv(rows, output_csv, metric_keys=metric_keys, precision=int(args.precision))
    write_latex_table(
        rows,
        output_tex,
        metric_keys=metric_keys,
        precision=int(args.precision),
        layout=layout,
    )

    print(f"Wrote aggregate metrics CSV: {output_csv}")
    print(f"Wrote aggregate metrics LaTeX table: {output_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
