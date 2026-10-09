#!/usr/bin/env python3
"""Bar figure: total reward and safety rate on structured slippery FrozenLake.

Reads the unified final-policy rollout aggregate written by
``evaluate_frozenlake_final_policies.py`` -- every bar therefore comes from the
same harness, the same 200 episode seeds and the same success definition, so
the panels compare policies rather than evaluation protocols.

PPO-Shield appears twice: once as deployed with its runtime shield, once with
the shield removed. They are the same trained weights, so they take the paper's
purple/plum colour pair. All bars and legend swatches use solid fills.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paper_style import METHOD_COLORS  # noqa: E402
from _aamas_reward_safety import BarMethod, BarPanel, compact_figure, save_compact_figure

DEFAULT_AGGREGATE = (
    REPO
    / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
    / "final_policy_rollouts_frozenlake128"
    / "aggregate.json"
)
DEFAULT_STEM = (
    REPO / "projects/safe_policy_optimisation/figures/aamas" / "frozenlake128_reward_safety_bars"
)


@dataclass(frozen=True)
class Series:
    key: str
    label: str
    color: str


# Short labels form a shared two-row legend at native single-column size.
SERIES = (
    Series("pspo", "PSPO", METHOD_COLORS["pspo"]),
    Series("ppo_shield_shielded", "PPO-Shield\nshield on", METHOD_COLORS["ppo_shield"]),
    Series(
        "ppo_shield", "PPO-Shield\nshield off",
        METHOD_COLORS["ppo_shield_nominal"],
    ),
    Series("cpo", "CPO", METHOD_COLORS["cpo"]),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--stem", type=Path, default=DEFAULT_STEM)
    parser.add_argument("--transpose", action="store_true",
                        help="Stack reward above safety, with a compact side legend.")
    parser.add_argument(
        "--title", default="FrozenLake 128×128",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    methods = json.loads(args.aggregate.read_text())["methods"]
    missing = [series.key for series in SERIES if series.key not in methods]
    if missing:
        raise SystemExit(f"Aggregate {args.aggregate} is missing: {missing}")

    rewards = [float(methods[s.key]["mean_total_reward"]) for s in SERIES]
    reward_errors = [float(methods[s.key]["mean_total_reward_2se"] or 0.0) for s in SERIES]
    safeties = [float(methods[s.key]["safety_rate"]) for s in SERIES]
    safety_errors = [float(methods[s.key]["safety_rate_2se"] or 0.0) for s in SERIES]

    specs = [BarMethod(s.key, s.label, s.color) for s in SERIES]
    panels = [BarPanel(args.title, rewards, reward_errors, safeties, safety_errors)]
    fig = compact_figure(panels, specs, single_column=True, title=args.title,
                         transpose=args.transpose)
    stem = args.stem
    if args.transpose and stem == DEFAULT_STEM:
        stem = stem.with_name(stem.name + "_transposed")
    save_compact_figure(
        fig, stem, panels=panels, methods=specs, se_multiplier=2.0,
        caption=("FrozenLake final-policy total reward and safety rate "
                 "(mean $\\pm$ two standard errors across seeds), evaluated "
                 "with the same rollout protocol for all methods. "
                 "Shield-on and shield-off use the same PPO-Shield weights."),
    )
    print(f"Wrote {stem}.pdf, .png, .csv and .tex")
    for series, reward, reward_error, safety, safety_error in zip(
        SERIES, rewards, reward_errors, safeties, safety_errors
    ):
        flat = series.label.replace("\n", " ")
        print(
            f"  {flat:30s} R {reward:7.3f} +/- {reward_error:5.3f}"
            f"   S {safety:5.3f} +/- {safety_error:5.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
