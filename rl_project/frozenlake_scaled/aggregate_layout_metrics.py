"""Aggregate per-policy run_summary metrics across seeds for one layout."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import pstdev

import yaml


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in YAML file: {path}")
    return data


def _seed_from_name(name: str) -> int | None:
    if not name.startswith("seed_"):
        return None
    suffix = name.removeprefix("seed_")
    if not suffix.isdigit():
        return None
    return int(suffix)


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(as_float):
        return None
    return as_float


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(pstdev(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan run_summary.yaml files for one layout and aggregate source/downstream "
            "total reward statistics per policy into a CSV table."
        ),
    )
    parser.add_argument("--layout", type=str, required=True, help="Layout name, e.g. diagonal_10x10.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Outputs root directory containing layout/seed_* subdirectories.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional explicit CSV output path. Default: outputs/<layout>/aggregate_layout_metrics.csv",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=None,
        help=(
            "Optional policy directory filter (e.g. source downstream_unconstrained "
            "downstream_ewc downstream_rashomon)."
        ),
    )
    args = parser.parse_args()

    layout_dir = args.outputs_root / args.layout
    if not layout_dir.exists():
        raise FileNotFoundError(f"Layout outputs directory not found: {layout_dir}")

    # policy -> seed -> {"source": float, "downstream": float}
    policy_seed_rewards: dict[str, dict[int, dict[str, float]]] = {}

    total_found = 0
    for seed_dir in sorted(layout_dir.glob("seed_*"), key=lambda p: p.name):
        if not seed_dir.is_dir():
            continue
        seed = _seed_from_name(seed_dir.name)
        if seed is None:
            continue

        for summary_path in sorted(seed_dir.glob("*/run_summary.yaml")):
            total_found += 1
            policy = summary_path.parent.name
            if args.policies is not None and policy not in args.policies:
                continue

            summary = _load_yaml(summary_path)
            source_reward = _safe_float(summary.get("source_mean_reward"))
            downstream_reward = _safe_float(summary.get("downstream_mean_reward"))
            if source_reward is None or downstream_reward is None:
                print(
                    f"[skip] Missing source_mean_reward/downstream_mean_reward in {summary_path}",
                )
                continue

            policy_seed_rewards.setdefault(policy, {})
            if seed in policy_seed_rewards[policy]:
                print(
                    f"[warn] Duplicate summary for policy={policy} seed={seed}; overwriting with {summary_path}",
                )
            policy_seed_rewards[policy][seed] = {
                "source": source_reward,
                "downstream": downstream_reward,
            }

    if total_found == 0:
        raise FileNotFoundError(f"No run_summary.yaml files found under {layout_dir}/seed_*/")

    policies = sorted(policy_seed_rewards.keys())
    if args.policies is not None:
        # Preserve requested order but only keep policies that had usable summaries.
        policies = [policy for policy in args.policies if policy in policy_seed_rewards]

    if not policies:
        raise RuntimeError(
            f"Found run summaries under {layout_dir}, but no usable entries matched the selection.",
        )

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = layout_dir / "aggregate_layout_metrics.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "layout",
        "policy",
        "num_seeds",
        "source_total_reward_mean",
        "source_total_reward_std",
        "downstream_total_reward_mean",
        "downstream_total_reward_std",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for policy in policies:
            seed_reward_map = policy_seed_rewards[policy]
            source_values = [seed_reward_map[s]["source"] for s in sorted(seed_reward_map)]
            downstream_values = [seed_reward_map[s]["downstream"] for s in sorted(seed_reward_map)]
            writer.writerow(
                {
                    "layout": args.layout,
                    "policy": policy,
                    "num_seeds": len(seed_reward_map),
                    "source_total_reward_mean": f"{_mean(source_values):.6f}",
                    "source_total_reward_std": f"{_std(source_values):.6f}",
                    "downstream_total_reward_mean": f"{_mean(downstream_values):.6f}",
                    "downstream_total_reward_std": f"{_std(downstream_values):.6f}",
                },
            )

    print(f"Wrote aggregate layout metrics CSV: {output_csv}")
    for policy in policies:
        seed_reward_map = policy_seed_rewards[policy]
        source_values = [seed_reward_map[s]["source"] for s in sorted(seed_reward_map)]
        downstream_values = [seed_reward_map[s]["downstream"] for s in sorted(seed_reward_map)]
        print(
            f"{policy}: "
            f"source mean={_mean(source_values):.4f} std={_std(source_values):.4f} | "
            f"downstream mean={_mean(downstream_values):.4f} std={_std(downstream_values):.4f} | "
            f"seeds={len(seed_reward_map)}",
        )


if __name__ == "__main__":
    main()
