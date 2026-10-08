#!/usr/bin/env python3
"""Emit a compact LaTeX table of final greedy-policy reward and safety rate.

Numbers come from each run's terminal ``metrics.json`` - the 100-episode
deterministic evaluation of the trained policy - not from the 20-episode
periodic checkpoints used for learning curves. Cells are the across-seed
mean +/- two standard errors; the best cell per environment and metric is
bold, with ties at the printed precision all marked.

Environment roots are imported from the learning-curve script so both
artefacts always describe the same runs. ``--include-shield-off`` adds
PPO-Shield's weights evaluated without the runtime shield and lists methods in
the bar-chart order of plot_final_policy_reward_safety_bars.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_extended_budget_learning_curves import (  # noqa: E402
    ENVIRONMENTS,
    EXPECTED_SEEDS,
    EnvironmentSpec,
)

DEFAULT_OUTPUT = (
    REPO
    / "projects/safe_policy_optimisation/figures"
    / "pspo_vs_rl_baselines_learning_curves_all_envs"
    / "final_policy_reward_safety_table.tex"
)
SHIELD_OFF_DEFAULT_OUTPUT = (
    REPO
    / "projects/safe_policy_optimisation/figures/aamas"
    / "final_policy_reward_safety_table_shield_off.tex"
)

ENV_LABELS = {
    "media_streaming": "Media Streaming",
    "colour_bomb": "Colour Bomb v1",
    "colour_bomb_v2": "Colour Bomb v2",
    "bridge_crossing": "Bridge Crossing v1",
    "bridge_crossing_v2": "Bridge Crossing v2",
    "mini_pacman": "MiniPacman",
}


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    relative_path: str
    # Keys walked into metrics.json to reach the evaluated variant.
    section: tuple[str, ...]
    adaptive: bool = False


# PPO-Shield is reported as deployed (shielded); this matches the variant
# drawn in the learning-curve figures.
METHODS = (
    MethodSpec("ppo_policy", "PPO", "ppo_policy/metrics.json", ()),
    MethodSpec(
        "ppo_lagrangian",
        "PPO-Lagrangian",
        "ppo_lagrangian/metrics.json",
        ("ppo_lagrangian",),
    ),
    MethodSpec(
        "ppo_pid_lagrangian",
        "PPO-PID-Lagrangian",
        "ppo_lagrangian/metrics.json",
        ("ppo_pid_lagrangian",),
    ),
    MethodSpec("cpo", "CPO", "cpo/metrics.json", ("cpo",)),
    MethodSpec("pspo", "PSPO", "metrics.json", (), adaptive=True),
    MethodSpec("ppo_shield", "PPO-Shield", "ppo_shield/metrics.json", ("shielded",)),
)

# The same trained PPO-Shield weights, evaluated without the runtime shield.
SHIELD_OFF_SPEC = MethodSpec(
    "ppo_shield_nominal", "PPO Shield (shield off)", "ppo_shield/metrics.json", ("nominal",)
)


def bar_order(include_shield_off: bool = False) -> tuple[MethodSpec, ...]:
    """Return METHODS in bar-chart order, with PSPO fixed in the final position.

    With ``include_shield_off`` the shield-off evaluation sits between
    PPO-Shield and PSPO.
    """
    specs = [spec for spec in METHODS if spec.key != "pspo"]
    if include_shield_off:
        shield = next(index for index, spec in enumerate(specs) if spec.key == "ppo_shield")
        specs.insert(shield + 1, SHIELD_OFF_SPEC)
    return tuple(specs) + tuple(spec for spec in METHODS if spec.key == "pspo")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .tex path (default depends on --include-shield-off).",
    )
    parser.add_argument(
        "--decimals", type=int, default=2, help="Printed decimal places (default: 2)."
    )
    parser.add_argument(
        "--ci-multiplier",
        type=float,
        default=2.0,
        help="Multiplier on the standard error (default: 2).",
    )
    parser.add_argument(
        "--include-shield-off",
        action="store_true",
        help="Add PPO Shield (shield off) and use the bar-chart method order.",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = SHIELD_OFF_DEFAULT_OUTPUT if args.include_shield_off else DEFAULT_OUTPUT
    return args


def _descend(payload: dict, section: tuple[str, ...], path: Path) -> dict:
    node = payload
    for key in section:
        if key not in node:
            raise KeyError(f"missing section {'/'.join(section)} in {path}")
        node = node[key]
    return node


def read_seed_values(
    spec: MethodSpec, environment: EnvironmentSpec
) -> tuple[list[float], list[float], set[int]]:
    """Return per-seed (reward, safety) and the episode counts behind them."""
    root = environment.adaptive_root if spec.adaptive else environment.baseline_root
    rewards: list[float] = []
    safeties: list[float] = []
    episode_counts: set[int] = set()
    for seed in EXPECTED_SEEDS:
        path = root / f"seed{seed}" / spec.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        node = _descend(json.loads(path.read_text()), spec.section, path)
        rewards.append(float(node["reward"]["mean_total_reward"]))
        safeties.append(float(node["safety"]["safety_rate"]))
        episode_counts.add(int(node["eval_episodes"]))
    return rewards, safeties, episode_counts


def mean_and_error(values: list[float], ci_multiplier: float) -> tuple[float, float]:
    count = len(values)
    mean = sum(values) / count
    if count < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return mean, ci_multiplier * math.sqrt(variance / count)


def format_cell(mean: float, error: float, decimals: int, best: bool) -> str:
    if round(mean, decimals) == 0:
        mean = 0.0  # print 0.00, not -0.00, for tiny negative means
    body = f"{mean:.{decimals}f} \\pm {error:.{decimals}f}"
    return f"$\\mathbf{{{body}}}$" if best else f"${body}$"


def build_table(
    results: dict[str, dict[str, tuple[float, float]]],
    *,
    decimals: int,
    ci_multiplier: float,
    methods: tuple[MethodSpec, ...] = METHODS,
) -> str:
    lines = [
        "% Final greedy-policy evaluation (100 deterministic episodes per seed).",
        f"% Cells are mean $\\pm$ {ci_multiplier:g} standard errors over "
        f"{len(EXPECTED_SEEDS)} seeds.",
        "% Bold marks the best value per environment and metric (ties included).",
    ]
    if SHIELD_OFF_SPEC in methods:
        lines.append(
            f"% {SHIELD_OFF_SPEC.label} is the trained PPO-Shield policy evaluated "
            "without its runtime shield."
        )
    lines += [
        "% Requires \\usepackage{booktabs} and \\usepackage{multirow}.",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Environment & Method & Total reward & Safety rate \\\\",
        "\\midrule",
    ]
    environment_keys = [environment.key for environment in ENVIRONMENTS]
    for position, environment_key in enumerate(environment_keys):
        per_method = results[environment_key]
        best_reward = max(
            round(per_method[spec.key][0][0], decimals) for spec in methods
        )
        best_safety = max(
            round(per_method[spec.key][1][0], decimals) for spec in methods
        )
        label = ENV_LABELS[environment_key]
        for index, spec in enumerate(methods):
            (reward_mean, reward_error), (safety_mean, safety_error) = per_method[
                spec.key
            ]
            first = f"\\multirow{{{len(methods)}}}{{*}}{{{label}}}" if index == 0 else ""
            lines.append(
                f"{first} & {spec.label} & "
                + format_cell(
                    reward_mean,
                    reward_error,
                    decimals,
                    round(reward_mean, decimals) == best_reward,
                )
                + " & "
                + format_cell(
                    safety_mean,
                    safety_error,
                    decimals,
                    round(safety_mean, decimals) == best_safety,
                )
                + " \\\\"
            )
        lines.append(
            "\\midrule" if position < len(environment_keys) - 1 else "\\bottomrule"
        )
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    methods = bar_order(include_shield_off=True) if args.include_shield_off else METHODS
    results: dict[str, dict[str, tuple[float, float]]] = {}
    all_episode_counts: set[int] = set()
    for environment in ENVIRONMENTS:
        per_method: dict[str, tuple[float, float]] = {}
        for spec in methods:
            rewards, safeties, episode_counts = read_seed_values(spec, environment)
            all_episode_counts |= episode_counts
            per_method[spec.key] = (
                mean_and_error(rewards, args.ci_multiplier),
                mean_and_error(safeties, args.ci_multiplier),
            )
        results[environment.key] = per_method

    table = build_table(
        results, decimals=args.decimals, ci_multiplier=args.ci_multiplier, methods=methods
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    print(table)
    print(f"Wrote {args.output}")
    print(f"Evaluation episodes per seed: {sorted(all_episode_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
