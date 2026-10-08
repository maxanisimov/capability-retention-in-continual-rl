"""Compare PSPO with the RL baselines under the extended training budgets.

The standard sweeps trained Bridge Crossing v2 for 200k steps and MiniPacman for
500k. The extended-budget runs re-train both PSPO and the RL baselines for
substantially longer, which answers a different question than the standard
tables in ``docs/pspo_reward_improvement_results.md``: whether PSPO's
reward gap to the reward-competitive baselines is a budget artefact.

Outputs a per-seed CSV, a cross-seed summary CSV, and a Markdown report into
``docs/extended_budget_comparison``.

Environments with no completed extended-budget baseline sweep are reported with
their PSPO rows only, and the missing methods are listed explicitly in the
report rather than silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "projects/safe_policy_optimisation/artifacts/paper_2503_07671/runs"
DEFAULT_OUTPUT_DIR = (
    REPO / "projects/safe_policy_optimisation/docs/extended_budget_comparison"
)

# (label, stage subdirectory, key inside metrics.json; None means the file itself)
BASELINE_METHODS = [
    ("PPO", "ppo_policy", None),
    ("PPO-Lagrangian", "ppo_lagrangian", "ppo_lagrangian"),
    ("PPO-PID-Lagrangian", "ppo_lagrangian", "ppo_pid_lagrangian"),
    ("CPO", "cpo", "cpo"),
    ("PPO-Shield", "ppo_shield", "shielded"),
    ("PPO-Shield-Nominal", "ppo_shield", "nominal"),
]


@dataclass(frozen=True)
class Block:
    """One (environment, budget setting, method family) group of seed runs."""

    environment: str
    setting: str  # "standard" or "extended"
    budget: int
    root: Path
    kind: str  # "baselines" or "pspo"
    method: str = "PSPO"
    note: str = ""


@dataclass
class EnvSpec:
    label: str
    blocks: list[Block] = field(default_factory=list)


ENVIRONMENTS = [
    EnvSpec(
        "Bridge Crossing v2",
        [
            Block(
                "Bridge Crossing v2", "standard", 400_000,
                RUNS / "pspo_reward_pilot_bcv2_m2_reuse_i200_f1_t400k/two_hidden/bridge_crossing_v2",
                "pspo", note="retained PSPO (margin 2, 200 iterations)",
            ),
            Block(
                "Bridge Crossing v2", "standard", 200_000,
                REPO / "outputs/_sweeps_2hidden_bridge_crossing_v2_baselines_only/bridge_crossing_v2",
                "baselines",
            ),
            Block(
                "Bridge Crossing v2", "extended", 1_600_000,
                RUNS / "pspo_adaptive_bridge_v2_safe_entropy_w1_min0p95_freq1_t1600k_reuse_base/two_hidden/bridge_crossing_v2",
                "pspo", note="safe-entropy base, frequency 1, reused base policy",
            ),
            Block(
                "Bridge Crossing v2", "extended", 1_000_000,
                RUNS / "rl_baselines_extended_budgets/bridge_crossing_v2",
                "baselines",
            ),
        ],
    ),
    EnvSpec(
        "MiniPacman",
        [
            Block(
                "MiniPacman", "standard", 500_000,
                RUNS / "pspo_adaptive_mini_pacman_safe_entropy_w1_min0p95_freq100/two_hidden/mini_pacman",
                "pspo", note="matched control for the extended run",
            ),
            Block(
                "MiniPacman", "standard", 500_000,
                REPO / "outputs/_sweeps_2hidden_minipacman_baselines_only/paper_2503_07671_minipacman",
                "baselines",
            ),
            Block(
                "MiniPacman", "extended", 2_000_000,
                RUNS / "pspo_adaptive_mini_pacman_safe_entropy_w1_min0p95_freq100_t2000k_reuse_base/two_hidden/mini_pacman",
                "pspo", note="safe-entropy base, frequency 100, reused base policy",
            ),
            Block(
                "MiniPacman", "extended", 2_000_000,
                RUNS / "rl_baselines_extended_budgets/mini_pacman",
                "baselines",
            ),
        ],
    ),
]


def _seed_number(path: Path) -> int:
    suffix = path.name.removeprefix("seed")
    return int(suffix) if suffix.isdigit() else 10**9


def _seed_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted((p for p in root.glob("seed*") if p.is_dir()), key=_seed_number)


def _load(seed_dir: Path, stage: str, algorithm: str | None) -> dict[str, Any] | None:
    path = seed_dir / stage / "metrics.json" if stage != "." else seed_dir / "metrics.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    payload = data if algorithm is None else data.get(algorithm)
    if not isinstance(payload, dict) or "reward" not in payload or "safety" not in payload:
        return None
    return payload


def _row(block: Block, method: str, seed: int, payload: dict[str, Any], source: Path) -> dict[str, Any]:
    return {
        "environment": block.environment,
        "setting": block.setting,
        "budget_timesteps": block.budget,
        "method": method,
        "seed": seed,
        "eval_episodes": payload.get("eval_episodes"),
        "total_reward": payload["reward"]["mean_total_reward"],
        "safety_rate": payload["safety"]["safety_rate"],
        "success_rate": payload.get("success", {}).get("success_rate"),
        "violation_count": payload["safety"].get("violation_count"),
        "source_metrics_json": str(source),
    }


def collect_per_seed() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (per-seed rows, missing-block records)."""
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for env in ENVIRONMENTS:
        for block in env.blocks:
            seed_dirs = _seed_dirs(block.root)
            if not seed_dirs:
                missing.append(
                    {
                        "environment": block.environment,
                        "setting": block.setting,
                        "kind": block.kind,
                        "budget_timesteps": block.budget,
                        "root": str(block.root),
                        "reason": "no seed directories found",
                    }
                )
                continue
            specs = (
                [(block.method, ".", None)]
                if block.kind == "pspo"
                else BASELINE_METHODS
            )
            found_any = False
            for method, stage, algorithm in specs:
                for seed_dir in seed_dirs:
                    payload = _load(seed_dir, stage, algorithm)
                    if payload is None:
                        continue
                    found_any = True
                    source = (
                        seed_dir / "metrics.json" if stage == "." else seed_dir / stage / "metrics.json"
                    )
                    rows.append(_row(block, method, _seed_number(seed_dir), payload, source))
            if not found_any:
                missing.append(
                    {
                        "environment": block.environment,
                        "setting": block.setting,
                        "kind": block.kind,
                        "budget_timesteps": block.budget,
                        "root": str(block.root),
                        "reason": "seed directories exist but hold no readable metrics.json",
                    }
                )
    return rows, missing


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _two_sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 2.0 * statistics.stdev(values) / math.sqrt(len(values))


def aggregate(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    order: list[tuple[str, str, str]] = []
    for row in per_seed:
        key = (row["environment"], row["setting"], row["method"])
        if key not in grouped:
            order.append(key)
        grouped.setdefault(key, []).append(row)

    summary = []
    for key in order:
        group = grouped[key]
        rewards = [float(r["total_reward"]) for r in group]
        safety = [float(r["safety_rate"]) for r in group]
        success = [float(r["success_rate"]) for r in group if r["success_rate"] is not None]
        summary.append(
            {
                "environment": key[0],
                "setting": key[1],
                "method": key[2],
                "budget_timesteps": group[0]["budget_timesteps"],
                "seed_count": len(group),
                "eval_episodes": group[0]["eval_episodes"],
                "total_reward_mean": _mean(rewards),
                "total_reward_2sem": _two_sem(rewards),
                "safety_rate_mean": _mean(safety),
                "safety_rate_2sem": _two_sem(safety),
                "success_rate_mean": _mean(success),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(mean: float, err: float) -> str:
    if math.isnan(mean):
        return "—"
    return f"{mean:.3f} +/- {err:.3f}"


def write_markdown(path: Path, summary: list[dict[str, Any]], missing: list[dict[str, Any]]) -> None:
    lines = [
        "# PSPO vs RL baselines under extended training budgets",
        "",
        f"Generated {datetime.now(timezone.utc).date().isoformat()} by",
        "`scripts/compare_extended_budget_pspo_vs_baselines.py`.",
        "",
        "All rewards and safety rates are means over training seeds of a fresh,",
        "unshielded, deterministic evaluation (100 episodes per seed). Uncertainty is",
        "two standard errors of the seed mean, not a confidence interval. `PPO-Shield`",
        "is the one method evaluated with its runtime shield attached.",
        "",
    ]

    for env in ENVIRONMENTS:
        env_rows = [r for r in summary if r["environment"] == env.label]
        if not env_rows:
            continue
        lines += [
            f"## {env.label}",
            "",
            "| Setting | Budget (steps) | Method | Reward (2SE) | Safety (2SE) | Seeds |",
            "|---|---:|---|---:|---:|---:|",
        ]
        for setting in ("standard", "extended"):
            for row in [r for r in env_rows if r["setting"] == setting]:
                lines.append(
                    "| {setting} | {budget:,} | {method} | {reward} | {safety} | {seeds} |".format(
                        setting=row["setting"],
                        budget=int(row["budget_timesteps"]),
                        method=row["method"],
                        reward=_fmt(row["total_reward_mean"], row["total_reward_2sem"]),
                        safety=_fmt(row["safety_rate_mean"], row["safety_rate_2sem"]),
                        seeds=row["seed_count"],
                    )
                )
        lines.append("")

    if missing:
        lines += ["## Missing result sets", ""]
        for entry in missing:
            lines.append(
                f"- {entry['environment']} / {entry['setting']} / {entry['kind']} "
                f"({int(entry['budget_timesteps']):,} steps): {entry['reason']} "
                f"(`{Path(entry['root']).relative_to(REPO)}`)"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_seed, missing = collect_per_seed()
    summary = aggregate(per_seed)

    _write_csv(args.output_dir / "extended_budget_per_seed.csv", per_seed)
    _write_csv(args.output_dir / "extended_budget_summary.csv", summary)
    write_markdown(args.output_dir / "extended_budget_comparison.md", summary, missing)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "blocks": [
                    {
                        "environment": b.environment,
                        "setting": b.setting,
                        "kind": b.kind,
                        "budget_timesteps": b.budget,
                        "root": str(b.root),
                        "exists": b.root.is_dir(),
                        "note": b.note,
                    }
                    for env in ENVIRONMENTS
                    for b in env.blocks
                ],
                "missing": missing,
                "summary": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote extended-budget comparison to {args.output_dir}")
    for entry in missing:
        print(f"  MISSING: {entry['environment']} {entry['setting']} {entry['kind']} -> {entry['reason']}")


if __name__ == "__main__":
    main()
