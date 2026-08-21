"""Collect two-hidden-layer baseline results into one canonical directory.

The raw two-hidden runs currently live in several sweep directories under
``outputs/_sweeps_2hidden_*_baselines_only``.  This script creates a compact,
paper-facing result directory containing:

* cross-seed reward/safety summary CSV;
* per-seed reward/safety CSV;
* a Markdown summary table;
* a provenance manifest linking every environment back to its raw sweep root.

Safe-RL baselines are included here:
PPO-Lagrangian, PPO-PID-Lagrangian, CPO, and shielded PPO where available.
Unshielded PPO and PPO-Shield-Nominal are also included as comparison
references. PSPO/Rashomon methods are deliberately excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = (
    REPO
    / "projects"
    / "safe_policy_optimisation"
    / "docs"
    / "two_hidden_safe_rl_baselines"
)


@dataclass(frozen=True)
class EnvironmentSpec:
    label: str
    env_id: str
    raw_root: Path


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    stage: str
    algorithm: str | None
    comparison_role: str


ENVIRONMENTS = [
    EnvironmentSpec(
        "Bridge Crossing v1",
        "bridge_crossing",
        REPO / "outputs/_sweeps_2hidden_bridge_crossing_baselines_only/bridge_crossing",
    ),
    EnvironmentSpec(
        "Bridge Crossing v2",
        "bridge_crossing_v2",
        REPO / "outputs/_sweeps_2hidden_bridge_crossing_v2_baselines_only/bridge_crossing_v2",
    ),
    EnvironmentSpec(
        "Colour Bomb v1",
        "colour_bomb",
        REPO / "outputs/_sweeps_2hidden_colour_bomb_baselines_only/colour_bomb",
    ),
    EnvironmentSpec(
        "Colour Bomb v2",
        "colour_bomb_v2",
        REPO / "outputs/_sweeps_2hidden_colour_bomb_v2_baselines_only/colour_bomb_v2",
    ),
    EnvironmentSpec(
        "Media Streaming",
        "media_streaming",
        REPO / "outputs/_sweeps_2hidden_media_streaming_baselines_only/media_streaming",
    ),
    EnvironmentSpec(
        "MiniPacman",
        "minipacman",
        REPO
        / "outputs/_sweeps_2hidden_minipacman_baselines_only/paper_2503_07671_minipacman",
    ),
]


METHODS = [
    MethodSpec("ppo_policy", "PPO", "ppo_policy", None, "reference_baseline"),
    MethodSpec(
        "ppo_lagrangian",
        "PPO-Lagrangian",
        "ppo_lagrangian",
        "ppo_lagrangian",
        "safe_rl_baseline",
    ),
    MethodSpec(
        "ppo_pid_lagrangian",
        "PPO-PID-Lagrangian",
        "ppo_lagrangian",
        "ppo_pid_lagrangian",
        "safe_rl_baseline",
    ),
    MethodSpec("cpo", "CPO", "cpo", "cpo", "safe_rl_baseline"),
    MethodSpec(
        "ppo_shield_shielded",
        "PPO-Shield",
        "ppo_shield",
        "shielded",
        "safe_rl_baseline",
    ),
    MethodSpec(
        "ppo_shield_nominal",
        "PPO-Shield-Nominal",
        "ppo_shield",
        "nominal",
        "reference_baseline",
    ),
]


def _seed_number(path: Path) -> int:
    suffix = path.name.removeprefix("seed")
    return int(suffix) if suffix.isdigit() else 10**9


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _sem(values: list[float]) -> float:
    return _std(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _load_method_metrics(seed_dir: Path, method: MethodSpec) -> dict[str, Any] | None:
    metrics_path = seed_dir / method.stage / "metrics.json"
    if not metrics_path.is_file():
        return None
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if method.algorithm is None:
        payload = data
    else:
        payload = data.get(method.algorithm)
    if not isinstance(payload, dict):
        return None
    if "reward" not in payload or "safety" not in payload:
        return None
    return payload


def collect_per_seed_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for env in ENVIRONMENTS:
        seed_dirs = sorted(
            [p for p in env.raw_root.glob("seed*") if p.is_dir()],
            key=_seed_number,
        )
        for seed_dir in seed_dirs:
            seed = _seed_number(seed_dir)
            for method in METHODS:
                payload = _load_method_metrics(seed_dir, method)
                if payload is None:
                    continue
                rows.append(
                    {
                        "environment": env.label,
                        "environment_id": env.env_id,
                        "method": method.label,
                        "method_key": method.key,
                        "comparison_role": method.comparison_role,
                        "seed": seed,
                        "eval_episodes": payload.get("eval_episodes"),
                        "total_reward": payload["reward"]["mean_total_reward"],
                        "safety_rate": payload["safety"]["safety_rate"],
                        "success_rate": payload.get("success", {}).get("success_rate"),
                        "violation_count": payload["safety"].get("violation_count"),
                        "mean_episode_cost": payload["safety"].get("mean_episode_cost"),
                        "source_metrics_json": str(seed_dir / method.stage / "metrics.json"),
                    }
                )
    return rows


def aggregate_rows(per_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_seed_rows:
        by_key.setdefault((row["environment"], row["method_key"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for env in ENVIRONMENTS:
        for method in METHODS:
            rows = by_key.get((env.label, method.key), [])
            rewards = [float(r["total_reward"]) for r in rows]
            safety = [float(r["safety_rate"]) for r in rows]
            success = [float(r["success_rate"]) for r in rows if r["success_rate"] is not None]
            eval_episodes = [
                float(r["eval_episodes"]) for r in rows if r["eval_episodes"] is not None
            ]
            summary.append(
                {
                    "environment": env.label,
                    "environment_id": env.env_id,
                    "method": method.label,
                    "method_key": method.key,
                    "comparison_role": method.comparison_role,
                    "seed_count": len(rows),
                    "seeds": " ".join(str(r["seed"]) for r in sorted(rows, key=lambda r: r["seed"])),
                    "eval_episodes_mean": _mean(eval_episodes),
                    "total_reward_mean": _mean(rewards),
                    "total_reward_std": _std(rewards),
                    "total_reward_sem": _sem(rewards),
                    "total_reward_min": min(rewards) if rewards else float("nan"),
                    "total_reward_max": max(rewards) if rewards else float("nan"),
                    "safety_rate_mean": _mean(safety),
                    "safety_rate_std": _std(safety),
                    "safety_rate_sem": _sem(safety),
                    "safety_rate_min": min(safety) if safety else float("nan"),
                    "safety_rate_max": max(safety) if safety else float("nan"),
                    "success_rate_mean": _mean(success),
                    "source_root": str(env.raw_root),
                }
            )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        return f"{value:.3f}"
    return str(value)


def write_markdown_summary(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Two-hidden-layer baseline results",
        "",
        "This table is generated from completed `metrics.json` files in the",
        "`outputs/_sweeps_2hidden_*_baselines_only` raw sweep directories.",
        "",
        "Included methods: PPO, PPO-Lagrangian, PPO-PID-Lagrangian, CPO, PPO-Shield,",
        "and PPO-Shield-Nominal where result files exist. PSPO/Rashomon runs are not",
        "included in this baseline summary.",
        "",
        "| Environment | Method | Role | n seeds | Total reward mean ± s.e. | Safety mean ± s.e. |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        if int(row["seed_count"]) == 0:
            reward = "—"
            safety = "—"
        else:
            reward = f'{_fmt(row["total_reward_mean"])} ± {_fmt(row["total_reward_sem"])}'
            safety = f'{_fmt(row["safety_rate_mean"])} ± {_fmt(row["safety_rate_sem"])}'
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["environment"]),
                    str(row["method"]),
                    str(row["comparison_role"]),
                    str(row["seed_count"]),
                    reward,
                    safety,
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(path: Path, output_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    sources = "\n".join(
        f"- `{env.label}`: `{env.raw_root.relative_to(REPO)}`" for env in ENVIRONMENTS
    )
    lines = [
        "# Two-hidden-layer baseline results",
        "",
        "This directory is the canonical paper-facing index for available",
        "two-hidden-layer baseline results. Safe-RL baselines are grouped together",
        "with PPO and PPO-Shield-Nominal reference baselines so the reward/safety",
        "tradeoff is visible in one place.",
        "",
        "Generated files:",
        "",
        "- `safe_rl_baseline_summary.csv`: cross-seed mean/std/standard-error for",
        "  total reward, safety rate, and success rate.",
        "- `safe_rl_baseline_summary.md`: compact human-readable table.",
        "- `safe_rl_baseline_per_seed.csv`: one row per environment/method/seed.",
        "- `manifest.json`: source sweep roots and generation metadata.",
        "",
        "Raw source sweep roots:",
        "",
        sources,
        "",
        "Result availability is determined from final `metrics.json` files. A method",
        "with `seed_count = 0` has no completed result in the current",
        "two-hidden sweep tree for that environment.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python3 projects/safe_policy_optimisation/scripts/collect_two_hidden_safe_rl_baselines.py",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, output_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "two_hidden",
        "output_dir": str(output_dir),
        "included_methods": [
            {
                "key": method.key,
                "label": method.label,
                "stage": method.stage,
                "algorithm": method.algorithm,
                "comparison_role": method.comparison_role,
            }
            for method in METHODS
        ],
        "excluded_methods": [
            "rashomon_policy",
            "rashomon_adaptive_policy",
        ],
        "source_roots": [
            {
                "environment": env.label,
                "environment_id": env.env_id,
                "raw_root": str(env.raw_root),
                "exists": env.raw_root.is_dir(),
            }
            for env in ENVIRONMENTS
        ],
        "summary": summary_rows,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write the collected result analytics. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows = collect_per_seed_rows()
    summary_rows = aggregate_rows(per_seed_rows)

    _write_csv(output_dir / "safe_rl_baseline_per_seed.csv", per_seed_rows)
    _write_csv(output_dir / "safe_rl_baseline_summary.csv", summary_rows)
    write_markdown_summary(output_dir / "safe_rl_baseline_summary.md", summary_rows)
    write_readme(output_dir / "README.md", output_dir, summary_rows)
    write_manifest(output_dir / "manifest.json", output_dir, summary_rows)

    print(f"Wrote two-hidden safe-RL baseline analytics to {output_dir}")


if __name__ == "__main__":
    main()
