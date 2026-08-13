"""Standardised final-evaluation metrics for trained policies.

Every training stage writes a ``metrics.json`` summarising the post-training
evaluation along the three axes the experiments care about:

- **success**  - fraction of evaluation episodes whose return clears the
  task's ``success_reward_threshold``;
- **reward**   - mean (and min/max) total episode return;
- **safety**   - safe-trajectory rate plus cost-budget violation counts.

Keeping one schema across stages makes runs directly comparable.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

METRICS_FILENAME = "metrics.json"

# Success semantics differ by task type. Reach tasks (a terminal goal to attain)
# score success as "return cleared the reward threshold". Avoid-only tasks have no
# reach goal - the objective is to survive the horizon without entering an unsafe
# zone - so their success flag is "the episode visited no unsafe state" (identical
# to the safe-trajectory flag). ``media_streaming`` is the avoid-only task here.
SUCCESS_MODE_REWARD_THRESHOLD = "reward_threshold"
SUCCESS_MODE_SAFE_TRAJECTORY = "safe_trajectory"


def success_mode_for_env(env_id: str | None) -> str:
    """Return the success definition to use for a given environment id.

    Avoid-only tasks (media streaming) use ``safe_trajectory``; every other task
    keeps the reward-threshold definition.
    """

    return (
        SUCCESS_MODE_SAFE_TRAJECTORY
        if "MediaStreaming" in str(env_id or "")
        else SUCCESS_MODE_REWARD_THRESHOLD
    )


def _as_row(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return record
    if dataclasses.is_dataclass(record):
        return dataclasses.asdict(record)
    raise TypeError(f"Cannot interpret evaluation record of type {type(record).__name__}.")


def summarise_evaluation(
    records: Iterable[Any],
    *,
    success_reward_threshold: float,
    cost_limit: float | None = None,
    algorithm: str | None = None,
    success_mode: str = SUCCESS_MODE_REWARD_THRESHOLD,
) -> dict[str, Any]:
    """Build the standardised success / reward / safety metrics for one policy.

    ``records`` is an iterable of per-episode evaluation results, each either a
    mapping or a dataclass with ``reward``/``cost``/``violated`` (and optionally
    ``safe_trajectory``) fields - i.e. the rows the stages already collect.

    ``success_mode`` selects the success definition: ``reward_threshold`` (return
    beats ``success_reward_threshold``, for reach tasks) or ``safe_trajectory``
    (episode visited no unsafe state, for avoid-only tasks such as media streaming).
    """

    rows = [_as_row(r) for r in records]
    n = len(rows)
    mean = (lambda xs: float(sum(xs) / n) if n else 0.0)

    rewards = [float(r["reward"]) for r in rows]
    violated_flags = [1 if r.get("violated") else 0 for r in rows]
    costs = [float(r.get("cost", 0.0)) for r in rows]
    safe_flags = [
        1 if r.get("safe_trajectory", float(r.get("cost", 0.0)) <= 0.0) else 0 for r in rows
    ]
    if success_mode == SUCCESS_MODE_SAFE_TRAJECTORY:
        # Avoid-only task: success == the episode never entered an unsafe zone.
        success_flags = list(safe_flags)
    else:
        success_flags = [1 if float(r["reward"]) > success_reward_threshold else 0 for r in rows]

    metrics: dict[str, Any] = {
        "eval_episodes": n,
        "success": {
            "success_rate": mean(success_flags),
            "success_count": int(sum(success_flags)),
            "success_mode": success_mode,
            "success_reward_threshold": float(success_reward_threshold),
        },
        "reward": {
            "mean_total_reward": mean(rewards),
            "min_total_reward": float(min(rewards)) if rows else 0.0,
            "max_total_reward": float(max(rewards)) if rows else 0.0,
        },
        "safety": {
            "safety_rate": mean(safe_flags),
            "safe_trajectory_count": int(sum(safe_flags)),
            "violation_count": int(sum(violated_flags)),
            "violation_percentage": 100.0 * sum(violated_flags) / n if n else 0.0,
            "mean_episode_cost": mean(costs),
            "cost_limit": None if cost_limit is None else float(cost_limit),
        },
    }
    if algorithm is not None:
        metrics = {"algorithm": algorithm, **metrics}
    return metrics


# --------------------------------------------------------------------------- #
# Cross-seed aggregation
# --------------------------------------------------------------------------- #

def _flatten_numeric(data: dict[str, Any], prefix: str) -> dict[str, float]:
    """Flatten nested numeric leaves to ``prefix.a.b -> float`` (skip bools/None/str)."""

    flat: dict[str, float] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(_flatten_numeric(value, path))
        elif isinstance(value, bool) or value is None:
            continue
        elif isinstance(value, (int, float)):
            flat[path] = float(value)
    return flat


def load_run_metrics(run_dir: Path) -> dict[str, float]:
    """Read every ``<run_dir>/<stage>/metrics.json`` into a flat metric map.

    Single-policy stages contribute keys like ``shielded_policy.success.success_rate``;
    baseline stages (whose ``metrics.json`` is keyed by algorithm) contribute keys
    like ``ppo_lagrangian/cpo.safety.safety_rate``.
    """

    run_dir = Path(run_dir)
    out: dict[str, float] = {}
    if not run_dir.is_dir():
        return out
    for stage_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        metrics_file = stage_dir / METRICS_FILENAME
        if not metrics_file.exists():
            continue
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        if "success" in data:  # single-policy stage
            sections = {stage_dir.name: data}
        else:  # baseline stage: {algorithm: {success/reward/safety...}}
            sections = {
                f"{stage_dir.name}/{algo}": payload
                for algo, payload in data.items()
                if isinstance(payload, dict) and "success" in payload
            }
        for prefix, payload in sections.items():
            out.update(_flatten_numeric(payload, prefix))
    return out


def aggregate_seed_metrics(seed_to_run_dir: dict[int, Path]) -> dict[str, Any]:
    """Aggregate per-stage metrics across seeds into mean/std/n/min/max/per_seed."""

    per_seed = {seed: load_run_metrics(run_dir) for seed, run_dir in seed_to_run_dir.items()}
    all_keys = sorted({key for metrics in per_seed.values() for key in metrics})
    aggregated: dict[str, Any] = {}
    for key in all_keys:
        values_by_seed = {seed: m[key] for seed, m in per_seed.items() if key in m}
        values = list(values_by_seed.values())
        aggregated[key] = {
            "mean": statistics.fmean(values) if values else 0.0,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
            "per_seed": {str(seed): val for seed, val in sorted(values_by_seed.items())},
        }
    return {"seeds": sorted(seed_to_run_dir), "metrics": aggregated}


def seed_metrics_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten :func:`aggregate_seed_metrics` output to CSV-friendly rows."""

    rows: list[dict[str, Any]] = []
    for metric, stats in aggregate.get("metrics", {}).items():
        rows.append(
            {
                "metric": metric,
                "mean": stats["mean"],
                "std": stats["std"],
                "n": stats["n"],
                "min": stats["min"],
                "max": stats["max"],
            }
        )
    return rows
