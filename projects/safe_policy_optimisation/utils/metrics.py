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
from collections.abc import Iterable
from typing import Any

METRICS_FILENAME = "metrics.json"


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
) -> dict[str, Any]:
    """Build the standardised success / reward / safety metrics for one policy.

    ``records`` is an iterable of per-episode evaluation results, each either a
    mapping or a dataclass with ``reward``/``cost``/``violated`` (and optionally
    ``safe_trajectory``) fields - i.e. the rows the stages already collect.
    """

    rows = [_as_row(r) for r in records]
    n = len(rows)
    mean = (lambda xs: float(sum(xs) / n) if n else 0.0)

    rewards = [float(r["reward"]) for r in rows]
    success_flags = [1 if float(r["reward"]) > success_reward_threshold else 0 for r in rows]
    violated_flags = [1 if r.get("violated") else 0 for r in rows]
    costs = [float(r.get("cost", 0.0)) for r in rows]
    safe_flags = [
        1 if r.get("safe_trajectory", float(r.get("cost", 0.0)) <= 0.0) else 0 for r in rows
    ]

    metrics: dict[str, Any] = {
        "eval_episodes": n,
        "success": {
            "success_rate": mean(success_flags),
            "success_count": int(sum(success_flags)),
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
