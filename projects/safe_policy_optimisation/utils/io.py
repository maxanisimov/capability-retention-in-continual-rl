"""Canonical result-IO helpers for stage scripts.

This is the single home for writing run artifacts. It re-exports the
``EpisodeMetrics``-based writers used by the safe-RL baselines (defined in
:mod:`...utils.safe_rl`) and adds the dict-record helpers used by the
single-policy stages (plain PPO, shielded PPO), which previously each carried a
private copy.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from projects.safe_policy_optimisation.utils.safe_rl import (  # noqa: F401
    episode_rows,
    training_episode_rows,
    write_episode_csv,
    write_json,
    write_training_episode_csv,
)

__all__ = [
    "write_json",
    "episode_rows",
    "training_episode_rows",
    "write_episode_csv",
    "write_training_episode_csv",
    "record_rows",
    "record_training_rows",
    "write_record_csv",
    "RECORD_CSV_FIELDNAMES",
]

# Columns written by the single-policy stages' per-episode CSVs.
RECORD_CSV_FIELDNAMES = [
    "algorithm",
    "episode",
    "reward",
    "cost",
    "length",
    "violated",
    "unsafe_state_visit_count",
    "safe_trajectory",
]


def record_rows(records: Iterable[dict[str, Any]], *, algorithm: str) -> list[dict[str, Any]]:
    """Tag dict-record evaluation episodes with the algorithm name."""

    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        row["algorithm"] = algorithm
        rows.append(row)
    return rows


def record_training_rows(
    records: Iterable[dict[str, Any]], *, algorithm: str
) -> list[dict[str, Any]]:
    """Tag training-exploration episodes with a cumulative end-timestep and name."""

    rows: list[dict[str, Any]] = []
    end_timestep = 0
    for record in records:
        row = dict(record)
        end_timestep += int(row["length"])
        row["end_timestep"] = end_timestep
        row["algorithm"] = algorithm
        rows.append(row)
    return rows


def write_record_csv(
    path: Path, rows: Iterable[dict[str, Any]], *, include_end_timestep: bool = False
) -> None:
    """Write single-policy per-episode rows with the standard column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RECORD_CSV_FIELDNAMES)
    if include_end_timestep:
        fieldnames.insert(2, "end_timestep")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
