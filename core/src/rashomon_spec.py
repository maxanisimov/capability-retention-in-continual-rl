"""Specification types for Rashomon-set computation: accuracy targets and results.

Replaces the scattered soft_min/hard_min/soft_metric/soft_temperature/aggregation
parameters with a single `target_accuracy`: the differentiable surrogate (a softmax
margin), its dataset-level aggregation (an exact order statistic), and the softmax
temperature it depends on are all constructed and calibrated automatically from this
one number. See `interval_utils._calibrate_temperature`/`_order_statistic_select`.

A target accuracy is just a plain `float | dict[int, float]` (see `AccuracyTarget`) -
a single float shared across every group, or a dict keyed by group id (the values
produced by `compute_rashomon_set`'s `group_by` function) for per-group targets. Use
`resolve_accuracy` to look up the value for a given group.
"""

from __future__ import annotations

import dataclasses

from abstract_gradient_training.bounded_models import IntervalBoundedModel

AccuracyTarget = float | dict[int, float]


def resolve_accuracy(target: AccuracyTarget, group: int | None) -> float:
    """Look up the target accuracy for a given group id, falling back to a shared
    float if `target` isn't a dict."""
    return target[group] if isinstance(target, dict) else target


@dataclasses.dataclass
class RashomonCertificate:
    """A certified accuracy result for one group at one checkpoint."""

    group: int | None  # None for the single global group (no group_by given)
    min_surrogate: float
    min_hard_acc: float


@dataclasses.dataclass
class RashomonResult:
    """Result of a Rashomon-set search."""

    bounded_models: list[IntervalBoundedModel]  # optimization-time boxes, one per checkpoint
    certificates: list[list[RashomonCertificate]]  # per checkpoint, per group
    temperatures: dict[int | None, float]  # per-group calibrated (or caller-supplied) softmax temperature
