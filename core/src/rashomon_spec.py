"""Specification types for Rashomon-set computation: accuracy requirements and results.

Replaces the scattered soft_min/hard_min/soft_metric/soft_temperature/aggregation
parameters with a single `target_accuracy`: the differentiable surrogate (a softmax
margin), its dataset-level aggregation (an exact order statistic), and the softmax
temperature it depends on are all constructed and calibrated automatically from this
one number. See `interval_utils._calibrate_temperature`/`_order_statistic_select`.
"""

from __future__ import annotations

import dataclasses

from abstract_gradient_training.bounded_models import IntervalBoundedModel


@dataclasses.dataclass
class AccuracyRequirement:
    """
    Specifies the minimum certified accuracy for a Rashomon-set search.

    `target_accuracy` may be a single float shared across every group, or a dict keyed
    by group id (the values produced by `compute_rashomon_set`'s `group_by` function)
    for per-group targets. Both the differentiable soft surrogate (a margin) and the
    strict hard certificate are aggregated across the dataset (or group) via the exact
    order statistic this target implies - `target_accuracy=1.0` requires every sample
    to clear the constraint (today's old strict "min" behavior); lower values tolerate
    a proportionally larger fraction of failing samples.
    """

    target_accuracy: float | dict[int, float]

    def resolve(self, group: int) -> float:
        """Look up the target accuracy for a given group id, falling back to a shared
        float if `target_accuracy` isn't a dict."""
        return (
            self.target_accuracy[group]
            if isinstance(self.target_accuracy, dict)
            else self.target_accuracy
        )


@dataclasses.dataclass
class RashomonCertificate:
    """A certified accuracy result for one group at one checkpoint."""

    group: int | None  # None for the single global group (no group_by given)
    min_soft_acc: float
    min_hard_acc: float


@dataclasses.dataclass
class RashomonResult:
    """Result of a Rashomon-set search."""

    bounded_models: list[IntervalBoundedModel]  # optimization-time boxes, one per checkpoint
    certificates: list[list[RashomonCertificate]]  # per checkpoint, per group
    temperatures: dict[int | None, float]  # per-group calibrated (or caller-supplied) softmax temperature
