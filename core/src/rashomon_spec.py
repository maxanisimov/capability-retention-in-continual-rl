"""Specification types for Rashomon-set computation: accuracy requirements and results.

Replaces the scattered min_acc_limit/min_soft_acc_limit/min_hard_acc_limit/
soft_acc_temperature/aggregation/multi_label_soft_metric parameters (and the
task_labels-positional-list-based per-task limits) with a single explicit
AccuracyRequirement, and the ad hoc tuple-of-lists return shape of
compute_rashomon_set with a uniformly-typed RashomonResult.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

from abstract_gradient_training.bounded_models import IntervalBoundedModel


@dataclasses.dataclass
class AccuracyRequirement:
    """
    Specifies the certified-accuracy constraint(s) for a Rashomon-set search.

    `soft_min`/`hard_min` may be a single float shared across every group, or a dict
    keyed by group id (the values produced by `compute_rashomon_set`'s `group_by`
    function) for per-group limits.

    Attributes:
        soft_min: Minimum soft (differentiable) accuracy used as the Lagrangian
            constraint during optimization.
        hard_min: Minimum hard (strict, certified) accuracy used as the strict
            constraint. Defaults to `soft_min` if None.
        soft_metric: Differentiable surrogate used for the soft constraint - either
            softmax probability mass on admissible/correct classes ("soft_accuracy")
            or a margin relative to the soundness threshold ("accuracy_margin").
        soft_temperature: Softmax temperature used by `soft_metric`.
        aggregation: How per-sample correctness is reduced to a single number within
            a group ("mean" or "min" - "min" is the worst-case sample in the group).
    """

    soft_min: float | dict[int, float]
    hard_min: float | dict[int, float] | None = None
    soft_metric: Literal["soft_accuracy", "accuracy_margin"] = "soft_accuracy"
    soft_temperature: float = 10.0
    aggregation: Literal["mean", "min"] = "mean"

    def resolve(self, group: int) -> tuple[float, float]:
        """
        Look up the (soft_min, hard_min) limits for a given group id, falling back to
        a shared float if `soft_min`/`hard_min` aren't dicts, and falling back to
        `soft_min` if `hard_min` is None.
        """
        soft = self.soft_min[group] if isinstance(self.soft_min, dict) else self.soft_min
        hard_source = self.hard_min if self.hard_min is not None else self.soft_min
        hard = hard_source[group] if isinstance(hard_source, dict) else hard_source
        return soft, hard


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
