"""Compatibility re-export: implementation now lives in experiments.utils.constrained_ppo."""

from experiments.utils.constrained_ppo import (
    ConstrainedPPOStats,
    LineSearchDecision,
    RashomonPayload,
    VerifiedMarginConstraint,
    allowed_action_accuracy,
    apply_safe_line_search,
    calibrate_margin_temperature,
    lagrangian_ppo_train,
    safe_line_search_ppo_train,
    validate_rashomon_payload,
)

__all__ = [
    "ConstrainedPPOStats",
    "LineSearchDecision",
    "RashomonPayload",
    "VerifiedMarginConstraint",
    "allowed_action_accuracy",
    "apply_safe_line_search",
    "calibrate_margin_temperature",
    "lagrangian_ppo_train",
    "safe_line_search_ppo_train",
    "validate_rashomon_payload",
]
