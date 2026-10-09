"""Left-boundary region shield for discrete ``MountainCar-v0``."""

from __future__ import annotations

from typing import Any

import numpy as np

from continuous_state_shields.base import FallbackRule, SafetyShield
from continuous_state_shields.config import MountainCarShieldConfig


class MountainCarShield(SafetyShield):
    """Push right while the car is moving left near the physical left wall.

    A state is safety-critical exactly when its position is in
    ``[critical_min_position, critical_max_position]`` and its velocity is
    negative. Push right (action 2) is the only admissible action there. The
    unsafe event is contact with the physical wall at ``min_position``.
    """

    def __init__(
        self,
        config: MountainCarShieldConfig | None = None,
        *,
        fallback_rule: FallbackRule = "lowest",
    ) -> None:
        super().__init__(3, fallback_rule=fallback_rule)
        self.config = config or MountainCarShieldConfig()

    def predict_next_state(self, state: Any, action: int) -> np.ndarray:
        """Return the configured one-step ``(position, velocity)`` prediction."""
        observation = np.asarray(state, dtype=float)
        if observation.shape != (2,):
            raise ValueError(
                f"MountainCar state must have shape (2,); got {observation.shape}."
            )
        action = self._validate_action(action)
        x, velocity = observation
        cfg = self.config
        next_velocity = np.clip(
            velocity + cfg.force * (action - 1) - cfg.gravity * np.cos(3.0 * x),
            -cfg.max_speed,
            cfg.max_speed,
        )
        next_position = np.clip(x + next_velocity, cfg.min_position, cfg.max_position)
        return np.asarray([next_position, next_velocity], dtype=float)

    def _evaluate(self, state: np.ndarray) -> tuple[list[int], dict[str, Any]]:
        if state.shape != (2,):
            raise ValueError(
                f"MountainCar state must have shape (2,); got {state.shape}."
            )
        cfg = self.config
        position, velocity = map(float, state)
        at_left_boundary = position <= cfg.min_position + cfg.tolerance
        in_critical_position_interval = (
            # Gym stores observations as float32, so its clipped -1.2 wall is
            # observed as roughly -1.20000005. Treat physical wall contact as
            # being on the lower edge of the semantic critical interval.
            (at_left_boundary or position >= cfg.critical_min_position - cfg.tolerance)
            and position <= cfg.critical_max_position + cfg.tolerance
        )
        moving_left = velocity < 0.0
        shield_active = in_critical_position_interval and moving_left
        violated = ["left_boundary_contact"] if at_left_boundary else []
        if shield_active:
            safe = [cfg.push_right_action]
            region = (
                "unsafe_boundary_safety_critical"
                if at_left_boundary
                else "safety_critical"
            )
            reason = (
                "state is near the left boundary and moving left; push right is "
                "the only safe action"
            )
        else:
            safe = list(range(self.n_actions))
            region = "unsafe_boundary" if at_left_boundary else "safe"
            reason = (
                "the left boundary has already been reached"
                if at_left_boundary
                else "state is not moving left in the critical position interval"
            )

        return safe, {
            "violated_constraints": violated,
            "reason": reason,
            "region": region,
            "position": position,
            "velocity": velocity,
            "unsafe_event": "left_boundary_contact",
            "left_boundary_position": cfg.min_position,
            "safety_critical_region": [
                cfg.critical_min_position,
                cfg.critical_max_position,
            ],
            "safety_critical_velocity": "velocity < 0",
            "push_right_action": cfg.push_right_action,
        }
