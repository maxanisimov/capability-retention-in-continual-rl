"""Transparent heuristic safety shield for discrete ``LunarLander-v3``."""

from __future__ import annotations

from typing import Any

import numpy as np

from continuous_state_shields.base import FallbackRule, SafetyShield
from continuous_state_shields.config import (
    LunarLanderDescentShieldConfig,
    LunarLanderShieldConfig,
)


class LunarLanderShield(SafetyShield):
    """Combine descent, attitude, and horizontal heuristic action sets.

    This shield deliberately avoids cloning Box2D. Side engine 1 is modelled as
    positive horizontal thrust and negative angular acceleration; engine 3 is the
    reverse. Their horizontal effect is multiplied by ``cos(theta)``. Main-engine
    horizontal thrust is approximated by ``-sin(theta)``. An action is rejected by
    an active horizontal/attitude component only when its projected absolute error
    is worse than taking no action.

    Component sets are intersected. If empty, priority is: dangerous descent
    (return ``{2}``), attitude, then horizontal position. Thus this shield never
    returns an empty set.
    """

    def __init__(
        self,
        config: LunarLanderShieldConfig | None = None,
        *,
        fallback_rule: FallbackRule = "lowest",
    ) -> None:
        super().__init__(4, fallback_rule=fallback_rule)
        self.config = config or LunarLanderShieldConfig()

    def _horizontal_predictions(self, state: np.ndarray) -> dict[int, float]:
        x, _, velocity_x, _, theta, _, _, _ = state
        cfg = self.config
        effects = {
            0: 0.0,
            1: cfg.side_engine_horizontal_effect * np.cos(theta),
            2: -cfg.main_engine_horizontal_effect * np.sin(theta),
            3: -cfg.side_engine_horizontal_effect * np.cos(theta),
        }
        return {
            action: float(x + cfg.horizontal_lookahead * (velocity_x + effect))
            for action, effect in effects.items()
        }

    def _attitude_predictions(self, state: np.ndarray) -> dict[int, float]:
        theta, angular_velocity = float(state[4]), float(state[5])
        cfg = self.config
        effects = {
            0: 0.0,
            1: -cfg.side_engine_angular_effect,
            2: 0.0,
            3: cfg.side_engine_angular_effect,
        }
        return {
            action: theta + cfg.attitude_lookahead * (angular_velocity + effect)
            for action, effect in effects.items()
        }

    def _evaluate(self, state: np.ndarray) -> tuple[list[int], dict[str, Any]]:
        if state.shape != (8,):
            raise ValueError(
                f"LunarLander state must have shape (8,); got {state.shape}."
            )
        x, y, _, velocity_y, theta, _, _, _ = (float(value) for value in state)
        cfg = self.config
        all_actions = set(range(4))
        component_sets: dict[str, set[int]] = {}
        horizontal_predictions = self._horizontal_predictions(state)
        attitude_predictions = self._attitude_predictions(state)

        dangerous_descent = (
            y < cfg.critical_height and velocity_y < cfg.safe_min_vertical_speed
        )
        if dangerous_descent:
            component_sets["vertical_descent"] = {2}

        horizontal_active = abs(x) >= cfg.safe_horizontal_position
        if horizontal_active:
            baseline_error = abs(horizontal_predictions[0])
            component_sets["horizontal_position"] = {
                action
                for action, predicted_x in horizontal_predictions.items()
                if abs(predicted_x) <= baseline_error + cfg.tolerance
            }

        attitude_active = (
            y < cfg.attitude_critical_height and abs(theta) >= cfg.safe_angle
        )
        if attitude_active:
            baseline_error = abs(attitude_predictions[0])
            component_sets["attitude"] = {
                action
                for action, predicted_theta in attitude_predictions.items()
                if abs(predicted_theta) <= baseline_error + cfg.tolerance
            }

        intersection = set(all_actions)
        for actions in component_sets.values():
            intersection.intersection_update(actions)

        priority_fallback = bool(component_sets and not intersection)
        if intersection:
            safe = sorted(intersection)
            reason = "intersection of all active LunarLander safety components"
        elif dangerous_descent:
            safe = [2]
            reason = (
                "empty intersection; vertical-descent priority requires the main engine"
            )
        elif attitude_active:
            safe = sorted(component_sets["attitude"])
            reason = "empty intersection; attitude safety has priority over horizontal safety"
        else:
            safe = sorted(component_sets["horizontal_position"])
            reason = "empty intersection; horizontal safety fallback"

        return safe, {
            "violated_constraints": list(component_sets),
            "reason": reason,
            "priority_fallback": priority_fallback,
            "component_safe_actions": {
                name: sorted(actions) for name, actions in component_sets.items()
            },
            "predicted_horizontal_positions": horizontal_predictions,
            "predicted_angles": attitude_predictions,
            "approximation": "orientation-aware deterministic impulse heuristic; Box2D is not cloned",
        }


class LunarLanderDescentShield(SafetyShield):
    """Require the main engine while descending fast near the ground.

    This is the single-component counterpart of :class:`LunarLanderShield`,
    built for interval certification rather than for runtime intervention
    alone.  Inside the safety-critical band ``y < critical_height`` and
    ``v_y < safe_min_vertical_speed`` the main engine is the only admissible
    action; everywhere else all actions are admissible.  The band is one
    axis-aligned box in the two observation coordinates it reads, so the
    complete critical set is expressible as a single interval certificate.

    Unlike the three-component :class:`LunarLanderShield`, the band is offset
    from the unsafe set it protects (defaults ``y < 0.6``/``v_y < -0.35``
    guarding ``y < 0.5``/``v_y < -0.4``), so the shield acts before the
    violation rather than on top of it.
    """

    def __init__(
        self,
        config: LunarLanderDescentShieldConfig | None = None,
        *,
        fallback_rule: FallbackRule = "lowest",
    ) -> None:
        cfg = config or LunarLanderDescentShieldConfig()
        super().__init__(int(cfg.n_actions), fallback_rule=fallback_rule)
        self.config = cfg

    def in_critical_band(self, state: np.ndarray) -> bool:
        """Whether the open safety-critical descent band contains ``state``."""
        cfg = self.config
        altitude, vertical_speed = float(state[1]), float(state[3])
        return (
            altitude < cfg.critical_height - cfg.tolerance
            and vertical_speed < cfg.safe_min_vertical_speed - cfg.tolerance
        )

    def _evaluate(self, state: np.ndarray) -> tuple[list[int], dict[str, Any]]:
        if state.shape != (8,):
            raise ValueError(
                f"LunarLander state must have shape (8,); got {state.shape}."
            )
        cfg = self.config
        if self.in_critical_band(state):
            return [int(cfg.main_engine_action)], {
                "violated_constraints": ["vertical_descent"],
                "reason": (
                    "inside the safety-critical descent band; the main engine is "
                    "the only admissible action"
                ),
                "critical_band": {
                    "critical_height": float(cfg.critical_height),
                    "safe_min_vertical_speed": float(cfg.safe_min_vertical_speed),
                },
                "approximation": "descent-rate rule only; horizontal and attitude components are not applied",
            }
        return list(range(self.n_actions)), {
            "violated_constraints": [],
            "reason": "outside the safety-critical descent band",
            "critical_band": {
                "critical_height": float(cfg.critical_height),
                "safe_min_vertical_speed": float(cfg.safe_min_vertical_speed),
            },
            "approximation": "descent-rate rule only; horizontal and attitude components are not applied",
        }
