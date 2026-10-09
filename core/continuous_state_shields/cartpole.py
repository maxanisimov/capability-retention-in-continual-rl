"""Model-based safety shield for discrete ``CartPole-v1``."""

from __future__ import annotations

from typing import Any

import numpy as np

from continuous_state_shields.base import FallbackRule, SafetyShield
from continuous_state_shields.config import CartPoleShieldConfig


class CartPoleShield(SafetyShield):
    """Score both actions with configurable CartPole lookahead dynamics.

    Zero-violation actions are all retained. If neither action predicts a state
    inside both margins, all minimum-score actions are retained. The weighted
    score is ``w_x max(0, |x|-x_safe) + w_theta max(0, |theta|-theta_safe)``.

    The default uses two Euler steps. Gymnasium's first Euler position/angle
    update uses the old velocities and is therefore identical for both actions;
    the second step makes the action's effect visible while remaining a short,
    deterministic application of the environment dynamics.
    """

    def __init__(
        self,
        config: CartPoleShieldConfig | None = None,
        *,
        fallback_rule: FallbackRule = "lowest",
    ) -> None:
        super().__init__(2, fallback_rule=fallback_rule)
        self.config = config or CartPoleShieldConfig()

    def predict_next_state(
        self, state: Any, action: int, *, steps: int | None = None
    ) -> np.ndarray:
        """Apply the configured Gymnasium CartPole equations one or more times."""
        predicted = np.asarray(state, dtype=float)
        if predicted.shape != (4,):
            raise ValueError(
                f"CartPole state must have shape (4,); got {predicted.shape}."
            )
        action = self._validate_action(action)
        n_steps = self.config.prediction_steps if steps is None else int(steps)
        if n_steps < 1:
            raise ValueError("steps must be at least one.")
        for _ in range(n_steps):
            predicted = self._step(predicted, action)
        return predicted

    def _step(self, state: np.ndarray, action: int) -> np.ndarray:
        x, x_dot, theta, theta_dot = state
        cfg = self.config
        total_mass = cfg.mass_cart + cfg.mass_pole
        polemass_length = cfg.mass_pole * cfg.half_pole_length
        force = cfg.force_magnitude if action == 1 else -cfg.force_magnitude
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        temp = (force + polemass_length * theta_dot**2 * sin_theta) / total_mass
        theta_acc = (cfg.gravity * sin_theta - cos_theta * temp) / (
            cfg.half_pole_length
            * (4.0 / 3.0 - cfg.mass_pole * cos_theta**2 / total_mass)
        )
        x_acc = temp - polemass_length * theta_acc * cos_theta / total_mass

        if cfg.kinematics_integrator == "euler":
            return np.asarray(
                [
                    x + cfg.tau * x_dot,
                    x_dot + cfg.tau * x_acc,
                    theta + cfg.tau * theta_dot,
                    theta_dot + cfg.tau * theta_acc,
                ]
            )
        next_x_dot = x_dot + cfg.tau * x_acc
        next_theta_dot = theta_dot + cfg.tau * theta_acc
        return np.asarray(
            [
                x + cfg.tau * next_x_dot,
                next_x_dot,
                theta + cfg.tau * next_theta_dot,
                next_theta_dot,
            ]
        )

    def _score(self, state: np.ndarray) -> tuple[float, float, float]:
        cfg = self.config
        x_violation = max(0.0, abs(float(state[0])) - cfg.safe_position)
        angle_violation = max(0.0, abs(float(state[2])) - cfg.safe_angle)
        return (
            cfg.position_weight * x_violation + cfg.angle_weight * angle_violation,
            x_violation,
            angle_violation,
        )

    def _evaluate(self, state: np.ndarray) -> tuple[list[int], dict[str, Any]]:
        if state.shape != (4,):
            raise ValueError(f"CartPole state must have shape (4,); got {state.shape}.")
        predictions = {
            action: self.predict_next_state(state, action) for action in range(2)
        }
        score_parts = {
            action: self._score(prediction)
            for action, prediction in predictions.items()
        }
        scores = {action: parts[0] for action, parts in score_parts.items()}
        cfg = self.config
        zero_violation = [
            action for action, score in scores.items() if score <= cfg.tolerance
        ]
        if zero_violation:
            safe = zero_violation
            reason = "predicted successor stays inside both safety margins"
        else:
            best = min(scores.values())
            safe = [
                action
                for action, score in scores.items()
                if score <= best + cfg.tolerance
            ]
            reason = (
                "no zero-violation action; minimum weighted-violation action retained"
            )

        violated = []
        if abs(float(state[0])) >= cfg.safe_position or all(
            parts[1] > 0 for parts in score_parts.values()
        ):
            violated.append("cart_position")
        if abs(float(state[2])) >= cfg.safe_angle or all(
            parts[2] > 0 for parts in score_parts.values()
        ):
            violated.append("pole_angle")
        return safe, {
            "violated_constraints": violated,
            "reason": reason,
            "predicted_next_states": {
                action: pred.tolist() for action, pred in predictions.items()
            },
            "predicted_violation_scores": scores,
            "predicted_position_violations": {
                action: parts[1] for action, parts in score_parts.items()
            },
            "predicted_angle_violations": {
                action: parts[2] for action, parts in score_parts.items()
            },
        }
