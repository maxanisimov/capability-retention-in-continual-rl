"""Configuration dataclasses and documented defaults for the three shields."""

from __future__ import annotations

from dataclasses import dataclass
from math import radians


@dataclass(frozen=True)
class MountainCarShieldConfig:
    """MountainCar physics and velocity-aware left-boundary safety region."""

    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    force: float = 0.001
    gravity: float = 0.0025
    critical_min_position: float = -1.2
    critical_max_position: float = -1.0
    push_right_action: int = 2
    tolerance: float = 1e-12
    unsafe_max_position: float | None = None

    def __post_init__(self) -> None:
        # Backward-compatible alias for launch commands written for the older
        # position-only shield. It now controls the lower edge of the critical
        # interval; the unsafe event itself is contact with ``min_position``.
        if self.unsafe_max_position is not None:
            if self.critical_min_position != -1.2:
                raise ValueError(
                    "Specify only critical_min_position; unsafe_max_position is a "
                    "deprecated alias for it."
                )
            object.__setattr__(
                self, "critical_min_position", float(self.unsafe_max_position)
            )
        if not (
            self.min_position
            <= self.critical_min_position
            < self.critical_max_position
            < self.max_position
        ):
            raise ValueError(
                "The safety-critical interval must be ordered inside the physical "
                "position interval."
            )
        if self.push_right_action != 2:
            raise ValueError("MountainCar's push-right action must be action 2.")
        if (
            self.max_speed <= 0
            or self.force <= 0
            or self.gravity < 0
            or self.tolerance < 0
        ):
            raise ValueError(
                "MountainCar speed/force must be positive and gravity/tolerance non-negative."
            )


@dataclass(frozen=True)
class CartPoleShieldConfig:
    """CartPole-v1 physics and margins preceding Gymnasium termination."""

    safe_position: float = 2.2
    safe_angle: float = radians(10.0)
    position_weight: float = 1.0
    angle_weight: float = 5.0
    prediction_steps: int = 2
    gravity: float = 9.8
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    half_pole_length: float = 0.5
    force_magnitude: float = 10.0
    tau: float = 0.02
    kinematics_integrator: str = "euler"
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        positive = (
            self.safe_position,
            self.safe_angle,
            self.position_weight,
            self.angle_weight,
            self.gravity,
            self.mass_cart,
            self.mass_pole,
            self.half_pole_length,
            self.force_magnitude,
            self.tau,
        )
        if any(value <= 0 for value in positive):
            raise ValueError(
                "CartPole thresholds, weights, and physical constants must be positive."
            )
        if self.prediction_steps < 1:
            raise ValueError("prediction_steps must be at least one.")
        if self.kinematics_integrator not in ("euler", "semi-implicit-euler"):
            raise ValueError(
                "kinematics_integrator must be 'euler' or 'semi-implicit-euler'."
            )
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative.")


@dataclass(frozen=True)
class LunarLanderShieldConfig:
    """Thresholds for the normalized LunarLander-v3 observation."""

    safe_horizontal_position: float = 0.8
    critical_height: float = 0.5
    safe_min_vertical_speed: float = -0.4
    attitude_critical_height: float = 0.75
    safe_angle: float = radians(20.0)
    horizontal_lookahead: float = 1.0
    attitude_lookahead: float = 1.0
    side_engine_horizontal_effect: float = 0.05
    main_engine_horizontal_effect: float = 0.05
    side_engine_angular_effect: float = 0.08
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        positive = (
            self.safe_horizontal_position,
            self.safe_angle,
            self.horizontal_lookahead,
            self.attitude_lookahead,
            self.side_engine_horizontal_effect,
            self.main_engine_horizontal_effect,
            self.side_engine_angular_effect,
        )
        if any(value <= 0 for value in positive):
            raise ValueError(
                "LunarLander thresholds, lookaheads, and effect sizes must be positive."
            )
        if self.attitude_critical_height < self.critical_height:
            raise ValueError(
                "attitude_critical_height must not be below critical_height."
            )
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative.")


@dataclass(frozen=True)
class LunarLanderDescentShieldConfig:
    """Thresholds for the descent-only LunarLander shield.

    The runtime shield is active on the open set ``y < critical_height`` and
    ``v_y < safe_min_vertical_speed``; the interval certificate covers its
    closed extension.  ``critical_height`` and ``safe_min_vertical_speed``
    deliberately sit *outside* the unsafe set they protect, so the shield acts
    before a violation rather than after it.
    """

    critical_height: float = 0.6
    safe_min_vertical_speed: float = -0.35
    main_engine_action: int = 2
    n_actions: int = 4
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive.")
        if not 0 <= self.main_engine_action < self.n_actions:
            raise ValueError(
                f"main_engine_action must lie in [0, {self.n_actions}); "
                f"got {self.main_engine_action}."
            )
        if self.safe_min_vertical_speed >= 0:
            raise ValueError("safe_min_vertical_speed must be negative.")
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
