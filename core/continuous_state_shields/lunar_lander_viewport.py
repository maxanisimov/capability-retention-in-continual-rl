"""Two-sided, position-only LunarLander viewport action constraint.

Side-engine impulse is leftwards for action 1 and rightwards for action 3
when upright. Orientation and momentum can reverse the intended movement;
certifying this rule is not a trajectory-invariance proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from continuous_state_shields.base import FallbackRule, SafetyShield


@dataclass(frozen=True)
class LunarLanderViewportShieldConfig:
    m: float = 0.8
    viewport_edge: float = 1.0
    push_left_action: int = 1
    push_right_action: int = 3
    n_actions: int = 4
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.m) or not 0 < self.m < 1:
            raise ValueError("m must be finite and strictly between 0 and 1")
        if self.viewport_edge != 1 or self.n_actions != 4:
            raise ValueError("LunarLander uses viewport edges ±1 and four actions")
        if self.push_left_action != 1 or self.push_right_action != 3:
            raise ValueError("Upright inward impulses use action 1 at right, action 3 at left")
        if self.tolerance != 0:
            raise ValueError("Viewport bands use exact closed endpoints without a tolerance")


class LunarLanderViewportShield(SafetyShield):
    """Allow only action 1 at x∈[m,1], only action 3 at x∈[-1,-m]."""

    def __init__(self, config: LunarLanderViewportShieldConfig | None = None,
                 *, fallback_rule: FallbackRule = "lowest") -> None:
        self.config = config or LunarLanderViewportShieldConfig()
        super().__init__(4, fallback_rule=fallback_rule)

    def _evaluate(self, state: np.ndarray) -> tuple[list[int], dict[str, Any]]:
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("LunarLander state must be a finite vector of shape (8,)")
        x = float(state[0])
        cfg = self.config
        if cfg.m <= x <= 1:
            actions, region = [cfg.push_left_action], "right_edge_band"
        elif -1 <= x <= -cfg.m:
            actions, region = [cfg.push_right_action], "left_edge_band"
        else:
            actions, region = list(range(4)), "outside_critical_bands"
        return actions, {
            "region": region, "position": x, "m": cfg.m,
            "unsafe_event": "out_of_view", "out_of_view": abs(x) >= 1,
            "violated_constraints": ["out_of_view"] if abs(x) >= 1 else [],
            "reason": "position-only inward side-engine action constraint",
        }


def viewport_certificate_arrays(m: float, observation_low: np.ndarray,
                                observation_high: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two complete 8-D boxes, left then right, spanning all other coordinates.

    Round x endpoints outwards if needed to ensure float32 certification covers
    the semantic real-valued closed bands, rather than accidentally shrinking.
    """
    LunarLanderViewportShieldConfig(m=m)
    low = np.asarray(observation_low, dtype=np.float32)
    high = np.asarray(observation_high, dtype=np.float32)
    if low.shape != (8,) or high.shape != (8,) or not (np.isfinite(low).all() and np.isfinite(high).all()):
        raise ValueError("Certificate needs finite eight-dimensional observation bounds")
    if np.any(low > high) or low[0] > -1 or high[0] < 1:
        raise ValueError("Observation bounds must be ordered and contain both viewport bands")
    lows, highs = np.tile(low, (2, 1)), np.tile(high, (2, 1))
    lower_m = np.float32(m)
    if float(lower_m) > m:
        lower_m = np.nextafter(lower_m, np.float32(-np.inf))
    lows[:, 0] = [-1, lower_m]
    highs[:, 0] = [-lower_m, 1]
    # -lower_m covers -m from above.
    masks = np.zeros((2, 4), dtype=bool)
    masks[0, 3] = True
    masks[1, 1] = True
    return lows, highs, masks
