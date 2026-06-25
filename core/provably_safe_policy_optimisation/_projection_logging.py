"""Helper to log per-window projection diagnostics to the SB3 logger.

Shared by :class:`ProjectedDQN` and :class:`ProjectedPPO`. Cumulative counters
live on the :class:`ProjectedAdam` optimizer; here we diff them against the
previous call to report *rates over the latest training window* (one SB3
``train()`` call), which reveals how projection frequency/magnitude evolves as
adaptation settles -- something cumulative totals hide.
"""

from __future__ import annotations

from typing import Any


def record_projection_window(model: Any) -> None:
    """Record per-window projection metrics to ``model.logger`` (if available)."""
    optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
    if optimizer is None or not getattr(optimizer, "has_bounds", False):
        return
    logger = getattr(model, "logger", None)
    if logger is None:
        return

    current = optimizer._raw_diag_counters()
    previous = getattr(model, "_projection_log_prev", None)
    model._projection_log_prev = current
    if previous is None:
        return

    delta_steps = current["bounded_steps"] - previous["bounded_steps"]
    if delta_steps <= 0:
        return
    delta_active = current["projection_active_steps"] - previous["projection_active_steps"]
    delta_elements = current["projected_elements"] - previous["projected_elements"]
    delta_l2 = current["displacement_l2_sum"] - previous["displacement_l2_sum"]

    logger.record("projection/active_step_fraction", delta_active / delta_steps)
    logger.record("projection/mean_projected_elements", delta_elements / delta_steps)
    logger.record("projection/mean_displacement_l2", delta_l2 / delta_steps)
    logger.record("projection/max_displacement_l2", current["max_displacement_l2"])
    logger.record("projection/bounded_steps_total", current["bounded_steps"])
