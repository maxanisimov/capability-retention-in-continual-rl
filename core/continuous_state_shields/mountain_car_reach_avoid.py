"""Synthesized reach-avoid shield for discrete ``MountainCar-v0`` actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from continuous_state_shields.base import SafetyShield
from continuous_state_shields.config import MountainCarShieldConfig


def _nearest_indices(
    values: np.ndarray, low: float, step: float, size: int
) -> np.ndarray:
    return np.clip(np.rint((values - low) / step), 0, size - 1).astype(np.int64)


def synthesise_reach_avoid_grid(
    *,
    n_positions: int = 721,
    n_velocities: int = 281,
    goal_horizon: int = 200,
    boundary_margin: float = 0.0,
) -> dict[str, Any]:
    """Compute a finite-grid viability kernel and safe goal-reachable set.

    Transitions use the exact deterministic MountainCar equations and are mapped
    to the nearest grid point. The returned action mask preserves the set from
    which the goal is reachable without entering the boundary set.
    """

    if n_positions < 3 or n_velocities < 3:
        raise ValueError("The reach-avoid grid needs at least three points per axis.")
    if goal_horizon <= 0:
        raise ValueError("goal_horizon must be positive.")
    if boundary_margin < 0:
        raise ValueError("boundary_margin must be non-negative.")

    cfg = MountainCarShieldConfig()
    positions = np.linspace(
        cfg.min_position, cfg.max_position, n_positions, dtype=np.float64
    )
    velocities = np.linspace(
        -cfg.max_speed, cfg.max_speed, n_velocities, dtype=np.float64
    )
    position_step = float(positions[1] - positions[0])
    velocity_step = float(velocities[1] - velocities[0])
    x = positions[:, None]
    velocity = velocities[None, :]
    shape = (n_positions, n_velocities)
    state_count = n_positions * n_velocities
    next_indices = np.empty((3, state_count), dtype=np.int64)
    next_boundary = np.empty((3, state_count), dtype=bool)
    next_goal = np.empty((3, state_count), dtype=bool)

    for action in range(3):
        next_velocity = np.clip(
            velocity + cfg.force * (action - 1) - cfg.gravity * np.cos(3.0 * x),
            -cfg.max_speed,
            cfg.max_speed,
        )
        next_position = np.clip(x + next_velocity, cfg.min_position, cfg.max_position)
        wall = next_position <= cfg.min_position + boundary_margin
        next_velocity = np.where(
            (next_position <= cfg.min_position) & (next_velocity < 0.0),
            0.0,
            next_velocity,
        )
        position_indices = _nearest_indices(
            next_position, cfg.min_position, position_step, n_positions
        )
        velocity_indices = _nearest_indices(
            next_velocity, -cfg.max_speed, velocity_step, n_velocities
        )
        next_indices[action] = np.ravel_multi_index(
            (position_indices, velocity_indices), shape
        ).reshape(-1)
        next_boundary[action] = wall.reshape(-1)
        next_goal[action] = (next_position >= 0.5).reshape(-1)

    current_boundary = np.broadcast_to(
        x <= cfg.min_position + boundary_margin, shape
    ).reshape(-1)
    current_goal = np.broadcast_to(x >= 0.5, shape).reshape(-1)
    viable = ~current_boundary
    viability_iterations = 0
    while True:
        successor_viable = np.stack(
            [
                ~next_boundary[action]
                & (next_goal[action] | viable[next_indices[action]])
                for action in range(3)
            ]
        )
        updated = (~current_boundary) & (current_goal | successor_viable.any(axis=0))
        viability_iterations += 1
        if np.array_equal(updated, viable):
            break
        viable = updated

    reachable = current_goal & viable
    rank = np.full(state_count, -1, dtype=np.int16)
    rank[reachable] = 0
    reachability_iterations = 0
    for distance in range(1, goal_horizon + 1):
        reaches_known = np.stack(
            [
                ~next_boundary[action]
                & (next_goal[action] | reachable[next_indices[action]])
                for action in range(3)
            ]
        ).any(axis=0)
        added = viable & ~reachable & reaches_known
        reachability_iterations = distance
        if not bool(added.any()):
            break
        rank[added] = distance
        reachable |= added

    action_mask = np.zeros((state_count, 3), dtype=bool)
    for action in range(3):
        action_mask[:, action] = (
            ~next_boundary[action]
            & (next_goal[action] | reachable[next_indices[action]])
            & reachable
        )
    action_mask[current_goal] = True

    return {
        "positions": positions.astype(np.float32),
        "velocities": velocities.astype(np.float32),
        "viable": viable.reshape(shape),
        "winning": reachable.reshape(shape),
        "goal_rank": rank.reshape(shape),
        "action_mask": action_mask.reshape((*shape, 3)),
        "metadata": {
            "format_version": 1,
            "environment": "MountainCar-v0",
            "method": "finite_grid_reach_avoid",
            "n_positions": int(n_positions),
            "n_velocities": int(n_velocities),
            "position_step": position_step,
            "velocity_step": velocity_step,
            "goal_horizon": int(goal_horizon),
            "boundary_position": float(cfg.min_position),
            "boundary_margin": float(boundary_margin),
            "goal_position": 0.5,
            "viability_iterations": int(viability_iterations),
            "reachability_iterations": int(reachability_iterations),
            "viable_state_count": int(viable.sum()),
            "winning_state_count": int(reachable.sum()),
            "grid_state_count": int(state_count),
            "discretisation_note": (
                "Transitions are exact before nearest-grid projection; runtime "
                "membership requires all adjacent grid vertices to be winning."
            ),
        },
    }


def save_reach_avoid_grid(path: Path, payload: dict[str, Any]) -> None:
    """Save a synthesized grid without pickle-dependent objects."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        positions=payload["positions"],
        velocities=payload["velocities"],
        viable=payload["viable"],
        winning=payload["winning"],
        goal_rank=payload["goal_rank"],
        action_mask=payload["action_mask"],
        metadata_json=np.asarray(json.dumps(payload["metadata"], sort_keys=True)),
    )


def box_reach_avoid_grid(
    *,
    positions: np.ndarray,
    velocities: np.ndarray,
    action_mask: np.ndarray,
    velocity_bands: int = 16,
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Conservatively cover curved grid regions with axis-aligned boxes.

    Each elementary grid cell first receives the intersection of the masks at
    its four vertices.  Velocity cells are then grouped into bands and the
    masks are intersected once more over each band.  Consequently, every box
    mask is a subset of the source mask everywhere in the covered grid cells.
    Adjacent rectangles with the same mask are merged without changing their
    meaning.

    Grid locations with an empty synthesized action set use MountainCar's
    fail-safe push-right fallback, matching :class:`MountainCarReachAvoidShield`.
    """

    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    action_mask = np.asarray(action_mask, dtype=bool)
    if positions.ndim != 1 or velocities.ndim != 1:
        raise ValueError("positions and velocities must be one-dimensional.")
    if len(positions) < 2 or len(velocities) < 2:
        raise ValueError("At least two grid points are required on each axis.")
    if not np.all(np.diff(positions) > 0) or not np.all(np.diff(velocities) > 0):
        raise ValueError("Grid axes must be strictly increasing.")
    expected = (len(positions), len(velocities), 3)
    if action_mask.shape != expected:
        raise ValueError(f"action_mask must have shape {expected}.")
    n_velocity_cells = len(velocities) - 1
    if velocity_bands <= 0 or velocity_bands > n_velocity_cells:
        raise ValueError(
            f"velocity_bands must be in [1, {n_velocity_cells}]."
        )

    cell_masks = (
        action_mask[:-1, :-1]
        & action_mask[1:, :-1]
        & action_mask[:-1, 1:]
        & action_mask[1:, 1:]
    )
    empty_cells = ~cell_masks.any(axis=2)
    cell_masks[empty_cells, 2] = True

    cuts = np.unique(
        np.linspace(0, n_velocity_cells, velocity_bands + 1, dtype=np.int64)
    )
    rectangles: list[tuple[int, int, int, int, tuple[bool, bool, bool]]] = []
    all_actions = np.ones(3, dtype=bool)
    for velocity_lo, velocity_hi in zip(cuts[:-1], cuts[1:], strict=True):
        band_masks = cell_masks[:, velocity_lo:velocity_hi].all(axis=1)
        empty = ~band_masks.any(axis=1)
        band_masks[empty, 2] = True
        position_lo = 0
        while position_lo < len(band_masks):
            if np.array_equal(band_masks[position_lo], all_actions):
                position_lo += 1
                continue
            position_hi = position_lo + 1
            while position_hi < len(band_masks) and np.array_equal(
                band_masks[position_hi], band_masks[position_lo]
            ):
                position_hi += 1
            rectangles.append(
                (
                    position_lo,
                    position_hi,
                    int(velocity_lo),
                    int(velocity_hi),
                    tuple(bool(value) for value in band_masks[position_lo]),
                )
            )
            position_lo = position_hi

    # Merge identical horizontal runs over consecutive velocity bands.
    merged: list[tuple[int, int, int, int, tuple[bool, bool, bool]]] = []
    active: dict[
        tuple[int, int, tuple[bool, bool, bool]], tuple[int, int]
    ] = {}
    by_velocity_lo: dict[
        int, list[tuple[int, int, int, int, tuple[bool, bool, bool]]]
    ] = {}
    for rectangle in rectangles:
        by_velocity_lo.setdefault(rectangle[2], []).append(rectangle)
    for velocity_lo, velocity_hi in zip(cuts[:-1], cuts[1:], strict=True):
        next_active: dict[
            tuple[int, int, tuple[bool, bool, bool]], tuple[int, int]
        ] = {}
        for position_lo, position_hi, _, _, mask in by_velocity_lo.get(
            int(velocity_lo), []
        ):
            key = (position_lo, position_hi, mask)
            start = (
                active[key][0]
                if key in active and active[key][1] == int(velocity_lo)
                else int(velocity_lo)
            )
            next_active[key] = (start, int(velocity_hi))
        for (position_lo, position_hi, mask), (start, stop) in active.items():
            if (position_lo, position_hi, mask) not in next_active:
                merged.append((position_lo, position_hi, start, stop, mask))
        active = next_active
    for (position_lo, position_hi, mask), (start, stop) in active.items():
        merged.append((position_lo, position_hi, start, stop, mask))

    box_lows = np.asarray(
        [[positions[x0], velocities[v0]] for x0, _, v0, _, _ in merged],
        dtype=np.float32,
    )
    box_highs = np.asarray(
        [[positions[x1], velocities[v1]] for _, x1, _, v1, _ in merged],
        dtype=np.float32,
    )
    safe_masks = np.asarray([mask for *_, mask in merged], dtype=bool)

    # Independently reconstruct every elementary cell's effective box mask.
    # This is also a useful invariant against accidental unsafe merging.
    effective = np.ones_like(cell_masks, dtype=bool)
    covered = np.zeros(cell_masks.shape[:2], dtype=bool)
    for x0, x1, v0, v1, mask in merged:
        effective[x0:x1, v0:v1] &= np.asarray(mask, dtype=bool)
        covered[x0:x1, v0:v1] = True
    effective[~covered] = True
    if np.any(effective & ~cell_masks):
        raise RuntimeError("Box conversion admitted an action excluded by the grid.")
    restricted = ~cell_masks.all(axis=2)
    if np.any(restricted & ~covered):
        raise RuntimeError("Box conversion failed to cover a restricted grid cell.")

    unique_masks, counts = np.unique(safe_masks, axis=0, return_counts=True)
    metadata = {
        "format_version": 1,
        "environment": "MountainCar-v0",
        "method": "conservative_interval_box_cover",
        "source_method": (source_metadata or {}).get("method"),
        "velocity_bands": int(len(cuts) - 1),
        "box_count": int(len(merged)),
        "elementary_cell_count": int(np.prod(cell_masks.shape[:2])),
        "restricted_elementary_cell_count": int(restricted.sum()),
        "covered_elementary_cell_count": int(covered.sum()),
        "overrestricted_elementary_cell_count": int((covered & ~restricted).sum()),
        "empty_source_cell_fallback_count": int(empty_cells.sum()),
        "box_mask_counts": {
            "".join("1" if bit else "0" for bit in mask): int(count)
            for mask, count in zip(unique_masks, counts, strict=True)
        },
        "construction": (
            "Four-vertex mask intersection per grid cell, followed by mask "
            "intersection within velocity bands and exact-mask rectangle merging."
        ),
        "boundary_semantics": (
            "Boxes are closed. At a shared boundary, all matching box masks are "
            "intersected."
        ),
    }
    return {
        "box_lows": box_lows,
        "box_highs": box_highs,
        "safe_masks": safe_masks,
        "metadata": metadata,
    }


def save_interval_box_shield(path: Path, payload: dict[str, Any]) -> None:
    """Save a boxed reach-avoid shield without pickle-dependent objects."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        box_lows=np.asarray(payload["box_lows"], dtype=np.float32),
        box_highs=np.asarray(payload["box_highs"], dtype=np.float32),
        safe_masks=np.asarray(payload["safe_masks"], dtype=bool),
        metadata_json=np.asarray(json.dumps(payload["metadata"], sort_keys=True)),
    )


class MountainCarIntervalBoxShield(SafetyShield):
    """A PSPO-compatible union of closed MountainCar interval boxes."""

    def __init__(
        self,
        *,
        box_lows: np.ndarray,
        box_highs: np.ndarray,
        safe_masks: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(3, fallback_rule="highest")
        self.box_lows = np.asarray(box_lows, dtype=np.float64)
        self.box_highs = np.asarray(box_highs, dtype=np.float64)
        self.safe_masks = np.asarray(safe_masks, dtype=bool)
        if self.box_lows.ndim != 2 or self.box_lows.shape[1:] != (2,):
            raise ValueError("box_lows must have shape (n_boxes, 2).")
        if self.box_highs.shape != self.box_lows.shape:
            raise ValueError("box_highs must have the same shape as box_lows.")
        if self.safe_masks.shape != (len(self.box_lows), 3):
            raise ValueError("safe_masks must have shape (n_boxes, 3).")
        if np.any(self.box_lows > self.box_highs):
            raise ValueError("Every interval box must satisfy lower <= upper.")
        if np.any(~self.safe_masks.any(axis=1)):
            raise ValueError("Every interval box needs at least one safe action.")
        self.metadata = dict(metadata or {})
        self.config = MountainCarShieldConfig()

    @classmethod
    def load(cls, path: str | Path) -> "MountainCarIntervalBoxShield":
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            return cls(
                box_lows=payload["box_lows"],
                box_highs=payload["box_highs"],
                safe_masks=payload["safe_masks"],
                metadata=metadata,
            )

    def _evaluate(self, state: np.ndarray) -> tuple[list[int], dict[str, Any]]:
        if state.shape != (2,):
            raise ValueError("MountainCar state must have shape (2,).")
        matches = np.all(
            (state[None, :] >= self.box_lows)
            & (state[None, :] <= self.box_highs),
            axis=1,
        )
        safe_mask = np.ones(3, dtype=bool)
        if bool(matches.any()):
            safe_mask = self.safe_masks[matches].all(axis=0)
        if not bool(safe_mask.any()):
            raise RuntimeError("Overlapping interval boxes have disjoint action masks.")
        return np.flatnonzero(safe_mask).astype(int).tolist(), {
            "region": "interval_box_cover" if bool(matches.any()) else "unrestricted",
            "matching_box_indices": np.flatnonzero(matches).astype(int).tolist(),
            "reason": (
                "intersection of every matching conservative interval-box mask"
                if bool(matches.any())
                else "state is outside every safety-critical interval box"
            ),
            "unsafe_event": "left_boundary_contact",
        }


class MountainCarReachAvoidShield(SafetyShield):
    """Runtime shield backed by a synthesized MountainCar reach-avoid grid."""

    def __init__(
        self,
        *,
        positions: np.ndarray,
        velocities: np.ndarray,
        winning: np.ndarray,
        goal_rank: np.ndarray,
        action_mask: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        super().__init__(3, fallback_rule="highest")
        self.positions = np.asarray(positions, dtype=np.float64)
        self.velocities = np.asarray(velocities, dtype=np.float64)
        self.winning = np.asarray(winning, dtype=bool)
        self.goal_rank = np.asarray(goal_rank, dtype=np.int64)
        self.action_mask = np.asarray(action_mask, dtype=bool)
        self.metadata = dict(metadata)
        expected_shape = (len(self.positions), len(self.velocities))
        if self.winning.shape != expected_shape:
            raise ValueError("winning grid shape does not match its axes.")
        if self.goal_rank.shape != expected_shape:
            raise ValueError("goal_rank grid shape does not match its axes.")
        if self.action_mask.shape != (*expected_shape, 3):
            raise ValueError(
                "action_mask must have shape (n_positions, n_velocities, 3)."
            )
        self.config = MountainCarShieldConfig()
        self.boundary_margin = float(self.metadata.get("boundary_margin", 0.0))
        self.goal_position = float(self.metadata.get("goal_position", 0.5))

    @classmethod
    def load(cls, path: str | Path) -> "MountainCarReachAvoidShield":
        """Load a shield saved by :func:`save_reach_avoid_grid`."""

        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            return cls(
                positions=payload["positions"],
                velocities=payload["velocities"],
                winning=payload["winning"],
                goal_rank=payload["goal_rank"],
                action_mask=payload["action_mask"],
                metadata=metadata,
            )

    def predict_next_state(self, state: Any, action: int) -> np.ndarray:
        """Apply the exact deterministic MountainCar transition."""

        observation = np.asarray(state, dtype=float)
        if observation.shape != (2,):
            raise ValueError("MountainCar state must have shape (2,).")
        action = self._validate_action(action)
        x, velocity = map(float, observation)
        cfg = self.config
        next_velocity = float(
            np.clip(
                velocity + cfg.force * (action - 1) - cfg.gravity * np.cos(3.0 * x),
                -cfg.max_speed,
                cfg.max_speed,
            )
        )
        next_position = float(
            np.clip(x + next_velocity, cfg.min_position, cfg.max_position)
        )
        if next_position <= cfg.min_position and next_velocity < 0.0:
            next_velocity = 0.0
        return np.asarray([next_position, next_velocity], dtype=float)

    @staticmethod
    def _axis_neighbours(axis: np.ndarray, value: float) -> tuple[int, ...]:
        upper = int(np.searchsorted(axis, value, side="left"))
        if upper <= 0:
            return (0,)
        if upper >= len(axis):
            return (len(axis) - 1,)
        if value == axis[upper]:
            return (upper,)
        return (upper - 1, upper)

    def _winning_cell(self, state: np.ndarray) -> bool:
        x, velocity = map(float, state)
        if x <= self.config.min_position + self.boundary_margin:
            return False
        if x >= self.goal_position:
            return True
        position_ids = self._axis_neighbours(self.positions, x)
        velocity_ids = self._axis_neighbours(self.velocities, velocity)
        return all(
            bool(self.winning[position_id, velocity_id])
            for position_id in position_ids
            for velocity_id in velocity_ids
        )

    def _nearest_grid_index(self, state: np.ndarray) -> tuple[int, int]:
        x, velocity = map(float, state)
        position_id = int(np.abs(self.positions - x).argmin())
        velocity_id = int(np.abs(self.velocities - velocity).argmin())
        return position_id, velocity_id

    def preferred_action(self, state: Any) -> int:
        """Return the admissible action with the lowest successor goal rank."""

        observation = np.asarray(state, dtype=float)
        safe = self.get_safe_actions(observation)
        ranked: list[tuple[int, int]] = []
        for action in safe:
            successor = self.predict_next_state(observation, action)
            if successor[0] >= self.goal_position:
                rank = 0
            else:
                rank = int(self.goal_rank[self._nearest_grid_index(successor)])
                if rank < 0:
                    rank = np.iinfo(np.int32).max
            ranked.append((rank, -action))
        return -min(ranked)[1]

    def shield_action(self, state: Any, proposed_action: int) -> int:
        """Keep admissible proposals; otherwise choose best goal-rank recovery."""

        proposed_action = self._validate_action(proposed_action)
        if self.is_safe_action(state, proposed_action):
            return proposed_action
        return self.preferred_action(state)

    def _evaluate(self, state: np.ndarray) -> tuple[list[int], dict[str, Any]]:
        if state.shape != (2,):
            raise ValueError("MountainCar state must have shape (2,).")
        x = float(state[0])
        at_boundary = x <= self.config.min_position + self.boundary_margin
        if x >= self.goal_position:
            safe_actions = [0, 1, 2]
            successor_winning = [True, True, True]
        elif at_boundary:
            safe_actions = [2]
            successor_winning = [False, False, False]
        else:
            successors = [self.predict_next_state(state, action) for action in range(3)]
            successor_winning = [self._winning_cell(item) for item in successors]
            safe_actions = [
                action for action, winning in enumerate(successor_winning) if winning
            ]
            if not safe_actions:
                safe_actions = [2]
        return safe_actions, {
            "region": "unsafe_boundary" if at_boundary else "reach_avoid",
            "violated_constraints": ["left_boundary_contact"] if at_boundary else [],
            "reason": "actions must preserve the synthesized safe goal-reachable set",
            "position": float(state[0]),
            "velocity": float(state[1]),
            "successor_winning": successor_winning,
            "winning_state": self._winning_cell(state),
            "unsafe_event": "left_boundary_contact",
            "goal_position": self.goal_position,
        }
