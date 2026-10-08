#!/usr/bin/env python
"""Build the minimal MountainCar left-edge shield and its matching certificate.

The specification is an *action constraint*, not an invariance property: push
left (action 0) is inadmissible within ``strip_width`` of the physical left wall,
and both remaining actions stay admissible there. Everywhere else all three
actions are admissible.

This is deliberately not the wall-avoidance property that
:class:`MountainCarShield` enforces. Wall contact is unavoidable from inside a
strip this thin -- at ``x = -1.15`` the reachable leftward speed is about
``-0.062`` while the maximum rightward acceleration is ``0.0034`` per step, so
braking needs roughly ``0.57`` of position and the strip offers ``0.05``. A
0.05-wide band can therefore never be inductively invariant, and only the
action-level constraint is soundly enforceable on it. PSPO certifies exactly
that: state -> admissible-action sets, never trajectories.

The runtime shield is emitted as the ``mountaincar-boxes`` ``.npz`` artifact and
the PSPO certificate as a ``TensorDataset(X_l, X_u, safe_mask)``. Both are
written from the same float32 arrays because ``train_pspo_continuous.py``
requires the certificate to match the runtime boxes to ``atol=1e-7``.

Usage:
    python projects/safe_policy_optimisation/scripts/build_mountaincar_left_edge_shield.py \
        --output-dir outputs/continuous_state_shields/synthesised/mountaincar_left_edge \
        --run-id left_edge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]

MIN_POSITION = -1.2
MAX_SPEED = 0.07
PUSH_LEFT_ACTION = 0
N_ACTIONS = 3

# Gym clips the wall contact position to -1.2 and stores observations as
# float32, so the wall is observed as roughly -1.20000005 -- just *below* the
# nominal bound. The box is opened a hair further down so wall states still fall
# inside it; widening a certified box is conservative for interval verification.
WALL_EPSILON = 1e-4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs/continuous_state_shields/synthesised/mountaincar_left_edge",
    )
    parser.add_argument("--run-id", default="left_edge")
    parser.add_argument(
        "--strip-width",
        type=float,
        default=0.05,
        help="Width of the no-push-left strip against the left wall (default: 0.05).",
    )
    return parser


def build_box(strip_width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ``(lows, highs, safe_masks)`` arrays for the single strip."""

    if not 0.0 < strip_width < 1.8:
        raise ValueError("--strip-width must lie inside the position range.")
    lows = np.asarray(
        [[MIN_POSITION - WALL_EPSILON, -MAX_SPEED]], dtype=np.float32
    )
    highs = np.asarray(
        [[MIN_POSITION + strip_width, MAX_SPEED]], dtype=np.float32
    )
    safe_masks = np.ones((1, N_ACTIONS), dtype=bool)
    safe_masks[0, PUSH_LEFT_ACTION] = False
    return lows, highs, safe_masks


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lows, highs, safe_masks = build_box(float(args.strip_width))

    run_dir = Path(args.output_dir) / str(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "shield": "mountaincar-left-edge",
        "specification": "push-left is inadmissible within the strip",
        "unsafe_event": "left_boundary_contact",
        "strip_width": float(args.strip_width),
        "strip_position_interval": [float(lows[0, 0]), float(highs[0, 0])],
        "forbidden_action": PUSH_LEFT_ACTION,
        "note": (
            "Action constraint only. Wall contact is not preventable from a strip "
            "this thin; see the module docstring."
        ),
    }

    shield_path = run_dir / "left_edge_shield.npz"
    np.savez_compressed(
        shield_path,
        box_lows=lows,
        box_highs=highs,
        safe_masks=safe_masks,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    certificate = TensorDataset(
        torch.from_numpy(lows.copy()),
        torch.from_numpy(highs.copy()),
        torch.from_numpy(safe_masks.astype(np.float32)),
    )
    certificate_path = run_dir / "critical_interval_dataset.pt"
    torch.save(certificate, certificate_path)

    # Fail closed: the runtime shield must reproduce the mask the certificate
    # states, on the strip and just outside it.
    from continuous_state_shields import MountainCarIntervalBoxShield

    shield = MountainCarIntervalBoxShield.load(shield_path)
    inside = shield.get_safe_actions(np.asarray([float(highs[0, 0]) - 1e-3, 0.0]))
    outside = shield.get_safe_actions(np.asarray([float(highs[0, 0]) + 1e-3, 0.0]))
    at_wall = shield.get_safe_actions(np.asarray([np.float32(MIN_POSITION), 0.0]))
    if sorted(inside) != [1, 2] or sorted(at_wall) != [1, 2]:
        raise RuntimeError(f"Strip must forbid push-left; got {inside} / {at_wall}.")
    if sorted(outside) != [0, 1, 2]:
        raise RuntimeError(f"Outside the strip all actions are safe; got {outside}.")

    print(f"shield:      {shield_path}")
    print(f"certificate: {certificate_path}")
    print(f"strip:       position [{lows[0, 0]}, {highs[0, 0]}], velocity full range")
    print(f"safe mask:   {safe_masks.tolist()} (action {PUSH_LEFT_ACTION} forbidden)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
