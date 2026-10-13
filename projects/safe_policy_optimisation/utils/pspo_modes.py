"""Shared mode defaults for active PSPO launchers."""

from __future__ import annotations

import argparse

DEFAULT_SAFE_REGION_SHAPE = "segment"
DEFAULT_SEGMENT_TOLERANCE = 1e-3
DEFAULT_SEGMENT_SPLITS = 4
DEFAULT_SEGMENT_MAX_SPLITS = 8


def boolean_mode(value: str) -> bool:
    if value.lower() in {"true", "1"}:
        return True
    if value.lower() in {"false", "0"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def add_pspo_mode_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--safe-region-shape", choices=("segment", "orthotope"),
        default=DEFAULT_SAFE_REGION_SHAPE,
        help="PSPO geometry (default: line segment); select orthotope explicitly.",
    )
    parser.add_argument(
        "--verify-first", nargs="?", const=True, default=False, type=boolean_mode,
        help="Verify proposals first; optionally pass true/false (default: false).",
    )
    parser.add_argument("--segment-tolerance", type=float, default=DEFAULT_SEGMENT_TOLERANCE)
    parser.add_argument("--segment-splits", type=int, default=DEFAULT_SEGMENT_SPLITS)
    parser.add_argument("--segment-max-splits", type=int, default=DEFAULT_SEGMENT_MAX_SPLITS)


def mode_settings(args: argparse.Namespace) -> dict:
    return {
        "safe_region_shape": getattr(args, "safe_region_shape", DEFAULT_SAFE_REGION_SHAPE),
        "verify_first": bool(getattr(args, "verify_first", False)),
        "segment_tolerance": getattr(args, "segment_tolerance", DEFAULT_SEGMENT_TOLERANCE),
        "segment_splits": getattr(args, "segment_splits", DEFAULT_SEGMENT_SPLITS),
        "segment_max_splits": getattr(args, "segment_max_splits", DEFAULT_SEGMENT_MAX_SPLITS),
    }


def variant_name(shape: str, verify_first: bool) -> str:
    geometry = "line_segment" if shape == "segment" else "orthotope"
    return geometry + ("_verify_first" if verify_first else "")
