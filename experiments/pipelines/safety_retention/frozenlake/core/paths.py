"""Path helpers for FrozenLake safety runs."""

from __future__ import annotations

from pathlib import Path


NOADAPT_POLICY_SUBDIR = "noadapt"
SOURCE_POLICY_SUBDIR = NOADAPT_POLICY_SUBDIR
DOWNSTREAM_UNCONSTRAINED_SUBDIR = "downstream_unconstrained"
DOWNSTREAM_EWC_SUBDIR = "downstream_ewc"
DOWNSTREAM_RASHOMON_SUBDIR = "downstream_rashomon"


def pipeline_root() -> Path:
    return Path(__file__).resolve().parents[1]


def artifacts_root() -> Path:
    return pipeline_root() / "artifacts"


def default_outputs_root() -> Path:
    return artifacts_root() / "runs"


def seed_run_dir(outputs_root: Path, layout: str, seed: int) -> Path:
    return outputs_root / layout / f"seed_{seed}"


def source_run_dir(outputs_root: Path, layout: str, seed: int) -> Path:
    return seed_run_dir(outputs_root, layout, seed) / NOADAPT_POLICY_SUBDIR


def resolve_source_run_dir(outputs_root: Path, layout: str, seed: int) -> Path:
    canonical = source_run_dir(outputs_root, layout, seed)
    if canonical.exists():
        return canonical
    legacy = seed_run_dir(outputs_root, layout, seed) / "source"
    if legacy.exists():
        return legacy
    return canonical


def mode_run_subdir(mode: str) -> str:
    if mode == "source":
        return NOADAPT_POLICY_SUBDIR
    if mode == "downstream_unconstrained":
        return DOWNSTREAM_UNCONSTRAINED_SUBDIR
    if mode == "downstream_ewc":
        return DOWNSTREAM_EWC_SUBDIR
    if mode == "downstream_rashomon":
        return DOWNSTREAM_RASHOMON_SUBDIR
    raise ValueError(f"Unsupported mode '{mode}'.")


def mode_run_dir(outputs_root: Path, layout: str, seed: int, mode: str) -> Path:
    return seed_run_dir(outputs_root, layout, seed) / mode_run_subdir(mode)

