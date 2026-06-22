"""Centralized path helpers for Mountain Car experiments."""

from __future__ import annotations

from pathlib import Path


NOADAPT_POLICY_SUBDIR = "noadapt"
DEFAULT_TASK_SETTING = "default"


def pipeline_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifacts_root() -> Path:
    return pipeline_root() / "artifacts"


def runs_root() -> Path:
    return artifacts_root() / "runs"


def default_outputs_root() -> Path:
    return runs_root()


def seed_run_dir(outputs_root: Path, task_setting: str, seed: int) -> Path:
    return outputs_root / task_setting / f"seed_{seed}"

