"""Centralized path helpers for LunarLander experiments."""

from __future__ import annotations

from pathlib import Path


def pipeline_root() -> Path:
    return Path(__file__).resolve().parents[2]


def settings_root() -> Path:
    return pipeline_root() / "settings"


def docs_root() -> Path:
    return pipeline_root() / "docs"


def artifacts_root() -> Path:
    return pipeline_root() / "artifacts"


def runs_root() -> Path:
    return artifacts_root() / "runs"


def logs_root() -> Path:
    return artifacts_root() / "logs"


def reports_root() -> Path:
    return artifacts_root() / "reports"


def legacy_outputs_root() -> Path:
    return pipeline_root() / "outputs"


def default_outputs_root() -> Path:
    canonical = runs_root()
    legacy = legacy_outputs_root()
    if canonical.exists() or not legacy.exists():
        return canonical
    return legacy


def default_task_settings_file() -> Path:
    return settings_root() / "tasks" / "task_settings.yaml"


def default_train_source_settings_file() -> Path:
    return settings_root() / "source" / "train_source_policy_settings.yaml"


def default_adapt_ppo_settings_file() -> Path:
    return settings_root() / "adaptation" / "ppo.yaml"


def default_adapt_ewc_settings_file() -> Path:
    return settings_root() / "adaptation" / "ewc.yaml"


def default_adapt_rashomon_settings_file() -> Path:
    return settings_root() / "adaptation" / "rashomon.yaml"


def seed_run_dir(outputs_root: Path, task_setting: str, seed: int) -> Path:
    return outputs_root / task_setting / f"seed_{seed}"


def resolve_default_source_run_dir(outputs_root: Path, task_setting: str, seed: int) -> Path:
    """Prefer new layout; fall back to legacy layout if needed."""
    preferred = seed_run_dir(outputs_root, task_setting, seed) / "source"
    legacy = outputs_root / f"seed_{seed}" / "source"
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy

    global_legacy_root = legacy_outputs_root()
    if outputs_root != global_legacy_root:
        global_legacy = seed_run_dir(global_legacy_root, task_setting, seed) / "source"
        global_flat_legacy = global_legacy_root / f"seed_{seed}" / "source"
        if global_legacy.exists():
            return global_legacy
        if global_flat_legacy.exists():
            return global_flat_legacy
    return legacy


def resolve_policy_dir(
    outputs_root: Path,
    train_task_setting: str,
    train_seed: int,
    policy_subdir: str,
) -> Path:
    """Prefer new layout; fall back to legacy layout if needed."""
    preferred = outputs_root / train_task_setting / f"seed_{train_seed}" / policy_subdir
    legacy = outputs_root / f"seed_{train_seed}" / policy_subdir
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy

    global_legacy_root = legacy_outputs_root()
    if outputs_root != global_legacy_root:
        global_legacy = global_legacy_root / train_task_setting / f"seed_{train_seed}" / policy_subdir
        global_flat_legacy = global_legacy_root / f"seed_{train_seed}" / policy_subdir
        if global_legacy.exists():
            return global_legacy
        if global_flat_legacy.exists():
            return global_flat_legacy
    return legacy
