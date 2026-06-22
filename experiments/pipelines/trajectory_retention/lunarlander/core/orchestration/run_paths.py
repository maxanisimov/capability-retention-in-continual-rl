"""Centralized path helpers for LunarLander experiments."""

from __future__ import annotations

from pathlib import Path


NOADAPT_POLICY_SUBDIR = "noadapt"
LEGACY_SOURCE_POLICY_SUBDIR = "source"


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
    return default_task_pipelines_file()


def default_task_pipelines_file() -> Path:
    return settings_root() / "tasks" / "task_pipelines.yaml"


def default_task_definitions_file() -> Path:
    return settings_root() / "tasks" / "task_definitions.yaml"


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


def _policy_subdir_candidates(policy_subdir: str) -> tuple[str, ...]:
    if policy_subdir in {NOADAPT_POLICY_SUBDIR, LEGACY_SOURCE_POLICY_SUBDIR}:
        return (NOADAPT_POLICY_SUBDIR, LEGACY_SOURCE_POLICY_SUBDIR)
    return (policy_subdir,)


def resolve_default_source_run_dir(outputs_root: Path, task_setting: str, seed: int) -> Path:
    """Prefer new layout; fall back to legacy layout if needed."""
    preferred_candidates = [
        seed_run_dir(outputs_root, task_setting, seed) / subdir
        for subdir in _policy_subdir_candidates(NOADAPT_POLICY_SUBDIR)
    ]
    legacy_candidates = [
        (outputs_root / f"seed_{seed}") / subdir
        for subdir in _policy_subdir_candidates(NOADAPT_POLICY_SUBDIR)
    ]

    for candidate in [*preferred_candidates, *legacy_candidates]:
        if candidate.exists():
            return candidate

    global_legacy_root = legacy_outputs_root()
    if outputs_root != global_legacy_root:
        global_candidates = [
            seed_run_dir(global_legacy_root, task_setting, seed) / subdir
            for subdir in _policy_subdir_candidates(NOADAPT_POLICY_SUBDIR)
        ] + [
            (global_legacy_root / f"seed_{seed}") / subdir
            for subdir in _policy_subdir_candidates(NOADAPT_POLICY_SUBDIR)
        ]
        for candidate in global_candidates:
            if candidate.exists():
                return candidate

    # Default to the canonical new location for fresh runs.
    return seed_run_dir(outputs_root, task_setting, seed) / NOADAPT_POLICY_SUBDIR


def resolve_policy_dir(
    outputs_root: Path,
    train_task_setting: str,
    train_seed: int,
    policy_subdir: str,
) -> Path:
    """Prefer new layout; fall back to legacy layout if needed."""
    candidate_subdirs = _policy_subdir_candidates(policy_subdir)

    preferred_candidates = [
        outputs_root / train_task_setting / f"seed_{train_seed}" / subdir
        for subdir in candidate_subdirs
    ]
    legacy_candidates = [
        outputs_root / f"seed_{train_seed}" / subdir
        for subdir in candidate_subdirs
    ]
    for candidate in [*preferred_candidates, *legacy_candidates]:
        if candidate.exists():
            return candidate

    global_legacy_root = legacy_outputs_root()
    if outputs_root != global_legacy_root:
        global_candidates = [
            global_legacy_root / train_task_setting / f"seed_{train_seed}" / subdir
            for subdir in candidate_subdirs
        ] + [
            global_legacy_root / f"seed_{train_seed}" / subdir
            for subdir in candidate_subdirs
        ]
        for candidate in global_candidates:
            if candidate.exists():
                return candidate

    return outputs_root / train_task_setting / f"seed_{train_seed}" / candidate_subdirs[0]
