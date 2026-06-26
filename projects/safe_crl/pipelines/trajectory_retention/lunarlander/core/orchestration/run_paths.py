"""Centralized path helpers for LunarLander projects.safe_crl."""

from __future__ import annotations

from pathlib import Path


NOADAPT_POLICY_SUBDIR = "noadapt"
LEGACY_SOURCE_POLICY_SUBDIR = "source"

RL_CHOICES = ("ppo",)

SOURCE_MODE = "source"
DOWNSTREAM_MODES = (
    "downstream_unconstrained",
    "downstream_ewc",
    "downstream_rashomon",
    "downstream_rashomon_expanded",
    "downstream_rashomon_plus",
)
ALL_MODES = (SOURCE_MODE, *DOWNSTREAM_MODES)

MODE_TO_DEFAULT_RUN_SUBDIR = {
    SOURCE_MODE: NOADAPT_POLICY_SUBDIR,
    "downstream_unconstrained": "downstream_unconstrained",
    "downstream_ewc": "downstream_ewc",
    "downstream_rashomon": "downstream_rashomon",
    "downstream_rashomon_expanded": "downstream_rashomon_expanded",
    "downstream_rashomon_plus": "downstream_rashomon_plus",
}

MODE_TO_REQUIRED_ARTIFACTS = {
    SOURCE_MODE: ("actor.pt", "critic.pt", "training_data.pt", "run_summary.yaml"),
    "downstream_unconstrained": ("actor.pt", "critic.pt", "training_data.pt", "run_summary.yaml"),
    "downstream_ewc": ("actor.pt", "critic.pt", "training_data.pt", "run_summary.yaml", "ewc_state.pt"),
    "downstream_rashomon": (
        "actor.pt",
        "critic.pt",
        "training_data.pt",
        "run_summary.yaml",
        "rashomon_dataset.pt",
        "rashomon_bounded_model.pt",
        "rashomon_param_bounds.pt",
        "rashomon_rollout_stats.yaml",
    ),
    "downstream_rashomon_expanded": (
        "actor.pt",
        "critic.pt",
        "training_data.pt",
        "run_summary.yaml",
        "rashomon_dataset.pt",
        "rashomon_bounded_model.pt",
        "rashomon_param_bounds.pt",
        "rashomon_union_interval_param_bounds.pt",
        "rashomon_rollout_stats.yaml",
    ),
    "downstream_rashomon_plus": (
        "actor.pt",
        "critic.pt",
        "training_data.pt",
        "run_summary.yaml",
        "rashomon_dataset.pt",
        "rashomon_bounded_model.pt",
        "rashomon_param_bounds.pt",
        "second_rashomon_bounded_model.pt",
        "second_rashomon_param_bounds.pt",
        "rashomon_union_interval_param_bounds.pt",
        "sampled_reference_actor.pt",
        "rashomon_rollout_stats.yaml",
    ),
}


def validate_rl(rl: str) -> None:
    if rl not in RL_CHOICES:
        raise NotImplementedError(f"Unsupported --rl={rl!r}. Supported values: {RL_CHOICES}.")


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


def seed_run_dir(outputs_root: Path, task_setting: str, seed: int, *, rl: str = "ppo") -> Path:
    return outputs_root / task_setting / rl / f"seed_{seed}"


def mode_run_dir(outputs_root: Path, task_setting: str, seed: int, mode: str, *, rl: str = "ppo") -> Path:
    return seed_run_dir(outputs_root, task_setting, seed, rl=rl) / MODE_TO_DEFAULT_RUN_SUBDIR[mode]


def is_mode_complete(outputs_root: Path, task_setting: str, seed: int, mode: str, *, rl: str = "ppo") -> bool:
    required = MODE_TO_REQUIRED_ARTIFACTS[mode]
    run_dir = mode_run_dir(outputs_root, task_setting, seed, mode, rl=rl)
    return all((run_dir / artifact).exists() for artifact in required)


def _policy_subdir_candidates(policy_subdir: str) -> tuple[str, ...]:
    if policy_subdir in {NOADAPT_POLICY_SUBDIR, LEGACY_SOURCE_POLICY_SUBDIR}:
        return (NOADAPT_POLICY_SUBDIR, LEGACY_SOURCE_POLICY_SUBDIR)
    return (policy_subdir,)


def _pretag_seed_run_dir(outputs_root: Path, task_setting: str, seed: int) -> Path:
    return outputs_root / task_setting / f"seed_{seed}"


def resolve_default_source_run_dir(
    outputs_root: Path,
    task_setting: str,
    seed: int,
    *,
    rl: str = "ppo",
) -> Path:
    """Prefer the rl-tagged layout; fall back to pretag/legacy layouts if needed."""
    candidate_subdirs = _policy_subdir_candidates(NOADAPT_POLICY_SUBDIR)

    tagged_candidates = [
        seed_run_dir(outputs_root, task_setting, seed, rl=rl) / subdir for subdir in candidate_subdirs
    ]
    pretag_candidates = [
        _pretag_seed_run_dir(outputs_root, task_setting, seed) / subdir for subdir in candidate_subdirs
    ]
    legacy_candidates = [(outputs_root / f"seed_{seed}") / subdir for subdir in candidate_subdirs]

    for candidate in [*tagged_candidates, *pretag_candidates, *legacy_candidates]:
        if candidate.exists():
            return candidate

    global_legacy_root = legacy_outputs_root()
    if outputs_root != global_legacy_root:
        global_candidates = (
            [
                seed_run_dir(global_legacy_root, task_setting, seed, rl=rl) / subdir
                for subdir in candidate_subdirs
            ]
            + [
                _pretag_seed_run_dir(global_legacy_root, task_setting, seed) / subdir
                for subdir in candidate_subdirs
            ]
            + [(global_legacy_root / f"seed_{seed}") / subdir for subdir in candidate_subdirs]
        )
        for candidate in global_candidates:
            if candidate.exists():
                return candidate

    # Default to the canonical new location for fresh runs.
    return seed_run_dir(outputs_root, task_setting, seed, rl=rl) / NOADAPT_POLICY_SUBDIR


def resolve_policy_dir(
    outputs_root: Path,
    train_task_setting: str,
    train_seed: int,
    policy_subdir: str,
    *,
    rl: str = "ppo",
) -> Path:
    """Prefer the rl-tagged layout; fall back to pretag/legacy layouts if needed."""
    candidate_subdirs = _policy_subdir_candidates(policy_subdir)

    tagged_candidates = [
        seed_run_dir(outputs_root, train_task_setting, train_seed, rl=rl) / subdir
        for subdir in candidate_subdirs
    ]
    pretag_candidates = [
        outputs_root / train_task_setting / f"seed_{train_seed}" / subdir for subdir in candidate_subdirs
    ]
    legacy_candidates = [outputs_root / f"seed_{train_seed}" / subdir for subdir in candidate_subdirs]

    for candidate in [*tagged_candidates, *pretag_candidates, *legacy_candidates]:
        if candidate.exists():
            return candidate

    global_legacy_root = legacy_outputs_root()
    if outputs_root != global_legacy_root:
        global_candidates = (
            [
                seed_run_dir(global_legacy_root, train_task_setting, train_seed, rl=rl) / subdir
                for subdir in candidate_subdirs
            ]
            + [
                global_legacy_root / train_task_setting / f"seed_{train_seed}" / subdir
                for subdir in candidate_subdirs
            ]
            + [global_legacy_root / f"seed_{train_seed}" / subdir for subdir in candidate_subdirs]
        )
        for candidate in global_candidates:
            if candidate.exists():
                return candidate

    return seed_run_dir(outputs_root, train_task_setting, train_seed, rl=rl) / candidate_subdirs[0]
