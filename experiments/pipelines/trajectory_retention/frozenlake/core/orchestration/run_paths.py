"""Centralized path helpers for FrozenLake experiments."""

from __future__ import annotations

from pathlib import Path


NOADAPT_POLICY_SUBDIR = "noadapt"
LEGACY_SOURCE_POLICY_SUBDIR = "source"

RL_CHOICES = ("ppo",)

SOURCE_MODE = "source"
DOWNSTREAM_MODES = ("downstream_unconstrained", "downstream_ewc", "downstream_rashomon")
ALL_MODES = (SOURCE_MODE, *DOWNSTREAM_MODES)

MODE_TO_DEFAULT_RUN_SUBDIR = {
    SOURCE_MODE: NOADAPT_POLICY_SUBDIR,
    "downstream_unconstrained": "downstream_unconstrained",
    "downstream_ewc": "downstream_ewc",
    "downstream_rashomon": "downstream_rashomon",
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
    ),
}


def validate_rl(rl: str) -> None:
    if rl not in RL_CHOICES:
        raise NotImplementedError(
            f"--rl '{rl}' is not implemented. Only {RL_CHOICES} exist today "
            f"(core/methods/ is PPO-only); add support there to use another algorithm.",
        )


def pipeline_root() -> Path:
    return Path(__file__).resolve().parents[2]


def settings_root() -> Path:
    return pipeline_root() / "settings"


def artifacts_root() -> Path:
    return pipeline_root() / "artifacts"


def runs_root() -> Path:
    return artifacts_root() / "runs"


def legacy_outputs_root() -> Path:
    return pipeline_root() / "outputs"


def default_outputs_root() -> Path:
    return runs_root()


def default_task_settings_file() -> Path:
    return default_task_pipelines_file()


def default_task_pipelines_file() -> Path:
    return settings_root() / "tasks" / "task_pipelines.yaml"


def default_task_definitions_file() -> Path:
    return settings_root() / "tasks" / "task_definitions.yaml"


def default_source_envs_file() -> Path:
    return settings_root() / "tasks" / "source_envs.yaml"


def default_downstream_envs_file() -> Path:
    return settings_root() / "tasks" / "downstream_envs.yaml"


def default_train_source_settings_file() -> Path:
    return settings_root() / "source" / "train_source_policy_settings.yaml"


def default_adapt_ppo_settings_file() -> Path:
    return settings_root() / "adaptation" / "ppo.yaml"


def default_adapt_ewc_settings_file() -> Path:
    return settings_root() / "adaptation" / "ewc.yaml"


def default_adapt_rashomon_settings_file() -> Path:
    return settings_root() / "adaptation" / "rashomon.yaml"


def seed_run_dir(outputs_root: Path, layout: str, seed: int, *, rl: str = "ppo") -> Path:
    return outputs_root / layout / rl / f"seed_{seed}"


def mode_run_dir(outputs_root: Path, layout: str, seed: int, mode: str, *, rl: str = "ppo") -> Path:
    return seed_run_dir(outputs_root, layout, seed, rl=rl) / MODE_TO_DEFAULT_RUN_SUBDIR[mode]


def is_mode_complete(outputs_root: Path, layout: str, seed: int, mode: str, *, rl: str = "ppo") -> bool:
    run_dir = mode_run_dir(outputs_root, layout, seed, mode, rl=rl)
    return all((run_dir / artifact).exists() for artifact in MODE_TO_REQUIRED_ARTIFACTS[mode])


def _policy_subdir_candidates(policy_subdir: str) -> tuple[str, ...]:
    if policy_subdir in {NOADAPT_POLICY_SUBDIR, LEGACY_SOURCE_POLICY_SUBDIR}:
        return (NOADAPT_POLICY_SUBDIR, LEGACY_SOURCE_POLICY_SUBDIR)
    if policy_subdir in {"downstream", "downstream_unconstrained"}:
        return ("downstream_unconstrained", "downstream")
    return (policy_subdir,)


def _pretag_seed_run_dir(outputs_root: Path, layout: str, seed: int) -> Path:
    """Path shape used before the <rl> tag was introduced: <outputs_root>/<layout>/seed_<n>."""
    return outputs_root / layout / f"seed_{seed}"


def resolve_default_source_run_dir(outputs_root: Path, layout: str, seed: int, *, rl: str = "ppo") -> Path:
    """Prefer the new noadapt layout; fall back to legacy source directories."""
    candidates = [
        seed_run_dir(outputs_root, layout, seed, rl=rl) / subdir
        for subdir in _policy_subdir_candidates(NOADAPT_POLICY_SUBDIR)
    ]
    pretag_candidates = [
        _pretag_seed_run_dir(outputs_root, layout, seed) / subdir
        for subdir in _policy_subdir_candidates(NOADAPT_POLICY_SUBDIR)
    ]
    legacy_candidates = [
        (outputs_root / f"seed_{seed}") / subdir
        for subdir in _policy_subdir_candidates(NOADAPT_POLICY_SUBDIR)
    ]
    for candidate in [*candidates, *pretag_candidates, *legacy_candidates]:
        if candidate.exists():
            return candidate

    global_legacy = legacy_outputs_root()
    if outputs_root != global_legacy:
        global_candidates = [
            seed_run_dir(global_legacy, layout, seed, rl=rl) / subdir
            for subdir in _policy_subdir_candidates(NOADAPT_POLICY_SUBDIR)
        ] + [
            _pretag_seed_run_dir(global_legacy, layout, seed) / subdir
            for subdir in _policy_subdir_candidates(NOADAPT_POLICY_SUBDIR)
        ] + [
            (global_legacy / f"seed_{seed}") / subdir
            for subdir in _policy_subdir_candidates(NOADAPT_POLICY_SUBDIR)
        ]
        for candidate in global_candidates:
            if candidate.exists():
                return candidate

    return seed_run_dir(outputs_root, layout, seed, rl=rl) / NOADAPT_POLICY_SUBDIR


def resolve_policy_dir(outputs_root: Path, layout: str, seed: int, policy_subdir: str, *, rl: str = "ppo") -> Path:
    """Resolve a policy directory with legacy fallback candidates."""
    candidate_subdirs = _policy_subdir_candidates(policy_subdir)
    candidates = [
        seed_run_dir(outputs_root, layout, seed, rl=rl) / subdir
        for subdir in candidate_subdirs
    ]
    pretag_candidates = [
        _pretag_seed_run_dir(outputs_root, layout, seed) / subdir
        for subdir in candidate_subdirs
    ]
    legacy_candidates = [
        (outputs_root / f"seed_{seed}") / subdir
        for subdir in candidate_subdirs
    ]
    for candidate in [*candidates, *pretag_candidates, *legacy_candidates]:
        if candidate.exists():
            return candidate

    global_legacy = legacy_outputs_root()
    if outputs_root != global_legacy:
        global_candidates = [
            seed_run_dir(global_legacy, layout, seed, rl=rl) / subdir
            for subdir in candidate_subdirs
        ] + [
            _pretag_seed_run_dir(global_legacy, layout, seed) / subdir
            for subdir in candidate_subdirs
        ] + [
            (global_legacy / f"seed_{seed}") / subdir
            for subdir in candidate_subdirs
        ]
        for candidate in global_candidates:
            if candidate.exists():
                return candidate

    return seed_run_dir(outputs_root, layout, seed, rl=rl) / candidate_subdirs[0]
