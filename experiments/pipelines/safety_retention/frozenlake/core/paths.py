"""Path helpers for FrozenLake safety runs."""

from __future__ import annotations

from pathlib import Path


NOADAPT_POLICY_SUBDIR = "noadapt"
SOURCE_POLICY_SUBDIR = NOADAPT_POLICY_SUBDIR
DOWNSTREAM_UNCONSTRAINED_SUBDIR = "downstream_unconstrained"
DOWNSTREAM_EWC_SUBDIR = "downstream_ewc"
DOWNSTREAM_RASHOMON_SUBDIR = "downstream_rashomon"

MODE_TO_REQUIRED_ARTIFACTS = {
    "source": ("actor.pt", "critic.pt", "training_data.pt", "run_summary.yaml", "rashomon_dataset.pt"),
    "downstream_unconstrained": ("actor.pt", "critic.pt", "training_data.pt", "run_summary.yaml"),
    "downstream_ewc": ("actor.pt", "critic.pt", "training_data.pt", "run_summary.yaml", "ewc_state.pt"),
    "downstream_rashomon": (
        "actor.pt",
        "critic.pt",
        "training_data.pt",
        "run_summary.yaml",
        "rashomon_dataset.pt",
        "rashomon_param_bounds.pt",
    ),
}


def pipeline_root() -> Path:
    return Path(__file__).resolve().parents[1]


def artifacts_root() -> Path:
    return pipeline_root() / "artifacts"


def default_outputs_root() -> Path:
    return artifacts_root() / "runs"


def run_settings_tag(rl: str, deterministic: bool) -> str:
    """Tag distinguishing artifacts produced with different --rl/--deterministic settings."""
    dynamics = "deterministic" if deterministic else "stochastic"
    return f"{rl}_{dynamics}"


def layout_run_root(outputs_root: Path, layout: str, rl: str, deterministic: bool) -> Path:
    return outputs_root / layout / run_settings_tag(rl, deterministic)


def seed_run_dir(outputs_root: Path, layout: str, rl: str, deterministic: bool, seed: int) -> Path:
    return layout_run_root(outputs_root, layout, rl, deterministic) / f"seed_{seed}"


def source_run_dir(outputs_root: Path, layout: str, rl: str, deterministic: bool, seed: int) -> Path:
    return seed_run_dir(outputs_root, layout, rl, deterministic, seed) / NOADAPT_POLICY_SUBDIR


def resolve_source_run_dir(outputs_root: Path, layout: str, rl: str, deterministic: bool, seed: int) -> Path:
    canonical = source_run_dir(outputs_root, layout, rl, deterministic, seed)
    if canonical.exists():
        return canonical
    legacy = seed_run_dir(outputs_root, layout, rl, deterministic, seed) / "source"
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


def mode_run_dir(outputs_root: Path, layout: str, rl: str, deterministic: bool, seed: int, mode: str) -> Path:
    return seed_run_dir(outputs_root, layout, rl, deterministic, seed) / mode_run_subdir(mode)


def is_mode_complete(outputs_root: Path, layout: str, rl: str, deterministic: bool, seed: int, mode: str) -> bool:
    run_dir = mode_run_dir(outputs_root, layout, rl, deterministic, seed, mode)
    return all((run_dir / artifact).exists() for artifact in MODE_TO_REQUIRED_ARTIFACTS[mode])
