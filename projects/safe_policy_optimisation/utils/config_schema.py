"""Typed schema + validation for pipeline and task settings.

The YAML settings remain the source of truth, but they are validated against
these dataclasses before a run starts. This gives a single, explicit definition
of every allowed section/field and fails fast on typos, missing required keys, or
stray sections instead of silently producing a different experiment.

The dataclasses describe the structured (sectioned) YAML; the existing flat
argparse-namespace flow in :mod:`...utils.config` is unchanged.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


class SettingsValidationError(ValueError):
    """Raised when a pipeline/task mapping does not match the schema."""


def _from_dict(cls: type[T], data: Any, *, ctx: str) -> T:
    """Build a dataclass from a mapping, rejecting unknown/missing keys."""

    if not isinstance(data, dict):
        raise SettingsValidationError(f"{ctx}: expected a mapping, got {type(data).__name__}.")
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise SettingsValidationError(
            f"{ctx}: unknown key(s) {sorted(unknown)}; allowed: {sorted(field_names)}."
        )
    try:
        return cls(**data)  # type: ignore[call-arg]
    except TypeError as exc:  # missing required field
        raise SettingsValidationError(f"{ctx}: {exc}") from exc


@dataclass
class OutputCfg:
    output_dir: str
    run_id: str


@dataclass
class RuntimeCfg:
    seed: int
    device: str


@dataclass
class ShieldSynthesisCfg:
    constraint: str
    init_safety_bound: float
    theta: float
    max_vi_steps: int
    granularity: int
    unsafe_cost_threshold: float
    constraint_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyCfg:
    cost_limit: float
    shield_path: str


@dataclass
class SafeRlBaselinesCfg:
    algorithms: list[str]


@dataclass
class TrainingCfg:
    total_timesteps: int
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    # Optional extras used by the paper-scale configs.
    gae_lambda: float | None = None
    clip_range: float | None = None
    max_grad_norm: float | None = None
    ent_coef: float | None = None
    vf_coef: float | None = None
    cost_gamma: float | None = None
    cost_gae_lambda: float | None = None
    lagrangian_multiplier_init: float | None = None


@dataclass
class RashomonSetCfg:
    rashomon_dir: str
    rashomon_n_iters: int
    rashomon_checkpoint: int
    rashomon_batch_size: int
    certificate_samples: int


@dataclass
class EarlyStoppingCfg:
    early_stop_eval_freq: int
    early_stop_eval_episodes: int
    early_stop_success_rate: float
    success_reward_threshold: float


@dataclass
class EvaluationCfg:
    eval_episodes: int
    shielded_evaluation_policy: str
    rashomon_evaluation_policy: str


@dataclass
class MonitoringCfg:
    curve_eval_freq: int
    curve_eval_episodes: int


@dataclass
class StagesCfg:
    skip_baselines: bool
    skip_shielded_policy: bool
    skip_rashomon_policy: bool


@dataclass
class TaskCfg:
    env_id: str
    description: str | None = None
    shield_task: str | None = None
    max_episode_steps: int | None = None
    env_kwargs: dict[str, Any] = field(default_factory=dict)


# section name -> dataclass; sections absent from a pipeline are simply skipped.
_SECTION_TYPES: dict[str, type] = {
    "output": OutputCfg,
    "runtime": RuntimeCfg,
    "shield_synthesis": ShieldSynthesisCfg,
    "safety": SafetyCfg,
    "safe_rl_baselines": SafeRlBaselinesCfg,
    "training": TrainingCfg,
    "rashomon_set": RashomonSetCfg,
    "early_stopping": EarlyStoppingCfg,
    "evaluation": EvaluationCfg,
    "monitoring": MonitoringCfg,
    "stages": StagesCfg,
}

_PIPELINE_METADATA_KEYS = {"description", "module", "default_task"}


@dataclass
class PipelineCfg:
    """Validated view of one pipeline's structured settings."""

    name: str
    default_task: str | None
    sections: dict[str, Any]
    module: str | None = None
    description: str | None = None


def validate_pipeline_mapping(name: str, mapping: dict[str, Any]) -> PipelineCfg:
    """Validate one pipeline mapping against the section schema."""

    if not isinstance(mapping, dict):
        raise SettingsValidationError(f"pipeline {name!r}: expected a mapping.")
    unknown = set(mapping) - _PIPELINE_METADATA_KEYS - set(_SECTION_TYPES)
    if unknown:
        raise SettingsValidationError(
            f"pipeline {name!r}: unknown section/key(s) {sorted(unknown)}; "
            f"allowed sections: {sorted(_SECTION_TYPES)}."
        )
    sections: dict[str, Any] = {}
    for section_name, section_type in _SECTION_TYPES.items():
        if section_name in mapping and mapping[section_name] is not None:
            sections[section_name] = _from_dict(
                section_type, mapping[section_name], ctx=f"pipeline {name!r} -> {section_name}"
            )
    return PipelineCfg(
        name=name,
        default_task=mapping.get("default_task"),
        module=mapping.get("module"),
        description=mapping.get("description"),
        sections=sections,
    )


def validate_task_mapping(name: str, mapping: dict[str, Any]) -> TaskCfg:
    """Validate one task mapping against :class:`TaskCfg`."""

    return _from_dict(TaskCfg, mapping, ctx=f"task {name!r}")
