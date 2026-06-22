"""Task settings and dynamics resolution for LunarLander experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


_TASK_DEFINITION_KEYS = (
    "env_id",
    "continuous",
    "gravity",
    "enable_wind",
    "wind_power",
    "turbulence_power",
    "initial_random_strength",
    "dispersion_strength",
    "main_engine_power",
    "side_engine_power",
    "leg_spring_torque",
    "lander_mass_scale",
    "leg_mass_scale",
    "linear_damping",
    "angular_damping",
    "terrain_heights",
    "action_repeat",
    "action_delay",
    "action_noise_prob",
    "action_noise_mode",
    "mark_out_of_viewport_as_unsafe",
)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data)}.")
    return data


def _role_task_id(task_role: str) -> float:
    if task_role == "source":
        return 0.0
    if task_role == "downstream":
        return 1.0
    raise ValueError(f"task_role must be 'source' or 'downstream', got '{task_role}'.")


def _build_env_cfg(
    *,
    env_cfg: dict[str, Any],
    task_role: str,
    append_task_id: bool,
    settings_format: str,
    task_pipelines_file: Path | None,
    task_definitions_file: Path | None,
    resolved_pipeline_name: str | None,
    resolved_definition_name: str,
) -> dict[str, Any]:
    cfg = {
        "env_id": env_cfg.get("env_id"),
        "gravity": env_cfg.get("gravity"),
        "task_id": _role_task_id(task_role),
        "enable_wind": env_cfg.get("enable_wind"),
        "wind_power": env_cfg.get("wind_power"),
        "turbulence_power": env_cfg.get("turbulence_power"),
        "initial_random_strength": env_cfg.get("initial_random_strength"),
        "dispersion_strength": env_cfg.get("dispersion_strength"),
        "main_engine_power": env_cfg.get("main_engine_power"),
        "side_engine_power": env_cfg.get("side_engine_power"),
        "leg_spring_torque": env_cfg.get("leg_spring_torque"),
        "lander_mass_scale": env_cfg.get("lander_mass_scale"),
        "leg_mass_scale": env_cfg.get("leg_mass_scale"),
        "linear_damping": env_cfg.get("linear_damping"),
        "angular_damping": env_cfg.get("angular_damping"),
        "terrain_heights": env_cfg.get("terrain_heights"),
        "action_repeat": env_cfg.get("action_repeat"),
        "action_delay": env_cfg.get("action_delay"),
        "action_noise_prob": env_cfg.get("action_noise_prob"),
        "action_noise_mode": env_cfg.get("action_noise_mode"),
        "mark_out_of_viewport_as_unsafe": env_cfg.get("mark_out_of_viewport_as_unsafe"),
        "append_task_id": bool(append_task_id),
        "continuous": env_cfg.get("continuous"),
        "_task_settings_format": settings_format,
        "_task_pipelines_file": (str(task_pipelines_file) if task_pipelines_file is not None else None),
        "_task_definitions_file": (str(task_definitions_file) if task_definitions_file is not None else None),
        "_resolved_pipeline_name": resolved_pipeline_name,
        "_resolved_definition_name": resolved_definition_name,
    }
    return cfg


def _is_pipeline_entry(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    source_cfg = entry.get("source")
    downstream_cfg = entry.get("downstream")
    if not isinstance(source_cfg, dict) or not isinstance(downstream_cfg, dict):
        return False
    return "env" in source_cfg and "env" in downstream_cfg


def _looks_like_split_pipeline_map(all_settings: dict[str, Any]) -> bool:
    for value in all_settings.values():
        if _is_pipeline_entry(value):
            return True
    return False


def _looks_like_legacy_monolithic_map(all_settings: dict[str, Any]) -> bool:
    for value in all_settings.values():
        if not isinstance(value, dict):
            continue
        if isinstance(value.get("source"), dict) and isinstance(value.get("downstream"), dict):
            return True
    return False


def _default_task_definitions_file(task_settings_file: Path) -> Path:
    if task_settings_file.name == "task_pipelines.yaml":
        return task_settings_file.with_name("task_definitions.yaml")
    return task_settings_file.parent / "task_definitions.yaml"


def _resolve_definition_cfg(
    *,
    task_definitions_file: Path,
    definition_name: str,
) -> dict[str, Any]:
    if not task_definitions_file.exists():
        raise FileNotFoundError(f"Task definitions file not found: {task_definitions_file}")
    all_definitions = _load_yaml_mapping(task_definitions_file)
    if definition_name not in all_definitions:
        raise ValueError(
            f"Task definition '{definition_name}' not found in {task_definitions_file}.",
        )
    definition_cfg = all_definitions[definition_name]
    if not isinstance(definition_cfg, dict):
        raise ValueError(
            f"Expected mapping for definition '{definition_name}' in {task_definitions_file}, "
            f"got {type(definition_cfg)}.",
        )
    return definition_cfg


def resolve_lunarlander_dynamics(cfg: dict[str, Any], *, cfg_name: str = "config") -> dict[str, Any]:
    """Normalize and validate dynamics-shift parameters for LunarLander."""
    enable_wind_raw = cfg.get("enable_wind", False)
    enable_wind = False if enable_wind_raw is None else bool(enable_wind_raw)
    wind_power_raw = cfg.get("wind_power", None)
    turbulence_power_raw = cfg.get("turbulence_power", None)
    wind_power = None if wind_power_raw is None else float(wind_power_raw)
    turbulence_power = None if turbulence_power_raw is None else float(turbulence_power_raw)
    initial_random_strength_raw = cfg.get("initial_random_strength", None)
    initial_random_strength = (
        None if initial_random_strength_raw is None else float(initial_random_strength_raw)
    )
    if initial_random_strength is not None and initial_random_strength < 0.0:
        raise ValueError(
            f"{cfg_name}: initial_random_strength must be >= 0, got {initial_random_strength}.",
        )
    dispersion_strength_raw = cfg.get("dispersion_strength", None)
    dispersion_strength = None if dispersion_strength_raw is None else float(dispersion_strength_raw)
    if dispersion_strength is not None and dispersion_strength < 0.0:
        raise ValueError(
            f"{cfg_name}: dispersion_strength must be >= 0, got {dispersion_strength}.",
        )
    main_engine_power_raw = cfg.get("main_engine_power", None)
    main_engine_power = None if main_engine_power_raw is None else float(main_engine_power_raw)
    if main_engine_power is not None and main_engine_power < 0.0:
        raise ValueError(
            f"{cfg_name}: main_engine_power must be >= 0, got {main_engine_power}.",
        )
    side_engine_power_raw = cfg.get("side_engine_power", None)
    side_engine_power = None if side_engine_power_raw is None else float(side_engine_power_raw)
    if side_engine_power is not None and side_engine_power < 0.0:
        raise ValueError(
            f"{cfg_name}: side_engine_power must be >= 0, got {side_engine_power}.",
        )
    leg_spring_torque_raw = cfg.get("leg_spring_torque", None)
    leg_spring_torque = None if leg_spring_torque_raw is None else float(leg_spring_torque_raw)
    if leg_spring_torque is not None and leg_spring_torque < 0.0:
        raise ValueError(
            f"{cfg_name}: leg_spring_torque must be >= 0, got {leg_spring_torque}.",
        )
    lander_mass_scale_raw = cfg.get("lander_mass_scale", None)
    lander_mass_scale = None if lander_mass_scale_raw is None else float(lander_mass_scale_raw)
    if lander_mass_scale is not None and lander_mass_scale <= 0.0:
        raise ValueError(
            f"{cfg_name}: lander_mass_scale must be > 0, got {lander_mass_scale}.",
        )
    leg_mass_scale_raw = cfg.get("leg_mass_scale", None)
    leg_mass_scale = None if leg_mass_scale_raw is None else float(leg_mass_scale_raw)
    if leg_mass_scale is not None and leg_mass_scale <= 0.0:
        raise ValueError(
            f"{cfg_name}: leg_mass_scale must be > 0, got {leg_mass_scale}.",
        )
    linear_damping_raw = cfg.get("linear_damping", None)
    linear_damping = None if linear_damping_raw is None else float(linear_damping_raw)
    if linear_damping is not None and linear_damping < 0.0:
        raise ValueError(
            f"{cfg_name}: linear_damping must be >= 0, got {linear_damping}.",
        )
    angular_damping_raw = cfg.get("angular_damping", None)
    angular_damping = None if angular_damping_raw is None else float(angular_damping_raw)
    if angular_damping is not None and angular_damping < 0.0:
        raise ValueError(
            f"{cfg_name}: angular_damping must be >= 0, got {angular_damping}.",
        )
    terrain_heights_raw = cfg.get("terrain_heights", None)
    terrain_heights: list[float] | None
    if terrain_heights_raw is None:
        terrain_heights = None
    else:
        if not isinstance(terrain_heights_raw, (list, tuple, np.ndarray)):
            raise ValueError(
                f"{cfg_name}: terrain_heights must be a sequence of numbers or null, "
                f"got {type(terrain_heights_raw)}.",
            )
        terrain_heights = [float(v) for v in terrain_heights_raw]

    action_repeat_raw = cfg.get("action_repeat", 1)
    action_repeat = 1 if action_repeat_raw is None else int(action_repeat_raw)
    if action_repeat < 0:
        raise ValueError(f"{cfg_name}: action_repeat must be >= 0, got {action_repeat}.")

    action_delay_raw = cfg.get("action_delay", 0)
    action_delay = 0 if action_delay_raw is None else int(action_delay_raw)
    if action_delay < 0:
        raise ValueError(f"{cfg_name}: action_delay must be >= 0, got {action_delay}.")

    action_noise_prob_raw = cfg.get("action_noise_prob", 0.0)
    action_noise_prob = 0.0 if action_noise_prob_raw is None else float(action_noise_prob_raw)
    if action_noise_prob < 0.0 or action_noise_prob > 1.0:
        raise ValueError(
            f"{cfg_name}: action_noise_prob must be in [0, 1], got {action_noise_prob}.",
        )

    action_noise_mode_raw = cfg.get("action_noise_mode", "noop")
    action_noise_mode = "noop" if action_noise_mode_raw is None else str(action_noise_mode_raw)
    if action_noise_mode not in {"noop", "previous", "random"}:
        raise ValueError(
            f"{cfg_name}: unsupported action_noise_mode '{action_noise_mode}'. "
            "Expected one of ['noop', 'previous', 'random'].",
        )
    mark_out_of_viewport_as_unsafe_raw = cfg.get("mark_out_of_viewport_as_unsafe", False)
    mark_out_of_viewport_as_unsafe = (
        False
        if mark_out_of_viewport_as_unsafe_raw is None
        else bool(mark_out_of_viewport_as_unsafe_raw)
    )

    return {
        "enable_wind": enable_wind,
        "wind_power": wind_power,
        "turbulence_power": turbulence_power,
        "initial_random_strength": initial_random_strength,
        "dispersion_strength": dispersion_strength,
        "main_engine_power": main_engine_power,
        "side_engine_power": side_engine_power,
        "leg_spring_torque": leg_spring_torque,
        "lander_mass_scale": lander_mass_scale,
        "leg_mass_scale": leg_mass_scale,
        "linear_damping": linear_damping,
        "angular_damping": angular_damping,
        "terrain_heights": terrain_heights,
        "action_repeat": action_repeat,
        "action_delay": action_delay,
        "action_noise_prob": action_noise_prob,
        "action_noise_mode": action_noise_mode,
        "mark_out_of_viewport_as_unsafe": mark_out_of_viewport_as_unsafe,
    }


def load_task_settings(
    settings_file: Path,
    setting_name: str,
    task_role: str,
) -> dict[str, Any]:
    if task_role not in ("source", "downstream"):
        raise ValueError(f"task_role must be 'source' or 'downstream', got '{task_role}'.")

    if not settings_file.exists():
        return {}
    all_settings = _load_yaml_mapping(settings_file)

    # New split schema: task_pipelines.yaml + task_definitions.yaml.
    if _looks_like_split_pipeline_map(all_settings):
        if setting_name in all_settings:
            pipeline_cfg = all_settings[setting_name]
            if not _is_pipeline_entry(pipeline_cfg):
                raise ValueError(
                    f"Expected split-pipeline entry for setting '{setting_name}' in {settings_file}.",
                )
            role_cfg = pipeline_cfg[task_role]
            definition_name = role_cfg.get("env")
            if not isinstance(definition_name, str) or len(definition_name.strip()) == 0:
                raise ValueError(
                    f"Expected non-empty string at {settings_file}:{setting_name}:{task_role}.env",
                )
            definitions_file = _default_task_definitions_file(settings_file)
            definition_cfg = _resolve_definition_cfg(
                task_definitions_file=definitions_file,
                definition_name=definition_name,
            )
            append_task_id = bool(pipeline_cfg.get("append_task_id", True))
            return _build_env_cfg(
                env_cfg=definition_cfg,
                task_role=task_role,
                append_task_id=append_task_id,
                settings_format="split_pipeline",
                task_pipelines_file=settings_file,
                task_definitions_file=definitions_file,
                resolved_pipeline_name=setting_name,
                resolved_definition_name=definition_name,
            )

        # For eval/video scripts, support direct environment definition lookup.
        definitions_file = _default_task_definitions_file(settings_file)
        if definitions_file.exists():
            all_definitions = _load_yaml_mapping(definitions_file)
            if setting_name in all_definitions:
                definition_cfg = all_definitions[setting_name]
                if not isinstance(definition_cfg, dict):
                    raise ValueError(
                        f"Expected mapping for definition '{setting_name}' in {definitions_file}, "
                        f"got {type(definition_cfg)}.",
                    )
                return _build_env_cfg(
                    env_cfg=definition_cfg,
                    task_role=task_role,
                    append_task_id=True,
                    settings_format="split_definition_direct",
                    task_pipelines_file=settings_file,
                    task_definitions_file=definitions_file,
                    resolved_pipeline_name=None,
                    resolved_definition_name=setting_name,
                )

        resolved_pipeline_name = "default"
        if resolved_pipeline_name in all_settings:
            pipeline_cfg = all_settings[resolved_pipeline_name]
            if not _is_pipeline_entry(pipeline_cfg):
                raise ValueError(
                    f"Expected split-pipeline entry for setting '{resolved_pipeline_name}' in {settings_file}.",
                )
            role_cfg = pipeline_cfg[task_role]
            definition_name = role_cfg.get("env")
            if not isinstance(definition_name, str) or len(definition_name.strip()) == 0:
                raise ValueError(
                    f"Expected non-empty string at {settings_file}:{resolved_pipeline_name}:{task_role}.env",
                )
            definition_cfg = _resolve_definition_cfg(
                task_definitions_file=definitions_file,
                definition_name=definition_name,
            )
            append_task_id = bool(pipeline_cfg.get("append_task_id", True))
            return _build_env_cfg(
                env_cfg=definition_cfg,
                task_role=task_role,
                append_task_id=append_task_id,
                settings_format="split_pipeline_default_fallback",
                task_pipelines_file=settings_file,
                task_definitions_file=definitions_file,
                resolved_pipeline_name=resolved_pipeline_name,
                resolved_definition_name=definition_name,
            )
        raise ValueError(f"Task setting '{setting_name}' not found in {settings_file}.")

    resolved_setting_name = setting_name if setting_name in all_settings else "default"
    if resolved_setting_name not in all_settings:
        raise ValueError(f"Task setting '{setting_name}' not found in {settings_file}.")

    setting_cfg = all_settings[resolved_setting_name] or {}
    if not isinstance(setting_cfg, dict):
        raise ValueError(
            f"Expected mapping for task setting '{resolved_setting_name}' in {settings_file}, "
            f"got {type(setting_cfg)}.",
        )

    # Legacy monolithic schema: one setting contains source/downstream sub-mappings.
    if _looks_like_legacy_monolithic_map({resolved_setting_name: setting_cfg}):
        role_cfg = setting_cfg.get(task_role, {}) or {}
        if not isinstance(role_cfg, dict):
            raise ValueError(
                f"Expected mapping for role '{task_role}' in setting '{resolved_setting_name}' "
                f"from {settings_file}.",
            )
        merged_env_cfg = {
            "env_id": setting_cfg.get("env_id"),
            "continuous": setting_cfg.get("continuous"),
        }
        for key in _TASK_DEFINITION_KEYS:
            if key in role_cfg:
                merged_env_cfg[key] = role_cfg.get(key)
        append_task_id = bool(setting_cfg.get("append_task_id", True))
        return _build_env_cfg(
            env_cfg=merged_env_cfg,
            task_role=task_role,
            append_task_id=append_task_id,
            settings_format="legacy_monolithic",
            task_pipelines_file=settings_file,
            task_definitions_file=None,
            resolved_pipeline_name=resolved_setting_name,
            resolved_definition_name=f"{resolved_setting_name}:{task_role}",
        )

    # Direct definitions file lookup (or definition-like maps).
    return _build_env_cfg(
        env_cfg=setting_cfg,
        task_role=task_role,
        append_task_id=True,
        settings_format="definition_direct",
        task_pipelines_file=None,
        task_definitions_file=settings_file,
        resolved_pipeline_name=None,
        resolved_definition_name=resolved_setting_name,
    )


# Backward-compatible aliases for older imports.
_resolve_lunarlander_dynamics = resolve_lunarlander_dynamics
_load_task_settings = load_task_settings
