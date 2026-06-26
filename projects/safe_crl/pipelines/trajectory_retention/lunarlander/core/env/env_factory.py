"""LunarLander environment construction helpers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from projects.safe_crl.pipelines.trajectory_retention.lunarlander.core.env.tunable_lunarlander import (
    ensure_lunarlander_v4_registered,
)
from projects.safe_crl.pipelines.trajectory_retention.lunarlander.core.env.wrappers import (
    ActionDelayWrapper,
    ActionNoiseWrapper,
    ActionRepeatWrapper,
    AppendTaskIDObservationWrapper,
    LunarLanderCrashSafetyWrapper,
)


def _retry_without_unsupported_kwargs(
    err: TypeError,
    kwargs: dict[str, Any],
    *,
    env_label: str,
) -> dict[str, Any] | None:
    err_text = str(err)
    unsupported_groups = {
        "wind/turbulence parameters": {"enable_wind", "wind_power", "turbulence_power"},
        "stochasticity parameters": {"initial_random_strength", "dispersion_strength"},
        "extended dynamics parameters": {
            "main_engine_power",
            "side_engine_power",
            "leg_spring_torque",
            "lander_mass_scale",
            "leg_mass_scale",
            "linear_damping",
            "angular_damping",
            "terrain_heights",
        },
    }
    drop_keys: set[str] = set()
    for group_name, group_keys in unsupported_groups.items():
        if any(key in err_text for key in group_keys):
            drop_keys.update(group_keys)
            print(
                f"{env_label} does not support {group_name} in this Gymnasium build; "
                "retrying without them.",
            )
    if not drop_keys:
        return None
    return {k: v for k, v in kwargs.items() if k not in drop_keys}


def make_lunarlander_env(
    env_id: str,
    *,
    gravity: float | None = None,
    task_id: float = 0.0,
    append_task_id: bool = True,
    enable_wind: bool | None = None,
    wind_power: float | None = None,
    turbulence_power: float | None = None,
    initial_random_strength: float | None = None,
    dispersion_strength: float | None = None,
    main_engine_power: float | None = None,
    side_engine_power: float | None = None,
    leg_spring_torque: float | None = None,
    lander_mass_scale: float | None = None,
    leg_mass_scale: float | None = None,
    linear_damping: float | None = None,
    angular_damping: float | None = None,
    terrain_heights: list[float] | None = None,
    action_repeat: int = 1,
    action_delay: int = 0,
    action_noise_prob: float = 0.0,
    action_noise_mode: str = "noop",
    mark_out_of_viewport_as_unsafe: bool = False,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a discrete-action LunarLander env with optional dynamics/task-id shifts."""
    ensure_lunarlander_v4_registered()

    make_kwargs: dict[str, Any] = {"continuous": False, "render_mode": render_mode}
    if gravity is not None:
        make_kwargs["gravity"] = float(gravity)
    if enable_wind is not None:
        make_kwargs["enable_wind"] = bool(enable_wind)
    if wind_power is not None:
        make_kwargs["wind_power"] = float(wind_power)
    if turbulence_power is not None:
        make_kwargs["turbulence_power"] = float(turbulence_power)
    if initial_random_strength is not None:
        make_kwargs["initial_random_strength"] = float(initial_random_strength)
    if dispersion_strength is not None:
        make_kwargs["dispersion_strength"] = float(dispersion_strength)
    if main_engine_power is not None:
        make_kwargs["main_engine_power"] = float(main_engine_power)
    if side_engine_power is not None:
        make_kwargs["side_engine_power"] = float(side_engine_power)
    if leg_spring_torque is not None:
        make_kwargs["leg_spring_torque"] = float(leg_spring_torque)
    if lander_mass_scale is not None:
        make_kwargs["lander_mass_scale"] = float(lander_mass_scale)
    if leg_mass_scale is not None:
        make_kwargs["leg_mass_scale"] = float(leg_mass_scale)
    if linear_damping is not None:
        make_kwargs["linear_damping"] = float(linear_damping)
    if angular_damping is not None:
        make_kwargs["angular_damping"] = float(angular_damping)
    if terrain_heights is not None:
        make_kwargs["terrain_heights"] = [float(v) for v in terrain_heights]

    try:
        env = gym.make(env_id, **make_kwargs)
    except TypeError as err:
        fallback_kwargs = _retry_without_unsupported_kwargs(err, make_kwargs, env_label=env_id)
        if fallback_kwargs is not None:
            env = gym.make(env_id, **fallback_kwargs)
        else:
            raise
    except gym.error.Error as err:
        if env_id == "LunarLander-v3":
            print("Could not create LunarLander-v3, retrying with LunarLander-v2 ...")
            try:
                env = gym.make("LunarLander-v2", **make_kwargs)
            except TypeError as type_err:
                fallback_kwargs = _retry_without_unsupported_kwargs(
                    type_err,
                    make_kwargs,
                    env_label="LunarLander-v2",
                )
                if fallback_kwargs is not None:
                    env = gym.make("LunarLander-v2", **fallback_kwargs)
                else:
                    raise
        else:
            raise RuntimeError(
                f"Failed to create env '{env_id}'. Ensure Box2D deps are installed "
                "(e.g., `pip install gymnasium[box2d]`)."
            ) from err

    # If manual terrain is requested, fail fast unless the instantiated env
    # actually exposes TunableLunarLander's manual-terrain support.
    if terrain_heights is not None:
        base_env = env.unwrapped
        manual_terrain = getattr(base_env, "_manual_terrain_heights", None)
        if manual_terrain is None:
            raise RuntimeError(
                "terrain_heights was provided, but the created environment does not "
                "support manual terrain injection. Use env_id='LunarLander-v4' "
                "(TunableLunarLander) and avoid fallbacks that drop custom kwargs.",
            )

    env = LunarLanderCrashSafetyWrapper(
        env,
        mark_out_of_viewport_as_unsafe=mark_out_of_viewport_as_unsafe,
    )
    # Treat action_repeat <= 1 as "no repeat wrapper" for backward compatibility.
    if action_repeat > 1:
        env = ActionRepeatWrapper(env, repeat=action_repeat)
    if action_delay != 0:
        env = ActionDelayWrapper(env, delay_steps=action_delay, noop_action=0)
    if action_noise_prob > 0.0:
        env = ActionNoiseWrapper(
            env,
            noise_prob=action_noise_prob,
            mode=action_noise_mode,
            noop_action=0,
        )
    if append_task_id:
        env = AppendTaskIDObservationWrapper(env, task_id=task_id)
    return env


# Backward-compatible alias for older imports.
_make_lunarlander_env = make_lunarlander_env
