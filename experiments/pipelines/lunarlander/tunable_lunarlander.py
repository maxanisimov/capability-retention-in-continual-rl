"""Tunable LunarLander environment with optional deterministic dynamics knobs."""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium.envs.box2d.lunar_lander as ll_mod
from gymnasium.envs.box2d.lunar_lander import LunarLander
from gymnasium.envs.registration import register, registry


LUNARLANDER_V4_ID = "LunarLander-v4"
LUNARLANDER_V4_ENTRY_POINT = "experiments.pipelines.lunarlander.tunable_lunarlander:TunableLunarLander"


class _DispersionRNGProxy:
    """Proxy RNG that rescales scalar uniform(-1, 1) samples used for engine dispersion."""

    def __init__(self, base_rng: np.random.Generator, owner_env: "TunableLunarLander"):
        self._base_rng = base_rng
        self._owner_env = owner_env

    def uniform(self, low=0.0, high=1.0, size=None):
        out = self._base_rng.uniform(low=low, high=high, size=size)
        if (
            size is None
            and np.isscalar(low)
            and np.isscalar(high)
            and np.isclose(low, -1.0)
            and np.isclose(high, 1.0)
        ):
            scale_factor = self._owner_env.dispersion_strength * ll_mod.SCALE
            return out * scale_factor
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_rng, name)


class TunableLunarLander(LunarLander):
    """LunarLander with explicit control over reset impulse and engine dispersion."""

    def __init__(
        self,
        *args,
        initial_random_strength: float = ll_mod.INITIAL_RANDOM,
        dispersion_strength: float = 1.0 / ll_mod.SCALE,
        main_engine_power: float = ll_mod.MAIN_ENGINE_POWER,
        side_engine_power: float = ll_mod.SIDE_ENGINE_POWER,
        leg_spring_torque: float = ll_mod.LEG_SPRING_TORQUE,
        lander_mass_scale: float = 1.0,
        leg_mass_scale: float = 1.0,
        linear_damping: float | None = None,
        angular_damping: float | None = None,
        deterministic: bool = False,
        **kwargs,
    ):
        if deterministic:
            initial_random_strength = 0.0
            dispersion_strength = 0.0
            kwargs["enable_wind"] = False
            kwargs["wind_power"] = 0.0
            kwargs["turbulence_power"] = 0.0

        super().__init__(*args, **kwargs)

        self.initial_random_strength = float(initial_random_strength)
        self.dispersion_strength = float(dispersion_strength)
        self.main_engine_power = float(main_engine_power)
        self.side_engine_power = float(side_engine_power)
        self.leg_spring_torque = float(leg_spring_torque)
        self.lander_mass_scale = float(lander_mass_scale)
        self.leg_mass_scale = float(leg_mass_scale)
        self.linear_damping = None if linear_damping is None else float(linear_damping)
        self.angular_damping = None if angular_damping is None else float(angular_damping)
        if self.initial_random_strength < 0.0:
            raise ValueError(
                f"initial_random_strength must be >= 0, got {self.initial_random_strength}.",
            )
        if self.dispersion_strength < 0.0:
            raise ValueError(
                f"dispersion_strength must be >= 0, got {self.dispersion_strength}.",
            )
        if self.main_engine_power < 0.0:
            raise ValueError(f"main_engine_power must be >= 0, got {self.main_engine_power}.")
        if self.side_engine_power < 0.0:
            raise ValueError(f"side_engine_power must be >= 0, got {self.side_engine_power}.")
        if self.leg_spring_torque < 0.0:
            raise ValueError(f"leg_spring_torque must be >= 0, got {self.leg_spring_torque}.")
        if self.lander_mass_scale <= 0.0:
            raise ValueError(f"lander_mass_scale must be > 0, got {self.lander_mass_scale}.")
        if self.leg_mass_scale <= 0.0:
            raise ValueError(f"leg_mass_scale must be > 0, got {self.leg_mass_scale}.")
        if self.linear_damping is not None and self.linear_damping < 0.0:
            raise ValueError(f"linear_damping must be >= 0, got {self.linear_damping}.")
        if self.angular_damping is not None and self.angular_damping < 0.0:
            raise ValueError(f"angular_damping must be >= 0, got {self.angular_damping}.")

        self._base_np_random: np.random.Generator | None = None
        self._install_rng_proxy()

    def _install_rng_proxy(self) -> None:
        current_rng = self.np_random
        if isinstance(current_rng, _DispersionRNGProxy):
            base_rng = current_rng._base_rng
        else:
            base_rng = current_rng
        self._base_np_random = base_rng
        self.np_random = _DispersionRNGProxy(base_rng, self)

    def set_stochasticity(
        self,
        *,
        initial_random_strength: float | None = None,
        dispersion_strength: float | None = None,
    ) -> None:
        if initial_random_strength is not None:
            value = float(initial_random_strength)
            if value < 0.0:
                raise ValueError(f"initial_random_strength must be >= 0, got {value}.")
            self.initial_random_strength = value
        if dispersion_strength is not None:
            value = float(dispersion_strength)
            if value < 0.0:
                raise ValueError(f"dispersion_strength must be >= 0, got {value}.")
            self.dispersion_strength = value

    def set_dynamics(
        self,
        *,
        main_engine_power: float | None = None,
        side_engine_power: float | None = None,
        leg_spring_torque: float | None = None,
        lander_mass_scale: float | None = None,
        leg_mass_scale: float | None = None,
        linear_damping: float | None = None,
        angular_damping: float | None = None,
    ) -> None:
        if main_engine_power is not None:
            value = float(main_engine_power)
            if value < 0.0:
                raise ValueError(f"main_engine_power must be >= 0, got {value}.")
            self.main_engine_power = value
        if side_engine_power is not None:
            value = float(side_engine_power)
            if value < 0.0:
                raise ValueError(f"side_engine_power must be >= 0, got {value}.")
            self.side_engine_power = value
        if leg_spring_torque is not None:
            value = float(leg_spring_torque)
            if value < 0.0:
                raise ValueError(f"leg_spring_torque must be >= 0, got {value}.")
            self.leg_spring_torque = value
        if lander_mass_scale is not None:
            value = float(lander_mass_scale)
            if value <= 0.0:
                raise ValueError(f"lander_mass_scale must be > 0, got {value}.")
            self.lander_mass_scale = value
        if leg_mass_scale is not None:
            value = float(leg_mass_scale)
            if value <= 0.0:
                raise ValueError(f"leg_mass_scale must be > 0, got {value}.")
            self.leg_mass_scale = value
        if linear_damping is not None:
            value = float(linear_damping)
            if value < 0.0:
                raise ValueError(f"linear_damping must be >= 0, got {value}.")
            self.linear_damping = value
        if angular_damping is not None:
            value = float(angular_damping)
            if value < 0.0:
                raise ValueError(f"angular_damping must be >= 0, got {value}.")
            self.angular_damping = value

    def make_deterministic(self) -> None:
        self.initial_random_strength = 0.0
        self.dispersion_strength = 0.0
        if hasattr(self, "enable_wind"):
            self.enable_wind = False
        if hasattr(self, "wind_power"):
            self.wind_power = 0.0
        if hasattr(self, "turbulence_power"):
            self.turbulence_power = 0.0

    def _apply_post_reset_body_dynamics(self) -> None:
        if self.lander is None:
            return

        # Adjust body/leg masses after upstream reset creates bodies and joints.
        if not np.isclose(self.lander_mass_scale, 1.0):
            self.lander.mass = float(self.lander.mass * self.lander_mass_scale)

        if hasattr(self, "legs"):
            for leg in self.legs:
                if not np.isclose(self.leg_mass_scale, 1.0):
                    leg.mass = float(leg.mass * self.leg_mass_scale)

        if self.linear_damping is not None:
            self.lander.linearDamping = float(self.linear_damping)
        if self.angular_damping is not None:
            self.lander.angularDamping = float(self.angular_damping)

    def step(self, action):  # type: ignore[override]
        old_main_engine_power = ll_mod.MAIN_ENGINE_POWER
        old_side_engine_power = ll_mod.SIDE_ENGINE_POWER
        ll_mod.MAIN_ENGINE_POWER = float(self.main_engine_power)
        ll_mod.SIDE_ENGINE_POWER = float(self.side_engine_power)
        try:
            return super().step(action)
        finally:
            ll_mod.MAIN_ENGINE_POWER = old_main_engine_power
            ll_mod.SIDE_ENGINE_POWER = old_side_engine_power

    def reset(self, *, seed=None, options=None):
        old_initial_random = ll_mod.INITIAL_RANDOM
        old_leg_spring_torque = ll_mod.LEG_SPRING_TORQUE
        ll_mod.INITIAL_RANDOM = float(self.initial_random_strength)
        ll_mod.LEG_SPRING_TORQUE = float(self.leg_spring_torque)
        try:
            obs, info = super().reset(seed=seed, options=options)
        finally:
            ll_mod.INITIAL_RANDOM = old_initial_random
            ll_mod.LEG_SPRING_TORQUE = old_leg_spring_torque
        self._apply_post_reset_body_dynamics()
        self._install_rng_proxy()
        return obs, info


def ensure_lunarlander_v4_registered() -> None:
    """Register LunarLander-v4 to point at TunableLunarLander."""
    existing = registry.get(LUNARLANDER_V4_ID)
    if existing is not None and str(getattr(existing, "entry_point", "")) == LUNARLANDER_V4_ENTRY_POINT:
        return

    if existing is not None:
        del registry[LUNARLANDER_V4_ID]

    register(
        id=LUNARLANDER_V4_ID,
        entry_point=LUNARLANDER_V4_ENTRY_POINT,
        max_episode_steps=1000,
        reward_threshold=200.0,
    )


ensure_lunarlander_v4_registered()
