"""Tunable LunarLander environment with optional deterministic dynamics knobs."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
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
        terrain_heights: list[float] | np.ndarray | None = None,
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

        self._manual_terrain_heights: np.ndarray | None = None
        self.set_manual_terrain(terrain_heights)

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

    def set_manual_terrain(self, terrain_heights: list[float] | np.ndarray | None) -> None:
        """Set custom terrain heights used at reset (before helipad flattening)."""
        if terrain_heights is None:
            self._manual_terrain_heights = None
            return

        arr = np.asarray(terrain_heights, dtype=np.float64).reshape(-1)
        # Gymnasium LunarLander reset uses CHUNKS=11 -> CHUNKS+1 == 12 heights.
        expected_len = 12
        if arr.shape[0] != expected_len:
            raise ValueError(
                f"terrain_heights must contain exactly {expected_len} values, got {arr.shape[0]}.",
            )
        if np.any(~np.isfinite(arr)):
            raise ValueError("terrain_heights contains non-finite values.")
        self._manual_terrain_heights = arr.copy()

    def _terrain_profile_for_reset(self, *, chunks: int, H: float) -> np.ndarray:
        if self._manual_terrain_heights is None:
            height = self.np_random.uniform(0, H / 2, size=(chunks + 1,))
        else:
            height = self._manual_terrain_heights.astype(np.float64, copy=True)

        helipad_y = H / 4
        center = chunks // 2
        height[center - 2] = helipad_y
        height[center - 1] = helipad_y
        height[center + 0] = helipad_y
        height[center + 1] = helipad_y
        height[center + 2] = helipad_y
        return height

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
        # Seed RNG like upstream reset() does.
        gym.Env.reset(self, seed=seed)

        old_initial_random = ll_mod.INITIAL_RANDOM
        old_leg_spring_torque = ll_mod.LEG_SPRING_TORQUE
        ll_mod.INITIAL_RANDOM = float(self.initial_random_strength)
        ll_mod.LEG_SPRING_TORQUE = float(self.leg_spring_torque)
        try:
            self._destroy()

            # Mirrors upstream workaround for world cleanup.
            self.world = ll_mod.Box2D.b2World(gravity=(0, self.gravity))
            self.world.contactListener_keepref = ll_mod.ContactDetector(self)
            self.world.contactListener = self.world.contactListener_keepref
            self.game_over = False
            self.prev_shaping = None

            W = ll_mod.VIEWPORT_W / ll_mod.SCALE
            H = ll_mod.VIEWPORT_H / ll_mod.SCALE

            # Create terrain from manual heights (if provided) or RNG, while
            # enforcing a flat central helipad.
            CHUNKS = 11
            height = self._terrain_profile_for_reset(chunks=CHUNKS, H=H)
            chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
            self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
            self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
            self.helipad_y = H / 4
            smooth_y = [
                0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
                for i in range(CHUNKS)
            ]

            self.moon = self.world.CreateStaticBody(
                shapes=ll_mod.edgeShape(vertices=[(0, 0), (W, 0)]),
            )
            self.sky_polys = []
            for i in range(CHUNKS - 1):
                p1 = (chunk_x[i], smooth_y[i])
                p2 = (chunk_x[i + 1], smooth_y[i + 1])
                self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
                self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

            self.moon.color1 = (0.0, 0.0, 0.0)
            self.moon.color2 = (0.0, 0.0, 0.0)

            # Create Lander body
            initial_y = ll_mod.VIEWPORT_H / ll_mod.SCALE
            initial_x = ll_mod.VIEWPORT_W / ll_mod.SCALE / 2
            self.lander = self.world.CreateDynamicBody(
                position=(initial_x, initial_y),
                angle=0.0,
                fixtures=ll_mod.fixtureDef(
                    shape=ll_mod.polygonShape(
                        vertices=[(x / ll_mod.SCALE, y / ll_mod.SCALE) for x, y in ll_mod.LANDER_POLY],
                    ),
                    density=5.0,
                    friction=0.1,
                    categoryBits=0x0010,
                    maskBits=0x001,
                    restitution=0.0,
                ),
            )
            self.lander.color1 = (128, 102, 230)
            self.lander.color2 = (77, 77, 128)

            # Apply (possibly disabled) initial random impulse.
            self.lander.ApplyForceToCenter(
                (
                    self.np_random.uniform(-ll_mod.INITIAL_RANDOM, ll_mod.INITIAL_RANDOM),
                    self.np_random.uniform(-ll_mod.INITIAL_RANDOM, ll_mod.INITIAL_RANDOM),
                ),
                True,
            )

            if self.enable_wind:
                self.wind_idx = self.np_random.integers(-9999, 9999)
                self.torque_idx = self.np_random.integers(-9999, 9999)

            # Create legs.
            self.legs = []
            for i in [-1, +1]:
                leg = self.world.CreateDynamicBody(
                    position=(initial_x - i * ll_mod.LEG_AWAY / ll_mod.SCALE, initial_y),
                    angle=(i * 0.05),
                    fixtures=ll_mod.fixtureDef(
                        shape=ll_mod.polygonShape(box=(ll_mod.LEG_W / ll_mod.SCALE, ll_mod.LEG_H / ll_mod.SCALE)),
                        density=1.0,
                        restitution=0.0,
                        categoryBits=0x0020,
                        maskBits=0x001,
                    ),
                )
                leg.ground_contact = False
                leg.color1 = (128, 102, 230)
                leg.color2 = (77, 77, 128)
                rjd = ll_mod.revoluteJointDef(
                    bodyA=self.lander,
                    bodyB=leg,
                    localAnchorA=(0, 0),
                    localAnchorB=(i * ll_mod.LEG_AWAY / ll_mod.SCALE, ll_mod.LEG_DOWN / ll_mod.SCALE),
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=ll_mod.LEG_SPRING_TORQUE,
                    motorSpeed=+0.3 * i,
                )
                if i == -1:
                    rjd.lowerAngle = +0.9 - 0.5
                    rjd.upperAngle = +0.9
                else:
                    rjd.lowerAngle = -0.9
                    rjd.upperAngle = -0.9 + 0.5
                leg.joint = self.world.CreateJoint(rjd)
                self.legs.append(leg)

            self.drawlist = [self.lander] + self.legs

            # Apply custom post-reset dynamics (masses + damping).
            self._apply_post_reset_body_dynamics()

            if self.render_mode == "human":
                self.render()

            # Upstream reset returns the observation after one no-op step.
            obs = self.step(np.array([0, 0]) if self.continuous else 0)[0]
            info = {}
        finally:
            ll_mod.INITIAL_RANDOM = old_initial_random
            ll_mod.LEG_SPRING_TORQUE = old_leg_spring_torque
            # Reinstall proxy after possible RNG reseeding in reset flow.
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
