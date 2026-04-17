"""Highway Parking setup with discrete actions and configurable parked spots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
from gymnasium.envs.registration import register, registry
import numpy as np
import yaml

import highway_env  # noqa: F401  # Registers highway-env environments.
from highway_env.envs.parking_env import ParkingEnv
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle


HIGHWAY_PARKING_ENV_ID = "HighwayParkingCustom-v0"
HIGHWAY_PARKING_ENTRY_POINT = "rl_project.experiments.highway.parking_setup:ConfigurableParkingEnv"


def _convert_lane_spots(
    spots: list[list[Any]] | list[tuple[Any, ...]] | None,
    *,
    field_name: str,
) -> list[tuple[str, str, int]] | None:
    """Convert YAML lane spots to tuple lane indices required by highway-env."""
    if spots is None:
        return None
    if not isinstance(spots, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of lane indices or null.")

    converted: list[tuple[str, str, int]] = []
    for idx, spot in enumerate(spots):
        if not isinstance(spot, (list, tuple)) or len(spot) != 3:
            raise ValueError(
                f"{field_name}[{idx}] must have exactly 3 entries like ['a', 'b', 0].",
            )
        converted.append((str(spot[0]), str(spot[1]), int(spot[2])))
    return converted


def normalize_parking_config(
    config: Mapping[str, Any],
    *,
    cfg_name: str = "config",
) -> dict[str, Any]:
    """Validate/normalize a parking config loaded from task settings."""
    cfg = dict(config)

    if "steering_range_deg" in cfg and "steering_range" not in cfg:
        cfg["steering_range"] = float(np.deg2rad(float(cfg.pop("steering_range_deg"))))

    controlled_vehicles = int(cfg.get("controlled_vehicles", 1))
    if controlled_vehicles < 1:
        raise ValueError(f"{cfg_name}: controlled_vehicles must be >= 1, got {controlled_vehicles}.")
    cfg["controlled_vehicles"] = controlled_vehicles

    goal_spots = _convert_lane_spots(
        cfg.get("goal_spots"),
        field_name=f"{cfg_name}.goal_spots",
    )
    if goal_spots is not None:
        if len(goal_spots) < controlled_vehicles:
            raise ValueError(
                f"{cfg_name}: goal_spots must contain at least {controlled_vehicles} entries "
                f"(got {len(goal_spots)}).",
            )
        cfg["goal_spots"] = goal_spots

    parked_vehicles_spots = _convert_lane_spots(
        cfg.get("parked_vehicles_spots"),
        field_name=f"{cfg_name}.parked_vehicles_spots",
    )
    vehicles_count_raw = cfg.get("vehicles_count", None)

    if parked_vehicles_spots is not None:
        if vehicles_count_raw is None:
            vehicles_count = len(parked_vehicles_spots)
        else:
            vehicles_count = int(vehicles_count_raw)
        if vehicles_count != len(parked_vehicles_spots):
            raise ValueError(
                f"{cfg_name}: vehicles_count ({vehicles_count}) must match "
                f"len(parked_vehicles_spots) ({len(parked_vehicles_spots)}).",
            )
        cfg["vehicles_count"] = vehicles_count
        cfg["parked_vehicles_spots"] = parked_vehicles_spots
    else:
        vehicles_count = int(vehicles_count_raw or 0)
        if vehicles_count < 0:
            raise ValueError(f"{cfg_name}: vehicles_count must be >= 0, got {vehicles_count}.")
        cfg["vehicles_count"] = vehicles_count

    return cfg


class ConfigurableParkingEnv(ParkingEnv):
    """ParkingEnv variant supporting exact parked-vehicle lane placement."""

    def _create_vehicles(self) -> None:
        empty_spots = list(self.road.network.lanes_dict().keys())

        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            x0 = float(i - self.config["controlled_vehicles"] // 2) * 10.0
            vehicle = self.action_type.vehicle_class(
                self.road,
                [x0, 0.0],
                2.0 * np.pi * self.np_random.uniform(),
                0.0,
            )
            vehicle.color = VehicleGraphics.EGO_COLOR
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
            if vehicle.lane_index in empty_spots:
                empty_spots.remove(vehicle.lane_index)

        goal_spots = self.config.get("goal_spots")
        for i, vehicle in enumerate(self.controlled_vehicles):
            if goal_spots is not None:
                lane_index = tuple(goal_spots[i])
                if lane_index not in empty_spots:
                    raise ValueError(
                        f"Goal spot {lane_index} for controlled vehicle {i} is invalid or occupied. "
                        f"Available spots: {empty_spots}",
                    )
            else:
                lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]

            lane = self.road.network.get_lane(lane_index)
            vehicle.goal = Landmark(
                self.road,
                lane.position(lane.length / 2, 0),
                heading=lane.heading,
            )
            self.road.objects.append(vehicle.goal)
            empty_spots.remove(lane_index)

        parked_vehicles_spots = self.config.get("parked_vehicles_spots")
        vehicles_count = int(self.config["vehicles_count"])
        if parked_vehicles_spots is not None and len(parked_vehicles_spots) < vehicles_count:
            raise ValueError(
                "parked_vehicles_spots must contain at least vehicles_count entries, "
                f"got {len(parked_vehicles_spots)} < {vehicles_count}.",
            )

        for i in range(vehicles_count):
            if not empty_spots:
                break
            if parked_vehicles_spots is not None:
                lane_index = tuple(parked_vehicles_spots[i])
                if lane_index not in empty_spots:
                    raise ValueError(
                        f"Parked vehicle spot {lane_index} for vehicle {i} is invalid or occupied. "
                        f"Available spots: {empty_spots}",
                    )
            else:
                lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]

            parked_vehicle = Vehicle.make_on_lane(
                self.road,
                lane_index,
                longitudinal=4.0,
                speed=0.0,
            )
            self.road.vehicles.append(parked_vehicle)
            empty_spots.remove(lane_index)

        if self.config["add_walls"]:
            width, height = 70, 42
            for y in [-height / 2, height / 2]:
                obstacle = Obstacle(self.road, [0, y])
                obstacle.LENGTH, obstacle.WIDTH = (width, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)
            for x in [-width / 2, width / 2]:
                obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
                obstacle.LENGTH, obstacle.WIDTH = (height, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)


def ensure_highway_parking_registered() -> None:
    """Register a dedicated configurable ParkingEnv ID for this project."""
    existing_spec = registry.get(HIGHWAY_PARKING_ENV_ID)
    if existing_spec is not None:
        existing_entry = str(existing_spec.entry_point)
        if existing_entry == HIGHWAY_PARKING_ENTRY_POINT:
            return
        del registry[HIGHWAY_PARKING_ENV_ID]

    register(
        id=HIGHWAY_PARKING_ENV_ID,
        entry_point=HIGHWAY_PARKING_ENTRY_POINT,
        max_episode_steps=100,
    )


class ParkingObservationWrapper(gym.ObservationWrapper):
    """Flatten ParkingEnv Dict observations into a single vector."""

    def __init__(
        self,
        env: gym.Env,
        *,
        use_goal: bool = True,
        task_id: float | None = None,
    ):
        super().__init__(env)

        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError("ParkingObservationWrapper requires a Dict observation space.")
        if "observation" not in env.observation_space:
            raise ValueError("Parking observation dict must include 'observation'.")
        if use_goal and "desired_goal" not in env.observation_space:
            raise ValueError("Parking observation dict must include 'desired_goal' when use_goal=True.")

        self.use_goal = bool(use_goal)
        self.task_id = None if task_id is None else float(task_id)

        obs_dim = int(np.prod(env.observation_space["observation"].shape))
        goal_dim = int(np.prod(env.observation_space["desired_goal"].shape)) if self.use_goal else 0
        task_dim = 1 if self.task_id is not None else 0
        out_dim = obs_dim + goal_dim + task_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(out_dim,),
            dtype=np.float32,
        )

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        pieces: list[np.ndarray] = [np.asarray(observation["observation"], dtype=np.float32).reshape(-1)]
        if self.use_goal:
            pieces.append(np.asarray(observation["desired_goal"], dtype=np.float32).reshape(-1))
        if self.task_id is not None:
            pieces.append(np.asarray([self.task_id], dtype=np.float32))
        return np.concatenate(pieces, axis=0)


class DiscreteParkingActionWrapper(gym.ActionWrapper):
    """Discretize continuous [acceleration, steering] actions into a grid."""

    def __init__(self, env: gym.Env, *, n_bins_accel: int = 5, n_bins_steer: int = 5):
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("DiscreteParkingActionWrapper requires a Box action space.")
        if tuple(env.action_space.shape) != (2,):
            raise ValueError(
                "DiscreteParkingActionWrapper expects a 2D action space "
                f"(acceleration, steering), got {env.action_space.shape}.",
            )

        self.n_bins_accel = int(n_bins_accel)
        self.n_bins_steer = int(n_bins_steer)
        if self.n_bins_accel < 2 or self.n_bins_steer < 2:
            raise ValueError("n_bins_accel and n_bins_steer must both be >= 2.")

        low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        self._accel_grid = np.linspace(low[0], high[0], self.n_bins_accel, dtype=np.float32)
        self._steer_grid = np.linspace(low[1], high[1], self.n_bins_steer, dtype=np.float32)

        self.action_space = gym.spaces.Discrete(self.n_bins_accel * self.n_bins_steer)

    def action(self, action: int) -> np.ndarray:
        action_index = int(action)
        accel_idx = action_index // self.n_bins_steer
        steer_idx = action_index % self.n_bins_steer
        if accel_idx < 0 or accel_idx >= self.n_bins_accel:
            raise ValueError(f"Discrete action {action_index} is out of range.")
        return np.asarray(
            [self._accel_grid[accel_idx], self._steer_grid[steer_idx]],
            dtype=np.float32,
        )


class ParkingSafetyWrapper(gym.Wrapper):
    """Expose a consistent boolean safety flag as info['safe']."""

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        info = dict(info)
        info["safe"] = not bool(info.get("crashed", False))
        return observation, info

    def step(self, action):  # type: ignore[override]
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["safe"] = not bool(info.get("crashed", False))
        return observation, reward, terminated, truncated, info


@dataclass(frozen=True)
class HighwayTaskSetup:
    """Resolved setup needed to build a discrete Highway Parking environment."""

    env_base_config: dict[str, Any]
    task_config: dict[str, Any]
    task_id: float
    append_task_id: bool
    use_goal: bool
    n_bins_accel: int
    n_bins_steer: int


def load_highway_task_setup(
    task_settings_file: str | Path,
    task_setting: str,
    task_role: str = "source",
) -> HighwayTaskSetup:
    """Load a named source/downstream task from the Highway task-settings YAML."""
    if task_role not in {"source", "downstream"}:
        raise ValueError(f"task_role must be 'source' or 'downstream', got '{task_role}'.")

    with Path(task_settings_file).open("r", encoding="utf-8") as handle:
        all_settings = yaml.safe_load(handle)
    if not isinstance(all_settings, dict):
        raise ValueError(f"{task_settings_file} did not parse to a YAML mapping.")
    if task_setting not in all_settings:
        raise ValueError(
            f"Task setting '{task_setting}' not found in {task_settings_file}. "
            f"Available keys: {sorted(all_settings.keys())}",
        )

    setting_cfg = all_settings[task_setting]
    if not isinstance(setting_cfg, dict):
        raise ValueError(f"Task setting '{task_setting}' must be a mapping.")

    env_base_cfg_raw = setting_cfg.get("env_base", {})
    task_cfg_raw = setting_cfg.get(task_role, {})
    if not isinstance(env_base_cfg_raw, dict):
        raise ValueError(f"task_settings[{task_setting}].env_base must be a mapping.")
    if not isinstance(task_cfg_raw, dict):
        raise ValueError(f"task_settings[{task_setting}].{task_role} must be a mapping.")

    env_base_cfg = dict(env_base_cfg_raw)
    task_cfg = dict(task_cfg_raw)
    task_id_default = 0.0 if task_role == "source" else 1.0

    task_id = float(task_cfg.pop("task_id", task_id_default))
    append_task_id = bool(task_cfg.pop("append_task_id", setting_cfg.get("append_task_id", True)))
    use_goal = bool(task_cfg.pop("use_goal", setting_cfg.get("use_goal", True)))
    n_bins_accel = int(task_cfg.pop("n_bins_accel", setting_cfg.get("n_bins_accel", 5)))
    n_bins_steer = int(task_cfg.pop("n_bins_steer", setting_cfg.get("n_bins_steer", 5)))

    return HighwayTaskSetup(
        env_base_config=env_base_cfg,
        task_config=task_cfg,
        task_id=task_id,
        append_task_id=append_task_id,
        use_goal=use_goal,
        n_bins_accel=n_bins_accel,
        n_bins_steer=n_bins_steer,
    )


def make_highway_parking_env(
    env_base_config: Mapping[str, Any],
    task_config: Mapping[str, Any],
    *,
    task_id: float,
    append_task_id: bool = True,
    use_goal: bool = True,
    n_bins_accel: int = 5,
    n_bins_steer: int = 5,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a discrete-action highway parking environment for one task."""
    ensure_highway_parking_registered()
    merged_config = dict(env_base_config)
    merged_config.update(dict(task_config))
    normalized_config = normalize_parking_config(merged_config, cfg_name="highway_parking")

    env = gym.make(HIGHWAY_PARKING_ENV_ID, render_mode=render_mode, config=normalized_config)
    task_indicator = task_id if append_task_id else None
    env = ParkingObservationWrapper(env, use_goal=use_goal, task_id=task_indicator)
    env = DiscreteParkingActionWrapper(
        env,
        n_bins_accel=n_bins_accel,
        n_bins_steer=n_bins_steer,
    )
    env = ParkingSafetyWrapper(env)
    return env


def make_highway_parking_env_from_task_settings(
    task_settings_file: str | Path,
    *,
    task_setting: str = "default",
    task_role: str = "source",
    render_mode: str | None = None,
) -> gym.Env:
    """Load task settings and build the corresponding discrete parking environment."""
    setup = load_highway_task_setup(task_settings_file, task_setting, task_role)
    return make_highway_parking_env(
        setup.env_base_config,
        setup.task_config,
        task_id=setup.task_id,
        append_task_id=setup.append_task_id,
        use_goal=setup.use_goal,
        n_bins_accel=setup.n_bins_accel,
        n_bins_steer=setup.n_bins_steer,
        render_mode=render_mode,
    )
