from gymnasium.envs.registration import register
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import math
import numpy as np
from highway_env.envs.parking_env import ParkingEnv
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle
from highway_env.vehicle.graphics import VehicleGraphics

### CartPole
class CustomCartPoleEnv(CartPoleEnv):
    def __init__(
        self, gravity=9.8, masscart=1.0, masspole=0.1, length=0.5, force_mag=10.0,
        theta_threshold_radians=12 * 2 * math.pi / 360, x_threshold=2.4,
        tau=0.02, kinematics_integrator="euler", reward_shaping="standard",
        **kwargs
    ):
        # Filter out custom parameters before calling super().__init__
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['reward_shaping']}
        super().__init__(**filtered_kwargs)
        self.reward_shaping = reward_shaping
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.length = length               # note: half-length
        self.force_mag = force_mag
        self.polemass_length = self.masspole * self.length
        self.tau = tau  # seconds between state updates
        self.kinematics_integrator = kinematics_integrator
        # Angle at which to fail the episode
        self.theta_threshold_radians = theta_threshold_radians
        self.x_threshold = x_threshold

    def step(self, action):
        """Override step to implement custom reward shaping."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Save failure info
        info = {'failure': terminated}
        
        # Apply reward shaping based on configuration
        if self.reward_shaping == "aggressive_movement":
            # Moderately reward movement - creates some conflict with stability
            x, x_dot, theta, theta_dot = obs
            movement_bonus = 2 * abs(x_dot)      # Moderate reward for speed
            angle_bonus = 2 * abs(theta)         # Moderate reward for large angles
            reward = reward + movement_bonus + angle_bonus
        elif self.reward_shaping == "extreme_stability":
            # Heavily penalize deviation - creates strong pressure for different policy
            x, x_dot, theta, theta_dot = obs
            stability_penalty = -3.0 * (abs(x) + abs(theta) + 0.5 * abs(x_dot) + 0.5 * abs(theta_dot))
            reward = reward + stability_penalty
        # "standard" uses default CartPole reward
        
        return obs, reward, terminated, truncated, info

# register at import time
try:
    register(id="CustomCartPole-v1", entry_point=CustomCartPoleEnv, max_episode_steps=500)
except Exception:
    # If already registered (e.g., due to hot-reload), ignore
    pass


### Highway Parking
class CustomParkingEnv(ParkingEnv):
    """
    Custom Parking environment with support for exact vehicle placement.

    This environment extends the highway-env ParkingEnv to allow precise control
    over where obstacle vehicles are placed via the 'parked_vehicles_spots' config parameter.

    Config additions:
        parked_vehicles_spots (list of tuples, optional): List of lane indices (tuples)
            specifying exact spots for parked vehicles. Each tuple should be a lane_index
            like ('a', 'b', 0) for lower lane or ('b', 'c', 0) for upper lane.
            If None, vehicles are placed randomly as in the standard ParkingEnv.

    Example usage:
        config = {
            "vehicles_count": 3,
            "goal_spots": [('b', 'c', 1)],  # Optional: specify goal spot for ego vehicle
            "parked_vehicles_spots": [
                ('a', 'b', 0),  # Lower lane, spot 0
                ('a', 'b', 1),  # Lower lane, spot 1
                ('b', 'c', 0),  # Upper lane, spot 0
                ('b', 'c', 2),  # Upper lane, spot 2

            ],
            # ... other config parameters
        }
        env = gymnasium.make('custom-parking-v0', config=config)
    """

    def _create_vehicles(self) -> None:
        """Create vehicles with support for exact spot placement via config."""
        empty_spots = list(self.road.network.lanes_dict().keys())

        # Controlled vehicles (ego vehicle)
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            x0 = float(i - self.config["controlled_vehicles"] // 2) * 10.0
            vehicle = self.action_type.vehicle_class(
                self.road, [x0, 0.0], 2.0 * np.pi * self.np_random.uniform(), 0.0
            )
            vehicle.color = VehicleGraphics.EGO_COLOR
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
            empty_spots.remove(vehicle.lane_index)

        # Goal (target parking spot for ego vehicle)
        for i, vehicle in enumerate(self.controlled_vehicles):
            if self.config.get("goal_spots") is not None:
                lane_index = self.config["goal_spots"][i]
                if lane_index not in empty_spots:
                    raise ValueError(
                        f"Goal spot index {lane_index} is not empty or invalid. "
                        f"Cannot place goal for vehicle {i} there. Available spots: {empty_spots}"
                    )
            else:
                lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
            lane = self.road.network.get_lane(lane_index)
            vehicle.goal = Landmark(
                self.road, lane.position(lane.length / 2, 0), heading=lane.heading
            )
            self.road.objects.append(vehicle.goal)
            empty_spots.remove(lane_index)

        # Other vehicles (obstacles) - with custom spot placement support
        for i in range(self.config["vehicles_count"]):
            if not empty_spots:
                continue

            # Check if custom spot indices are provided
            if self.config.get("parked_vehicles_spots") is not None:
                lane_index = self.config["parked_vehicles_spots"][i]
                if lane_index not in empty_spots:
                    raise ValueError(
                        f"Spot index {lane_index} is not empty or invalid. "
                        f"Cannot place vehicle {i} there. Available spots: {empty_spots}"
                    )
            else:
                # Default behavior: random placement
                lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]

            v = Vehicle.make_on_lane(self.road, lane_index, longitudinal=4.0, speed=0.0)
            self.road.vehicles.append(v)
            empty_spots.remove(lane_index)

        # Walls
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


# Register at import time
try:
    register(
        id="custom-parking-v0",
        entry_point="rl_project.utils.custom_envs:CustomParkingEnv",
        max_episode_steps=100,
    )
except Exception:
    # If already registered (e.g., due to hot-reload), ignore
    pass