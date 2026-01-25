from gymnasium.envs.registration import register
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import math

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