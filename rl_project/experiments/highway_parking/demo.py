#%%
import sys
import numpy as np
import gymnasium
import highway_env
import matplotlib.pyplot as plt

sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning/rl_project')
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning')
from utils.ppo_utils import PPOConfig, ppo_train, evaluate
from utils.gymnasium_utils import render_gymnasium_agent

# Import to register the custom environment
from utils.custom_envs import CustomParkingEnv

class ParkingObservationWrapper(gymnasium.ObservationWrapper):
    """Convert Dict observation (OrderedDict with 'observation', 'achieved_goal',
    'desired_goal') to a flat array for MLP input.
    
    With use_goal=True: concatenates observation + desired_goal → 12 dims
    With use_goal=False: uses only observation → 6 dims
    """
    def __init__(self, env, use_goal=True):
        super().__init__(env)
        self.use_goal = use_goal
        obs_dim = 12 if use_goal else 6
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def observation(self, obs):
        if self.use_goal:
            return np.concatenate([
                obs['observation'].flatten(),
                obs['desired_goal'].flatten()
            ]).astype(np.float32)
        else:
            return obs['observation'].flatten().astype(np.float32)


class SafetyWrapperHighway(gymnasium.Wrapper):
    """Adds a 'safe' flag to info dict. safe=True when not crashed."""
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['safe'] = self.check_safety(obs, info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['safe'] = self.check_safety(obs, info)
        return obs, reward, terminated, truncated, info

    def check_safety(self, obs, info):
        return not info.get('crashed', False)

"""
Task 1: Parking with a specific goal and parrked vehicles.
Task 2: Different layout.

Idea: let's make Task 1 more difficult for safety (e.g. the goal is between two parked cars)
"""

env1_config = {
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False,
    },
    # "action": {"type": "ContinuousAction"},
    "action": {"type": "DiscreteMetaAction"}, # NOTE: Discrete action space for testing
    "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
    "success_goal_reward": 0.12,
    "collision_reward": -5,
    "steering_range": np.deg2rad(45),
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 100,
    "screen_width": 600,
    "screen_height": 300,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "controlled_vehicles": 1,
    "goal_spots": [('b', 'c', 1)],  # Optional: specify goal spot for ego vehicle
    "vehicles_count": 4, # NOTE
    "parked_vehicles_spots": [ # NOTE ('a', 'b', i) is the lower lane; ('b', 'c', i) is the upper lane;
        ('a', 'b', 0), ('a', 'b', 1),
        ('b', 'c', 0), ('b', 'c', 2),
    ], # NOTE: Removed - causes TypeError
    "add_walls": True,
}

env1 = gymnasium.make('custom-parking-v0', render_mode=None, config=env1_config)
env1 = ParkingObservationWrapper(env1, use_goal=True)
env1 = SafetyWrapperHighway(env1)

ppo_config = PPOConfig(
    seed=42,
    # total_timesteps=100_000,
)
actor1, critic1, training_data1= ppo_train(
    env=env1, cfg=ppo_config, 
    actor_warm_start=None,
    critic_warm_start=None,
    actor_param_bounds_l=None,
    actor_param_bounds_u=None,
    return_training_data=True
)

#%%
avg_total_reward, std_total_reward, failure_rate = evaluate(
    env=env1, actor=actor1, episodes=100
)
print(f"Evaluation results over 100 episodes:")
print(f"Average Total Reward: {avg_total_reward}")
print(f"Standard Deviation of Total Reward: {std_total_reward}")
print(f"Failure Rate: {failure_rate}")

# %%
# Render the trained agent in the environment
env1_render = gymnasium.make('custom-parking-v0', render_mode='human', config=env1_config)
env1_render = ParkingObservationWrapper(env1_render, use_goal=True)
env1_render = SafetyWrapperHighway(env1_render)
render_gymnasium_agent(
    actor1, 
    env=env1_render,
    num_episodes=4,
    seed=42, 
)

# %%
