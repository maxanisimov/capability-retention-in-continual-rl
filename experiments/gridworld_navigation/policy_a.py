### Learn a Policy A which is trained only on Task 1. 
### This policy follows the optimal trajectory fo Task 1 and thefore is not safe for Task 2.

#%%
### Imports
import sys
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning')
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning/experiments')
import gymnasium as gym
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from _navigation_env import PointNavSafetyEnv, PointNavConfig
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.utils.data import TensorDataset, DataLoader
import copy
from src.trainer import IntervalTrainer
import torch
from _safety_critic_utils import SafetyCritic, collect_traces_dataset
from _ppo_utils import PPOConfig, ppo_train, evaluate
from collections import OrderedDict

# ----- Define Task 1 (big hazard on the shortest path) -----
task1_cfg = PointNavConfig(
    start=(-0.9, -0.9),
    goal=(0.9, 0.9),
    hazard_circles=[(0.0, 0.0, 0.35)],  # big hazard on the shortest path
    hazard_rects=[],                       # not used here
    terminate_on_hazard=True
)

# ----- Define Task 2 (smaller hazard on the shortest path + flooding on the previous best paths) -----
task2_cfg = PointNavConfig(
    start=(-0.9, -0.9),
    goal=(0.9, 0.9),
    hazard_circles=[
        # Flooding #1 hazards on previous best paths
        # (-0.5, -0.9, 0.1),
        (-0.4, -0.8, 0.1),
        (-0.3, -0.7, 0.1),
        (-0.2, -0.6, 0.1),
        (-0.1, -0.5, 0.1),
        (0.0, -0.4, 0.1),
        (0.1, -0.3, 0.1),
        (0.2, -0.2, 0.1),
        (0.3, -0.1, 0.1),
        (0.4, 0.0, 0.1),
        (0.5, 0.1, 0.1),
        (0.6, 0.2, 0.1),
        (0.7, 0.3, 0.1),
        (0.8, 0.4, 0.1),
        # (0.9, 0.5, 0.1),
        # Flooding #2 hazards on previous best paths
        # (-0.9, -0.5, 0.1),
        (-0.8, -0.4, 0.1),
        (-0.7, -0.3, 0.1),
        (-0.6, -0.2, 0.1),
        (-0.5, -0.1, 0.1),
        (-0.4, 0.0, 0.1),
        (-0.3, 0.1, 0.1),
        (-0.2, 0.2, 0.1),
        (-0.1, 0.3, 0.1),
        (0.0, 0.4, 0.1),
        (0.1, 0.5, 0.1),
        (0.2, 0.6, 0.1),
        (0.3, 0.7, 0.1),
        (0.4, 0.8, 0.1),
        # (0.5, 0.9, 0.1),
    ],
    terminate_on_hazard=True, # terminate on hazard to encourage safe paths
    progress_coef=100.0  # bigger coeff to encourage finding the shortest path
)

def make_env(cfg):
    return Monitor(PointNavSafetyEnv(cfg))

# ----- Evaluate safety transfer: run Task-2 policy in Task-1 env -----


final_metrics = {
    'Policy A': { # original policy trained on Task 1
        'Task 1': {
            'avg_total_reward': None,
            'failure_rate': None,
        },
        'Task 2': {
            'avg_total_reward': None,
            'failure_rate': None,
        },
    },
    'Policy B': {
                'Task 1': {
            'avg_total_reward': None,
            'failure_rate': None,
        },
        'Task 2': {
            'avg_total_reward': None,
            'failure_rate': None,
        },
    },
    'Policy C': {
                'Task 1': {
            'avg_total_reward': None,
            'failure_rate': None,
        },
        'Task 2': {
            'avg_total_reward': None,
            'failure_rate': None,
        },
    }
}


### Create the Grid World Environments for Task 1 and Task 2
env1 = make_env(task1_cfg)
env2 = make_env(task2_cfg)

### Train Policy A on Task 1 only
task1_ppo_cfg = PPOConfig( 
    total_timesteps=200_000, # 200_000 works with the gridworld; default 100_000,
    eval_episodes=100,    # Reduced from 1_000 to match SB3 evaluation
    rollout_steps=2048,   # SB3 default
    update_epochs=10,     # SB3 default
    minibatch_size=64,    # SB3 default
    gamma=0.99,           # SB3 default
    gae_lambda=0.95,      # SB3 default
    clip_coef=0.2,        # SB3 default
    ent_coef=0.01,        # SB3 default
    vf_coef=0.5,          # SB3 default
    lr=3e-4,              # SB3 default
    max_grad_norm=0.5,    # SB3 default
    seed=42,              # Fixed seed
    device='cpu'          # Ensure same device
)
task1_actor, task1_critic = ppo_train(env=env1, cfg=task1_ppo_cfg)

### Rollout and evaluate Policy A on Task 1 and Task 2
avg_reward_task1_a, std_task1, failure_rate_task1_a = evaluate(env1, task1_actor, episodes=1)
print(f"Policy A - Task 1: Avg Reward: {avg_reward_task1_a}, Failure Rate: {failure_rate_task1_a}")
avg_reward_task2_a, std_task2, failure_rate_task2_a = evaluate(env2, task1_actor, episodes=1)
print(f"Policy A - Task 2: Avg Reward: {avg_reward_task2_a}, Failure Rate: {failure_rate_task2_a}")

#%%
### Show the trajectory of Policy A on Task 2
# Plot trajectory learned by task1_actor
print("Plotting trajectory for Policy A (Task 1)...")
env1_traj = PointNavSafetyEnv(task1_cfg)
obs, _ = env1_traj.reset()
trajectory = [obs.copy()]
done = False

while not done:
    action = task1_actor(torch.tensor(obs)).argmax().item()
    obs, reward, terminated, truncated, info = env1_traj.step(action)
    trajectory.append(obs.copy())
    done = terminated or truncated

trajectory = np.array(trajectory)

### Render environment and plot trajectory
fig, ax = plt.subplots(figsize=(8, 8))
img = env1_traj.render()
ax.imshow(img)

# Convert trajectory from world coordinates [-0.9, 0.9] to image pixel coordinates
# The environment typically scales to image space automatically in render()
# Get the image dimensions
img_height, img_width = img.shape[:2]

# World coordinates are in range [-0.9, 0.9], we need to map to [0, img_width/height]
# Assuming the environment center is at (0.9, 0.9) in world coords maps to center of image
trajectory_pixels = trajectory.copy()
trajectory_pixels[:, 0] = (trajectory[:, 0] + 1.0) / 2.0 * img_width   # X: [-0.9, 0.9] -> [0, img_width]
trajectory_pixels[:, 1] = (trajectory[:, 1] + 1.0) / 2.0 * img_height  # Y: [-0.9, 0.9] -> [0, img_height]

ax.plot(trajectory_pixels[:, 0], trajectory_pixels[:, 1], 'r-', linewidth=2, label='Trajectory')
ax.scatter(trajectory_pixels[0, 0], trajectory_pixels[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
ax.scatter(trajectory_pixels[-1, 0], trajectory_pixels[-1, 1], c='blue', s=100, marker='*', label='End', zorder=5)
ax.set_title('Task 1 - Policy A Trajectory')
ax.legend()
ax.set_xlim(-10, img_width+10)
ax.set_ylim(img_height+10, -10)  # Flip Y-axis to match image coordinates
ax.axis('off')
plt.tight_layout()
# plt.savefig('figures/policy_a_trajectory.png', dpi=300, bbox_inches='tight')
plt.show()
env1_traj.close()
# %%
