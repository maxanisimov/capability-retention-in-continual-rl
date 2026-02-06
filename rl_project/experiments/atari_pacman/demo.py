"""
Train a PPO agent for a Pacman environment using RL Zoo hyperparameters
"""
#%%
import os
import sys
import torch
import numpy as np
import pandas as pd
import yaml
import gymnasium
from gymnasium.wrappers import FlattenObservation, GrayScaleObservation, ResizeObservation
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning/rl_project')
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning')
from utils.ppo_utils import ppo_train, PPOConfig
from src.trainer import IntervalTrainer
from utils.gymnasium_utils import render_gymnasium_agent

#%%
# Create environment with RAM observations (simpler than RGB)
env1 = gymnasium.make("ALE/Pacman-v5", obs_type='ram')

#%%
# Apply RL Zoo hyperparameters for Atari Pacman
# Source: https://huggingface.co/alfredowh/ppo-Pacman-v5
ppo_cfg = PPOConfig(
    # Training
    total_timesteps=500_000,      # n_timesteps from config (reduced for testing)
    
    # PPO hyperparameters
    rollout_steps=128,                  # Rollout buffer size
    minibatch_size=256,               # Minibatch size
    update_epochs=4,                   # Number of epochs per update
    
    # Discount and GAE
    gamma=0.99,                   # Discount factor
    gae_lambda=0.95,              # GAE lambda
    
    # Learning rates (linear schedule)
    lr=2.5e-4,         # Starting learning rate (lin_2.5e-4)
    # Note: You may need to implement linear schedule: starts at 2.5e-4, decays to 0
    
    # Clipping
    clip_coef=0.1,               # PPO clipping parameter (lin_0.1)
    
    # Coefficients
    ent_coef=0.01,                # Entropy coefficient
    vf_coef=0.5,                  # Value function coefficient
    max_grad_norm=0.5,            # Gradient clipping
    
    # Note: Original uses n_envs=8 (parallel environments)
    # and frame_stack=4, but these require different setup with RAM
)

print("PPO Config:")
print(f"  Total timesteps: {ppo_cfg.total_timesteps:,}")
print(f"  Batch size: {ppo_cfg.minibatch_size}")
print(f"  Learning rate: {ppo_cfg.lr}")
print(f"  Entropy coef: {ppo_cfg.ent_coef}")

#%%
# Train the agent
print("\n=== Training PPO Agent on Pacman ===")
actor, critic, training_data_env1 = ppo_train(
    env=env1, 
    cfg=ppo_cfg, 
    return_training_data=True
)

#%%
# Visualize the trained agent
print("\n=== Rendering Trained Agent ===")
env1_render = gymnasium.make(
    "ALE/Pacman-v5", obs_type='ram', render_mode='human'
)
render_gymnasium_agent(
    env=env1_render,
    actor=actor,
    num_episodes=3,
    deterministic=True,
)

# %%
# # Evaluate performance
# from utils.safety_utils import evaluate_agent_performance

# print("\n=== Evaluating Agent ===")
# results = evaluate_agent_performance(
#     env=env1,
#     actor=actor,
#     num_episodes=100
# )
# print(f"Average reward: {results['avg_reward']:.2f}")
# print(f"Average episode length: {results['avg_length']:.2f}")
# %%