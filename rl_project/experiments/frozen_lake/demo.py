#%%
import os
import sys
import torch
import numpy as np
import pandas as pd
import yaml
import gymnasium as gym
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning/rl_project')
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning')
from utils.ppo_utils import ppo_train, PPOConfig
from src.trainer import IntervalTrainer

"""
Frozen Lake Environment Demo: Safe Continual Learning with Rashomon Sets

This demonstration adapts the poisoned apple safe continual learning approach to the
Gymnasium FrozenLake environment, where an agent must navigate from start to goal
while avoiding holes in the ice.

Scenario:
- Environment 1 (Env1): Standard 4x4 FrozenLake with fixed hole positions
- Environment 2 (Env2): Modified FrozenLake with different hole positions (distribution shift)
- Challenge: Adapt to Env2 while maintaining safety in Env1 (avoid catastrophic forgetting)

Approach: Safe Optimal Policy Demonstrations
--------------------------------------------
Collects state-action pairs from a deterministic policy that is both safe AND optimal
in Env1 (reaches goal while avoiding holes).

Key characteristics:
- Data collection: Deterministic rollouts using argmax policy
- Dataset structure: Single action per state (deterministic trajectory)
- Rashomon constraint: Uses multi_label=False for strict enforcement
- Guarantees: Maintains both SAFETY and NEAR-OPTIMAL PERFORMANCE on Env1
"""

sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning/rl_project')
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

current_script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(current_script_dir, 'plots')
frames_dir = os.path.join(current_script_dir, 'frames')
tables_dir = os.path.join(current_script_dir, 'tables')
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

############### Utils #################################

def evaluate_policy(env, actor, num_episodes=100):
    """
    Evaluate a policy in the given environment.
    
    Returns:
        dict: Metrics including avg_reward, avg_success, avg_safety_success, avg_steps
    """
    total_reward = 0
    total_success = 0
    total_safety_success = 0
    total_steps = 0
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        fell_in_hole = False
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_logits = actor(obs_tensor)
                action = torch.argmax(action_logits, dim=1).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
            
            # Check if agent fell in hole (terminated but no reward)
            if terminated and reward == 0:
                fell_in_hole = True
        
        total_reward += episode_reward
        total_success += int(reward > 0)  # Reached goal
        total_safety_success += int(not fell_in_hole)  # Didn't fall in hole
        total_steps += episode_steps
    
    return {
        'avg_reward': total_reward / num_episodes,
        'avg_success': total_success / num_episodes,
        'avg_safety_success': total_safety_success / num_episodes,
        'avg_steps': total_steps / num_episodes
    }


def visualize_agent_trajectory(env, actor, num_episodes=3, max_steps=100, 
                               env_name=None, cfg_name=None, actor_name=None, save_dir=None):
    """
    Visualize the trained agent's trajectory in FrozenLake.
    """
    action_names = ["LEFT", "DOWN", "RIGHT", "UP"]
    
    for episode in range(num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print('='*50)
        
        obs, info = env.reset()
        trajectory = []
        rewards_list = []
        actions_list = []
        
        # Store initial state
        trajectory.append(obs)
        
        done = False
        step = 0
        total_reward = 0
        
        while not done and step < max_steps:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_logits = actor(obs_tensor)
                action = torch.argmax(action_logits, dim=1).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            trajectory.append(obs)
            rewards_list.append(reward)
            actions_list.append(action)
            
            action_name = action_names[action]
            print(f"Step {step + 1}: {action_name}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
            step += 1
        
        print(f"\nEpisode finished! Total reward: {total_reward:.2f}")
        
        # Plot trajectory
        plot_trajectory(env, trajectory, rewards_list, actions_list, 
                       episode_num=episode + 1 if num_episodes > 1 else None,
                       env_name=env_name, cfg_name=cfg_name, actor_name=actor_name, 
                       save_dir=save_dir)
    
    if save_dir is None:
        plt.show()


def plot_trajectory(env, trajectory, rewards_list, actions_list, 
                   episode_num=None, env_name=None, cfg_name=None, actor_name=None, save_dir=None):
    """
    Plot a single trajectory for FrozenLake.
    """
    # Get environment layout
    desc = env.unwrapped.desc
    nrow, ncol = desc.shape
    num_steps = len(trajectory)
    
    # Create figure with subplots for each step
    cols = min(5, num_steps)
    rows = (num_steps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if num_steps == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    action_names = ["←", "↓", "→", "↑"]
    
    for step_idx, state in enumerate(trajectory):
        row = step_idx // cols
        col = step_idx % cols
        ax = axes[row, col]
        
        # Create grid
        ax.set_xlim(-0.5, ncol - 0.5)
        ax.set_ylim(-0.5, nrow - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.set_xticks(range(ncol))
        ax.set_yticks(range(nrow))
        ax.tick_params(labelsize=12)
        ax.invert_yaxis()
        
        # Draw environment
        for i in range(nrow):
            for j in range(ncol):
                cell = desc[i, j].decode('utf-8') if isinstance(desc[i, j], bytes) else desc[i, j]
                
                if cell == 'S':  # Start
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                            color='lightgreen', alpha=0.3)
                    ax.add_patch(rect)
                    ax.text(j, i, 'S', ha='center', va='center', 
                           fontsize=14, fontweight='bold', color='green')
                elif cell == 'F':  # Frozen (safe)
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                            color='lightblue', alpha=0.2)
                    ax.add_patch(rect)
                elif cell == 'H':  # Hole (unsafe)
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                            color='red', alpha=0.3)
                    ax.add_patch(rect)
                    ax.text(j, i, 'H', ha='center', va='center', 
                           fontsize=14, fontweight='bold', color='darkred')
                elif cell == 'G':  # Goal
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                            color='gold', alpha=0.4)
                    ax.add_patch(rect)
                    ax.text(j, i, 'G', ha='center', va='center', 
                           fontsize=14, fontweight='bold', color='darkgoldenrod')
        
        # Draw agent position
        agent_row = state // ncol
        agent_col = state % ncol
        circle = patches.Circle((agent_col, agent_row), 0.3, color='blue', alpha=0.8)
        ax.add_patch(circle)
        ax.text(agent_col, agent_row, '●', ha='center', va='center',
               fontsize=20, fontweight='bold', color='white')
        
        # Title for each step
        if step_idx == 0:
            ax.set_title(f'Start', fontsize=13, fontweight='bold')
        else:
            action = action_names[actions_list[step_idx - 1]]
            reward = rewards_list[step_idx - 1]
            reward_color = 'green' if reward > 0 else ('red' if reward < 0 else 'gray')
            ax.set_title(f'Step {step_idx}: {action} (r={reward:.2f})', 
                        fontsize=13, fontweight='bold', color=reward_color)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)
    
    # Hide empty subplots
    for step_idx in range(num_steps, rows * cols):
        row = step_idx // cols
        col = step_idx % cols
        axes[row, col].axis('off')
    
    suptitle = ''
    if cfg_name is not None:
        suptitle = suptitle + cfg_name
    if env_name is not None:
        suptitle = suptitle + ' - ' + env_name
    if actor_name is not None:
        suptitle = suptitle + ' - ' + actor_name
    if episode_num is not None:
        suptitle = suptitle + ' - ' + f'Episode {episode_num}'
    fig.suptitle(suptitle, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_dir is not None:
        filename_parts = []
        if cfg_name is not None:
            filename_parts.append(cfg_name)
        if env_name is not None:
            clean_env_name = env_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename_parts.append(clean_env_name)
        if actor_name is not None:
            clean_actor_name = actor_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename_parts.append(clean_actor_name)
        if episode_num is not None:
            filename_parts.append(f"episode_{episode_num}")

        filename = "_".join(filename_parts) + ".png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {filepath}")
        plt.close(fig)


def one_hot_encode_state(state, num_states, task_num: int = 0) -> np.ndarray:
    """Convert discrete state to one-hot encoding."""
    encoded = np.zeros(num_states, dtype=np.float32)
    encoded[state] = 1.0
    # Also, add the task indicator if needed (for multi-task settings)
    encoded = np.append(encoded, task_num)
    return encoded


class SafetyFlagWrapper(gym.Wrapper):
    """Wrapper that adds a safety flag to the info dict indicating if current state is safe (not a hole)."""
    
    def __init__(self, env):
        super().__init__(env)
        # Get the map description
        self.desc = env.unwrapped.desc
        self.nrow, self.ncol = self.desc.shape
        
    def _is_safe_state(self, state):
        """Check if a state is safe (not a hole)."""
        row = state // self.ncol
        col = state % self.ncol
        cell = self.desc[row, col]
        cell = cell.decode('utf-8') if isinstance(cell, bytes) else cell
        return cell != 'H'  # Safe if not a hole
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Add safety flag for initial state
        if isinstance(obs, int):
            state = obs
        else:
            state_ohe = obs[:-1]  # Exclude task indicator
            state = np.argmax(state_ohe)
        info['safe'] = self._is_safe_state(state)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Add safety flag - True if current state is not a hole
        if isinstance(obs, int):
            current_state = obs
        else:
            state_ohe = obs[:-1]  # Exclude task indicator
            current_state = np.argmax(state_ohe)
        info['safe'] = self._is_safe_state(current_state)
        return obs, reward, terminated, truncated, info


#%%
### CONFIGS
cfg_name = 'frozen_lake_4x4'
safe_state_action_data_name = 'Safe Optimal Policy Data' # 'Safe Optimal Policy Data' or 'Safe Training Data'
save_results = True
seed = 42

env1_map = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]
env2_map = [
    "SHFF",
    "FFFH",
    "FHFF",
    "HFFG"
]

# FrozenLake parameters
max_steps = 100
safe_demonstrations_policy_env1_num_episodes = 10
unadaptable_actor_timesteps = 50000
rashomon_timesteps = 20000

if not save_results:
    plots_dir = None
    tables_dir = None

#%%
######################################################
print("\n\n=== FrozenLake Environment Demo ===")

# Create Env1: Standard FrozenLake
env1 = gym.make('FrozenLake-v1', desc=env1_map, is_slippery=False, render_mode=None)
env1.reset(seed=seed)
num_states = env1.observation_space.n
num_actions = env1.action_space.n

# Wrapper to provide one-hot encoded observations
class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env, task_num: int):
        super().__init__(env)
        low_bounds = np.zeros(env.observation_space.n + 1, dtype=np.float32)
        high_bounds = np.ones(env.observation_space.n + 1, dtype=np.float32)
        high_bounds[-1] = np.inf  # Last dimension (task indicator) is unbounded above
        self.observation_space = gym.spaces.Box(
            low=low_bounds, high=high_bounds, dtype=np.float32
        )
        self.task_num = task_num
    
    def observation(self, obs):
        return one_hot_encode_state(obs, self.env.observation_space.n, self.task_num)

env1 = OneHotWrapper(env1, task_num=0)
env1 = SafetyFlagWrapper(env1)

#%%
### Train using PPO
print("\n=== Training Standard Actor on Env1 ===")
ppo_cfg = PPOConfig(
    total_timesteps=unadaptable_actor_timesteps,
)
standard_actor, standard_critic, standard_training_data = ppo_train(
    env=env1,
    cfg=ppo_cfg,
    return_training_data=True
)

#%%
# ### Visualize the trained agent in Env 1
# visualize_agent_trajectory(
#     env1, standard_actor, num_episodes=1, max_steps=max_steps,
#     env_name='Env 1', cfg_name=cfg_name, actor_name='Standard Actor', save_dir=plots_dir
# )

# from utils.gymnasium_utils import render_gymnasium_agent
# env1_render = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human')
# env1_render = OneHotWrapper(env1_render, task_num=0)
# render_gymnasium_agent(
#     env=env1_render,
#     actor=standard_actor,
#     num_episodes=1,
#     deterministic=True,
#     seed=seed
# )

standard_actor_env1_eval_results = evaluate_policy(env1, standard_actor, num_episodes=1)

#%%
### Create Env2 with modified hole positions (distribution shift)
print("\n=== Creating Env2 with Distribution Shift ===")
# Custom map with different hole positions
env2 = gym.make('FrozenLake-v1', desc=env2_map, is_slippery=False, render_mode=None)
env2 = OneHotWrapper(env2, task_num=1)  # Different task_num to distinguish from env1
env2 = SafetyFlagWrapper(env2)
standard_actor_env2_eval_results = evaluate_policy(env2, standard_actor, num_episodes=1)

# # Visualize the standard actor in Env 2
# visualize_agent_trajectory(
#     env2, standard_actor, num_episodes=1, max_steps=max_steps,
#     env_name='Env 2', cfg_name=cfg_name, actor_name='Standard Actor',
#     save_dir=plots_dir
# )

#%%
# from utils.gymnasium_utils import render_gymnasium_agent
# env2_render = gym.make('FrozenLake-v1', desc=env2_map, is_slippery=False, render_mode='human')
# env2_render = OneHotWrapper(env2_render, task_num=1)
# render_gymnasium_agent(
#     env=env2_render,
#     actor=standard_actor,
#     num_episodes=1,
#     deterministic=True,
#     seed=seed
# )

#%%
# ### Generate safe state-action dataset from Env1
# print("\n=== Generating Safe State-Action Dataset ===")
# states = []
# actions = []

# for episode in range(safe_env1_state_action_data_num_rollouts):
#     obs, info = env1.reset()
#     done = False
#     visited_states = set()
    
#     while not done:
#         state_tuple = tuple(obs)
#         if state_tuple not in visited_states:
#             visited_states.add(state_tuple)
            
#             with torch.no_grad():
#                 obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
#                 action_logits = standard_actor(obs_tensor)
#                 action = torch.argmax(action_logits, dim=1).item()
            
#             states.append(obs)
#             actions.append(action)
            
#             obs, reward, terminated, truncated, info = env1.step(action)
#             done = terminated or truncated
#         else:
#             break

# states = torch.FloatTensor(np.array(states))
# actions = torch.LongTensor(np.array(actions))
# safe_optimal_policy_data_torch_dataset = torch.utils.data.TensorDataset(states, actions)

# print(f"Collected {len(states)} safe state-action pairs")

from utils.safety_utils import get_unique_safe_state_action_pairs, generate_safe_optimal_policy_data
print("\n=== Generating Safe State-Action Dataset ===")

safe_state_action_data_name = 'Safe Optimal Policy Data'  # 'Safe Training Data' or 'Safe Optimal Policy Data'

if safe_state_action_data_name == 'Safe Training Data':
    safe_state_action_torch_dataset = get_unique_safe_state_action_pairs(
        training_data=standard_training_data
    )
else:
    safe_state_action_torch_dataset = generate_safe_optimal_policy_data(
        env=env1,
        safe_actor=standard_actor,
        num_episodes=safe_demonstrations_policy_env1_num_episodes,
        deterministic=True
    )

print(f"Collected {len(safe_state_action_torch_dataset)} unique safe state-action pairs")

from visualize_safe_dataset import visualize_safe_state_action_dataset
save_path = None
if plots_dir is not None:
    save_path=plots_dir + f'/safe_state_action_dataset_Env1_{safe_state_action_data_name}.png'

visualize_safe_state_action_dataset(
    dataset=safe_state_action_torch_dataset, 
    env_map=env1_map,
    save_path=save_path
)

#%%
### Compute Rashomon Set
print("\n=== Computing Rashomon Set ===")
interval_trainer = IntervalTrainer(
    model=standard_actor,
    min_acc_limit=0.99,
    seed=seed
)
interval_trainer.compute_rashomon_set(
    dataset=safe_state_action_torch_dataset,
    multi_label=True
)

assert len(interval_trainer.bounds) == 1, "Expected exactly one bounded model"
certificate = interval_trainer.certificates[0]
print(f"\nRashomon set computed. Certified accuracy on safe action dataset: {certificate:.2f}")

bounded_model = interval_trainer.bounds[0]
param_bounds_l = [bound.detach().cpu() for bound in bounded_model.param_l]
param_bounds_u = [bound.detach().cpu() for bound in bounded_model.param_u]

#%%
### Train Rashomon-constrained actor on Env2
print("\n=== Training Rashomon Actor on Env2 ===")
ppo_cfg_rashomon = PPOConfig(
    total_timesteps=rashomon_timesteps,
)
rashomon_actor, _ = ppo_train(
    env=env2,
    cfg=ppo_cfg_rashomon,
    actor_warm_start=standard_actor,
    critic_warm_start=standard_critic,
    actor_param_bounds_l=param_bounds_l,
    actor_param_bounds_u=param_bounds_u
)

# # Visualize the Rashomon actor in Env 1
# visualize_agent_trajectory(
#     env1, rashomon_actor, num_episodes=1, max_steps=max_steps,
#     env_name='Env 1', cfg_name=cfg_name, 
#     actor_name=f'Rashomon Actor ({safe_state_action_data_name})', save_dir=plots_dir
# )

# # Visualize the Rashomon actor in Env 2
# visualize_agent_trajectory(
#     env2, rashomon_actor, num_episodes=1, max_steps=max_steps,
#     env_name='Env 2', cfg_name=cfg_name, 
#     actor_name=f'Rashomon Actor ({safe_state_action_data_name})',
#     save_dir=plots_dir
# )

#%%
### Evaluate policies
print("\n\n=== Policy Evaluations ===")
num_eval_episodes = 100

standard_actor_env1_metrics = evaluate_policy(env1, standard_actor, num_episodes=num_eval_episodes)
standard_actor_env2_metrics = evaluate_policy(env2, standard_actor, num_episodes=num_eval_episodes)

rashomon_actor_env1_metrics = evaluate_policy(env1, rashomon_actor, num_episodes=num_eval_episodes)
rashomon_actor_env2_metrics = evaluate_policy(env2, rashomon_actor, num_episodes=num_eval_episodes)

#%%
# Create dataframe to summarize results
results_df = pd.DataFrame({
    'Standard Actor / Env 1': standard_actor_env1_metrics,
    'Standard Actor / Env 2': standard_actor_env2_metrics,
    'Rashomon Actor / Env 1': rashomon_actor_env1_metrics,
    'Rashomon Actor / Env 2': rashomon_actor_env2_metrics
})
print("\n=== Evaluation Results ===")
print(results_df.round(2))

#%%
# Verify safety properties
assert results_df.loc['avg_safety_success', 'Rashomon Actor / Env 1'] >= 0.95, \
    "Rashomon actor should maintain safety in Env 1"

if save_results and tables_dir is not None:
    results_df.to_csv(
        f'{tables_dir}/frozen_lake_demo_results_{cfg_name}_{safe_state_action_data_name}.csv'
    )
    print(f"\nResults saved to {tables_dir}")
#%%
### Collect trajectory frames for visualization
print("\n=== Collecting Trajectory Frames ===")

def collect_trajectory_frames(env_map, task_num, actor, actor_name, env_name, seed=42, max_steps=100):
    """Collect frames showing agent trajectory in the environment."""
    env_render = gym.make('FrozenLake-v1', desc=env_map, is_slippery=False, render_mode='rgb_array')
    env_render = OneHotWrapper(env_render, task_num=task_num)
    env_render = SafetyFlagWrapper(env_render)
    
    frames = []
    obs, info = env_render.reset(seed=seed)
    frames.append(env_render.render())
    
    done = False
    step = 0
    
    while not done and step < max_steps:
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_logits = actor(obs_tensor)
            action = torch.argmax(action_logits, dim=1).item()
        
        obs, reward, terminated, truncated, info = env_render.step(action)
        frames.append(env_render.render())
        done = terminated or truncated
        step += 1
    
    env_render.close()
    
    print(f"Collected {len(frames)} frames for {actor_name} in {env_name}")
    return frames

# Collect frames for all combinations
standard_env1_frames = collect_trajectory_frames(
    env1_map, task_num=0, actor=standard_actor, 
    actor_name="Standard Actor", env_name="Env1", seed=seed
)

standard_env2_frames = collect_trajectory_frames(
    env2_map, task_num=1, actor=standard_actor,
    actor_name="Standard Actor", env_name="Env2", seed=seed
)

rashomon_env1_frames = collect_trajectory_frames(
    env1_map, task_num=0, actor=rashomon_actor,
    actor_name="Rashomon Actor", env_name="Env1", seed=seed
)

rashomon_env2_frames = collect_trajectory_frames(
    env2_map, task_num=1, actor=rashomon_actor,
    actor_name="Rashomon Actor", env_name="Env2", seed=seed
)

# Store frames in a dictionary for easy access
trajectory_frames = {
    'standard_env1': standard_env1_frames,
    'standard_env2': standard_env2_frames,
    'rashomon_env1': rashomon_env1_frames,
    'rashomon_env2': rashomon_env2_frames
}

print(f"\nAll trajectory frames collected successfully!")
print(f"Frame counts:")
for key, frames in trajectory_frames.items():
    print(f"  {key}: {len(frames)} frames")

# Optional: Save frames as images
if save_results and frames_dir is not None:
    import matplotlib.pyplot as plt
    
    print("\n=== Saving Trajectory Frames ===")
    
    for key, frames in trajectory_frames.items():
        frame_dir = os.path.join(frames_dir, f'frames_{key}')
        os.makedirs(frame_dir, exist_ok=True)
        
        for idx, frame in enumerate(frames):
            plt.figure(figsize=(6, 6))
            plt.imshow(frame)
            plt.axis('off')
            plt.title(f"{key.replace('_', ' ').title()} - Step {idx}")
            plt.tight_layout()
            plt.savefig(os.path.join(frame_dir, f'frame_{idx:03d}.png'), dpi=100, bbox_inches='tight')
            plt.close()
        
        print(f"Saved {len(frames)} frames to {frame_dir}")
    
    print("\nAll frames saved successfully!")
