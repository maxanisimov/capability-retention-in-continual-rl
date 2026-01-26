#%%
####### Imports #################################
import sys
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning/rl_project/poisoned_apple')
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning')
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning/rl_project')
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from poisoned_apple_env import PoisonedAppleEnv, evaluate_policy
from utils.ppo_utils import ppo_train, PPOConfig
from src.trainer import IntervalTrainer

plots_dir = '/Users/ma5923/Documents/_projects/CertifiedContinualLearning/rl_project/plots'

############### Utils #################################
### Visualize trained agent's trajectory
def visualize_agent_trajectory(env, actor, num_episodes=3, max_steps=None, env_name=None, save_dir=None):
    """
    Visualize the trained agent's trajectory in the environment.
    
    Args:
        env: The environment
        actor: Trained actor network
        num_episodes: Number of episodes to visualize
        max_steps: Maximum steps per episode (default: env.max_steps)
        env_name: Optional name for the environment (used in plot titles and filenames)
        save_dir: Optional directory to save plots. If None, plots are only displayed.
    """
    # if max_steps is None:
    #     max_steps = env.max_steps
    # if max_steps is None:
    #     max_steps = np.inf
    max_steps = np.inf
    
    action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
    
    for episode in range(num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print('='*50)
        
        obs, info = env.reset()
        trajectory = []
        rewards_list = []
        actions_list = []
        
        # Store initial state
        trajectory.append({
            'agent_pos': tuple(env.agent_pos),
            'safe_apples': set(env.safe_apples),
            'poisoned_apples': set(env.poisoned_apples)
        })
        
        done = False
        step = 0
        total_reward = 0
        
        while not done:
            # Get action from actor
            import torch
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_logits = actor(obs_tensor)
                action = torch.argmax(action_logits, dim=1).item()
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action) # type: ignore
            done = terminated or truncated
            total_reward += reward
            
            # Store trajectory
            trajectory.append({
                'agent_pos': tuple(env.agent_pos),
                'safe_apples': set(env.safe_apples),
                'poisoned_apples': set(env.poisoned_apples)
            })
            rewards_list.append(reward)
            actions_list.append(action)
            
            action_name = action_names[action] # type: ignore
            print(f"Step {step + 1}: {action_name}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
            step += 1

            if step >= max_steps:
                print("Reached maximum steps for this episode.")
                break
        
        print(f"\nEpisode finished! Total reward: {total_reward:.2f}")
        print(f"Apples remaining: {info['safe_apples_remaining']} safe, {info['poisoned_apples_remaining']} poisoned")
        
        # Plot trajectory
        plot_trajectory(env, trajectory, rewards_list, actions_list, episode + 1, env_name=env_name, save_dir=save_dir)
    
    if save_dir is None:
        plt.show()

def plot_trajectory(env, trajectory, rewards_list, actions_list, episode_num, env_name=None, save_dir=None):
    """
    Plot a single trajectory as a static image.
    
    Args:
        env: The environment
        trajectory: List of states
        rewards_list: List of rewards
        actions_list: List of actions
        episode_num: Episode number for title
        env_name: Optional environment name for title
        save_dir: Optional directory to save the plot. If None, plot is only displayed.
    """
    grid_size = env.grid_size
    num_steps = len(trajectory)
    
    # Create figure with subplots for each step
    cols = min(5, num_steps)
    rows = (num_steps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if num_steps == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    action_names = ["↑", "→", "↓", "←"]
    
    for step_idx, state in enumerate(trajectory):
        row = step_idx // cols
        col = step_idx % cols
        ax = axes[row, col]
        
        # Create grid
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.invert_yaxis()
        
        # Draw safe apples (green circles)
        for pos in state['safe_apples']:
            circle = patches.Circle((pos[1], pos[0]), 0.3, color='green', alpha=0.6)
            ax.add_patch(circle)
            ax.text(pos[1], pos[0], 'A', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white')
        
        # Draw poisoned apples (red circles)
        for pos in state['poisoned_apples']:
            circle = patches.Circle((pos[1], pos[0]), 0.3, color='red', alpha=0.6)
            ax.add_patch(circle)
            ax.text(pos[1], pos[0], 'P', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
        
        # Draw agent (blue square)
        agent_pos = state['agent_pos']
        rect = patches.Rectangle((agent_pos[1] - 0.35, agent_pos[0] - 0.35), 
                                0.7, 0.7, color='blue', alpha=0.8)
        ax.add_patch(rect)
        ax.text(agent_pos[1], agent_pos[0], '●', ha='center', va='center',
               fontsize=16, fontweight='bold', color='white')
        
        # Title for each step
        if step_idx == 0:
            ax.set_title(f'Start', fontsize=10, fontweight='bold')
        else:
            action = action_names[actions_list[step_idx - 1]]
            reward = rewards_list[step_idx - 1]
            reward_color = 'green' if reward > 0 else ('red' if reward < 0 else 'gray')
            ax.set_title(f'Step {step_idx}: {action} (r={reward:.2f})', 
                        fontsize=10, fontweight='bold', color=reward_color)
    
    # Hide empty subplots
    for step_idx in range(num_steps, rows * cols):
        row = step_idx // cols
        col = step_idx % cols
        axes[row, col].axis('off')
    
    suptitle = f'Episode {episode_num} - Agent Trajectory'
    if env_name is not None:
        suptitle = env_name + ' - ' + suptitle
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save the figure if save_dir is provided
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        filename = f"trajectory_episode_{episode_num}"
        if env_name is not None:
            # Clean env_name for filename (replace spaces and special chars)
            clean_env_name = env_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename = f"{clean_env_name}_{filename}"
        filepath = os.path.join(save_dir, f"{filename}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {filepath}")
        plt.close(fig)

#%%
### CONFIGS (shared across all environments)
max_steps = 9
grid_size = 5
observation_type = "flat"
agent_start_pos = (0, 0)
safe_env1_state_action_data_num_rollouts = 1  # NOTE: one episode is sufficient because Env1 and standard_actor are deterministic
seed = 42

#%%
######################################################
print("\n\n=== Flat Observation Demo ===")
# Create environment with flat observations
env = PoisonedAppleEnv(
    grid_size=grid_size,
    agent_start_pos=agent_start_pos,
    safe_apple_positions=[(1, 1), (2, 2)],
    poisoned_apple_positions=[(3, 3)],
    observation_type=observation_type,
    render_mode="human",
    max_steps=max_steps,
    seed=seed
)

#%%
### Train using ppo
ppo_cfg = PPOConfig(
    total_timesteps=1_000,
)
standard_actor, standard_critic = ppo_train(
    env=env,
    cfg=ppo_cfg,
)

#%%
### Visualize the trained agent in Env 1
visualize_agent_trajectory(
    env, standard_actor, num_episodes=1, env_name='Env 1',
    # save_dir=plots_dir
)
# %%
### In Env 2, one of the safe apples becomes poisoned :(
env2 = PoisonedAppleEnv(
    grid_size=grid_size,
    agent_start_pos=agent_start_pos,
    safe_apple_positions=[(2, 2)],
    poisoned_apple_positions=[(1, 1), (3, 3)],
    observation_type=observation_type,
    render_mode="human",
    max_steps=max_steps,
    seed=seed
)

# Visualize the trained agent in Env 2
visualize_agent_trajectory(
    env2, standard_actor, num_episodes=1, max_steps=max_steps, env_name='Env 2',
    # save_dir=plots_dir
)

#%%
# ### OPTIONAL: train a new agent from scratch in Env 2
# print("\n\n=== Training new agent in Env 2 (with poisoned apple) ===")
# # NOTE: For good performance, it is critical to:
# # set total_timesteps = 10_000
# # use actor and critic warm starts from Env 1
# ppo_cfg_amnesic = PPOConfig(
#     total_timesteps=20_000,
#     # ent_coef=1,
#     # lr=0.01
# )
# amnesic_actor, _ = ppo_train(
#     env=env2,
#     cfg=ppo_cfg_amnesic,
#     actor_warm_start=standard_actor,
#     critic_warm_start=standard_critic
# )
# # How does the new amnesic_actor perform in Env 1?
# visualize_agent_trajectory(env, amnesic_actor, num_episodes=1, max_steps=max_steps, env_name='Env 1 - Amnesic Actor')
# # Visualize the amnesic_actor in Env 2
# visualize_agent_trajectory(env2, amnesic_actor, num_episodes=1, max_steps=max_steps, env_name='Env 2 - Amnesic Actor')

#%%
### Generate dataset that contains safe actions for each state visited by standard_actor in Env1
### SAFE RL POLICY UPDATE
# The idea is to update the policy on Task 2 using a safe RL method
# such that it retains safety on Task 1 while improving performance on Task 2.
# Here, we compute the Rashomon set based on safety constraints from Task 1
# and then update the policy to perform well in the Task 2.

# ### Hardcoded safe actions
# # For each state in Env1, provide a safe action that avoids poisoned apples.
# safe_action_map = {}
# for row in range(env.grid_size):
#     for col in range(env.grid_size):
#         state = np.zeros((env.grid_size, env.grid_size), dtype=int)
#         state[row, col] = 1  # Agent position
#         # Safe apples
#         for apple_pos in [(1, 1), (2, 2)]:
#             state[apple_pos] = 2
#         # Poisoned apple
#         state[(3, 3)] = 3
#         flat_state = state.flatten()
        
#         # Define safe actions based on agent position
#         if (row, col) == (0, 0):
#             safe_action = PoisonedAppleEnv.DOWN  # Move down
#         elif (row, col) == (1, 0):
#             safe_action = PoisonedAppleEnv.RIGHT  # Move right
#         elif (row, col) == (1, 1):
#             safe_action = PoisonedAppleEnv.DOWN  # Move down
#         elif (row, col) == (2, 1):
#             safe_action = PoisonedAppleEnv.RIGHT  # Move right
#         elif (row, col) == (2, 2):
#             safe_action = PoisonedAppleEnv.UP  # Move up
#         else:
#             safe_action = PoisonedAppleEnv.UP  # Default safe action
        
#         safe_action_map[tuple(flat_state)] = safe_action

# # Generate dataset
# states = []
# actions = []
# obs, info = env.reset()
# done = False
# while not done:
#     # Get action from standard_actor
#     with torch.no_grad():
#         obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
#         action_logits = standard_actor(obs_tensor)
#         action = torch.argmax(action_logits, dim=1).item()
    
#     # Record state and safe action
#     flat_obs = tuple(obs)
#     if flat_obs in safe_action_map:
#         safe_action = safe_action_map[flat_obs]
#         states.append(obs)
#         actions.append(safe_action)
    
#     # Take step
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated
# states = torch.FloatTensor(states)
# actions = torch.LongTensor(actions)

# NOTE: it is important that standard_actor's behaviour is safe in Env1 along its trajectory
# NOTE: I do not think we require safety for all possible states in Env1, only those visited by standard_actor


### Generate dataset from standard_actor's behavior in Env1 (no hardcoding needed!)
states = []
actions = []

# Collect multiple rollouts to get diverse state coverage
for episode in range(safe_env1_state_action_data_num_rollouts):
    obs, info = env.reset()
    done = False
    while not done:
        # Get action from standard_actor (the trained safe policy)
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_logits = standard_actor(obs_tensor)
            action = torch.argmax(action_logits, dim=1).item()
        
        # Record state-action pair (standard_actor's behavior IS the safe behavior)
        states.append(obs)
        actions.append(action)
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action) # type: ignore
        done = terminated or truncated

states = torch.FloatTensor(states)
actions = torch.LongTensor(actions)

state_action_torch_dataset = torch.utils.data.TensorDataset(states, actions)

# %%
### Rashomon Set
interval_trainer = IntervalTrainer(
    model=standard_actor, # policy which is an instance of nn.Sequential
    min_acc_limit=0.99, # NOTE: should be not greater than accuracy of the model
    seed=seed
    # n_iters=10_000, # default 2000; running longer may not translate into higher OOS accuracy
)
interval_trainer.compute_rashomon_set(
    dataset=state_action_torch_dataset, # states and safe actions
    multi_label=False # NOTE: when set to True, the policy deviates more from the original policy in Env1 (but is still safe)
)
# Extract parameter bounds from the bounded model
assert len(interval_trainer.bounds) == 1, "Expected exactly one bounded model"
certificate = interval_trainer.certificates[0]
print(f"\nRashomon set computed. Certified accuracy on safe action dataset: {certificate:.2f}")
bounded_model = interval_trainer.bounds[0]
param_bounds_l = [bound.detach().cpu() for bound in bounded_model.param_l]
param_bounds_u = [bound.detach().cpu() for bound in bounded_model.param_u]

#%%
### Train a safe actor
ppo_cfg_rashomon = PPOConfig(
    total_timesteps=5_000, # >4000 needed to be safe and performant in Env2
)
rashomon_actor, _ = ppo_train(
    env=env2,
    cfg=ppo_cfg_rashomon,
    actor_warm_start=standard_actor,
    critic_warm_start=standard_critic,
    actor_param_bounds_l=param_bounds_l,
    actor_param_bounds_u=param_bounds_u
)

# Visualize the trained safe actor in Env 1
visualize_agent_trajectory(
    env, rashomon_actor, num_episodes=1, max_steps=10, env_name='Env 1',
    # save_dir=plots_dir
)

# Visualize the trained safe actor in Env 2
visualize_agent_trajectory(
    env2, rashomon_actor, num_episodes=1, max_steps=10, env_name='Env 2',
    # save_dir=plots_dir
)

# %%
### Evaluate policies
print("\n\n=== Policy Evaluations ===")
# Evaluate standard_actor
num_eval_episodes = 1 # it is ok because environments and actors are deterministic
standard_actor_env1_metrics = evaluate_policy(env, standard_actor, num_episodes=num_eval_episodes)
standard_actor_env2_metrics = evaluate_policy(env2, standard_actor, num_episodes=num_eval_episodes)

# Evaluate rashomon_actor
rashomon_actor_env1_metrics = evaluate_policy(env, rashomon_actor, num_episodes=num_eval_episodes)
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
# Make sure standard actor is unsafe in Env 2
assert results_df.loc['avg_safety_success', 'Standard Actor / Env 2'] < 1.0, "Standard actor should be unsafe in Env 2" # type: ignore
# Make sure Rashomon actor is safe in Env 1 # TODO: I probably should compare to the certificate value here
assert results_df.loc['avg_safety_success', 'Rashomon Actor / Env 1'] == 1.0, "Rashomon actor should be safe in Env 1"
# And in Env 2
assert results_df.loc['avg_safety_success', 'Rashomon Actor / Env 2'] == 1.0, "Rashomon actor should be safe in Env 2"

#%%
# Ablation study TODO:
# 1) Show that total_timesteps increase for Actor 1 training does not help Actor 1 to be safe and performant in Env 2