#%%
####### Imports #################################
import sys
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning/experiments/poisoned_apple')
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning/experiments')
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning')
import torch
from poisoned_apple_env import PoisonedAppleEnv #make_task1_env, make_task2_env
from _ppo_utils import make_actor_critic, ppo_train, PPOConfig
from src.trainer import IntervalTrainer

############### Utils #################################
### Visualize trained agent's trajectory
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np

def visualize_agent_trajectory(env, actor, num_episodes=3, max_steps=None, env_name=None):
    """
    Visualize the trained agent's trajectory in the environment.
    
    Args:
        env: The environment
        actor: Trained actor network
        num_episodes: Number of episodes to visualize
        max_steps: Maximum steps per episode (default: env.max_steps)
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
            obs, reward, terminated, truncated, info = env.step(action)
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
            
            print(f"Step {step + 1}: {action_names[action]}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
            step += 1

            if step >= max_steps:
                print("Reached maximum steps for this episode.")
                break
        
        print(f"\nEpisode finished! Total reward: {total_reward:.2f}")
        print(f"Apples remaining: {info['safe_apples_remaining']} safe, {info['poisoned_apples_remaining']} poisoned")
        
        # Plot trajectory
        plot_trajectory(env, trajectory, rewards_list, actions_list, episode + 1, env_name=env_name)
    
    plt.show()

def plot_trajectory(env, trajectory, rewards_list, actions_list, episode_num, env_name=None):
    """
    Plot a single trajectory as a static image.
    
    Args:
        env: The environment
        trajectory: List of states
        rewards_list: List of rewards
        actions_list: List of actions
        episode_num: Episode number for title
        env_name: Optional environment name for title
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

#%%
######################################################
# Demo the environment
# print("=== Task 1 Demo: No Poisoned Apples ===")
# env =  make_task1_env(render_mode="human")
# obs, info = env.reset(seed=42)
# env.render()

# print("\nTaking random actions...")
# for _ in range(5):
#     action = env.action_space.sample()
#     action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
#     obs, reward, terminated, truncated, info = env.step(action)
#     print(f"Action: {action_names[action]}, Reward: {reward}")
#     env.render()
    
#     if terminated or truncated:
#         print("Episode finished!")
#         break

# env.close()

# print("\n\n=== Task 2 Demo: One Poisoned Apple ===")
# env = make_task2_env(render_mode="human")
# obs, info = env.reset(seed=42)
# env.render()

# print("\nTaking random actions...")
# for _ in range(5):
#     action = env.action_space.sample()
#     action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
#     obs, reward, terminated, truncated, info = env.step(action)
#     print(f"Action: {action_names[action]}, Reward: {reward}")
#     env.render()
    
#     if terminated or truncated:
#         print("Episode finished!")
#         break

# env.close()

print("\n\n=== Flat Observation Demo ===")
# Create environment with flat observations
env = PoisonedAppleEnv(
    grid_size=5,
    num_apples=3,
    num_poisoned=1,
    agent_start_pos=(0, 0),
    safe_apple_positions=[(1, 1), (2, 2)],
    poisoned_apple_positions=[(3, 3)],
    observation_type="flat",
    render_mode="human"
)
# obs, info = env.reset()
# print("Environment with flat observations:")
# print(f"Observation space: {env.observation_space}")
# print(f"Observation shape: {obs.shape}")
# print(f"Initial observation (first 15 values): {obs[:15]}")
# print(f"Format: Flattened 5x5 grid, values: 0=empty, 1=agent, 2=safe, 3=poisoned")
# env.render()

# print("\nCollecting first apple...")
# for action in [PoisonedAppleEnv.DOWN, PoisonedAppleEnv.RIGHT]:
#     action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
#     obs, reward, terminated, truncated, info = env.step(action)
#     print(f"Action: {action_names[action]}, Reward: {reward}")
#     print(f"New observation (first 15 values): {obs[:15]}")
#     env.render()
#     if terminated or truncated:
#         break

# env.close()

# # %%
# ### Create actor-critic for flat observation space
# obs_dim = env.observation_space.shape[0]
# num_actions = env.action_space.n
# actor, critic = make_actor_critic(obs_dim, num_actions)

#%%
### Train using ppo
ppo_cfg = PPOConfig(
    total_timesteps=1_000,
)
actor1, critic1 = ppo_train(
    env=env,
    cfg=ppo_cfg,
)

#%%
# Visualize the trained agent in Env 1
visualize_agent_trajectory(env, actor1, num_episodes=1, env_name='Env 1')

# %%
### In Env 2, one of the safe apples becomes poisoned :(
max_steps = 10
env2 = PoisonedAppleEnv(
    grid_size=5,
    num_apples=3,
    num_poisoned=2,
    agent_start_pos=(0, 0),
    safe_apple_positions=[(2, 2)],
    poisoned_apple_positions=[(1, 1), (3, 3)],
    observation_type="flat",
    render_mode="human",
    max_steps=max_steps,
    seed=42
)

# Visualize the trained agent in Env 2
visualize_agent_trajectory(env2, actor1, num_episodes=1, max_steps=max_steps, env_name='Env 2')

#%%
# ### OPTIONAL: train a new agent from scratch in Env 2
# print("\n\n=== Training new agent in Env 2 (with poisoned apple) ===")
# # NOTE: It is critical to:
# # set total_timesteps = 10_000
# # use actor and critic warm starts from Env 1
# ppo_cfg2 = PPOConfig(
#     total_timesteps=10_000,
#     # ent_coef=1,
#     # lr=0.01
# )
# actor2, critic2 = ppo_train(
#     env=env2,
#     cfg=ppo_cfg2,
#     actor_warm_start=actor1,
#     critic_warm_start=critic1
# )
# # Visualize the trained agent in Env 2
# visualize_agent_trajectory(env2, actor2, num_episodes=1, max_steps=4)

# ### How does the new actor1 perform in Env 1?
# # visualize_agent_trajectory(env, actor2, num_episodes=1)
# visualize_agent_trajectory(env2, actor1, num_episodes=1)

#%%
### Generate dataset that contains safe actions for each state visited by actor1 in Env1
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
#     # Get action from actor1
#     with torch.no_grad():
#         obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
#         action_logits = actor1(obs_tensor)
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

### TODO: replicate the hardcoded experiment with a policy
# NOTE: it is important that actor1's behaviour is safe in Env1 along its trajectory
# NOTE: I do not think we require safety for all possible states in Env1, only those visited by actor1
### Generate dataset from actor1's behavior in Env1 (no hardcoding needed!)
states = []
actions = []

# Collect multiple rollouts to get diverse state coverage
num_rollouts = 10  # Collect from multiple episodes
for episode in range(num_rollouts):
    obs, info = env.reset()
    done = False
    while not done:
        # Get action from actor1 (the trained safe policy)
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_logits = actor1(obs_tensor)
            action = torch.argmax(action_logits, dim=1).item()
        
        # Record state-action pair (actor1's behavior IS the safe behavior)
        states.append(obs)
        actions.append(action)
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

states = torch.FloatTensor(states)
actions = torch.LongTensor(actions)

state_action_torch_dataset = torch.utils.data.TensorDataset(states, actions)

# %%
### Rashomon Set
interval_trainer = IntervalTrainer(
    model=actor1, # policy which is an instance of nn.Sequential
    min_acc_limit=0.99, # NOTE: should be not greater than accuracy of the model
    seed=2025,
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
ppo_cfg3 = PPOConfig(
    total_timesteps=10_000,
)
actor3, critic3 = ppo_train(
    env=env2,
    cfg=ppo_cfg3,
    actor_warm_start=actor1,
    critic_warm_start=critic1,
    actor_param_bounds_l=param_bounds_l,
    actor_param_bounds_u=param_bounds_u
)

# Visualize the trained safe actor in Env 1
visualize_agent_trajectory(env, actor3, num_episodes=1, max_steps=10, env_name='Env 1')

# Visualize the trained safe actor in Env 2
visualize_agent_trajectory(env2, actor3, num_episodes=1, max_steps=10, env_name='Env 2')
# %%
