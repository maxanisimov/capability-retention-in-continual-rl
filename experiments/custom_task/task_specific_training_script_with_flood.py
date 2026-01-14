"""
The aim of this script is to create a custom task for an experiment.
This task should be constructed so that its reward reflects both performance and safety,
i.e. the agent is rewarded for completing the task quickly but penalized for unsafe actions.

I want to create two versions of this task such that if I comlete a task 1 safely,
I cannot complete task 2 safely, and vice versa.

In this script, I want to use my custom PPO implementation because it is important for me 
that actor is an instance of nn.Sequential. 

This task should be as simple as possible.
"""
#%%
import sys
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning')
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning/experiments')
import pandas as pd
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
from _ppo_utils import PPOConfig, ppo_train, make_actor_critic
from src.trainer import IntervalTrainer
from safety_flooded_grid_utils import *
from safety_flooded_grid_utils import _calc_obs

#%%
# Create environments for Task 1 and Task 2
task1_env = make_safe_grid_env(task_version=1, max_steps=30)
task2_env = make_safe_grid_env(task_version=2, max_steps=30)

# Visualize both tasks
print("Task 1 Layout:")
task1_env.reset()
task1_env.render()

print("\nTask 2 Layout:")
task2_env.reset()
task2_env.render()

#%%
# =============================================================================
# Training Configuration
# =============================================================================
# Note: For this simple environment, fewer timesteps are needed.
# The agent just needs to learn to go right while avoiding the unsafe zone.

ppo_cfg = PPOConfig( 
    total_timesteps=50_000,   # Reduced - simple task converges quickly
    eval_episodes=100,
    rollout_steps=2048,
    update_epochs=10,
    minibatch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.05,           # Higher entropy for exploration
    vf_coef=0.5,
    lr=3e-4,
    max_grad_norm=0.5,
    seed=42,
    device='cpu'
)

#%%
# =============================================================================
# Train Policy on Task 1
# =============================================================================
print("=" * 60)
print("Training on Task 1")
print("Optimal strategy: go right, pass through Zone A safely")
print("=" * 60)

task1_actor, task1_critic = ppo_train(env=task1_env, cfg=ppo_cfg)

#%%
# =============================================================================
# Evaluate Task 1 Policy on Both Tasks
# =============================================================================

print("\n" + "=" * 60)
print("Evaluating Task 1 policy on Task 1 (should be safe)")
print("=" * 60)
task1_on_task1 = evaluate_with_safety(task1_env, task1_actor)
print(f"Mean Reward: {task1_on_task1['mean_reward']:.2f} ± {task1_on_task1['std_reward']:.2f}")
print(f"Mean Safety Violations: {task1_on_task1['mean_violations']:.2f}")
print(f"Success Rate: {task1_on_task1['success_rate']:.2%}")

print("\n" + "=" * 60)
print("Evaluating Task 1 policy on Task 2 (should be UNSAFE)")
print("=" * 60)
task1_on_task2 = evaluate_with_safety(task2_env, task1_actor)
print(f"Mean Reward: {task1_on_task2['mean_reward']:.2f} ± {task1_on_task2['std_reward']:.2f}")
print(f"Mean Safety Violations: {task1_on_task2['mean_violations']:.2f}")
print(f"Success Rate: {task1_on_task2['success_rate']:.2%}")

#%%
# =============================================================================
# Visualize Task 1 Policy Trajectories
# =============================================================================
print("\n" + "=" * 60)
print("Visualizing Task 1 Policy Trajectories")
print("=" * 60)

plot_policy_trajectories(
    task1_env, task1_actor, n_episodes=1, 
    title="Policy A \n Task 1: No Flooding"
)
plot_policy_trajectories(
    task2_env, task1_actor, n_episodes=1, 
    title="Policy B \n Task 2: Flooded Areas"
)

# #%%
# # =============================================================================
# # Train Policy on Task 2 (for comparison)
# # =============================================================================
# print("\n" + "=" * 60)
# print("Training on Task 2")
# print("Optimal strategy: go right, pass through Zone B safely")
# print("=" * 60)

# task2_actor, task2_critic = ppo_train(env=task2_env, cfg=ppo_cfg)

# print("\n" + "=" * 60)
# print("Evaluating Task 2 policy on Task 2 (should be safe)")
# print("=" * 60)
# task2_on_task2 = evaluate_with_safety(task2_env, task2_actor)
# print(f"Mean Reward: {task2_on_task2['mean_reward']:.2f} ± {task2_on_task2['std_reward']:.2f}")
# print(f"Mean Safety Violations: {task2_on_task2['mean_violations']:.2f}")
# print(f"Success Rate: {task2_on_task2['success_rate']:.2%}")

# print("\n" + "=" * 60)
# print("Evaluating Task 2 policy on Task 1 (should be UNSAFE)")
# print("=" * 60)
# task2_on_task1 = evaluate_with_safety(task1_env, task2_actor)
# print(f"Mean Reward: {task2_on_task1['mean_reward']:.2f} ± {task2_on_task1['std_reward']:.2f}")
# print(f"Mean Safety Violations: {task2_on_task1['mean_violations']:.2f}")
# print(f"Success Rate: {task2_on_task1['success_rate']:.2%}")

# #%%
# # =============================================================================
# # Visualize Task 2 Policy Trajectories
# # =============================================================================
# print("\n" + "=" * 60)
# print("Visualizing Task 2 Policy Trajectories")
# print("=" * 60)

# plot_policy_trajectories(
#     task2_env, task2_actor, n_episodes=1, 
#     title="Policy B \n Task 2: Flooding"
# )
# # plot_policy_trajectories(task1_env, task2_actor, n_episodes=1, 
# #                          title="Policy B \n Task 2: Safe | Task 1: Unsafe")

# #%%
# # =============================================================================
# # Summary
# # =============================================================================
# print("\n" + "=" * 60)
# print("SUMMARY: Safety Conflict Demonstration")
# print("=" * 60)
# print(f"Task 1 policy on Task 1: {task1_on_task1['mean_violations']:.1f} violations (safe)")
# print(f"Task 1 policy on Task 2: {task1_on_task2['mean_violations']:.1f} violations (unsafe)")
# print(f"Task 2 policy on Task 2: {task2_on_task2['mean_violations']:.1f} violations (safe)")
# print(f"Task 2 policy on Task 1: {task2_on_task1['mean_violations']:.1f} violations (unsafe)")
# print("\n→ This demonstrates that safe completion of one task")
# print("  implies unsafe behavior on the other task.")

#%%
### Store policies and value functions for later use
# save_to = '/Users/ma5923/Documents/_projects/CertifiedContinualLearning/experiments/custom_task/neural_nets'
# torch.save(task1_actor.state_dict(), f"{save_to}/task1_actor.pth")
# torch.save(task1_critic.state_dict(), f"{save_to}/task1_critic.pth")
# torch.save(task2_actor.state_dict(), f"{save_to}/task2_actor.pth")
# torch.save(task2_critic.state_dict(), f"{save_to}/task2_critic.pth")

#################################################################################

#%%
### SAFE RL POLICY UPDATE
# The idea is to update the policy on Task 2 using a safe RL method
# such that it retains safety on Task 1 while improving performance on Task 2.
# Here, we compute the Rashomon set based on safety constraints from Task 1
# and then update the policy to perform well in the Task 2.

# ### Hardcoded safe trajectories: provide only the omni-safe trajectories to create the safe state-action dataset
# # Create a trajectory that goes from the start to the bottom row, then goes all the way right, then goes up to the goal.
# states = torch.tensor([
#     _calc_obs(task1_env.start_pos, n_rows=task1_env.n_rows, n_cols=task1_env.n_cols),
#     _calc_obs((2, 0), n_rows=task1_env.n_rows, n_cols=task1_env.n_cols),
#     _calc_obs((3, 0), n_rows=task1_env.n_rows, n_cols=task1_env.n_cols),
#     _calc_obs((3, 1), n_rows=task1_env.n_rows, n_cols=task1_env.n_cols),
#     _calc_obs((3, 2), n_rows=task1_env.n_rows, n_cols=task1_env.n_cols),
#     _calc_obs((3, 3), n_rows=task1_env.n_rows, n_cols=task1_env.n_cols),
#     _calc_obs((3, 4), n_rows=task1_env.n_rows, n_cols=task1_env.n_cols),
#     _calc_obs((2, 4), n_rows=task1_env.n_rows, n_cols=task1_env.n_cols),
#     # _calc_obs(task1_env.goal_pos, n_rows=task1_env.n_rows, n_cols=task1_env.n_cols),
# ])
# states = torch.cat([states] * 10_000, dim=0) # repeat to have enough data
# actions = torch.tensor([
#     1, # down
#     1, # down
#     3, # right
#     3, # right
#     3, # right
#     3, # right
#     0, # up 
#     0, # up
# ])
# actions = torch.cat([actions] * 10_000, dim=0) # repeat to have enough data

### Create dataset of safe state-action pairs from Task 1 with exploration by storing only safe (s,a) pairs
exploration_rate = 0.9 # NOTE: 0.5 is not enough to have example of omni-safe path with 200 episodes
state_action_data = []
n_safe_episodes = 10_000
for _ in range(n_safe_episodes):
    obs, _ = task1_env.reset()
    done = False
    
    while not done:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = task1_actor(obs_tensor)
            if np.random.rand() < exploration_rate:
                action = task1_env.action_space.sample()
            else:
                action = torch.argmax(logits, dim=-1).item()
        
        next_obs, reward, terminated, truncated, info = task1_env.step(action)
        
        # Only store state-action pairs if no safety violation occurred
        if info.get("cumulative_safety_violations", 0) == 0:
            state_action_data.append((obs, action))
        
        obs = next_obs
        done = terminated or truncated
# Convert to tensors
states = torch.tensor([s for s, a in state_action_data], dtype=torch.float32)
actions = torch.tensor([a for s, a in state_action_data], dtype=torch.long)

state_action_torch_dataset = torch.utils.data.TensorDataset(states, actions)

#%%
# Analyze safe dataset
state_action_df = pd.DataFrame({
    'state_row': states[:, 0].numpy(),
    'state_col': states[:, 1].numpy(),
    # 'zone_a_safe': states[:, 2].numpy(),
    'action': actions.numpy()
})

### Plot the distribution of states in the safe dataset
print(f"\nSafe Dataset Statistics:")
print(f"Total safe state-action pairs: {len(state_action_df)}")
print(f"\nAction distribution:")
print(state_action_df['action'].value_counts().sort_index())

# Create visualization of safe states on the grid
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Heatmap of state visitation frequency
ax1 = axes[0]
state_counts = state_action_df.groupby(['state_row', 'state_col']).size().reset_index(name='count')

# Create a grid for the heatmap
n_rows, n_cols = task1_env.n_rows, task1_env.n_cols
heatmap = np.zeros((n_rows, n_cols))
for _, row in state_counts.iterrows():
    r_idx = int(row['state_row'] * (n_rows - 1))
    c_idx = int(row['state_col'] * (n_cols - 1))
    heatmap[r_idx, c_idx] = row['count']

# Plot heatmap
im1 = ax1.imshow(heatmap, cmap='YlOrRd', aspect='equal')
ax1.set_title('Safe State Visitation Frequency\n(Task 1 Training)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Column')
ax1.set_ylabel('Row')
ax1.set_xticks(range(n_cols))
ax1.set_yticks(range(n_rows))

# Add grid
for i in range(n_rows + 1):
    ax1.axhline(i - 0.5, color='gray', linewidth=0.5)
for j in range(n_cols + 1):
    ax1.axvline(j - 0.5, color='gray', linewidth=0.5)

# Annotate cells with counts
for i in range(n_rows):
    for j in range(n_cols):
        text = ax1.text(j, i, int(heatmap[i, j]),
                       ha="center", va="center", color="black", fontsize=10)

# Mark special cells
start_r, start_c = task1_env.start_pos
goal_r, goal_c = task1_env.goal_pos
ax1.plot(start_c, start_r, 'bs', markersize=15, markeredgecolor='white', markeredgewidth=2)
ax1.plot(goal_c, goal_r, 'g*', markersize=20, markeredgecolor='white', markeredgewidth=2)

plt.colorbar(im1, ax=ax1, label='Visit Count')

# Plot 2: Action distribution per state
ax2 = axes[1]
action_map = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
state_action_df['action_name'] = state_action_df['action'].map(action_map)

# Get top visited states
top_states = state_action_df.groupby(['state_row', 'state_col']).size().nlargest(10).reset_index(name='count')

action_dist_data = []
for _, state in top_states.iterrows():
    mask = (state_action_df['state_row'] == state['state_row']) & \
           (state_action_df['state_col'] == state['state_col'])
    actions_at_state = state_action_df[mask]['action_name'].value_counts()
    
    r_idx = int(state['state_row'] * (n_rows - 1))
    c_idx = int(state['state_col'] * (n_cols - 1))
    state_label = f"({r_idx},{c_idx})"
    
    for action_name in ['Up', 'Down', 'Left', 'Right']:
        action_dist_data.append({
            'State': state_label,
            'Action': action_name,
            'Count': actions_at_state.get(action_name, 0)
        })

action_dist_df = pd.DataFrame(action_dist_data)
action_pivot = action_dist_df.pivot(index='State', columns='Action', values='Count').fillna(0)

# Reorder columns
action_order = ['Up', 'Down', 'Left', 'Right']
action_pivot = action_pivot[[col for col in action_order if col in action_pivot.columns]]

action_pivot.plot(kind='bar', stacked=True, ax=ax2, 
                 color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax2.set_title('Action Distribution at Top 10 Visited States', fontsize=14, fontweight='bold')
ax2.set_xlabel('State (row, col)')
ax2.set_ylabel('Action Count')
ax2.legend(title='Action', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print(f"\nDataset ready for Rashomon set computation with {len(state_action_df)} safe examples")

#%%
### 1. Compute the Rashomon set based on Task 1 safe actions
interval_trainer = IntervalTrainer(
    model=task1_actor, # policy which is safe for Task 1
    min_acc_limit=0.99, # NOTE: cannot be > than the accuracy of task1_actor
    seed=2025,
    # n_iters=10_000, # default 2000; running longer may not translate into higher OOS accuracy
)
interval_trainer.compute_rashomon_set(
    dataset=state_action_torch_dataset, # states and safe actions from Task 1
    multi_label=True # NOTE
)
# Extract parameter bounds from the bounded model
assert len(interval_trainer.bounds) == 1, "Expected exactly one bounded model"
certificate = interval_trainer.certificates[0]
print(f"\nRashomon set computed. Certified accuracy on safe action dataset: {certificate:.2f}")
bounded_model = interval_trainer.bounds[0]
param_bounds_l = [bound.detach().cpu() for bound in bounded_model.param_l]
param_bounds_u = [bound.detach().cpu() for bound in bounded_model.param_u]

#%%
### Update the task1 actor and critic using PGD within the Rashomon set
safe_actor, safe_critic = ppo_train(
    env=task2_env,
    cfg=ppo_cfg,
    actor_warm_start=copy.deepcopy(task1_actor),
    critic_warm_start=copy.deepcopy(task1_critic),
    actor_param_bounds_l=param_bounds_l,
    actor_param_bounds_u=param_bounds_u,
)

### Check that the weights of safe_actor are within the Rashomon bounds
with torch.no_grad():
    for i, (param, p_l, p_u) in enumerate(zip(safe_actor.parameters(), param_bounds_l, param_bounds_u)):
        assert torch.all(param.cpu() >= p_l), f"Parameter {i} below lower bound"
        assert torch.all(param.cpu() <= p_u), f"Parameter {i} above upper bound"
print("\nSafe policy parameters are within Rashomon bounds.")

### Plot the trajectories of the safe updated policy on both tasks
print("\n" + "=" * 60)
print("Visualizing Safe Updated Policy Trajectories")
print("=" * 60)
plot_policy_trajectories(task1_env, safe_actor, n_episodes=1, 
                         title="Omni-safe Policy C (?)")
# plot_policy_trajectories(task2_env, safe_actor, n_episodes=1, 
#                          title="Safe Updated Policy on Task 2 Environment")

# %%
# Visualize parameter bounds
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (p_l, p_u) in enumerate(zip(param_bounds_l, param_bounds_u)):
    if i >= 4:  # Only plot first 4 parameter tensors
        break
    
    ax = axes[i]
    
    # Flatten the parameter tensors for visualization
    p_l_flat = p_l.flatten().numpy()
    p_u_flat = p_u.flatten().numpy()
    width = (p_u_flat - p_l_flat)
    center = (p_l_flat + p_u_flat) / 2
    
    # Create a scatter plot showing bounds
    indices = np.arange(len(p_l_flat))
    ax.fill_between(indices, p_l_flat, p_u_flat, alpha=0.3, label='Bound Range')
    ax.plot(indices, center, 'r-', linewidth=1, label='Center', alpha=0.7)
    ax.plot(indices, p_l_flat, 'b--', linewidth=0.5, label='Lower Bound', alpha=0.5)
    ax.plot(indices, p_u_flat, 'g--', linewidth=0.5, label='Upper Bound', alpha=0.5)
    
    ax.set_title(f'Parameter Tensor {i}\nShape: {p_l.shape}, Mean Width: {width.mean():.4f}', 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Parameter Index')
    ax.set_ylabel('Parameter Value')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for i in range(len(param_bounds_l), 4):
    axes[i].axis('off')

plt.tight_layout()
plt.suptitle('Parameter Bounds Visualization (Rashomon Set)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.show()

# Additional histogram visualization of bound widths
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
all_widths = []
for p_l, p_u in zip(param_bounds_l, param_bounds_u):
    width = (p_u - p_l).flatten().numpy()
    all_widths.extend(width)

ax.hist(all_widths, bins=50, alpha=0.7, edgecolor='black')
ax.set_xlabel('Bound Width')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Parameter Bound Widths', fontsize=14, fontweight='bold')
ax.axvline(np.mean(all_widths), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_widths):.4f}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nParameter bounds information:")
print(f"Number of parameter tensors: {len(param_bounds_l)}")

total_params = 0
for i, (p_l, p_u) in enumerate(zip(param_bounds_l, param_bounds_u)):
    width = (p_u - p_l).abs().mean().item()
    total_params += p_l.numel()
    print(f"  Parameter {i}: shape={p_l.shape}, avg_width={width:.6f}, params={p_l.numel()}")
print(f"Total parameters: {total_params}")

# Certificate information
assert len(interval_trainer.certificates) == 1, "Expected exactly one certificate"
certificate = interval_trainer.certificates[0]
print(f"Certified accuracy on the safe action dataset: {certificate:.2f}")

#%%