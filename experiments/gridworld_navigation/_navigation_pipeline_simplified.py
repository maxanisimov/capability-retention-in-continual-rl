#%%
# TODO: try a bigger blob, maybe agent will be enforce

#%%
# train_two_tasks.py
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
import sys
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning')
from src.trainer import IntervalTrainer
import torch
from _safety_critic_utils import SafetyCritic, collect_traces_dataset
from _ppo_utils import PPOConfig, ppo_train, evaluate
from collections import OrderedDict

# ----- Define Task 1 (big hazard on the shortest path) -----
task1_cfg = PointNavConfig(
    start=(-0.9, -0.9),
    goal=(0.9, 0.9),
    hazard_circles=[(-0.8, -0.8, 0.05)],  # big hazard on the shortest path
    hazard_rects=[],                       # not used here
    terminate_on_hazard=True
)

# ----- Define Task 2 (smaller hazard on the shortest path + flooding on the previous best paths) -----
task2_cfg = PointNavConfig(
    start=(-0.9, -0.9),
    goal=(0.9, 0.9),
    hazard_circles=[(0.8, 0.8, 0.05)],  # big hazard on the shortest path
    hazard_rects=[],                       # not used here
    terminate_on_hazard=True
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

## Rendering the environments
env1 = make_env(task1_cfg)
obs, _ = env1.reset()
plt.imshow(env1.render())
plt.title('Task 1 Grid World')
plt.axis('off')
plt.savefig('figures/laval_gridworld_task1.png', dpi=300, bbox_inches="tight")
plt.show()
env1.close()


env2 = make_env(task2_cfg)
obs, _ = env2.reset()
plt.imshow(env2.render())
plt.title('Task 2 Grid World')
plt.axis('off')
plt.savefig('figures/laval_gridworld_task2.png', dpi=300, bbox_inches="tight")
plt.show()
env2.close()

#%%
# ----- Train PPO Policy A on Task 1 -----
print('Training PPO Policy A on Task 1...')
env1 = PointNavSafetyEnv(task1_cfg)
if isinstance(env1.action_space, gym.spaces.Discrete):
    num_actions = env1.action_space.n
else:
    num_actions = env1.action_space.shape[0]
if isinstance(env1.observation_space, gym.spaces.Discrete):
    num_obs = env1.observation_space.n
else:
    num_obs = env1.observation_space.shape[0]
 
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
trajectory_pixels[:, 0] = (trajectory[:, 0] + 0.9) / 1.8 * img_width   # X: [-0.9, 0.9] -> [0, img_width]
trajectory_pixels[:, 1] = (trajectory[:, 1] + 0.9) / 1.8 * img_height  # Y: [-0.9, 0.9] -> [0, img_height]

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


#%%
# ----- Train PPO Policy B on Task 2 -----
print('Training PPO Policy B on Task 2...')
env2 = PointNavSafetyEnv(task2_cfg)
task2_ppo_cfg = task1_ppo_cfg # copy the same config for Task 2
task2_actor, task2_critic = ppo_train(
    env=env2,
    cfg=task2_ppo_cfg,
    actor_warm_start=task1_actor,  # Use SB3-style architecture
    critic_warm_start=task1_critic  # Use SB3-style architecture
)

# Plot trajectory learned by task2_actor
print("Plotting trajectory for Policy B (Task 2)...")
env2_traj = PointNavSafetyEnv(task2_cfg)
obs, _ = env2_traj.reset()
trajectory = [obs.copy()]
done = False

while not done:
    # Use the same action selection as task1_actor (direct tensor forward pass)
    action = task2_actor(torch.tensor(obs, dtype=torch.float32)).argmax().item()
    obs, reward, terminated, truncated, info = env2_traj.step(action)
    trajectory.append(obs.copy())
    done = terminated or truncated

trajectory = np.array(trajectory)

# Render environment and plot trajectory
fig, ax = plt.subplots(figsize=(8, 8))
img = env2_traj.render()
ax.imshow(img)

# Convert trajectory from world coordinates [-0.9, 0.9] to image pixel coordinates
img_height, img_width = img.shape[:2]

trajectory_pixels = trajectory.copy()
trajectory_pixels[:, 0] = (trajectory[:, 0] + 0.9) / 1.8 * img_width   # X: [-0.9, 0.9] -> [0, img_width]
trajectory_pixels[:, 1] = (trajectory[:, 1] + 0.9) / 1.8 * img_height  # Y: [-0.9, 0.9] -> [0, img_height]

ax.plot(trajectory_pixels[:, 0], trajectory_pixels[:, 1], 'r-', linewidth=2, label='Trajectory')
ax.scatter(trajectory_pixels[0, 0], trajectory_pixels[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
ax.scatter(trajectory_pixels[-1, 0], trajectory_pixels[-1, 1], c='blue', s=100, marker='*', label='End', zorder=5)
ax.set_title('Task 2 - Policy B Trajectory')
ax.legend()
ax.set_xlim(-10, img_width+10)
ax.set_ylim(img_height+10, -10)  # Flip Y-axis to match image coordinates
ax.axis('off')
plt.tight_layout()
# plt.savefig('figures/policy_b_trajectory.png', dpi=300, bbox_inches='tight')
plt.show()
env2_traj.close()

#%%
#################################################################################
# ----- Train PPO Policy C on Task 2 with Rashomon bounds -----
# 1) Train a safety critic
# Collect diverse data from task 1
print("\n=== Step 1: Train Safety Critic on Task 1 ===")
device = 'cpu'
seed = 2025
obs_arr, action_arr, next_obs_arr, failure_arr, done_arr = collect_traces_dataset(
    policy_model=task1_actor, env=env1,
    num_episodes=10_000,
    epsilon=0.5,
    device=device,
    seed=seed
)
print(f'Collected safety dataset. Failure rate: {np.sum(failure_arr) / 100}')

# TODO: find a good safety critic!
print('Safety Critic training...')
# TD training
S_t = torch.from_numpy(obs_arr).to(device)
A_t = torch.from_numpy(action_arr).to(device)
SP_t = torch.from_numpy(next_obs_arr).to(device)
D_t = torch.from_numpy(done_arr).to(device)
F_t = torch.from_numpy(failure_arr).to(device)


safety_critic_in_dim = S_t.shape[1]
if isinstance(env1.action_space, gym.spaces.Discrete):
    safety_critic_in_dim += 1
else:
    safety_critic_in_dim += env1.action_space.shape[0]

# Safety critic hyperaprams:
safety_critic_batch_size = 4096
safety_critic_td_epochs = 100
safety_critic_lr = 1e-3
gamma_safe = 0.99 # 0.0
safety_critic = SafetyCritic(safety_critic_in_dim, hidden_dim=128, dropout=0.2).to(device)
optimizer = torch.optim.Adam(safety_critic.parameters(), lr=safety_critic_lr)
bce_logits = torch.nn.BCEWithLogitsLoss()

safety_dataset_length = S_t.shape[0]
for epoch in range(safety_critic_td_epochs):
    idx = np.random.permutation(safety_dataset_length)
    total_loss = 0.0
    for i in range(0, safety_dataset_length, safety_critic_batch_size):
        j = idx[i:i + safety_critic_batch_size]
        s_b = S_t[j]
        a_b = A_t[j].unsqueeze(-1)
        state_action_b = torch.cat([s_b, a_b], dim=1)
        d_b = D_t[j]
        f_b = F_t[j]

        sp_b = SP_t[j]
        # Apply the policy to all next states
        if isinstance(env1.action_space, gym.spaces.Discrete):
            with torch.no_grad():
                ### For SB3 model
                if isinstance(task1_actor, ActorCriticPolicy):
                    ap_b_np, _ = task1_actor.predict(sp_b.cpu().numpy(), deterministic=True)
                else:  ### For my PPO from scratch
                    cur_sp_tensor = torch.tensor(sp_b, dtype=torch.float32).unsqueeze(0).to(device)
                    logits = task1_actor.forward(cur_sp_tensor)
                    ap_b_np = logits.argmax(dim=-1).cpu().numpy()
            ap_b = torch.from_numpy(ap_b_np).to(device).unsqueeze(-1)
        else:
            with torch.no_grad():
                ### For SB3 model
                if isinstance(task1_actor, ActorCriticPolicy):
                    ap_b_np, _ = task1_actor.predict(sp_b.cpu().numpy(), deterministic=True)
                else:  ### For my PPO from scratch
                    cur_sp_tensor = torch.tensor(sp_b, dtype=torch.float32).unsqueeze(0).to(device)
                    logits = task1_actor.forward(cur_sp_tensor)
                    ap_b_np = logits.argmax(dim=-1).cpu().numpy()
            ap_b = torch.from_numpy(ap_b_np).to(device)
        if len(ap_b.shape) > 2:
            ap_b = ap_b.reshape(-1, 1)
        next_state_action_b = torch.cat([sp_b, ap_b], dim=1)

        with torch.no_grad():
            next_logits = safety_critic(next_state_action_b)
            next_p = torch.sigmoid(next_logits)
            target = f_b + (1.0 - d_b) * gamma_safe * next_p
            target = target.clamp(0.0, 1.0)

        logits = safety_critic(state_action_b)
        loss = bce_logits(logits, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(safety_critic.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * s_b.size(0)

    print(f"[TD {epoch+1}/{safety_critic_td_epochs}] BCE(TD) loss={total_loss / safety_dataset_length:.6f}")

safety_critic.eval()

# Check the accuracy of the safety critic on the training data
with torch.no_grad():
    if isinstance(env1.action_space, gym.spaces.Discrete):
        action_tensor = torch.from_numpy(action_arr).to(device).unsqueeze(-1)
    else:
        action_tensor = torch.from_numpy(action_arr).to(device)
    state_action_tensor = torch.cat([S_t, action_tensor], dim=1)
    logits = safety_critic(state_action_tensor)
    preds = (torch.sigmoid(logits) >= 0.5).float() # whether next state is a failure state
    accuracy = (preds.squeeze() == F_t).float().mean().item()
    print(f"Safety Critic training accuracy on collected data: {accuracy:.4f}")

### a) Collect a large exploratory dataset using the trained policy with epsilon-greedy exploration
episodes_exp = 1500   # make sure you have enough data
epsilon_exp = 0.5    # and make sure the data is diverse
S_exp, A_exp, SP_exp, F_exp, D_exp = collect_traces_dataset(
    policy_model=task1_actor, env=env1, num_episodes=episodes_exp, epsilon=epsilon_exp
)
# TODO: plot traces used in safety critic training; we want to have a good coverage of the state-action space!
# TODO: provide the policy safe for Task 1 and 2 explicitly to the Rashomon set computation (in S_exp, A_exp, SP_exp, F_exp, D_exp)
# TODO: simplify the grid world and run the script

N = S_exp.shape[0]
avg_len = N / episodes_exp
fail_terms = int(F_exp.sum())

print(f"Exploration dataset collected:")
print(f"- episodes={episodes_exp}, epsilon={epsilon_exp}")
print(f"- transitions={N}, mean_ep_len≈{avg_len:.1f}, fail terminals={fail_terms}")

# Optional: estimate number of unique observations (rounded to reduce near-duplicates)
rounded_S = np.round(S_exp, 3)
unique_obs = np.unique(rounded_S, axis=0)
print(f"- approx unique observations (rounded 3 d.p.)={unique_obs.shape[0]}")

#%%
### Plot trajectories in S_exp used in safety critic training
fig, ax = plt.subplots(figsize=(10, 10))
num_episodes_to_plot = 100

# Group trajectories by episode
episode_starts = np.where(D_exp[:-1] == 1)[0] + 1
episode_starts = np.concatenate([[0], episode_starts])

# Plot each episode trajectory
colors = cm.viridis(np.linspace(0, 1, len(episode_starts)))  # Use the imported cm module
for ep_idx, start_idx in enumerate(episode_starts[:min(num_episodes_to_plot, len(episode_starts))]):  # Limit to first 50 episodes for clarity
    end_idx = episode_starts[ep_idx + 1] if ep_idx + 1 < len(episode_starts) else len(S_exp)
    original_traj = S_exp[start_idx:end_idx]
    # adjust trajectory so that starting in (-1, -1) means starting from top left corner (1, -1)
    traj = original_traj.copy()
    traj[:, 0] = (traj[:, 0] + 0.9) / 1.8 * img_width   # X: [-0.9, 0.9] -> [0, img_width]
    traj[:, 1] = (traj[:, 1] + 0.9) / 1.8 * img_height  # Y: [-0.9, 0.9] -> [0, img_height]
    ax.plot(traj[:, 0], traj[:, 1], alpha=0.5, linewidth=1, color=colors[ep_idx])
    ax.scatter(traj[0, 0], traj[0, 1], c='green', s=30, marker='o', zorder=5)  # Start
    ax.scatter(traj[-1, 0], traj[-1, 1], c='blue', s=30, marker='D', zorder=5)  # End
    # mark a failure point 
    ax.scatter(
        traj[F_exp[start_idx:end_idx].astype(bool), 0], 
        traj[F_exp[start_idx:end_idx].astype(bool), 1], 
        c='red', s=200, marker='x', zorder=5,
    )

ax.set_xlim(-25, img_width+25)
ax.set_ylim(img_height+25, -25)  # Flip Y-axis to match image coordinates
ax.set_xlabel('Position X')
ax.set_ylabel('Position Y')
ax.set_title(f'Trajectories in Exploration Dataset (first {num_episodes_to_plot} episodes)')
ax.grid(True, alpha=0.3)
ax.axis('off')
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
plt.tight_layout()
plt.show()

#%%
# Plot traces used in safety critic training for state-action space coverage
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

state_trajectories = S_exp.copy()
state_trajectories[:, 0] = (state_trajectories[:, 0] + 0.9) / 1.8 * img_width
state_trajectories[:, 1] = (state_trajectories[:, 1] + 0.9) / 1.8 * img_height

# Plot 1: State space coverage
axes[0].scatter(state_trajectories[:, 0], state_trajectories[:, 1], alpha=0.3, s=10, c='blue', label='Explored states')
# axes[0].scatter(rounded_S[:, 0], rounded_S[:, 1], alpha=0.6, s=20, c='red', label='Unique states')
axes[0].set_xlabel('Position X')
axes[0].set_ylabel('Position Y')
axes[0].set_title('State Space Coverage (Exploration Dataset)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Failure distribution
failure_states = S_exp[F_exp.astype(bool)]
safe_states = S_exp[~F_exp.astype(bool)]
axes[1].scatter(safe_states[:, 0], safe_states[:, 1], alpha=0.3, s=10, c='green', label='Safe transitions')
axes[1].scatter(failure_states[:, 0], failure_states[:, 1], alpha=0.6, s=20, c='red', label='Failure transitions')
axes[1].set_xlabel('Position X')
axes[1].set_ylabel('Position Y')
axes[1].set_title('Failure Distribution (Exploration Dataset)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('figures/exploration_dataset_coverage.png', dpi=300, bbox_inches='tight')
plt.show()

#%%%
### b) Use the safety critic to label safe actions in the exploratory dataset
safety_critic.eval()
N = S_exp.shape[0]
batch_size_eval = 32768  # adjust if needed
max_failure_prob = 0.01 # 0.5 # NOTE: if the failure probability is at this value or below this threshold, the action is considered safe

safe_action_mask = np.zeros((N, num_actions), dtype=bool)
safe_actions_per_state = []
with torch.no_grad():
    for i in range(0, N, batch_size_eval):
        j = min(i + batch_size_eval, N)
        s_batch = torch.from_numpy(S_exp[i:j]).to(device=device, dtype=torch.float32)

        # Build (state, action) pairs for all discrete actions
        action_vals = torch.arange(num_actions, device=device, dtype=s_batch.dtype)  # [0..num_actions-1]
        s_expanded = s_batch.unsqueeze(1).expand(-1, num_actions, -1)                # (B, A, num_obs)
        a_expanded = action_vals.view(1, num_actions, 1).expand(s_batch.size(0), num_actions, 1)  # (B, A, 1)
        sa_flat = torch.cat([s_expanded, a_expanded], dim=2).reshape(-1, num_obs + 1)        # (B*A, num_obs+1)

        logits = safety_critic(sa_flat)                         # (B*A,)
        probs = torch.sigmoid(logits).reshape(-1, num_actions)    # (B, A)
        mask = (probs <= max_failure_prob).cpu().numpy()                # (B, A) bool

        safe_action_mask[i:j] = mask
        safe_actions_per_state.extend([np.where(row)[0].tolist() for row in mask])

print(f"Computed safe actions for {N} states (threshold={max_failure_prob}, num_actions={num_actions}).")
print('Distribution of number of safe actions per state:')
print(pd.Series([len(lst) for lst in safe_actions_per_state]).value_counts().sort_index())

### c) Prepare safe action matrix where -1 is used for padding
final_safe_actions_per_state = []
for safe_action_indices in safe_actions_per_state:
    safe_action_indices = torch.tensor(safe_action_indices, dtype=torch.long)
    pad_len = max(0, num_actions - safe_action_indices.numel())
    if pad_len > 0:
        pad = torch.full((pad_len,), -1, dtype=safe_action_indices.dtype, device=safe_action_indices.device)
        cur_tensor = torch.cat([safe_action_indices, pad], dim=0)
    else:
        cur_tensor = safe_action_indices
    final_safe_actions_per_state.append(cur_tensor.numpy())

safe_actions_tensor = torch.from_numpy(np.array(final_safe_actions_per_state))
states_tensor = torch.from_numpy(S_exp)
state_action_torch_dataset = TensorDataset(states_tensor, safe_actions_tensor)
state_action_loader = DataLoader(state_action_torch_dataset, batch_size=8, shuffle=False)

#%%
### d) Compute Rashomon set using IntervalTrainer
interval_trainer = IntervalTrainer(
    model=task1_actor, # policy which is an instance of nn.Sequential
    min_acc_limit=0.99, # NOTE: tweak only if task 2 adaptation is poor
    seed=2025,
    # n_iters=10_000, # default 2000; running longer may not translate into higher OOS accuracy
)
interval_trainer.compute_rashomon_set(
    dataset=state_action_torch_dataset, # states and safe actions
    multi_label=True # NOTE
)
# Extract parameter bounds from the bounded model
assert len(interval_trainer.bounds) == 1, "Expected exactly one bounded model"
bounded_model = interval_trainer.bounds[0]
param_bounds_l = [bound.detach().cpu() for bound in bounded_model.param_l]
param_bounds_u = [bound.detach().cpu() for bound in bounded_model.param_u]

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

safe_actor, safe_critic = ppo_train(
    env=env2,
    cfg=task2_ppo_cfg,
    actor_warm_start=task1_actor,
    critic_warm_start=task1_critic,
    actor_param_bounds_l=param_bounds_l,
    actor_param_bounds_u=param_bounds_u,
)
safe_eval_avg_r_task1, safe_eval_std_r_task1, safe_eval_fail_rate_task1 = evaluate(
    actor=safe_actor, 
    env=env1, episodes=1, render_mode='rgb_array'
)
safe_eval_avg_r_task2, safe_eval_std_r_task2, safe_eval_fail_rate_task2 = evaluate(
    actor=safe_actor, 
    env=env2, episodes=1, render_mode='rgb_array'
)

# Plot trajectory learned by safe_actor (Policy C)
print("Plotting trajectory for Policy C (Task 2 with Rashomon bounds)...")
env2_traj = PointNavSafetyEnv(task2_cfg)
obs, _ = env2_traj.reset()
trajectory = [obs.copy()]
done = False

while not done:
    # Use the same action selection as safe_actor (direct tensor forward pass)
    action = safe_actor(torch.tensor(obs, dtype=torch.float32)).argmax().item()
    obs, reward, terminated, truncated, info = env2_traj.step(action)
    trajectory.append(obs.copy())
    done = terminated or truncated

trajectory = np.array(trajectory)

# Render environment and plot trajectory
fig, ax = plt.subplots(figsize=(8, 8))
img = env2_traj.render()
ax.imshow(img)

# Convert trajectory from world coordinates [-0.9, 0.9] to image pixel coordinates
img_height, img_width = img.shape[:2]

trajectory_pixels = trajectory.copy()
trajectory_pixels[:, 0] = (trajectory[:, 0] + 0.9) / 1.8 * img_width   # X: [-0.9, 0.9] -> [0, img_width]
trajectory_pixels[:, 1] = (trajectory[:, 1] + 0.9) / 1.8 * img_height  # Y: [-0.9, 0.9] -> [0, img_height]

ax.plot(trajectory_pixels[:, 0], trajectory_pixels[:, 1], 'r-', linewidth=2, label='Trajectory')
ax.scatter(trajectory_pixels[0, 0], trajectory_pixels[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
ax.scatter(trajectory_pixels[-1, 0], trajectory_pixels[-1, 1], c='blue', s=100, marker='*', label='End', zorder=5)
ax.set_title('Task 2 - Policy C Trajectory (Rashomon Bounds)')
ax.legend()
ax.set_xlim(0, img_width)
ax.set_ylim(img_height, 0)  # Flip Y-axis to match image coordinates
ax.axis('off')
plt.tight_layout()
plt.savefig('figures/policy_c_trajectory.png', dpi=300, bbox_inches='tight')
plt.show()
env2_traj.close()


#%%
###### FINAL EVALUATION ####################################
policies = {'Policy A': task1_actor, 'Policy B': task2_actor, 'Policy C': safe_actor}  # Policy C is same as A for demo
task_configs = {'Task 1': task1_cfg, 'Task 2': task2_cfg}
for policy_name, actor in policies.items():
    for task_name, cfg in task_configs.items():
        env = PointNavSafetyEnv(cfg)
        avg_total_reward, std_total_reward, failure_rate = evaluate(actor=actor, env=env, episodes=10)
        final_metrics[policy_name][task_name]['avg_total_reward'] = avg_total_reward
        # final_metrics[policy_name][task_name]['std_total_reward'] = std_total_reward
        final_metrics[policy_name][task_name]['failure_rate'] = failure_rate
        env.close()

print("\nFinal Evaluation Metrics:")
for policy_name, tasks in final_metrics.items():
    print(f"\n{policy_name}:")
    for task_name, metrics in tasks.items():
        print(f"  {task_name} -> Avg Reward: {round(metrics['avg_total_reward'], 2)}, Failure Rate: {round(metrics['failure_rate'], 2)}")

# # Original policy eval
# env1_eval = make_env(task1_cfg)
# avg_total_reward, failure_rate = evaluate(model1, env1_eval, n_episodes=1, render_mode='rgb_array')
# print(f"Original policy in Task-1 env -> avg reward: {avg_total_reward}, failure rate: {failure_rate}")
# env1_eval.close()

# # New policy eval on Task 2
# env2_eval = make_env(task2_cfg)
# avg_total_reward, failure_rate = evaluate(model2, env2_eval, n_episodes=1, render_mode='rgb_array')
# print(f"Task-2 policy in Task-2 env -> avg reward: {avg_total_reward}, failure rate: {failure_rate}")
# env2_eval.close()

# # New policy eval on Task 1
# env1_eval = make_env(task1_cfg)
# avg_total_reward, failure_rate = evaluate(model2, env1_eval, n_episodes=1, render_mode='rgb_array')
# print(f"Task-2 policy in Task-1 env -> avg reward: {avg_total_reward}, failure rate: {failure_rate}")
# env1_eval.close()

# %%
