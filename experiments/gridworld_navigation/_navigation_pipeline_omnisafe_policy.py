#%%
# TODO list
# DONE: plot traces used in safety critic training; we want to have a good coverage of the state-action space!
# DONE: provide the policy safe for Task 1 and 2 explicitly to the Rashomon set computation (in S_exp, A_exp, SP_exp, F_exp, D_exp)
# TODO: use a model policy which is not safe for Task 1 so that we see
# the effect of Rashomon set computation
# best policy for Task 1 should not be the same as for Task 2
# TODO: start with simpler env setup and then make the env more complex
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
import torch.nn as nn
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
        # (-0.4, -0.8, 0.1),
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
        # (0.8, 0.4, 0.1),
        # (0.9, 0.5, 0.1),
        # Flooding #2 hazards on previous best paths
        # (-0.9, -0.5, 0.1),
        # (-0.8, -0.4, 0.1),
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
        # (0.4, 0.8, 0.1),
        # (0.5, 0.9, 0.1),
    ],
    terminate_on_hazard=True, # terminate on hazard to encourage safe paths
    progress_coef=100.0  # bigger coeff to encourage finding the shortest path
)

# Create task 3 which combines the hazards from task 1 and task 2
# ----- Define Task 3 (both big hazard and flooding hazards) -----
task3_cfg = PointNavConfig(
    start=(-0.9, -0.9),
    goal=(0.9, 0.9),
    hazard_circles=[
        # Big hazard from Task 1
        (0.0, 0.0, 0.35),
        # Flooding #1 hazards from Task 2
        # (-0.5, -0.9, 0.1),
        # (-0.4, -0.8, 0.1),
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
        # (0.8, 0.4, 0.1),
        # (0.9, 0.5, 0.1),
        # Flooding #2 hazards from Task 2
        # (-0.9, -0.5, 0.1),
        # (-0.8, -0.4, 0.1),
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
        # (0.4, 0.8, 0.1),
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

# ## Rendering the environments
# env1 = make_env(task1_cfg)
# obs, _ = env1.reset()
# plt.imshow(env1.render())
# plt.title('Task 1 Grid World')
# plt.axis('off')
# # plt.savefig('figures/laval_gridworld_task1.png', dpi=300, bbox_inches="tight")
# plt.show()
# env1.close()


# env2 = make_env(task2_cfg)
# obs, _ = env2.reset()
# plt.imshow(env2.render())
# plt.title('Task 2 Grid World')
# plt.axis('off')
# # plt.savefig('figures/laval_gridworld_task2.png', dpi=300, bbox_inches="tight")
# plt.show()
# env2.close()

#%%
### FUNCTIONS
def render_policy_trajectory(env, trajectory_x, trajectory_y, title='Policy Trajectory'):
    # Get te rendered image
    img = env.render()
    img_height, img_width = img.shape[:2]

    def env_to_pixel(x, y, img_width, img_height):
        """Convert environment coords [-1, 1] to pixel coords [0, img_size]"""
        # Map [-1, 1] to [0, img_size]
        px = (x + 1.0) / 2.0 * img_width
        py = (y + 1.0) / 2.0 * img_height  # No flip - match the render() convention
        return px, py
    
    # Transform all trajectory points
    trajectory_px = []
    trajectory_py = []
    for x, y in zip(trajectory_x, trajectory_y):
        px, py = env_to_pixel(x, y, img_width, img_height)
        trajectory_px.append(px)
        trajectory_py.append(py)

    # Plot with transformed coordinates
    plt.figure(figsize=(8, 8))
    plt.imshow(img, origin='lower')
    plt.plot(trajectory_px, trajectory_py, 'r-', linewidth=2, label='Policy Trajectory')
    plt.plot(trajectory_px[0], trajectory_py[0], 'go', markersize=10, label='Start')
    plt.plot(trajectory_px[-1], trajectory_py[-1], 'r*', markersize=15, label='End')
    plt.legend()
    plt.title(title)
    plt.axis('off')
    plt.show()

#%%
# ----- POLICY WHICH IS SAFE FOR BOTH TASK 1 AND TASK 2 -----
env1 = PointNavSafetyEnv(task1_cfg)
if isinstance(env1.action_space, gym.spaces.Discrete):
    num_actions = env1.action_space.n
else:
    num_actions = env1.action_space.shape[0]
if isinstance(env1.observation_space, gym.spaces.Discrete):
    num_obs = env1.observation_space.n
else:
    num_obs = env1.observation_space.shape[0]
 
class PerimeterPolicy(nn.Module):
    """Hardcoded policy that follows the left-top perimeter."""
    def __init__(self, num_actions=5, goal_radius=0.08):
        super().__init__()
        self.num_actions = num_actions
        assert goal_radius > 0.0, "goal_radius must be positive"
        self.goal_radius = goal_radius
        # Action mapping: 0=stay, 1=up, 2=down, 3=left, 4=right
        
    def forward(self, obs):
        """
        Path: (-0.9,-0.9) -> (-0.9,0.9) -> (0.9,0.9)
        Step 1: Go UP along left edge until y ≈ 0.9
        Step 2: Go RIGHT along top edge until x ≈ 0.9
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        batch_size = obs.shape[0]
        x, y = obs[:, 0], obs[:, 1]
        
        actions = torch.zeros(batch_size, dtype=torch.long, device=obs.device)
        
        for i in range(batch_size):
            xi, yi = x[i].item(), y[i].item()

            # first check the x dimension
            if xi < 0.9 - self.goal_radius: # to the left of the goal
                actions[i] =  4  # right
            elif xi > 0.9 + self.goal_radius: # to the right of the goal
                actions[i] = 3  # left
            elif yi < 0.9 - self.goal_radius: # below the goal
                actions[i] = 1  # up
            elif yi > 0.9 + self.goal_radius: # above the goal
                actions[i] = 2  # down
            else:
                actions[i] = 0  # stay
        
        # Return logits (one-hot encoded)
        logits = torch.full((batch_size, self.num_actions), -10.0, device=obs.device)
        logits[range(batch_size), actions] = 10.0
        return logits

omnisafe_actor = PerimeterPolicy(num_actions=num_actions)
task1_critic = None  # Not needed for hardcoded policy

#%%
# ----- Deploy the omnisafe policy in the Task 1 -----
done = False
step_count = 0
total_reward = 0.0
env1 = make_env(task1_cfg)
obs, _ = env1.reset()

print(f"Initial obs: {obs}")
print(f"Action space: {env1.action_space}")
print(f"Observation space: {env1.observation_space}")

trajectory_x = [obs[0]]
trajectory_y = [obs[1]]

while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        logits = omnisafe_actor(obs_tensor)
    action = torch.argmax(logits).item()
    # print(f"Step {step_count}: obs={obs}, action={action}")
    obs, reward, terminated, truncated, info = env1.step(action)
    total_reward += reward
    # print(f"  -> new_obs={obs}, reward={reward}, terminated={terminated}, truncated={truncated}, info={info}")
    done = terminated or truncated
    trajectory_x.append(obs[0])
    trajectory_y.append(obs[1])
    step_count += 1
    # if step_count > 200:  # Safety limit
    #     print("Reached 200 steps, breaking")
    #     break

    if obs[0] == 0.9 and obs[1] == 0.9:
        print("Reached goal!")
        break

print('Last obs:', obs)
print(f"Total steps: {step_count}")
print(f"Trajectory length: {len(trajectory_x)}")
print('Total reward:', total_reward)

### Get the rendered image
img = env1.render()
img_height, img_width = img.shape[:2]

# Transform environment coordinates [-1, 1] to pixel coordinates [0, img_width/height]
# def env_to_pixel(x, y, img_width, img_height):
#     """Convert environment coords [-1, 1] to pixel coords [0, img_size]"""
#     # Map [-1, 1] to [0, img_size]
#     px = (x + 1.0) / 2.0 * img_width
#     py = (1.0 - (y + 1.0) / 2.0) * img_height  # Flip Y: high y values -> low pixel row
#     return px, py
def env_to_pixel(x, y, img_width, img_height):
    """Convert environment coords [-1, 1] to pixel coords [0, img_size]"""
    # Map [-1, 1] to [0, img_size]
    px = (x + 1.0) / 2.0 * img_width
    py = (y + 1.0) / 2.0 * img_height  # No flip - match the render() convention
    return px, py

# Transform all trajectory points
trajectory_px = []
trajectory_py = []
for x, y in zip(trajectory_x, trajectory_y):
    px, py = env_to_pixel(x, y, img_width, img_height)
    trajectory_px.append(px)
    trajectory_py.append(py)
# trajectory_df = pd.DataFrame({
#     'x': trajectory_x,
#     'y': trajectory_y
# })
# trajectory_px_df = pd.DataFrame({
#     'px': trajectory_px,
#     'py': trajectory_py
# })

# Plot with transformed coordinates
plt.figure(figsize=(8, 8))
plt.imshow(img, origin='lower')
plt.plot(trajectory_px, trajectory_py, 'r-', linewidth=2, label='Policy Trajectory')
plt.plot(trajectory_px[0], trajectory_py[0], 'go', markersize=10, label='Start')
plt.plot(trajectory_px[-1], trajectory_py[-1], 'r*', markersize=15, label='End')
plt.legend()
plt.title('Perimeter Policy Trajectory in Task 1')
plt.axis('off')
plt.show()

# %%
### Deploy the omnisafe policy in the Task 2 -----
done = False
step_count = 0
total_reward = 0.0
env2 = make_env(task2_cfg) 
obs, _ = env2.reset()
print(f"Initial obs: {obs}")
print(f"Action space: {env2.action_space}")
print(f"Observation space: {env2.observation_space}")
trajectory_x = [obs[0]]
trajectory_y = [obs[1]]

while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        logits = omnisafe_actor(obs_tensor)
    action = torch.argmax(logits).item()
    obs, reward, terminated, truncated, info = env2.step(action)
    total_reward += reward
    done = terminated or truncated
    trajectory_x.append(obs[0])
    trajectory_y.append(obs[1])
    step_count += 1

    if obs[0] == 0.9 and obs[1] == 0.9:
        print("Reached goal!")
        break

print('Last obs:', obs)
print(f"Total steps: {step_count}")
print(f"Trajectory length: {len(trajectory_x)}")
print('Total reward:', total_reward)
### Get the rendered image
img = env2.render()
img_height, img_width = img.shape[:2]
# Transform all trajectory points
trajectory_px = []
trajectory_py = []
for x, y in zip(trajectory_x, trajectory_y):
    px, py = env_to_pixel(x, y, img_width, img_height)
    trajectory_px.append(px)
    trajectory_py.append(py)
# Plot with transformed coordinates
plt.figure(figsize=(8, 8))
plt.imshow(img, origin='lower')
plt.plot(trajectory_px, trajectory_py, 'r-', linewidth=2, label='Policy Trajectory')
plt.plot(trajectory_px[0], trajectory_py[0], 'go', markersize=10, label='Start')
plt.plot(trajectory_px[-1], trajectory_py[-1], 'r*', markersize=15, label='End')
plt.legend()
plt.title('Perimeter Policy Trajectory in Task 2')
plt.axis('off')
plt.show()
env2.close()
# %%
### Deploy the omnisafe policy in the Task 3 -----
done = False
step_count = 0
total_reward = 0.0
env3 = make_env(task3_cfg) 
obs, _ = env3.reset()
print(f"Initial obs: {obs}")
print(f"Action space: {env3.action_space}")
print(f"Observation space: {env3.observation_space}")
trajectory_x = [obs[0]]
trajectory_y = [obs[1]]
while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        logits = omnisafe_actor(obs_tensor)
    action = torch.argmax(logits).item()
    obs, reward, terminated, truncated, info = env3.step(action)
    total_reward += reward
    done = terminated or truncated
    trajectory_x.append(obs[0])
    trajectory_y.append(obs[1])
    step_count += 1

    if obs[0] == 0.9 and obs[1] == 0.9:
        print("Reached goal!")
        break

print('Last obs:', obs)
print(f"Total steps: {step_count}")
print(f"Trajectory length: {len(trajectory_x)}")
print('Total reward:', total_reward)
### Get the rendered image
img = env3.render()
img_height, img_width = img.shape[:2]
# Transform all trajectory points
trajectory_px = []
trajectory_py = []
for x, y in zip(trajectory_x, trajectory_y):
    px, py = env_to_pixel(x, y, img_width, img_height)
    trajectory_px.append(px)
    trajectory_py.append(py)
# Plot with transformed coordinates
plt.figure(figsize=(8, 8))
plt.imshow(img, origin='lower')
plt.plot(trajectory_px, trajectory_py, 'r-', linewidth=2, label='Policy Trajectory')
plt.plot(trajectory_px[0], trajectory_py[0], 'go', markersize=10, label='Start')
plt.plot(trajectory_px[-1], trajectory_py[-1], 'r*', markersize=15, label='End')
plt.legend()
plt.title('Perimeter Policy Trajectory in Task 3')
plt.axis('off')
plt.show()
env3.close()

# %%
##############################################################################
### Use the omnisafe policy to generate trajectories for Rashomon set computation
# Collect state-action pairs from safe trajectories in Task 1 environment
obs_arr, action_arr, next_obs_arr, failure_arr, done_arr = collect_traces_dataset(
    policy_model=omnisafe_actor, 
    env=env1,
    num_episodes=10,
    epsilon=0.5,
    device='cpu',
    seed=2025
)

final_final_safe_actions_per_state = []
for i in range(obs_arr.shape[0]):
    cur_state = obs_arr[i]
    cur_safe_action = action_arr[i] # aciton is safe because it comes from safe trajs
    # Find all indices where this state occurs
    pad_len = max(0, num_actions - 1)
    cur_pad_arr = np.full((num_actions-1,), -1, dtype=np.int64)
    cur_safe_actions_arr = np.append(cur_safe_action, cur_pad_arr)
    # make sure the final array is of int64 type
    cur_safe_actions_arr = cur_safe_actions_arr.astype(np.int64)
    final_final_safe_actions_per_state.append(cur_safe_actions_arr)
# Identify safe actions for each unique state

safe_actions_tensor = torch.from_numpy(np.array(final_final_safe_actions_per_state))
# Only use the first 2 dimensions (x    , y) to match the neural network input size
states_tensor = torch.from_numpy(obs_arr)
state_action_torch_dataset = TensorDataset(states_tensor, safe_actions_tensor)
state_action_loader = DataLoader(state_action_torch_dataset, batch_size=8, shuffle=False)

#%%
##########################################################################################
### Train a neural policy that imitates the omnisafe policy
print("Training neural network to mimic PerimeterPolicy...")

# Create a simple feedforward neural network
class NeuralPolicy(nn.Module):
    def __init__(self, num_obs=4, num_actions=5, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_obs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
    
    def forward(self, x):
        return self.net(x)

# Create the neural network with explicit dimensions (x, y coordinates only)
neural_policy = NeuralPolicy(num_obs=4, num_actions=env1.action_space.n)

# Generate training data from the PerimeterPolicy
print("Generating training data from PerimeterPolicy...")
num_train_samples = 10000
train_states = np.random.uniform(-0.9, 0.9, size=(num_train_samples, 2)).astype(np.float32)
# add columns with the goal coordinates to match the env observation space if needed
train_states = np.hstack([train_states, np.zeros((num_train_samples, 1), dtype=np.float32)])
train_states[:, 2] = 0.9  # goal x
train_states = np.hstack([train_states, np.zeros((num_train_samples, 1), dtype=np.float32)])
train_states[:, 3] = 0.9  # goal y
train_states_tensor = torch.from_numpy(train_states)

# Get labels from the hardcoded policy
with torch.no_grad():
    train_logits = omnisafe_actor(train_states_tensor)
    train_labels = torch.argmax(train_logits, dim=1)

# Train the neural network
train_dataset = TensorDataset(train_states_tensor, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

optimizer = torch.optim.Adam(neural_policy.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

neural_policy.train()
for epoch in range(100):
    total_loss = 0
    correct = 0
    total = 0
    for batch_states, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = neural_policy(batch_states)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)
    
    if (epoch + 1) % 10 == 0:
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/50, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {acc:.2f}%")

neural_policy.eval()
print("Neural policy training complete!")

# Test the neural policy
print("\nTesting neural policy vs PerimeterPolicy...")
test_states = np.array([[-0.9, -0.9, 0.9, 0.9], [-0.5, 0.0, 0.9, 0.9], [0.0, 0.5, 0.9, 0.9], [0.5, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9]], dtype=np.float32)
test_states_tensor = torch.from_numpy(test_states)

with torch.no_grad():
    original_actions = torch.argmax(omnisafe_actor(test_states_tensor), dim=1)
    neural_actions = torch.argmax(neural_policy(test_states_tensor), dim=1)
    
print("State -> Original Action | Neural Action")
for i, state in enumerate(test_states):
    print(f"{state} -> {original_actions[i].item()} | {neural_actions[i].item()}")

# Check the accuracy of the neural policy on the training dataset
with torch.no_grad():
    train_outputs = neural_policy(train_states_tensor)
    train_predicted = torch.argmax(train_outputs, dim=1)
    train_accuracy = (train_predicted == train_labels).float().mean().item() * 100
print(f"\nNeural policy training accuracy on training dataset: {train_accuracy:.2f}%")

neural_omnisafe_actor = copy.deepcopy(neural_policy)

#%%
### Show the trajectory of the neural policy in Task 3 environment
print("Plotting trajectory for Neural Omnisafe Policy in Task 3...")
env3_traj = PointNavSafetyEnv(task3_cfg)
obs, _ = env3_traj.reset()
trajectory = [obs.copy()]
done = False 
while not done:
    # Use the same action selection as neural_omnisafe_actor
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        logits = neural_omnisafe_actor(obs_tensor)
    action = torch.argmax(logits).item()
    obs, reward, terminated, truncated, info = env3_traj.step(action)
    trajectory.append(obs.copy())
    done = terminated or truncated

trajectory = np.array(trajectory)

### Rendering
render_policy_trajectory(env3_traj, trajectory[:, 0], trajectory[:, 1])

#%%
#### Compute Rashomon set using IntervalTrainer
interval_trainer = IntervalTrainer(
    model=neural_omnisafe_actor.net, # policy which is an instance of nn.Sequential
    min_acc_limit=0.99, # NOTE: tweak only if task 2 adaptation is poor
    seed=2025,
    # n_iters=10_000, # default 2000; running longer may not translate into higher OOS accuracy
)
interval_trainer.compute_rashomon_set(
    dataset=state_action_torch_dataset, # states and safe actions
    multi_label=True # NOTE must be True when there are multiple safe actions per state
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

#%%
### Train PPO on Task 2 with Rashomon bounds

# Define Task 2 PPO configuration
task2_ppo_cfg = PPOConfig(
    total_timesteps=100_000,
    lr=3e-4,
    rollout_steps=2048,
    minibatch_size=64,
    update_epochs=10,
)
env2 = make_env(task2_cfg) # create a new instance of Task 2 environment
obs, _ = env2.reset()

### Train PPO with parameter bounds from Rashomon set
print("\nTraining PPO on Task 2 with Rashomon parameter bounds...")
safe_actor, safe_critic = ppo_train(
    env=env2,
    cfg=task2_ppo_cfg,
    actor_warm_start=neural_omnisafe_actor.net,  # Use the neural network, not the hardcoded policy
    critic_warm_start=None,
    actor_param_bounds_l=param_bounds_l,
    actor_param_bounds_u=param_bounds_u,
)

### Evaluate the safe_actor on both Task 1 and Task 2
print("\nEvaluating safe_actor on Task 1 and Task 2...")
env1_eval = make_env(task1_cfg)
env2_eval = make_env(task2_cfg)
safe_eval_avg_r_task1, safe_eval_std_r_task1, safe_eval_fail_rate_task1 = evaluate(
    actor=safe_actor, 
    env=env1_eval, episodes=1, render_mode='rgb_array'
)
safe_eval_avg_r_task2, safe_eval_std_r_task2, safe_eval_fail_rate_task2 = evaluate(
    actor=safe_actor, 
    env=env2_eval, episodes=1, render_mode='rgb_array'
)

#%% 
### Rendering

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

render_policy_trajectory(env2_traj, trajectory[:, 0], trajectory[:, 1], title='Policy C Trajectory (Rashomon Bounds)')
