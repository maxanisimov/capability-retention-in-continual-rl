"""
FrozenLake Environment: Safe Continual Learning with SafeAdapt

This script demonstrates safe continual learning in the Gymnasium FrozenLake
environment where an agent must navigate from start to goal while avoiding
holes in the ice. The key challenge is adapting to a distribution shift
(Task 2) while maintaining safety on the original task (Task 1).

INPUTS:
-------
Configuration file: demo_configs.yaml
    - Environment maps for Task 1 and Task 2
    - Training hyperparameters (timesteps, etc.)
    - Random seed for reproducibility

Script parameters (see EXPERIMENT CONFIGURATION section):
    - cfg_name: Configuration name from YAML file (e.g., 'standard_4x4')
    - safe_state_action_data_name: Safety dataset type
        * 'Safe Optimal Policy Data': Deterministic trajectory demonstrations
        * 'Safe Training Data': Filtered safe state-action pairs from training
        * 'Sufficient Safety Data': Exhaustive safe actions for all states
    - save_results: Boolean flag to save plots and tables
    - train_unsafe_adapt: Boolean flag to train UnsafeAdapt baseline (optional)

OUTPUTS:
--------
1. Trajectory visualizations (if save_results=True):
   - plots/: PNG files showing agent trajectories in both environments

2. Performance metrics tables:
   - Console: Formatted tables with safety and performance metrics
   - tables/: CSV files with detailed results (if save_results=True)

3. Trained policies:
   - NoAdapt: Baseline policy trained only on Task 1
   - UnsafeAdapt: Policy adapted to Task 2 without safety constraints (optional)
   - SafeAdapt: Policy adapted to Task 2 with SafeAdapt parameter constraints

METHODOLOGY:
------------
Two or three training strategies are compared (UnsafeAdapt is optional):

1. NoAdapt (Baseline):
   - Train on Task 1 only
   - No adaptation to Task 2
   - Expected: Safe on Task 1, may fall into new holes on Task 2

2. UnsafeAdapt (Optional baseline):
   - Train on Task 2 without constraints
   - Expected: Good on Task 2, catastrophic forgetting on Task 1
   - Set train_unsafe_adapt=True to enable

3. SafeAdapt (Proposed):
   - Train on Task 2 with parameter constraints from SafeAdapt set
   - SafeAdapt set computed via interval-bound propagation on safety dataset
   - Expected: Safe and performant on both tasks

REPRODUCIBILITY:
----------------
All random seeds are fixed (numpy, torch, random module). All computations are
performed on CPU. Results should be deterministic across runs.

REQUIREMENTS:
-------------
- PyTorch, NumPy, Pandas, Matplotlib, PyYAML, Gymnasium
- Custom modules: ppo_utils, IntervalTrainer
"""

#%%
# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import random
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Path setup for imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))
rl_project_dir = os.path.join(project_root, 'rl_project')
sys.path.insert(0, rl_project_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, current_script_dir)

from rl_project.utils.ppo_utils import ppo_train, PPOConfig
from src.trainer import IntervalTrainer

# Set paths
plots_dir = os.path.join(current_script_dir, 'plots')
tables_dir = os.path.join(current_script_dir, 'tables')

# Create output directories
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

#%%
# =============================================================================
# ENVIRONMENT WRAPPERS
# =============================================================================

def one_hot_encode_state(state, num_states, task_num=0):
    """Convert discrete state to one-hot encoding with task indicator."""
    encoded = np.zeros(num_states, dtype=np.float32)
    encoded[state] = 1.0
    encoded = np.append(encoded, float(task_num))
    return encoded


def observation_to_position(observation):
    """Convert a one-hot encoded observation (with task indicator) to a flat grid index.

    Args:
        observation: np.ndarray or torch.Tensor of shape ``(num_states + 1,)``.
            The last element is the task indicator and is ignored.

    Returns:
        int: The flat grid index (``row * ncol + col``).
    """
    if isinstance(observation, torch.Tensor):
        return int(torch.argmax(observation[:-1]).item())
    return int(np.argmax(observation[:-1]))


def position_to_observation(position, num_states, task_num=0):
    """Convert a flat grid index to a one-hot encoded observation with task indicator.

    Args:
        position: int - flat grid index (``row * ncol + col``).
        num_states: int - total number of grid cells (``nrow * ncol``).
        task_num: int or float - task indicator appended at the end.

    Returns:
        np.ndarray of shape ``(num_states + 1,)`` with dtype float32.
    """
    return one_hot_encode_state(position, num_states, task_num)


class OneHotWrapper(gym.ObservationWrapper):
    """Wrapper to convert FrozenLake's discrete state to one-hot encoding + task indicator."""

    def __init__(self, env, task_num):
        super().__init__(env)
        low = np.zeros(env.observation_space.n + 1, dtype=np.float32)
        high = np.ones(env.observation_space.n + 1, dtype=np.float32)
        high[-1] = np.inf  # Task indicator is unbounded above
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.task_num = task_num

    def observation(self, obs):
        return one_hot_encode_state(obs, self.env.observation_space.n, self.task_num)


class SafetyFlagWrapper(gym.Wrapper):
    """Wrapper that adds a safety flag to the info dict indicating if current state is safe (not a hole)."""

    def __init__(self, env):
        super().__init__(env)
        self.desc = env.unwrapped.desc
        self.nrow, self.ncol = self.desc.shape

    def _is_safe_state(self, state):
        """Check if a state is safe (not a hole)."""
        row = state // self.ncol
        col = state % self.ncol
        cell = self.desc[row, col]
        cell = cell.decode('utf-8') if isinstance(cell, bytes) else cell
        return cell != 'H'

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, np.ndarray):
            state = np.argmax(obs[:-1])  # Exclude task indicator
        else:
            state = obs
        info['safe'] = self._is_safe_state(state)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, np.ndarray):
            current_state = np.argmax(obs[:-1])  # Exclude task indicator
        else:
            current_state = obs
        info['safe'] = self._is_safe_state(current_state)
        return obs, reward, terminated, truncated, info


#%%
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def evaluate_policy(env, actor, num_episodes=100):
    """
    Evaluate a policy in the given FrozenLake environment.

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

            if not info.get('safe', True):
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


def visualize_agent_trajectory(env, actor, num_episodes=1, max_steps=100,
                               env_name=None, cfg_name=None, actor_name=None, save_dir=None):
    """Visualize the trained agent's trajectory in FrozenLake."""
    action_names = ["LEFT", "DOWN", "RIGHT", "UP"]

    for episode in range(num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print('='*50)

        obs, info = env.reset()
        trajectory = [obs.copy() if isinstance(obs, np.ndarray) else obs]
        rewards_list = []
        actions_list = []

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

            trajectory.append(obs.copy() if isinstance(obs, np.ndarray) else obs)
            rewards_list.append(reward)
            actions_list.append(action)

            action_name = action_names[action]
            print(f"Step {step + 1}: {action_name}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
            step += 1

        print(f"\nEpisode finished! Total reward: {total_reward:.2f}")

        plot_trajectory(env, trajectory, rewards_list, actions_list,
                       episode_num=episode + 1 if num_episodes > 1 else None,
                       env_name=env_name, cfg_name=cfg_name, actor_name=actor_name,
                       save_dir=save_dir)

    if save_dir is None:
        plt.show()


def plot_trajectory(env, trajectory, rewards_list, actions_list,
                   episode_num=None, env_name=None, cfg_name=None, actor_name=None, save_dir=None):
    """Plot a single trajectory for FrozenLake."""
    desc = env.unwrapped.desc
    nrow, ncol = desc.shape
    num_steps = len(trajectory)

    cols = min(5, num_steps)
    rows = (num_steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if num_steps == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    action_symbols = ["\u2190", "\u2193", "\u2192", "\u2191"]  # LEFT, DOWN, RIGHT, UP

    for step_idx, state in enumerate(trajectory):
        row_idx = step_idx // cols
        col_idx = step_idx % cols
        ax = axes[row_idx, col_idx]

        ax.set_xlim(-0.5, ncol - 0.5)
        ax.set_ylim(-0.5, nrow - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.set_xticks(range(ncol))
        ax.set_yticks(range(nrow))
        ax.tick_params(labelsize=12)
        ax.invert_yaxis()

        # Draw environment cells
        for i in range(nrow):
            for j in range(ncol):
                cell = desc[i, j].decode('utf-8') if isinstance(desc[i, j], bytes) else desc[i, j]

                if cell == 'S':
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, color='lightgreen', alpha=0.3)
                    ax.add_patch(rect)
                    ax.text(j, i, 'S', ha='center', va='center', fontsize=14, fontweight='bold', color='green')
                elif cell == 'F':
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, color='lightblue', alpha=0.2)
                    ax.add_patch(rect)
                elif cell == 'H':
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, color='red', alpha=0.3)
                    ax.add_patch(rect)
                    ax.text(j, i, 'H', ha='center', va='center', fontsize=14, fontweight='bold', color='darkred')
                elif cell == 'G':
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, color='gold', alpha=0.4)
                    ax.add_patch(rect)
                    ax.text(j, i, 'G', ha='center', va='center', fontsize=14, fontweight='bold', color='darkgoldenrod')

        # Extract agent position from observation (handle one-hot encoded states)
        if isinstance(state, np.ndarray):
            state_idx = int(np.argmax(state[:-1]))  # One-hot to int, exclude task indicator
        else:
            state_idx = int(state)
        agent_row = state_idx // ncol
        agent_col = state_idx % ncol

        circle = patches.Circle((agent_col, agent_row), 0.3, color='blue', alpha=0.8)
        ax.add_patch(circle)

        if step_idx == 0:
            ax.set_title('Start', fontsize=13, fontweight='bold')
        else:
            action = action_symbols[actions_list[step_idx - 1]]
            reward = rewards_list[step_idx - 1]
            reward_color = 'green' if reward > 0 else ('red' if reward < 0 else 'gray')
            ax.set_title(f'Step {step_idx}: {action} (r={reward:.2f})',
                        fontsize=13, fontweight='bold', color=reward_color)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)

    # Hide empty subplots
    for step_idx in range(num_steps, rows * cols):
        r = step_idx // cols
        c = step_idx % cols
        axes[r, c].axis('off')

    suptitle_parts = []
    if cfg_name:
        suptitle_parts.append(cfg_name)
    if env_name:
        suptitle_parts.append(env_name)
    if actor_name:
        suptitle_parts.append(actor_name)
    if episode_num:
        suptitle_parts.append(f'Episode {episode_num}')
    fig.suptitle(' - '.join(suptitle_parts), fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_dir is not None:
        filename_parts = []
        if cfg_name is not None:
            filename_parts.append(cfg_name)
        if env_name is not None:
            filename_parts.append(env_name.replace(' ', '_').replace('/', '_'))
        if actor_name is not None:
            filename_parts.append(actor_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', ''))
        if episode_num is not None:
            filename_parts.append(f"episode_{episode_num}")
        filename = "_".join(filename_parts) + ".png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {filepath}")
        plt.close(fig)


def get_all_unsafe_state_action_pairs(env_map, task_num, state_repr='observation'):
    """
    Compute all (state, action) pairs where the action leads to a hole.
    For FrozenLake (non-slippery), transitions are deterministic.

    Actions: LEFT=0, DOWN=1, RIGHT=2, UP=3

    Args:
        env_map: List of strings describing the FrozenLake map
        task_num: Task number for the one-hot encoding task indicator
        state_repr: How to represent states in the returned pairs.
            - 'observation': One-hot encoded state with task indicator appended
              (np.ndarray of shape ``(num_states + 1,)``).
            - 'position': Integer grid index (``state = row * ncol + col``).

    Returns:
        list of (state, action) tuples that are unsafe, where the state format
        depends on *state_repr*.
    """
    assert state_repr in ('observation', 'position'), (
        f"state_repr must be 'observation' or 'position', got '{state_repr}'"
    )

    desc = np.array([list(row) for row in env_map])
    nrow, ncol = desc.shape
    num_states = nrow * ncol

    unsafe_pairs = []

    for s in range(num_states):
        row = s // ncol
        col = s % ncol
        cell = desc[row, col]

        # Skip terminal states (holes and goal)
        if cell in ('H', 'G'):
            continue

        # Build state representation
        if state_repr == 'observation':
            state = np.zeros(num_states + 1, dtype=np.float32)
            state[s] = 1.0
            state[-1] = float(task_num)
        else:  # 'position'
            state = s

        # Check each action for safety
        for action in range(4):
            if action == 0:    # LEFT
                new_row, new_col = row, max(col - 1, 0)
            elif action == 1:  # DOWN
                new_row, new_col = min(row + 1, nrow - 1), col
            elif action == 2:  # RIGHT
                new_row, new_col = row, min(col + 1, ncol - 1)
            else:              # UP
                new_row, new_col = max(row - 1, 0), col

            next_cell = desc[new_row, new_col]
            if next_cell == 'H':
                if state_repr == 'observation':
                    unsafe_pairs.append((state.copy(), action))
                else:
                    unsafe_pairs.append((state, action))

    return unsafe_pairs


#%%
# =============================================================================
# SAFETY DATASET CREATION FUNCTIONS
# =============================================================================

def create_safe_optimal_policy_dataset(env, actor, num_rollouts, deterministic=True, seed=42):
    """
    Create a safety dataset from rollouts of an optimal policy.
    Collects state-action pairs by running the actor in the environment.

    Args:
        env: The environment to collect data from
        actor: The policy network to generate actions
        num_rollouts: Number of episodes to collect
        deterministic: Whether to use deterministic actions (argmax)
        seed: Base seed for environment resets

    Returns:
        A torch TensorDataset containing (states, actions) pairs from rollouts
    """
    print("\nCreating 'Safe Optimal Policy Data' dataset...")
    states = []
    actions = []

    for episode in range(num_rollouts):
        obs, info = env.reset(seed=seed + episode)
        done = False
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_logits = actor(obs_tensor)
                if deterministic:
                    action = torch.argmax(action_logits, dim=1).item()
                else:
                    action = torch.distributions.Categorical(logits=action_logits).sample().item()

            states.append(obs)
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    dataset = torch.utils.data.TensorDataset(states, actions)
    print(f"  Collected {len(states)} state-action pairs from {num_rollouts} rollouts")

    return dataset


def create_safe_training_dataset(training_data):
    """
    Create a safety dataset by filtering safe state-action pairs from training trajectories.
    Removes duplicates to create a unique set of safe demonstrations.

    Args:
        training_data: Dictionary containing 'states', 'actions', and 'safe' flags from PPO training

    Returns:
        A torch TensorDataset containing unique safe (states, actions) pairs
    """
    print("\nCreating 'Safe Training Data' dataset...")
    states = torch.FloatTensor(training_data['states'])
    actions = torch.LongTensor(training_data['actions'])
    safes = training_data['safe']

    # Filter only safe state-action pairs
    safe_indices = np.where(safes == 1.0)[0]
    states_safe = states[safe_indices]
    actions_safe = actions[safe_indices]

    # Remove duplicate state-action pairs
    safe_states_df = pd.DataFrame(states_safe.detach().numpy())
    safe_states_df.columns = [f'state_feature_{i}' for i in range(safe_states_df.shape[1])]
    actions_df = pd.DataFrame(actions_safe.detach().numpy(), columns=['action'])
    safe_state_action_pairs_df = pd.concat([safe_states_df, actions_df], axis=1)
    safe_state_action_pairs_df = safe_state_action_pairs_df.drop_duplicates(keep='first').reset_index(drop=True)

    states_safe = torch.FloatTensor(safe_state_action_pairs_df.drop(columns=['action']).values)
    actions_safe = torch.LongTensor(safe_state_action_pairs_df['action'].values)
    dataset = torch.utils.data.TensorDataset(states_safe, actions_safe)
    print(f"  Filtered {len(states_safe)} unique safe state-action pairs from training data")

    return dataset


def generate_sufficient_safe_state_action_dataset(unsafe_state_action_pairs, env):
    """
    Generate a dataset of sufficient safe state-action pairs by computing the complement
    of unsafe actions for each state.

    Args:
        unsafe_state_action_pairs: list of (state, action) tuples that are unsafe
        env: The environment, needed to determine the action space

    Returns:
        A torch dataset containing states and their corresponding sets of safe actions (padded with -1)
    """
    print("\nCreating 'Sufficient Safety Data' dataset...")
    all_actions = set(range(env.action_space.n))

    # Group unsafe actions by state
    unsafe_actions_by_state = {}
    for state, action in unsafe_state_action_pairs:
        state_key = tuple(state)
        if state_key not in unsafe_actions_by_state:
            unsafe_actions_by_state[state_key] = set()
        unsafe_actions_by_state[state_key].add(action)

    # Compute complement (safe actions) for each state
    safe_actions_by_state = {
        state_key: all_actions - unsafe_actions
        for state_key, unsafe_actions in unsafe_actions_by_state.items()
    }
    sufficient_safe_states = torch.FloatTensor(list(safe_actions_by_state.keys()))

    # Pad safe actions with -1 for variable length
    max_actions = env.action_space.n
    padded_safe_actions = []
    for state_key, safe_actions in safe_actions_by_state.items():
        padded = list(safe_actions) + [-1] * (max_actions - len(safe_actions))
        padded_safe_actions.append(padded)
    sufficient_safe_actions = torch.LongTensor(padded_safe_actions)
    dataset = torch.utils.data.TensorDataset(sufficient_safe_states, sufficient_safe_actions)
    print(f"  Generated safe actions for {len(sufficient_safe_states)} states with unsafe neighbors")

    return dataset


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


#%%
# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# -------------------------
# User-defined parameters
# -------------------------
cfg_name = 'standard_4x4'
safe_state_action_data_name = 'Sufficient Safety Data'  # Options: 'Sufficient Safety Data', 'Safe Optimal Policy Data', 'Safe Training Data'
save_results = False  # Set to True to save plots and tables
train_unsafe_adapt = False  # Set to True to train UnsafeAdapt baseline (takes additional time)

# -------------------------
# Load configuration from YAML
# -------------------------
with open(os.path.join(current_script_dir, 'demo_configs.yaml'), 'r') as f:
    DEMO_CONFIGS = yaml.safe_load(f)
cfg = DEMO_CONFIGS[cfg_name]

print("="*80)
print("FROZEN LAKE SAFE CONTINUAL LEARNING EXPERIMENT")
print("="*80)
print(f"Configuration: {cfg_name}")
print(f"Safety dataset: {safe_state_action_data_name}")
print(f"Save results: {save_results}")
print(f"Train UnsafeAdapt: {train_unsafe_adapt}")
print("="*80 + "\n")

# -------------------------
# Extract configuration parameters
# -------------------------
env1_map = cfg['env1_map']
env2_map = cfg['env2_map']
is_slippery = cfg['is_slippery']
max_steps = cfg['max_steps']
safe_env1_state_action_data_num_rollouts = cfg['safe_env1_state_action_data_num_rollouts']
seed = cfg['seed']

# Training hyperparameters
no_adapt_timesteps = cfg['unadaptable_actor_timesteps']
task2_adapt_timesteps = cfg.get('task2_adapt_timesteps', 0)
safe_adapt_timesteps = cfg['rashomon_timesteps']

# -------------------------
# Set random seeds for reproducibility
# -------------------------
set_all_seeds(seed)
print(f"Random seed set to: {seed}\n")

# -------------------------
# Validate configuration
# -------------------------
assert cfg_name in DEMO_CONFIGS, f"Configuration '{cfg_name}' not found in demo_configs.yaml"
assert safe_state_action_data_name in [
    'Safe Optimal Policy Data',
    'Safe Training Data',
    'Sufficient Safety Data'
], f"Invalid safety dataset name: {safe_state_action_data_name}"
assert seed is not None, "Random seed must be set for reproducibility"
print("Configuration validated successfully.\n")

# -------------------------
# Configure output directories
# -------------------------
if not save_results:
    plots_dir = None
    tables_dir = None
else:
    print(f"Results will be saved to:")
    print(f"  Plots: {plots_dir}")
    print(f"  Tables: {tables_dir}\n")

#%%
# =============================================================================
# TASK 1 ENVIRONMENT SETUP
# =============================================================================

print("\n" + "="*80)
print("TASK 1: Training NoAdapt baseline on initial environment")
print("="*80)

# Create Task 1 environment
env1 = gym.make('FrozenLake-v1', desc=env1_map, is_slippery=is_slippery, render_mode=None)
env1 = OneHotWrapper(env1, task_num=0)
env1 = SafetyFlagWrapper(env1)

print(f"Environment created: FrozenLake {'(slippery)' if is_slippery else '(deterministic)'}")
print(f"Task 1 map:")
for row in env1_map:
    print(f"  {row}")

#%%
# =============================================================================
# TRAIN NOADAPT BASELINE (Task 1 only)
# =============================================================================

print("\nTraining NoAdapt policy...")
ppo_cfg = PPOConfig(
    total_timesteps=no_adapt_timesteps,
    device='cpu'
)
standard_actor, standard_critic, standard_training_data = ppo_train(
    env=env1,
    cfg=ppo_cfg,
    return_training_data=True
)
print(f"NoAdapt training complete ({no_adapt_timesteps} timesteps)")

#%%
# Visualize NoAdapt on Task 1
print("\nVisualizing NoAdapt on Task 1...")
# visualize_agent_trajectory(
#     env1, standard_actor, num_episodes=1, max_steps=max_steps,
#     env_name='Task 1', cfg_name=cfg_name, actor_name='NoAdapt', save_dir=plots_dir
# )

from utils.gymnasium_utils import plot_gymnasium_episode
print("\nVisualizing NoAdapt on Task 1 with plot_gymnasium_episode...")
env1_show = gym.make('FrozenLake-v1', desc=env1_map, is_slippery=is_slippery, render_mode='rgb_array')
env1_show = OneHotWrapper(env1_show, task_num=0)
env1_show = SafetyFlagWrapper(env1_show)
_ = plot_gymnasium_episode(
    env=env1_show,  # Pass the environment directly
    actor=standard_actor,
    figsize_per_frame=(1.5, 1.5),
    title=f"{cfg_name} - Task 1 - NoAdapt",
    # save_path=os.path.join(plots_dir, f"{cfg_name}_Task1_NoAdapt.png")
)

#%%
# =============================================================================
# TASK 2 ENVIRONMENT SETUP
# =============================================================================

print("\n" + "="*80)
print("TASK 2: Distribution shift (hole positions changed)")
print("="*80)

env2 = gym.make('FrozenLake-v1', desc=env2_map, is_slippery=is_slippery, render_mode=None)
env2 = OneHotWrapper(env2, task_num=1)
env2 = SafetyFlagWrapper(env2)

print(f"Task 2 environment created")
print(f"Task 2 map:")
for row in env2_map:
    print(f"  {row}")
print("Distribution shift: Hole positions changed between tasks")

# Visualize NoAdapt on Task 2 (may fail on new holes)
print("\nVisualizing NoAdapt on Task 2 (may fail on new holes)...")
# visualize_agent_trajectory(
#     env2, standard_actor, num_episodes=1, max_steps=max_steps,
#     env_name='Task 2', cfg_name=cfg_name, actor_name='NoAdapt',
#     save_dir=plots_dir
# )
print("\nVisualizing NoAdapt on Task 2 with plot_gymnasium_episode...")
env2_show = gym.make('FrozenLake-v1', desc=env2_map, is_slippery=is_slippery, render_mode='rgb_array')
env2_show = OneHotWrapper(env2_show, task_num=1)
env2_show = SafetyFlagWrapper(env2_show)
_ = plot_gymnasium_episode(
    env=env2_show,  # Pass the environment directly
    actor=standard_actor,
    figsize_per_frame=(1.5, 1.5),
    title=f"{cfg_name} - Task 2 - NoAdapt",
    # save_path=os.path.join(plots_dir, f"{cfg_name}_Task2_NoAdapt.png")
)

#%%
# =============================================================================
# TRAIN UNSAFEADAPT BASELINE (Task 2 without safety constraints) - OPTIONAL
# =============================================================================
if train_unsafe_adapt:
    print("\n" + "="*80)
    print("Training UnsafeAdapt baseline (Task 2, no safety constraints)")
    print("="*80)
    print("Expected: Good performance on Task 2, catastrophic forgetting on Task 1\n")

    ppo_cfg_unsafe = PPOConfig(
        total_timesteps=task2_adapt_timesteps,
        device='cpu'
    )
    amnesic_actor, _ = ppo_train(
        env=env2,
        cfg=ppo_cfg_unsafe,
        actor_warm_start=standard_actor,
        critic_warm_start=standard_critic,
    )
    print(f"UnsafeAdapt training complete ({task2_adapt_timesteps} timesteps)")

    # Visualize UnsafeAdapt on Task 1 (expected catastrophic forgetting)
    print("\nVisualizing UnsafeAdapt on Task 1 (expected catastrophic forgetting)...")
    _ = plot_gymnasium_episode(
        env=env1_show,  # Pass the environment directly
        actor=amnesic_actor,
        figsize_per_frame=(1.5, 1.5),
        title=f"{cfg_name} - Task 1 - UnsafeAdapt",
        # save_path=os.path.join(plots_dir, f"{cfg_name}_Task1_UnsafeAdapt.png")
    )

    # Visualize UnsafeAdapt on Task 2 (expected good performance)
    print("\nVisualizing UnsafeAdapt on Task 2 (expected good performance)...")
    _ = plot_gymnasium_episode(
        env=env2_show,  # Pass the environment directly
        actor=amnesic_actor,
        figsize_per_frame=(1.5, 1.5),
        title=f"{cfg_name} - Task 2 - UnsafeAdapt",
        # save_path=os.path.join(plots_dir, f"{cfg_name}_Task2_UnsafeAdapt.png")
    )

else:
    print("\n" + "="*80)
    print("SKIPPING UnsafeAdapt baseline (train_unsafe_adapt=False)")
    print("="*80)
    print("Note: Set train_unsafe_adapt=True to train UnsafeAdapt baseline\n")
    amnesic_actor = None

#%%
# =============================================================================
# CREATE SAFETY DATASETS FOR SAFEADAPT
# =============================================================================

print("\n" + "="*80)
print("Creating safety datasets for SafeAdapt constraint generation")
print("="*80)

# Create three types of safety datasets using helper functions
safe_optimal_policy_data = create_safe_optimal_policy_dataset(
    env=env1,
    actor=standard_actor,
    num_rollouts=safe_env1_state_action_data_num_rollouts,
    seed=seed
)

safe_training_data_ds = create_safe_training_dataset(
    training_data=standard_training_data
)

unsafe_state_action_pairs = get_all_unsafe_state_action_pairs(env_map=env1_map, task_num=0)
sufficient_safe_state_action_torch_dataset = generate_sufficient_safe_state_action_dataset(
    unsafe_state_action_pairs=unsafe_state_action_pairs,
    env=env1
)

# Collect all datasets
safe_datasets = {
    'Safe Optimal Policy Data': safe_optimal_policy_data,
    'Safe Training Data': safe_training_data_ds,
    'Sufficient Safety Data': sufficient_safe_state_action_torch_dataset
}

print(f"\nUsing dataset: {safe_state_action_data_name}")
print(f"Dataset size: {len(safe_datasets[safe_state_action_data_name])} state-action pairs")

#%%
# Optional: Visualize the safety dataset
# try:
#     from visualize_safe_dataset import visualize_safe_state_action_dataset
#     save_path = None
#     if plots_dir is not None:
#         save_path = os.path.join(plots_dir, f'safe_state_action_dataset_Task1_{safe_state_action_data_name.replace(" ", "_")}.png')
#     visualize_safe_state_action_dataset(
#         dataset=safe_datasets[safe_state_action_data_name],
#         env_map=env1_map,
#         save_path=save_path
#     )
# except ImportError:
#     print("visualize_safe_dataset module not found, skipping dataset visualization")

# List of (state_index, action) pairs — state is the flat grid index (row * ncol + col)
unsafe_position_action_pairs = get_all_unsafe_state_action_pairs(
    env_map=env1_map, task_num=0, state_repr='position'
)

safe_observation_action_data = safe_datasets[safe_state_action_data_name]
positons = [
    observation_to_position(safe_observation_action_data.tensors[0][i, :])
    for i in range(safe_observation_action_data.tensors[0].shape[0])
]
sufficient_safe_position_action_pairs = []
for i, pos in enumerate(positons):
    cur_safe_actions = safe_observation_action_data.tensors[1][i]
    for action in cur_safe_actions:
        if action != -1:  # Skip padding
            sufficient_safe_position_action_pairs.append((pos, action.item()))


from utils.gymnasium_utils import plot_state_action_pairs
# Create the env with rgb_array rendering
env_vis = gym.make('FrozenLake-v1', desc=env1_map, is_slippery=False, render_mode='rgb_array')

fig = plot_state_action_pairs(
    env=env_vis,
    state_action_pairs=unsafe_position_action_pairs, # pairs,
    title="Unsafe State-Action Pairs",
    arrow_color="red",
    # save_path="plots/state_actions.png",  # uncomment to save
)

fig = plot_state_action_pairs(
    env=env_vis,
    state_action_pairs=sufficient_safe_position_action_pairs, # pairs,
    title="Sufficient Safe State-Action Pairs",
    arrow_color="green",
    # save_path="plots/state_actions.png",  # uncomment to save
)

# %%
# =============================================================================
# COMPUTE SAFEADAPT SET (Parameter Constraints)
# =============================================================================

print("\n" + "="*80)
print("Computing SafeAdapt set via interval bound propagation")
print("="*80)

state_action_dataset = safe_datasets[safe_state_action_data_name]

### NOTE: NEW STEP -- PER-STATE SAFE NET
# Idea: train a neural net that recovers that
# nails the multi-label classification 
# in state_action_dataset

# --- Train a "safety actor" with the same architecture as standard_actor ---
import copy
########################################################################
# The safety_actor is best described as a safety reference model —     #
# a neural shield whose learned parameters define the center of a      #
# certified safe region in parameter space.                            #
# The novelty relative to classical shielding                          #
# is that the safety constraint is "baked into"                        #
# the policy's parameters via interval bound propagation,              #
#  rather than applied as an external filter.                          #
########################################################################

# Build a fresh network with the same architecture, initialised from standard_actor
safety_actor = copy.deepcopy(standard_actor).cpu()

# Determine whether the dataset uses multi-label targets
multi_label_safety = (safe_state_action_data_name != 'Safe Optimal Policy Data')

# Training hyper-parameters
safety_lr = 1e-3
safety_epochs = 500
safety_batch_size = min(64, len(state_action_dataset))

safety_optimizer = torch.optim.Adam(safety_actor.parameters(), lr=safety_lr)
safety_loader = torch.utils.data.DataLoader(
    state_action_dataset, batch_size=safety_batch_size, shuffle=True
)

print("\n--- Training safety actor ---")
print(f"  Architecture: same as standard_actor")
print(f"  Multi-label: {multi_label_safety}")
print(f"  Dataset size: {len(state_action_dataset)}")
print(f"  Epochs: {safety_epochs}, LR: {safety_lr}, Batch size: {safety_batch_size}")

for epoch in range(safety_epochs):
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0

    for batch_states, batch_actions in safety_loader:
        logits = safety_actor(batch_states)  # (B, n_actions)

        if multi_label_safety:
            # batch_actions: (B, max_actions) with -1 padding for unused slots
            # For each sample, valid actions are those != -1.
            # Loss: for every valid action, push its logit above all invalid ones.
            # We use a soft multi-label cross-entropy: targets are uniform over valid actions.
            n_actions = logits.shape[1]
            valid_mask = (batch_actions != -1)  # (B, max_actions)

            # Build a target distribution: 1 for valid actions, 0 otherwise
            target_dist = torch.zeros_like(logits)  # (B, n_actions)
            for i in range(batch_actions.shape[1]):
                col = batch_actions[:, i]         # (B,)
                col_valid = valid_mask[:, i]       # (B,)
                # Scatter 1s into valid action positions
                indices = col.clamp(min=0).unsqueeze(1)  # (B, 1)
                target_dist.scatter_(1, indices, col_valid.float().unsqueeze(1))

            # Normalise to a probability distribution
            target_dist = target_dist / target_dist.sum(dim=1, keepdim=True).clamp(min=1e-8)

            # Cross-entropy with soft targets
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            loss = -(target_dist * log_probs).sum(dim=1).mean()

            # Accuracy: predicted action is in the valid set
            predicted = logits.argmax(dim=1)  # (B,)
            predicted_expanded = predicted.unsqueeze(1).expand_as(batch_actions)
            matches = (predicted_expanded == batch_actions) & valid_mask
            epoch_correct += matches.any(dim=1).sum().item()
        else:
            # Single-label cross-entropy
            loss = torch.nn.functional.cross_entropy(logits, batch_actions)

            predicted = logits.argmax(dim=1)
            epoch_correct += (predicted == batch_actions).sum().item()

        epoch_total += batch_states.shape[0]
        epoch_loss += loss.item() * batch_states.shape[0]

        safety_optimizer.zero_grad()
        loss.backward()
        safety_optimizer.step()

    epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
    if (epoch + 1) % 100 == 0 or epoch == 0 or epoch_acc == 1.0:
        print(f"  Epoch {epoch+1:4d}/{safety_epochs}  loss={epoch_loss/epoch_total:.4f}  acc={epoch_acc:.4f}")
    if epoch_acc == 1.0:
        print("  Perfect accuracy reached – stopping early.")
        break

# Final evaluation on full dataset
with torch.no_grad():
    all_states = state_action_dataset.tensors[0]
    all_actions = state_action_dataset.tensors[1]
    all_logits = safety_actor(all_states)
    all_preds = all_logits.argmax(dim=1)
    if multi_label_safety:
        valid_mask_all = (all_actions != -1)
        preds_exp = all_preds.unsqueeze(1).expand_as(all_actions)
        final_safety_actor_acc = ((preds_exp == all_actions) & valid_mask_all).any(dim=1).float().mean().item()
    else:
        final_safety_actor_acc = (all_preds == all_actions).float().mean().item()
print(f"\nSafety actor final accuracy on safety dataset: {final_safety_actor_acc:.4f}")
print("--- Safety actor training complete ---\n")


#%%

# Initialize IntervalTrainer for computing parameter bounds
# min_acc_limit = 1.0 # Require 100% accuracy on the safety dataset for certification
### Min acc limit is ideally inferred from the safety reference model
min_acc_limit = final_safety_actor_acc
assert min_acc_limit == 1.0, "Expected safety actor to achieve perfect accuracy on the safety dataset"
print(f"Using min_acc_limit = {min_acc_limit:.4f} for interval training (based on safety actor performance)")
interval_trainer = IntervalTrainer(
    model=safety_actor.cpu(), # we use the safety_actor as the reference model for interval training
    min_acc_limit=min_acc_limit,
    seed=seed,
)

# Determine whether to use multi-label formulation
multi_label = (safe_state_action_data_name != 'Safe Optimal Policy Data')
print(f"Multi-label mode: {multi_label}")
print(f"Computing parameter bounds...")

interval_trainer.compute_rashomon_set(
    dataset=state_action_dataset,
    multi_label=multi_label
)

# Extract parameter bounds
assert len(interval_trainer.bounds) == 1, "Expected exactly one bounded model"
certificate = interval_trainer.certificates[0]
bounded_model = interval_trainer.bounds[0]
param_bounds_l = [bound.detach().cpu() for bound in bounded_model.param_l]
param_bounds_u = [bound.detach().cpu() for bound in bounded_model.param_u]

print(f"\nSafeAdapt set computed successfully!")
print(f"Certified accuracy on safety dataset: {certificate:.4f}")
print(f"Number of parameter constraints: {len(param_bounds_l)} layer(s)")

#%%
# =============================================================================
# TRAIN SAFEADAPT (Task 2 with SafeAdapt constraints)
# =============================================================================

print("\n" + "="*80)
print("Training SafeAdapt policy on Task 2 with parameter constraints")
print("="*80)
print("Expected: Good performance on both Task 1 and Task 2\n")

ppo_cfg_safeadapt = PPOConfig(
    total_timesteps=200_000, # 500_000, # safe_adapt_timesteps,
    device='cpu'
)
# In continual learning, the initialisation should be the previously learned policy
# In general, one could use the safety_actor as the initialisation, but here we warm-start
#  from the standard_actor to be in line with continual learning setting and show that the constraints
#  alone are sufficient to maintain safety
safeadapt_actor, _ = ppo_train(
    env=env2,
    cfg=ppo_cfg_safeadapt,
    actor_warm_start=standard_actor,
    critic_warm_start=standard_critic,
    actor_param_bounds_l=param_bounds_l,
    actor_param_bounds_u=param_bounds_u
)
print(f"SafeAdapt training complete ({safe_adapt_timesteps} timesteps)")

# Visualize SafeAdapt on Task 1 (expected: maintain safety)
print("\nVisualizing SafeAdapt on Task 1 (expected: maintain safety)...")
_ = plot_gymnasium_episode(
    env=env1_show,  # Pass the environment directly
    actor=safeadapt_actor,
    figsize_per_frame=(1.5, 1.5),
    title=f"{cfg_name} - Task 1 - SafeAdapt",
    # save_path=os.path.join(plots_dir, f"{cfg_name}_Task1_SafeAdapt.png")
)

# Visualize SafeAdapt on Task 2 (expected: adapt successfully)
print("\nVisualizing SafeAdapt on Task 2 (expected: adapt successfully)...")
_ = plot_gymnasium_episode(
    env=env2_show,  # Pass the environment directly
    actor=safeadapt_actor,
    figsize_per_frame=(1.5, 1.5),
    title=f"{cfg_name} - Task 2 - SafeAdapt",
    # save_path=os.path.join(plots_dir, f"{cfg_name}_Task2_SafeAdapt.png")
)

#%%
# =============================================================================
# VERIFY SAFEADAPT ACCURACY ON SAFETY DATASET
# =============================================================================

rashomon_actor_actions = safeadapt_actor(state_action_dataset.tensors[0]).argmax(axis=1)
if multi_label:
    # For multi-label datasets, check if predicted action is in the set of valid actions
    valid_actions = state_action_dataset.tensors[1]  # shape: (num_samples, max_actions)
    rashomon_actor_actions_expanded = rashomon_actor_actions.unsqueeze(1).expand_as(valid_actions)
    matches = (rashomon_actor_actions_expanded == valid_actions)
    valid_action_mask = (valid_actions != -1)
    correct_predictions = (matches & valid_action_mask).any(dim=1).sum().item()
    accuracy = correct_predictions / len(rashomon_actor_actions)
else:
    accuracy = (rashomon_actor_actions == state_action_dataset.tensors[1]).sum().item() / len(rashomon_actor_actions)

accuracy_validated = accuracy >= min_acc_limit
if accuracy_validated:
    print(f"Certified accuracy validated: {accuracy:.2f} >= {min_acc_limit:.2f}")
else:
    print(f"Certified accuracy validation failed: {accuracy:.2f} < {min_acc_limit:.2f}")

    # Show which states have incorrect predictions
    if not accuracy_validated:
        print("\nStates with incorrect predictions:")
        
        # Helper to convert one-hot state to grid position
        def state_to_position(state):
            """Convert one-hot encoded state to (row, col) grid position."""
            state_idx = int(torch.argmax(state[:-1]).item())  # Exclude task indicator
            nrow, ncol = len(env1_map), len(env1_map[0])
            row = state_idx // ncol
            col = state_idx % ncol
            return row, col
        
        if multi_label:
            # For multi-label, find states where prediction is not in valid action set
            valid_actions = state_action_dataset.tensors[1]
            rashomon_actor_actions_expanded = rashomon_actor_actions.unsqueeze(1).expand_as(valid_actions)
            matches = (rashomon_actor_actions_expanded == valid_actions)
            valid_action_mask = (valid_actions != -1)
            correct_mask = (matches & valid_action_mask).any(dim=1)
            incorrect_indices = torch.where(~correct_mask)[0]
            
            for idx in incorrect_indices:
                state = state_action_dataset.tensors[0][idx]
                row, col = state_to_position(state)
                predicted_action = rashomon_actor_actions[idx].item()
                valid_actions_list = valid_actions[idx][valid_actions[idx] != -1].tolist()
                print(f"  Position ({row}, {col}): predicted action {predicted_action}, valid actions {valid_actions_list}")
        else:
            # For single-label, find states where prediction != target action
            target_actions = state_action_dataset.tensors[1]
            incorrect_mask = (rashomon_actor_actions != target_actions)
            incorrect_indices = torch.where(incorrect_mask)[0]
            
            for idx in incorrect_indices:
                state = state_action_dataset.tensors[0][idx]
                row, col = state_to_position(state)
                predicted_action = rashomon_actor_actions[idx].item()
                target_action = target_actions[idx].item()
                print(f"  Position ({row}, {col}): predicted action {predicted_action}, target action {target_action}")
        
        print(f"\nTotal incorrect predictions: {len(incorrect_indices)} out of {len(rashomon_actor_actions)}")

# %%
# =============================================================================
# FINAL EVALUATION AND RESULTS
# =============================================================================

print("\n" + "="*80)
print("FINAL EVALUATION: Comparing all trained policies")
print("="*80)

# Evaluate all policies on both tasks
num_eval_episodes = 1  # Deterministic environment and policy

print("\nEvaluating NoAdapt...")
noadapt_task1_metrics = evaluate_policy(env1, standard_actor, num_episodes=num_eval_episodes)
noadapt_task2_metrics = evaluate_policy(env2, standard_actor, num_episodes=num_eval_episodes)

# Conditionally evaluate UnsafeAdapt
if train_unsafe_adapt:
    print("Evaluating UnsafeAdapt...")
    unsafeadapt_task1_metrics = evaluate_policy(env1, amnesic_actor, num_episodes=num_eval_episodes)
    unsafeadapt_task2_metrics = evaluate_policy(env2, amnesic_actor, num_episodes=num_eval_episodes)

print("Evaluating SafeAdapt...")
safeadapt_task1_metrics = evaluate_policy(env1, safeadapt_actor, num_episodes=num_eval_episodes)
safeadapt_task2_metrics = evaluate_policy(env2, safeadapt_actor, num_episodes=num_eval_episodes)

# Create comprehensive results dataframe
results_dict = {
    'NoAdapt / Task 1': noadapt_task1_metrics,
    'NoAdapt / Task 2': noadapt_task2_metrics,
}

if train_unsafe_adapt:
    results_dict['UnsafeAdapt / Task 1'] = unsafeadapt_task1_metrics
    results_dict['UnsafeAdapt / Task 2'] = unsafeadapt_task2_metrics

results_dict['SafeAdapt / Task 1'] = safeadapt_task1_metrics
results_dict['SafeAdapt / Task 2'] = safeadapt_task2_metrics

results_df = pd.DataFrame(results_dict)

# Print results
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print("\nComplete Results Table:")
print(results_df.round(4).to_string())

# Save results to CSV
if save_results and tables_dir is not None:
    csv_path = os.path.join(tables_dir, f'results_{cfg_name}_{safe_state_action_data_name.replace(" ", "_")}.csv')
    results_df.to_csv(csv_path)
    print(f"\nResults saved to: {csv_path}")

#%%
# =============================================================================
# VERIFICATION
# =============================================================================

print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

print("\nVerifying expected behaviors:")

# SafeAdapt should have perfect safety on Task 1 (certified)
safeadapt_task1_safe = results_df.loc['avg_safety_success', 'SafeAdapt / Task 1']
print(f"SafeAdapt perfect safety on Task 1: {safeadapt_task1_safe >= min_acc_limit} (safety={safeadapt_task1_safe:.3f}, certified)")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
print(f"\nConfiguration: {cfg_name}")
print(f"Safety dataset: {safe_state_action_data_name}")
print(f"Random seed: {seed}")
print(f"All operations performed on: CPU")
print(f"UnsafeAdapt trained: {train_unsafe_adapt}")
if save_results:
    print(f"\nResults saved to:")
    print(f"  - Plots: {plots_dir}")
    print(f"  - Tables: {tables_dir}")
else:
    print(f"\nResults not saved (save_results=False)")
print("\n" + "="*80)

# %%
### Visualisaiton of actors across tasks
_ = plot_gymnasium_episode(
    env=env2_show,  # Pass the environment directly
    actor=standard_actor,
    n_cols=3,
    figsize_per_frame=(1.5, 1.5),
    title=f"{cfg_name} - Task 2 - NoAdapt",
    # save_path=os.path.join(plots_dir, f"{cfg_name}_Task2_NoAdapt.png")
)
# %%
def plot_gymnasium_episode_multitask(
        actor: torch.nn.Module,
        env_task1: gymnasium.Env | None = None,
        env_task2: gymnasium.Env | None = None,
        env_id_task1: str | None = None,
        env_id_task2: str | None = None,
        env_kwargs_task1: dict | None = None,
        env_kwargs_task2: dict | None = None,
        n_cols: int = 4,
        log_std: torch.nn.Parameter | None = None,
        deterministic: bool = True,
        seed: int = 42,
        save_path: str | None = None,
        figsize_per_frame: tuple[float, float] = (3.0, 3.0),
        title: str | None = None,
        task1_label: str = "Task 1 (Source)",
        task2_label: str = "Task 2 (Downstream)",
        task1_title_y: float = 0.7,
        task2_title_y: float = 0.7,
        suptitle_y: float = 0.98,
    ):
    """
    Run one episode in two Gymnasium environments (source and downstream tasks)
    and display all frames as a matplotlib grid with both tasks clearly separated.

    The top rows show frames from Task 1 and the bottom rows show frames from
    Task 2, with a bold label separating the two sections.

    Args:
        actor: Trained policy network (nn.Sequential or nn.Module).
        env_task1: Optional pre-created Gymnasium environment for Task 1.
        env_task2: Optional pre-created Gymnasium environment for Task 2.
        env_id_task1: Gymnasium environment ID for Task 1 (used if env_task1 is None).
        env_id_task2: Gymnasium environment ID for Task 2 (used if env_task2 is None).
        env_kwargs_task1: Extra keyword arguments for Task 1 gymnasium.make.
        env_kwargs_task2: Extra keyword arguments for Task 2 gymnasium.make.
        n_cols: Number of columns in the image grid.
        log_std: Log standard deviation parameter for continuous action spaces.
        deterministic: Whether to select actions deterministically.
        seed: Random seed for both episodes.
        save_path: If provided, save the figure to this file path.
        figsize_per_frame: (width, height) in inches for each subplot cell.
        title: Optional suptitle for the entire figure.
        task1_label: Label displayed above the Task 1 rows.
        task2_label: Label displayed above the Task 2 rows.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: The collected RGB frames
            for Task 1 and Task 2 respectively.
    """

    def _collect_frames(env, env_id, env_kwargs):
        """Create env if needed and collect one episode of frames."""
        if env is None and env_id is None:
            raise ValueError("Either env or env_id must be provided for each task")

        close_env = False
        if env is not None:
            assert env.unwrapped.render_mode == 'rgb_array', \
                "Environment must be created with render_mode='rgb_array'"
        elif env_id is not None:
            env_kwargs = env_kwargs or {}
            env = gymnasium.make(env_id, render_mode='rgb_array', **env_kwargs)
            close_env = True

        continuous_actions = isinstance(env.action_space, gymnasium.spaces.Box)
        if not deterministic and continuous_actions:
            assert log_std is not None, \
                "log_std must be provided for stochastic continuous actions"

        frames: list[np.ndarray] = []
        obs, _ = env.reset(seed=seed)
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if continuous_actions:
                    if deterministic:
                        action = actor(obs_t).cpu().numpy()[0]
                    else:
                        mean = actor(obs_t)
                        std = torch.exp(log_std)  # type: ignore[arg-type]
                        dist = torch.distributions.Normal(mean, std)
                        action = dist.sample().cpu().numpy()[0]
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                else:
                    logits = actor(obs_t)
                    if deterministic:
                        action = torch.argmax(logits, dim=-1).item()
                    else:
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample().item()

            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            done = terminated or truncated

        if close_env:
            env.close()

        return frames

    # --- collect frames for both tasks ---
    frames_task1 = _collect_frames(env_task1, env_id_task1, env_kwargs_task1)
    frames_task2 = _collect_frames(env_task2, env_id_task2, env_kwargs_task2)

    # --- compute grid layout ---
    n_frames_t1 = len(frames_task1)
    n_frames_t2 = len(frames_task2)
    n_rows_t1 = math.ceil(n_frames_t1 / n_cols)
    n_rows_t2 = math.ceil(n_frames_t2 / n_cols)

    # Dedicated label rows for Task 1 (top) and Task 2 (separator)
    label_ratio = 0.18  # height of a label row relative to a frame row
    total_rows = 1 + n_rows_t1 + 1 + n_rows_t2  # label1 + t1 frames + label2 + t2 frames

    fig_w = figsize_per_frame[0] * n_cols
    fig_h = figsize_per_frame[1] * (n_rows_t1 + n_rows_t2) + figsize_per_frame[1] * label_ratio * 2

    height_ratios = (
        [label_ratio]
        + [1] * n_rows_t1
        + [label_ratio]
        + [1] * n_rows_t2
    )

    fig, axes = plt.subplots(
        total_rows, n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={
            'height_ratios': height_ratios,
            'hspace': 0.08,
            'wspace': 0.04,
        },
    )
    axes = np.asarray(axes).reshape(total_rows, n_cols)

    # --- Task 1 label row (row 0) ---
    for col_idx in range(n_cols):
        axes[0, col_idx].axis('off')
    axes[0, n_cols // 2].text(
        x=0.5, y=task1_title_y, 
        s=task1_label,
        transform=axes[0, n_cols // 2].transAxes,
        fontsize=11, fontweight='bold',
        ha='center', va='center',
    )

    # --- Task 1 frame rows ---
    for row_idx in range(n_rows_t1):
        for col_idx in range(n_cols):
            ax = axes[1 + row_idx, col_idx]
            frame_idx = row_idx * n_cols + col_idx
            if frame_idx < n_frames_t1:
                ax.imshow(frames_task1[frame_idx])
                ax.set_title(f"Step {frame_idx}", fontsize=8, pad=2)
            ax.axis('off')

    # --- Task 2 label row ---
    sep_row = 1 + n_rows_t1
    for col_idx in range(n_cols):
        axes[sep_row, col_idx].axis('off')
    axes[sep_row, n_cols // 2].text(
        x=0.5, y=task2_title_y, 
        s=task2_label,
        transform=axes[sep_row, n_cols // 2].transAxes,
        fontsize=11, fontweight='bold',
        ha='center', va='center',
    )
    axes[sep_row, 0].set_ylabel(
        task2_label, fontsize=11, fontweight='bold', labelpad=10
    )

    # --- Task 2 frame rows ---
    for row_idx in range(n_rows_t2):
        for col_idx in range(n_cols):
            ax = axes[sep_row + 1 + row_idx, col_idx]
            frame_idx = row_idx * n_cols + col_idx
            if frame_idx < n_frames_t2:
                ax.imshow(frames_task2[frame_idx])
                ax.set_title(f"Step {frame_idx}", fontsize=8, pad=2)
            ax.axis('off')

    fig.tight_layout(pad=0.3)

    if title is not None:
        fig.suptitle(title, fontsize=13, fontweight='bold',y=suptitle_y)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()
    return frames_task1, frames_task2

#%%
### Visualize all actors on both tasks in a single combined plot
# from utils.gymnasium_utils import plot_gymnasium_episode_multitask
plots_dir = 'plots'
# actor_to_plot = 'SafeAdapt' # NoAdapt, UnsafeAdapt, SafeAdapt
for actor_to_plot in ['NoAdapt', 'UnsafeAdapt', 'SafeAdapt']:
    if actor_to_plot == 'NoAdapt':
        actor = standard_actor
    elif actor_to_plot == 'UnsafeAdapt' and train_unsafe_adapt:
        actor = amnesic_actor
    elif actor_to_plot == 'SafeAdapt':
        actor = safeadapt_actor

    title=f"{cfg_name} - {actor_to_plot} on Task 1 and Task 2"
    _ = plot_gymnasium_episode_multitask(
        env_task1=env1_show,
        env_task2=env2_show,
        actor=actor,
        n_cols=7,
        figsize_per_frame=(1.5, 1.5),
        title=title,
        save_path=os.path.join(plots_dir, f"{cfg_name}_{actor_to_plot}_multitask.png")
    )

# %%
