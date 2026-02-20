"""
Poisoned Apple Environment: Safe Continual Learning with SafeAdapt

This script demonstrates safe continual learning in a gridworld environment where an 
agent must avoid poisoned apples while collecting safe ones. The key challenge is 
adapting to a distribution shift (Task 2) while maintaining safety on the original 
task (Task 1).

INPUTS:
-------
Configuration file: demo_configs.yaml
    - Environment parameters (grid size, apple positions, etc.)
    - Training hyperparameters (timesteps, learning rates, etc.)
    - Random seed for reproducibility

Script parameters (lines 350-353):
    - cfg_name: Configuration name from YAML file (e.g., 'simple_5x5')
    - safe_state_action_data_name: Safety dataset type
        * 'Safe Optimal Policy Data': Deterministic trajectory demonstrations
        * 'Safe Training Data': Filtered safe state-action pairs from training
        * 'Tabular Safety Critic Data': Exhaustive safe actions for all states
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
   - Expected: Safe on Task 1, unsafe on Task 2

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
- PyTorch, NumPy, Pandas, Matplotlib, PyYAML
- Custom modules: poisoned_apple_env, ppo_utils, IntervalTrainer
"""

#%%
# =============================================================================
# IMPORTS
# =============================================================================
import os
import random
import gymnasium
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from poisoned_apple_env import (
    PoisonedAppleEnv, 
    evaluate_policy,
    get_all_unsafe_state_action_pairs,
    visualize_agent_trajectory
)
from rl_project.utils.ppo_utils import ppo_train, PPOConfig
from src.trainer import IntervalTrainer

# Set paths
current_script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(current_script_dir, 'plots')
tables_dir = os.path.join(current_script_dir, 'tables')

# Create output directories
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

#%%
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_safe_optimal_policy_dataset(env, actor, num_rollouts, deterministic: bool = True, seed: int = 42):
    """
    Create a safety dataset from rollouts of an optimal policy.
    Collects state-action pairs by running the actor in the environment.
    
    Args:
        env: The environment to collect data from
        actor: The policy network to generate actions
        num_rollouts: Number of episodes to collect
        deterministic: Whether to use deterministic actions (argmax) or sample from the policy distribution
        
    Returns:
        A torch TensorDataset containing (states, actions) pairs from deterministic rollouts
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
            obs, reward, terminated, truncated, info = env.step(action)  # type: ignore
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

def generate_sufficient_safe_state_action_dataset(
    unsafe_state_action_pairs: list[tuple], env: gymnasium.Env
):
    """
    Generate a dataset of sufficient safe state-action pairs by computing the complement of unsafe actions for each state.
    This dataset can be used to compute a Rashomon set that enforces safety constraints without requiring an optimal policy demonstration.
    
    Args:
        unsafe_state_action_pairs: list of (state, action) tuples that are unsafe in Env 1. 
        These can be collected by running random rollouts in Env 1 and recording state-action pairs where the safety flag is False.
        env: The environment, needed to determine the action space for computing the complement set of safe actions.
    Returns:
        A torch dataset containing states and their corresponding sets of safe actions (padded for variable length)
    """
    
    # Given unsafe state-action pairs, we can generate the complement set of actions for each state
    all_actions = set(range(env.action_space.n)) # type: ignore NOTE: works for discrete action spaces
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

    # Each action row contains valid class indices (padded with -1 for variable length)
    max_actions = env.action_space.n # type: ignore
    padded_safe_actions = []
    for state_key, safe_actions in safe_actions_by_state.items():
        padded_actions = list(safe_actions) + [-1] * (max_actions - len(safe_actions))
        padded_safe_actions.append(padded_actions)
    sufficient_safe_actions = torch.LongTensor([list(actions) for actions in padded_safe_actions])
    sufficient_safe_state_action_torch_dataset = torch.utils.data.TensorDataset(sufficient_safe_states, sufficient_safe_actions)
    
    return sufficient_safe_state_action_torch_dataset

def set_all_seeds(seed):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
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
cfg_name = 'simple_5x5'
safe_state_action_data_name = 'Sufficient Safety Data' # Options: 'Sufficient Safety Data', 'Safe Optimal Policy Data', 'Safe Training Data', 'Tabular Safety Critic Data'
save_results = False  # Set to True to save plots and tables
train_unsafe_adapt = False  # Set to True to train UnsafeAdapt baseline (takes additional time)

# -------------------------
# Load configuration from YAML
# -------------------------
with open('demo_configs.yaml', 'r') as f:
    DEMO_CONFIGS = yaml.safe_load(f)
cfg = DEMO_CONFIGS[cfg_name]

print("="*80)
print("POISONED APPLE SAFE CONTINUAL LEARNING EXPERIMENT")
print("="*80)
print(f"Configuration: {cfg_name}")
print(f"Safety dataset: {safe_state_action_data_name}")
print(f"Save results: {save_results}")
print(f"Train UnsafeAdapt: {train_unsafe_adapt}")
print("="*80 + "\n")

# -------------------------
# Extract configuration parameters
# -------------------------
# General environment parameters
grid_size = cfg['grid_size']
agent_start_pos = tuple(cfg['agent_start_pos'])
observation_type = cfg['observation_type']
max_steps = cfg['max_steps']
safe_env1_state_action_data_num_rollouts = cfg['safe_env1_state_action_data_num_rollouts']
seed = cfg['seed']

# Task 1 (Env1) configuration
env1_safe_apple_positions = [tuple(pos) for pos in cfg['env1_safe_apples']]
env1_poisoned_apple_positions = [tuple(pos) for pos in cfg['env1_poisoned_apples']]

# Task 2 (Env2) configuration  
env2_safe_apple_positions = [tuple(pos) for pos in cfg['env2_safe_apples']]
env2_poisoned_apple_positions = [tuple(pos) for pos in cfg['env2_poisoned_apples']]

# Training hyperparameters
no_adapt_timesteps = cfg['unadaptable_actor_timesteps']
task2_adapt_timesteps = cfg['task2_adapt_timesteps']
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
env = PoisonedAppleEnv(
    grid_size=grid_size,
    agent_start_pos=agent_start_pos,
    safe_apple_positions=env1_safe_apple_positions,
    poisoned_apple_positions=env1_poisoned_apple_positions,
    observation_type=observation_type,
    render_mode="human",
    max_steps=max_steps,
    seed=seed
)

print(f"Environment created: {grid_size}x{grid_size} grid")
print(f"Safe apples: {env1_safe_apple_positions}")
print(f"Poisoned apples: {env1_poisoned_apple_positions}")

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
    env=env,
    cfg=ppo_cfg,
    return_training_data=True
)
print(f"NoAdapt training complete ({no_adapt_timesteps} timesteps)")

#%%
# Visualize NoAdapt on Task 1
print("\nVisualizing NoAdapt on Task 1...")
visualize_agent_trajectory(
    env, standard_actor, num_episodes=1, max_steps=max_steps, 
    env_name='Task 1', cfg_name=cfg_name, actor_name='NoAdapt', save_dir=plots_dir
)

#%%
# =============================================================================
# TASK 2 ENVIRONMENT SETUP  
# =============================================================================

print("\n" + "="*80)
print("TASK 2: Distribution shift (poisoned apple location changed)")
print("="*80)

env2 = PoisonedAppleEnv(
    grid_size=grid_size,
    agent_start_pos=agent_start_pos,
    safe_apple_positions=env2_safe_apple_positions,
    poisoned_apple_positions=env2_poisoned_apple_positions,
    observation_type=observation_type,
    render_mode="human",
    max_steps=max_steps,
    seed=seed
)

print(f"Task 2 environment created: {grid_size}x{grid_size} grid")
print(f"Safe apples: {env2_safe_apple_positions}")
print(f"Poisoned apples: {env2_poisoned_apple_positions}")
print("Distribution shift: One safe apple became poisoned")

# Visualize NoAdapt on Task 2 (expected to fail)
print("\nVisualizing NoAdapt on Task 2 (expected catastrophic failure)...")
visualize_agent_trajectory(
    env2, standard_actor, num_episodes=1, max_steps=max_steps, 
    env_name='Task 2', cfg_name=cfg_name, actor_name='NoAdapt',
    save_dir=plots_dir
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
    visualize_agent_trajectory(
        env, amnesic_actor, num_episodes=1, max_steps=max_steps, 
        env_name='Task 1', cfg_name=cfg_name, actor_name='UnsafeAdapt',
        save_dir=plots_dir
    )

    # Visualize UnsafeAdapt on Task 2 (expected good performance)
    print("\nVisualizing UnsafeAdapt on Task 2 (expected good performance)...")
    visualize_agent_trajectory(
        env2, amnesic_actor, num_episodes=1, max_steps=max_steps, 
        env_name='Task 2', cfg_name=cfg_name, actor_name='UnsafeAdapt',
        save_dir=plots_dir
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
    env=env,
    actor=standard_actor,
    num_rollouts=safe_env1_state_action_data_num_rollouts
)

safe_training_data = create_safe_training_dataset(
    training_data=standard_training_data
)

unsafe_state_action_pairs = get_all_unsafe_state_action_pairs(env=env)
sufficient_safe_state_action_torch_dataset = generate_sufficient_safe_state_action_dataset(
    unsafe_state_action_pairs=unsafe_state_action_pairs,
    env=env
)

# Collect all datasets
safe_datasets = {
    'Safe Optimal Policy Data': safe_optimal_policy_data,
    'Safe Training Data': safe_training_data,
    'Sufficient Safety Data': sufficient_safe_state_action_torch_dataset
}

print(f"\nUsing dataset: {safe_state_action_data_name}")
print(f"Dataset size: {len(safe_datasets[safe_state_action_data_name])} state-action pairs")

# %%
# =============================================================================
# COMPUTE SAFEADAPT SET (Parameter Constraints)
# =============================================================================

print("\n" + "="*80)
print("Computing SafeAdapt set via interval bound propagation")
print("="*80)

state_action_dataset = safe_datasets[safe_state_action_data_name]

# Ensure model is on CPU
standard_actor_cpu = standard_actor.cpu()

# Initialize IntervalTrainer for computing parameter bounds
min_acc_limit = 1.0  # Require 100% accuracy on the safety dataset for certification
interval_trainer = IntervalTrainer(
    model=standard_actor_cpu,
    min_acc_limit=min_acc_limit,  # Require min_acc_limit accuracy on the safety dataset for certification
    seed=seed,
    n_iters=2_000, # Number of iterations for interval bound optimization (can be adjusted for faster computation during demos
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
    total_timesteps=safe_adapt_timesteps,
    device='cpu'
)
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
visualize_agent_trajectory(
    env, safeadapt_actor, num_episodes=1, max_steps=max_steps, 
    env_name='Task 1', cfg_name=cfg_name, 
    actor_name=f'SafeAdapt ({safe_state_action_data_name})', 
    save_dir=plots_dir
)

# Visualize SafeAdapt on Task 2 (expected: adapt successfully)
print("\nVisualizing SafeAdapt on Task 2 (expected: adapt successfully)...")
visualize_agent_trajectory(
    env2, safeadapt_actor, num_episodes=1, max_steps=max_steps, 
    env_name='Task 2', cfg_name=cfg_name, 
    actor_name=f'SafeAdapt ({safe_state_action_data_name})', 
    save_dir=plots_dir
)

#%%
### Verify accuracy of the Rashomon actor on the safe state-action dataset
rashomon_actor_actions = safeadapt_actor(state_action_dataset.tensors[0]).argmax(axis=1) # type: ignore
if multi_label:
    # For multi-label datasets, we need to check if the predicted action is in the set of valid actions for each state
    valid_actions = state_action_dataset.tensors[1]  # shape: (num_samples, max_actions)
    rashomon_actor_actions_expanded = rashomon_actor_actions.unsqueeze(1).expand_as(valid_actions)  # shape: (num_samples, max_actions)
    matches = (rashomon_actor_actions_expanded == valid_actions)  # shape: (num_samples, max_actions), bool tensor
    valid_action_mask = (valid_actions != -1)  # shape: (num_samples, max_actions), bool tensor
    correct_predictions = (matches & valid_action_mask).any(dim=1).sum().item()  # count samples where at least one valid action matches
    accuracy = correct_predictions / len(rashomon_actor_actions)
else:
    accuracy = (rashomon_actor_actions == state_action_dataset.tensors[1]).sum().item() / len(rashomon_actor_actions)

accuracy_validated = accuracy >= min_acc_limit
if accuracy_validated:
    print(f"Certified accuracy validated: {accuracy:.2f} >= {min_acc_limit:.2f}")
else:
    raise ValueError(f"Certified accuracy validation failed: {accuracy:.2f} < {min_acc_limit:.2f}")

# %%
# =============================================================================
# FINAL EVALUATION AND RESULTS
# =============================================================================

print("\n" + "="*80)
print("FINAL EVALUATION: Comparing all trained policies")
print("="*80)

# Evaluate all policies on both tasks
num_eval_episodes = 1  # Deterministic environments and policies

print("\nEvaluating NoAdapt...")
noadapt_task1_metrics = evaluate_policy(env, standard_actor, num_episodes=num_eval_episodes)
noadapt_task2_metrics = evaluate_policy(env2, standard_actor, num_episodes=num_eval_episodes)

# Conditionally evaluate UnsafeAdapt
if train_unsafe_adapt:
    print("Evaluating UnsafeAdapt...")
    unsafeadapt_task1_metrics = evaluate_policy(env, amnesic_actor, num_episodes=num_eval_episodes)
    unsafeadapt_task2_metrics = evaluate_policy(env2, amnesic_actor, num_episodes=num_eval_episodes)

print("Evaluating SafeAdapt...")
safeadapt_task1_metrics = evaluate_policy(env, safeadapt_actor, num_episodes=num_eval_episodes)
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
# LATEX TABLE GENERATION
# =============================================================================

def generate_latex_table(df, caption="Results", label="tab:results"):
    """
    Generate LaTeX table from pandas DataFrame.
    
    Args:
        df: pandas DataFrame with results
        caption: Table caption for LaTeX
        label: Table label for LaTeX references
    
    Returns:
        str: LaTeX table code
    """
    # Key metrics to include in LaTeX table
    key_metrics = [
        'avg_reward',
        'avg_safety_success',
        'avg_poisoned_apples_collected',
        'avg_safe_apples_collected'
    ]
    
    # Filter to key metrics
    df_filtered = df.loc[key_metrics]
    
    # Create metric name mapping for better display
    metric_names = {
        'avg_reward': 'Avg. Reward',
        'avg_safety_success': 'Safety Rate',
        'avg_poisoned_apples_collected': 'Poisoned Collected',
        'avg_safe_apples_collected': 'Safe Collected'
    }
    
    # Rename index
    df_filtered.index = [metric_names.get(idx, idx) for idx in df_filtered.index]
    
    # Generate LaTeX
    latex_str = "\\begin{table}[htbp]\n"
    latex_str += "\\centering\n"
    latex_str += "\\caption{" + caption + "}\n"
    latex_str += "\\label{" + label + "}\n"
    latex_str += "\\begin{tabular}{l" + "c" * len(df_filtered.columns) + "}\n"
    latex_str += "\\toprule\n"
    
    # Header
    latex_str += "Metric"
    for col in df_filtered.columns:
        # Split column name for better formatting
        parts = col.split(" / ")
        if len(parts) == 2:
            method, task = parts
            latex_str += f" & \\multicolumn{{1}}{{c}}{{{method} / {task}}}"
        else:
            latex_str += f" & \\multicolumn{{1}}{{c}}{{{col}}}"
    latex_str += " \\\\\n"
    latex_str += "\\midrule\n"
    
    # Data rows
    for idx, row in df_filtered.iterrows():
        latex_str += str(idx)
        for val in row:
            # Format based on metric type
            if 'Safety Rate' in str(idx):
                latex_str += f" & {val:.3f}"
            elif 'Collected' in str(idx):
                latex_str += f" & {val:.2f}"
            else:
                latex_str += f" & {val:.3f}"
        latex_str += " \\\\\n"
    
    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{table}\n"
    
    return latex_str

# Generate and print LaTeX tables
print("\n" + "="*80)
print("LATEX TABLE CODE (for paper)")
print("="*80)

methods_str = "NoAdapt and SafeAdapt"
if train_unsafe_adapt:
    methods_str = "NoAdapt, UnsafeAdapt, and SafeAdapt"

# latex_table = generate_latex_table(
#     results_df,
#     caption=f"Performance comparison of {methods_str} on Tasks 1 and 2. "
#             f"SafeAdapt uses {safe_state_action_data_name} for constraint generation. "
#             f"Safety Rate indicates fraction of episodes without poisoned apple consumption.",
#     label=f"tab:results_{cfg_name}"
# )
# print("\n" + latex_table)

# # Save LaTeX table to file
# if save_results and tables_dir is not None:
#     latex_path = os.path.join(tables_dir, f'latex_table_{cfg_name}_{safe_state_action_data_name.replace(" ", "_")}.tex')
#     with open(latex_path, 'w') as f:
#         f.write(latex_table)
#     print(f"\nLaTeX table saved to: {latex_path}")

#%%
# =============================================================================
# VERIFICATION AND ASSERTIONS
# =============================================================================

print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

### Make sure the following is true:
# 1) NoAdapt should be unsafe on Task 2 (safety success < threshold)
# 2) UnsafeAdapt should show catastrophic forgetting on Task 1 (if trained) (safety success < threshold)
# 3) SafeAdapt should have perfect safety on Task 1 (safety success = 1.0 due to certification) and high safety on Task 2 (safety success >= threshold)

# Check expected behaviors
# Note: safety_threshold is used to verify unsafe behaviors (NoAdapt, UnsafeAdapt)
# SafeAdapt on Task 1 should have perfect safety (1.0) due to min_acc_limit=1.0 certification
task2_safety_threshold = 0.99

print("\nVerifying expected behaviors:")

# NoAdapt should be unsafe on Task 2
# noadapt_task2_safe = results_df.loc['avg_safety_success', 'NoAdapt / Task 2']
# print(f"✓ NoAdapt unsafe on Task 2: {noadapt_task2_safe < task2_safety_threshold} (safety={noadapt_task2_safe:.3f})")

# # UnsafeAdapt should show catastrophic forgetting on Task 1 (if trained)
# if train_unsafe_adapt:
#     unsafeadapt_task1_safe = results_df.loc['avg_safety_success', 'UnsafeAdapt / Task 1']
#     print(f"✓ UnsafeAdapt shows forgetting on Task 1: {unsafeadapt_task1_safe < task2_safety_threshold} (safety={unsafeadapt_task1_safe:.3f})")

# SafeAdapt should have perfect safety on Task 1 (certified) and high safety on Task 2 (generalization)
safeadapt_task1_safe = results_df.loc['avg_safety_success', 'SafeAdapt / Task 1']
safeadapt_task2_safe = results_df.loc['avg_safety_success', 'SafeAdapt / Task 2']
print(f"✓ SafeAdapt perfect safety on Task 1: {safeadapt_task1_safe > min_acc_limit} (safety={safeadapt_task1_safe:.3f}, certified)")
# print(f"✓ SafeAdapt safe on Task 2: {safeadapt_task2_safe >= task2_safety_threshold} (safety={safeadapt_task2_safe:.3f}, generalization)")

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