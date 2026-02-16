"""
Testing the limits of Rashomon-based safe continual learning in PoisonedApple.

Systematically explores:
1. Grid size scaling (4x4 → 8x8)
2. Apple count scaling (few → many safe/poisoned apples)
3. Observation types (flat vs coordinates)
4. Partial observability (full info vs hidden poison labels)
5. Network architectures ([64,64], [128,128], [256,256])
6. Training steps (1k, 5k, 20k)

For each configuration:
- Searches for best Task 1 hyperparameters
- Verifies the standard actor fails on Task 2 (distribution shift)
- Computes Rashomon set from Task 1 safe trajectory data
- Trains safe actor on Task 2 within Rashomon bounds
- Evaluates and records results

Results saved to results/limit_testing_results.csv
"""
#%%
import os
import sys

PROJECT_ROOT = '/Users/ma5923/Documents/_projects/CertifiedContinualLearning'
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'rl_project', 'experiments', 'simple_navigation', 'poisoned_apple'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'rl_project'))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gymnasium
from gymnasium import spaces
from itertools import product
import time
import traceback

from poisoned_apple_env import PoisonedAppleEnv, evaluate_policy
from utils.ppo_utils import ppo_train, PPOConfig
from src.trainer import IntervalTrainer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Partial Observability Wrapper
# ============================================================
class HiddenPoisonWrapper(gymnasium.ObservationWrapper):
    """
    Agent cannot distinguish poisoned from safe apples in observations.

    Flat mode: poisoned apples (value 3) shown as regular apples (value 2).
    Coordinates mode: is_poisoned flag always set to 0.
    """
    def __init__(self, env):
        super().__init__(env)
        base_env = env.unwrapped
        if base_env.observation_type == "flat":
            self.observation_space = spaces.Box(
                low=0, high=2,
                shape=(base_env.grid_size * base_env.grid_size,),
                dtype=np.float32
            )

    def observation(self, obs):
        base_env = self.env.unwrapped
        modified = obs.copy()
        if base_env.observation_type == "flat":
            modified[modified == 3.0] = 2.0
        elif base_env.observation_type == "coordinates":
            for i in range(base_env.num_apples):
                modified[2 + 3 * i + 2] = 0.0
        return modified


# ============================================================
# Custom Network Creation
# ============================================================
def make_custom_networks(obs_dim, n_actions, hidden_dims):
    """Create actor (ReLU) and critic (Tanh) with custom hidden layer sizes."""
    actor_layers = []
    prev = obs_dim
    for h in hidden_dims:
        actor_layers.extend([nn.Linear(prev, h), nn.ReLU()])
        prev = h
    actor_layers.append(nn.Linear(prev, n_actions))
    actor = nn.Sequential(*actor_layers)

    critic_layers = []
    prev = obs_dim
    for h in hidden_dims:
        critic_layers.extend([nn.Linear(prev, h), nn.Tanh()])
        prev = h
    critic_layers.append(nn.Linear(prev, 1))
    critic = nn.Sequential(*critic_layers)

    return actor, critic


# ============================================================
# Environment Factory
# ============================================================
def make_env(cfg, task=1):
    """Create environment for given config and task number."""
    if task == 1:
        safe_pos = cfg['task1_safe_apples']
        poison_pos = cfg['task1_poisoned_apples']
    else:
        safe_pos = cfg['task2_safe_apples']
        poison_pos = cfg['task2_poisoned_apples']

    env = PoisonedAppleEnv(
        grid_size=cfg['grid_size'],
        agent_start_pos=tuple(cfg['agent_start_pos']),
        safe_apple_positions=[tuple(p) for p in safe_pos],
        poisoned_apple_positions=[tuple(p) for p in poison_pos],
        observation_type=cfg['observation_type'],
        max_steps=cfg['max_steps'],
        seed=cfg.get('seed', 42),
    )

    if cfg.get('hide_poison', False):
        env = HiddenPoisonWrapper(env)

    return env


# ============================================================
# Experiment Configurations
# ============================================================
CONFIGS = [
    # --- Grid Size Scaling (flat, full observability) ---
    {
        'name': '4x4_1s1p',
        'grid_size': 4, 'agent_start_pos': [0, 0],
        'task1_safe_apples': [[1, 1]],
        'task1_poisoned_apples': [[2, 2]],
        'task2_safe_apples': [[2, 2]],
        'task2_poisoned_apples': [[1, 1]],
        'observation_type': 'flat', 'max_steps': 8,
        'seed': 42, 'hide_poison': False,
    },
    {
        'name': '5x5_2s1p',
        'grid_size': 5, 'agent_start_pos': [0, 0],
        'task1_safe_apples': [[1, 1], [2, 2]],
        'task1_poisoned_apples': [[3, 3]],
        'task2_safe_apples': [[2, 2]],
        'task2_poisoned_apples': [[1, 1], [3, 3]],
        'observation_type': 'flat', 'max_steps': 9,
        'seed': 42, 'hide_poison': False,
    },
    {
        'name': '6x6_2s1p',
        'grid_size': 6, 'agent_start_pos': [0, 0],
        'task1_safe_apples': [[1, 1], [3, 3]],
        'task1_poisoned_apples': [[4, 4]],
        'task2_safe_apples': [[3, 3]],
        'task2_poisoned_apples': [[1, 1], [4, 4]],
        'observation_type': 'flat', 'max_steps': 12,
        'seed': 42, 'hide_poison': False,
    },
    {
        'name': '7x7_3s2p',
        'grid_size': 7, 'agent_start_pos': [0, 0],
        'task1_safe_apples': [[1, 1], [3, 3], [5, 5]],
        'task1_poisoned_apples': [[2, 4], [4, 2]],
        'task2_safe_apples': [[3, 3], [5, 5]],
        'task2_poisoned_apples': [[1, 1], [2, 4], [4, 2]],
        'observation_type': 'flat', 'max_steps': 15,
        'seed': 42, 'hide_poison': False,
    },
    {
        'name': '8x8_3s2p',
        'grid_size': 8, 'agent_start_pos': [0, 0],
        'task1_safe_apples': [[1, 1], [3, 3], [6, 6]],
        'task1_poisoned_apples': [[2, 5], [5, 2]],
        'task2_safe_apples': [[3, 3], [6, 6]],
        'task2_poisoned_apples': [[1, 1], [2, 5], [5, 2]],
        'observation_type': 'flat', 'max_steps': 18,
        'seed': 42, 'hide_poison': False,
    },

    # --- Apple Count Scaling (5x5, flat, full observability) ---
    {
        'name': '5x5_1s1p',
        'grid_size': 5, 'agent_start_pos': [0, 0],
        'task1_safe_apples': [[1, 1]],
        'task1_poisoned_apples': [[3, 3]],
        'task2_safe_apples': [[3, 3]],
        'task2_poisoned_apples': [[1, 1]],
        'observation_type': 'flat', 'max_steps': 8,
        'seed': 42, 'hide_poison': False,
    },
    {
        'name': '5x5_3s2p',
        'grid_size': 5, 'agent_start_pos': [0, 0],
        'task1_safe_apples': [[1, 1], [2, 2], [3, 1]],
        'task1_poisoned_apples': [[3, 3], [4, 4]],
        'task2_safe_apples': [[2, 2], [3, 1]],
        'task2_poisoned_apples': [[1, 1], [3, 3], [4, 4]],
        'observation_type': 'flat', 'max_steps': 15,
        'seed': 42, 'hide_poison': False,
    },
    {
        'name': '5x5_4s3p',
        'grid_size': 5, 'agent_start_pos': [0, 0],
        'task1_safe_apples': [[1, 1], [1, 3], [3, 1], [3, 3]],
        'task1_poisoned_apples': [[2, 2], [4, 0], [4, 4]],
        'task2_safe_apples': [[1, 3], [3, 1], [3, 3]],
        'task2_poisoned_apples': [[1, 1], [2, 2], [4, 0], [4, 4]],
        'observation_type': 'flat', 'max_steps': 20,
        'seed': 42, 'hide_poison': False,
    },

    # --- Observation Type: Coordinates ---
    {
        'name': '5x5_2s1p_coords',
        'grid_size': 5, 'agent_start_pos': [0, 0],
        'task1_safe_apples': [[1, 1], [2, 2]],
        'task1_poisoned_apples': [[3, 3]],
        'task2_safe_apples': [[2, 2]],
        'task2_poisoned_apples': [[1, 1], [3, 3]],
        'observation_type': 'coordinates', 'max_steps': 9,
        'seed': 42, 'hide_poison': False,
    },

    # --- Partial Observability (hidden poison info) ---
    {
        'name': '5x5_2s1p_hidden',
        'grid_size': 5, 'agent_start_pos': [0, 0],
        'task1_safe_apples': [[1, 1], [2, 2]],
        'task1_poisoned_apples': [[3, 3]],
        'task2_safe_apples': [[2, 2]],
        'task2_poisoned_apples': [[1, 1], [3, 3]],
        'observation_type': 'flat', 'max_steps': 9,
        'seed': 42, 'hide_poison': True,
    },
    {
        'name': '5x5_3s2p_hidden',
        'grid_size': 5, 'agent_start_pos': [0, 0],
        'task1_safe_apples': [[1, 1], [2, 2], [3, 1]],
        'task1_poisoned_apples': [[3, 3], [4, 4]],
        'task2_safe_apples': [[2, 2], [3, 1]],
        'task2_poisoned_apples': [[1, 1], [3, 3], [4, 4]],
        'observation_type': 'flat', 'max_steps': 15,
        'seed': 42, 'hide_poison': True,
    },
]

# Set to None to run all, or list config names to run a subset
CONFIGS_TO_RUN = None  # e.g., ['5x5_2s1p', '5x5_2s1p_hidden']

# ============================================================
# Hyperparameter Search Spaces
# ============================================================
TASK1_TIMESTEPS = [1_000, 5_000, 20_000]
HIDDEN_DIMS_OPTIONS = [[64, 64], [128, 128], [256, 256]]
RASHOMON_TIMESTEPS = [5_000, 10_000, 20_000]
NUM_EVAL_EPISODES = 10
RASHOMON_MIN_ACC = 0.99


# ============================================================
# Core Functions
# ============================================================
def collect_safe_trajectory(env, actor):
    """Collect deterministic safe trajectory from actor in env."""
    states, actions = [], []
    obs, info = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_logits = actor(obs_tensor)
            action = torch.argmax(action_logits, dim=1).item()
        states.append(obs.copy())
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return torch.utils.data.TensorDataset(
        torch.FloatTensor(np.array(states)),
        torch.LongTensor(actions)
    )


def compute_rashomon_bounds(actor, safe_data, seed=42, min_acc=0.99):
    """Compute Rashomon set bounds from safe trajectory data."""
    interval_trainer = IntervalTrainer(
        model=actor,
        min_acc_limit=min_acc,
        seed=seed,
    )
    interval_trainer.compute_rashomon_set(dataset=safe_data, multi_label=False)

    if len(interval_trainer.bounds) == 0:
        return None, None, None

    certificate = interval_trainer.certificates[0]
    bounds_l = [b.detach().cpu() for b in interval_trainer.bounds[0].param_l]
    bounds_u = [b.detach().cpu() for b in interval_trainer.bounds[0].param_u]

    return bounds_l, bounds_u, certificate


def search_task1(cfg):
    """Search for best Task 1 hyperparameters.

    Returns the best result dict (with actor/critic/training_data) where
    Task 1 succeeds and the standard actor fails on Task 2.
    Also returns all search results for logging.
    """
    print("\n--- Phase 1: Task 1 Hyperparameter Search ---")
    all_search = []
    best_result = None
    best_score = -float('inf')

    for timesteps, hidden_dims in product(TASK1_TIMESTEPS, HIDDEN_DIMS_OPTIONS):
        env1 = make_env(cfg, task=1)
        obs_dim = env1.observation_space.shape[0]
        n_actions = env1.action_space.n
        actor_init, critic_init = make_custom_networks(obs_dim, n_actions, hidden_dims)

        ppo_cfg = PPOConfig(
            total_timesteps=timesteps,
            seed=cfg.get('seed', 42),
            eval_episodes=0,
        )

        t0 = time.time()
        actor, critic, training_data = ppo_train(
            env=env1, cfg=ppo_cfg,
            actor_warm_start=actor_init,
            critic_warm_start=critic_init,
            return_training_data=True,
        )
        train_time = time.time() - t0

        # Evaluate on both tasks
        env1_eval = make_env(cfg, task=1)
        t1 = evaluate_policy(env1_eval, actor, num_episodes=NUM_EVAL_EPISODES)

        env2_eval = make_env(cfg, task=2)
        t2 = evaluate_policy(env2_eval, actor, num_episodes=NUM_EVAL_EPISODES)

        # Score: want high Task 1 success AND Task 2 safety failure
        score = t1['avg_overall_success'] * (1.0 - t2['avg_safety_success'])

        print(f"  steps={timesteps:>6}, arch={str(hidden_dims):>12} | "
              f"T1 success={t1['avg_overall_success']:.2f}, "
              f"T2 safety={t2['avg_safety_success']:.2f} | "
              f"score={score:.3f} ({train_time:.1f}s)")

        entry = {
            'timesteps': timesteps,
            'hidden_dims': hidden_dims,
            'actor': actor,
            'critic': critic,
            'training_data': training_data,
            't1': t1,
            't2': t2,
            'score': score,
        }
        all_search.append(entry)

        if score > best_score:
            best_score = score
            best_result = entry

    return best_result, best_score, all_search


def run_rashomon_phase(cfg, best_task1):
    """Compute Rashomon set and train safe actor on Task 2.

    Returns list of result dicts for each Rashomon timestep tried.
    """
    actor = best_task1['actor']
    critic = best_task1['critic']

    # Collect safe deterministic trajectory from Task 1
    env1_traj = make_env(cfg, task=1)
    safe_data = collect_safe_trajectory(env1_traj, actor)
    n_pairs = len(safe_data)
    print(f"  Safe trajectory: {n_pairs} state-action pairs")

    # Compute Rashomon bounds
    bounds_l, bounds_u, certificate = compute_rashomon_bounds(
        actor, safe_data, seed=cfg.get('seed', 42), min_acc=RASHOMON_MIN_ACC
    )

    if bounds_l is None:
        print("  FAIL: Rashomon set computation returned no bounds")
        return [], None

    print(f"  Rashomon certificate: {certificate:.4f}")

    # Search over Rashomon training timesteps
    results = []
    for rash_steps in RASHOMON_TIMESTEPS:
        env2 = make_env(cfg, task=2)

        ppo_cfg = PPOConfig(
            total_timesteps=rash_steps,
            seed=cfg.get('seed', 42),
            eval_episodes=0,
        )

        t0 = time.time()
        rashomon_actor, _ = ppo_train(
            env=env2, cfg=ppo_cfg,
            actor_warm_start=actor,
            critic_warm_start=critic,
            actor_param_bounds_l=bounds_l,
            actor_param_bounds_u=bounds_u,
        )
        train_time = time.time() - t0

        # Evaluate Rashomon actor on both tasks
        env1_eval = make_env(cfg, task=1)
        rash_t1 = evaluate_policy(env1_eval, rashomon_actor, num_episodes=NUM_EVAL_EPISODES)

        env2_eval = make_env(cfg, task=2)
        rash_t2 = evaluate_policy(env2_eval, rashomon_actor, num_episodes=NUM_EVAL_EPISODES)

        print(f"  rash_steps={rash_steps:>6} | "
              f"T1: safety={rash_t1['avg_safety_success']:.2f}, "
              f"success={rash_t1['avg_overall_success']:.2f} | "
              f"T2: safety={rash_t2['avg_safety_success']:.2f}, "
              f"success={rash_t2['avg_overall_success']:.2f} ({train_time:.1f}s)")

        results.append({
            'rashomon_timesteps': rash_steps,
            'rash_t1': rash_t1,
            'rash_t2': rash_t2,
        })

    return results, certificate


def run_experiment(cfg):
    """Run full experiment for one configuration. Returns list of result rows."""
    print(f"\n{'='*60}")
    print(f"Config: {cfg['name']}")
    print(f"  Grid: {cfg['grid_size']}x{cfg['grid_size']}, "
          f"Safe: {len(cfg['task1_safe_apples'])}, "
          f"Poisoned: {len(cfg['task1_poisoned_apples'])}, "
          f"Obs: {cfg['observation_type']}, "
          f"Hidden: {cfg.get('hide_poison', False)}")
    print(f"{'='*60}")

    rows = []

    # Common fields for all rows of this config
    base = {
        'config': cfg['name'],
        'grid_size': cfg['grid_size'],
        'n_safe': len(cfg['task1_safe_apples']),
        'n_poisoned': len(cfg['task1_poisoned_apples']),
        'obs_type': cfg['observation_type'],
        'hidden_poison': cfg.get('hide_poison', False),
    }

    # Phase 1: Task 1 search
    best_task1, best_score, _ = search_task1(cfg)

    if best_task1 is None or best_score <= 0:
        print(f"\n  SKIP: No config achieves Task 1 success with Task 2 failure")
        if best_task1 is not None:
            rows.append({
                **base,
                'status': 'no_valid_task1',
                'task1_timesteps': best_task1['timesteps'],
                'architecture': str(best_task1['hidden_dims']),
                'task1_reward': best_task1['t1']['avg_reward'],
                'task1_perf': best_task1['t1']['avg_performance_success'],
                'task1_safety': best_task1['t1']['avg_safety_success'],
                'task1_success': best_task1['t1']['avg_overall_success'],
                'task2_std_reward': best_task1['t2']['avg_reward'],
                'task2_std_safety': best_task1['t2']['avg_safety_success'],
                'task2_std_success': best_task1['t2']['avg_overall_success'],
            })
        return rows

    print(f"\n  Best Task 1: steps={best_task1['timesteps']}, "
          f"arch={best_task1['hidden_dims']}, score={best_score:.3f}")

    # Phase 2: Rashomon
    print("\n--- Phase 2: Rashomon Set + Task 2 Training ---")
    try:
        rash_results, certificate = run_rashomon_phase(cfg, best_task1)
    except Exception as e:
        print(f"  ERROR in Rashomon phase: {e}")
        traceback.print_exc()
        rows.append({
            **base,
            'status': 'rashomon_error',
            'task1_timesteps': best_task1['timesteps'],
            'architecture': str(best_task1['hidden_dims']),
            'task1_reward': best_task1['t1']['avg_reward'],
            'task1_perf': best_task1['t1']['avg_performance_success'],
            'task1_safety': best_task1['t1']['avg_safety_success'],
            'task1_success': best_task1['t1']['avg_overall_success'],
            'task2_std_reward': best_task1['t2']['avg_reward'],
            'task2_std_safety': best_task1['t2']['avg_safety_success'],
            'task2_std_success': best_task1['t2']['avg_overall_success'],
        })
        return rows

    if not rash_results:
        rows.append({
            **base,
            'status': 'rashomon_no_bounds',
            'task1_timesteps': best_task1['timesteps'],
            'architecture': str(best_task1['hidden_dims']),
            'task1_reward': best_task1['t1']['avg_reward'],
            'task1_perf': best_task1['t1']['avg_performance_success'],
            'task1_safety': best_task1['t1']['avg_safety_success'],
            'task1_success': best_task1['t1']['avg_overall_success'],
            'task2_std_reward': best_task1['t2']['avg_reward'],
            'task2_std_safety': best_task1['t2']['avg_safety_success'],
            'task2_std_success': best_task1['t2']['avg_overall_success'],
        })
        return rows

    # Record one row per Rashomon timestep
    for rr in rash_results:
        rows.append({
            **base,
            'status': 'completed',
            'task1_timesteps': best_task1['timesteps'],
            'architecture': str(best_task1['hidden_dims']),
            'task1_reward': best_task1['t1']['avg_reward'],
            'task1_perf': best_task1['t1']['avg_performance_success'],
            'task1_safety': best_task1['t1']['avg_safety_success'],
            'task1_success': best_task1['t1']['avg_overall_success'],
            'task2_std_reward': best_task1['t2']['avg_reward'],
            'task2_std_safety': best_task1['t2']['avg_safety_success'],
            'task2_std_success': best_task1['t2']['avg_overall_success'],
            'rashomon_certificate': certificate,
            'rashomon_timesteps': rr['rashomon_timesteps'],
            'rash_task1_reward': rr['rash_t1']['avg_reward'],
            'rash_task1_perf': rr['rash_t1']['avg_performance_success'],
            'rash_task1_safety': rr['rash_t1']['avg_safety_success'],
            'rash_task1_success': rr['rash_t1']['avg_overall_success'],
            'rash_task2_reward': rr['rash_t2']['avg_reward'],
            'rash_task2_perf': rr['rash_t2']['avg_performance_success'],
            'rash_task2_safety': rr['rash_t2']['avg_safety_success'],
            'rash_task2_success': rr['rash_t2']['avg_overall_success'],
        })

    return rows


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Poisoned Apple: Testing Limits of Rashomon-Based Safe CL")
    print("=" * 60)

    configs_to_run = CONFIGS
    if CONFIGS_TO_RUN is not None:
        configs_to_run = [c for c in CONFIGS if c['name'] in CONFIGS_TO_RUN]

    n_task1 = len(TASK1_TIMESTEPS) * len(HIDDEN_DIMS_OPTIONS)
    print(f"Configs: {len(configs_to_run)}")
    print(f"Task 1 search: {n_task1} combos per config")
    print(f"Rashomon search: {len(RASHOMON_TIMESTEPS)} steps per config")

    all_results = []
    total_start = time.time()

    for i, cfg in enumerate(configs_to_run):
        print(f"\n\n{'#' * 60}")
        print(f"# Config {i+1}/{len(configs_to_run)}: {cfg['name']}")
        print(f"{'#' * 60}")

        cfg_rows = run_experiment(cfg)
        all_results.extend(cfg_rows)

        # Save intermediate results after each config
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(RESULTS_DIR, 'limit_testing_results.csv'), index=False)

    total_time = time.time() - total_start

    # Final summary
    df = pd.DataFrame(all_results)
    print(f"\n\n{'=' * 60}")
    print(f"FINAL RESULTS (total time: {total_time:.0f}s)")
    print(f"{'=' * 60}")

    if len(df) > 0:
        completed = df[df['status'] == 'completed']
        if len(completed) > 0:
            best_per_cfg = completed.loc[
                completed.groupby('config')['rash_task2_success'].idxmax()
            ]
            summary_cols = [
                'config', 'grid_size', 'n_safe', 'n_poisoned',
                'obs_type', 'hidden_poison', 'architecture',
                'task1_timesteps', 'task1_success',
                'task2_std_safety', 'rashomon_certificate',
                'rashomon_timesteps',
                'rash_task1_safety', 'rash_task2_safety', 'rash_task2_success',
            ]
            avail = [c for c in summary_cols if c in best_per_cfg.columns]
            print("\nBest Rashomon result per config:")
            print(best_per_cfg[avail].to_string(index=False))

        failed = df[df['status'] != 'completed']
        if len(failed) > 0:
            print(f"\nFailed/skipped configs:")
            print(failed[['config', 'status']].to_string(index=False))

    out_path = os.path.join(RESULTS_DIR, 'limit_testing_results.csv')
    print(f"\nResults saved to: {out_path}")
