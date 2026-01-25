
########################################################################################
### TODO: this is the final script which allows to run any experiment end-to-end #######
########################################################################################

### Imports
import gymnasium as gym
import torch
import numpy as np
import random
import argparse
import time
import copy
import pandas as pd

from utils.ppo_utils import PPOConfig, ppo_train, evaluate
from utils.custom_envs import CustomCartPoleEnv
from utils.safety_critic_utils import collect_traces_dataset, SafetyCritic

AVAILABLE_ENVS = ['CustomCartPole-v1', 'SafeGridWorld-v0']

if __name__ == "__main__":

    ### Arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--env', type=str, choices=AVAILABLE_ENVS, required=True,
    #     help='Environment to run the experiment on'
    # )
    # args = parser.parse_args()
    # env_name = args.env
    # General:
    env_name = 'CustomCartPole-v1'
    random_seed = 2025
    # Task 1 training
    task1_total_train_timesteps = 50_000
    # Safety critic hyperaprams:
    safety_demonstrations_num_episodes = 1_000
    safety_demonstrations_epsilon = 0.1
    safety_critic_batch_size = 4096
    safety_critic_td_epochs = 100
    safety_critic_lr = 1e-3
    safety_critic_epsilon_collect = 0.2
    safety_critic_num_episodes_collect = 400
    gamma_safe = 0.99
    # Task 2 hyperparams:
    
    ### Prelims
    # Fix random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    
    ### 0. Train a PPO policy on Task 1
    task1_env = gym.make(env_name)
    if isinstance(task1_env.action_space, gym.spaces.Discrete):
        num_actions = task1_env.action_space.n
    else:
        num_actions = task1_env.action_space.shape[0] # type: ignore
    if isinstance(task1_env.observation_space, gym.spaces.Discrete):
        num_obs = task1_env.observation_space.n
    else:
        num_obs = task1_env.observation_space.shape[0] # type: ignore
    print(f"Environment: {env_name}, num_actions={num_actions}, num_obs={num_obs}")

    print("\n=== Step 0: Train PPO Policy on Task 1 ===")
    task1_cfg = PPOConfig(
        total_timesteps=task1_total_train_timesteps,
        seed=random_seed
    )
    task1_actor, task1_critic = ppo_train(env=task1_env, cfg=task1_cfg) # NOTE: includes final evaluation
    original_policy_task1_avg_reward, original_policy_task1_std_reward, original_policy_task1_failure_rate =\
        evaluate(env=task1_env, actor=task1_actor, device=device, seed=random_seed)

    ### 1. Train a safety critic on Task 1
    # Collect diverse data from task 1
    print("\n=== Step 1: Train Safety Critic on Task 1 ===")
    obs_arr, action_arr, next_obs_arr, failure_arr, done_arr = collect_traces_dataset(
        policy_model=task1_actor, env=task1_env, 
        num_episodes=safety_demonstrations_num_episodes, 
        epsilon=safety_demonstrations_epsilon, 
        device=device, seed=random_seed
    )
    print(f'Collected safety dataset. Failure rate: {np.sum(failure_arr) / safety_demonstrations_num_episodes}')

    # Train the safety critic
    print('Safety Critic training...')
    # TD training
    S_t = torch.from_numpy(obs_arr).to(device)
    A_t = torch.from_numpy(action_arr).to(device)
    SP_t = torch.from_numpy(next_obs_arr).to(device)
    D_t = torch.from_numpy(done_arr).to(device)
    F_t = torch.from_numpy(failure_arr).to(device)


    safety_critic_in_dim = S_t.shape[1]
    if isinstance(task1_env.action_space, gym.spaces.Discrete):
        safety_critic_in_dim += 1
    else:
        safety_critic_in_dim += task1_env.action_space.shape[0] # type: ignore
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
            if isinstance(task1_env.action_space, gym.spaces.Discrete):
                with torch.no_grad():
                    ### For SB3 model
                    # ap_b_np, _ = model.predict(sp_b.cpu().numpy(), deterministic=True)
                    ### For my PPO from scratch
                    cur_sp_tensor = torch.tensor(sp_b, dtype=torch.float32).unsqueeze(0).to(device)
                    logits = task1_actor.forward(cur_sp_tensor)
                    ap_b_np = logits.argmax(dim=-1).cpu().numpy()
                ap_b = torch.from_numpy(ap_b_np).to(device).unsqueeze(-1)
            else:
                with torch.no_grad():
                    ### For SB3 model
                    # ap_b_np, _ = model.predict(sp_b.cpu().numpy(), deterministic=True)
                    ### For my PPO from scratch
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
        if isinstance(task1_env.action_space, gym.spaces.Discrete):
            action_tensor = torch.from_numpy(action_arr).to(device).unsqueeze(-1)
        else:
            action_tensor = torch.from_numpy(action_arr).to(device)
        state_action_tensor = torch.cat([S_t, action_tensor], dim=1)
        logits = safety_critic(state_action_tensor)
        preds = (torch.sigmoid(logits) >= 0.5).float() # whether next state is a failure state
        accuracy = (preds.squeeze() == F_t).float().mean().item()
        print(f"Safety Critic training accuracy on collected data: {accuracy:.4f}")

    ### 2. Task 2
    # a) Define the task2 environment (adjusted more loose safety constraints!)
    task2_env = gym.make(env_name, reward_shaping="aggressive_movement")
   
    # b) Use that task1 policy for task2 (expected to be suboptimal)
    original_policy_task2_avg_reward, original_policy_task2_std_reward, original_policy_task2_failure_rate =\
        evaluate(env=task2_env, actor=task1_actor, device=device, seed=random_seed)
    print(
        f"Original policy on task 2 - Avg Reward: {original_policy_task2_avg_reward}, "
        f"Std Reward: {original_policy_task2_std_reward}, "
        f"Failure Rate: {original_policy_task2_failure_rate}"
    )

    # c) Adapt the task1 policy using PPO (expected to be good but unsafe for task1)
    task2_total_train_timesteps = 50_000

    print("\n=== Train PPO Policy on Task 2 ===")
    task2_cfg = PPOConfig(
        total_timesteps=task2_total_train_timesteps,
        seed=random_seed
    )
    task2_actor, task2_critic = ppo_train(env=task2_env, cfg=task2_cfg)

    # d) Adapt the task1 policy using PPO + PGD on the Rashomon set
    # (expected to be safe for task1 and good for task2)
    # Use IntervalTrainer to create the Rashmonon set
    # Extract the Rashmon set bounds
    # Use PGD and PPO to adapt the policy within the Rashmon set
    print("\n=== Train PPO + PGD Policy on Task 2 ===")

    # --- Collect safe actions demonstrations ---
     ### a) Collect a large exploratory CartPole dataset using the trained policy with epsilon-greedy exploration
    episodes_exp = 1500   # increase for more data
    epsilon_exp = 0.5    # higher -> more random actions

    S_exp, A_exp, SP_exp, F_exp, D_exp = collect_traces_dataset(
        policy_model=task1_actor,
        env=task1_env, # NOTE: task 1!
        num_episodes=episodes_exp,
        epsilon=0.0 # NOTE: no exploration! 
    )

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

    # Optional: save to disk for reuse
    # np.savez_compressed(
    #     "cartpole_exploration_dataset.npz",
    #     S=S_exp, A=A_exp, SP=SP_exp, D=D_exp, F=F_exp, epsilon=epsilon_exp
    # )
    # print("Saved to cartpole_exploration_dataset.npz")

    ### b) Label the dataset with the trained safety critic
    # For each state in S_exp, find all discrete actions with failure probability < 0.5 (per safety_critic)
    # Determine the number of discrete actions

    safety_critic.eval()
    N = S_exp.shape[0]
    batch_size_eval = 32768  # adjust if needed
    max_failure_prob = 0.5 # threshold for safe actions

    safe_action_mask = np.zeros((N, num_actions), dtype=bool)
    safe_actions_per_state = []
    # final_safe_actions_per_state = []
    with torch.no_grad():
        for i in range(0, N, batch_size_eval):
            j = min(i + batch_size_eval, N)
            s_batch = torch.from_numpy(S_exp[i:j]).to(device=device, dtype=torch.float32)

            # Build (state, action) pairs for all discrete actions
            action_vals = torch.arange(num_actions, device=device, dtype=s_batch.dtype)  # [0..num_actions-1] # type: ignore
            s_expanded = s_batch.unsqueeze(1).expand(-1, num_actions, -1)                # (B, A, num_obs) # type: ignore
            a_expanded = action_vals.view(1, num_actions, 1).expand(s_batch.size(0), num_actions, 1)  # (B, A, 1)
            sa_flat = torch.cat([s_expanded, a_expanded], dim=2).reshape(-1, num_obs + 1)        # (B*A, num_obs+1) # type: ignore

            logits = safety_critic(sa_flat)                         # (B*A,)
            probs = torch.sigmoid(logits).reshape(-1, num_actions)    # (B, A) # type: ignore
            mask = (probs < max_failure_prob).cpu().numpy()                # (B, A) bool

            safe_action_mask[i:j] = mask
            safe_actions_per_state.extend([np.where(row)[0].tolist() for row in mask])  # type: ignore

    print(f"Computed safe actions for {N} states (threshold={max_failure_prob}, num_actions={num_actions}).")
    print('Distribution of number of safe actions per state:')
    print(pd.Series([len(lst) for lst in safe_actions_per_state]).value_counts().sort_index())

    ### c) Prepare safe action matrix where -1 is used for padding
    final_safe_actions_per_state = []
    for safe_action_indices in safe_actions_per_state:
        safe_action_indices = torch.tensor(safe_action_indices, dtype=torch.long)
        pad_len = max(0, num_actions - safe_action_indices.numel())
        if pad_len > 0:
            pad = torch.full((pad_len,), -1, dtype=safe_action_indices.dtype, device=safe_action_indices.device) # type: ignore
            cur_tensor = torch.cat([safe_action_indices, pad], dim=0)
        else:
            cur_tensor = safe_action_indices
        final_safe_actions_per_state.append(cur_tensor.numpy())

    ### 3. Final evaluation

    # Comare all metrics of the three policies on both tasks: performance and safety
    final_metrics = {
        'original_policy': {
            'task1': {
                'performance': original_policy_task1_avg_reward, 
                'failure_rate': original_policy_task1_failure_rate
            },
            'task2': {
                'performance': original_policy_task2_avg_reward, 
                'failure_rate': original_policy_task2_failure_rate
            }
        },
        'ppo_adapted_policy': {
            'task1': {
                'performance': None, 
                'safety': None
            },
            'task2': {
                'performance': None, 
                'safety': None
            }
        },
        'ppo_pgd_adapted_policy': {
            'task1': {'performance': None, 'safety': None},
            'task2': {'performance': None, 'safety': None}
        }
    }
    print(final_metrics)