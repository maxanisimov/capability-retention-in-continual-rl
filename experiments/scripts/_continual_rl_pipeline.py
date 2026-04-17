import sys
import os
import torch
import random
import time
import argparse
import copy
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
sns.set_style("whitegrid")

# Resolve repository root dynamically so the script is portable across machines.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from scripts._sqrl_pretrain import SQRLPretrainConfig, pretrain_sqrl, default_failure_fn
from scripts.custom_tasks import *
from src.trainer import IntervalTrainer
from torch.utils.data import TensorDataset, DataLoader
from stable_baselines3 import PPO

### Functions
def run_eval(policy_model, env, num_episodes=100, max_steps=100):
    policy_model.eval()
    total_rewards = []
    success_count = 0
    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0.0
            for _ in range(max_steps):
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                logits = policy_model(obs_tensor)
                action = torch.argmax(logits, dim=-1).item()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    if terminated and reward > 0:
                        success_count += 1
                    break
            total_rewards.append(episode_reward)
    avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
    success_rate = success_count / num_episodes if num_episodes > 0 else 0.0
    return avg_reward, success_rate

name_to_task_dct = {
    'drunk spider': DrunkSpiderEnv(render_mode='rgb_array'),
    'minitaur': None, # TODO
    'cube rotation': None, # TODO
}
name_to_failure_fn_dct = {
    'drunk spider': lambda obs, action, next_obs, info: bool(info['is_failure']), # failure if next state is in lava
    'mintaur': None, # TODO
    'cube rotation': None, # TODO
}


if __name__ == '__main__':

    ### Arguments
    # parser = argparse.ArgumentParser(description="Safe Continual RL.")
    # parser.add_argument('-e', '--env', type=str, default='drunk spider', help='Environment name')
    # parser.add_argument('-tt', '--training_timesteps', type=int, default=10_000, help='Number of training timesteps')
    # parser.add_argument('-s', '-seed', type=int, default=2025)

    # args = parser.parse_args()
    # env_name = args.env
    # training_timesteps = args.training_timesteps
    # seed = args.seed
    env_name = 'drunk spider'
    training_timesteps = 2_000 # TODO: find the best value
    seed = 2025
    sqrl_tau_threshold = 0.2 # Safety threshold for SQRL
    interval_trainer_min_accuracy = None # TODO
    num_eval_episodes = 200

    # Set the random seed
    # torch.random.seed(seed)

    ### 1) Task 1 policy learning ##########################################################################################

    task1_env = name_to_task_dct[env_name] # initialise the env
    failure_fn = name_to_failure_fn_dct[env_name] # get the failure function

    # # Visualize the environment
    # plt.imshow(drunk_spider_env.render())
    # plt.tight_layout()
    # plt.show()

    ### Train the policy and safety critic using SQRL (safety Q-functions for RL)
    print("Starting SQRL pretraining on Task 1...")
    cfg = SQRLPretrainConfig(env=task1_env, total_env_steps=training_timesteps, tau_threshold=sqrl_tau_threshold)
    start = time.time()
    policy, safety_critic, logs = pretrain_sqrl(cfg, failure_fn=failure_fn)
    end = time.time()
    print(f"SQRL pretraining completed in {end - start:.1f} seconds")
    print("Done. Average training return:", np.mean(logs["episode_return"]), "episodes.")

    ### Evaluate the trained policy on task 1
    avg_return, success_rate = run_eval(policy, task1_env, num_episodes=num_eval_episodes)
    print(f"Avg evaluation return: {avg_return:.3f}, Success rate: {success_rate:.3f}")

    ### Plot training logs
    episode_logs = {key: value for key, value in logs.items() if key.startswith("episode_")}
    timestep_logs = {key: value for key, value in logs.items() if not key.startswith("episode_")}

    # Plot episode-level logs
    fig, axes = plt.subplots(len(episode_logs), 1, figsize=(12, 3 * len(episode_logs)), sharex=True)
    for i, (k, v) in enumerate(episode_logs.items()):
        axes[i].plot(v)
        axes[i].set_title(k)
    plt.xlabel("Episode")
    plt.show()

    # Plot timestep-level logs
    fig, axes = plt.subplots(len(timestep_logs), 1, figsize=(12, 3 * len(timestep_logs)), sharex=True)
    for i, (k, v) in enumerate(timestep_logs.items()):
        axes[i].plot(v)
        axes[i].set_title(k)
    plt.xlabel("Timestep")
    plt.show()


    ### 2) Get safe action set per state ###################################################################################
    # TODO: make this code block generalisable to any environment
    safe_actions_per_state = {}
    highest_failure_prob = 0.0
    for height in range(task1_env.H):
        for width in range(task1_env.W):
            agent = task1_env._onehot(height, width)
            goal = task1_env._onehot(task1_env.goal[0], task1_env.goal[1])
            lava = task1_env._lava.astype(np.float32).reshape(-1)
            cur_obs = np.concatenate([agent, goal, lava], axis=0).astype(np.float32)
            failure_probs = safety_critic(torch.from_numpy(cur_obs))
            safe_actions_mask = failure_probs <= cfg.tau_threshold
            if not torch.any(safe_actions_mask).item():
                print('No safe actions for the observation!')
                lowest_failure_prob = torch.min(failure_probs).item()
                print(f'Lowest failure probability is {lowest_failure_prob}. Threshold is {cfg.tau_threshold}.')
                safest_action_idx = torch.argmin(failure_probs).item()
                safe_actions_mask = torch.zeros(len(failure_probs), dtype=torch.bool)
                safe_actions_mask[safest_action_idx] = True
            safe_failure_probs = failure_probs[safe_actions_mask]
            
            # save the highest probability of failure
            if torch.min(safe_failure_probs).item() > highest_failure_prob:
                highest_failure_prob = torch.min(safe_failure_probs).item()
            
            safe_action_indices = torch.nonzero(safe_actions_mask, as_tuple=False).squeeze(-1)
            # pad the list with -1 to make sure every row has the length of the number of actions
            num_actions = getattr(task1_env.action_space, 'n', failure_probs.numel())
            pad_len = max(0, num_actions - safe_action_indices.numel())
            if pad_len > 0:
                pad = torch.full((pad_len,), -1, dtype=safe_action_indices.dtype, device=safe_action_indices.device)
                safe_action_indices = torch.cat([safe_action_indices, pad], dim=0)
            elif safe_action_indices.numel() > num_actions: # TODO: do not allow empty safe action lists
                safe_action_indices = safe_action_indices[:num_actions]
            # print("Safe action indices:", safe_action_indices.tolist())
            # print("Safe failure probabilities:", safe_failure_probs.tolist())
            # Convert observation array to tuple for use as dictionary key
            obs_key = tuple(cur_obs)
            safe_actions_per_state[obs_key] = safe_action_indices

    print(f'Highest failure probability is {highest_failure_prob}')

    ### 3) Build the Rashomon set ##########################################################################################

    ### Build a dataset in which for each state, a safe action is taken
    # TODO: how to allow taking more than one action?
    state_action_dataset = []
    for obs_tuple, action_indices in safe_actions_per_state.items():
        if len(action_indices) == 0:
            continue
        
        # Convert action_indices tensor to list and randomly select one action
        safe_actions_list = action_indices.tolist() if isinstance(action_indices, torch.Tensor) else action_indices
        if isinstance(safe_actions_list, int):
            safe_actions_list = [safe_actions_list]
        
        # Convert tuple back to numpy array for the dataset
        obs_array = np.array(obs_tuple, dtype=np.float32)
        safe_actions_array = np.array(safe_actions_list, dtype=np.int32)
        state_action_dataset.append((obs_array, safe_actions_list))

    print(f"Number of samples: {len(state_action_dataset)}")
    print("First 5 samples (obs shape -> action):")
    for s, a in state_action_dataset[:5]:
        print(f"obs shape: {s.shape} -> action: {a}")
    
    # Create a tensor with all possible states and their safe actions
    states_list = []
    actions_list = []
    for obs_array, action in state_action_dataset:
        states_list.append(torch.from_numpy(obs_array))
        actions_list.append(action)

    states = torch.stack(states_list)
    actions = torch.tensor(actions_list, dtype=torch.long)
    state_action_torch_dataset = TensorDataset(states, actions)
    state_action_loader = DataLoader(state_action_torch_dataset, batch_size=8, shuffle=True)

    print(f"Dataset size: {len(state_action_torch_dataset)}")
    print("States shape:", states.shape, "Actions shape:", actions.shape)
    print("Expected observation shape:", task1_env.observation_space.shape)

    # Use IntervalTrainer to compute the Rashomon set around the pretrained policy
    interval_trainer = IntervalTrainer( # TODO: set accuracy
        model=policy, # SQRL policy network (CategoricalPolicy)
        # min_acc_limit=interval_trainer_min_accuracy,
        seed=seed,
    )
    interval_trainer.compute_rashomon_set(
        dataset=state_action_torch_dataset, # states and safe actions
        multi_label=True # NOTE
    )

    # Store parameter bounds and certificates
    print("Rashomon set computation completed!")
    print(f"Number of bounded models: {len(interval_trainer.bounds)}")
    print(f"Number of certificates: {len(interval_trainer.certificates)}")

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


    ### 4) Task 2: Projected Gradient Descent Adaptation within Rashomon Bounds ############################################
    
    print("\n" + "="*80)
    print("TASK 2: SAC WITH PROJECTED GRADIENT DESCENT IN RASHOMON BOUNDS")
    print("="*80)
    
    # Create a new environment for Task 2 (more challenging version)
    task2_env = DrunkSpiderEnv(render_mode='rgb_array')
    # Modify the environment to be more challenging (different goal, more lava)
    task2_env.goal = (task2_env.H - 2, task2_env.W - 2)  # Different goal position
    # Add more lava in strategic locations
    for i in range(2, 8):
        for j in range(2, 8):
            if random.random() < 0.15:  # 15% chance of additional lava
                task2_env._lava[i, j] = True
    
    print(f"Task 2 Environment created with goal at {task2_env.goal}")
    print(f"Total lava cells in Task 2: {task2_env._lava.sum()}")
    
    # Initialize the adapted policy as a copy of the original policy
    adapted_policy = copy.deepcopy(policy)
    print("Created adapted policy as copy of original SQRL policy")
    
    # Parameter projection function
    def project_parameters_to_rashomon_bounds(model, param_bounds_l, param_bounds_u, device='cpu'):
        """Project model parameters to stay within Rashomon bounds."""
        total_projected = 0
        total_params = 0
        
        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                if i < len(param_bounds_l) and i < len(param_bounds_u):
                    # Get bounds for this parameter
                    p_l = param_bounds_l[i].view(param.shape).to(device)
                    p_u = param_bounds_u[i].view(param.shape).to(device)
                    
                    # Count violations before projection
                    violations = ((param.data < p_l) | (param.data > p_u)).sum().item()
                    total_projected += violations
                    total_params += param.numel()
                    
                    # Project parameters to bounds
                    param.data.clamp_(min=p_l, max=p_u)
        
        return total_projected, total_params
    
    # SAC Components - Create individual Q-critics
    from scripts._sqrl_pretrain import MLP
    
    # Individual Q-Critics for SAC (not the twin version)
    obs_dim = task2_env.observation_space.shape[0] if task2_env.observation_space.shape else 300
    act_dim = task2_env.action_space.n if hasattr(task2_env.action_space, 'n') else 8
    
    q_critic1 = MLP(obs_dim, act_dim, hidden=(256, 256), act=torch.nn.ReLU)
    q_critic2 = MLP(obs_dim, act_dim, hidden=(256, 256), act=torch.nn.ReLU)
    q_critic1_target = copy.deepcopy(q_critic1)
    q_critic2_target = copy.deepcopy(q_critic2)
    
    # Optimizers
    policy_optimizer = torch.optim.Adam(adapted_policy.parameters(), lr=3e-4)
    q1_optimizer = torch.optim.Adam(q_critic1.parameters(), lr=3e-4)
    q2_optimizer = torch.optim.Adam(q_critic2.parameters(), lr=3e-4)
    
    # SAC hyperparameters
    gamma = 0.99
    tau = 0.005  # Target network update rate
    alpha = 0.2   # Temperature parameter
    # For discrete actions with 8 actions: target_entropy = -log(1/8) * 0.98 ≈ -2.04
    target_entropy = -np.log(1.0 / 8) * 0.98  # Target entropy for 8 discrete actions
    log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)
    
    # Replay buffer
    class SAC_ReplayBuffer:
        def __init__(self, capacity):
            self.buffer = deque(maxlen=capacity)
        
        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))
        
        def sample(self, batch_size):
            batch = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = map(np.array, zip(*batch))
            return state, action, reward, next_state, done
        
        def __len__(self):
            return len(self.buffer)
    
    replay_buffer = SAC_ReplayBuffer(100000)
    
    # Training parameters
    max_episodes = 200
    max_steps_per_episode = 200
    batch_size = 256
    min_replay_size = 1000
    update_frequency = 1
    
    # Logging
    episode_rewards = []
    episode_lengths = []
    projection_counts = []
    policy_losses = []
    q_losses = []
    
    print(f"\nStarting SAC training with Rashomon bounds projection...")
    print(f"Max episodes: {max_episodes}")
    print(f"Batch size: {batch_size}")
    print(f"Target entropy: {target_entropy:.3f}")
    
    total_steps = 0
    for episode in range(max_episodes):
        state, _ = task2_env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps_per_episode):
            # Select action using current policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                logits = adapted_policy(state_tensor)
                action_probs = torch.softmax(logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = task2_env.step(action)
            done = terminated or truncated
            
            # Store in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # Update networks if we have enough samples
            if len(replay_buffer) > min_replay_size and total_steps % update_frequency == 0:
                # Sample batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)
                
                # Update Q-networks
                with torch.no_grad():
                    next_logits = adapted_policy(next_states)
                    next_action_probs = torch.softmax(next_logits, dim=-1)
                    next_log_probs = torch.log_softmax(next_logits, dim=-1)
                    
                    # Now q_critics return single tensors (B, n_actions)
                    next_q1 = q_critic1_target(next_states)
                    next_q2 = q_critic2_target(next_states)
                    next_q = torch.min(next_q1, next_q2)
                    
                    # Soft value function
                    next_v = (next_action_probs * (next_q - alpha * next_log_probs)).sum(dim=-1, keepdim=True)
                    target_q = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * next_v
                
                # Current Q-values
                current_q1 = q_critic1(states).gather(1, actions.unsqueeze(1))
                current_q2 = q_critic2(states).gather(1, actions.unsqueeze(1))
                
                # Q-loss
                q1_loss = torch.nn.functional.mse_loss(current_q1, target_q)
                q2_loss = torch.nn.functional.mse_loss(current_q2, target_q)
                
                # Update Q-networks
                q1_optimizer.zero_grad()
                q1_loss.backward()
                q1_optimizer.step()
                
                q2_optimizer.zero_grad()
                q2_loss.backward()
                q2_optimizer.step()
                
                # Update policy
                logits = adapted_policy(states)
                action_probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                
                with torch.no_grad():
                    q1_values = q_critic1(states)
                    q2_values = q_critic2(states)
                    q_values = torch.min(q1_values, q2_values)
                
                # Policy loss (maximize entropy-regularized reward)
                policy_loss = (action_probs * (alpha * log_probs - q_values)).sum(dim=-1).mean()
                
                # Update policy with gradient step
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()
                
                # PROJECT PARAMETERS TO RASHOMON BOUNDS (KEY INNOVATION!)
                projected_params, total_params = project_parameters_to_rashomon_bounds(
                    adapted_policy, param_bounds_l, param_bounds_u
                )
                
                # Update temperature parameter
                with torch.no_grad():
                    entropy = -(action_probs * log_probs).sum(dim=-1).mean()
                alpha_loss = -log_alpha * (entropy - target_entropy)
                
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                alpha = log_alpha.exp().item()
                
                # Update target networks
                for target_param, param in zip(q_critic1_target.parameters(), q_critic1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for target_param, param in zip(q_critic2_target.parameters(), q_critic2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                # Log training metrics
                if total_steps % 1000 == 0:
                    projection_counts.append(projected_params)
                    policy_losses.append(policy_loss.item())
                    q_losses.append((q1_loss.item() + q2_loss.item()) / 2)
            
            if done:
                break
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Print progress
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0
            avg_length = np.mean(episode_lengths[-20:]) if episode_lengths else 0
            recent_projections = np.mean(projection_counts[-10:]) if projection_counts else 0
            
            print(f"Episode {episode:3d} | Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Length: {avg_length:5.1f} | Steps: {total_steps:6d} | "
                  f"Alpha: {alpha:.3f} | Projected Params: {recent_projections:5.0f}")
    
    print(f"\nSAC training completed after {max_episodes} episodes!")
    
    # Final evaluation on both tasks
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Evaluate on Task 1 (original environment)
    print("Evaluating on Task 1 (original environment)...")
    original_task1_reward, original_task1_success = run_eval(policy, task1_env, num_episodes=100)
    adapted_task1_reward, adapted_task1_success = run_eval(adapted_policy, task1_env, num_episodes=100)
    
    print(f"Task 1 - Original Policy: Reward={original_task1_reward:.3f}, Success={original_task1_success:.3f}")
    print(f"Task 1 - Adapted Policy:  Reward={adapted_task1_reward:.3f}, Success={adapted_task1_success:.3f}")
    
    # Evaluate on Task 2 (new environment)
    print("Evaluating on Task 2 (new environment)...")
    original_task2_reward, original_task2_success = run_eval(policy, task2_env, num_episodes=100)
    adapted_task2_reward, adapted_task2_success = run_eval(adapted_policy, task2_env, num_episodes=100)
    
    print(f"Task 2 - Original Policy: Reward={original_task2_reward:.3f}, Success={original_task2_success:.3f}")
    print(f"Task 2 - Adapted Policy:  Reward={adapted_task2_reward:.3f}, Success={adapted_task2_success:.3f}")
    
    # Verify parameter bounds compliance
    print("\nVerifying Rashomon bounds compliance...")
    final_projected, final_total = project_parameters_to_rashomon_bounds(
        adapted_policy, param_bounds_l, param_bounds_u
    )
    compliance_rate = 1.0 - (final_projected / max(final_total, 1))
    print(f"Parameter compliance: {compliance_rate:.4f} ({final_total - final_projected}/{final_total} parameters within bounds)")
    
    # Plot training results
    plt.figure(figsize=(15, 10))
    
    # Episode rewards
    plt.subplot(2, 3, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Episode lengths
    plt.subplot(2, 3, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    # Parameter projections
    plt.subplot(2, 3, 3)
    if projection_counts:
        plt.plot(projection_counts)
        plt.title('Parameter Projections')
        plt.xlabel('Update Step (x1000)')
        plt.ylabel('Projected Parameters')
        plt.grid(True)
    
    # Policy loss
    plt.subplot(2, 3, 4)
    if policy_losses:
        plt.plot(policy_losses)
        plt.title('Policy Loss')
        plt.xlabel('Update Step (x1000)')
        plt.ylabel('Loss')
        plt.grid(True)
    
    # Q loss
    plt.subplot(2, 3, 5)
    if q_losses:
        plt.plot(q_losses)
        plt.title('Q-Network Loss')
        plt.xlabel('Update Step (x1000)')
        plt.ylabel('Loss')
        plt.grid(True)
    
    # Performance comparison
    plt.subplot(2, 3, 6)
    tasks = ['Task 1', 'Task 2']
    original_rewards = [original_task1_reward, original_task2_reward]
    adapted_rewards = [adapted_task1_reward, adapted_task2_reward]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    plt.bar(x - width/2, original_rewards, width, label='Original Policy', alpha=0.8)
    plt.bar(x + width/2, adapted_rewards, width, label='Adapted Policy', alpha=0.8)
    plt.xlabel('Task')
    plt.ylabel('Average Reward')
    plt.title('Performance Comparison')
    plt.xticks(x, tasks)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ Successfully trained SAC with Rashomon bounds projection")
    print(f"📊 Task 1 performance change: {adapted_task1_reward - original_task1_reward:+.3f}")
    print(f"📊 Task 2 performance change: {adapted_task2_reward - original_task2_reward:+.3f}")
    print(f"🔒 Parameter compliance: {compliance_rate:.1%}")
    print(f"🎯 Total training episodes: {max_episodes}")
    print(f"🎯 Total environment steps: {total_steps}")
    
    if adapted_task2_reward > original_task2_reward:
        print("🎉 Successfully adapted to Task 2 while maintaining safety bounds!")
    else:
        print("⚠️  Limited adaptation to Task 2, but safety bounds maintained.")
    
    print("\nProjected Gradient Descent SAC training completed! 🚀")
