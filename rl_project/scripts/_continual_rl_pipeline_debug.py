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
sns.set_style("whitegrid")

project_root = os.path.abspath('/Users/ma5923/Documents/_projects/CertifiedContinualLearning')
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts._sqrl_pretrain import SQRLPretrainConfig, pretrain_sqrl, default_failure_fn, MLP
from scripts.custom_tasks import *
from src.trainer import IntervalTrainer
from torch.utils.data import TensorDataset, DataLoader

print("🚀 Starting Safe Continual RL Pipeline with Debug Info")
print("="*60)

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

if __name__ == '__main__':
    try:
        ### Arguments
        env_name = 'drunk spider'
        training_timesteps = 5000  # Reduced for testing
        seed = 2025

        ### Hyperparameters
        sqrl_tau_threshold = 0.2 # Safety threshold for SQRL
        
        print(f"📋 Configuration:")
        print(f"   Environment: {env_name}")
        print(f"   Training timesteps: {training_timesteps}")
        print(f"   Seed: {seed}")
        print(f"   SQRL tau threshold: {sqrl_tau_threshold}")

        ### 1) Task 1 policy learning
        print("\n🎯 PHASE 1: SQRL Policy Training")
        print("-" * 40)
        
        if env_name == 'drunk spider':
            task1_env = DrunkSpiderEnv(render_mode='rgb_array')
            failure_fn_drunk_spider = lambda obs, action, next_obs, info: bool(info['is_failure'])
            print(f"✅ Environment created: {task1_env.__class__.__name__}")
        else:
            raise NotImplementedError()

        # Train the policy and safety critic using SQRL
        print("Starting SQRL pretraining...")
        cfg = SQRLPretrainConfig(env=task1_env, total_env_steps=training_timesteps, tau_threshold=sqrl_tau_threshold)
        start = time.time()
        policy, safety_critic, logs = pretrain_sqrl(cfg, failure_fn=failure_fn_drunk_spider)
        end = time.time()
        print(f"✅ SQRL pretraining completed in {end - start:.1f} seconds")
        print(f"   Episodes: {len(logs['episode_return'])}")
        print(f"   Average training return: {np.mean(logs['episode_return']):.3f}")

        # Evaluate the trained policy on task 1
        print("Evaluating trained policy...")
        avg_r, succ = run_eval(policy, task1_env, num_episodes=50)  # Reduced for speed
        print(f"✅ Task 1 evaluation - Reward: {avg_r:.3f}, Success: {succ:.3f}")

        ### 2) Get safe action set per state
        print("\n🛡️ PHASE 2: Safe Action Set Generation")
        print("-" * 40)
        
        safe_actions_per_state = {}
        total_states = task1_env.H * task1_env.W
        processed_states = 0
        
        for height in range(task1_env.H):
            for width in range(task1_env.W):
                agent = task1_env._onehot(height, width)
                goal = task1_env._onehot(task1_env.goal[0], task1_env.goal[1])
                lava = task1_env._lava.astype(np.float32).reshape(-1)
                cur_obs = np.concatenate([agent, goal, lava], axis=0).astype(np.float32)
                failure_probs = safety_critic(torch.from_numpy(cur_obs))
                safe_actions_mask = failure_probs <= cfg.tau_threshold
                safe_action_indices = torch.nonzero(safe_actions_mask, as_tuple=False).squeeze(-1)
                
                # Convert observation array to tuple for use as dictionary key
                obs_key = tuple(cur_obs)
                safe_actions_per_state[obs_key] = safe_action_indices
                
                processed_states += 1
                if processed_states % 20 == 0:
                    print(f"   Processed {processed_states}/{total_states} states...")

        print(f"✅ Safe actions computed for {len(safe_actions_per_state)} states")

        ### Build a dataset
        print("Building safe action dataset...")
        state_action_dataset = []
        for obs_tuple, action_indices in safe_actions_per_state.items():
            if len(action_indices) == 0:
                continue
            
            # Convert action_indices tensor to list and randomly select one action
            safe_actions_list = action_indices.tolist() if isinstance(action_indices, torch.Tensor) else action_indices
            if isinstance(safe_actions_list, int):
                safe_actions_list = [safe_actions_list]
            
            cur_action = random.choice(safe_actions_list)
            # Convert tuple back to numpy array for the dataset
            obs_array = np.array(obs_tuple, dtype=np.float32)
            state_action_dataset.append((obs_array, cur_action))

        print(f"✅ Dataset created with {len(state_action_dataset)} samples")

        ### 3) Build the Rashomon set
        print("\n📊 PHASE 3: Rashomon Set Computation")
        print("-" * 40)
        
        # Create dataset
        states_list = []
        actions_list = []
        for obs_array, action in state_action_dataset:
            states_list.append(torch.from_numpy(obs_array))
            actions_list.append(action)

        states = torch.stack(states_list)
        actions = torch.tensor(actions_list, dtype=torch.long)
        state_action_torch_dataset = TensorDataset(states, actions)

        print(f"Dataset - States: {states.shape}, Actions: {actions.shape}")

        # Use IntervalTrainer to compute the Rashomon set
        print("Computing Rashomon set...")
        interval_trainer = IntervalTrainer(model=policy, seed=2025)
        interval_trainer.compute_rashomon_set(dataset=state_action_torch_dataset)

        print(f"✅ Rashomon set computation completed!")
        print(f"   Bounded models: {len(interval_trainer.bounds)}")
        print(f"   Certificates: {len(interval_trainer.certificates)}")

        # Extract parameter bounds
        bounded_model = interval_trainer.bounds[0]
        param_bounds_l = [bound.detach().cpu() for bound in bounded_model.param_l]
        param_bounds_u = [bound.detach().cpu() for bound in bounded_model.param_u]

        total_params = sum(p_l.numel() for p_l in param_bounds_l)
        certificate = interval_trainer.certificates[0]
        
        print(f"   Total parameters: {total_params}")
        print(f"   Certificate: {certificate:.3f}")

        ### 4) Task 2: SAC with Projected Gradient Descent
        print("\n🎯 PHASE 4: SAC with Rashomon Bounds")
        print("-" * 40)
        
        # Create Task 2 environment (more challenging)
        task2_env = DrunkSpiderEnv(render_mode='rgb_array')
        task2_env.goal = (task2_env.H - 2, task2_env.W - 2)  # Different goal
        print(f"✅ Task 2 environment created with goal at {task2_env.goal}")
        
        # Initialize adapted policy
        adapted_policy = copy.deepcopy(policy)
        print("✅ Adapted policy initialized")
        
        # Parameter projection function
        def project_parameters_to_rashomon_bounds(model, param_bounds_l, param_bounds_u):
            total_projected = 0
            total_params = 0
            
            with torch.no_grad():
                for i, param in enumerate(model.parameters()):
                    if i < len(param_bounds_l) and i < len(param_bounds_u):
                        p_l = param_bounds_l[i].view(param.shape)
                        p_u = param_bounds_u[i].view(param.shape)
                        
                        violations = ((param.data < p_l) | (param.data > p_u)).sum().item()
                        total_projected += violations
                        total_params += param.numel()
                        
                        param.data.clamp_(min=p_l, max=p_u)
            
            return total_projected, total_params
        
        # SAC Components
        obs_dim = 300  # DrunkSpider observation dimension
        act_dim = 8    # DrunkSpider action dimension
        
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
        tau = 0.005
        alpha = 0.2
        target_entropy = -np.log(1.0 / 8) * 0.98
        log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)
        
        print("✅ SAC components initialized")
        
        # Simple replay buffer
        class SimpleReplayBuffer:
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
        
        replay_buffer = SimpleReplayBuffer(50000)  # Smaller buffer for testing
        
        # Training parameters (reduced for testing)
        max_episodes = 50
        max_steps_per_episode = 100
        batch_size = 128
        min_replay_size = 500
        
        print(f"Training parameters - Episodes: {max_episodes}, Batch size: {batch_size}")
        
        # Training loop
        print("Starting SAC training with projection...")
        episode_rewards = []
        projection_counts = []
        
        total_steps = 0
        for episode in range(max_episodes):
            state, _ = task2_env.reset()
            episode_reward = 0
            
            for step in range(max_steps_per_episode):
                # Select action
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    logits = adapted_policy(state_tensor)
                    action_probs = torch.softmax(logits, dim=-1)
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample().item()
                
                # Take action
                next_state, reward, terminated, truncated, info = task2_env.step(action)
                done = terminated or truncated
                
                # Store transition
                replay_buffer.push(state, action, reward, next_state, done)
                
                episode_reward += reward
                total_steps += 1
                
                # Update networks
                if len(replay_buffer) > min_replay_size and total_steps % 4 == 0:
                    # Sample batch (simplified SAC update)
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                    
                    states = torch.FloatTensor(states)
                    actions = torch.LongTensor(actions)
                    rewards = torch.FloatTensor(rewards)
                    next_states = torch.FloatTensor(next_states)
                    dones = torch.FloatTensor(dones)
                    
                    # Simple Q-update (simplified for testing)
                    with torch.no_grad():
                        next_q1 = q_critic1_target(next_states)
                        next_q2 = q_critic2_target(next_states)
                        next_q = torch.min(next_q1, next_q2)
                        max_next_q = next_q.max(dim=1, keepdim=True)[0]
                        target_q = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * max_next_q
                    
                    current_q1 = q_critic1(states).gather(1, actions.unsqueeze(1))
                    current_q2 = q_critic2(states).gather(1, actions.unsqueeze(1))
                    
                    q1_loss = torch.nn.functional.mse_loss(current_q1, target_q)
                    q2_loss = torch.nn.functional.mse_loss(current_q2, target_q)
                    
                    # Update Q-networks
                    q1_optimizer.zero_grad()
                    q1_loss.backward()
                    q1_optimizer.step()
                    
                    q2_optimizer.zero_grad()
                    q2_loss.backward()
                    q2_optimizer.step()
                    
                    # Update policy (simplified)
                    logits = adapted_policy(states)
                    action_probs = torch.softmax(logits, dim=-1)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    with torch.no_grad():
                        q_values = torch.min(q_critic1(states), q_critic2(states))
                    
                    policy_loss = (action_probs * (alpha * log_probs - q_values)).sum(dim=-1).mean()
                    
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()
                    
                    # PROJECT PARAMETERS TO RASHOMON BOUNDS
                    projected_params, total_params_check = project_parameters_to_rashomon_bounds(
                        adapted_policy, param_bounds_l, param_bounds_u
                    )
                    
                    if total_steps % 1000 == 0:
                        projection_counts.append(projected_params)
                    
                    # Update targets
                    for target_param, param in zip(q_critic1_target.parameters(), q_critic1.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    
                    for target_param, param in zip(q_critic2_target.parameters(), q_critic2.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                if done:
                    break
                
                state = next_state
            
            episode_rewards.append(episode_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                recent_projections = np.mean(projection_counts[-5:]) if projection_counts else 0
                print(f"   Episode {episode:2d} | Avg Reward: {avg_reward:6.2f} | "
                      f"Steps: {total_steps:5d} | Projections: {recent_projections:4.0f}")
        
        print(f"✅ SAC training completed!")
        
        ### Final evaluation
        print("\n🏁 FINAL EVALUATION")
        print("-" * 40)
        
        # Evaluate on both tasks
        print("Evaluating on Task 1 (original)...")
        orig_task1_reward, orig_task1_success = run_eval(policy, task1_env, num_episodes=50)
        adapt_task1_reward, adapt_task1_success = run_eval(adapted_policy, task1_env, num_episodes=50)
        
        print("Evaluating on Task 2 (new)...")
        orig_task2_reward, orig_task2_success = run_eval(policy, task2_env, num_episodes=50)
        adapt_task2_reward, adapt_task2_success = run_eval(adapted_policy, task2_env, num_episodes=50)
        
        print("\n📊 RESULTS SUMMARY")
        print("=" * 50)
        print(f"Task 1 - Original: {orig_task1_reward:.3f} reward, {orig_task1_success:.3f} success")
        print(f"Task 1 - Adapted:  {adapt_task1_reward:.3f} reward, {adapt_task1_success:.3f} success")
        print(f"Task 2 - Original: {orig_task2_reward:.3f} reward, {orig_task2_success:.3f} success")
        print(f"Task 2 - Adapted:  {adapt_task2_reward:.3f} reward, {adapt_task2_success:.3f} success")
        
        # Verify bounds compliance
        final_projected, final_total = project_parameters_to_rashomon_bounds(
            adapted_policy, param_bounds_l, param_bounds_u
        )
        compliance_rate = 1.0 - (final_projected / max(final_total, 1))
        print(f"\n🔒 Parameter compliance: {compliance_rate:.4f}")
        
        print("\n🎉 Safe Continual RL Pipeline completed successfully!")
        
        if adapt_task2_reward > orig_task2_reward:
            print("✅ Successfully adapted to Task 2 while maintaining safety bounds!")
        else:
            print("⚠️ Limited adaptation, but safety bounds maintained.")
            
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {e}")
        print("Full traceback:")
        traceback.print_exc()