import gymnasium as gym
import numpy
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn

class SafetyCritic(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),  # logits for failure probability
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def collect_traces_dataset(
    policy_model: nn.Sequential, 
    env: gym.Env,
    num_episodes=1_000, 
    epsilon=0.2, 
    device='cpu',
    seed: int = 42
):
    """
    Collect a dataset of observations, actions, failures and dones from the environment
    using the given policy model with epsilon-greedy exploration.
    """
    obs_list, action_list, next_obs_list, failure_list, done_list = [], [], [], [], []

    for episode_num in range(num_episodes):
        obs, _ = env.reset(seed=seed*episode_num)
        ep_obs = []
        ep_actions = []
        ep_next_obs = []
        ep_failures = []
        ep_dones = []
        done = False
        while not done:
            ep_obs.append(obs.copy())

            if numpy.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                if isinstance(policy_model, ActorCriticPolicy):
                    ### For SB3 model
                    action, _ = policy_model.predict(obs, deterministic=True)
                else:
                    ### For my PPO from scratch
                    cur_obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    logits = policy_model.forward(cur_obs_tensor)
                    action = logits.argmax(dim=-1).item()
                if isinstance(action, numpy.ndarray) and action.shape == (1,):
                    action = action.item()
            ep_actions.append(action)

            next_obs, _r, terminated, truncated, info = env.step(action)
            ep_next_obs.append(next_obs.copy())
            done = terminated or truncated
            fail = 1 if info['failure'] else 0
            ep_failures.append(fail)
            ep_dones.append(done)
            if done:
                obs_list.extend(ep_obs)
                action_list.extend(ep_actions)
                next_obs_list.extend(ep_next_obs)
                failure_list.extend(ep_failures)
                done_list.extend(ep_dones)
            obs = next_obs

    env.close()
    obs_arr = numpy.asarray(obs_list, dtype=numpy.float32)
    action_arr = numpy.asarray(action_list, dtype=numpy.float32)
    next_obs_arr = numpy.asarray(next_obs_list, dtype=numpy.float32)
    failure_arr = numpy.asarray(failure_list, dtype=numpy.float32)
    done_arr = numpy.asarray(done_list, dtype=numpy.float32)
    return obs_arr, action_arr, next_obs_arr, failure_arr, done_arr