import gymnasium
import torch
import numpy as np
import pandas as pd

def get_unique_safe_state_action_pairs(training_data: dict[str, np.ndarray | torch.Tensor]) -> torch.utils.data.TensorDataset:
    """
    Extract unique safe state-action pairs from the standard training data.

    Args:
        standard_training_data: A dictionary with keys 'states', 'actions', and 'safe'.
            'states' should be a numpy array or torch tensor of shape (N, state_dim).
            'actions' should be a numpy array or torch tensor of shape (N,).
            'safe' should be a numpy array or torch tensor of shape (N,), with 1.0 for safe and 0.0 for unsafe. 
    Returns:
        safe_training_data_torch_dataset: A torch.utils.data.TensorDataset containing unique safe state-action
     pairs.
    """

    states = torch.FloatTensor(training_data['states'])
    actions = torch.LongTensor(training_data['actions'])
    safety_flags = torch.FloatTensor(training_data['safe'])  # 1.0 for safe, 0.0 for unsafe
    # Filter only safe state-action pairs
    safe_indices = np.where(safety_flags == 1.0)[0]
    states = states[safe_indices]
    actions = actions[safe_indices]

    # Remove duplicate safe state-action pairs
    safe_states_df = pd.DataFrame(states.detach().numpy())
    safe_states_df.columns = [f'state_feature_{i}' for i in range(safe_states_df.shape[1])]
    actions_df = pd.DataFrame(actions.detach().numpy())
    actions_df.columns = ['action'] if actions_df.shape[1] == 1 else [f'action_{i}' for i in range(actions_df.shape[1])]
    action_cols = actions_df.columns.tolist()
    safe_state_action_pairs_df = pd.concat([safe_states_df, actions_df], axis=1)
    safe_state_action_pairs_df = safe_state_action_pairs_df.drop_duplicates(keep='first').reset_index(drop=True)
    # Convert back to torch dataset
    states = torch.FloatTensor(safe_state_action_pairs_df.drop(columns=action_cols).values)
    actions = torch.LongTensor(safe_state_action_pairs_df[action_cols].values)
    safe_training_data_torch_dataset = torch.utils.data.TensorDataset(states, actions)
    return safe_training_data_torch_dataset

def generate_safe_optimal_policy_data(
        env,
        safe_actor,
        num_episodes: int = 1,
        deterministic: bool = True,
        log_std: float | None = None,
    ) -> torch.utils.data.TensorDataset:
    """
    Generate safe optimal policy data by rolling out the safe_actor in the environment.

    Args:
        env: The Gymnasium environment to roll out in.
        safe_actor: The trained safe policy network (safety guaranteed when deterministic=True).
        num_episodes: Number of rollouts to collect data from.
        deterministic: Whether to use deterministic actions.
        log_std: Log standard deviation for continuous action spaces (if applicable).

    Returns:
        safe_optimal_policy_data_torch_dataset: A torch.utils.data.TensorDataset containing
         (states, actions) pairs from the safe optimal policy.
    """
    states = []
    actions = []

    continuouse_actions = isinstance(env.action_space, gymnasium.spaces.Box)

    # Collect multiple rollouts to get diverse state coverage
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            # Get action from safe_actor (the trained safe policy)
            with torch.no_grad():
                if continuouse_actions:
                    # Continuous actions: use mean from safe_actor
                    if deterministic:
                        action = safe_actor(obs_t).cpu().numpy()[0]
                    else:
                        mean = safe_actor(obs_t)
                        std = np.exp(log_std).expand_as(mean)
                        dist = torch.distributions.Normal(mean, std)
                        action = dist.sample().cpu().numpy()[0]
                    # Clip to action space bounds
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                else:
                    # Discrete actions: use argmax
                    if deterministic:
                        logits = safe_actor(obs_t)
                        action = torch.argmax(logits, dim=-1).item()
                    else:
                        logits = safe_actor(obs_t)
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample().item()
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action) # type: ignore
            done = terminated or truncated
            
            # Record state-action pair (if it is safe)
            safety_flag = info.get('is_safe', 1.0)  # Default to safe if not provided
            if safety_flag == 1.0:
                states.append(obs)
                actions.append(action)

            obs = next_obs

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)

    # Remove duplicate safe state-action pairs
    safe_states_df = pd.DataFrame(states.detach().numpy())
    safe_states_df.columns = [f'state_feature_{i}' for i in range(safe_states_df.shape[1])]
    actions_df = pd.DataFrame(actions.detach().numpy())
    actions_df.columns = ['action'] if actions_df.shape[1] == 1 else [f'action_{i}' for i in range(actions_df.shape[1])]
    action_cols = actions_df.columns.tolist()
    safe_state_action_pairs_df = pd.concat([safe_states_df, actions_df], axis=1)
    safe_state_action_pairs_df = safe_state_action_pairs_df.drop_duplicates(keep='first').reset_index(drop=True)
    # Convert back to torch dataset
    states = torch.FloatTensor(safe_state_action_pairs_df.drop(columns=action_cols).values)
    actions = torch.LongTensor(safe_state_action_pairs_df[action_cols].values)
    safe_state_action_torch_dataset = torch.utils.data.TensorDataset(states, actions)
    return safe_state_action_torch_dataset