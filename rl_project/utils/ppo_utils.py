import torch
from dataclasses import dataclass
import random
import numpy
import numpy as np
import gymnasium as gym
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import matplotlib.pyplot as plt

#### --- PPO Utilities --- ####
@dataclass
class PPOConfig:
    seed: int = 42
    total_timesteps: int = 100_000
    eval_episodes: int = 1_000
    rollout_steps: int = 2048
    update_epochs: int = 10
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def make_actor_critic(
    obs_dim: int, n_actions: int, 
    actor_warm_start: nn.Sequential | None = None,
    critic_warm_start: nn.Sequential | None = None,
    ):
    """
    Create simple feedforward actor and critic networks.
    Actor is a policy network outputting logits for discrete actions.
    Critic is a value network outputting a single value.
    """
    if actor_warm_start is not None:
        actor = copy.deepcopy(actor_warm_start)
        assert isinstance(actor, torch.nn.Sequential), "Warm-start actor must be nn.Sequential"
        last = actor[-1]
        assert hasattr(last, "out_features"), "Last layer must expose out_features"
        assert last.out_features == n_actions, "Actor output dim must match env action space"
    else:
        # actor = torch.nn.Sequential(
        #     torch.nn.Linear(obs_dim, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, n_actions),
        # )
        actor = torch.nn.Sequential( # matching SB3 MlpPolicy architecture
            torch.nn.Linear(obs_dim, 256),
            # torch.nn.Tanh(),  # SB3 uses Tanh by default
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            # torch.nn.Tanh(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_actions),
        )
    if critic_warm_start is not None:
        critic = copy.deepcopy(critic_warm_start)
        assert isinstance(critic, torch.nn.Sequential), "Warm-start critic must be nn.Sequential"
        last = critic[-1]
        assert hasattr(last, "out_features"), "Last layer must expose out_features"
        assert last.out_features == 1, "Critic output dim must be 1"
    else:
        # critic = torch.nn.Sequential(
        #     torch.nn.Linear(obs_dim, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 1),
        # )
        critic = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 1),
        )
    return actor, critic

def set_seed(env, seed: int):
    """
    Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()
        env.action_space.seed(seed)

def evaluate(
        env: gym.Env, actor: nn.Sequential, episodes=10, seed: int = 2025,
        device: str = 'cpu', render_mode: str | None = None
    ):
    """
    Evaluate the actor policy over a number of episodes and return mean and std of rewards.
    """
    assert render_mode in (None, 'rgb_array')
    actor.eval()
    scores = []
    failures = 0
    for episode_num in range(episodes):
        obs, _ = env.reset(seed=seed*episode_num)
        episodic_reward = 0.0
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = actor(obs_t)
                action = torch.argmax(logits, dim=-1).item()
            obs, reward, terminated, truncated, info = env.step(action)
            if render_mode == 'rgb_array':
                plt.imshow(env.render()) # type: ignore
                plt.axis('off')
                plt.show()
            if info.get('failure', False):
                failures += 1
            episodic_reward += reward # type: ignore
            done = terminated or truncated
        scores.append(episodic_reward)
    actor.train()
    avg_total_reward = float(np.mean(scores))
    std_total_reward = float(np.std(scores))
    failure_rate = failures / episodes
    return avg_total_reward, std_total_reward, failure_rate

def ppo_train(
    env: gym.Env, cfg: PPOConfig, 
    actor_warm_start: nn.Sequential | None = None,
    critic_warm_start: nn.Sequential | None = None,
    actor_param_bounds_l: list[torch.Tensor] | None = None,
    actor_param_bounds_u: list[torch.Tensor] | None = None,
    return_training_data: bool = False
):
    """
    Train a PPO agent. If actor_param_bounds_l and actor_param_bounds_u are provided,
    uses projected gradient descent to keep actor parameters within bounds.
    
    Args:
        env: Gymnasium environment
        cfg: PPO configuration
        actor_warm_start: Optional pre-trained actor network
        critic_warm_start: Optional pre-trained critic network
        actor_param_bounds_l: Optional lower bounds for actor parameters (for PGD)
        actor_param_bounds_u: Optional upper bounds for actor parameters (for PGD)
        return_training_data: If True, returns state-action pairs collected during training
        
    Returns:
        If return_training_data is False: (actor, critic)
        If return_training_data is True: (actor, critic, training_data)
            where training_data is a dict containing:
                - 'states': numpy array of shape (N, obs_dim) containing all states visited
                - 'actions': numpy array of shape (N,) containing all actions taken
                - 'rewards': numpy array of shape (N,) containing all rewards received
    """
    # env_kwargs = cfg.env_kwargs if cfg.env_kwargs is not None else {}
    # env = gym.make(cfg.env_id, **env_kwargs)
    set_seed(env, cfg.seed)

    obs_dim = env.observation_space.shape[0] if isinstance(env.observation_space, gym.spaces.Box) else env.observation_space.n # type: ignore
    n_actions = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0] # type: ignore

    actor, critic = make_actor_critic(
        obs_dim=obs_dim,
        n_actions=n_actions, # type: ignore
        actor_warm_start=actor_warm_start,
        critic_warm_start=critic_warm_start
    )
    device = torch.device(cfg.device)
    actor.to(device)
    critic.to(device)

    # Check if we have parameter bounds for projected gradient descent
    use_pgd = (actor_param_bounds_l is not None and actor_param_bounds_u is not None)
    print('Use PGD:', use_pgd)
    bounds_l = None
    bounds_u = None
    if use_pgd:
        bounds_l = [bound.to(device) for bound in actor_param_bounds_l] # type: ignore
        bounds_u = [bound.to(device) for bound in actor_param_bounds_u] # type: ignore
        print(f"Using projected gradient descent with parameter bounds")
        
        # Ensure initial parameters are within bounds
        with torch.no_grad():
            for param, lb, ub in zip(actor.parameters(), bounds_l, bounds_u):
                param.data.clamp_(lb, ub)

    optimizer = torch.optim.Adam([
        {"params": actor.parameters(), "lr": cfg.lr},
        {"params": critic.parameters(), "lr": cfg.lr},
    ])

    obs, _ = env.reset(seed=cfg.seed)
    global_step = 0
    start_time = time.time()
    pgd_projections = 0  # Count total parameter projections for logging
    
    # Training data tracking
    if return_training_data:
        training_data = {
            'states': [],
            'actions': [],
            'rewards': []
        }

    while global_step < cfg.total_timesteps:
        # Storage for rollout
        obs_buf = np.zeros((cfg.rollout_steps, obs_dim), dtype=np.float32)
        act_buf = np.zeros((cfg.rollout_steps,), dtype=np.int64)
        logp_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        rew_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        done_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        val_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)

        # Collect rollout
        for t in range(cfg.rollout_steps):
            obs_buf[t] = obs
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = actor(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                value = critic(obs_t).squeeze(-1)
                action = dist.sample()
                logp = dist.log_prob(action)
            act = int(action.item())
            next_obs, reward, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            
            # Collect state-action pairs if recording training data
            if return_training_data:
                training_data['states'].append(obs.copy()) # type: ignore
                training_data['actions'].append(act) # type: ignore
                training_data['rewards'].append(float(reward)) # type: ignore
            
            act_buf[t] = act
            logp_buf[t] = float(logp.item())
            rew_buf[t] = float(reward)
            done_buf[t] = float(done)
            val_buf[t] = float(value.item())

            obs = next_obs
            global_step += 1
            if done:
                obs, _ = env.reset()

            if global_step >= cfg.total_timesteps:
                # If we hit total_timesteps mid-rollout, cut here
                obs_buf = obs_buf[:t+1]
                act_buf = act_buf[:t+1]
                logp_buf = logp_buf[:t+1]
                rew_buf = rew_buf[:t+1]
                done_buf = done_buf[:t+1]
                val_buf = val_buf[:t+1]
                break

        # Bootstrap with value of last observation
        with torch.no_grad():
            last_val = critic(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).item()

        # Compute GAE advantages and returns
        T = len(rew_buf)
        adv_buf = np.zeros_like(rew_buf)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - done_buf[t]
            next_value = last_val if t == T - 1 else val_buf[t + 1]
            delta = rew_buf[t] + cfg.gamma * next_value * next_nonterminal - val_buf[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            adv_buf[t] = last_gae
        ret_buf = adv_buf + val_buf

        # Normalize advantages
        adv_t = torch.tensor(adv_buf, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        obs_t = torch.tensor(obs_buf, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_buf, dtype=torch.int64, device=device)
        old_logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_buf, dtype=torch.float32, device=device)

        # PPO updates
        batch_size = T
        idxs = np.arange(batch_size)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_idx = idxs[start:end]
                mb_obs = obs_t[mb_idx]
                mb_act = act_t[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                logits = actor(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_old_logp)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v = critic(mb_obs).squeeze(-1)
                v_loss = F.mse_loss(v, mb_ret)

                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), cfg.max_grad_norm)
                optimizer.step()

                # Projected gradient descent: clamp actor parameters to bounds
                if use_pgd:
                    with torch.no_grad():
                        for param, lb, ub in zip(actor.parameters(), bounds_l, bounds_u): # type: ignore
                            # Count parameters that need projection
                            violations = ((param.data < lb) | (param.data > ub)).sum().item()
                            if violations > 0:
                                pgd_projections += violations
                            param.data.clamp_(lb, ub)

        if global_step % (10 * cfg.rollout_steps) < cfg.rollout_steps:
            mean_r, std_r, failure_rate = evaluate(env=env, actor=actor, device=device, episodes=10) # type: ignore
            # Reset the environment after evaluation since the last episode ended
            obs, _ = env.reset()
            elapsed = time.time() - start_time
            log_msg = f"Steps={global_step} | meanR={mean_r:.1f} +/- {std_r:.1f} | elapsed={elapsed:.1f}s"
            log_msg += f" | failure_rate={failure_rate:.2f}"
            if use_pgd:
                log_msg += f" | PGD projections={pgd_projections}"
            print(log_msg)

    env.close()

    if cfg.eval_episodes is not None and cfg.eval_episodes > 0:
        # Final checks and evaluation
        mean_r, std_r, failure_rate = evaluate(env=env, actor=actor, device=device, episodes=cfg.eval_episodes) # type: ignore
        final_msg = f"Final evaluation over {cfg.eval_episodes} episodes: mean_reward={mean_r:.2f} +/- {std_r:.2f}"
        final_msg += f" | failure_rate={failure_rate:.2f}"
        if use_pgd:
            final_msg += f" | Total PGD projections during training: {pgd_projections}"
        print(final_msg)

    # Return results
    if return_training_data:
        # Convert lists to numpy arrays
        training_data['states'] = np.array(training_data['states']) # type: ignore
        training_data['actions'] = np.array(training_data['actions']) # type: ignore
        # training_data['rewards'] = np.array(training_data['rewards']) # type: ignore
        return actor, critic, training_data # type: ignore
    else:
        return actor, critic